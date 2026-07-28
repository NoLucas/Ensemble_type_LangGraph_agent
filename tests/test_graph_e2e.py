"""
그래프 통합(E2E) 테스트.

실제 도구(calculate, read_text_file, write_text_file)와 FakeChatModel을
결합해서 그래프 전체가
dispatcher -> [calculate_node, read_file_node, write_file_node] (병렬
팬아웃) -> reporter -> END 순서로 정상 동작하는지 검증한다. LLM 자체는
가짜지만, 도구 실행과 상태 병합(add_messages reducer, iteration 누적)은
실제 코드 경로를 그대로 탄다.

가장 중요한 두 케이스:
1. test_graph_fans_out_to_all_three_tools_then_reports — dispatcher가 한
   응답에서 세 도구를 동시에 요청하면, llm 호출은 정확히 2번(dispatcher 1번
   + reporter 1번)만 발생해야 한다. 순차 ReAct 루프였다면 도구마다 model을
   한 번씩 더 거쳐야 하므로, 이 테스트가 깨지면 팬아웃이 무너졌다는 뜻이다.
2. test_graph_ignores_tool_calls_returned_by_reporter — reporter는
   bind_tools되지 않은 llm을 쓰므로 실제로는 tool_call을 만들 수 없지만,
   FakeChatModel은 무엇이든 반환할 수 있다. 혹시 reporter가 tool_call이
   섞인 응답을 반환하더라도(예: 모델이 프롬프트를 무시하는 경우), 그래프
   구조상 reporter -> END는 무조건 엣지라 라우팅이 그 tool_call을 아예
   보지 않는다는 것, 즉 안전장치가 "프롬프트 준수"가 아니라 "그래프 구조"
   라는 것을 검증한다.
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.graph import build_graph


def test_graph_dispatches_calculate_then_reporter_reports(fake_llm_factory):
    responses = [
        AIMessage(
            content="",
            tool_calls=[{"name": "calculate", "args": {"expression": "2+2"}, "id": "call_1"}],
        ),
        AIMessage(content="2+2는 4입니다."),
    ]
    llm = fake_llm_factory(responses)
    graph = build_graph(llm)

    result = graph.invoke(
        {"messages": [HumanMessage(content="2+2는 얼마야?")], "iteration": 0}
    )

    assert "4" in result["messages"][-1].content
    assert result["iteration"] == 2
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].content == "4"


def test_graph_dispatches_read_then_reporter_reports(fake_llm_factory, sandbox_dir):
    (sandbox_dir / "report.txt").write_text("매출 100억", encoding="utf-8")
    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "read_sandbox_file", "args": {"filename": "report.txt"}, "id": "call_1"}
            ],
        ),
        AIMessage(content="보고서 내용: 매출 100억"),
    ]
    llm = fake_llm_factory(responses)
    graph = build_graph(llm, sandbox_dir=sandbox_dir)

    result = graph.invoke(
        {"messages": [HumanMessage(content="report.txt 내용 알려줘")], "iteration": 0}
    )

    assert "매출 100억" in result["messages"][-1].content


def test_graph_dispatches_write_then_reporter_reports(fake_llm_factory, sandbox_dir):
    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "write_sandbox_file",
                    "args": {"filename": "note.txt", "content": "저장된 메모"},
                    "id": "call_1",
                }
            ],
        ),
        AIMessage(content="note.txt에 저장했습니다."),
    ]
    llm = fake_llm_factory(responses)
    graph = build_graph(llm, sandbox_dir=sandbox_dir)

    result = graph.invoke(
        {"messages": [HumanMessage(content="메모 저장해줘")], "iteration": 0}
    )

    assert "저장" in result["messages"][-1].content
    assert (sandbox_dir / "note.txt").read_text(encoding="utf-8") == "저장된 메모"


def test_graph_fans_out_to_all_three_tools_then_reports(fake_llm_factory, sandbox_dir):
    (sandbox_dir / "sales.txt").write_text("1200", encoding="utf-8")
    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "calculate", "args": {"expression": "2+2"}, "id": "call_calc"},
                {
                    "name": "read_sandbox_file",
                    "args": {"filename": "sales.txt"},
                    "id": "call_read",
                },
                {
                    "name": "write_sandbox_file",
                    "args": {"filename": "result.txt", "content": "4"},
                    "id": "call_write",
                },
            ],
        ),
        AIMessage(content="계산 결과 4, 매출 파일 내용 1200 확인, result.txt 저장까지 완료했습니다."),
    ]
    llm = fake_llm_factory(responses)
    graph = build_graph(llm, sandbox_dir=sandbox_dir)

    result = graph.invoke(
        {"messages": [HumanMessage(content="계산하고 파일도 읽고 결과도 저장해줘")], "iteration": 0}
    )

    # dispatcher 1번 + reporter 1번 = 정확히 2번. 세 도구가 순차가 아니라
    # 같은 라운드에서 병렬로 처리됐다는 증거다.
    assert llm.calls == 2
    assert result["iteration"] == 2

    tool_messages = {m.tool_call_id: m.content for m in result["messages"] if isinstance(m, ToolMessage)}
    assert tool_messages["call_calc"] == "4"
    assert tool_messages["call_read"] == "1200"
    assert tool_messages["call_write"].startswith("OK")
    assert (sandbox_dir / "result.txt").read_text(encoding="utf-8") == "4"

    final_message = result["messages"][-1]
    assert final_message.content == "계산 결과 4, 매출 파일 내용 1200 확인, result.txt 저장까지 완료했습니다."


def test_graph_skips_reporter_when_no_tools_needed(fake_llm_factory):
    llm = fake_llm_factory([AIMessage(content="바로 답할 수 있어요.")])
    graph = build_graph(llm)

    result = graph.invoke({"messages": [HumanMessage(content="안녕")], "iteration": 0})

    # dispatcher가 도구를 요청하지 않으면 reporter는 아예 호출되지 않는다
    # (llm.calls == 1) — 불필요한 LLM 호출을 만들지 않는 게 "효율적"이라는
    # dispatcher의 역할과 일치한다.
    assert llm.calls == 1
    assert result["iteration"] == 1
    assert result["messages"][-1].content == "바로 답할 수 있어요."


def test_graph_ignores_tool_calls_returned_by_reporter(fake_llm_factory):
    # reporter가 (실수로든 뭐든) tool_call이 섞인 응답을 내더라도, reporter
    # -> END는 무조건 엣지라 그래프가 그 tool_call을 절대 실행하지 않는다.
    # 안전장치가 "프롬프트를 잘 따르길 바란다"가 아니라 그래프 구조 자체임을
    # 보여주는 테스트다.
    responses = [
        AIMessage(
            content="",
            tool_calls=[{"name": "calculate", "args": {"expression": "1+1"}, "id": "call_1"}],
        ),
        AIMessage(
            content="계산 끝났습니다.",
            tool_calls=[{"name": "calculate", "args": {"expression": "9+9"}, "id": "call_2"}],
        ),
    ]
    llm = fake_llm_factory(responses)
    graph = build_graph(llm)

    result = graph.invoke(
        {"messages": [HumanMessage(content="1+1 계산해줘")], "iteration": 0}
    )

    # reporter 이후 추가 라운드가 없으므로 llm은 정확히 2번만 호출된다.
    assert llm.calls == 2
    assert result["iteration"] == 2
    assert result["messages"][-1].content == "계산 끝났습니다."
    # call_2에 대한 ToolMessage는 존재하지 않는다 — 실행되지 않았다는 뜻.
    tool_call_ids = {m.tool_call_id for m in result["messages"] if isinstance(m, ToolMessage)}
    assert tool_call_ids == {"call_1"}
