"""
그래프 통합(E2E) 테스트.

실제 도구(calculate, read_text_file)와 FakeChatModel을 결합해서 그래프
전체가 model -> [calculate_node, read_file_node] (병렬 팬아웃) -> model
(팬인) -> ... -> END 순서로 정상 순환하는지 검증한다. LLM 자체는 가짜지만,
도구 실행과 상태 병합(add_messages reducer, iteration 누적)은 실제 코드
경로를 그대로 탄다.

가장 중요한 케이스는 test_graph_fans_out_to_both_tools_in_a_single_round다:
모델이 한 번의 응답에서 calculate와 read_sandbox_file을 동시에 요청하면,
model 호출 2번(요청 1번 + 최종 답변 1번)만으로 두 도구가 모두 실행돼야
한다. 순차 ReAct 루프였다면 도구마다 model을 한 번씩 더 거쳐야 하므로,
이 테스트가 깨지면 팬아웃이 아니라 다시 순차 구조로 퇴행했다는 뜻이다.
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.graph import build_graph
from agent.nodes import MAX_ITERATIONS


def test_graph_calls_calculate_tool_then_returns_final_answer(fake_llm_factory):
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


def test_graph_reads_sandbox_file_via_tool(fake_llm_factory, sandbox_dir):
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


def test_graph_fans_out_to_all_three_tools_in_a_single_round(fake_llm_factory, sandbox_dir):
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

    # model이 정확히 2번만 호출됐다는 것 자체가 세 도구가 순차가 아니라
    # 같은 라운드에서 병렬로 처리됐다는 증거다. 순차 구조였다면 도구마다
    # model을 한 번씩 더 거쳐야 하므로 llm.calls가 4가 됐을 것이다.
    assert llm.calls == 2
    assert result["iteration"] == 2

    tool_messages = {m.tool_call_id: m.content for m in result["messages"] if isinstance(m, ToolMessage)}
    assert tool_messages["call_calc"] == "4"
    assert tool_messages["call_read"] == "1200"
    assert tool_messages["call_write"].startswith("OK")
    assert (sandbox_dir / "result.txt").read_text(encoding="utf-8") == "4"


def test_graph_writes_sandbox_file_via_tool(fake_llm_factory, sandbox_dir):
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


def test_graph_stops_without_calling_tools_when_model_answers_directly(fake_llm_factory):
    llm = fake_llm_factory([AIMessage(content="바로 답할 수 있어요.")])
    graph = build_graph(llm)

    result = graph.invoke(
        {"messages": [HumanMessage(content="안녕")], "iteration": 0}
    )

    assert result["iteration"] == 1
    assert result["messages"][-1].content == "바로 답할 수 있어요."


def test_graph_hits_recursion_limit_when_model_always_calls_tools(fake_llm_factory):
    def make_tool_call_response(i: int) -> AIMessage:
        # 매번 새 AIMessage 인스턴스를 만들어야 한다. add_messages reducer는
        # 메시지 id로 병합 여부를 판단하는데, 같은 객체(=같은 id)를 재사용하면
        # 리스트 끝에 추가되지 않고 기존 위치에서 덮어써져 마지막 메시지가
        # 달라지는 테스트 아티팩트가 생긴다.
        return AIMessage(
            content="",
            tool_calls=[{"name": "calculate", "args": {"expression": "1+1"}, "id": f"call_{i}"}],
        )

    # 상한을 넉넉히 넘도록 동일 응답을 충분히 준비한다.
    llm = fake_llm_factory([make_tool_call_response(i) for i in range(MAX_ITERATIONS + 5)])
    graph = build_graph(llm)

    result = graph.invoke(
        {"messages": [HumanMessage(content="계속 계산해줘")], "iteration": 0},
        config={"recursion_limit": 100},
    )

    # route_after_model이 MAX_ITERATIONS에서 강제로 END를 반환하므로
    # 정확히 MAX_ITERATIONS번만 model이 호출되고 멈춰야 한다.
    assert result["iteration"] == MAX_ITERATIONS
