"""
그래프 통합(E2E) 테스트.

실제 도구(calculate, read_text_file, write_text_file)와 FakeChatModel을
결합해서 그래프 전체가
dispatcher -> [calculate_node, read_file_node, write_file_node] (도구
팬아웃) -> [draft_concise, draft_detailed, draft_action] (보고서 앙상블
팬아웃) -> aggregate_reports -> END 순서로 정상 동작하는지 검증한다. LLM
자체는 가짜지만, 도구 실행과 상태 병합(add_messages/operator.add reducer)은
실제 코드 경로를 그대로 탄다.

가장 중요한 두 케이스:
1. test_graph_fans_out_tools_then_fans_out_three_report_drafts — dispatcher가
   한 응답에서 세 도구를 동시에 요청하면, llm 호출은 정확히 4번(dispatcher 1
   + draft 3)만 발생해야 한다. 도구가 순차 처리됐거나 draft가 순차 처리됐다면
   호출 수가 더 늘어나므로, 이 숫자가 깨지면 팬아웃 두 단계 중 하나가
   무너졌다는 뜻이다.
2. test_graph_ignores_tool_calls_returned_by_draft_node — draft 노드는
   bind_tools되지 않은 llm을 쓰므로 실제로는 tool_call을 만들 수 없지만,
   FakeChatModel은 무엇이든 반환할 수 있다. draft 응답에 tool_call이 섞여
   있어도 그래프가 절대 실행하지 않는다는 것 — 안전장치가 "프롬프트 준수"가
   아니라 "그래프 구조(bind_tools 여부 + 고정 엣지)"라는 것을 검증한다.
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.graph import build_graph


def test_graph_dispatches_calculate_then_drafts_report(fake_llm_factory):
    responses = [
        AIMessage(
            content="",
            tool_calls=[{"name": "calculate", "args": {"expression": "2+2"}, "id": "call_1"}],
        ),
        AIMessage(content="핵심은 4입니다."),
        AIMessage(content="calculate(2+2)를 실행해 4를 얻었습니다."),
        AIMessage(content="다음으로 이 값을 활용해 보세요."),
    ]
    llm = fake_llm_factory(responses)
    graph = build_graph(llm)

    result = graph.invoke(
        {"messages": [HumanMessage(content="2+2는 얼마야?")], "iteration": 0, "report_drafts": []}
    )

    final_text = result["messages"][-1].content
    assert "핵심은 4입니다." in final_text
    assert "calculate(2+2)를 실행해 4를 얻었습니다." in final_text
    assert "다음으로 이 값을 활용해 보세요." in final_text
    # 세 섹션 헤딩이 항상 이 순서로 존재해야 한다 (aggregate_reports_node가
    # 병렬 도착 순서와 무관하게 고정 순서로 조합하기 때문).
    assert final_text.index("핵심 요약") < final_text.index("상세 설명") < final_text.index("실무 제안")

    assert llm.calls == 4  # dispatcher 1 + draft 3
    assert result["iteration"] == 4
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].content == "4"


def test_graph_dispatches_read_then_drafts_report(fake_llm_factory, sandbox_dir):
    (sandbox_dir / "report.txt").write_text("매출 100억", encoding="utf-8")
    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "read_sandbox_file", "args": {"filename": "report.txt"}, "id": "call_1"}
            ],
        ),
        AIMessage(content="매출 100억을 확인했습니다."),
        AIMessage(content="report.txt에서 매출 100억을 읽었습니다."),
        AIMessage(content="이 수치를 다음 보고서에 반영하세요."),
    ]
    llm = fake_llm_factory(responses)
    graph = build_graph(llm, sandbox_dir=sandbox_dir)

    result = graph.invoke(
        {"messages": [HumanMessage(content="report.txt 내용 알려줘")], "iteration": 0, "report_drafts": []}
    )

    assert "매출 100억" in result["messages"][-1].content


def test_graph_dispatches_write_then_drafts_report(fake_llm_factory, sandbox_dir):
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
        AIMessage(content="저장을 완료했습니다."),
        AIMessage(content="note.txt에 '저장된 메모'를 기록했습니다."),
        AIMessage(content="필요하면 다시 읽어 확인하세요."),
    ]
    llm = fake_llm_factory(responses)
    graph = build_graph(llm, sandbox_dir=sandbox_dir)

    result = graph.invoke(
        {"messages": [HumanMessage(content="메모 저장해줘")], "iteration": 0, "report_drafts": []}
    )

    assert "저장" in result["messages"][-1].content
    assert (sandbox_dir / "note.txt").read_text(encoding="utf-8") == "저장된 메모"


def test_graph_fans_out_tools_then_fans_out_three_report_drafts(fake_llm_factory, sandbox_dir):
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
        AIMessage(content="요약 draft"),
        AIMessage(content="상세 draft"),
        AIMessage(content="제안 draft"),
    ]
    llm = fake_llm_factory(responses)
    graph = build_graph(llm, sandbox_dir=sandbox_dir)

    result = graph.invoke(
        {
            "messages": [HumanMessage(content="계산하고 파일도 읽고 결과도 저장해줘")],
            "iteration": 0,
            "report_drafts": [],
        }
    )

    # dispatcher 1번 + draft 3번 = 정확히 4번. 도구 3개와 draft 3개 모두
    # 순차가 아니라 각자의 라운드에서 병렬로 처리됐다는 증거다.
    assert llm.calls == 4
    assert result["iteration"] == 4
    assert len(result["report_drafts"]) == 3

    tool_messages = {m.tool_call_id: m.content for m in result["messages"] if isinstance(m, ToolMessage)}
    assert tool_messages["call_calc"] == "4"
    assert tool_messages["call_read"] == "1200"
    assert tool_messages["call_write"].startswith("OK")
    assert (sandbox_dir / "result.txt").read_text(encoding="utf-8") == "4"

    final_text = result["messages"][-1].content
    assert "요약 draft" in final_text
    assert "상세 draft" in final_text
    assert "제안 draft" in final_text


def test_graph_skips_report_drafting_when_no_tools_needed(fake_llm_factory):
    llm = fake_llm_factory([AIMessage(content="바로 답할 수 있어요.")])
    graph = build_graph(llm)

    result = graph.invoke(
        {"messages": [HumanMessage(content="안녕")], "iteration": 0, "report_drafts": []}
    )

    # dispatcher가 도구를 요청하지 않으면 draft 3개도 aggregate도 전혀
    # 실행되지 않는다(llm.calls == 1) — 불필요한 LLM 호출을 만들지 않는 게
    # "효율적"이라는 dispatcher의 역할과 일치한다.
    assert llm.calls == 1
    assert result["iteration"] == 1
    assert result["messages"][-1].content == "바로 답할 수 있어요."
    assert result.get("report_drafts", []) == []


def test_graph_ignores_tool_calls_returned_by_draft_node(fake_llm_factory):
    # draft 노드가 (실수로든 뭐든) tool_call이 섞인 응답을 내더라도, draft ->
    # aggregate_reports -> END는 모두 무조건 엣지라 그래프가 그 tool_call을
    # 절대 실행하지 않는다. 안전장치가 "프롬프트를 잘 따르길 바란다"가 아니라
    # 그래프 구조(고정 엣지 + bind_tools 안 된 llm) 자체임을 보여주는 테스트다.
    responses = [
        AIMessage(
            content="",
            tool_calls=[{"name": "calculate", "args": {"expression": "1+1"}, "id": "call_1"}],
        ),
        AIMessage(
            content="요약 draft",
            tool_calls=[{"name": "calculate", "args": {"expression": "9+9"}, "id": "call_2"}],
        ),
        AIMessage(content="상세 draft"),
        AIMessage(content="제안 draft"),
    ]
    llm = fake_llm_factory(responses)
    graph = build_graph(llm)

    result = graph.invoke(
        {"messages": [HumanMessage(content="1+1 계산해줘")], "iteration": 0, "report_drafts": []}
    )

    # draft 단계 이후 추가 라운드가 없으므로 llm은 정확히 4번만 호출된다.
    assert llm.calls == 4
    assert result["iteration"] == 4
    # call_2에 대한 ToolMessage는 존재하지 않는다 — 실행되지 않았다는 뜻.
    tool_call_ids = {m.tool_call_id for m in result["messages"] if isinstance(m, ToolMessage)}
    assert tool_call_ids == {"call_1"}
