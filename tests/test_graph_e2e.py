"""
그래프 통합(E2E) 테스트.

실제 도구(calculate, read_text_file)와 FakeChatModel을 결합해서 그래프
전체가 model -> tools -> model -> END 순서로 정상 순환하는지 검증한다.
LLM 자체는 가짜지만, 도구 실행과 상태 병합(add_messages reducer,
iteration 누적)은 실제 코드 경로를 그대로 탄다.
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
