"""
조건부 엣지(라우팅 함수) 테스트.

route_after_model은 LLM을 전혀 호출하지 않는 순수 함수다 — state의 마지막
메시지만 보고 다음 노드 이름을 문자열로 반환한다. 그래서 FakeChatModel조차
필요 없이, AIMessage를 직접 손으로 만들어서 검증한다.

반복 상한(iteration cap)은 모델이 tool_calls를 계속 반환하더라도 무한
루프에 빠지지 않도록 하는 안전장치이므로, "도구 호출이 있어도 상한을
넘으면 무조건 종료"하는 케이스를 반드시 커버한다.
"""

from langchain_core.messages import AIMessage
from langgraph.graph import END

from agent.nodes import MAX_ITERATIONS, route_after_model


def _ai_message_with_tool_call() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "calculate", "args": {"expression": "1+1"}, "id": "call_1"}],
    )


def test_routes_to_tools_when_tool_call_present():
    state = {"messages": [_ai_message_with_tool_call()], "iteration": 1}
    assert route_after_model(state) == "tools"


def test_routes_to_end_when_no_tool_call():
    state = {"messages": [AIMessage(content="최종 답변입니다.")], "iteration": 1}
    assert route_after_model(state) == END


def test_routes_to_end_at_iteration_cap_even_with_pending_tool_call():
    state = {"messages": [_ai_message_with_tool_call()], "iteration": MAX_ITERATIONS}
    assert route_after_model(state) == END


def test_routes_to_tools_just_below_iteration_cap():
    state = {"messages": [_ai_message_with_tool_call()], "iteration": MAX_ITERATIONS - 1}
    assert route_after_model(state) == "tools"
