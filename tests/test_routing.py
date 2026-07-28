"""
조건부 엣지(라우팅 함수) 테스트.

route_after_model은 LLM을 전혀 호출하지 않는 순수 함수다 — state의 마지막
메시지만 보고 다음에 실행할 노드 이름(들)을 반환한다. 그래서 FakeChatModel조차
필요 없이, AIMessage를 직접 손으로 만들어서 검증한다.

이 프로젝트는 model -> [calculate_node, read_file_node] 팬아웃 -> model
팬인 구조를 쓴다. tool_call이 하나만 와도 두 노드를 모두 리스트로 반환해서
"항상 둘 다 병렬로 깨우고, 자기 담당이 아니면 조용히 통과한다"는 원칙을
지킨다 — add_edge(["calculate_node", "read_file_node"], "model") 형태의
팬인 조인은 두 노드가 매번 함께 실행되어야만 데드락 없이 동작하기 때문이다.

반복 상한(iteration cap)은 모델이 tool_calls를 계속 반환하더라도 무한
루프에 빠지지 않도록 하는 안전장치이므로, "도구 호출이 있어도 상한을
넘으면 무조건 종료"하는 케이스를 반드시 커버한다.
"""

from langchain_core.messages import AIMessage
from langgraph.graph import END

from agent.nodes import FANOUT_TOOL_NODES, MAX_ITERATIONS, route_after_model


def _ai_message_with_tool_call(name: str = "calculate") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": {"expression": "1+1"}, "id": "call_1"}],
    )


def test_routes_to_all_fanout_tool_nodes_when_any_tool_call_present():
    state = {"messages": [_ai_message_with_tool_call("calculate")], "iteration": 1}
    assert route_after_model(state) == FANOUT_TOOL_NODES


def test_routes_to_all_fanout_tool_nodes_regardless_of_which_tool_was_called():
    # read_sandbox_file만 요청했어도 calculate_node까지 함께 깨워야
    # 팬인 조인(add_edge(["calculate_node","read_file_node"], "model"))이
    # 데드락 없이 동작한다. calculate_node는 담당 tool_call이 없으면
    # 그냥 통과(pass-through)한다.
    state = {"messages": [_ai_message_with_tool_call("read_sandbox_file")], "iteration": 1}
    assert route_after_model(state) == FANOUT_TOOL_NODES


def test_routes_to_end_when_no_tool_call():
    state = {"messages": [AIMessage(content="최종 답변입니다.")], "iteration": 1}
    assert route_after_model(state) == END


def test_routes_to_end_at_iteration_cap_even_with_pending_tool_call():
    state = {"messages": [_ai_message_with_tool_call()], "iteration": MAX_ITERATIONS}
    assert route_after_model(state) == END


def test_routes_to_fanout_just_below_iteration_cap():
    state = {"messages": [_ai_message_with_tool_call()], "iteration": MAX_ITERATIONS - 1}
    assert route_after_model(state) == FANOUT_TOOL_NODES
