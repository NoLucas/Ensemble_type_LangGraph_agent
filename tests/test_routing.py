"""
조건부 엣지(라우팅 함수) 테스트.

route_after_dispatcher는 LLM을 전혀 호출하지 않는 순수 함수다 — dispatcher가
반환한 마지막 메시지만 보고 다음에 실행할 노드 이름(들)을 반환한다. 그래서
FakeChatModel조차 필요 없이, AIMessage를 직접 손으로 만들어서 검증한다.

이 프로젝트는 dispatcher -> [repo_overview_node, repo_source_node]
(팬아웃) -> voter 팬아웃 -> vote_for_best_report -> END 구조를 쓴다.
dispatcher가 tool_call을 하나만 냈어도 두 노드를 모두 리스트로 반환해서
"항상 둘 다 병렬로 깨우고, 자기 담당이 아니면 조용히 통과한다"는 원칙을
지킨다 — add_edge(FANOUT_TOOL_NODES, voter_node) 형태의 팬인 조인은 두
노드가 매번 함께 실행되어야만 데드락 없이 동작하기 때문이다.

dispatcher -> 팬아웃 -> voter 팬아웃 -> vote -> END는 사이클이 없는
구조라(voter는 도구가 바인딩되지 않은 llm을 쓰므로 구조적으로 다시
tool_call을 낼 수 없다), "반복 상한(iteration cap)" 개념은 필요 없다 —
그래프 자체가 무한 루프를 만들 수 없는 DAG이기 때문이다.
"""

from langchain_core.messages import AIMessage
from langgraph.graph import END

from agent.nodes import FANOUT_TOOL_NODES, route_after_dispatcher


def _ai_message_with_tool_call(name: str = "fetch_repo_overview") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": {"repo": "octocat/hello-world"}, "id": "call_1"}],
    )


def test_routes_to_all_fanout_tool_nodes_when_any_tool_call_present():
    state = {"messages": [_ai_message_with_tool_call("fetch_repo_overview")], "iteration": 1}
    assert route_after_dispatcher(state) == FANOUT_TOOL_NODES


def test_routes_to_all_fanout_tool_nodes_regardless_of_which_tool_was_called():
    # fetch_repo_source_sample만 요청했어도 repo_overview_node까지 함께
    # 깨워야 팬인 조인(add_edge(FANOUT_TOOL_NODES, voter_node))이 데드락
    # 없이 동작한다. repo_overview_node는 담당 tool_call이 없으면 그냥 통과한다.
    state = {
        "messages": [_ai_message_with_tool_call("fetch_repo_source_sample")],
        "iteration": 1,
    }
    assert route_after_dispatcher(state) == FANOUT_TOOL_NODES


def test_routes_to_end_when_no_tool_call():
    # 도구가 전혀 필요 없으면 voter를 거치지 않고 dispatcher의 답변이
    # 곧바로 최종 답변이 된다 (불필요한 LLM 호출을 만들지 않는 게 "효율적").
    state = {"messages": [AIMessage(content="최종 답변입니다.")], "iteration": 1}
    assert route_after_dispatcher(state) == END


def test_fanout_has_two_specialized_tool_nodes():
    # 개요(overview) / 소스 코드(source)로 역할이 세분화된 2-way 팬아웃이어야 한다.
    assert FANOUT_TOOL_NODES == ["repo_overview_node", "repo_source_node"]
