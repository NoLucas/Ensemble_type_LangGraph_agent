"""
AgentState 스키마에 대한 테스트.

AgentState는 그래프의 모든 노드가 공유하는 상태 타입이다. TypedDict라 런타임
강제는 없지만, add_messages reducer가 실제로 "덮어쓰기"가 아니라 "누적"으로
동작하는지는 반드시 검증해야 한다. 이 reducer가 깨지면 노드가 model 응답을
반환할 때마다 이전 대화 기록이 사라지는 조용한 버그가 생기기 때문이다.
"""

from langchain_core.messages import HumanMessage, AIMessage

from agent.state import AgentState, merge_state


def test_agent_state_accepts_required_keys():
    state: AgentState = {
        "messages": [HumanMessage(content="hi")],
        "iteration": 0,
    }
    assert state["iteration"] == 0
    assert len(state["messages"]) == 1


def test_add_messages_reducer_appends_not_overwrites():
    existing = [HumanMessage(content="hi")]
    incoming = [AIMessage(content="hello")]

    merged = merge_state(existing, incoming)

    assert len(merged) == 2
    assert merged[0].content == "hi"
    assert merged[1].content == "hello"
