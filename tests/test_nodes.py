"""
call_model 노드 단위 테스트.

실제 LLM을 호출하지 않고 conftest.py의 FakeChatModel을 주입해서, 노드가
(1) state 전체가 아니라 변경된 필드만 반환하는지, (2) iteration을 정확히
1씩 증가시키는지, (3) 시스템 프롬프트를 매 호출마다 포함시키는지를 검증한다.

"노드는 partial state만 반환해야 한다"는 원칙은 add_messages reducer가
정상 동작하기 위한 전제조건이라 별도 테스트로 고정해둔다 (전체 state를
반환하면 메시지 리스트가 이중으로 누적되는 조용한 버그가 생긴다).
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.nodes import call_model


def test_call_model_returns_only_changed_keys(fake_llm_factory):
    llm = fake_llm_factory([AIMessage(content="4")])
    state = {"messages": [HumanMessage(content="2+2는?")], "iteration": 0}

    result = call_model(state, llm=llm)

    assert set(result.keys()) == {"messages", "iteration"}
    assert result["iteration"] == 1
    assert len(result["messages"]) == 1
    assert result["messages"][0].content == "4"


def test_call_model_increments_iteration_from_current_value(fake_llm_factory):
    llm = fake_llm_factory([AIMessage(content="ok")])
    state = {"messages": [HumanMessage(content="hi")], "iteration": 5}

    result = call_model(state, llm=llm)

    assert result["iteration"] == 6


def test_call_model_prepends_system_prompt(fake_llm_factory):
    llm = fake_llm_factory([AIMessage(content="ok")])
    state = {"messages": [HumanMessage(content="hi")], "iteration": 0}

    call_model(state, llm=llm)

    sent_messages = llm.received_messages[0]
    assert isinstance(sent_messages[0], SystemMessage)
    assert sent_messages[1].content == "hi"
