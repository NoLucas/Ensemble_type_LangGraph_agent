"""
노드 단위 테스트.

call_model은 실제 LLM을 호출하지 않고 conftest.py의 FakeChatModel을 주입해서,
노드가 (1) state 전체가 아니라 변경된 필드만 반환하는지, (2) iteration을
정확히 1씩 증가시키는지, (3) 시스템 프롬프트를 매 호출마다 포함시키는지를
검증한다.

calculate_node/read_file_node는 팬아웃 구조의 "담당 도구별 병렬 노드"다.
model이 반환한 마지막 AIMessage의 tool_calls 중 자기 이름과 일치하는
것만 실행하고, 없으면 아무 것도 반환하지 않는다(pass-through) — 이 두
노드는 model이 tool_call을 하나만 냈어도 항상 함께 깨워지기 때문에,
자기 몫이 없을 때 조용히 통과하는 능력이 팬인 조인의 전제조건이다.

"노드는 partial state만 반환해야 한다"는 원칙은 add_messages reducer가
정상 동작하기 위한 전제조건이라 별도 테스트로 고정해둔다 (전체 state를
반환하면 메시지 리스트가 이중으로 누적되는 조용한 버그가 생긴다).
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.nodes import calculate_node, read_file_node
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


# ---------------------------------------------------------------------------
# calculate_node: calculate tool_call만 담당, 나머지는 통과
# ---------------------------------------------------------------------------


def test_calculate_node_executes_matching_tool_call():
    ai_message = AIMessage(
        content="",
        tool_calls=[{"name": "calculate", "args": {"expression": "2+2"}, "id": "call_1"}],
    )
    state = {"messages": [HumanMessage(content="계산해줘"), ai_message], "iteration": 1}

    result = calculate_node(state)

    assert len(result["messages"]) == 1
    tool_message = result["messages"][0]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == "4"
    assert tool_message.tool_call_id == "call_1"


def test_calculate_node_passes_through_when_no_matching_tool_call():
    ai_message = AIMessage(
        content="",
        tool_calls=[{"name": "read_sandbox_file", "args": {"filename": "a.txt"}, "id": "call_1"}],
    )
    state = {"messages": [HumanMessage(content="파일 읽어줘"), ai_message], "iteration": 1}

    result = calculate_node(state)

    assert result == {}


def test_calculate_node_returns_error_message_for_invalid_args_without_raising():
    ai_message = AIMessage(
        content="",
        tool_calls=[{"name": "calculate", "args": {"wrong_key": "2+2"}, "id": "call_1"}],
    )
    state = {"messages": [HumanMessage(content="계산해줘"), ai_message], "iteration": 1}

    result = calculate_node(state)  # 예외 없이 에러 메시지로 반환되어야 한다

    assert result["messages"][0].content.startswith("Error")


# ---------------------------------------------------------------------------
# read_file_node: read_sandbox_file tool_call만 담당, 나머지는 통과
# ---------------------------------------------------------------------------


def test_read_file_node_executes_matching_tool_call(sandbox_dir):
    (sandbox_dir / "note.txt").write_text("hello", encoding="utf-8")
    ai_message = AIMessage(
        content="",
        tool_calls=[
            {"name": "read_sandbox_file", "args": {"filename": "note.txt"}, "id": "call_2"}
        ],
    )
    state = {"messages": [HumanMessage(content="읽어줘"), ai_message], "iteration": 1}

    node = read_file_node(sandbox_dir)
    result = node(state)

    assert len(result["messages"]) == 1
    tool_message = result["messages"][0]
    assert tool_message.content == "hello"
    assert tool_message.tool_call_id == "call_2"


def test_read_file_node_passes_through_when_no_matching_tool_call(sandbox_dir):
    ai_message = AIMessage(
        content="",
        tool_calls=[{"name": "calculate", "args": {"expression": "1+1"}, "id": "call_1"}],
    )
    state = {"messages": [HumanMessage(content="계산해줘"), ai_message], "iteration": 1}

    node = read_file_node(sandbox_dir)
    result = node(state)

    assert result == {}
