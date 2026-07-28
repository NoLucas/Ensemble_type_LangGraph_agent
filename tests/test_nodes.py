"""
노드 단위 테스트.

이 에이전트는 model 노드 하나가 아니라 역할이 다른 두 LLM 노드를 쓴다.

- dispatcher(call_dispatcher_model): 사용자의 "첫 입력"을 받아 calculate/
  read/write 세 도구 중 필요한 것을 효율적으로 지시하는 역할. tool_calls를
  낼 수 있도록 tool-bound llm을 받는다.
- reporter(call_reporter_model): 팬아웃/팬인으로 도구 실행이 끝난 뒤, 그
  결과를 사용자에게 효과적으로 보고하는 역할. tool을 다시 호출할 필요가
  없으므로 tool이 바인딩되지 않은 llm을 받는다 — 이렇게 하면 프롬프트로만
  "도구 부르지 마"라고 당부하는 게 아니라, 애초에 모델이 도구 스키마 자체를
  모르게 만들어 구조적으로 재호출을 막는다.

두 노드 모두 실제 LLM을 호출하지 않고 conftest.py의 FakeChatModel을
주입해서, (1) state 전체가 아니라 변경된 필드만 반환하는지, (2) iteration을
정확히 1씩 증가시키는지, (3) 각자의 시스템 프롬프트를 매 호출마다
포함시키는지 검증한다. 두 프롬프트가 서로 뒤바뀌면(=역할이 섞이면) 테스트가
바로 깨지도록, 실제 프롬프트 상수와 content가 정확히 일치하는지까지 비교한다.

calculate_node/read_file_node/write_file_node는 팬아웃 구조의 "담당 도구별
병렬 노드"다. dispatcher가 반환한 마지막 AIMessage의 tool_calls 중 자기
이름과 일치하는 것만 실행하고, 없으면 아무 것도 반환하지 않는다
(pass-through) — 이 세 노드는 tool_call을 하나만 냈어도 항상 함께
깨워지기 때문에, 자기 몫이 없을 때 조용히 통과하는 능력이 팬인 조인의
전제조건이다.

"노드는 partial state만 반환해야 한다"는 원칙은 add_messages reducer가
정상 동작하기 위한 전제조건이라 별도 테스트로 고정해둔다 (전체 state를
반환하면 메시지 리스트가 이중으로 누적되는 조용한 버그가 생긴다).
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.nodes import calculate_node, read_file_node, write_file_node
from agent.nodes import (
    DISPATCHER_SYSTEM_PROMPT,
    REPORTER_SYSTEM_PROMPT,
    call_dispatcher_model,
    call_reporter_model,
)


def test_call_dispatcher_model_returns_only_changed_keys(fake_llm_factory):
    llm = fake_llm_factory([AIMessage(content="4")])
    state = {"messages": [HumanMessage(content="2+2는?")], "iteration": 0}

    result = call_dispatcher_model(state, llm=llm)

    assert set(result.keys()) == {"messages", "iteration"}
    assert result["iteration"] == 1
    assert len(result["messages"]) == 1
    assert result["messages"][0].content == "4"


def test_call_dispatcher_model_increments_iteration_from_current_value(fake_llm_factory):
    llm = fake_llm_factory([AIMessage(content="ok")])
    state = {"messages": [HumanMessage(content="hi")], "iteration": 5}

    result = call_dispatcher_model(state, llm=llm)

    assert result["iteration"] == 6


def test_call_dispatcher_model_prepends_dispatcher_system_prompt(fake_llm_factory):
    llm = fake_llm_factory([AIMessage(content="ok")])
    state = {"messages": [HumanMessage(content="hi")], "iteration": 0}

    call_dispatcher_model(state, llm=llm)

    sent_messages = llm.received_messages[0]
    assert isinstance(sent_messages[0], SystemMessage)
    assert sent_messages[0].content == DISPATCHER_SYSTEM_PROMPT
    assert sent_messages[1].content == "hi"


def test_call_reporter_model_returns_only_changed_keys(fake_llm_factory):
    llm = fake_llm_factory([AIMessage(content="계산 결과는 4입니다.")])
    state = {
        "messages": [HumanMessage(content="2+2는?"), AIMessage(content="")],
        "iteration": 1,
    }

    result = call_reporter_model(state, llm=llm)

    assert set(result.keys()) == {"messages", "iteration"}
    assert result["iteration"] == 2
    assert result["messages"][0].content == "계산 결과는 4입니다."


def test_call_reporter_model_increments_iteration_from_current_value(fake_llm_factory):
    llm = fake_llm_factory([AIMessage(content="보고 완료")])
    state = {"messages": [HumanMessage(content="hi")], "iteration": 5}

    result = call_reporter_model(state, llm=llm)

    assert result["iteration"] == 6


def test_call_reporter_model_prepends_reporter_system_prompt(fake_llm_factory):
    llm = fake_llm_factory([AIMessage(content="ok")])
    state = {"messages": [HumanMessage(content="hi")], "iteration": 1}

    call_reporter_model(state, llm=llm)

    sent_messages = llm.received_messages[0]
    assert isinstance(sent_messages[0], SystemMessage)
    assert sent_messages[0].content == REPORTER_SYSTEM_PROMPT
    assert sent_messages[1].content == "hi"


def test_dispatcher_and_reporter_prompts_are_distinct():
    # 역할이 실수로 서로 뒤바뀌면(=같은 프롬프트를 공유하면) 여기서 바로 걸린다.
    assert DISPATCHER_SYSTEM_PROMPT != REPORTER_SYSTEM_PROMPT


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


# ---------------------------------------------------------------------------
# write_file_node: write_sandbox_file tool_call만 담당, 나머지는 통과
# ---------------------------------------------------------------------------


def test_write_file_node_executes_matching_tool_call(sandbox_dir):
    ai_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "write_sandbox_file",
                "args": {"filename": "out.txt", "content": "결과"},
                "id": "call_3",
            }
        ],
    )
    state = {"messages": [HumanMessage(content="저장해줘"), ai_message], "iteration": 1}

    node = write_file_node(sandbox_dir)
    result = node(state)

    assert len(result["messages"]) == 1
    tool_message = result["messages"][0]
    assert tool_message.content.startswith("OK")
    assert tool_message.tool_call_id == "call_3"
    assert (sandbox_dir / "out.txt").read_text(encoding="utf-8") == "결과"


def test_write_file_node_passes_through_when_no_matching_tool_call(sandbox_dir):
    ai_message = AIMessage(
        content="",
        tool_calls=[{"name": "calculate", "args": {"expression": "1+1"}, "id": "call_1"}],
    )
    state = {"messages": [HumanMessage(content="계산해줘"), ai_message], "iteration": 1}

    node = write_file_node(sandbox_dir)
    result = node(state)

    assert result == {}
