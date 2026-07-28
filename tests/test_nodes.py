"""
노드 단위 테스트.

이 에이전트는 model 노드 하나가 아니라 역할이 다른 LLM 노드들을 쓴다.

- dispatcher(call_dispatcher_model): 사용자의 "첫 입력"을 받아 calculate/
  read/write 세 도구 중 필요한 것을 효율적으로 지시하는 역할. tool_calls를
  낼 수 있도록 tool-bound llm을 받는다.
- report draft 3개(call_report_draft_model): 팬아웃/팬인으로 도구 실행이
  끝난 뒤, 같은 입력을 세 가지 다른 관점(간결 요약/상세 설명/실무 제안)으로
  각자 독립적으로 서술하는 역할. 앙상블 패턴이지만 계산/파일 결과처럼
  "정답이 하나뿐인" 사실을 다루므로, 관점을 다르게 해서 다양성을 주되
  temperature로 무작위성을 주지는 않는다(사실이 흔들릴 위험).
- aggregate_reports_node: 이 3개 draft를 **LLM을 다시 호출하지 않고** 고정된
  순서(간결→상세→제안)로 그대로 이어붙여 최종 답변을 만드는 결정론적 함수.
  "종합"을 또 다른 LLM 호출로 하면 그 종합 단계 자체가 환각을 일으킬 수
  있으므로, 여기서는 아예 LLM을 빼서 그 위험을 구조적으로 없앴다.

세 draft 노드 모두 실제 LLM을 호출하지 않고 conftest.py의 FakeChatModel을
주입해서, (1) state 전체가 아니라 변경된 필드만 반환하는지, (2) iteration을
정확히 1씩 증가시키는지, (3) 각자의 관점 프롬프트를 매 호출마다 포함시키는지
검증한다.

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
    REPORT_ANGLES,
    aggregate_reports_node,
    call_dispatcher_model,
    call_report_draft_model,
)


def test_call_dispatcher_model_returns_only_changed_keys(fake_llm_factory):
    llm = fake_llm_factory([AIMessage(content="4")])
    state = {"messages": [HumanMessage(content="2+2는?")], "iteration": 0}

    result = call_dispatcher_model(state, llm=llm)

    assert set(result.keys()) == {"messages", "iteration"}
    assert result["iteration"] == 1
    assert len(result["messages"]) == 1
    assert result["messages"][0].content == "4"


def test_call_dispatcher_model_returns_delta_of_one_regardless_of_current_state(fake_llm_factory):
    # iteration은 operator.add 리듀서가 누적하는 "델타"다. 병렬로 실행되는
    # draft 노드들과 동일한 계약을 지키기 위해, dispatcher도 state의 현재
    # iteration 값과 무관하게 항상 1만 반환해야 한다 (절대값 "현재+1"을
    # 계산하면 병렬 노드끼리 충돌한다 — state.py 주석 참고).
    llm = fake_llm_factory([AIMessage(content="ok")])
    state = {"messages": [HumanMessage(content="hi")], "iteration": 5}

    result = call_dispatcher_model(state, llm=llm)

    assert result["iteration"] == 1


def test_call_dispatcher_model_prepends_dispatcher_system_prompt(fake_llm_factory):
    llm = fake_llm_factory([AIMessage(content="ok")])
    state = {"messages": [HumanMessage(content="hi")], "iteration": 0}

    call_dispatcher_model(state, llm=llm)

    sent_messages = llm.received_messages[0]
    assert isinstance(sent_messages[0], SystemMessage)
    assert sent_messages[0].content == DISPATCHER_SYSTEM_PROMPT
    assert sent_messages[1].content == "hi"


# ---------------------------------------------------------------------------
# call_report_draft_model: 관점 하나를 맡아 report_drafts에 항목 하나를 쌓는다
# ---------------------------------------------------------------------------


def test_call_report_draft_model_returns_only_changed_keys(fake_llm_factory):
    llm = fake_llm_factory([AIMessage(content="계산 결과는 4입니다.")])
    state = {
        "messages": [HumanMessage(content="2+2는?"), AIMessage(content="")],
        "iteration": 1,
        "report_drafts": [],
    }

    result = call_report_draft_model(
        state, llm=llm, system_prompt="간결하게 답하세요.", label="concise"
    )

    assert set(result.keys()) == {"report_drafts", "iteration"}
    assert result["iteration"] == 1  # 델타(operator.add가 누적) — state의 현재값과 무관
    assert result["report_drafts"] == [{"label": "concise", "text": "계산 결과는 4입니다."}]


def test_call_report_draft_model_does_not_touch_messages(fake_llm_factory):
    # draft는 사용자에게 바로 보이는 messages가 아니라 report_drafts에만
    # 쌓인다 — 3개의 draft가 전부 대화창에 노출되면 지저분해지기 때문이다.
    llm = fake_llm_factory([AIMessage(content="draft text")])
    state = {"messages": [HumanMessage(content="hi")], "iteration": 1, "report_drafts": []}

    result = call_report_draft_model(state, llm=llm, system_prompt="p", label="detailed")

    assert "messages" not in result


def test_call_report_draft_model_prepends_given_system_prompt(fake_llm_factory):
    llm = fake_llm_factory([AIMessage(content="ok")])
    state = {"messages": [HumanMessage(content="hi")], "iteration": 0, "report_drafts": []}

    call_report_draft_model(state, llm=llm, system_prompt="이것은 관점 프롬프트", label="action")

    sent_messages = llm.received_messages[0]
    assert isinstance(sent_messages[0], SystemMessage)
    assert sent_messages[0].content == "이것은 관점 프롬프트"
    assert sent_messages[1].content == "hi"


def test_report_angles_has_three_distinct_entries():
    assert len(REPORT_ANGLES) == 3
    keys = [a["key"] for a in REPORT_ANGLES]
    assert len(set(keys)) == 3  # 중복 없이 서로 다른 관점
    prompts = [a["system_prompt"] for a in REPORT_ANGLES]
    assert len(set(prompts)) == 3  # 프롬프트도 서로 달라야 실제로 "다른 의견"이 나온다


# ---------------------------------------------------------------------------
# aggregate_reports_node: LLM을 호출하지 않는 결정론적 종합
# ---------------------------------------------------------------------------


def test_aggregate_reports_node_never_receives_an_llm_argument():
    # 함수 시그니처 자체에 llm 파라미터가 없다 — "LLM을 다시 부르지 않는다"는
    # 설계를 코드 구조로 강제한다 (프롬프트로 당부하는 게 아니라).
    import inspect

    params = inspect.signature(aggregate_reports_node).parameters
    assert "llm" not in params


def test_aggregate_reports_node_orders_sections_by_fixed_angle_order_regardless_of_arrival():
    # 병렬 실행이라 report_drafts 도착 순서가 뒤섞여 있어도(action이 먼저 와도)
    # 출력은 항상 간결 -> 상세 -> 제안 고정 순서여야 한다.
    state = {
        "messages": [],
        "iteration": 3,
        "report_drafts": [
            {"label": "action", "text": "다음엔 이걸 하세요"},
            {"label": "concise", "text": "요약: 4"},
            {"label": "detailed", "text": "1단계... 2단계..."},
        ],
    }

    result = aggregate_reports_node(state)

    assert set(result.keys()) == {"messages"}
    final_text = result["messages"][0].content
    idx_concise = final_text.index("요약: 4")
    idx_detailed = final_text.index("1단계... 2단계...")
    idx_action = final_text.index("다음엔 이걸 하세요")
    assert idx_concise < idx_detailed < idx_action


def test_aggregate_reports_node_handles_missing_angle_gracefully():
    # 어떤 이유로든 draft 하나가 비어도(예: 도구 에러) 예외 없이 나머지로 조합한다.
    state = {
        "messages": [],
        "iteration": 2,
        "report_drafts": [{"label": "concise", "text": "요약만 있음"}],
    }

    result = aggregate_reports_node(state)

    assert "요약만 있음" in result["messages"][0].content


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
