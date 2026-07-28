"""
노드 단위 테스트.

이 에이전트는 model 노드 하나가 아니라 역할이 다른 LLM 노드들을 쓴다.

- dispatcher(call_dispatcher_model): 사용자가 언급한 GitHub 저장소를 받아
  fetch_repo_overview/fetch_repo_source_sample 중 필요한 것을 효율적으로
  지시하는 역할. tool_calls를 낼 수 있도록 tool-bound llm을 받는다.
- voter 3개(call_report_draft_model): 도구 실행이 끝난 뒤, **동일한
  프롬프트(VOTER_SYSTEM_PROMPT)**로 최종 리뷰를 각자 독립적으로 3번
  시도하는 역할 — 관점을 다르게 하는 앙상블이 아니라, 같은 과제를 여러 번
  독립 시도해서 다수결로 검증하는 투표(voting) 앙상블이다. 실제 서비스에서는
  모델 샘플링 자체의 변동성(temperature)이 다양성의 원천이 된다.
- vote_for_best_report_node: 3개의 독립 시도 중 **도구 실행 결과(ToolMessage
  내용)를 실제로 정확히 포함한 draft만 통과**시키고, 그중 하나를 결정론적
  규칙으로 선택하는 함수. **LLM을 다시 호출하지 않는다** — 어떤 draft가
  사실과 일치하는지는 이미 알고 있는 도구 결과와 문자열 포함 여부로 검증할
  수 있으므로, "어느 게 더 그럴듯한가"를 LLM 판사에게 다시 묻지 않고
  코드로 결정론적으로 판정한다.

세 voter 노드 모두 실제 LLM을 호출하지 않고 conftest.py의 FakeChatModel을
주입해서, (1) state 전체가 아니라 변경된 필드만 반환하는지, (2) iteration을
정확히 1씩 증가시키는지, (3) 매 호출마다 시스템 프롬프트를 포함시키는지
검증한다.

repo_overview_node/repo_source_node는 팬아웃 구조의 "담당 도구별 병렬
노드"다. dispatcher가 반환한 마지막 AIMessage의 tool_calls 중 자기 이름과
일치하는 것만 실행하고, 없으면 아무 것도 반환하지 않는다(pass-through) —
이 두 노드는 tool_call을 하나만 냈어도 항상 함께 깨워지기 때문에, 자기
몫이 없을 때 조용히 통과하는 능력이 팬인 조인의 전제조건이다. 실제 GitHub
API 호출은 mock_github_get(conftest.py)으로 대체한다.

"노드는 partial state만 반환해야 한다"는 원칙은 add_messages reducer가
정상 동작하기 위한 전제조건이라 별도 테스트로 고정해둔다 (전체 state를
반환하면 메시지 리스트가 이중으로 누적되는 조용한 버그가 생긴다).
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.nodes import (
    DISPATCHER_SYSTEM_PROMPT,
    VOTER_LABELS,
    VOTER_SYSTEM_PROMPT,
    call_dispatcher_model,
    call_report_draft_model,
    repo_overview_node,
    repo_source_node,
    vote_for_best_report_node,
)
from tests.conftest import FakeResponse


def test_call_dispatcher_model_returns_only_changed_keys(fake_llm_factory):
    llm = fake_llm_factory([AIMessage(content="4")])
    state = {"messages": [HumanMessage(content="octocat/hello-world 리뷰해줘")], "iteration": 0}

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
# call_report_draft_model: 독립 시도 하나를 맡아 report_drafts에 항목 하나를 쌓는다
# ---------------------------------------------------------------------------


def test_call_report_draft_model_returns_only_changed_keys(fake_llm_factory):
    llm = fake_llm_factory([AIMessage(content="요약: ...\n코드 리뷰: ...")])
    state = {
        "messages": [HumanMessage(content="octocat/hello-world 리뷰해줘"), AIMessage(content="")],
        "iteration": 1,
        "report_drafts": [],
    }

    result = call_report_draft_model(
        state, llm=llm, system_prompt="간결하게 답하세요.", label="concise"
    )

    assert set(result.keys()) == {"report_drafts", "iteration"}
    assert result["iteration"] == 1  # 델타(operator.add가 누적) — state의 현재값과 무관
    assert result["report_drafts"] == [
        {"label": "concise", "text": "요약: ...\n코드 리뷰: ..."}
    ]


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


def test_voter_labels_has_three_distinct_entries():
    assert len(VOTER_LABELS) == 3
    assert len(set(VOTER_LABELS)) == 3  # 중복 없이 서로 다른 후보 식별자


def test_all_voters_share_the_same_system_prompt():
    # 투표형 앙상블은 "다른 관점"이 아니라 "같은 과제의 독립 시도"다.
    # 프롬프트가 voter마다 다르면 그건 더 이상 투표가 아니라 지난번의
    # 관점 앙상블로 되돌아간 것이므로, 단일 프롬프트임을 고정해둔다.
    assert isinstance(VOTER_SYSTEM_PROMPT, str) and len(VOTER_SYSTEM_PROMPT) > 0


# ---------------------------------------------------------------------------
# vote_for_best_report_node: LLM을 호출하지 않는 결정론적 다수결 선택
# ---------------------------------------------------------------------------


def test_vote_for_best_report_node_never_receives_an_llm_argument():
    # 함수 시그니처 자체에 llm 파라미터가 없다 — "판사 LLM을 또 부르지 않는다"는
    # 설계를 코드 구조로 강제한다 (프롬프트로 당부하는 게 아니라).
    import inspect

    params = inspect.signature(vote_for_best_report_node).parameters
    assert "llm" not in params


def test_vote_for_best_report_node_picks_candidate_containing_all_tool_facts():
    # ToolMessage(도구 실행 결과)에 있는 사실("stars: 42")을 실제로 포함한
    # candidate만 승자가 될 수 있다. voter_2/voter_3는 이를 언급하지
    # 않으므로 탈락한다.
    state = {
        "messages": [
            HumanMessage(content="octocat/hello-world 리뷰해줘"),
            AIMessage(content="", tool_calls=[]),
            ToolMessage(content="stars: 42", name="fetch_repo_overview", tool_call_id="call_1"),
        ],
        "iteration": 4,
        "report_drafts": [
            {"label": "voter_1", "text": "이 저장소는 stars: 42를 받았습니다."},
            {"label": "voter_2", "text": "죄송하지만 정보를 확인하지 못했습니다."},
            {"label": "voter_3", "text": "리뷰를 시도했습니다."},
        ],
    }

    result = vote_for_best_report_node(state)

    assert set(result.keys()) == {"messages"}
    assert result["messages"][0].content == "이 저장소는 stars: 42를 받았습니다."


def test_vote_for_best_report_node_breaks_ties_by_fixed_voter_order():
    # 여러 candidate가 모두 사실을 포함해 동점이면, VOTER_LABELS에서 가장
    # 먼저 나오는 voter를 선택한다 — 매 실행마다 승자가 무작위로 바뀌면
    # 안 되므로 순서 기반 타이브레이크가 필요하다.
    state = {
        "messages": [ToolMessage(content="stars: 42", name="fetch_repo_overview", tool_call_id="call_1")],
        "iteration": 4,
        "report_drafts": [
            {"label": "voter_3", "text": "stars: 42"},
            {"label": "voter_1", "text": "이 저장소의 stars: 42입니다"},
            {"label": "voter_2", "text": "stars: 42가 나왔습니다"},
        ],
    }

    result = vote_for_best_report_node(state)

    assert result["messages"][0].content == "이 저장소의 stars: 42입니다"


def test_vote_for_best_report_node_handles_missing_candidate_gracefully():
    # voter 하나가 어떤 이유로든 draft를 못 남겨도(예: 도구 에러) 예외 없이
    # 나머지 후보만으로 투표를 진행한다.
    state = {
        "messages": [ToolMessage(content="stars: 42", name="fetch_repo_overview", tool_call_id="call_1")],
        "iteration": 2,
        "report_drafts": [{"label": "voter_2", "text": "stars: 42"}],
    }

    result = vote_for_best_report_node(state)

    assert "42" in result["messages"][0].content


def test_vote_for_best_report_node_falls_back_when_no_candidate_matches_facts():
    # 셋 다 정답을 포함하지 못해도(예: 모두 실패) 예외를 던지지 않고 첫 번째
    # voter의 답을 그대로 반환한다 — "아무도 못 맞혀도 침묵하지 않는다".
    state = {
        "messages": [ToolMessage(content="stars: 42", name="fetch_repo_overview", tool_call_id="call_1")],
        "iteration": 4,
        "report_drafts": [
            {"label": "voter_1", "text": "모르겠습니다"},
            {"label": "voter_2", "text": "아마도요"},
            {"label": "voter_3", "text": "확인할 수 없습니다"},
        ],
    }

    result = vote_for_best_report_node(state)

    assert result["messages"][0].content == "모르겠습니다"


# ---------------------------------------------------------------------------
# repo_overview_node: fetch_repo_overview tool_call만 담당, 나머지는 통과
# ---------------------------------------------------------------------------


def test_repo_overview_node_executes_matching_tool_call(mock_github_get):
    mock_github_get(
        [
            FakeResponse(200, json_data={"full_name": "octocat/hello-world"}),
            FakeResponse(200, text="README 내용"),
        ]
    )
    ai_message = AIMessage(
        content="",
        tool_calls=[
            {"name": "fetch_repo_overview", "args": {"repo": "octocat/hello-world"}, "id": "call_1"}
        ],
    )
    state = {"messages": [HumanMessage(content="리뷰해줘"), ai_message], "iteration": 1}

    result = repo_overview_node(state)

    assert len(result["messages"]) == 1
    tool_message = result["messages"][0]
    assert isinstance(tool_message, ToolMessage)
    assert "octocat/hello-world" in tool_message.content
    assert tool_message.tool_call_id == "call_1"


def test_repo_overview_node_passes_through_when_no_matching_tool_call():
    ai_message = AIMessage(
        content="",
        tool_calls=[
            {"name": "fetch_repo_source_sample", "args": {"repo": "a/b"}, "id": "call_1"}
        ],
    )
    state = {"messages": [HumanMessage(content="리뷰해줘"), ai_message], "iteration": 1}

    result = repo_overview_node(state)

    assert result == {}


def test_repo_overview_node_returns_error_message_for_invalid_args_without_raising():
    ai_message = AIMessage(
        content="",
        tool_calls=[{"name": "fetch_repo_overview", "args": {"wrong_key": "a/b"}, "id": "call_1"}],
    )
    state = {"messages": [HumanMessage(content="리뷰해줘"), ai_message], "iteration": 1}

    result = repo_overview_node(state)  # 예외 없이 에러 메시지로 반환되어야 한다

    assert result["messages"][0].content.startswith("Error")


# ---------------------------------------------------------------------------
# repo_source_node: fetch_repo_source_sample tool_call만 담당, 나머지는 통과
# ---------------------------------------------------------------------------


def test_repo_source_node_executes_matching_tool_call(mock_github_get):
    mock_github_get(
        [
            FakeResponse(200, json_data={"default_branch": "main"}),
            FakeResponse(200, json_data={"tree": [{"path": "main.py", "type": "blob"}]}),
            FakeResponse(200, text="print('hi')"),
        ]
    )
    ai_message = AIMessage(
        content="",
        tool_calls=[
            {"name": "fetch_repo_source_sample", "args": {"repo": "octocat/hello-world"}, "id": "call_2"}
        ],
    )
    state = {"messages": [HumanMessage(content="리뷰해줘"), ai_message], "iteration": 1}

    result = repo_source_node(state)

    assert len(result["messages"]) == 1
    tool_message = result["messages"][0]
    assert "print('hi')" in tool_message.content
    assert tool_message.tool_call_id == "call_2"


def test_repo_source_node_passes_through_when_no_matching_tool_call():
    ai_message = AIMessage(
        content="",
        tool_calls=[{"name": "fetch_repo_overview", "args": {"repo": "a/b"}, "id": "call_1"}],
    )
    state = {"messages": [HumanMessage(content="리뷰해줘"), ai_message], "iteration": 1}

    result = repo_source_node(state)

    assert result == {}
