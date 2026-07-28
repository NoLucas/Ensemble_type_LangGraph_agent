"""
그래프 노드 로직.

이 에이전트는 model 노드 하나가 아니라 역할이 다른 LLM 노드들을 쓰는
dispatcher -> [calculate_node, read_file_node, write_file_node] (도구
팬아웃) -> [voter_1_node, voter_2_node, voter_3_node] (투표형 앙상블
팬아웃) -> vote_for_best_report_node (팬인) -> END 구조다.

- dispatcher(call_dispatcher_model): 사용자의 "첫 입력"을 해석해서 세 도구
  중 필요한 것을 정확하고 효율적으로 지시하는 역할. 여러 도구가 동시에
  필요하면 한 번에 다 요청하도록 프롬프트로 유도해서, 팬아웃이 실제로 병렬
  이득을 보게 만든다.
- voter 3개(call_report_draft_model): 도구 실행이 끝난 뒤, **동일한
  프롬프트(VOTER_SYSTEM_PROMPT)**로 최종 보고서를 각자 독립적으로 3번
  시도하는 역할. 이전 버전은 "관점을 다르게" 하는 앙상블이었지만, 이번
  버전은 "같은 과제를 여러 번 독립 시도해서 다수결로 검증"하는 투표
  (voting) 앙상블이다 — 실제 서비스에서는 모델 샘플링의 자연스러운
  변동성(temperature)이 다양성의 원천이 된다. 각 voter는 바인딩되지 않은
  llm을 받아 도구를 다시 호출할 수 없다.
- vote_for_best_report_node: 세 voter의 독립 시도 중, **도구 실행 결과
  (ToolMessage 내용)를 실제로 정확히 포함한 candidate만 통과**시키고 그중
  하나를 결정론적 규칙으로 고르는 함수. "어느 게 더 그럴듯한가"를 LLM
  판사에게 다시 묻지 않는다 — 정답(도구 결과)을 이미 알고 있으므로, 각
  candidate가 그 정답을 실제로 언급했는지 문자열 포함 여부로 계산하고,
  가장 많은 사실을 포함한 candidate를 선택한다. 함수 시그니처에 llm
  파라미터 자체가 없어서, "종합 단계에서 LLM이 새 내용을 지어내는" 환각
  경로가 구조적으로 존재하지 않는다.

dispatcher가 tool_call을 하나만 냈어도 세 도구 노드를 항상 함께 깨우고,
자기 담당이 아닌 노드는 조용히 통과(pass-through)한다. voter 3개도 마찬가지로
tool 팬인 이후 항상 함께 실행된다. 이렇게 해야 graph.py의 list 기반
add_edge(...) 팬인 조인들이 데드락 없이 동작한다 (조인은 나열된 노드가
"이번 라운드에 모두 실행됐는지"를 기준으로 트리거되므로, 하나라도 조건부로
스킵하면 나머지가 영원히 기다리게 된다).

dispatcher -> 도구 팬아웃 -> voter 팬아웃 -> vote -> END는 사이클이 없는
구조라, 이전에 있었던 반복 상한(MAX_ITERATIONS) 같은 무한 루프 방지 장치가
필요 없다.

call_dispatcher_model/call_report_draft_model 모두 llm을 인자로 주입받는다
(전역으로 인스턴스화하지 않는다). 이렇게 해야 테스트에서 FakeChatModel을
넣어 실제 API 호출 없이 노드 동작을 검증할 수 있고, 프로덕션에서는
dispatcher에 tools가 bind_tools된 ChatModel을, voter 노드들에 바인딩되지
않은 ChatModel을 그대로 넣어 재사용할 수 있다.
"""

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.graph import END

from agent.state import AgentState
from agent.tools import calculate, make_read_text_file_tool, make_write_text_file_tool

DISPATCHER_SYSTEM_PROMPT = """당신은 사용자 요청을 분석해서 calculate(계산),
read_sandbox_file(파일 읽기), write_sandbox_file(파일 쓰기) 세 도구 중 필요한
것을 정확하고 효율적으로 호출 지시하는 디스패처입니다. 여러 도구가 동시에
필요하면 반드시 한 번의 응답에서 모두 요청해서 병렬로 처리되게 하세요 (도구를
하나씩 순서대로 요청하면 병렬 처리의 이점이 사라집니다). 도구가 전혀 필요
없는 요청이면 도구를 부르지 말고 바로 답변하세요."""

# route_after_dispatcher가 팬아웃할 때 항상 함께 반환하는 노드 이름 목록.
# 계산(calculate) / 읽기(read) / 쓰기(write)로 역할을 세분화한 3-way 팬아웃.
# graph.py의 add_edge(FANOUT_TOOL_NODES, ...) 팬인과 반드시 짝이 맞아야 한다.
FANOUT_TOOL_NODES = ["calculate_node", "read_file_node", "write_file_node"]

# 세 voter가 공유하는 단일 프롬프트. 관점을 나누지 않는다 — 투표형
# 앙상블은 "같은 과제를 독립적으로 여러 번 시도"하는 것이지 "다른 역할을
# 나눠 맡는" 것이 아니다. 다양성은 프롬프트가 아니라 모델 샘플링 자체의
# 변동성(temperature)에서 나온다.
VOTER_SYSTEM_PROMPT = """당신은 도구 실행 결과를 사용자에게 명확하게
보고하는 리포터입니다. 지금까지의 대화와 도구 실행 결과(Tool 메시지들)를
바탕으로, 사용자가 무엇을 물었고 각 도구가 무엇을 반환했는지 빠짐없이
반영한 최종 답변을 정리해서 응답하세요. 도구는 이미 모두 실행됐으니 다시
호출할 필요가 없습니다."""

# 세 독립 시도를 구분하는 라벨. 동점일 때 이 순서대로 타이브레이크한다.
VOTER_LABELS = ["voter_1", "voter_2", "voter_3"]

# route_after_dispatcher가 도구 팬인 이후 항상 함께 팬아웃하는 voter 노드
# 이름 목록. graph.py의 조인/팬아웃 배선과 반드시 짝이 맞아야 한다.
FANOUT_VOTE_NODES = [f"{label}_node" for label in VOTER_LABELS]


def call_dispatcher_model(state: AgentState, llm) -> dict:
    """state 전체가 아니라 변경된 필드(messages, iteration)만 반환한다.

    LangGraph가 이 partial dict를 기존 state에 merge하는데, messages는
    add_messages reducer 덕분에 "덮어쓰기"가 아니라 "추가"로 처리된다.
    """
    messages = [SystemMessage(content=DISPATCHER_SYSTEM_PROMPT)] + list(state["messages"])
    response = llm.invoke(messages)
    return {
        "messages": [response],
        "iteration": 1,  # operator.add 리듀서가 누적하는 델타(항상 +1)
    }


def call_report_draft_model(state: AgentState, llm, system_prompt: str, label: str) -> dict:
    """관점 하나(label)를 맡아 report_drafts에 항목 하나를 쌓는다.

    messages는 건드리지 않는다 — draft 3개가 전부 대화창에 노출되면
    지저분해지므로, 최종 사용자 메시지는 aggregate_reports_node만 만든다.
    report_drafts는 operator.add reducer(state.py)로 누적되므로, 병렬
    실행되는 세 draft 노드가 서로의 항목을 덮어쓰지 않는다.
    """
    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
    response = llm.invoke(messages)
    text = response.content if isinstance(response.content, str) else str(response.content)
    return {
        "report_drafts": [{"label": label, "text": text}],
        "iteration": 1,  # operator.add 리듀서가 누적하는 델타(항상 +1).
        # draft 3개가 병렬로 같은 state 스냅샷을 읽으므로, "현재값+1" 절대값
        # 방식은 서로 충돌하거나 값이 유실된다. 델타를 더하는 방식이어야
        # 실행 순서와 무관하게 "총 LLM 호출 수"가 정확히 합산된다.
    }


def vote_for_best_report_node(state: AgentState) -> dict:
    """세 voter의 독립 시도 중, 도구 실행 결과를 가장 정확히 반영한 하나를
    LLM 호출 없이 결정론적으로 고른다.

    "어느 draft가 더 그럴듯한가"를 다시 LLM에게 묻는 대신, 이미 알고 있는
    정답(ToolMessage 내용)이 각 candidate 텍스트에 실제로 포함되는지 세어서
    점수를 매긴다 — 점수가 가장 높은 candidate가 사실을 가장 정확히
    반영했다고 보는 것이다. 동점이면 VOTER_LABELS 순서상 먼저 나오는 voter를
    선택해서 실행할 때마다 승자가 무작위로 바뀌지 않게 한다. voter 일부가
    누락돼도(예: 도구 에러) 있는 candidate만으로 투표하며, 아무도 정답을
    맞히지 못해도 예외 없이 첫 번째 voter의 답을 반환한다("아무도 못 맞혀도
    침묵하지 않는다").
    """
    required_facts = [m.content for m in state["messages"] if isinstance(m, ToolMessage)]
    drafts_by_label = {d["label"]: d["text"] for d in state.get("report_drafts", [])}

    def score(text: str) -> int:
        return sum(1 for fact in required_facts if fact and fact in text)

    best_label = None
    best_score = -1
    for label in VOTER_LABELS:
        if label not in drafts_by_label:
            continue
        candidate_score = score(drafts_by_label[label])
        if candidate_score > best_score:
            best_score = candidate_score
            best_label = label

    if best_label is None:
        return {"messages": [AIMessage(content="보고서를 생성하지 못했습니다.")]}
    return {"messages": [AIMessage(content=drafts_by_label[best_label])]}


def route_after_dispatcher(state: AgentState):
    """dispatcher 다음에 도구를 병렬로 실행할지, 보고서 단계 없이 바로 끝낼지
    결정하는 조건부 엣지.

    LLM을 호출하지 않는 순수 함수라 테스트가 빠르고 결정적이다. 도구가
    전혀 필요 없으면 draft/aggregate 단계를 거치지 않고 dispatcher의 답변을
    그대로 최종 답변으로 쓴다 — 보고할 도구 결과가 없는데 draft 3개를 또
    돌리는 건 불필요한 LLM 호출이기 때문이다.

    tool_call이 하나만 요청됐어도 FANOUT_TOOL_NODES 전체를 반환한다 —
    calculate_node/read_file_node/write_file_node 각각이 자기 몫이 없으면
    통과하는 방식으로 "항상 셋 다 병렬로 깨운다"는 팬인 조인의 전제를
    지킨다.
    """
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return FANOUT_TOOL_NODES
    return END


def _run_matching_tool_calls(state: AgentState, tool_name: str, tool_fn) -> dict:
    """마지막 AIMessage의 tool_calls 중 tool_name과 일치하는 것만 실행한다.

    담당 tool_call이 하나도 없으면 빈 dict를 반환해 상태를 건드리지 않는다
    (팬아웃 구조에서 "내 몫이 없으면 통과"하는 표준 동작).
    LLM이 도구 스키마와 맞지 않는 args를 만들어낼 수 있으므로(신뢰할 수 없는
    외부 입력), tool_fn.invoke()가 예외를 던지더라도 그래프 전체가 죽지
    않도록 에러 메시지로 변환해서 반환한다.
    """
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []
    matching = [tc for tc in tool_calls if tc["name"] == tool_name]
    if not matching:
        return {}

    results = []
    for tc in matching:
        try:
            output = tool_fn.invoke(tc["args"])
        except Exception as exc:  # LLM이 준 args가 도구 스키마와 안 맞을 수 있다.
            output = f"Error: 도구 호출에 실패했습니다 ({exc})"
        results.append(ToolMessage(content=str(output), name=tool_name, tool_call_id=tc["id"]))

    return {"messages": results}


def calculate_node(state: AgentState) -> dict:
    return _run_matching_tool_calls(state, "calculate", calculate)


def read_file_node(sandbox_dir):
    """sandbox_dir에 고정된 read_file_node 함수를 만든다.

    read_sandbox_file 도구 자체가 base_dir을 클로저로 고정해야 하는 것과
    같은 이유로, 그래프 노드도 sandbox_dir을 미리 받아 고정해둔다.
    """
    read_tool = make_read_text_file_tool(sandbox_dir)

    def _node(state: AgentState) -> dict:
        return _run_matching_tool_calls(state, "read_sandbox_file", read_tool)

    return _node


def write_file_node(sandbox_dir):
    """sandbox_dir에 고정된 write_file_node 함수를 만든다. read_file_node와
    동일한 클로저 패턴이다."""
    write_tool = make_write_text_file_tool(sandbox_dir)

    def _node(state: AgentState) -> dict:
        return _run_matching_tool_calls(state, "write_sandbox_file", write_tool)

    return _node
