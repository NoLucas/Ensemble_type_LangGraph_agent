"""
그래프 노드 로직.

이 에이전트는 model 노드 하나가 아니라 역할이 다른 LLM 노드들을 쓰는
dispatcher -> [calculate_node, read_file_node, write_file_node] (도구
팬아웃) -> [draft_concise_node, draft_detailed_node, draft_action_node]
(보고서 팬아웃, 3-way 앙상블) -> aggregate_reports_node (팬인) -> END 구조다.

- dispatcher(call_dispatcher_model): 사용자의 "첫 입력"을 해석해서 세 도구
  중 필요한 것을 정확하고 효율적으로 지시하는 역할. 여러 도구가 동시에
  필요하면 한 번에 다 요청하도록 프롬프트로 유도해서, 팬아웃이 실제로 병렬
  이득을 보게 만든다.
- report draft 3개(call_report_draft_model): 도구 실행이 끝난 뒤, 그 결과를
  세 가지 다른 관점(REPORT_ANGLES: 간결 요약/상세 설명/실무 제안)으로 각자
  독립적으로 서술하는 역할. 앙상블(⑥) 패턴이되, calculate/read/write 결과는
  "정답이 하나뿐인 사실"이라 temperature로 무작위성을 주면 사실이 흔들릴
  위험이 있다 — 그래서 다양성의 원천을 온도가 아니라 관점(프롬프트)에 둔다.
  각 draft 노드는 바인딩되지 않은 llm을 받아 도구를 다시 호출할 수 없다.
- aggregate_reports_node: 세 draft를 **LLM을 다시 호출하지 않고** 고정된
  순서로 그대로 이어붙여 최종 답변을 만드는 결정론적 함수. "종합"까지
  LLM에게 맡기면 그 종합 단계 자체가 원본 draft에 없던 내용을 지어내는
  환각의 새 진입점이 되므로, 아예 LLM을 빼서 그 위험을 구조적으로
  없앴다 — 함수 시그니처에 llm 파라미터 자체가 없다.

dispatcher가 tool_call을 하나만 냈어도 세 도구 노드를 항상 함께 깨우고,
자기 담당이 아닌 노드는 조용히 통과(pass-through)한다. draft 3개도 마찬가지로
tool 팬인 이후 항상 함께 실행된다. 이렇게 해야 graph.py의 list 기반
add_edge(...) 팬인 조인들이 데드락 없이 동작한다 (조인은 나열된 노드가
"이번 라운드에 모두 실행됐는지"를 기준으로 트리거되므로, 하나라도 조건부로
스킵하면 나머지가 영원히 기다리게 된다).

dispatcher -> 도구 팬아웃 -> draft 팬아웃 -> aggregate -> END는 사이클이 없는
구조라, 이전에 있었던 반복 상한(MAX_ITERATIONS) 같은 무한 루프 방지 장치가
필요 없다.

call_dispatcher_model/call_report_draft_model 모두 llm을 인자로 주입받는다
(전역으로 인스턴스화하지 않는다). 이렇게 해야 테스트에서 FakeChatModel을
넣어 실제 API 호출 없이 노드 동작을 검증할 수 있고, 프로덕션에서는
dispatcher에 tools가 bind_tools된 ChatModel을, draft 노드들에 바인딩되지
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

# 도구 실행 결과를 "어떻게 서술할지"에 대한 세 가지 독립적인 관점.
# 각 관점은 같은 사실(도구 결과)을 다루므로 내용이 갈릴 이유가 없어야 정상이고,
# 다양성은 "무엇을 강조/생략하는가"에서만 나온다 — 그래서 temperature가 아니라
# 프롬프트(system_prompt)로 관점을 나눈다.
REPORT_ANGLES = [
    {
        "key": "concise",
        "node_name": "draft_concise_node",
        "heading": "핵심 요약",
        "system_prompt": """당신은 도구 실행 결과를 핵심만 1~2문장으로
간결하게 요약하는 역할입니다. 지금까지의 대화와 도구 실행 결과(Tool
메시지들)를 바탕으로, 사용자가 가장 먼저 알아야 할 결론만 짧게 정리해서
응답하세요. 도구는 이미 모두 실행됐으니 다시 호출하지 마세요.""",
    },
    {
        "key": "detailed",
        "node_name": "draft_detailed_node",
        "heading": "상세 설명",
        "system_prompt": """당신은 도구 실행 결과를 빠짐없이 단계별로
설명하는 역할입니다. 지금까지의 대화와 도구 실행 결과(Tool 메시지들)를
바탕으로, 어떤 도구가 어떤 입력으로 무엇을 반환했는지 하나도 빠뜨리지 않고
차례대로 서술하세요. 도구는 이미 모두 실행됐으니 다시 호출하지 마세요.""",
    },
    {
        "key": "action",
        "node_name": "draft_action_node",
        "heading": "실무 제안",
        "system_prompt": """당신은 도구 실행 결과를 바탕으로 사용자가 다음에
취할 수 있는 실무적인 행동을 제안하는 역할입니다. 지금까지의 대화와 도구
실행 결과(Tool 메시지들)를 바탕으로, 결과가 의미하는 바와 다음 단계로 무엇을
하면 좋을지 제안하세요. 도구는 이미 모두 실행됐으니 다시 호출하지 마세요.""",
    },
]

# route_after_dispatcher가 도구 팬인 이후 항상 함께 팬아웃하는 draft 노드
# 이름 목록. graph.py의 조인/팬아웃 배선과 반드시 짝이 맞아야 한다.
FANOUT_REPORT_DRAFT_NODES = [angle["node_name"] for angle in REPORT_ANGLES]

_HEADING_BY_KEY = {angle["key"]: angle["heading"] for angle in REPORT_ANGLES}
_ANGLE_ORDER = [angle["key"] for angle in REPORT_ANGLES]


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


def aggregate_reports_node(state: AgentState) -> dict:
    """세 관점의 draft를 LLM 호출 없이 고정된 순서로 이어붙인다.

    report_drafts는 병렬 실행 순서에 따라 도착 순서가 뒤섞일 수 있으므로,
    _ANGLE_ORDER(간결 -> 상세 -> 제안) 기준으로 정렬해서 항상 같은 순서로
    조합한다. draft가 일부 누락돼도(예: 도구 에러로 내용이 부실했던 경우)
    있는 것만으로 조합하며 예외를 던지지 않는다.
    """
    drafts_by_key = {d["label"]: d["text"] for d in state.get("report_drafts", [])}
    sections = []
    for key in _ANGLE_ORDER:
        if key in drafts_by_key:
            heading = _HEADING_BY_KEY[key]
            sections.append(f"### {heading}\n{drafts_by_key[key]}")
    combined = "\n\n".join(sections)
    return {"messages": [AIMessage(content=combined)]}


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
