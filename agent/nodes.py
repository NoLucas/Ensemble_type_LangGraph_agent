"""
그래프 노드 로직.

이 에이전트는 model -> [calculate_node, read_file_node] (병렬 팬아웃)
-> model (팬인) 구조를 쓴다. model이 tool_call을 하나만 냈어도 두 도구
노드를 항상 함께 깨우고, 자기 담당이 아닌 노드는 조용히 통과
(pass-through)한다. 이렇게 해야 graph.py의
add_edge(["calculate_node", "read_file_node"], "model") 팬인 조인이
매 라운드마다 데드락 없이 동작한다 (조인은 나열된 노드가 "이번 라운드에
모두 실행됐는지"를 기준으로 트리거되므로, 한쪽만 조건부로 스킵하면 다른
쪽이 영원히 기다리게 된다).

call_model은 llm을 인자로 주입받는다(전역으로 인스턴스화하지 않는다). 이렇게
해야 테스트에서 FakeChatModel을 넣어 실제 API 호출 없이 노드 동작을 검증할
수 있고, 프로덕션에서는 tools가 bind_tools된 실제 ChatModel을 넣어 그대로
재사용할 수 있다.
"""

from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import END

from agent.state import AgentState
from agent.tools import calculate, make_read_text_file_tool

SYSTEM_PROMPT = """당신은 계산과 파일 조회를 도와주는 작업 에이전트입니다.
산술 계산이 필요하면 calculate 도구를, 샌드박스 안의 텍스트 파일을 읽어야 하면
read_sandbox_file 도구를 사용하세요. 한 번에 여러 도구가 필요하면 동시에
요청해도 됩니다. 더 이상 도구가 필요 없다면 최종 답변을 말로 정리해서
응답하세요."""

# 모델이 tool_calls를 계속 반환해도 여기서 강제로 그래프를 종료시킨다.
# 이게 없으면 모델이 같은 도구를 무한히 호출하는 경우 API 비용이 무한정 발생한다.
MAX_ITERATIONS = 10

# route_after_model이 팬아웃할 때 항상 함께 반환하는 노드 이름 목록.
# graph.py의 add_edge(["calculate_node", "read_file_node"], "model") 팬인과
# 반드시 짝이 맞아야 한다.
FANOUT_TOOL_NODES = ["calculate_node", "read_file_node"]


def call_model(state: AgentState, llm) -> dict:
    """state 전체가 아니라 변경된 필드(messages, iteration)만 반환한다.

    LangGraph가 이 partial dict를 기존 state에 merge하는데, messages는
    add_messages reducer 덕분에 "덮어쓰기"가 아니라 "추가"로 처리된다.
    """
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
    response = llm.invoke(messages)
    return {
        "messages": [response],
        "iteration": state.get("iteration", 0) + 1,
    }


def route_after_model(state: AgentState):
    """model 노드 다음에 도구를 병렬로 실행할지 종료할지 결정하는 조건부 엣지.

    LLM을 호출하지 않는 순수 함수라 테스트가 빠르고 결정적이다.
    반복 상한(MAX_ITERATIONS)을 먼저 검사해서, 도구 호출이 남아있어도
    상한에 도달했으면 무조건 종료시킨다.

    tool_call이 하나만 요청됐어도 FANOUT_TOOL_NODES 전체를 반환한다 —
    calculate_node/read_file_node 각각이 자기 몫이 없으면 통과하는 방식으로
    "항상 둘 다 병렬로 깨운다"는 팬인 조인의 전제를 지킨다.
    """
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        return END

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
