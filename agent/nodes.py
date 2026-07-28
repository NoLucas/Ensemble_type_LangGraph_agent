"""
그래프 노드 로직.

이 에이전트는 model 노드 하나가 아니라 역할이 다른 두 LLM 노드를 쓰는
dispatcher -> [calculate_node, read_file_node, write_file_node] (병렬
팬아웃) -> reporter (팬인) -> END 구조다.

- dispatcher(call_dispatcher_model): 사용자의 "첫 입력"을 해석해서 세 도구
  중 필요한 것을 정확하고 효율적으로 지시하는 역할. 여러 도구가 동시에
  필요하면 한 번에 다 요청하도록 프롬프트로 유도해서, 팬아웃이 실제로 병렬
  이득을 보게 만든다.
- reporter(call_reporter_model): 팬아웃/팬인으로 도구 실행이 끝난 뒤, 그
  결과를 사용자에게 명확하고 효과적으로 "보고"하는 역할. 프롬프트로 "도구
  다시 부르지 마"라고 당부하는 데 그치지 않고, graph.py에서 tool이
  바인딩되지 않은 llm을 넘겨 애초에 도구 스키마 자체를 모르게 만든다 —
  구조적으로 tool_call을 낼 수 없으므로 역할 경계가 프롬프트 준수 여부에
  의존하지 않는다.

dispatcher가 tool_call을 하나만 냈어도 세 도구 노드를 항상 함께 깨우고,
자기 담당이 아닌 노드는 조용히 통과(pass-through)한다. 이렇게 해야
graph.py의 add_edge(FANOUT_TOOL_NODES, "reporter") 팬인 조인이 데드락 없이
동작한다 (조인은 나열된 노드가 "이번 라운드에 모두 실행됐는지"를 기준으로
트리거되므로, 하나라도 조건부로 스킵하면 나머지가 영원히 기다리게 된다).

dispatcher -> 팬아웃 -> reporter -> END는 사이클이 없는 구조라, 이전에
있었던 반복 상한(MAX_ITERATIONS) 같은 무한 루프 방지 장치가 필요 없다.

두 call_*_model 함수 모두 llm을 인자로 주입받는다(전역으로 인스턴스화하지
않는다). 이렇게 해야 테스트에서 FakeChatModel을 넣어 실제 API 호출 없이
노드 동작을 검증할 수 있고, 프로덕션에서는 dispatcher에 tools가
bind_tools된 ChatModel을, reporter에 바인딩되지 않은 ChatModel을 그대로
넣어 재사용할 수 있다.
"""

from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import END

from agent.state import AgentState
from agent.tools import calculate, make_read_text_file_tool, make_write_text_file_tool

DISPATCHER_SYSTEM_PROMPT = """당신은 사용자 요청을 분석해서 calculate(계산),
read_sandbox_file(파일 읽기), write_sandbox_file(파일 쓰기) 세 도구 중 필요한
것을 정확하고 효율적으로 호출 지시하는 디스패처입니다. 여러 도구가 동시에
필요하면 반드시 한 번의 응답에서 모두 요청해서 병렬로 처리되게 하세요 (도구를
하나씩 순서대로 요청하면 병렬 처리의 이점이 사라집니다). 도구가 전혀 필요
없는 요청이면 도구를 부르지 말고 바로 답변하세요."""

REPORTER_SYSTEM_PROMPT = """당신은 도구 실행 결과를 사용자에게 명확하고
효과적으로 보고하는 리포터입니다. 지금까지의 대화와 도구 실행 결과(Tool
메시지들)를 바탕으로, 사용자가 무엇을 물었고 각 도구가 무엇을 반환했는지
빠짐없이 반영한 최종 답변만 정리해서 응답하세요. 도구는 이미 모두 실행됐으니
다시 호출할 필요가 없습니다."""

# route_after_dispatcher가 팬아웃할 때 항상 함께 반환하는 노드 이름 목록.
# 계산(calculate) / 읽기(read) / 쓰기(write)로 역할을 세분화한 3-way 팬아웃.
# graph.py의 add_edge(FANOUT_TOOL_NODES, "reporter") 팬인과 반드시 짝이
# 맞아야 한다.
FANOUT_TOOL_NODES = ["calculate_node", "read_file_node", "write_file_node"]


def _call_model(state: AgentState, llm, system_prompt: str) -> dict:
    """state 전체가 아니라 변경된 필드(messages, iteration)만 반환한다.

    LangGraph가 이 partial dict를 기존 state에 merge하는데, messages는
    add_messages reducer 덕분에 "덮어쓰기"가 아니라 "추가"로 처리된다.
    dispatcher/reporter가 시스템 프롬프트만 다르고 나머지 로직은 동일해서
    공용 구현으로 뽑아뒀다.
    """
    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
    response = llm.invoke(messages)
    return {
        "messages": [response],
        "iteration": state.get("iteration", 0) + 1,
    }


def call_dispatcher_model(state: AgentState, llm) -> dict:
    return _call_model(state, llm, DISPATCHER_SYSTEM_PROMPT)


def call_reporter_model(state: AgentState, llm) -> dict:
    return _call_model(state, llm, REPORTER_SYSTEM_PROMPT)


def route_after_dispatcher(state: AgentState):
    """dispatcher 다음에 도구를 병렬로 실행할지, reporter 없이 바로 끝낼지
    결정하는 조건부 엣지.

    LLM을 호출하지 않는 순수 함수라 테스트가 빠르고 결정적이다. 도구가
    전혀 필요 없으면 reporter를 거치지 않고 dispatcher의 답변을 그대로
    최종 답변으로 쓴다 — 보고할 도구 결과가 없는데 reporter를 한 번 더
    호출하는 건 불필요한 LLM 호출이기 때문이다.

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
