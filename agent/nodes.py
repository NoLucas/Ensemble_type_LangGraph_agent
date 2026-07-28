"""
그래프 노드 로직.

call_model은 llm을 인자로 주입받는다(전역으로 인스턴스화하지 않는다). 이렇게
해야 테스트에서 FakeChatModel을 넣어 실제 API 호출 없이 노드 동작을 검증할
수 있고, 프로덕션에서는 tools가 bind_tools된 실제 ChatModel을 넣어 그대로
재사용할 수 있다.
"""

from langchain_core.messages import SystemMessage
from langgraph.graph import END

from agent.state import AgentState

SYSTEM_PROMPT = """당신은 계산과 파일 조회를 도와주는 작업 에이전트입니다.
산술 계산이 필요하면 calculate 도구를, 샌드박스 안의 텍스트 파일을 읽어야 하면
read_text_file 도구를 사용하세요. 더 이상 도구가 필요 없다면 최종 답변을 말로
정리해서 응답하세요."""

# 모델이 tool_calls를 계속 반환해도 여기서 강제로 그래프를 종료시킨다.
# 이게 없으면 모델이 같은 도구를 무한히 호출하는 경우 API 비용이 무한정 발생한다.
MAX_ITERATIONS = 10


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


def route_after_model(state: AgentState) -> str:
    """model 노드 다음에 tools로 갈지 종료할지 결정하는 조건부 엣지.

    LLM을 호출하지 않는 순수 함수라 테스트가 빠르고 결정적이다.
    반복 상한(MAX_ITERATIONS)을 먼저 검사해서, 도구 호출이 남아있어도
    상한에 도달했으면 무조건 종료시킨다.
    """
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        return END

    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END
