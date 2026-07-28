"""
GitHub 저장소 리뷰 에이전트 - Chainlit 웹 인터페이스.

main.py의 make_llm()/extract_text()를 그대로 재사용한다. 오케스트레이션
로직(build_graph)은 agent/ 패키지에만 있고, 이 파일은 그걸 Chainlit 채팅
UI로 보여주는 인터페이스일 뿐이다.

app.py(Streamlit)와의 차이는 도구 실행 과정을 보여주는 방식이다: 여기서는
graph.invoke() 대신 graph.stream(..., stream_mode="updates")로 그래프를
한 번만 실행하면서, 도구 노드(repo_overview_node/repo_source_node)가
반환한 결과를 cl.Step으로 실시간 시각화한다. invoke()와 stream()을 각각
호출하면 그래프가 두 번 실행되어 LLM 호출 비용이 두 배가 되므로, stream()
결과만으로 최종 상태를 직접 재구성한다(_merge_update).
"""

import chainlit as cl
from langchain_core.messages import HumanMessage, ToolMessage

from agent.graph import build_graph
from main import extract_text, make_llm

# 도구 노드 이름 -> Step에 표시할 라벨. agent/nodes.py의 FANOUT_TOOL_NODES와
# 짝이 맞아야 한다.
TOOL_STEP_LABELS = {
    "repo_overview_node": "📋 저장소 개요",
    "repo_source_node": "🔍 소스 코드 발췌",
}

_graph = None


def get_graph():
    """그래프를 프로세스당 한 번만 조립해서 재사용한다(요청마다 다시 만들지 않는다)."""
    global _graph
    if _graph is None:
        _graph = build_graph(make_llm())
    return _graph


def _merge_update(state: dict, update: dict) -> None:
    """agent.state.AgentState의 reducer와 동일한 규칙으로 update를 state에 반영한다.

    messages는 add_messages처럼 끝에 추가한다. iteration은 이 버전(투표형
    앙상블)에서 operator.add 리듀서를 쓰므로, 각 노드가 절대값이 아니라
    델타(항상 1)를 반환한다 — 그래서 여기서도 마지막 값으로 덮어쓰면 안 되고
    누적(+=)해야 한다. report_drafts도 operator.add(list)라 extend로
    누적한다. 그래프에 체크포인터를 달지 않았으므로, stream()만으로 최종
    상태를 얻으려면 이 병합을 직접 해줘야 한다.
    """
    if "messages" in update:
        state["messages"].extend(update["messages"])
    if "iteration" in update:
        state["iteration"] = state.get("iteration", 0) + update["iteration"]
    if "report_drafts" in update:
        state.setdefault("report_drafts", []).extend(update["report_drafts"])


@cl.on_chat_start
async def on_chat_start():
    # 사람이 보낸 메시지는 처음부터 HumanMessage로 넣어둔다. main.py/app.py처럼
    # ("human", text) 튜플을 넣으면 graph.invoke()가 알아서 정규화해주지만,
    # 여기서는 stream()의 부분 업데이트만으로 state를 직접 관리하기 때문에
    # 그 정규화 과정을 거치지 않는다 — 튜플을 남겨두면 다음 턴에 리스트 중간에
    # Message가 아닌 튜플이 섞여 add_messages의 id 기반 처리와 어긋날 수 있다.
    cl.user_session.set("agent_state", {"messages": [], "iteration": 0, "report_drafts": []})
    await cl.Message(
        content=(
            "🔎 GitHub 저장소 리뷰 에이전트입니다.\n\n"
            "리뷰할 저장소를 owner/repo 형식으로 알려주세요 "
            "(예: `langchain-ai/langgraph 리뷰해줘`)."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    graph = get_graph()
    state = cl.user_session.get("agent_state")
    state["messages"].append(HumanMessage(content=message.content))

    for update in graph.stream(state, stream_mode="updates"):
        for node_name, node_output in update.items():
            if node_name in TOOL_STEP_LABELS:
                for tool_message in node_output.get("messages", []):
                    if isinstance(tool_message, ToolMessage):
                        async with cl.Step(
                            name=TOOL_STEP_LABELS[node_name], type="tool"
                        ) as step:
                            step.output = tool_message.content
            _merge_update(state, node_output)

    cl.user_session.set("agent_state", state)

    final_text = extract_text(state["messages"][-1].content)
    await cl.Message(content=final_text).send()
