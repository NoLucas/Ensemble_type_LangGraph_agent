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
import sniffio
from langchain_core.messages import HumanMessage, ToolMessage

from agent.graph import build_graph
from main import extract_text, make_llm

# chainlit run은 시작할 때 nest_asyncio.apply()로 이벤트 루프를 재진입
# 가능하게 패치하는데, 이 패치가 asyncio.current_task() 기반 태스크 추적을
# 깨뜨려서 sniffio.current_async_library()가 실행 중인 코루틴 안에서도
# "async 라이브러리를 찾지 못했다"고 오판한다. 그 결과 anyio가 파일
# 정적 응답(FileResponse -> anyio.to_thread.run_sync)마다
# NoEventLoopError를 던져 프론트엔드 정적 자산(SPA index.html/JS 번들)이
# 전혀 로드되지 않는다 — chainlit/nest_asyncio/anyio 4.x 조합에서 발생하는
# 환경 차원의 알려진 비호환 문제다. sniffio가 공식으로 노출하는
# contextvar를 직접 세팅해서 "현재 asyncio다"를 강제하면, 매 요청마다
# 벌어지는 태스크 기반 자동 감지(고장 난 경로) 대신 이 값을 먼저 읽으므로
# 문제를 우회할 수 있다. 이 프로세스는 항상 uvicorn(asyncio) 위에서만
# 돌기 때문에 값을 고정해도 안전하다.
sniffio.current_async_library_cvar.set("asyncio")

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


# 채팅 입력창 위에 뜨는 클릭형 예시 프롬프트. 저장소를 직접 타이핑하기
# 귀찮은 사용자를 위한 진입점일 뿐, 그래프 로직에는 영향을 주지 않는다.
@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="작은 저장소로 빠르게 체험",
            message="octocat/Hello-World 저장소 개요만 알려줘.",
        ),
        cl.Starter(
            label="요약 + 코드 리뷰 둘 다",
            message="langchain-ai/langgraph 저장소 요약이랑 코드 리뷰 둘 다 해줘.",
        ),
        cl.Starter(
            label="저장소 두 개 동시 비교",
            message="octocat/Hello-World랑 octocat/Spoon-Knife 둘 다 리뷰해줘.",
        ),
    ]


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
            "리뷰할 저장소를 `owner/repo` 형식으로 알려주세요 "
            "(예: `langchain-ai/langgraph 리뷰해줘`). "
            "저장소를 여러 개 언급하면 병렬로 처리됩니다."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    graph = get_graph()
    state = cl.user_session.get("agent_state")
    state["messages"].append(HumanMessage(content=message.content))

    try:
        for update in graph.stream(state, stream_mode="updates"):
            for node_name, node_output in update.items():
                if node_output is None:
                    # 이번 라운드에 이 노드가 담당할 tool_call이 없어서
                    # {}(빈 dict)를 반환했을 때, LangGraph는 stream_mode=
                    # "updates"에서 그 기여분을 None으로 보고한다(빈 dict가
                    # 아니라). repo_overview_node/repo_source_node는 항상
                    # 함께 깨워지지만 자기 몫이 없으면 통과하므로(nodes.py의
                    # "팬인 조인" 설계), 실제로 매 요청마다 발생하는 정상
                    # 케이스다 — 반영할 상태 변화가 없으므로 건너뛴다.
                    continue
                if node_name in TOOL_STEP_LABELS:
                    for tool_message in node_output.get("messages", []):
                        if isinstance(tool_message, ToolMessage):
                            async with cl.Step(
                                name=TOOL_STEP_LABELS[node_name], type="tool"
                            ) as step:
                                step.output = tool_message.content
                _merge_update(state, node_output)
    except Exception as exc:
        # LLM 호출 자체가 실패하는 경우(API 키 누락/오류, 요금 한도, 일시적
        # 장애 등)는 도구처럼 "Error: ..." 문자열로 정상 반환되지 않고
        # 예외로 그래프 밖까지 새어나온다. 여기서 잡지 않으면 Chainlit이
        # 스택 트레이스를 그대로 노출하므로, 사용자가 이해할 수 있는 메시지로
        # 바꿔서 보여준다. 실패한 human 메시지를 state에 남겨두면 다음 턴에
        # 같은 요청이 중복 전송되므로 되돌린다.
        state["messages"].pop()
        cl.user_session.set("agent_state", state)
        await cl.Message(
            content=f"⚠️ 리뷰 중 오류가 발생했습니다: {exc}\n\n"
            "ANTHROPIC_API_KEY가 올바른지, 네트워크 연결이 되는지 확인한 뒤 다시 시도해주세요."
        ).send()
        return

    cl.user_session.set("agent_state", state)

    final_text = extract_text(state["messages"][-1].content)
    await cl.Message(content=final_text).send()
