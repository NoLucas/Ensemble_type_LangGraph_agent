"""
GitHub 저장소 리뷰 에이전트 - Streamlit 웹 인터페이스.

main.py에 정의된 make_llm()/extract_text()를 그대로 재사용한다. 오케스트레이션
로직(build_graph)은 agent/ 패키지에만 있고, 이 파일은 그걸 채팅 UI로
보여주는 인터페이스일 뿐이다 — 그래프 로직을 건드리지 않고도 새로운
인터페이스를 추가할 수 있다는 걸 보여주는 예시이기도 하다.
"""

import streamlit as st

from agent.graph import build_graph
from main import NORMAL_NUM_VOTERS, STUDY_MODE_NUM_VOTERS, extract_text, make_llm

st.set_page_config(page_title="GitHub 저장소 리뷰 에이전트", page_icon="🔎")
st.title("🔎 GitHub 저장소 리뷰 에이전트")
st.caption("언급된 GitHub 저장소의 개요와 소스 코드를 가져와 요약 + 코드 리뷰를 병렬로 작성하는 LangGraph 에이전트")


@st.cache_resource
def get_graph(num_voters: int):
    # num_voters별로 캐시가 갈라진다(st.cache_resource가 인자값으로 키를
    # 만든다) — 모드를 토글해도 이전 그래프를 재사용하지 않고 새로 만든다.
    return build_graph(make_llm(), num_voters=num_voters)


with st.sidebar:
    st.subheader("모드")
    study_mode = st.toggle(
        "스터디 모드",
        value=False,
        help=(
            "다수결 검증(voter 3개) 대신 voter 1개만 써서 LLM 호출을 "
            "절반 이하로 줄입니다. 대형 저장소를 가볍게 훑어보며 공부할 때 "
            "추천합니다 — 정확도(다수결 검증)보다 토큰 비용을 우선합니다."
        ),
    )
    if study_mode:
        st.caption("💡 '구조만 보여줘'로 가볍게 훑고, 관심 파일을 콕 집어 드릴다운해보세요.")

num_voters = STUDY_MODE_NUM_VOTERS if study_mode else NORMAL_NUM_VOTERS

if "agent_state" not in st.session_state:
    # 그래프에 체크포인터가 없으므로, 대화 기록은 세션 상태가 파이썬
    # 변수로 들고 있다가 매 턴 그래프에 다시 넘겨준다. report_drafts는
    # 투표형 앙상블(voter_1..voter_N)이 채우는 필드라, operator.add
    # 리듀서가 첫 병렬 쓰기부터 안전하게 누적하도록 빈 리스트로 시작한다.
    st.session_state.agent_state = {"messages": [], "iteration": 0, "report_drafts": []}

graph = get_graph(num_voters)

# 기존 대화 렌더링. ToolMessage 등 내부 처리 메시지는 사용자에게 보여주지
# 않고, human/ai 메시지만 채팅창에 표시한다.
for message in st.session_state.agent_state["messages"]:
    role = getattr(message, "type", None)
    if role == "human":
        with st.chat_message("user"):
            st.write(message.content)
    elif role == "ai":
        text = extract_text(message.content)
        if text:
            with st.chat_message("assistant"):
                st.write(text)

user_input = st.chat_input("리뷰할 저장소를 입력하세요 (예: 'langchain-ai/langgraph 리뷰해줘')")
if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    st.session_state.agent_state["messages"].append(("human", user_input))
    with st.spinner("처리 중..."):
        st.session_state.agent_state = graph.invoke(st.session_state.agent_state)

    final_message = st.session_state.agent_state["messages"][-1]
    with st.chat_message("assistant"):
        st.write(extract_text(final_message.content))
