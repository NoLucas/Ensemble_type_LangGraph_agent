"""
GitHub 저장소 리뷰 에이전트 - Streamlit 웹 인터페이스.

main.py에 정의된 make_llm()/extract_text()를 그대로 재사용한다. 오케스트레이션
로직(build_graph)은 agent/ 패키지에만 있고, 이 파일은 그걸 채팅 UI로
보여주는 인터페이스일 뿐이다 — 그래프 로직을 건드리지 않고도 새로운
인터페이스를 추가할 수 있다는 걸 보여주는 예시이기도 하다.
"""

import streamlit as st

from agent.graph import build_graph
from main import extract_text, make_llm

st.set_page_config(page_title="GitHub 저장소 리뷰 에이전트", page_icon="🔎")
st.title("🔎 GitHub 저장소 리뷰 에이전트")
st.caption("언급된 GitHub 저장소의 개요와 소스 코드를 가져와 요약 + 코드 리뷰를 병렬로 작성하는 LangGraph 에이전트")


@st.cache_resource
def get_graph():
    return build_graph(make_llm())


if "agent_state" not in st.session_state:
    # 그래프에 체크포인터가 없으므로, 대화 기록은 세션 상태가 파이썬
    # 변수로 들고 있다가 매 턴 그래프에 다시 넘겨준다. report_drafts는
    # 3-way 투표형 앙상블(voter_1/2/3)이 채우는 필드라, operator.add
    # 리듀서가 첫 병렬 쓰기부터 안전하게 누적하도록 빈 리스트로 시작한다.
    st.session_state.agent_state = {"messages": [], "iteration": 0, "report_drafts": []}

graph = get_graph()

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
