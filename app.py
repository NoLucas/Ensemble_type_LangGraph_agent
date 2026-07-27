"""
Java 학습 도우미 - Streamlit 웹 인터페이스

실행:
    streamlit run app.py

이 파일은 main.py의 그래프(build_graph())를 그대로 재사용하면서,
콘솔 input()/print() 대신 브라우저에서 동작하는 웹 폼으로 감싼 것이다.
즉 LangGraph 그래프 자체(노드/엣지/상태)는 main.py에 그대로 있고,
이 파일은 "그래프를 어떻게 호출하고 결과를 어떻게 보여줄 것인가"만 담당한다.
"""

import streamlit as st

# main.py에서 그래프를 만드는 함수만 가져와 재사용한다.
# (main.py를 import하면 그 파일의 최상위 코드도 실행되지만,
#  main.py의 실행 로직은 `if __name__ == "__main__":` 안에 있으므로
#  이 파일에서 import할 때는 실행되지 않고 build_graph 정의만 가져오게 된다.)
from main import build_graph

# 브라우저 탭 제목, 파비콘(이모지), 레이아웃(넓게)을 설정한다.
# 반드시 스크립트의 다른 st.* 호출보다 먼저 한 번만 호출해야 한다.
st.set_page_config(page_title="Java 학습 도우미", page_icon="📚", layout="wide")

# ---------------------------------------------------------------------------
# st.session_state: Streamlit에서 "새로고침(rerun)되어도 유지되는" 상태 저장소.
# ---------------------------------------------------------------------------
# Streamlit은 사용자가 버튼을 누르거나 폼을 제출할 때마다 이 파일 전체를
# 위에서부터 아래로 다시 실행(rerun)한다. 그래서 build_graph()를 매번 새로
# 호출하면 매 상호작용마다 그래프를 다시 컴파일하게 되어 비효율적이다.
# st.session_state는 rerun 사이에도 값이 살아남는 딕셔너리 같은 객체이므로,
# 그래프를 "최초 1번만" 만들어 캐싱해두고 이후 rerun에서는 재사용한다.
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

st.title("📚 Java 학습 도우미")
st.caption("Java 기초 · 객체지향 · 백엔드+DB · 실전 프로젝트 코치 — 4개 에이전트 병렬 오케스트레이션")

# ---------------------------------------------------------------------------
# st.form: 내부의 위젯 값들을 "제출 버튼을 누르는 순간"에만 한 번에 반영한다.
# ---------------------------------------------------------------------------
# form을 쓰지 않으면 텍스트 영역에 한 글자를 입력할 때마다 스크립트 전체가
# rerun되어 매우 비효율적이다. st.form으로 감싸면 form_submit_button을
# 누르기 전까지는 rerun이 발생하지 않고, 누르는 순간에만 아래 with 블록
# 밖의 `if submitted:` 코드가 실행된다.
with st.form("question_form"):
    # 화면을 3개의 세로 컬럼으로 나눠 콘솔 버전의 "1/2/3 메뉴"를
    # 웹에서는 나란히 놓인 입력창 3개로 표현한다.
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Java 기초")
        # key="java_q"는 이 위젯의 값을 st.session_state["java_q"]로도
        # 접근할 수 있게 해주는 고유 식별자다(위젯 자체를 구분하는 용도).
        java_question = st.text_area(
            "변수/자료형, 조건문, 반복문, 배열, 메서드, 예외처리",
            key="java_q",
            height=120,
        )

    with col2:
        st.subheader("객체지향")
        oop_question = st.text_area(
            "클래스/객체, 생성자, 상속, 인터페이스, 다형성",
            key="oop_q",
            height=120,
        )

    with col3:
        st.subheader("백엔드+DB")
        backend_db_question = st.text_area(
            "HTTP, REST API, Controller/Service/Repository, SQL, JPA",
            key="backend_q",
            height=120,
        )

    # 콘솔 버전의 "실전 프로젝트 코치의 통합 코멘트도 받을까요? (y/n)" 질문을
    # 체크박스로 대체한 것. 기본값은 체크 해제(False)다.
    ask_project_coach = st.checkbox("실전 프로젝트 코치의 통합 코멘트도 받기", value=False)

    # use_container_width=True: 버튼을 폼(컬럼 3개 너비) 전체 너비로 늘려서
    # 시각적으로 잘 보이게 한다. 이 버튼을 눌러야 아래 `if submitted:` 블록이 실행된다.
    submitted = st.form_submit_button("질문하기", use_container_width=True)

# ---------------------------------------------------------------------------
# 폼 제출 이후 처리: main.py의 run_interactive() 안에 있던 로직과 대응된다.
# ---------------------------------------------------------------------------
if submitted:
    # GraphState 입력을 구성한다. 콘솔 버전의 graph_input과 동일한 역할.
    graph_input = {"ask_project_coach": ask_project_coach}

    # 빈 입력(공백만 입력한 경우 포함)은 아예 key를 넣지 않는다.
    # -> main.py의 각 튜터 노드는 state.get("xxx_question")이 falsy(None/빈 문자열)면
    #    LLM을 호출하지 않고 통과하므로, 여기서 미리 걸러도 되고 안 걸러도 되지만
    #    "질문을 입력했는지" 여부(len(graph_input))를 아래에서 판단하기 위해 걸러낸다.
    if java_question.strip():
        graph_input["java_question"] = java_question.strip()
    if oop_question.strip():
        graph_input["oop_question"] = oop_question.strip()
    if backend_db_question.strip():
        graph_input["backend_db_question"] = backend_db_question.strip()

    # graph_input에는 항상 "ask_project_coach" 키가 최소 1개 들어있으므로,
    # 길이가 1이라는 것은 세 질문 중 아무것도 입력하지 않았다는 뜻이다.
    if len(graph_input) == 1:  # ask_project_coach만 있고 질문이 하나도 없는 경우
        st.warning("최소 한 주제에는 질문을 입력해주세요.")
    else:
        # st.spinner: with 블록이 실행되는 동안(=그래프 invoke가 끝날 때까지)
        # 화면에 로딩 스피너와 안내 문구를 보여준다. LLM 응답을 기다리는 동안
        # 사용자에게 "동작 중"임을 알려주기 위한 UI 장치일 뿐, 로직에는 영향 없다.
        with st.spinner("에이전트들이 답변을 작성 중입니다..."):
            # main.py의 run_interactive()에서 app.invoke(graph_input)을 호출하던 부분과
            # 완전히 동일하다. 캐싱해둔 st.session_state.graph를 그대로 사용한다.
            result = st.session_state.graph.invoke(graph_input)

        # 구분선: 입력 폼과 결과 영역을 시각적으로 분리한다.
        st.divider()

        # 결과 출력: 콘솔 버전의 ANSWER_LABEL_MAP 순회 + print()에 대응된다.
        # st.expander로 감싸면 접었다 펼 수 있는 카드 형태가 되고,
        # expanded=True라서 처음부터 펼쳐진 상태로 보인다.
        # st.markdown은 답변에 포함된 마크다운 문법(제목, 코드블록, 표 등)을
        # 그대로 렌더링해주므로, print()보다 훨씬 읽기 좋은 형태로 표시된다.
        if result.get("java_answer"):
            with st.expander("🟦 Java 기초 답변", expanded=True):
                st.markdown(result["java_answer"])

        if result.get("oop_answer"):
            with st.expander("🟩 객체지향 답변", expanded=True):
                st.markdown(result["oop_answer"])

        if result.get("backend_db_answer"):
            with st.expander("🟧 백엔드+DB 답변", expanded=True):
                st.markdown(result["backend_db_answer"])

        if result.get("project_summary"):
            with st.expander("🎯 실전 프로젝트 코치의 통합 코멘트", expanded=True):
                st.markdown(result["project_summary"])
