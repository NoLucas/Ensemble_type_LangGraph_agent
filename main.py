"""
Java 학습 도우미 - LangGraph 기반 4-에이전트 오케스트레이션

구조:
  START -> [java_tutor, oop_tutor, backend_db_tutor] (병렬 fan-out)
        -> project_coach (fan-in, 세 결과를 참고해 통합 코멘트)
        -> END

각 에이전트는 담당 영역의 질문만 처리하고, 질문이 없으면 그냥 통과한다.
"""

import os
from typing import TypedDict, Optional

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END

# .env 파일을 읽어서 os.environ에 등록한다.
# 이 코드 자체에는 API 키가 하드코딩되어 있지 않고, ChatAnthropic이 내부적으로
# os.environ["ANTHROPIC_API_KEY"]를 읽어서 사용한다.
load_dotenv()

# 이 프로젝트에서 사용할 Claude 모델 이름. 한 곳에서만 관리하도록 상수로 분리했다.
MODEL_NAME = "claude-sonnet-5"


# ---------------------------------------------------------------------------
# GraphState: LangGraph의 모든 노드가 공유하는 "전체 상태" 타입
# ---------------------------------------------------------------------------
# LangGraph는 각 노드 함수가 dict(state 전체 또는 일부)를 리턴하면,
# 그 값을 기존 상태에 자동으로 merge(덮어쓰기)해서 다음 노드로 넘겨준다.
# 즉 노드는 "내가 바꾸고 싶은 필드만" 리턴하면 되고, 나머지 필드는 그대로 유지된다.
#
# TypedDict로 선언하는 이유: 타입 힌트만 제공할 뿐 런타임 검증은 하지 않지만,
# 어떤 키들이 상태에 존재할 수 있는지 문서화 + IDE 자동완성 지원을 위해 사용한다.
#
# Optional[str] 필드들의 의미:
#   - *_question: 사용자가 해당 주제를 선택하지 않으면 None (=키 자체가 없을 수도 있음)
#   - *_answer:    해당 튜터가 아직 실행되지 않았거나, 질문이 없어서 스킵한 경우 None
class GraphState(TypedDict):
    java_question: Optional[str]
    oop_question: Optional[str]
    backend_db_question: Optional[str]
    java_answer: Optional[str]
    oop_answer: Optional[str]
    backend_db_answer: Optional[str]
    ask_project_coach: bool          # 통합 코멘트(project_coach) 실행 여부
    project_summary: Optional[str]   # project_coach가 생성한 통합 코멘트


def extract_text(content) -> str:
    """
    ChatAnthropic 응답의 content를 순수 텍스트 문자열로 변환한다.
    extended thinking이 켜진 모델은 content가 문자열이 아니라
    [{"type": "thinking", ...}, {"type": "text", "text": "..."}] 형태의
    블록 리스트로 온다. 여기서 thinking 블록(추론 과정, 서명 blob)은 버리고
    text 블록만 이어붙여야 화면에 실제 답변만 출력된다.
    """
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def make_llm() -> ChatAnthropic:
    """
    호출될 때마다 새 ChatAnthropic 인스턴스를 만든다.
    LangGraph는 병렬로 여러 노드(java_tutor, oop_tutor, backend_db_tutor)를
    동시에 실행할 수 있으므로, 클라이언트 인스턴스를 전역으로 공유하지 않고
    노드마다 새로 만들어서 스레드 간 상태 공유 문제를 원천 차단한다.
    """
    return ChatAnthropic(model=MODEL_NAME, max_tokens=1024)


# ---------------------------------------------------------------------------
# 시스템 프롬프트: 각 튜터(에이전트)의 "역할/성격"을 정의하는 부분.
# LLM에게 "너는 이런 역할이고, 이런 범위만 담당하고, 이렇게 설명해라"라고
# 미리 지시해두면, 같은 질문이라도 어떤 튜터가 답하느냐에 따라 관점이 달라진다.
# ---------------------------------------------------------------------------
JAVA_SYSTEM_PROMPT = """당신은 Java 기초 문법 전담 튜터입니다.
담당 범위: 변수와 자료형, 조건문, 반복문, 배열, 메서드, 예외 처리.
학습자가 개념을 스스로 이해하도록 짧은 코드 예시와 함께 쉽게 설명하세요.
정답만 던지지 말고, 왜 그렇게 동작하는지 이유를 짚어주세요."""

OOP_SYSTEM_PROMPT = """당신은 Java 객체지향 전담 튜터입니다.
담당 범위: 클래스와 객체, 생성자, 상속, 인터페이스, 다형성.
실생활 비유와 간단한 코드 예시를 함께 사용해 개념 간 관계(예: 상속과 다형성의 연결)를 짚어주며 설명하세요."""

BACKEND_DB_SYSTEM_PROMPT = """당신은 백엔드/데이터베이스 전담 튜터입니다.
담당 범위:
- 백엔드 기초: HTTP, REST API, JSON, Controller, Service, Repository
- 데이터베이스: SQL, 테이블 설계, CRUD, JOIN, JPA
REST API 요청이 Controller -> Service -> Repository -> DB로 흐르는 과정을 항상 염두에 두고,
질문이 어느 계층에 해당하는지 짚어가며 설명하세요."""

PROJECT_COACH_SYSTEM_PROMPT = """당신은 실전 프로젝트 코치입니다.
담당 범위: 요구사항 분석, API 설계, DB 설계, 기능 구현, 테스트, 배포.
아래는 학습자가 기초 문법/객체지향/백엔드+DB 튜터에게서 받은 답변입니다.
이 내용을 참고하여, 학습자가 실전 프로젝트에 이 개념들을 어떻게 적용하면 좋을지
통합적인 관점에서 조언하세요. 각 튜터 답변을 반복 설명하지 말고, 다음 학습 단계를 제안하세요."""


# ---------------------------------------------------------------------------
# 노드 함수들 (java_tutor / oop_tutor / backend_db_tutor)
# ---------------------------------------------------------------------------
# LangGraph에서 "노드"는 (state: GraphState) -> dict 형태의 함수다.
# 세 함수 모두 구조가 동일한 패턴을 따른다:
#   1) 자신이 담당하는 질문 필드를 state에서 꺼낸다.
#   2) 질문이 없으면(None 또는 빈 문자열) 곧바로 answer=None을 리턴하고 끝낸다.
#      -> 이것이 "질문이 없는 튜터는 그냥 통과한다"는 동작의 실제 구현부다.
#   3) 질문이 있으면 LLM을 호출해 답변을 받아 리턴한다.
#      리턴값은 state 전체가 아니라 자신이 바꾼 필드(예: java_answer)만 담은
#      작은 dict이며, LangGraph가 이를 전체 state에 병합해준다.
def java_tutor(state: GraphState) -> GraphState:
    question = state.get("java_question")
    if not question:
        return {"java_answer": None}
    llm = make_llm()
    response = llm.invoke(
        [("system", JAVA_SYSTEM_PROMPT), ("human", question)]
    )
    return {"java_answer": extract_text(response.content)}


def oop_tutor(state: GraphState) -> GraphState:
    question = state.get("oop_question")
    if not question:
        return {"oop_answer": None}
    llm = make_llm()
    response = llm.invoke(
        [("system", OOP_SYSTEM_PROMPT), ("human", question)]
    )
    return {"oop_answer": extract_text(response.content)}


def backend_db_tutor(state: GraphState) -> GraphState:
    question = state.get("backend_db_question")
    if not question:
        return {"backend_db_answer": None}
    llm = make_llm()
    response = llm.invoke(
        [("system", BACKEND_DB_SYSTEM_PROMPT), ("human", question)]
    )
    return {"backend_db_answer": extract_text(response.content)}


# ---------------------------------------------------------------------------
# project_coach: "fan-in" 노드
# ---------------------------------------------------------------------------
# 그래프 구조상 java_tutor / oop_tutor / backend_db_tutor 세 노드가 각각
# 끝나야만 project_coach가 실행된다(아래 build_graph의 edge 참고).
# 즉 병렬로 흩어졌던 실행 흐름이 이 노드에서 다시 하나로 "합류(fan-in)"한다.
#
# 동작 순서:
#   1) ask_project_coach가 False면 애초에 실행할 필요가 없으니 바로 종료.
#   2) 세 튜터의 답변 중 실제로 존재하는(질문을 받았던) 것만 골라 리스트로 모은다.
#   3) 모을 답변이 하나도 없으면(예: 통합 코멘트만 원했는데 질문을 아무것도 안 한 경우)
#      LLM 호출 없이 안내 문구만 리턴해 불필요한 API 호출을 막는다.
#   4) 모은 답변들을 하나의 사람 메시지로 합쳐 project_coach LLM에게 전달한다.
def project_coach(state: GraphState) -> GraphState:
    if not state.get("ask_project_coach"):
        return {"project_summary": None}

    # 실제로 답변이 존재하는 튜터만 골라 라벨과 함께 문자열로 정리한다.
    collected = []
    if state.get("java_answer"):
        collected.append(f"[Java 기초 답변]\n{state['java_answer']}")
    if state.get("oop_answer"):
        collected.append(f"[객체지향 답변]\n{state['oop_answer']}")
    if state.get("backend_db_answer"):
        collected.append(f"[백엔드+DB 답변]\n{state['backend_db_answer']}")

    if not collected:
        return {"project_summary": "참고할 튜터 답변이 없어 통합 코멘트를 생략합니다."}

    llm = make_llm()
    response = llm.invoke(
        [
            ("system", PROJECT_COACH_SYSTEM_PROMPT),
            # 여러 튜터의 답변을 빈 줄 두 개로 구분해 하나의 human 메시지로 합친다.
            ("human", "\n\n".join(collected)),
        ]
    )
    return {"project_summary": extract_text(response.content)}


# ---------------------------------------------------------------------------
# build_graph: 노드들을 연결해 실행 가능한 그래프(app)를 만든다.
# ---------------------------------------------------------------------------
def build_graph():
    # GraphState를 상태 스키마로 사용하는 그래프 빌더를 생성한다.
    graph = StateGraph(GraphState)

    # 1) 그래프에 노드(실행 단위)를 등록한다. "이름" -> "실제 함수" 매핑.
    graph.add_node("java_tutor", java_tutor)
    graph.add_node("oop_tutor", oop_tutor)
    graph.add_node("backend_db_tutor", backend_db_tutor)
    graph.add_node("project_coach", project_coach)

    # 2) fan-out: START(그래프 시작점)에서 세 튜터로 동시에 edge를 연결한다.
    #    이렇게 하나의 노드에서 여러 edge가 뻗어나가면 LangGraph는
    #    해당 노드들을 병렬(동시)로 실행한다.
    graph.add_edge(START, "java_tutor")
    graph.add_edge(START, "oop_tutor")
    graph.add_edge(START, "backend_db_tutor")

    # 3) fan-in: 세 튜터 모두 project_coach로 edge를 연결한다.
    #    LangGraph는 project_coach로 들어오는 모든 선행 노드(3개)가
    #    끝날 때까지 자동으로 기다린 뒤 project_coach를 실행한다(barrier).
    graph.add_edge("java_tutor", "project_coach")
    graph.add_edge("oop_tutor", "project_coach")
    graph.add_edge("backend_db_tutor", "project_coach")

    # 4) project_coach가 끝나면 그래프 종료.
    graph.add_edge("project_coach", END)

    # compile()을 호출해야 실제로 실행 가능한(app.invoke(...)를 호출할 수 있는)
    # 객체가 만들어진다. 여기까지는 "설계도"만 만든 것이고, compile 이후가 "실행체"다.
    return graph.compile()


# ---------------------------------------------------------------------------
# 아래부터는 LangGraph와 직접적인 관련은 없는, 콘솔 UI(사용자 입출력) 담당 코드.
# ---------------------------------------------------------------------------

TOPIC_MENU = """
어떤 주제에 질문하시겠어요? (쉼표로 여러 개 동시 선택 가능, 예: 1,3)
  1. Java 기초
  2. 객체지향
  3. 백엔드+DB
  종료하려면 q 입력
"""

# 사용자가 입력한 메뉴 번호("1","2","3")를 GraphState의 실제 질문 필드 이름으로 변환하는 표.
TOPIC_FIELD_MAP = {
    "1": "java_question",
    "2": "oop_question",
    "3": "backend_db_question",
}

# 그래프 실행 결과(GraphState)의 답변 필드 이름을, 화면에 보여줄 한글 라벨로 변환하는 표.
ANSWER_LABEL_MAP = {
    "java_answer": "Java 기초 튜터",
    "oop_answer": "객체지향 튜터",
    "backend_db_answer": "백엔드+DB 튜터",
}


def run_interactive(app):
    """
    콘솔에서 무한 루프를 돌며 사용자 입력을 받고,
    매 반복(turn)마다 build_graph()로 만든 app(컴파일된 그래프)을 한 번 invoke한다.
    즉 이 함수는 "그래프를 어떻게 만드는가"가 아니라 "그래프를 어떻게 반복 호출하는가"를 담당한다.
    """
    while True:
        choice = input(TOPIC_MENU + "> ").strip()
        if choice.lower() in ("q", "quit", "exit"):
            print("종료합니다.")
            break

        # "1,3" 같은 입력을 쉼표로 나눈 뒤, TOPIC_FIELD_MAP에 실제로 존재하는
        # 번호("1","2","3")만 남긴다. "4"나 빈 문자열처럼 잘못된 입력은 자동으로 걸러진다.
        selected_numbers = [c.strip() for c in choice.split(",") if c.strip() in TOPIC_FIELD_MAP]
        if not selected_numbers:
            print("1, 2, 3 중에서 선택해주세요.")
            continue

        # 선택된 주제 각각에 대해 실제 질문 문장을 입력받아,
        # 그래프에 넘길 입력 dict(graph_input)을 채운다.
        # 예: {"java_question": "...", "backend_db_question": "..."}
        graph_input = {}
        for number in selected_numbers:
            field = TOPIC_FIELD_MAP[number]
            label = {"java_question": "Java 기초", "oop_question": "객체지향", "backend_db_question": "백엔드+DB"}[field]
            question = input(f"[{label}] 질문 입력: ").strip()
            if question:
                graph_input[field] = question

        # 번호는 선택했지만 정작 질문을 아무것도 입력하지 않은 경우(전부 빈 문자열)
        if not graph_input:
            print("질문이 비어 있어 건너뜁니다.")
            continue

        # 통합 코멘트(project_coach) 실행 여부도 graph_input에 함께 담아 전달한다.
        want_coach = input("실전 프로젝트 코치의 통합 코멘트도 받을까요? (y/n): ").strip().lower() == "y"
        graph_input["ask_project_coach"] = want_coach

        # 여기서 실제로 그래프가 실행된다:
        #   START -> (선택된 튜터들 병렬 실행) -> project_coach -> END
        # graph_input에 없는 필드는 GraphState 상에서 자동으로 없는 값(None)으로 취급된다.
        result = app.invoke(graph_input)

        # 결과(result)는 GraphState 전체 dict이므로, 답변이 채워진 필드만 골라 출력한다.
        for field, label in ANSWER_LABEL_MAP.items():
            if result.get(field):
                print(f"\n=== {label} ===")
                print(result[field])

        if result.get("project_summary"):
            print("\n=== 프로젝트 코치 통합 코멘트 ===")
            print(result["project_summary"])


if __name__ == "__main__":
    # 실제로 그래프를 실행(LLM 호출)하기 전에 API 키 존재 여부부터 확인한다.
    # 이렇게 하면 질문을 다 입력한 뒤 첫 LLM 호출에서야 에러가 나는 대신,
    # 프로그램 시작과 동시에 바로 문제를 알 수 있다.
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit(".env에 ANTHROPIC_API_KEY를 설정하세요 (.env.example 참고)")

    # build_graph()로 그래프를 컴파일한 뒤, run_interactive에 넘겨 콘솔 루프를 시작한다.
    run_interactive(build_graph())
