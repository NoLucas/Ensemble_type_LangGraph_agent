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

load_dotenv()

MODEL_NAME = "claude-sonnet-5"


class GraphState(TypedDict):
    java_question: Optional[str]
    oop_question: Optional[str]
    backend_db_question: Optional[str]
    java_answer: Optional[str]
    oop_answer: Optional[str]
    backend_db_answer: Optional[str]
    ask_project_coach: bool
    project_summary: Optional[str]


def make_llm() -> ChatAnthropic:
    return ChatAnthropic(model=MODEL_NAME, max_tokens=1024)


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


def java_tutor(state: GraphState) -> GraphState:
    question = state.get("java_question")
    if not question:
        return {"java_answer": None}
    llm = make_llm()
    response = llm.invoke(
        [("system", JAVA_SYSTEM_PROMPT), ("human", question)]
    )
    return {"java_answer": response.content}


def oop_tutor(state: GraphState) -> GraphState:
    question = state.get("oop_question")
    if not question:
        return {"oop_answer": None}
    llm = make_llm()
    response = llm.invoke(
        [("system", OOP_SYSTEM_PROMPT), ("human", question)]
    )
    return {"oop_answer": response.content}


def backend_db_tutor(state: GraphState) -> GraphState:
    question = state.get("backend_db_question")
    if not question:
        return {"backend_db_answer": None}
    llm = make_llm()
    response = llm.invoke(
        [("system", BACKEND_DB_SYSTEM_PROMPT), ("human", question)]
    )
    return {"backend_db_answer": response.content}


def project_coach(state: GraphState) -> GraphState:
    if not state.get("ask_project_coach"):
        return {"project_summary": None}

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
            ("human", "\n\n".join(collected)),
        ]
    )
    return {"project_summary": response.content}


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("java_tutor", java_tutor)
    graph.add_node("oop_tutor", oop_tutor)
    graph.add_node("backend_db_tutor", backend_db_tutor)
    graph.add_node("project_coach", project_coach)

    graph.add_edge(START, "java_tutor")
    graph.add_edge(START, "oop_tutor")
    graph.add_edge(START, "backend_db_tutor")

    graph.add_edge("java_tutor", "project_coach")
    graph.add_edge("oop_tutor", "project_coach")
    graph.add_edge("backend_db_tutor", "project_coach")

    graph.add_edge("project_coach", END)

    return graph.compile()


TOPIC_MENU = """
어떤 주제에 질문하시겠어요? (쉼표로 여러 개 동시 선택 가능, 예: 1,3)
  1. Java 기초
  2. 객체지향
  3. 백엔드+DB
  종료하려면 q 입력
"""

TOPIC_FIELD_MAP = {
    "1": "java_question",
    "2": "oop_question",
    "3": "backend_db_question",
}

ANSWER_LABEL_MAP = {
    "java_answer": "Java 기초 튜터",
    "oop_answer": "객체지향 튜터",
    "backend_db_answer": "백엔드+DB 튜터",
}


def run_interactive(app):
    while True:
        choice = input(TOPIC_MENU + "> ").strip()
        if choice.lower() in ("q", "quit", "exit"):
            print("종료합니다.")
            break

        selected_numbers = [c.strip() for c in choice.split(",") if c.strip() in TOPIC_FIELD_MAP]
        if not selected_numbers:
            print("1, 2, 3 중에서 선택해주세요.")
            continue

        graph_input = {}
        for number in selected_numbers:
            field = TOPIC_FIELD_MAP[number]
            label = {"java_question": "Java 기초", "oop_question": "객체지향", "backend_db_question": "백엔드+DB"}[field]
            question = input(f"[{label}] 질문 입력: ").strip()
            if question:
                graph_input[field] = question

        if not graph_input:
            print("질문이 비어 있어 건너뜁니다.")
            continue

        want_coach = input("실전 프로젝트 코치의 통합 코멘트도 받을까요? (y/n): ").strip().lower() == "y"
        graph_input["ask_project_coach"] = want_coach

        result = app.invoke(graph_input)

        for field, label in ANSWER_LABEL_MAP.items():
            if result.get(field):
                print(f"\n=== {label} ===")
                print(result[field])

        if result.get("project_summary"):
            print("\n=== 프로젝트 코치 통합 코멘트 ===")
            print(result["project_summary"])


if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit(".env에 ANTHROPIC_API_KEY를 설정하세요 (.env.example 참고)")

    run_interactive(build_graph())
