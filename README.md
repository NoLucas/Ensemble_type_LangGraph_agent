# UP_Simple_LangGraph

Java 학습 도우미 - LangGraph 기반 4-에이전트 오케스트레이션 예제입니다.

## 구조

```
        START
          |
   ┌──────┼──────────────┐
   ▼      ▼              ▼
java_tutor oop_tutor  backend_db_tutor      (병렬 fan-out)
   └──────┼──────────────┘
          ▼
   project_coach                            (fan-in, 통합 코멘트)
          │
          ▼
         END
```

- **java_tutor**: Java 기초 문법(변수/자료형, 조건문, 반복문, 배열, 메서드, 예외 처리) 담당
- **oop_tutor**: 객체지향(클래스/객체, 생성자, 상속, 인터페이스, 다형성) 담당
- **backend_db_tutor**: 백엔드/DB(HTTP, REST API, Controller-Service-Repository, SQL, JPA) 담당
- **project_coach**: 위 세 튜터의 답변을 참고해 실전 프로젝트 적용 관점의 통합 코멘트 생성

세 튜터는 병렬로 실행되며, 각 노드는 자신이 담당한 질문이 없으면 바로 통과(pass-through)합니다.
`project_coach`는 `ask_project_coach` 값이 참이고 참고할 답변이 하나라도 있을 때만 통합 코멘트를 만듭니다.

## 동작 방식

1. `main.py`를 실행하면 콘솔 메뉴가 뜹니다.
2. 질문할 주제 번호를 선택합니다 (쉼표로 복수 선택 가능, 예: `1,3`).
3. 선택한 주제별로 질문을 입력합니다.
4. 실전 프로젝트 코치의 통합 코멘트를 받을지 `y`/`n`으로 선택합니다.
5. LangGraph가 선택된 튜터 노드들을 병렬로 호출하고, 필요 시 `project_coach`까지 실행한 뒤 결과를 출력합니다.
6. `q`를 입력하면 종료합니다.

## 사용법

### 1. 저장소 클론

```bash
git clone https://github.com/NoLucas/UP_Simple_LangGraph.git
cd UP_Simple_LangGraph
```

### 2. 가상환경 생성 및 활성화

```bash
python -m venv venv
# Windows (Git Bash)
source venv/Scripts/activate
# macOS/Linux
source venv/bin/activate
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. API 키 설정

`.env.example`을 복사해 `.env`를 만들고, 자신의 Anthropic API 키를 넣습니다.

```bash
cp .env.example .env
```

```
ANTHROPIC_API_KEY=sk-ant-실제-발급받은-키
```

`.env`는 `.gitignore`에 등록되어 있어 git에 올라가지 않으므로, 각자 자신의 키를 안전하게 사용할 수 있습니다.

### 5. 실행

```bash
python main.py
```

## 요구 사항

- Python 3.10+
- Anthropic API 키 ([console.anthropic.com](https://console.anthropic.com)에서 발급)

## 주요 의존성

- `langgraph` — 에이전트 그래프 오케스트레이션
- `langchain-anthropic` — Anthropic(Claude) 모델 연동
- `python-dotenv` — `.env` 환경변수 로드
