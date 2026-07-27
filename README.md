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

## 이 오케스트레이션의 목적

Java를 배우는 사람이 한 번에 여러 영역(문법, 객체지향, 백엔드/DB)에 걸친 질문을 해도, 각 영역에
특화된 시스템 프롬프트를 가진 튜터가 각자의 관점에서 답하도록 하는 것이 목적입니다. 하나의
범용 프롬프트로 모든 질문에 답하게 하는 대신, 역할을 4개의 에이전트(3개의 병렬 튜터 + 1개의
통합 코치)로 분리해 "각 에이전트는 좁고 명확한 책임만 진다"는 원칙을 실습해보는 예제입니다.

## 이 구조의 장점

- **관심사 분리**: 튜터마다 시스템 프롬프트(담당 범위, 설명 방식)가 독립적이라, 한 튜터의 프롬프트를
  수정해도 다른 튜터의 답변 품질에 영향을 주지 않습니다.
- **병렬 실행으로 인한 지연 감소**: `java_tutor`/`oop_tutor`/`backend_db_tutor`는 서로 의존성이 없으므로
  LangGraph가 동시에 실행합니다. 세 질문을 모두 물어봐도 순차 호출(3번의 왕복 대기) 대신 가장 느린
  튜터 1명의 응답 시간만큼만 기다리면 됩니다.
- **불필요한 LLM 호출 생략**: 질문하지 않은 주제의 튜터는 아예 LLM을 호출하지 않고 즉시 통과하며,
  `project_coach`도 통합 코멘트를 요청하지 않으면 실행되지 않습니다. 비용과 응답 시간을 최소화합니다.
- **fan-in을 통한 통합 관점 제공**: `project_coach`가 세 튜터의 답변을 모아 "실전 프로젝트에 어떻게
  적용할지"라는 상위 관점의 조언을 추가로 제공해, 단편적인 지식이 아니라 통합된 학습 경험을 만듭니다.
- **인터페이스와 로직의 분리**: 그래프(`build_graph()`)는 `main.py`에 한 번만 정의되어 있고,
  콘솔(`main.py`)과 웹(`app.py`) 두 인터페이스가 이를 그대로 재사용합니다. 오케스트레이션 로직을
  건드리지 않고도 새로운 인터페이스(예: API 서버)를 추가할 수 있습니다.

## 실행 방식 두 가지

이 프로젝트는 동일한 그래프(`main.py`의 `build_graph()`)를 두 가지 인터페이스로 실행할 수 있습니다.

| 파일 | 인터페이스 | 실행 명령 |
|------|-----------|----------|
| `main.py` | 콘솔(터미널) | `python main.py` |
| `app.py`  | 웹 브라우저(Streamlit) | `streamlit run app.py` |

### 콘솔 버전 (`main.py`) 동작 방식

1. `main.py`를 실행하면 콘솔 메뉴가 뜹니다.
2. 질문할 주제 번호를 선택합니다 (쉼표로 복수 선택 가능, 예: `1,3`).
3. 선택한 주제별로 질문을 입력합니다.
4. 실전 프로젝트 코치의 통합 코멘트를 받을지 `y`/`n`으로 선택합니다.
5. LangGraph가 선택된 튜터 노드들을 병렬로 호출하고, 필요 시 `project_coach`까지 실행한 뒤 결과를 출력합니다.
6. `q`를 입력하면 종료합니다.

### 웹 버전 (`app.py`) 동작 방식

1. `streamlit run app.py`를 실행하면 브라우저가 자동으로 열립니다.
2. Java 기초 / 객체지향 / 백엔드+DB 세 칸의 입력창 중 원하는 만큼 질문을 입력합니다 (하나만 입력해도 됩니다).
3. 실전 프로젝트 코치의 통합 코멘트를 받을지 체크박스로 선택합니다.
4. **질문하기** 버튼을 누르면 콘솔 버전과 동일하게 그래프가 실행되고, 각 튜터의 답변이 접었다 펼 수 있는 카드(마크다운 렌더링)로 표시됩니다.
5. 다시 질문하려면 입력창을 수정하고 **질문하기**를 다시 누르면 됩니다(페이지를 새로고침할 필요 없음).

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

콘솔 버전:

```bash
python main.py
```

웹(Streamlit) 버전:

```bash
streamlit run app.py
```

웹 버전을 실행하면 기본적으로 `http://localhost:8501`에서 브라우저가 자동으로 열립니다.

## 요구 사항

- Python 3.10+
- Anthropic API 키 ([console.anthropic.com](https://console.anthropic.com)에서 발급)

## 주요 의존성

- `langgraph` — 에이전트 그래프 오케스트레이션
- `langchain-anthropic` — Anthropic(Claude) 모델 연동
- `python-dotenv` — `.env` 환경변수 로드
- `streamlit` — `app.py`의 웹 인터페이스 렌더링

---

# English

A Java learning assistant example built with LangGraph, orchestrating 4 agents.

## Architecture

```
        START
          |
   ┌──────┼──────────────┐
   ▼      ▼              ▼
java_tutor oop_tutor  backend_db_tutor      (parallel fan-out)
   └──────┼──────────────┘
          ▼
   project_coach                            (fan-in, combined feedback)
          │
          ▼
         END
```

- **java_tutor**: handles Java basics — variables/types, conditionals, loops, arrays, methods, exception handling
- **oop_tutor**: handles OOP — classes/objects, constructors, inheritance, interfaces, polymorphism
- **backend_db_tutor**: handles backend/DB — HTTP, REST API, Controller-Service-Repository flow, SQL, JPA
- **project_coach**: reads the three tutors' answers and gives integrated advice on applying them in a real project

The three tutors run in parallel; each node passes through untouched if it has no assigned question.
`project_coach` only produces a combined comment when `ask_project_coach` is true and at least one tutor answer is available.

## Purpose of this orchestration

When a Java learner asks about multiple areas at once (syntax, OOP, backend/DB) in a single turn, each
area is handled by a tutor agent with its own specialized system prompt, answering from its own
perspective. Instead of one general-purpose prompt trying to answer everything, the responsibility is
split across 4 agents (3 parallel tutors + 1 integrating coach) — a small, practical example of giving
each agent a narrow, well-defined responsibility.

## Advantages of this structure

- **Separation of concerns**: each tutor's system prompt (scope, explanation style) is independent, so
  editing one tutor's prompt doesn't affect the quality of the others' answers.
- **Reduced latency via parallelism**: `java_tutor`, `oop_tutor`, and `backend_db_tutor` have no
  dependency on each other, so LangGraph runs them concurrently. Asking all three topics at once only
  costs as long as the slowest single tutor's response, instead of three sequential round trips.
- **Fewer unnecessary LLM calls**: a tutor for a topic that wasn't asked about never calls the LLM and
  passes through immediately; `project_coach` likewise doesn't run unless a combined comment was
  requested. This keeps both cost and response time down.
- **Integrated perspective via fan-in**: `project_coach` gathers all three tutors' answers and adds
  higher-level advice on applying them to a real project, turning isolated facts into a more coherent
  learning experience.
- **Decoupled interface from logic**: the graph (`build_graph()`) is defined once in `main.py`, and both
  the console (`main.py`) and web (`app.py`) interfaces reuse it as-is. A new interface (e.g. an API
  server) can be added without touching the orchestration logic.

## Two ways to run it

Both interfaces run the exact same graph (`build_graph()` in `main.py`).

| File | Interface | Run command |
|------|-----------|--------------|
| `main.py` | Console (terminal) | `python main.py` |
| `app.py`  | Web browser (Streamlit) | `streamlit run app.py` |

### Console version (`main.py`)

1. Running `main.py` shows a console menu.
2. Pick the topic number(s) you want to ask about (comma-separated for multiple, e.g. `1,3`).
3. Enter your question for each selected topic.
4. Choose `y`/`n` for whether you also want the project coach's combined comment.
5. LangGraph invokes the selected tutor nodes in parallel, then runs `project_coach` if requested, and prints the results.
6. Enter `q` to quit.

### Web version (`app.py`)

1. Running `streamlit run app.py` opens a browser automatically.
2. Fill in as many of the three question boxes (Java basics / OOP / Backend+DB) as you like — even just one is fine.
3. Check the checkbox if you also want the project coach's combined comment.
4. Click **질문하기 (Ask)** to invoke the same graph as the console version; each tutor's answer appears in a collapsible, markdown-rendered card.
5. To ask again, just edit the inputs and click the button again — no page refresh needed.

## Usage

### 1. Clone the repository

```bash
git clone https://github.com/NoLucas/UP_Simple_LangGraph.git
cd UP_Simple_LangGraph
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
# Windows (Git Bash)
source venv/Scripts/activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

Copy `.env.example` to `.env` and add your own Anthropic API key.

```bash
cp .env.example .env
```

```
ANTHROPIC_API_KEY=sk-ant-your-actual-key
```

`.env` is listed in `.gitignore` and is never committed, so each user can safely use their own key.

### 5. Run

Console version:

```bash
python main.py
```

Web (Streamlit) version:

```bash
streamlit run app.py
```

The web version opens a browser automatically at `http://localhost:8501` by default.

## Requirements

- Python 3.10+
- Anthropic API key (get one at [console.anthropic.com](https://console.anthropic.com))

## Key dependencies

- `langgraph` — agent graph orchestration
- `langchain-anthropic` — Anthropic (Claude) model integration
- `python-dotenv` — loads environment variables from `.env`
- `streamlit` — renders the web interface in `app.py`
