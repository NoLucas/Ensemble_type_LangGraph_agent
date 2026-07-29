# Ensemble_type_LangGraph_agent

LangGraph 기반 GitHub 저장소 리뷰/학습 에이전트입니다. TDD(Red-Green-Refactor)로 개발되었으며, 사용자가 대화에서 언급한 `owner/repo` 저장소의 개요(README)·구조·소스 코드를 GitHub API로 가져와 병렬 처리합니다. 최종 답변은 동일한 과제를 여러 번(기본 3회) 독립 시도한 뒤 도구 실행 결과와 실제로 일치하는 답을 결정론적으로 골라 채택하는 투표(voting) 앙상블 구조입니다.

**정식 리뷰**(요약 + 코드 리뷰, 토큰 많이 씀)와 **스터디 모드**(구조 지도만 가볍게 훑고 관심 파일만 드릴다운, 토큰 적게 씀) 두 흐름을 지원합니다 — 무료 티어처럼 토큰이 빠듯한 상태에서도 대형 저장소를 공부할 수 있게 하려는 목적입니다. 자세한 내용은 [스터디 모드](#스터디-모드-토큰-절약) 섹션을 참고하세요.

## 구조

```
        START
          │
          ▼
      dispatcher                          (첫 입력: 언급된 저장소에 맞는 도구 지시)
          │
   tool_call 없음 ──────────────────► END  (도구 불필요 시 dispatcher 답변이 곧 최종 답변)
          │
   tool_call 있음
          │
   ┌──────┼───────┬───────┐
   ▼      ▼       ▼       ▼               (도구 팬아웃 — 정적 4-way)
repo_overview  repo_source  repo_structure  repo_file
_node          _node        _node           _node
   └──────┼───────┴───────┘
          ▼
   ┌──────┼───────────────┬──────────┐
   ▼      ▼               ▼          (투표형 앙상블 팬아웃 — 정적 N-way,
voter_1_node   voter_2_node   voter_3_node   기본 N=3, 스터디 모드는 N=1,
   └──────┼───────────────┴──────────┘        다 같은 프롬프트로 독립 시도)
          ▼
   vote_for_best_report                (팬인: LLM 재호출 없이 결정론적 다수결)
          │
          ▼
         END
```

- **dispatcher**: 사용자 입력에서 언급된 GitHub 저장소(`owner/repo`)를 해석해 도구 네 개(`fetch_repo_overview`/`fetch_repo_source_sample`/`fetch_repo_structure`/`fetch_repo_file`) 중 필요한 것을 지시합니다. "정식 리뷰"인지 "가볍게 훑어보기+드릴다운"인지 사용자 의도에 맞는 조합을 고릅니다. 여러 저장소·도구가 필요하면 한 번의 응답에서 모두 요청하도록 유도해 팬아웃이 실제로 병렬 이득을 보게 합니다.
- **도구 팬아웃/팬인**: 네 도구 노드는 항상 함께 깨워지고, 자기 담당 tool_call이 없으면 조용히 통과(pass-through)합니다.
- **투표형 앙상블 팬아웃/팬인**: `voter_1..voter_N`(기본 N=3)은 **모두 같은 프롬프트(`VOTER_SYSTEM_PROMPT`)**로 사용자 요청에 맞는 답변(리뷰/구조 설명/특정 파일 설명)을 각자 독립적으로 시도합니다. 관점을 나누는 앙상블이 아니라 같은 과제를 여러 번 독립 시도해서 검증하는 투표 앙상블이며, 다양성은 프롬프트가 아니라 모델 샘플링 자체의 변동성(temperature)에서 나옵니다. `build_graph(llm, num_voters=1)`(스터디 모드)이면 다수결 없이 그 한 번의 시도가 그대로 채택됩니다.
- **vote_for_best_report**: 도구 실행 결과(`ToolMessage`)에서 "사실 토큰"(숫자, 영문 파일명/식별자, 한글 단어)을 뽑아, candidate 중 이 토큰을 가장 많이 포함한 것을 **LLM을 다시 호출하지 않고** 결정론적으로 고르는 함수입니다. `ToolMessage` 내용이 수백~수천 자짜리 자유 서술형 텍스트라 voter가 그 블록을 통째로 인용하는 일은 없으므로, "전체 문자열이 그대로 포함되는가"가 아니라 "토큰 단위로 얼마나 겹치는가"로 채점합니다 — voter가 문장을 바꿔 써도(패러프레이즈해도) 숫자/고유명사 같은 사실은 그대로 잡아냅니다. 동점이면 `voter_1 → voter_2 → ...` 순서로 타이브레이크하고, 아무도 사실을 못 맞혀도 예외 없이 첫 번째 voter의 답을 반환합니다. "어느 게 더 그럴듯한가"를 LLM 판사에게 다시 묻지 않으므로, 이 종합 단계에는 환각이 새로 끼어들 여지가 없습니다.
- **도구가 필요 없으면** dispatcher의 답변이 그대로 최종 답변이 되고, 도구 팬아웃/투표 팬아웃 단계는 아예 실행되지 않습니다 (불필요한 LLM 호출을 만들지 않습니다).

이 그래프는 사이클이 없는 DAG입니다 — `dispatcher`는 tool이 바인딩된 llm을, `voter_*` 노드들은 바인딩되지 않은 llm을 받아서 구조적으로 도구를 다시 호출할 수 없습니다. 그래서 반복 상한(iteration cap) 같은 무한 루프 방지 장치가 필요 없습니다.

## 파일 구성

```
agent/
  state.py    # AgentState: messages(add_messages), iteration(operator.add 델타), report_drafts(operator.add)
  tools.py    # fetch_repo_overview/source_sample/structure/file (GitHub REST API, GITHUB_TOKEN 선택적 인증)
  nodes.py    # dispatcher/voter/vote_for_best_report 노드, 라우팅 함수, 도구 실행 노드
  graph.py    # build_graph(llm, num_voters=3) — 전체 그래프 조립
tests/
  conftest.py         # FakeChatModel + mock_github_get(QueuedGet)/mock_github_get_by_path — 실제 API 호출 없이 결정적 테스트
  test_state.py        # state reducer 테스트
  test_tools.py         # 도구 단위 테스트 (정상/실패 케이스)
  test_nodes.py          # 노드 단위 테스트
  test_routing.py         # 조건부 라우팅 테스트
  test_graph_e2e.py        # 그래프 통합 테스트
```

## 도구

| 도구 | 설명 | 안전장치 |
|---|---|---|
| `fetch_repo_overview` | 저장소 메타데이터(설명/언어/stars)와 README 발췌 | `owner/repo` 형식 검증, 저장소 없음(404)/rate limit(403)/네트워크 오류를 예외 대신 에러 문자열로 반환 |
| `fetch_repo_source_sample` | 저장소를 디렉토리 단위로 탐색해 대표 소스 파일 최대 3개의 **전체 내용** 발췌 (정식 코드 리뷰용) | 소스 확장자 화이트리스트, `tests/vendor/node_modules/examples` 등 경로 및 `_test.go`/`.spec.ts` 등 파일명 패턴 제외, 디렉토리 탐색 호출 상한, 파일당/README 길이 상한 |
| `fetch_repo_structure` | 저장소를 훨씬 넓게(최대 10개) 훑되, 파일 본문이 아니라 **함수/클래스 시그니처만** 추출한 구조 지도 (스터디 모드 1단계) | 위와 동일한 탐색/제외 로직 재사용, 시그니처 추출 미지원 확장자는 자동으로 건너뜀 |
| `fetch_repo_file` | 사용자가 콕 집은 파일 **하나**의 전체 내용 (스터디 모드 2단계, 드릴다운) | `owner/repo` 형식 검증, path 필수, 404/네트워크 오류 처리. 탐색 없이 바로 조회하므로 API 호출이 가장 적음 |

모든 도구는 예외를 던지지 않고 `"Error: ..."` 문자열을 반환합니다 — LLM이 스키마와 맞지 않는 입력을 줄 수 있고 GitHub API도 실패할 수 있으므로, 실패도 정상적인 반환값으로 표현해서 그래프 전체가 죽지 않게 합니다.

### `fetch_repo_source_sample`이 큰 저장소에서도 가벼운 이유

저장소 전체 파일 목록을 `/git/trees/{branch}?recursive=1`로 한 번에 받는 방식은 파일 몇 개만 보면 되는 리뷰 목적에 비해 너무 무겁습니다 — 실측으로 `kubernetes/kubernetes`는 응답이 10MB/37,390개 항목, `torvalds/linux`는 17.6MB/71,798개 항목(그나마도 `truncated: true`로 일부 누락)이었습니다. 대신 `/repos/{repo}/contents/{path}`로 디렉토리를 하나씩(비재귀) 나열하며 **깊이 우선(DFS)**으로 탐색합니다 — `src`/`lib`/`pkg`/`cmd` 같은 소스 디렉토리 힌트를 먼저 파고들고, 호출 횟수(`_MAX_DIR_LISTING_CALLS`, 기본 8)나 모인 후보 수가 충분해지면 멈춥니다. 너비 우선(BFS)으로 처음 만들었다가 `kubernetes/kubernetes`로 검증하며 문제를 발견했습니다: 루트 바로 아래엔 컴포넌트별 디렉토리만 잔뜩 있고 실제 코드는 몇 단계 더 안쪽(`pkg/volume/*.go` 등)에 있어서, BFS는 얕은 디렉토리를 옆으로 훑다 호출 예산을 다 쓰고 파일을 하나도 못 찾았습니다. DFS는 유망한 경로 하나를 바닥까지 파고든 뒤에야 옆 가지로 넘어가므로 이런 구조에도 잘 맞습니다(`kubernetes/kubernetes`로 재검증: 10MB → 약 4KB, 파일 3개, 3초 이내).

찾은 후보는 다음 신호를 점수로 합산해 가장 대표성 있는 파일을 고릅니다(`agent/tools.py`의 `_score_source_candidate`):

- **저장소의 대표 언어(`language` 메타데이터)와 확장자가 일치**하는 파일 우선 (예: Python 저장소면 `.py` 우선)
- **진입점으로 보이는 파일명**(`main`/`index`/`app`/`cli`/`__init__`/`server`/`run`) 우선
- **경로가 얕을수록** 약간 가산 (저장소 핵심에 가까운 코드일 가능성)
- **적당한 크기**(50~20,000바이트)의 파일 우선 — 빈 파일(스텁)은 사실상 배제하고, 지나치게 큰 파일(생성 코드·데이터 덤프)은 감점

`examples`/`demo`/`sample` 디렉토리와 `_test.go`/`.spec.ts`/`test_*.py` 같은 테스트 파일명 패턴은 아예 후보에서 제외합니다 — `examples/`의 데모 코드는 흔히 `index.js` 같은 진입점 파일명을 그대로 써서 실제 라이브러리 구현(`lib/`, `src/`)보다 먼저 뽑히는 문제가 있었고, Go처럼 구현 파일과 테스트 파일을 같은 디렉토리에 나란히 두는 언어는 디렉토리 이름만으로 걸러지지 않기 때문입니다.

GitHub API는 기본적으로 **비인증으로 호출**합니다(시간당 60회 제한). `.env`에 `GITHUB_TOKEN`을 설정하면 모든 요청에 자동으로 `Authorization: Bearer <GITHUB_TOKEN>` 헤더가 붙어 시간당 5000회로 늘어납니다(`agent/tools.py`의 `_auth_headers()`). 별도 권한이 필요 없는(공개 저장소 읽기 전용) [개인용 액세스 토큰](https://github.com/settings/tokens)이면 충분합니다. 토큰을 설정하지 않으면 기존과 동일하게 비인증으로 동작합니다.

## 스터디 모드 (토큰 절약)

큰 저장소를 리뷰가 아니라 **공부** 목적으로 가볍게 훑어보고 싶을 때, `build_graph(llm, num_voters=1)`로 그래프를 조립하면 다수결 검증(voter 3개) 대신 voter 1개만 씁니다 — LLM 호출이 4번(dispatcher + voter 3)에서 **2번**(dispatcher + voter 1)으로 줄어 토큰을 절반 이하로 아낍니다. 무료 티어처럼 토큰이 빠듯한 상황을 염두에 둔 트레이드오프로, 다수결의 오류 검증 효과는 포기합니다.

세 인터페이스 모두 스터디 모드를 켤 수 있습니다:

| 인터페이스 | 켜는 방법 |
|---|---|
| `main.py` (콘솔) | 시작할 때 "스터디 모드로 시작할까요? [y/N]" 프롬프트에 `y` |
| `app.py` (Streamlit) | 사이드바의 "스터디 모드" 토글 |
| `chainlit_app.py` (Chainlit) | 오른쪽 위 설정(⚙️)의 "스터디 모드" 스위치 — 대화 중에도 켜고 끌 수 있습니다 |

스터디 모드에서 대형 저장소를 효과적으로 공부하는 흐름은 "구조 먼저, 필요한 파일만 나중에"입니다:

1. **"구조만 보여줘"** — `fetch_repo_overview` + `fetch_repo_structure`만 호출됩니다. 전체 코드 본문 없이 함수/클래스 시그니처만 담은 지도를 최대 10개 파일까지 보여줘서, 코드를 한 줄도 안 읽고도 저장소가 대략 어떻게 생겼는지 감을 잡을 수 있습니다.
2. **"app.py 자세히 보여줘"** — 구조 지도를 본 뒤 관심 가는 파일을 콕 집으면 `fetch_repo_file`이 그 파일 하나의 전체 내용만 가져옵니다(탐색 없이 바로 조회하므로 API 호출도 가장 적습니다).

dispatcher가 이 흐름을 자동으로 판단합니다(`DISPATCHER_SYSTEM_PROMPT` 참고) — "리뷰해줘"처럼 포괄적인 요청이면 정식 리뷰(`fetch_repo_source_sample`)로, "구조만"/"공부하고 싶다"처럼 가벼운 요청이면 스터디 흐름으로 자동 전환됩니다.

## 설치 및 테스트

```bash
python -m venv venv
venv/Scripts/activate   # Windows
pip install -r requirements.txt
pytest
```

현재 81개 테스트가 전부 통과합니다 (state/tools/nodes/routing/graph E2E 계층별 TDD 79개 + 실LLM 통합 테스트 2개).

느린 테스트(실LLM + 실제 GitHub API 호출)는 `integration` 마커(`tests/test_integration.py`)로 분리되어 있으며 기본 실행에서 제외하는 것을 권장합니다:

```bash
pytest -m "not integration"
```

`test_integration.py`는 `.env`의 `ANTHROPIC_API_KEY`가 설정된 경우에만 실행되고, 없으면 실패하지 않고 자동으로 skip됩니다. 키가 있는 상태에서 통합 테스트만 따로 돌리려면:

```bash
pytest -m integration
```

`tests/conftest.py`의 `FakeChatModel`/`QueuedGet`은 `threading.Lock`으로 보호됩니다 — voter들이 그래프 실행 중 같은 인스턴스를 스레드 풀에서 동시에 호출하므로, 락이 없으면 경합으로 테스트가 간헐적으로 실패할 수 있습니다. `mock_github_get`(`QueuedGet`)은 호출 순서 기반으로 응답을 내주는 더미라 도구 하나가 순차적으로 여러 번 호출하는 경우에 적합합니다. 서로 다른 두 도구 노드(예: `repo_structure_node`+`repo_file_node`)가 같은 팬아웃 라운드에서 병렬로 각자 여러 번 호출하면 순서를 신뢰할 수 없으므로, 그런 경우엔 URL 부분 문자열로 응답을 매칭하는 `mock_github_get_by_path`(`PathAwareQueuedGet`)를 씁니다.

## 그래프 사용 예시

```python
from langchain_anthropic import ChatAnthropic
from agent.graph import build_graph

llm = ChatAnthropic(model="claude-sonnet-5")
graph = build_graph(llm)  # 정식 리뷰(voter 3개). 스터디 모드는 num_voters=1

result = graph.invoke({
    "messages": [("human", "langchain-ai/langgraph 요약이랑 코드 리뷰 둘 다 해줘")],
    "iteration": 0,
    "report_drafts": [],
})
print(result["messages"][-1].content)
```

`ANTHROPIC_API_KEY`는 `.env.example`을 참고해 `.env` 파일로 설정합니다.

## 실행 방법

동일한 그래프(`agent/graph.py`의 `build_graph()`)를 세 가지 인터페이스로 실행할 수 있습니다. 오케스트레이션 로직은 `agent/`에만 있고, 인터페이스들은 이를 그대로 재사용합니다.

| 파일 | 인터페이스 | 실행 명령 |
|------|-----------|----------|
| `main.py` | 콘솔(터미널) | `python main.py` |
| `app.py`  | 웹 브라우저(Streamlit) | `streamlit run app.py` |
| `chainlit_app.py` | 웹 브라우저(Chainlit, 도구 실행 과정 실시간 시각화) | `chainlit run chainlit_app.py` (또는 `python run_chainlit.py chainlit_app.py`) |

세 파일 모두 `main.py`의 `make_llm()`/`extract_text()`를 공유합니다. 그래프에 체크포인터가 없으므로, 대화 기록은 각 인터페이스가 파이썬 변수(콘솔: 지역 변수, Streamlit: `st.session_state`, Chainlit: `cl.user_session`)로 들고 있다가 매 턴 그래프에 다시 넘겨줍니다.

`chainlit run`이 몇몇 환경(`nest_asyncio` + 최신 `anyio` 조합)에서 정적 프론트엔드 자산을 못 띄우고 500 에러만 뱉는 경우, `run_chainlit.py`를 대신 쓰세요 — `chainlit run`과 똑같은 CLI(`-h`, `--port` 등)를 그대로 지원하면서, 문제의 원인인 `nest_asyncio.apply()` 호출만 무해화합니다. 자세한 원인은 `run_chainlit.py`의 모듈 docstring을 참고하세요.

초기 state에는 `report_drafts: []`를 반드시 포함해야 합니다 — N-way 투표형 앙상블(`voter_1..voter_N`)이 병렬로 채우는 `operator.add` 리듀서 필드라, 첫 병렬 쓰기 전에 채널이 초기화돼 있어야 안전합니다. `chainlit_app.py`처럼 `graph.invoke()` 대신 `graph.stream()`으로 상태를 직접 재구성하는 경우, `iteration`과 `report_drafts` 모두 **델타/리스트를 누적**해야 합니다(마지막 값으로 덮어쓰면 안 됩니다) — 두 필드 모두 `operator.add` 리듀서를 쓰기 때문입니다.

## 개발 진행 이력 (Chainlit 실행 불가 버그)

`venv/Scripts/python.exe -m chainlit run chainlit_app.py`가 실행이 안 된다는
문제를 초기 진단부터 최종 수정까지 단계별로 정리한다.

1. **최초 증상 확인** — 명령을 실행하면 `Your app is available at
   http://localhost:8000` 로그까지는 뜨지만, 실제 요청(정적 프론트엔드 자산
   포함)마다 `ERROR: Exception in ASGI application`과 함께
   `anyio.NoEventLoopError: Not currently running on any asynchronous event
   loop`가 발생해 화면이 전혀 뜨지 않는 것을 확인했다.
2. **Python 3.14가 원인이라는 가설 → 재현 실패로 기각** — 이 환경의 venv가
   Python 3.14.6이었고, chainlit이 쓰는 `nest_asyncio`(1.6.0, 2년 넘게
   업데이트 없음)가 최신 Python의 asyncio 내부 구현과 충돌할 가능성을
   의심했다. `winget`으로 Python 3.12.10을 추가 설치하고, 기존 venv를
   `venv_py314_old/`로 옮긴 뒤 3.12로 venv를 새로 만들어 `requirements.txt`를
   재설치했다. 그 과정에서 `grpcio`가 한 번 DLL 로드 실패(`ImportError:
   DLL load failed while importing cygrpc`)를 냈는데, 원인은 Python 버전이
   아니라 `pip install -r requirements.txt`가 `-q` 플래그 때문에 중간에
   조용히 일부만 설치된 상태였던 것으로 밝혀졌다 — verbose 모드로 다시
   설치하니 정상적으로 전체 의존성이 들어갔다. 이렇게 Python 3.12로 재설치를
   마친 뒤에도 **동일한 `anyio.NoEventLoopError`가 그대로 재현**되어,
   "Python 3.14가 원인"이라는 최초 가설을 기각했다.
3. **최소 재현 스크립트로 진짜 원인 격리** — `nest_asyncio.apply()` +
   `starlette.staticfiles.StaticFiles` + `uvicorn`만 남긴 독립 스크립트로
   좁혀 들어갔다. `uvicorn.run()`(top-level 래퍼)으로 재현하면 nest_asyncio가
   패치한 `asyncio.run()`이 최신 uvicorn이 넘기는 `loop_factory` 키워드
   인자를 못 받아 `TypeError`가 났고(이건 chainlit의 실제 호출 경로가 아니라
   허위 단서였다), chainlit이 실제로 쓰는 방식(`asyncio.run(server.serve())`)
   그대로 재현하자 `NoEventLoopError`가 나타났다. `nest_asyncio.apply()`
   호출 한 줄만 빼면 동일한 코드가 정상 동작(HTTP 200)하는 것으로 확인해,
   **원인이 Python 버전이 아니라 `chainlit/cli/__init__.py`가 무조건 호출하는
   `nest_asyncio.apply()`가 설치된 anyio(4.14.2)의 이벤트 루프 감지를 깨뜨리는
   것**임을 확정했다.
4. **`run_chainlit.py` 런처 작성** — chainlit이 `nest_asyncio`를 import하기
   전에 `sys.modules["nest_asyncio"]`를 `apply()`가 아무 것도 하지 않는
   가짜 모듈로 먼저 등록한 뒤 `chainlit.cli.cli()`를 그대로 호출하는 얇은
   래퍼를 추가했다. 이 프로젝트는 Jupyter 같은 재진입 이벤트 루프가 필요
   없으므로 `nest_asyncio`의 실제 패치 기능 자체가 애초에 불필요하다.
   `chainlit run`과 동일한 CLI 옵션(서브커맨드 인자를 그대로 전달)을
   지원한다.
5. **검증 및 정리** — `run_chainlit.py chainlit_app.py`로 루트 페이지와
   `/favicon` 정적 자산이 각각 HTTP 200을 반환하는 것을 확인했다. 재현에만
   쓰인 `venv_py314_old/` 백업은 사용자 확인 후 삭제했다. 결과적으로 venv는
   Python 3.12 기준으로 남았지만, 실제 버그 원인은 Python 버전이 아니었다는
   점이 이 조사의 핵심 결론이다 — 이후 동일 증상이 재발하면 `nest_asyncio`/
   `anyio`/`uvicorn` 버전 조합부터 의심할 것.
6. **(정정) venv는 현재 다시 Python 3.14.6이다** — 이후 세션에서 venv가
   Python 3.12로 손상되어 있는 것을 발견해(경위 불명) 원래대로 3.14.6으로
   재생성했다. `run_chainlit.py`는 Python 버전과 무관하게(원인이 애초에
   Python 버전이 아니었으므로) 3.14.6에서도 동일하게 동작을 확인했다.
7. **새로운 별개 문제 발견: Windows 애플리케이션 제어 정책이 `grpc` 로드를
   차단** — 위 nest_asyncio 문제와는 무관한 환경 문제를 추가로 발견했다.
   `import chainlit`이 `chainlit → literalai → traceloop → opentelemetry
   otlp grpc exporter → grpc`로 이어지는 원격 텔레메트리 의존성 체인을
   타는데, `grpc`의 네이티브 확장(`cygrpc.cp314-win_amd64.pyd`) 로드가
   `ImportError: DLL load failed while importing cygrpc: 애플리케이션 제어
   정책에서 이 파일을 차단했습니다`로 실패한다 — Windows Defender
   Application Control(WDAC)/AppLocker 같은 시스템 보안 정책이 서명되지
   않았거나(또는 새로 설치된) DLL의 실행을 막는 것으로 보인다. 이건 코드
   버그가 아니라 시스템 보안 설정이라 애플리케이션 코드로 우회할 수
   없다 — Claude Code의 안전 정책상으로도 시스템/보안 설정 변경은 직접
   수행할 수 없는 영역이다. 이 문제를 겪으면: (a) 시스템 관리자에게 해당
   DLL 또는 이 프로젝트의 venv 경로를 허용 목록에 추가해달라고 요청하거나,
   (b) `pip uninstall grpcio grpcio-tools`로 텔레메트리 경로 자체를 없애는
   방법을 시도해볼 수 있다(단, chainlit/literalai가 grpc를 완전히
   선택적으로 다루는지는 검증하지 않았다). 이 문제는 `run_chainlit.py`로
   해결되지 않는다 — `nest_asyncio` 문제와 별개로, `import chainlit` 자체가
   막히기 때문이다.
