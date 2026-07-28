# Ensemble_type_LangGraph_agent

LangGraph 기반 GitHub 저장소 리뷰 에이전트입니다. TDD(Red-Green-Refactor)로 개발되었으며, 사용자가 대화에서 언급한 `owner/repo` 저장소의 개요(README)와 대표 소스 코드를 GitHub API로 가져와 병렬 처리하고, 최종 리뷰(요약 + 코드 리뷰)는 동일한 과제를 3번 독립 시도한 뒤 도구 실행 결과와 실제로 일치하는 답을 결정론적으로 골라 채택하는 투표(voting) 앙상블 구조입니다.

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
   ┌──────┴──────┐
   ▼             ▼                       (도구 팬아웃 — 정적 2-way)
repo_overview_node  repo_source_node
   └──────┬──────┘
          ▼
   ┌──────┼───────────────┬──────────┐
   ▼      ▼               ▼          (투표형 앙상블 팬아웃 — 정적 3-way,
voter_1_node   voter_2_node   voter_3_node   셋 다 같은 프롬프트로 독립 시도)
   └──────┼───────────────┴──────────┘
          ▼
   vote_for_best_report                (팬인: LLM 재호출 없이 결정론적 다수결)
          │
          ▼
         END
```

- **dispatcher**: 사용자 입력에서 언급된 GitHub 저장소(`owner/repo`)를 해석해 `fetch_repo_overview`(개요/README) / `fetch_repo_source_sample`(대표 소스 코드) 중 필요한 도구를 지시합니다. 여러 저장소·도구가 필요하면 한 번의 응답에서 모두 요청하도록 유도해 팬아웃이 실제로 병렬 이득을 보게 합니다.
- **도구 팬아웃/팬인**: 두 도구 노드는 항상 함께 깨워지고, 자기 담당 tool_call이 없으면 조용히 통과(pass-through)합니다.
- **투표형 앙상블 팬아웃/팬인**: `voter_1`/`voter_2`/`voter_3`는 **모두 같은 프롬프트(`VOTER_SYSTEM_PROMPT`)**로 "요약 + 코드 리뷰" 두 섹션을 포함한 최종 리뷰를 각자 독립적으로 시도합니다. 관점을 나누는 앙상블이 아니라 같은 과제를 여러 번 독립 시도해서 검증하는 투표 앙상블이며, 다양성은 프롬프트가 아니라 모델 샘플링 자체의 변동성(temperature)에서 나옵니다.
- **vote_for_best_report**: 도구 실행 결과(`ToolMessage`: 저장소 개요/소스 코드)에서 "사실 토큰"(숫자, 영문 파일명/식별자, 한글 단어)을 뽑아, 세 candidate 중 이 토큰을 가장 많이 포함한 것을 **LLM을 다시 호출하지 않고** 결정론적으로 고르는 함수입니다. `ToolMessage` 내용이 수백~수천 자짜리 자유 서술형 텍스트라 voter가 그 블록을 통째로 인용하는 일은 없으므로, "전체 문자열이 그대로 포함되는가"가 아니라 "토큰 단위로 얼마나 겹치는가"로 채점합니다 — voter가 문장을 바꿔 써도(패러프레이즈해도) 숫자/고유명사 같은 사실은 그대로 잡아냅니다. 동점이면 `voter_1 → voter_2 → voter_3` 순서로 타이브레이크하고, 아무도 사실을 못 맞혀도 예외 없이 첫 번째 voter의 답을 반환합니다. "어느 게 더 그럴듯한가"를 LLM 판사에게 다시 묻지 않으므로, 이 종합 단계에는 환각이 새로 끼어들 여지가 없습니다.
- **도구가 필요 없으면** dispatcher의 답변이 그대로 최종 답변이 되고, 도구 팬아웃/투표 팬아웃 단계는 아예 실행되지 않습니다 (불필요한 LLM 호출을 만들지 않습니다).

이 그래프는 사이클이 없는 DAG입니다 — `dispatcher`는 tool이 바인딩된 llm을, `voter_*` 노드들은 바인딩되지 않은 llm을 받아서 구조적으로 도구를 다시 호출할 수 없습니다. 그래서 반복 상한(iteration cap) 같은 무한 루프 방지 장치가 필요 없습니다.

## 파일 구성

```
agent/
  state.py    # AgentState: messages(add_messages), iteration(operator.add 델타), report_drafts(operator.add)
  tools.py    # fetch_repo_overview, fetch_repo_source_sample (GitHub REST API, GITHUB_TOKEN 선택적 인증)
  nodes.py    # dispatcher/voter/vote_for_best_report 노드, 라우팅 함수, 도구 실행 노드
  graph.py    # build_graph(llm) — 전체 그래프 조립
tests/
  conftest.py         # FakeChatModel + mock_github_get(QueuedGet) — 실제 API 호출 없이 결정적 테스트
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
| `fetch_repo_source_sample` | 저장소 트리에서 대표 소스 파일 최대 3개의 내용 발췌 | 소스 확장자 화이트리스트, `tests/vendor/node_modules` 등 경로 제외, 파일당/README 길이 상한 |

모든 도구는 예외를 던지지 않고 `"Error: ..."` 문자열을 반환합니다 — LLM이 스키마와 맞지 않는 입력을 줄 수 있고 GitHub API도 실패할 수 있으므로, 실패도 정상적인 반환값으로 표현해서 그래프 전체가 죽지 않게 합니다.

GitHub API는 기본적으로 **비인증으로 호출**합니다(시간당 60회 제한). `.env`에 `GITHUB_TOKEN`을 설정하면 모든 요청에 자동으로 `Authorization: Bearer <GITHUB_TOKEN>` 헤더가 붙어 시간당 5000회로 늘어납니다(`agent/tools.py`의 `_auth_headers()`). 별도 권한이 필요 없는(공개 저장소 읽기 전용) [개인용 액세스 토큰](https://github.com/settings/tokens)이면 충분합니다. 토큰을 설정하지 않으면 기존과 동일하게 비인증으로 동작합니다.

## 설치 및 테스트

```bash
python -m venv venv
venv/Scripts/activate   # Windows
pip install -r requirements.txt
pytest
```

현재 47개 테스트가 전부 통과합니다 (state/tools/nodes/routing/graph E2E 계층별 TDD 45개 + 실LLM 통합 테스트 2개).

느린 테스트(실LLM + 실제 GitHub API 호출)는 `integration` 마커(`tests/test_integration.py`)로 분리되어 있으며 기본 실행에서 제외하는 것을 권장합니다:

```bash
pytest -m "not integration"
```

`test_integration.py`는 `.env`의 `ANTHROPIC_API_KEY`가 설정된 경우에만 실행되고, 없으면 실패하지 않고 자동으로 skip됩니다. 키가 있는 상태에서 통합 테스트만 따로 돌리려면:

```bash
pytest -m integration
```

`tests/conftest.py`의 `FakeChatModel`은 `threading.Lock`으로 `invoke()`를 보호합니다 — voter 3개가 그래프 실행 중 같은 `FakeChatModel` 인스턴스를 스레드 풀에서 동시에 호출하므로, 락이 없으면 `calls` 증가와 응답 인덱싱이 경합해 테스트가 간헐적으로 실패할 수 있습니다. `mock_github_get`(`QueuedGet`)은 `requests.get`을 호출 순서 기반 더미로 교체해 실제 GitHub API를 타지 않고 도구 로직을 검증합니다.

## 그래프 사용 예시

```python
from langchain_anthropic import ChatAnthropic
from agent.graph import build_graph

llm = ChatAnthropic(model="claude-sonnet-5")
graph = build_graph(llm)

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

초기 state에는 `report_drafts: []`를 반드시 포함해야 합니다 — 3-way 투표형 앙상블(`voter_1`/`voter_2`/`voter_3`)이 병렬로 채우는 `operator.add` 리듀서 필드라, 첫 병렬 쓰기 전에 채널이 초기화돼 있어야 안전합니다. `chainlit_app.py`처럼 `graph.invoke()` 대신 `graph.stream()`으로 상태를 직접 재구성하는 경우, `iteration`과 `report_drafts` 모두 **델타/리스트를 누적**해야 합니다(마지막 값으로 덮어쓰면 안 됩니다) — 두 필드 모두 `operator.add` 리듀서를 쓰기 때문입니다.
