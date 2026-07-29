# CLAUDE.md

이 파일은 다음 세션(Claude Code)이 이 프로젝트의 현재 작업 상황을 빠르게 파악할 수 있도록 만든 인계 문서입니다.

## 프로젝트 한 줄 요약

LangGraph 기반 **GitHub 저장소 리뷰/학습 에이전트**. TDD(Red-Green-Refactor)로 개발됨. 대화에서 언급된 `owner/repo` 저장소의 개요(README)·구조·소스 코드를 GitHub REST API(기본 비인증, `GITHUB_TOKEN` 설정 시 자동 인증)로 병렬 팬아웃 조회하고, 최종 답변은 **같은 과제를 여러 번(기본 3번) 독립 시도한 뒤 도구 실행 결과와 실제로 일치하는 답을 결정론적으로 골라 채택하는 투표(voting) 앙상블** 구조. "정식 리뷰"(`fetch_repo_source_sample`, 토큰 많이 씀)와 **스터디 모드**("구조 먼저, 필요한 파일만 나중에" — `fetch_repo_structure`+`fetch_repo_file`, 토큰 적게 씀 + `num_voters=1`로 다수결도 생략)를 지원 — 무료 티어처럼 토큰이 빠듯할 때 대형 저장소를 공부할 수 있게 하려는 목적.

## 이 저장소와 자매 저장소 (역사적 배경, 현재는 목적이 갈림)

| | 로컬 경로 | GitHub | 현재 목적 |
|---|---|---|---|
| **이 저장소** | `C:\Users\Y\Agent\ensemble_agent` | [Ensemble_type_LangGraph_agent](https://github.com/NoLucas/Ensemble_type_LangGraph_agent) | GitHub 저장소 리뷰 (voter_1/2/3 → 다수결) |
| 자매 저장소 | `C:\Users\Y\Agent\normal_agent` | [Code_Simple_LangGraph](https://github.com/NoLucas/Code_Simple_LangGraph) | 계산/파일 읽기·쓰기 에이전트 (단일 reporter) |

두 저장소는 원래 같은 git 히스토리(계산/파일 도구 + 투표형 앙상블 vs 관점형 앙상블)에서 갈라져 나왔지만, **이 저장소는 이후 GitHub 저장소 리뷰 전용 에이전트로 목적 자체가 바뀌었습니다**: `calculate`/`read_sandbox_file`/`write_sandbox_file` 도구를 완전히 제거하고 `fetch_repo_overview`/`fetch_repo_source_sample` 두 GitHub API 도구로 교체했습니다. 그래프 골격(dispatcher → 도구 2-way 팬아웃 → voter 3-way 팬아웃 → 결정론적 다수결)은 이전과 동일하지만, 도구/프롬프트/테스트는 전부 저장소 리뷰에 맞게 다시 작성되었습니다. `normal_agent`와의 기능 이식 관계는 더 이상 유지하지 않습니다 — 앞으로 `normal_agent`의 변경 사항(체크포인터 등)을 이 저장소로 자동으로 가져올 필요는 없습니다.

## 그래프 구조

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
voter_1_node   voter_2_node   voter_3_node   기본 N=3, 스터디 모드 N=1)
   └──────┼───────────────┴──────────┘
          ▼
   vote_for_best_report                (팬인: LLM 재호출 없이 결정론적 다수결)
          │
          ▼
         END
```

자세한 설명은 [README.md](README.md)를 참고하세요. 핵심만 요약하면:

- **dispatcher**: 도구 네 개(`fetch_repo_overview`/`fetch_repo_source_sample`/`fetch_repo_structure`/`fetch_repo_file`)를 bind_tools한 llm 사용. 사용자가 언급한 저장소마다 필요한 도구를 한 번에 요청하도록 유도. "정식 리뷰"(source_sample)와 "가볍게 훑기+드릴다운"(structure→file) 흐름을 사용자 의도로 구분해서 고르도록 `DISPATCHER_SYSTEM_PROMPT`가 안내.
- **repo_overview_node**: 저장소 메타데이터(설명/언어/stars/forks/topics) + README 발췌(최대 3000자)를 텍스트로 조립.
- **repo_source_node**: 저장소를 `/repos/{repo}/contents/{path}`로 디렉토리 단위(비재귀) DFS 탐색해서(`_discover_source_candidates()`, 최대 8회 호출) 소스 확장자(.py/.js/.ts/.go/...) 파일 후보를 모으고, `tests`/`vendor`/`node_modules`/`examples` 등 경로와 `_test.go`/`.spec.ts` 등 파일명 패턴(`_looks_like_test_file()`)을 제외한 뒤, `_score_source_candidate()`로 점수(주 언어 일치 +10, 진입점 파일명 +5, 얕은 경로 가산, 빈 파일 -100/적당한 크기 +2)를 매겨 상위 3개의 **전체 내용**(파일당 최대 1500자)을 가져옴. `/git/trees/{branch}?recursive=1`(저장소 전체를 한 번에 받음)은 더 이상 쓰지 않음.
- **repo_structure_node**: `repo_source_node`와 동일한 탐색/채점 로직을 재사용하되, 최대 10개 파일까지 더 넓게 훑고 각 파일은 전체 본문 대신 `_extract_signatures()`(언어별 정규식 휴리스틱)로 뽑은 함수/클래스 시그니처만 담는다 — 스터디 모드 1단계, 토큰을 훨씬 적게 씀.
- **repo_file_node**: 사용자가 지정한 `path` 하나만 탐색 없이 바로 조회(`fetch_repo_file_text`, API 호출 1번) — 스터디 모드 2단계(드릴다운).
- **voter_1..voter_N**(기본 N=3, `build_graph(num_voters=)`로 조절): **모두 같은 프롬프트**(`VOTER_SYSTEM_PROMPT`)로 사용자 요청에 맞는 답변(리뷰/구조 설명/특정 파일 설명)을 독립 시도. 관점을 나누지 않는다 — 다양성은 모델 샘플링 자체의 변동성에서 나온다(실서비스에서는 temperature). bind_tools 안 된 llm이라 구조적으로 도구를 재호출 못 함. 두 섹션(요약/코드 리뷰)을 더 이상 강제하지 않음 — 어떤 도구가 호출됐는지에 맞춰 알아서 형식을 정함.
- **vote_for_best_report**: `llm` 파라미터가 아예 없는 순수 함수. 도구 실행 결과(`ToolMessage`)에서 뽑은 "사실 토큰"(숫자/영문 파일명·식별자/한글 단어)을 candidate가 몇 개나 포함하는지로 채점해 다수결로 선택. 동점이면 `voter_1 → voter_2 → ...` 순서로 타이브레이크. `num_voters=1`이면 다수결 없이 단일 candidate를 그대로 채택(로직은 동일, 그냥 후보가 하나뿐).

## 핵심 설계 결정 (왜 이렇게 했는지)

- **GitHub API는 기본 비인증, `GITHUB_TOKEN`이 있으면 자동 인증**: 처음엔 사용자가 명시적으로 비인증만 선택했지만(토큰 발급/관리 부담 없음), 시간당 60회 제한이 저장소 하나당 최대 5회(overview 2회 + source 3회) 소모라 금방 걸리는 문제가 있어 이후 토큰 인증을 선택적으로 추가했다. `agent/tools.py`의 `_auth_headers()`가 매 `requests.get` 호출 시점에 `os.environ["GITHUB_TOKEN"]`을 읽어 있으면 `Authorization: Bearer ...` 헤더를 붙이고, 없으면 기존과 동일하게 비인증으로 호출한다 — 토큰을 안 넣는 사용자에게는 아무 동작 변화가 없다. 기존 Accept 헤더(raw 콘텐츠 요청)와 병합해야 하므로 헤더를 통째로 덮어쓰지 않고 `dict(extra or {})`에 추가하는 방식을 썼다.
- **`iteration`은 절대값이 아니라 델타(`Annotated[int, operator.add]`)**: voter 3개가 병렬로 같은 state 스냅샷을 읽고 "현재+1"을 계산하면 `InvalidUpdateError`가 나거나 값이 유실된다(실제로 겪은 버그, `bugfix.md` 참고). 그래서 각 노드가 항상 델타 `1`만 반환하고 리듀서가 합산한다.
- **`vote_for_best_report_node`에 `llm` 파라미터가 없음**: "종합" 단계에 LLM을 또 넣으면 그 단계 자체가 draft에 없던 내용을 지어내는 환각 진입점이 된다. 함수 시그니처 자체에 `llm`이 없어서 구조적으로 그 경로가 존재하지 않는다.
- **`vote_for_best_report_node`는 "전체 문자열 포함"이 아니라 "토큰 겹침"으로 채점**: 계산기 버전은 `ToolMessage`가 `"4"`처럼 짧아서 candidate가 그대로 인용하기 쉬웠지만, 저장소 리뷰는 `ToolMessage`가 수백~수천 자짜리 자유 서술형 텍스트라 voter가 그 블록을 통째로 인용하는 일이 없다. 전체 문자열 포함 여부만 봤다면 거의 항상 0점이 나와 다수결이 사실상 무력화되고 매번 `voter_1`로 fallback됐을 것이다(실제로 겪은 버그는 아니지만, 구조를 바꾸지 않고 도구만 교체했다면 반드시 겪었을 문제라 사전에 고쳤다). `_FACT_TOKEN_PATTERN`(`agent/nodes.py`)으로 숫자/영문 식별자/한글 단어를 토큰화해서 겹치는 개수로 채점하도록 바꿨다 — `tests/test_nodes.py`의
  `test_vote_for_best_report_node_uses_token_overlap_not_exact_substring`이 "정답을 정확히 패러프레이즈한 candidate가 무의미한 첫 번째 candidate를 이긴다"는 걸 회귀 테스트로 고정해둔다.
- **`FakeChatModel`(`tests/conftest.py`)에 `threading.Lock`**: voter 3개가 그래프 실행 중 같은 인스턴스를 스레드 풀에서 동시 호출한다. 락 없이는 `calls` 증가/응답 인덱싱이 경합한다(실제로 겪은 버그).
- **GitHub API 테스트는 `QueuedGet`(`tests/conftest.py`)으로 모킹**: 도구가 `requests.get`을 순차적으로 여러 번(메타데이터 → README/트리 → 파일 내용) 호출하므로, URL별 매칭 대신 "호출 순서대로 응답을 미리 큐에 넣어두는" 방식이 가장 단순했다. `FakeChatModel`과 동일한 설계 철학(결정적, 실제 네트워크 없음).
- **사이클 없는 DAG**: dispatcher/voter가 서로 다른 llm 바인딩(tools 유무)을 쓰기 때문에 무한 루프 방지용 반복 상한(`MAX_ITERATIONS`) 같은 장치가 아예 필요 없다.

## 실행 / 테스트

```bash
venv/Scripts/python.exe -m pytest -q        # 81개 테스트 (FakeChatModel + QueuedGet/PathAwareQueuedGet 기반 79개 + 실LLM 통합 2개, 전부 통과해야 정상)
python main.py                              # 콘솔
streamlit run app.py                        # Streamlit 웹
chainlit run chainlit_app.py                # Chainlit 웹 (도구 실행 과정 시각화)
```

이 폴더는 자체 `venv/`를 가지고 있습니다. 의존성 설치:
```bash
venv/Scripts/python.exe -m pip install -r requirements.txt
```

세 진입점 모두 초기 state에 `"report_drafts": []`를 반드시 포함해야 합니다 (`operator.add` 리듀서 채널이 첫 병렬 쓰기 전에 초기화돼 있어야 함).

## 현재 상태

### 완료됨
- GitHub 도구 2개(`fetch_repo_overview`/`fetch_repo_source_sample`) + 팬아웃/팬인, 정상/실패(저장소 없음·rate limit·네트워크 오류) 케이스 전부 테스트 포함
- 투표형 앙상블(voter × 3 + 결정론적 다수결)을 리뷰(요약+코드 리뷰) 프롬프트로 재구성
- **다수결 채점 로직을 "전체 문자열 포함"에서 "사실 토큰 겹침"으로 개선**: 저장소 리뷰처럼 `ToolMessage`가 길고 자유 서술형일 때 옛 방식(전체 문자열 포함)은 거의 항상 0점이 나와 다수결이 무력화되던 문제를 고쳤다. 회귀 테스트로 "정답을 정확히 패러프레이즈한 candidate가 무의미한 답을 이기는지" 고정해둠.
- 콘솔(`main.py`) / Streamlit(`app.py`) / Chainlit(`chainlit_app.py`) 진입점을 새 목적에 맞게 텍스트/라벨 갱신
- 기존 계산/파일 도구(`calculate`/`read_sandbox_file`/`write_sandbox_file`)와 관련 테스트·`sandbox_data/` 완전 제거
- **실LLM 통합 테스트(`tests/test_integration.py`) 작성 및 실제 API 키로 통과 확인**: `claude-haiku-4-5-20251001` + 실제 GitHub API(`octocat/Hello-World`)로 (1) dispatcher가 대화에서 `owner/repo`를 정확히 도구 인자로 추출하는지, (2) 전체 파이프라인이 voter 3개를 채우고 최종 리뷰를 만들어내는지 검증. `integration` 마커라 기본 `pytest`/CI에서는 제외되고, `ANTHROPIC_API_KEY`가 없으면 실패 대신 skip된다.
- **`GITHUB_TOKEN` 선택적 인증 지원**: `.env`에 설정하면 모든 GitHub API 호출에 자동으로 `Authorization` 헤더가 붙어 시간당 60회 → 5000회로 늘어난다. 미설정 시 기존과 동일하게 비인증으로 동작(하위 호환).
- **CI 실제 통과 확인**: GitHub Actions에서 이 세션의 push 7개 전부 `success` 확인함(로컬 통과만이 아니라 실제 Actions 로그로 검증).
- **Chainlit 인터페이스 다듬기 + 실제 브라우저 수동 테스트**: `chainlit.md`/`.chainlit/config.toml`의 이름·설명을 새 목적에 맞게 갱신, `@cl.set_starters`로 예시 프롬프트 3개 추가, `on_message`에 `try/except`로 LLM 호출 실패 시 친절한 에러 메시지 추가. 실제 브라우저로 "개요만" / "요약+코드 리뷰" 두 시나리오를 끝까지 실행해서 Step 시각화·투표·한글 렌더링이 정상임을 확인했고, GitHub rate limit에 실제로 걸렸을 때도 도구가 에러 문자열을 반환하고 voter가 이를 자연스럽게 리포트에 반영하는 것까지 확인했다(설계대로 동작).
- **수동 테스트로 실제 버그 발견 및 수정**: `chainlit_app.py`의 `graph.stream(..., stream_mode="updates")` 루프가 `node_output.get(...)`을 호출하는데, 설치된 LangGraph 1.2.9에서는 노드가 아무 것도 반환하지 않으면(`{}`가 아니라) 업데이트 값 자체가 `None`으로 온다 — `repo_overview_node`/`repo_source_node`가 항상 함께 깨워지지만 자기 몫이 없으면 통과하는 설계상 **매 요청마다 발생하는 정상 케이스**인데도 `.get()`이 `NoneType`에서 죽었다. `node_output is None`이면 continue하도록 수정. 이 프로젝트의 FakeChatModel 기반 테스트는 `graph.invoke()`만 검증하고 `graph.stream(stream_mode="updates")`의 실제 반환 모양은 검증하지 않아서 놓쳤던 버그 — 실제 LLM + 실제 브라우저로 돌려보지 않았다면 못 잡았을 것.
- **`fetch_repo_source_sample`의 파일 선택을 "경로 깊이만"에서 "점수 기반"으로 고도화**: 저장소의 대표 언어(`language` 메타데이터)와 확장자 일치, 진입점 파일명(`main`/`index`/`app`/`__init__` 등), 얕은 경로, 적당한 파일 크기(50~20000바이트, 빈 파일은 사실상 배제)를 점수로 합산해 상위 3개를 고른다(`_score_source_candidate()`). `psf/requests`/`expressjs/express` 등 실제 저장소로 수동 검증하는 과정에서 `examples/`의 데모 `index.js`가 진입점 가산점 때문에 실제 라이브러리 구현(`lib/`)보다 먼저 뽑히는 문제를 발견해 `_SKIP_PATH_PARTS`에 `examples`/`demo`/`sample` 계열을 추가로 제외했다(`bugfix.md` 참고 — 유닛 테스트 픽스처만으로는 못 잡고 실제 저장소로 눈으로 확인해야 드러난 문제).
- **대형 저장소에서 GitHub 응답이 너무 커서 사실상 리뷰 불가능하던 문제 해결**: `/git/trees/{branch}?recursive=1`(저장소 전체 파일 목록을 한 번에 받음)을 `/repos/{repo}/contents/{path}` 디렉토리 단위 DFS 탐색(`_discover_source_candidates()`)으로 교체했다. 실측: `kubernetes/kubernetes` 10MB/37,390개 항목 → 약 4KB/파일 3개, `torvalds/linux` 17.6MB/71,798개 항목(그나마 `truncated: true`로 일부 누락) → 마찬가지로 가벼워짐. 처음엔 너비 우선(BFS)으로 만들었다가 `kubernetes/kubernetes`로 검증하며 "루트 바로 아래엔 컴포넌트별 얕은 디렉토리만 잔뜩 있고 실제 코드는 몇 단계 안쪽"이라 BFS가 호출 예산(당시 5회)을 옆으로 훑다가 다 쓰고 파일을 하나도 못 찾는 걸 발견 — DFS(+힌트 디렉토리 우선, 예산 8회)로 바꿔서 고쳤다. 이 과정에서 `_test.go`/`.spec.ts` 같은 파일명 기반 테스트 파일(디렉토리 관례가 아니라 명명 관례를 쓰는 언어)이 안 걸러지는 것도 추가로 발견해 `_looks_like_test_file()`로 고쳤다. 회귀 테스트 6개 추가(재귀 트리 엔드포인트 미사용, 호출 횟수 상한, 하위 디렉토리 실패 허용, 깊이 우선 탐색, 테스트 파일명 제외 등).
- **스터디 모드 + 도구 2개 추가**: 사용자가 "무료 티어라 토큰이 부족한데 대형 저장소를 효과적으로 공부할 방법"을 요청해서 구현했다.
  - `fetch_repo_structure`(신규): `repo_source_node`와 같은 탐색/채점 로직을 재사용하되 파일을 최대 10개까지 더 넓게 훑고, 전체 본문 대신 `_extract_signatures()`(Python/JS·TS/Go/Rust/Java·Kotlin/Ruby용 정규식 휴리스틱)로 뽑은 함수/클래스 시그니처만 담는다.
  - `fetch_repo_file`(신규): 사용자가 지정한 `path` 하나만 탐색 없이 바로 조회(`_fetch_file_raw()` 공용 헬퍼로 `fetch_repo_source_sample`과 코드 공유). API 호출이 1번뿐이라 가장 저렴.
  - `build_graph(llm, num_voters=3)`: voter 개수를 조절 가능하게 만들었다. `num_voters=1`(스터디 모드)이면 다수결 없이 단일 시도를 그대로 채택 — LLM 호출이 4번(dispatcher+voter 3)에서 2번(dispatcher+voter 1)으로 줄어든다. `VOTER_LABELS[:num_voters]`로 슬라이싱만 하면 되고, `vote_for_best_report_node`는 코드 수정 없이도 그대로 동작한다(누락된 라벨은 원래 조용히 건너뛰는 설계였기 때문).
  - `DISPATCHER_SYSTEM_PROMPT`/`VOTER_SYSTEM_PROMPT`를 4-도구/가변 형식에 맞게 재작성 — voter가 "요약+코드 리뷰" 두 섹션을 항상 강제하지 않고 사용자 요청(구조 지도/특정 파일/포괄적 리뷰)에 맞춰 알아서 형식을 고른다.
  - `main.py`(콘솔: 시작 시 y/N 프롬프트) / `app.py`(Streamlit: 사이드바 토글, `st.cache_resource`가 `num_voters` 인자별로 캐시 분리) / `chainlit_app.py`(Chainlit: `cl.ChatSettings` + `Switch` 위젯, `on_settings_update`가 세션별 그래프 캐시를 무효화)에 모두 연결.
  - 테스트: `_extract_signatures` 언어별 4개, `fetch_repo_structure_text`/`fetch_repo_file_text` 정상/실패 케이스, 두 노드 단위 테스트, `num_voters=1` 그래프 E2E, `num_voters` 범위 검증(`ValueError`), 구조+파일 도구 동시 호출 E2E.
  - **테스트 인프라 개선**: `repo_structure_node`+`repo_file_node`처럼 서로 다른 두 도구 노드가 같은 팬아웃 라운드에서 병렬로(스레드 풀) 각자 여러 번 `requests.get`을 호출하면, 기존 `QueuedGet`(순수 FIFO)은 어느 응답이 어느 노드로 갔는지 보장 못 한다는 걸 발견 — URL 부분 문자열로 매칭하는 `PathAwareQueuedGet`(`mock_github_get_by_path` 픽스처)을 새로 추가했다. 기존 `QueuedGet`에도 `FakeChatModel`과 동일한 이유로 `threading.Lock`을 뒤늦게 추가함(경합 자체를 막을 뿐 순서 뒤섞임은 못 막으므로, 여러 도구 노드가 동시에 여러 번 호출하는 시나리오는 반드시 `PathAwareQueuedGet`을 써야 한다).

### 아직 안 된 것 / 알려진 한계
- **체크포인터(SqliteSaver) 미지원**: `agent/graph.py`의 `build_graph()`가 `checkpointer` 파라미터를 받지 않는다. (사용자 확인: 필요 없음.)
- **GitHub rate limit은 `GITHUB_TOKEN`으로 완화 가능하지만 기본값은 여전히 비인증**: `.env`에 토큰을 넣지 않으면 시간당 60회 제한 그대로다. `README.md`에 발급 방법을 안내해뒀지만, 자동으로 토큰을 발급/설정해주지는 않는다.
- **투표 로직이 여전히 완벽하지 않음**: 토큰 겹침으로 개선했지만 `_FACT_TOKEN_PATTERN`은 숫자를 아라비아 숫자로만 인식한다(예: stars 수를 "42"가 아니라 "마흔두 개"로 표현하면 못 잡음). 한글 형태소 분석기가 아니라 정규식 기반이라 조사가 붙은 단어("저장소로", "저장소는")는 원형("저장소")과 정확히 일치하지 않으면 놓칠 수 있다(코드 리뷰용 영문 식별자·숫자는 이 문제가 없다).
- **이 개발 환경(Python 3.14.6)에서 `chainlit run chainlit_app.py`이 그대로는 안 뜬다 → `run_chainlit.py`로 해결**: chainlit의 `chainlit.cli`가 임포트 시점에 `nest_asyncio.apply()`를 무조건 호출하는데, 이 패치가 `asyncio.current_task()` 기반 추적을 깨뜨려서(Python 3.14 + anyio 4.14.2 조합) anyio의 `CancelScope`/`_task_states` 조회가 `TypeError: cannot create weak reference to 'NoneType' object`로 죽는다 — 정적 프론트엔드 자산(SPA index.html/JS 번들)이 전혀 로드되지 않는다. `chainlit_app.py`에 추가한 `sniffio.current_async_library_cvar.set("asyncio")`는 이 문제의 1단계(`sniffio` 오판)만 고치고 근본 원인(`nest_asyncio`가 태스크 추적 자체를 깨는 것)은 남기므로 단독으로는 부족했다. 최종 해법은 프로젝트 루트의 `run_chainlit.py` — `chainlit.cli`가 `nest_asyncio`를 import하기 **전에** `sys.modules["nest_asyncio"]`를 `apply()`가 no-op인 가짜 모듈로 바꿔치기한 뒤 `chainlit.cli.cli()`를 그대로 호출한다. `chainlit run`과 동일한 CLI 인터페이스(`-h`, `--port` 등)를 그대로 지원하면서 문제의 패치 호출만 무력화하는 방식이라, `chainlit run`을 완전히 재구현하는 것보다 훨씬 얇고 안전하다. 실제 브라우저로 "개요만"/"코드 리뷰"/"도구 불필요" 세 시나리오를 이 스크립트로 끝까지 검증했다. (참고: 이 파일은 이 세션에서 예상치 못하게 프로젝트 디렉토리에 이미 생성되어 있었다 — 정확한 생성 경위는 파악하지 못했지만 내용을 직접 검증한 뒤 채택했다.)
- **[NEW, 미해결] `import chainlit` 자체가 Windows 애플리케이션 제어 정책에 막힘**: 위 `nest_asyncio` 문제와 별개로, `chainlit → literalai → traceloop → opentelemetry otlp grpc exporter → grpc`로 이어지는 텔레메트리 의존성 체인에서 `grpc`의 네이티브 확장(`cygrpc.cp314-win_amd64.pyd`)이 `ImportError: DLL load failed while importing cygrpc: 애플리케이션 제어 정책에서 이 파일을 차단했습니다`로 실패한다(WDAC/AppLocker로 추정). `run_chainlit.py`로도 우회 불가능 — `chainlit.cli` import 자체가 이 시점에 이미 실패한다. 시스템 보안 정책이라 코드로 못 고친다(Claude Code 정책상 시스템 설정 변경 자체도 직접 수행 불가). 스터디 모드 관련 chainlit_app.py 변경사항은 소스 코드 검토(`chainlit.input_widget.Switch`/`ChatSettings` 실제 시그니처를 소스 파일로 직접 대조)로만 검증했고, 실제 브라우저로는 확인 못 했다. 다음 세션에서 이 문제가 사라졌으면 그냥 진행하면 되고, 여전하면 사용자에게 WDAC 예외 등록을 요청하거나 `pip uninstall grpcio`로 텔레메트리 경로 자체를 없애는 시도를 해볼 것(단 chainlit/literalai가 grpc를 선택적으로 다루는지는 미검증).
- **[NEW] `.env`의 `ANTHROPIC_API_KEY`가 현재 무효(401 invalid x-api-key)**: `test_integration.py` 2개와 `main.py`/`app.py`를 통한 실LLM 수동 검증이 막혀 있다. 사용자가 이전에 채팅에 노출된 키를 폐기했을 가능성이 높다(이 세션 초반에 그렇게 권장한 바 있음) — 유효한 키로 교체 필요.

## 관련 문서
- [README.md](README.md) — 사용자 대상 설명 (구조, 도구, 설치법)
- [bugfix.md](bugfix.md) — 개발 중 실제로 겪은 버그와 수정 기록 (계산기 버전 시절 버그 포함, 여전히 유효한 교훈)
