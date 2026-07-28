# CLAUDE.md

이 파일은 다음 세션(Claude Code)이 이 프로젝트의 현재 작업 상황을 빠르게 파악할 수 있도록 만든 인계 문서입니다.

## 프로젝트 한 줄 요약

LangGraph 기반 **GitHub 저장소 리뷰 에이전트**. TDD(Red-Green-Refactor)로 개발됨. 대화에서 언급된 `owner/repo` 저장소의 개요(README)와 대표 소스 코드를 GitHub REST API(비인증)로 병렬 팬아웃 조회하고, 최종 리뷰(요약 + 코드 리뷰)는 **같은 과제를 3번 독립 시도한 뒤 도구 실행 결과와 실제로 일치하는 답을 결정론적으로 골라 채택하는 투표(voting) 앙상블** 구조.

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

자세한 설명은 [README.md](README.md)를 참고하세요. 핵심만 요약하면:

- **dispatcher**: `fetch_repo_overview`/`fetch_repo_source_sample`를 bind_tools한 llm 사용. 사용자가 언급한 저장소마다 필요한 도구를 한 번에 요청하도록 유도.
- **repo_overview_node**: 저장소 메타데이터(설명/언어/stars/forks/topics) + README 발췌(최대 3000자)를 텍스트로 조립.
- **repo_source_node**: 저장소 트리에서 소스 확장자(.py/.js/.ts/.go/...) 파일 중 `tests`/`vendor`/`node_modules` 등을 제외하고, 경로 깊이가 얕은 순으로 최대 3개를 골라 내용(파일당 최대 1500자)을 가져옴.
- **voter_1/2/3**: **모두 같은 프롬프트**(`VOTER_SYSTEM_PROMPT`)로 "요약 + 코드 리뷰" 두 섹션을 포함한 리뷰를 독립 시도. 관점을 나누지 않는다 — 다양성은 모델 샘플링 자체의 변동성에서 나온다(실서비스에서는 temperature). bind_tools 안 된 llm이라 구조적으로 도구를 재호출 못 함.
- **vote_for_best_report**: `llm` 파라미터가 아예 없는 순수 함수. 도구 실행 결과(`ToolMessage`)를 실제로 포함한 candidate만 통과시켜 다수결로 선택. 동점이면 `voter_1 → voter_2 → voter_3` 순서로 타이브레이크.

## 핵심 설계 결정 (왜 이렇게 했는지)

- **GitHub API는 비인증으로 호출**: 사용자가 명시적으로 선택한 방향(토큰 발급/관리 부담 없음). 시간당 60회 제한이라 저장소 하나를 리뷰할 때마다 최대 5회(overview 2회 + source 3회) 요청이 나가므로, 짧은 시간에 여러 저장소를 반복 리뷰하면 금방 제한에 걸릴 수 있다. 필요해지면 `agent/tools.py`의 `requests.get` 호출에 `Authorization` 헤더를 추가해 토큰 인증으로 전환 가능(코드에 훅은 없음, 아직 요청받지 않아 구현 안 함).
- **`iteration`은 절대값이 아니라 델타(`Annotated[int, operator.add]`)**: voter 3개가 병렬로 같은 state 스냅샷을 읽고 "현재+1"을 계산하면 `InvalidUpdateError`가 나거나 값이 유실된다(실제로 겪은 버그, `bugfix.md` 참고). 그래서 각 노드가 항상 델타 `1`만 반환하고 리듀서가 합산한다.
- **`vote_for_best_report_node`에 `llm` 파라미터가 없음**: "종합" 단계에 LLM을 또 넣으면 그 단계 자체가 draft에 없던 내용을 지어내는 환각 진입점이 된다. 함수 시그니처 자체에 `llm`이 없어서 구조적으로 그 경로가 존재하지 않는다.
- **`FakeChatModel`(`tests/conftest.py`)에 `threading.Lock`**: voter 3개가 그래프 실행 중 같은 인스턴스를 스레드 풀에서 동시 호출한다. 락 없이는 `calls` 증가/응답 인덱싱이 경합한다(실제로 겪은 버그).
- **GitHub API 테스트는 `QueuedGet`(`tests/conftest.py`)으로 모킹**: 도구가 `requests.get`을 순차적으로 여러 번(메타데이터 → README/트리 → 파일 내용) 호출하므로, URL별 매칭 대신 "호출 순서대로 응답을 미리 큐에 넣어두는" 방식이 가장 단순했다. `FakeChatModel`과 동일한 설계 철학(결정적, 실제 네트워크 없음).
- **사이클 없는 DAG**: dispatcher/voter가 서로 다른 llm 바인딩(tools 유무)을 쓰기 때문에 무한 루프 방지용 반복 상한(`MAX_ITERATIONS`) 같은 장치가 아예 필요 없다.

## 실행 / 테스트

```bash
venv/Scripts/python.exe -m pytest -q        # 42개 테스트 (FakeChatModel + QueuedGet 기반, 전부 통과해야 정상)
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
- 투표형 앙상블(voter × 3 + 결정론적 다수결)을 리뷰(요약+코드 리뷰) 프롬프트로 재구성, 42개 단위/통합(FakeChatModel) 테스트 통과
- 콘솔(`main.py`) / Streamlit(`app.py`) / Chainlit(`chainlit_app.py`) 진입점을 새 목적에 맞게 텍스트/라벨 갱신
- 기존 계산/파일 도구(`calculate`/`read_sandbox_file`/`write_sandbox_file`)와 관련 테스트·`sandbox_data/` 완전 제거
- **실LLM 통합 테스트(`tests/test_integration.py`) 작성 및 실제 API 키로 통과 확인**: `claude-haiku-4-5-20251001` + 실제 GitHub API(`octocat/Hello-World`)로 (1) dispatcher가 대화에서 `owner/repo`를 정확히 도구 인자로 추출하는지, (2) 전체 파이프라인이 voter 3개를 채우고 최종 리뷰를 만들어내는지 검증. `integration` 마커라 기본 `pytest`/CI에서는 제외되고, `ANTHROPIC_API_KEY`가 없으면 실패 대신 skip된다.

### 아직 안 된 것 / 알려진 한계
- **체크포인터(SqliteSaver) 미지원**: `agent/graph.py`의 `build_graph()`가 `checkpointer` 파라미터를 받지 않는다.
- **GitHub 비인증 rate limit**: 시간당 60회라 여러 저장소를 짧은 시간에 반복 리뷰하는 시나리오는 취약하다. 필요해지면 토큰 인증으로 전환.
- **투표 로직이 단순 문자열 포함 검사**: `vote_for_best_report_node`가 숫자/표현 차이(예: stars 수를 "42"가 아니라 "마흔두 개"로 표현)에는 관대하지 않다. 실LLM 통합 테스트가 구조(voter 3개 채워짐, 최종 답변에 저장소명 포함)까지는 확인했지만, "다수결이 항상 가장 정확한 draft를 고르는지"까지는 검증하지 않는다.
- **CI(`.github/workflows/tests.yml`)가 새 테스트 스위트로 통과하는지 로컬에서만 확인함**: 실제 GitHub Actions 실행 로그는 아직 못 봤다. `test_integration.py`는 `integration` 마커라 CI(`pytest -m "not integration"`)에서는 애초에 실행되지 않는다.

## 관련 문서
- [README.md](README.md) — 사용자 대상 설명 (구조, 도구, 설치법)
- [bugfix.md](bugfix.md) — 개발 중 실제로 겪은 버그와 수정 기록 (계산기 버전 시절 버그 포함, 여전히 유효한 교훈)
