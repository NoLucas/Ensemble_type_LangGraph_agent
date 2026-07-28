# CLAUDE.md

이 파일은 다음 세션(Claude Code)이 이 프로젝트의 현재 작업 상황을 빠르게 파악할 수 있도록 만든 인계 문서입니다.

## 프로젝트 한 줄 요약

LangGraph 기반 코드/데이터 작업 에이전트. TDD(Red-Green-Refactor)로 개발됨. 계산·파일 읽기·파일 쓰기 도구를 병렬 팬아웃으로 처리하고, 최종 보고서는 **같은 과제를 3번 독립 시도한 뒤 도구 실행 결과와 실제로 일치하는 답을 결정론적으로 골라 채택하는 투표(voting) 앙상블** 구조.

## 이 저장소와 자매 저장소

이 프로젝트는 두 버전 중 **투표형 앙상블(ensemble) 버전**입니다.

| | 로컬 경로 | GitHub | 팬인 방식 |
|---|---|---|---|
| **이 저장소** | `C:\Users\Y\Agent\ensemble agent` | [Ensemble_type_LangGraph_agent](https://github.com/NoLucas/Ensemble_type_LangGraph_agent) | 투표형 앙상블 (voter_1/2/3 → 다수결) |
| 자매 저장소 | `C:\Users\Y\Agent\normal agent` | [Code_Simple_LangGraph](https://github.com/NoLucas/Code_Simple_LangGraph) | 단일 reporter (정적 팬아웃/팬인만) |

두 저장소는 원래 같은 git 히스토리에서 갈라져 나왔습니다 (분기점: `a6bcea4` "관점 앙상블을 투표형 앙상블로 교체" 이전). `normal agent`가 나중에 앙상블 단계를 리버트(`c0fd429`)해서 정적 구조로 돌아갔고, 그 이후 `normal agent`에만 추가된 기능(체크포인터, 실LLM 통합 테스트, main.py/app.py/chainlit_app.py 진입점, CI)을 **이 저장소로 수동 이식**했습니다. 즉:
- 그래프 코어(`agent/state.py`, `agent/nodes.py`, `agent/graph.py`)와 테스트는 이 저장소가 투표형 앙상블 기준으로 독자적입니다.
- 진입점 파일(`main.py`, `app.py`, `chainlit_app.py`)과 CI는 `normal agent`에서 이식하면서 이 저장소의 state 스키마(`report_drafts` 필드, `iteration` 델타 방식)에 맞게 수정했습니다.

## 그래프 구조

```
        START
          │
          ▼
      dispatcher                          (첫 입력: 효율적으로 도구 지시)
          │
   tool_call 없음 ──────────────────► END  (도구 불필요 시 dispatcher 답변이 곧 최종 답변)
          │
   tool_call 있음
          │
   ┌──────┼───────────────┬──────────┐
   ▼      ▼               ▼          (도구 팬아웃 — 정적 3-way)
calculate_node   read_file_node  write_file_node
   └──────┼───────────────┴──────────┘
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

- **dispatcher**: 도구를 bind_tools한 llm 사용. 여러 도구를 한 번에 요청하도록 유도.
- **voter_1/2/3**: **모두 같은 프롬프트**(`VOTER_SYSTEM_PROMPT`)로 독립 시도. 관점을 나누지 않는다 — 다양성은 모델 샘플링 자체의 변동성에서 나온다(실서비스에서는 temperature). bind_tools 안 된 llm이라 구조적으로 도구를 재호출 못 함.
- **vote_for_best_report**: `llm` 파라미터가 아예 없는 순수 함수. 도구 실행 결과(`ToolMessage`)를 실제로 포함한 candidate만 통과시켜 다수결로 선택. 동점이면 `voter_1 → voter_2 → voter_3` 순서로 타이브레이크.

## 핵심 설계 결정 (왜 이렇게 했는지)

- **`iteration`은 절대값이 아니라 델타(`Annotated[int, operator.add]`)**: voter 3개가 병렬로 같은 state 스냅샷을 읽고 "현재+1"을 계산하면 `InvalidUpdateError`가 나거나 값이 유실된다(실제로 겪은 버그, `bugfix.md` 참고). 그래서 각 노드가 항상 델타 `1`만 반환하고 리듀서가 합산한다.
- **`vote_for_best_report_node`에 `llm` 파라미터가 없음**: "종합" 단계에 LLM을 또 넣으면 그 단계 자체가 draft에 없던 내용을 지어내는 환각 진입점이 된다. 함수 시그니처 자체에 `llm`이 없어서 구조적으로 그 경로가 존재하지 않는다.
- **`FakeChatModel`(`tests/conftest.py`)에 `threading.Lock`**: voter 3개가 그래프 실행 중 같은 인스턴스를 스레드 풀에서 동시 호출한다. 락 없이는 `calls` 증가/응답 인덱싱이 경합한다(실제로 겪은 버그).
- **사이클 없는 DAG**: dispatcher/voter가 서로 다른 llm 바인딩(tools 유무)을 쓰기 때문에 무한 루프 방지용 반복 상한(`MAX_ITERATIONS`) 같은 장치가 아예 필요 없다.

## 실행 / 테스트

```bash
venv/Scripts/python.exe -m pytest -q        # 48개 테스트 (FakeChatModel 기반, 전부 통과해야 정상)
python main.py                              # 콘솔
streamlit run app.py                        # Streamlit 웹
chainlit run chainlit_app.py                # Chainlit 웹 (도구 실행 과정 시각화)
```

이 폴더는 자체 `venv/`를 가지고 있습니다 (`normal agent`의 venv와 별개로 새로 만듦). 의존성 설치:
```bash
venv/Scripts/python.exe -m pip install -r requirements.txt
```

세 진입점 모두 초기 state에 `"report_drafts": []`를 반드시 포함해야 합니다 (`operator.add` 리듀서 채널이 첫 병렬 쓰기 전에 초기화돼 있어야 함).

## 현재 상태

### 완료됨
- 도구 3개(calculate/read_sandbox_file/write_sandbox_file) + 팬아웃/팬인, 전부 안전장치 테스트 포함
- 투표형 앙상블(voter × 3 + 결정론적 다수결) 완성, 48개 테스트 통과
- 콘솔(`main.py`) / Streamlit(`app.py`) / Chainlit(`chainlit_app.py`) 진입점 연결
- GitHub Actions CI (`.github/workflows/tests.yml`) — push/PR마다 `pytest -m "not integration"`

### 아직 안 된 것 (자매 저장소 `normal agent`에는 있지만 여기엔 이식 안 함)
- **체크포인터(SqliteSaver) 미지원**: `agent/graph.py`의 `build_graph()`가 `checkpointer` 파라미터를 받지 않는다. `normal agent`에는 있음 — 필요하면 그 구현을 참고해서 이식.
- **실LLM 통합 테스트 없음**: `tests/test_integration.py`가 이 저장소엔 없다. `normal agent`의 버전을 참고해서 만들 수 있음 (단, dispatcher/voter 구조에 맞게 iteration 기대값 등을 조정해야 함).
- **실제 API 스모크 테스트 미확인**: `.env`의 `ANTHROPIC_API_KEY`가 비어 있어서 실제 LLM 호출 테스트를 아직 못 해봤다. 실제 키를 넣으면 확인 가능.
- **투표 로직이 단순 문자열 포함 검사**: `vote_for_best_report_node`가 숫자 표현 차이(`"4"` vs `"4.0"` vs `"네 개"`)에는 관대하지 않다.

## 관련 문서
- [README.md](README.md) — 사용자 대상 설명 (구조, 도구, 설치법)
- [bugfix.md](bugfix.md) — 개발 중 실제로 겪은 버그와 수정 기록
