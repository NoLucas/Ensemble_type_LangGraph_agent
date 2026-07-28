# Code_Simple_LangGraph

LangGraph 기반 코드/데이터 작업 에이전트입니다. TDD(Red-Green-Refactor)로 개발되었으며, 계산·파일 읽기·파일 쓰기 도구를 병렬로 처리하고, 최종 보고서는 동일한 과제를 3번 독립 시도한 뒤 도구 실행 결과와 실제로 일치하는 답을 결정론적으로 골라 채택하는 투표(voting) 앙상블 구조입니다.

## 구조

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

- **dispatcher**: 사용자 입력을 해석해 `calculate`(계산) / `read_sandbox_file`(파일 읽기) / `write_sandbox_file`(파일 쓰기) 중 필요한 도구를 지시합니다. 여러 도구가 필요하면 한 번의 응답에서 모두 요청하도록 유도해 팬아웃이 실제로 병렬 이득을 보게 합니다.
- **도구 팬아웃/팬인**: 세 도구 노드는 항상 함께 깨워지고, 자기 담당 tool_call이 없으면 조용히 통과(pass-through)합니다.
- **투표형 앙상블 팬아웃/팬인**: `voter_1`/`voter_2`/`voter_3`는 **모두 같은 프롬프트(`VOTER_SYSTEM_PROMPT`)**로 최종 보고서를 각자 독립적으로 시도합니다. 관점을 나누는 앙상블이 아니라 같은 과제를 여러 번 독립 시도해서 검증하는 투표 앙상블이며, 다양성은 프롬프트가 아니라 모델 샘플링 자체의 변동성(temperature)에서 나옵니다.
- **vote_for_best_report**: 세 candidate 중 도구 실행 결과(`ToolMessage`)를 실제로 포함한 것만 통과시키고, **LLM을 다시 호출하지 않고** 문자열 일치 개수로 결정론적으로 승자를 고르는 함수입니다. 동점이면 `voter_1 → voter_2 → voter_3` 순서로 타이브레이크하고, 아무도 사실을 못 맞혀도 예외 없이 첫 번째 voter의 답을 반환합니다. "어느 게 더 그럴듯한가"를 LLM 판사에게 다시 묻지 않으므로, 이 종합 단계에는 환각이 새로 끼어들 여지가 없습니다.
- **도구가 필요 없으면** dispatcher의 답변이 그대로 최종 답변이 되고, 도구 팬아웃/투표 팬아웃 단계는 아예 실행되지 않습니다 (불필요한 LLM 호출을 만들지 않습니다).

이 그래프는 사이클이 없는 DAG입니다 — `dispatcher`는 tool이 바인딩된 llm을, `voter_*` 노드들은 바인딩되지 않은 llm을 받아서 구조적으로 도구를 다시 호출할 수 없습니다. 그래서 반복 상한(iteration cap) 같은 무한 루프 방지 장치가 필요 없습니다.

## 파일 구성

```
agent/
  state.py    # AgentState: messages(add_messages), iteration(operator.add 델타), report_drafts(operator.add)
  tools.py    # calculate, read_text_file, write_text_file (+ 샌드박스 경로 탈출 방지)
  nodes.py    # dispatcher/voter/vote_for_best_report 노드, 라우팅 함수, 도구 실행 노드
  graph.py    # build_graph(llm, sandbox_dir) — 전체 그래프 조립
  sandbox_data/  # read/write 도구가 접근 가능한 샌드박스 디렉토리
tests/
  conftest.py         # FakeChatModel — 실제 API 호출 없이 결정적 테스트
  test_state.py        # state reducer 테스트
  test_tools.py         # 도구 단위 테스트 (정상/거부 케이스)
  test_nodes.py          # 노드 단위 테스트
  test_routing.py         # 조건부 라우팅 테스트
  test_graph_e2e.py        # 그래프 통합 테스트
```

## 도구

| 도구 | 설명 | 안전장치 |
|---|---|---|
| `calculate` | 사칙연산 + 괄호 산술 계산 | `eval` 없이 `ast` 화이트리스트로만 평가 |
| `read_sandbox_file` | 샌드박스 안 텍스트 파일 읽기 | `resolve()` 후 base_dir 하위인지 검사, 경로 탈출 차단 |
| `write_sandbox_file` | 샌드박스 안 텍스트 파일 쓰기/덮어쓰기 | 동일한 경계 검사로 중첩 경로 탈출도 차단 |

모든 도구는 예외를 던지지 않고 `"Error: ..."` 문자열을 반환합니다 — LLM이 스키마와 맞지 않는 입력을 줄 수 있으므로, 실패도 정상적인 반환값으로 표현해서 그래프 전체가 죽지 않게 합니다.

## 설치 및 테스트

```bash
python -m venv venv
venv/Scripts/activate   # Windows
pip install -r requirements.txt
pytest
```

느린 테스트(실LLM 호출)는 `integration` 마커로 분리되어 있으며 기본 실행에서 제외하는 것을 권장합니다:

```bash
pytest -m "not integration"
```

## 그래프 사용 예시

```python
from langchain_anthropic import ChatAnthropic
from agent.graph import build_graph

llm = ChatAnthropic(model="claude-sonnet-5")
graph = build_graph(llm)

result = graph.invoke({
    "messages": [("human", "2+2 계산하고 sales.txt 내용도 읽어줘")],
    "iteration": 0,
    "report_drafts": [],
})
print(result["messages"][-1].content)
```

`ANTHROPIC_API_KEY`는 `.env.example`을 참고해 `.env` 파일로 설정합니다.

> 현재 이 저장소는 그래프 코어(`agent/`)와 테스트만 포함하며, 콘솔/웹 실행 진입점(예: `main.py`, `app.py`)은 아직 별도로 연결되어 있지 않습니다.
