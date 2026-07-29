# Bugfix: 같은 `AIMessage` 객체를 재사용하면 반복 상한 테스트가 조기 종료되던 문제

## 증상

`route_after_dispatcher`(구 `route_after_model`)가 `MAX_ITERATIONS`에서 정확히 멈추는지
검증하는 테스트에서, 모델이 항상 `tool_calls`를 반환하도록 응답 리스트를 준비했는데도
그래프가 `MAX_ITERATIONS`(10)이 아니라 2번 만에 멈췄다.

```python
always_tool_call = AIMessage(
    content="", tool_calls=[{"name": "calculate", ...}],
)
llm = FakeChatModel([always_tool_call] * (MAX_ITERATIONS + 5))  # 같은 객체를 재사용
...
# 기대: iteration == MAX_ITERATIONS(10)
# 실제: iteration == 2
```

## 원인

LangGraph의 `add_messages` reducer는 메시지를 리스트 끝에 무조건 추가하지 않고,
**같은 `id`를 가진 메시지가 이미 있으면 그 자리에서 업데이트(치환)**한다. `AIMessage()`는
인스턴스를 생성하는 순간 고유한 `id`가 자동 부여되는데, 위 코드는 `[always_tool_call] * N`으로
**같은 객체(=같은 id)를 N번 재사용**했다.

그 결과 두 번째 model 호출이 반환한 응답이 messages 리스트 **끝에 추가되지 않고**,
첫 번째 응답이 있던 자리(리스트 중간)에서 치환되어버렸다. 그래서 리스트의 실제 마지막
메시지는 그사이 실행된 `ToolMessage`가 되었고, `route_after_dispatcher`가
`state["messages"][-1]`을 봤을 때 `tool_calls`가 없는 것으로 오판해 그래프가 조기 종료됐다.

디버깅은 `graph.stream(..., stream_mode="updates")`로 각 스텝의 상태를 직접 찍어보고,
`AIMessage`의 `id` 필드가 두 번의 model 호출에서 완전히 동일하다는 것을 확인해서 원인을
좁혔다.

## 해결

테스트에서 스크립트 응답을 준비할 때 **매번 새 인스턴스**를 만들도록 고쳤다.

```python
def make_tool_call_response(i: int) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "calculate", "args": {"expression": "1+1"}, "id": f"call_{i}"}],
    )

llm = fake_llm_factory([make_tool_call_response(i) for i in range(MAX_ITERATIONS + 5)])
```

그래프/노드 로직 자체에는 버그가 없었다 — `FakeChatModel`을 쓰는 테스트 픽스처의 함정이었다.
같은 이유로, 앞으로 `AIMessage`/`ToolMessage` 등을 반복적으로 만들 때는 리스트 컴프리헨션 등으로
**항상 새 인스턴스를 생성**해야 한다는 규칙을 팀 컨벤션으로 남겨둔다.

## 관련 파일

- `tests/test_graph_e2e.py` — 반복 상한 테스트의 응답 준비 로직 수정

---

# Bugfix: 병렬 draft 노드 3개가 `iteration`을 동시에 갱신하며 `InvalidUpdateError` 발생

## 증상

reporter를 3-way 앙상블(`draft_concise`/`draft_detailed`/`draft_action`)로 바꾼 뒤 그래프를
처음 실행하자 다음 예외가 발생했다.

```
langgraph.errors.InvalidUpdateError: At key 'iteration': Can receive only one value per step.
Use an Annotated key to handle multiple values.
```

## 원인

기존 `iteration` 필드는 리듀서가 없는 평범한 `int`였고, 각 노드가
`state.get("iteration", 0) + 1`처럼 **현재 상태값을 읽어 절대값을 계산**해서 반환했다.
이 방식은 노드가 한 번에 하나씩만 실행될 때는 문제가 없었지만, 3개의 draft 노드가
**같은 슈퍼스텝에서 병렬로 실행**되면서 상황이 달라졌다.

세 노드 모두 실행 시작 시점에 같은 `iteration` 값(예: 1)을 읽어 각자 "현재+1"(=2)을
계산해 반환했다. 리듀서가 없는 채널은 한 스텝에 값이 하나만 들어와야 하는데, 세 노드가
동시에 값을 쓰려고 하니 LangGraph가 이를 감지해 `InvalidUpdateError`를 던졌다.

## 해결

`iteration`을 절대값이 아니라 **델타(항상 1)**를 반환하도록 바꾸고, `state.py`에서
`Annotated[int, operator.add]` 리듀서를 지정했다.

```python
# state.py
class AgentState(TypedDict):
    ...
    iteration: Annotated[int, operator.add]

# nodes.py — 절대값(state.get("iteration", 0) + 1) 대신 델타만 반환
return {
    "report_drafts": [{"label": label, "text": text}],
    "iteration": 1,
}
```

이렇게 하면 병렬로 실행되는 노드가 몇 개든, 각자 자신의 "호출 1회"만 보고하고
리듀서가 이를 모두 더해준다. 실행 순서나 동시성과 무관하게 "총 LLM 호출 수"가 정확히
합산된다. `report_drafts` 필드도 같은 이유로 처음부터 `Annotated[list[dict], operator.add]`로
설계해서 병렬 쓰기 충돌을 피했다.

## 관련 파일

- `agent/state.py` — `iteration`을 `Annotated[int, operator.add]`로 변경
- `agent/nodes.py` — `call_dispatcher_model`/`call_report_draft_model`이 절대값 대신
  델타(1)를 반환하도록 수정
- `tests/test_nodes.py` — 관련 단위 테스트를 델타 기준으로 갱신

---

# Bugfix 아님 — GitHub 도구 전환 시 미리 방어한 실패 케이스

계산/파일 도구를 `fetch_repo_overview`/`fetch_repo_source_sample`(GitHub REST API)로
교체하면서, 실제로 겪은 버그는 아니지만 **외부 API를 호출하는 도구라서 반드시 처음부터
막아둬야 했던 실패 케이스**를 정리해둔다. 이전 도구(calculate/read/write)는 실패 원인이
전부 로컬(잘못된 표현식, 경로 탈출)이었지만, 이번 도구는 실패 원인이 로컬(잘못된 repo
형식)과 원격(GitHub 쪽 상태)에 걸쳐 있어서 각각 다른 방어가 필요했다.

## 1. `owner/repo` 형식이 아닌 입력

LLM이 저장소를 `https://github.com/owner/repo` 전체 URL이나 `owner`만, 또는 완전히
관련 없는 문자열로 넘길 수 있다. 이 경우 GitHub API에 요청 자체를 보내지 않고
`_validate_repo()`(정규식 `^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$`)에서 즉시
`"Error: 저장소는 'owner/repo' 형식이어야 합니다"`로 걸러낸다. 불필요한 API 호출(=
비인증 rate limit 소모)을 막는 것도 이 검증의 목적이다.

## 2. 저장소가 존재하지 않음 (404)

오타가 있는 저장소명이나 삭제된 저장소를 리뷰해달라고 하면 `/repos/{repo}`가 404를
반환한다. `_fetch_repo_metadata()`가 상태 코드를 보고 `f"Error: 저장소를 찾을 수
없습니다 ({repo})."`를 반환하며, 이후 README/트리/파일 조회로 넘어가지 않는다(연쇄
실패를 막기 위해 초반에 조기 반환).

## 3. 비인증 GitHub API rate limit 초과 (403)

의도적으로 비인증 호출을 선택했기 때문에(시간당 60회), 저장소를 몇 개만 연속으로
리뷰해도 한도를 넘기기 쉽다. 403을 다른 종류의 실패(404, 5xx)와 구분해서
`"Error: GitHub API 요청 한도를 초과했습니다 (비인증 시 시간당 60회)."`로 원인을
명시했다 — 단순히 "요청 실패"라고만 하면 사용자가 저장소명이 잘못됐다고 오해하기
쉽기 때문이다.

## 4. README가 없는 저장소

메타데이터 조회는 성공했지만 `/repos/{repo}/readme`가 404인 경우(README 파일이 없는
저장소는 흔하다), 전체 개요 조회를 실패로 처리하지 않고 `(README 없음)` 문구만 붙여서
나머지 메타데이터(설명/언어/stars 등)는 그대로 반환한다. README 하나가 없다고 저장소
개요 전체를 못 가져오는 것은 과도한 실패 처리라고 판단했다.

## 5. 코드 리뷰 대상 소스 파일이 하나도 없는 저장소

문서 전용 저장소, 데이터셋 저장소, 또는 확장자가 화이트리스트(`_SOURCE_EXTENSIONS`)에
없는 언어로만 이루어진 저장소는 트리 조회는 성공해도 `candidates`가 빈 리스트가 된다.
이 경우 `Error`가 아니라 `f"({repo}에서 리뷰할 소스 파일을 찾지 못했습니다.)"`를
반환한다 — "실패"가 아니라 "리뷰할 코드가 없다는 사실 자체가 결과"이므로,
`vote_for_best_report_node`가 이 문자열을 정답(ToolMessage)으로 놓고 다수결을 매길
때도 동일하게 취급된다.

## 6. 네트워크 오류(타임아웃/연결 실패)

`requests.get()`이 `requests.RequestException`을 던질 수 있으므로(DNS 실패, 타임아웃
등), 모든 호출을 개별 `try/except`로 감싸 `f"Error: ... 실패했습니다 ({exc})"` 형태로
변환한다. 특히 `fetch_repo_source_sample_text`는 선택된 파일 여러 개를 순차적으로
가져오는데, 파일 하나가 네트워크 오류로 실패해도 나머지 파일 조회를 계속 진행하고
해당 파일 섹션에만 실패 메시지를 남긴다(전체를 포기하지 않는다).

이 다섯/여섯 케이스 모두 `tests/test_tools.py`에 `mock_github_get`(conftest.py의
`QueuedGet`)으로 각각 대응하는 테스트가 있다. 실제 GitHub API를 타지 않고도 상태
코드만 조작해서 결정적으로 재현할 수 있다.

## 관련 파일

- `agent/tools.py` — `_validate_repo`, `_fetch_repo_metadata`,
  `fetch_repo_overview_text`, `fetch_repo_source_sample_text`
- `tests/test_tools.py` — 케이스별 정상/실패 테스트
- `tests/conftest.py` — `FakeResponse`/`QueuedGet` (requests.get 모킹)

---

# Bugfix: `examples/` 폴더의 데모 스크립트가 실제 라이브러리 구현보다 먼저 뽑히던 문제

## 증상

`fetch_repo_source_sample_text`의 파일 선택 로직을 "경로 깊이만" 보던 방식에서
"주 언어 일치 + 진입점 파일명 + 깊이 + 크기" 점수 합산 방식으로 개선한 뒤,
`langgraph`/`requests` 같은 저장소로는 잘 동작했지만 `expressjs/express`로 수동
검증하자 결과가 이상했다.

```
### index.js
### examples/auth/index.js
### examples/content-negotiation/index.js
```

세 번째 항목까지 전부 데모/예제 코드였고, 정작 라이브러리의 실제 구현(`lib/`)은
하나도 뽑히지 않았다.

## 원인

Node.js/npm 생태계는 관례적으로 거의 모든 디렉토리에 `index.js`를 둔다
(모듈 해석 규칙 때문에). 진입점 파일명에 주는 가산점(+5)이 경로 깊이 페널티
(레벨당 -1)보다 커서, `examples/auth/index.js`(깊이 2, 점수 10+5-2+2=15)가
`lib/application.js`(진입점 아님, 깊이 1, 점수 10+0-1+2=11)를 점수로 이겨버렸다.
"진입점처럼 보이는 파일명"이라는 신호가 예제 코드에는 아무 의미가 없는데도
똑같이 가산점을 준 것이 근본 원인이다.

## 해결

`_SKIP_PATH_PARTS`(테스트/벤더/빌드 산출물 제외 목록)에 `example`/`examples`/
`demo`/`demos`/`sample`/`samples`를 추가해서, 점수를 매기기도 전에 후보에서
아예 제외했다. 진입점 가산점의 가중치를 조정하는 대신 "애초에 후보가 아니다"로
접근한 이유는, `tests/`를 제외하는 것과 동일한 종류의 판단(저장소의 "진짜 코드"가
아니다)이라 기존 필터와 일관되고, 다른 언어의 예제 디렉토리(`examples/`,
`sample/` 등 명명 관례가 JS 생태계 밖에서도 흔하다)에도 똑같이 적용되기 때문이다.

```python
_SKIP_PATH_PARTS = {
    "test", "tests", "vendor", "node_modules", "dist", "build", ".github",
    "example", "examples", "demo", "demos", "sample", "samples",
}
```

수정 후 재검증:

```
### index.js
### lib/application.js
### lib/express.js
```

## 교훈

`tests/test_tools.py`의 모킹된 단위 테스트는 이 문제를 잡지 못했다 — 테스트
픽스처가 애초에 "그럴듯한" 파일 몇 개만 만들어뒀지, 실제 저장소처럼 같은
파일명이 여러 디렉토리에 반복되는 상황을 재현하지 않았기 때문이다. 휴리스틱/
스코어링 로직을 짤 때는 실제 유명 오픈소스 저장소 몇 개(언어를 다양하게)로
직접 돌려보고 결과를 눈으로 확인하는 과정이 유닛 테스트만으로는 못 잡는
"각 신호가 실제로 어떻게 상호작용하는지"를 드러낸다.

## 관련 파일

- `agent/tools.py` — `_SKIP_PATH_PARTS`에 예제/데모 경로 추가
- `tests/test_tools.py` — `test_fetch_repo_source_sample_text_excludes_examples_directory`
  회귀 테스트 추가

---

# Bugfix: 대형 저장소에서 GitHub 응답이 너무 커서 리뷰가 사실상 불가능하던 문제

## 증상

사용자가 "GitHub 용량이 너무 커서 리뷰가 불가능하다"고 지적했다. API 키 없이
(GitHub API만으로) 재현해보니 실제로 심각했다.

```python
r = requests.get('https://api.github.com/repos/torvalds/linux/git/trees/master',
                  params={'recursive': '1'})
# truncated: True, tree 항목 71,798개, 응답 17.6MB
r = requests.get('https://api.github.com/repos/kubernetes/kubernetes/git/trees/master',
                  params={'recursive': '1'})
# truncated: False, tree 항목 37,390개, 응답 10.3MB
```

`fetch_repo_source_sample_text`가 파일 3개만 골라 쓰려고 저장소 **전체** 파일
목록을 `/git/trees/{branch}?recursive=1`로 한 번에 받고 있었던 것이 원인이었다.
linux 커널 같은 극단적인 경우는 GitHub이 아예 응답을 잘라버려서(`truncated:
true`) 일부 파일은 애초에 후보에 들어오지도 못했다.

## 원인

recursive=1 트리 API는 "저장소 전체를 한 번에 훑고 싶을 때" 쓰라고 있는
엔드포인트인데, 우리 용도는 정반대(파일 몇 개만 대표로 보면 됨)였다. 요청
자체는 1번이라 GitHub API 호출 횟수(rate limit) 관점에서는 저렴해 보이지만,
응답 크기·다운로드 시간·JSON 파싱 비용은 저장소 크기에 비례해서 커진다.

## 해결

디렉토리를 하나씩(비재귀) 나열하는 `/repos/{repo}/contents/{path}`로 바꾸고,
**깊이 우선(DFS)** + 힌트 디렉토리(`src`/`lib`/`pkg`/`cmd`/...) 우선 탐색으로
필요한 만큼만 내려간다(`_discover_source_candidates()`). 호출 횟수
(`_MAX_DIR_LISTING_CALLS`, 8회)나 모인 후보 수(15개)가 충분해지면 멈춘다.

처음엔 너비 우선(BFS)으로 만들었는데, `kubernetes/kubernetes`로 검증하자마자
빈 결과가 나왔다. 원인을 보니: 그 저장소는 루트 바로 아래에 `cmd`/`pkg`
같은 컴포넌트 디렉토리가 20개 넘게 있고, 그 안에도 또 컴포넌트별 하위
디렉토리가 잔뜩 있어서 실제 `.go` 코드는 3~4단계는 더 들어가야 나온다. BFS는
"같은 깊이의 모든 디렉토리를 다 훑은 뒤에야 한 단계 더 내려가는" 방식이라,
호출 예산(당시 5회)을 얕은 형제 디렉토리들을 옆으로 훑는 데 다 써버리고 단
하나의 파일도 못 찾았다. DFS로 바꾸고(힌트 디렉토리를 스택에 나중에 넣어서
먼저 꺼내지게 함) 예산을 8회로 늘리자 `pkg/volume/*.go` 같은 실제 구현
파일을 찾아냈다.

```
# 이전 (recursive=1): kubernetes/kubernetes에서 10MB 다운로드 후에도 무거움
# 이후 (BFS 디렉토리 나열, 5회 예산): "리뷰할 소스 파일을 찾지 못했습니다"
# 이후 (DFS + 힌트 우선, 8회 예산): 3.8KB, 2.8초
### pkg/volume/doc.go
### pkg/volume/metrics_block.go
### pkg/volume/metrics_cached.go
```

이 과정에서 파생 문제도 하나 발견했다: Go는 `tests/` 같은 디렉토리 관례
대신 같은 디렉토리에 `foo.go`/`foo_test.go`를 나란히 두는 명명 관례를
쓰는데, 기존 `_SKIP_PATH_PARTS`(디렉토리명 기반)로는 `metrics_block_linux_test.go`
같은 파일이 걸러지지 않았다. `_looks_like_test_file()`을 추가해서 파일명
접미사(`_test`/`.test`/`.spec`/`test_` 접두사)로도 걸러내도록 고쳤다.

## 교훈

- API 호출 "횟수"만 아끼면 안 된다 — 응답 "크기"도 비용이다. recursive=1은
  호출 1번이라 저렴해 보였지만 실제로는 리뷰 목적에 안 맞는 과도한 데이터였다.
- 휴리스틱(BFS든 DFS든)을 코드만 보고 옳다고 판단하지 말고, 실제로 구조가
  극단적인 저장소(이번엔 kubernetes/kubernetes)로 돌려봐야 문제가 드러난다.
  작은 테스트 픽스처는 "얕고 좁은" 구조만 만들 뿐, "얕고 넓다가 깊은 곳에
  코드가 몰린" 실제 모노레포 구조를 우연히라도 재현하기 어렵다.

## 관련 파일

- `agent/tools.py` — `_list_directory`, `_discover_source_candidates`,
  `_looks_like_test_file`, `fetch_repo_source_sample_text` 재작성
- `tests/test_tools.py` — 재귀 트리 엔드포인트 미사용 회귀, 호출 횟수 상한,
  하위 디렉토리 실패 허용, 깊이 우선 탐색(얕은 형제보다 깊은 코드 우선),
  테스트 파일명 제외 등 회귀 테스트 추가
