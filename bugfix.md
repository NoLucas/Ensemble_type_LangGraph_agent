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
