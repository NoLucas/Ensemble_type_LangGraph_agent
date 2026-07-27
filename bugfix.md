# Bugfix: 튜터 답변에 thinking 서명(signature) blob이 그대로 출력되던 문제

## 증상

`project_coach`(및 다른 튜터)의 답변을 출력하면, 실제 텍스트 답변 대신 아래처럼
`{'signature': '...', 'thinking': '', 'type': 'thinking'}`, `{'text': '...', 'type': 'text'}`
형태의 파이썬 리스트가 그대로 화면에 찍혔다.

```
=== 프로젝트 코치 통합 코멘트 ===
[{'signature': 'ErYGCokBCBAYAipAAE+U...(생략)...', 'thinking': '', 'type': 'thinking'}, {'text': '# 실전 프로젝트 관점에서 본 for문 활용법 ...', 'type': 'text'}]
```

## 원인

`ChatAnthropic`이 `claude-sonnet-5` 모델과 함께 **extended thinking**을 사용할 때,
`response.content`는 더 이상 단순 문자열이 아니라 다음과 같은 **콘텐츠 블록 리스트**로 온다.

```python
[
    {"type": "thinking", "thinking": "...", "signature": "..."},  # 모델의 내부 추론 과정 + 서명
    {"type": "text", "text": "..."},                               # 실제 사용자에게 보여줄 답변
]
```

기존 코드(`java_tutor`, `oop_tutor`, `backend_db_tutor`, `project_coach`)는 모두

```python
return {"java_answer": response.content}
```

처럼 `response.content`를 문자열이라고 가정하고 그대로 상태(state)에 저장했다.
그 결과 `run_interactive`에서 `print(result[field])`를 호출하면 리스트 전체(= thinking 블록의
서명 blob까지 포함)가 그대로 출력되었다.

## 해결

`response.content`에서 `type == "text"`인 블록만 골라 이어붙이는 `extract_text()` 헬퍼를
추가하고, 네 곳의 노드 함수(`java_tutor`, `oop_tutor`, `backend_db_tutor`, `project_coach`)에서
`response.content` 대신 `extract_text(response.content)`를 상태에 저장하도록 수정했다.

```python
def extract_text(content) -> str:
    """
    ChatAnthropic 응답의 content를 순수 텍스트 문자열로 변환한다.
    extended thinking이 켜진 모델은 content가 문자열이 아니라
    [{"type": "thinking", ...}, {"type": "text", "text": "..."}] 형태의
    블록 리스트로 온다. 여기서 thinking 블록(추론 과정, 서명 blob)은 버리고
    text 블록만 이어붙여야 화면에 실제 답변만 출력된다.
    """
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )
```

`content`가 문자열인 경우(= thinking이 꺼져 있거나 다른 모델을 쓰는 경우)도 그대로
지원하도록 `isinstance(content, str)` 분기를 두었다.

## 관련 파일

- `main.py` — `extract_text()` 추가, `java_tutor` / `oop_tutor` / `backend_db_tutor` /
  `project_coach`에서 `response.content` → `extract_text(response.content)`로 변경
