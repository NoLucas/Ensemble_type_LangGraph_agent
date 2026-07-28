"""
코드/데이터 작업 에이전트의 공유 상태(State) 정의.

LangGraph의 모든 노드는 이 AgentState를 읽고, 자신이 바꾸고 싶은 필드만
partial dict로 반환한다. LangGraph가 반환값을 기존 상태에 merge하는데,
merge 방식은 필드별로 다르게 지정할 수 있다 (Annotated[..., reducer]).

- messages: add_messages reducer를 사용해 "덮어쓰기"가 아니라 "누적"되도록 한다.
  노드가 새 메시지 하나만 반환해도 이전 대화 기록이 사라지지 않는다.
- iteration: operator.add reducer로 "누적"된다. dispatcher와 draft 3개는
  모두 LLM을 한 번씩 호출할 때마다 절대값(현재값+1)이 아니라 델타(항상 1)를
  반환한다. 절대값 방식을 쓰면 병렬로 실행되는 draft 3개가 같은 시점의
  state를 읽어 똑같이 "현재값+1"을 계산해버려서 충돌하거나(리듀서 없이는
  InvalidUpdateError), 값이 갱신되지 않는다. 델타를 더하는 방식이어야
  "LLM 호출이 총 몇 번 일어났는지"가 병렬 실행 순서와 무관하게 정확히
  합산된다.
- report_drafts: operator.add reducer로 "누적"된다. 3개의 voter 노드가
  같은 과제를 독립적으로 병렬 시도해 각자 하나씩 {"label": ..., "text": ...}
  항목을 반환하는데, 리스트를 누적하는 reducer가 없으면 나중에 실행된
  노드의 결과가 앞선 노드의 결과를 덮어써서 2개가 사라지는 버그가 생긴다.
  도착 순서(=병렬 실행이 끝난 순서)는 보장되지 않으므로, 이 리스트를
  소비하는 vote_for_best_report_node는 label로 candidate를 찾아 도구
  실행 결과와의 일치 여부로 다수결 판정해야 한다.
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    iteration: Annotated[int, operator.add]
    report_drafts: Annotated[list[dict], operator.add]


def merge_state(existing: list[BaseMessage], incoming: list[BaseMessage]) -> list[BaseMessage]:
    """add_messages reducer를 직접 호출하는 헬퍼.

    그래프 밖에서 reducer 동작 자체를 단위 테스트하기 위해 존재한다.
    """
    return add_messages(existing, incoming)
