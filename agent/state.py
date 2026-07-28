"""
코드/데이터 작업 에이전트의 공유 상태(State) 정의.

LangGraph의 모든 노드는 이 AgentState를 읽고, 자신이 바꾸고 싶은 필드만
partial dict로 반환한다. LangGraph가 반환값을 기존 상태에 merge하는데,
merge 방식은 필드별로 다르게 지정할 수 있다 (Annotated[..., reducer]).

- messages: add_messages reducer를 사용해 "덮어쓰기"가 아니라 "누적"되도록 한다.
  노드가 새 메시지 하나만 반환해도 이전 대화 기록이 사라지지 않는다.
- iteration: reducer를 지정하지 않았으므로 마지막으로 반환된 값으로 덮어써진다.
  (call_model 노드가 매번 iteration + 1을 반환하는 방식으로 증가시킨다.)
"""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    iteration: int


def merge_state(existing: list[BaseMessage], incoming: list[BaseMessage]) -> list[BaseMessage]:
    """add_messages reducer를 직접 호출하는 헬퍼.

    그래프 밖에서 reducer 동작 자체를 단위 테스트하기 위해 존재한다.
    """
    return add_messages(existing, incoming)
