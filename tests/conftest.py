"""
공용 테스트 픽스처.

FakeChatModel은 실제 LLM API를 호출하지 않고, 미리 정해진 AIMessage 응답을
순서대로 반환하는 더미 모델이다. 노드/그래프 테스트가 결정적(deterministic)이고
빠르게 돌아가도록 하기 위한 것으로, 실제 llm.invoke(messages) 인터페이스와
동일한 모양(.invoke, .bind_tools)만 흉내 낸다.
"""

import threading

import pytest
from langchain_core.messages import AIMessage


class FakeChatModel:
    def __init__(self, responses: list[AIMessage]):
        self._responses = list(responses)
        self.calls = 0
        # invoke()에 매번 어떤 messages가 들어왔는지 기록해서
        # 테스트에서 "모델이 무엇을 봤는지" 검증할 수 있게 한다.
        self.received_messages: list[list] = []
        # 3-way 보고서 앙상블(draft_concise/detailed/action)이 같은
        # FakeChatModel 인스턴스를 스레드 풀에서 동시에 호출하므로, calls
        # 증가와 _responses 인덱싱을 락으로 보호하지 않으면 두 스레드가
        # 같은 인덱스를 읽거나 응답을 건너뛰는 경합이 생길 수 있다.
        self._lock = threading.Lock()

    def bind_tools(self, tools):
        # 실제 ChatModel과 동일한 체이닝 형태(llm.bind_tools(tools))를 지원하되,
        # 테스트에서는 도구 스키마를 실제로 LLM에 전달할 필요가 없으므로 self를 반환한다.
        return self

    def invoke(self, messages):
        with self._lock:
            self.received_messages.append(messages)
            if self.calls >= len(self._responses):
                raise AssertionError(
                    f"FakeChatModel: 예정된 응답 {len(self._responses)}개를 모두 "
                    f"소진했는데 {self.calls + 1}번째 호출이 발생했습니다."
                )
            response = self._responses[self.calls]
            self.calls += 1
            return response


@pytest.fixture
def fake_llm_factory():
    """responses 리스트를 받아 FakeChatModel을 만드는 팩토리 픽스처."""

    def _make(responses: list[AIMessage]) -> FakeChatModel:
        return FakeChatModel(responses)

    return _make


@pytest.fixture
def sandbox_dir(tmp_path):
    """read_text_file 도구가 접근할 수 있는 격리된 샌드박스 디렉토리."""
    d = tmp_path / "sandbox"
    d.mkdir()
    return d
