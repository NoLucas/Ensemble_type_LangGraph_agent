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
        # 3-way 투표형 앙상블(voter_1/voter_2/voter_3)이 같은
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


class FakeResponse:
    """requests.Response를 흉내 내는 최소한의 더미(status_code/json/text만)."""

    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data if json_data is not None else {}
        self.text = text

    def json(self):
        return self._json_data


class QueuedGet:
    """requests.get을 대체하는 더미. 호출될 때마다 큐에서 응답을 하나씩 꺼내
    반환한다. GitHub 도구가 여러 번(메타데이터 -> README/트리 -> 파일 내용)
    순차적으로 requests.get을 호출하므로, URL 매칭 대신 호출 순서로 응답을
    미리 정해두는 편이 테스트를 단순하게 만든다.

    도구 노드 하나가 순차적으로 호출하는 경우에만 안전하다 — 서로 다른 두
    도구 노드가 그래프의 같은 팬아웃 라운드에서 병렬로(스레드 풀) 각자
    여러 번 호출하면 두 노드의 호출 순서가 뒤섞이므로, 그런 시나리오는
    PathAwareQueuedGet(mock_github_get_by_path)을 대신 써야 한다. 락은
    "두 스레드가 같은 응답을 훔쳐가는" 경합만 막아줄 뿐, 뒤섞인 순서
    자체는 못 막는다 — FakeChatModel과 동일한 이유(voter 3개 동시 호출)로
    필요하다.
    """

    def __init__(self, responses: list):
        self._responses = list(responses)
        self.calls: list[tuple] = []
        self._lock = threading.Lock()

    def __call__(self, url, **kwargs):
        with self._lock:
            self.calls.append((url, kwargs))
            if not self._responses:
                raise AssertionError("QueuedGet: 예정된 응답을 모두 소진했습니다.")
            return self._responses.pop(0)


@pytest.fixture
def mock_github_get(monkeypatch):
    """agent.tools.requests.get을 QueuedGet으로 교체하는 팩토리 픽스처."""
    import agent.tools as tools_module

    def _install(responses: list) -> QueuedGet:
        fake = QueuedGet(responses)
        monkeypatch.setattr(tools_module.requests, "get", fake)
        return fake

    return _install


class PathAwareQueuedGet:
    """requests.get을 대체하는 더미. 호출 순서가 아니라 URL에 포함된 부분
    문자열로 어느 큐에서 응답을 꺼낼지 정한다.

    QueuedGet(순수 FIFO)은 도구 하나가 순차적으로 여러 번 requests.get을
    호출하는 경우엔 충분하지만, 서로 다른 두 도구 노드(예: repo_structure_node
    + repo_file_node)가 그래프의 같은 팬아웃 라운드에서 병렬로(스레드 풀)
    각자 여러 번 호출하면, 두 노드의 호출이 뒤섞여 들어와 순서를 신뢰할 수
    없다 — 그래서 "어떤 URL이냐"로 응답을 골라내는 이 클래스가 필요하다.
    """

    def __init__(self, response_map: dict[str, list]):
        self._queues = {key: list(values) for key, values in response_map.items()}
        self._lock = threading.Lock()
        self.calls: list[tuple] = []

    def __call__(self, url, **kwargs):
        with self._lock:
            self.calls.append((url, kwargs))
            for key, queue in self._queues.items():
                if key in url and queue:
                    return queue.pop(0)
            raise AssertionError(f"PathAwareQueuedGet: {url}에 대해 예정된 응답이 없습니다.")


@pytest.fixture
def mock_github_get_by_path(monkeypatch):
    """agent.tools.requests.get을 PathAwareQueuedGet으로 교체하는 팩토리 픽스처.

    response_map은 {URL에 포함될 부분 문자열: [FakeResponse, ...]} 형태다.
    """
    import agent.tools as tools_module

    def _install(response_map: dict[str, list]) -> PathAwareQueuedGet:
        fake = PathAwareQueuedGet(response_map)
        monkeypatch.setattr(tools_module.requests, "get", fake)
        return fake

    return _install
