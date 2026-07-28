"""
도구(tools) 단위 테스트.

GitHub API 호출은 실제 네트워크를 타지 않고 conftest.py의 mock_github_get으로
requests.get을 큐 기반 더미로 교체해서 검증한다. 각 도구는 (1) 정상 케이스와
(2) 실패 케이스(저장소 없음/rate limit/네트워크 오류)를 반드시 가진다 —
도구는 에이전트 그래프 안에서 호출되므로, 여기서 예외가 새면 그래프 전체가
죽는다.
"""

import requests

from agent.tools import (
    fetch_repo_overview,
    fetch_repo_overview_text,
    fetch_repo_source_sample,
    fetch_repo_source_sample_text,
)
from tests.conftest import FakeResponse

# ---------------------------------------------------------------------------
# 공통: repo 형식 검증
# ---------------------------------------------------------------------------


def test_fetch_repo_overview_text_rejects_invalid_repo_format(mock_github_get):
    fake = mock_github_get([])  # requests.get이 아예 호출되면 안 된다

    result = fetch_repo_overview_text("not-a-valid-repo")

    assert result.startswith("Error")
    assert fake.calls == []


def test_fetch_repo_source_sample_text_rejects_invalid_repo_format(mock_github_get):
    fake = mock_github_get([])

    result = fetch_repo_source_sample_text("just-owner")

    assert result.startswith("Error")
    assert fake.calls == []


# ---------------------------------------------------------------------------
# fetch_repo_overview_text: 메타데이터 + README 발췌
# ---------------------------------------------------------------------------


def test_fetch_repo_overview_text_success(mock_github_get):
    mock_github_get(
        [
            FakeResponse(
                200,
                json_data={
                    "full_name": "octocat/hello-world",
                    "description": "테스트용 저장소",
                    "language": "Python",
                    "stargazers_count": 42,
                    "forks_count": 7,
                    "topics": ["demo", "test"],
                },
            ),
            FakeResponse(200, text="# Hello World\n환영합니다."),
        ]
    )

    result = fetch_repo_overview_text("octocat/hello-world")

    assert "octocat/hello-world" in result
    assert "테스트용 저장소" in result
    assert "Python" in result
    assert "42" in result
    assert "Hello World" in result


def test_fetch_repo_overview_text_repo_not_found(mock_github_get):
    mock_github_get([FakeResponse(404)])

    result = fetch_repo_overview_text("octocat/does-not-exist")

    assert result.startswith("Error")
    assert "찾을 수 없습니다" in result


def test_fetch_repo_overview_text_rate_limited(mock_github_get):
    mock_github_get([FakeResponse(403)])

    result = fetch_repo_overview_text("octocat/hello-world")

    assert result.startswith("Error")
    assert "한도" in result


def test_fetch_repo_overview_text_network_error(monkeypatch):
    import agent.tools as tools_module

    def _raise(*args, **kwargs):
        raise requests.RequestException("connection failed")

    monkeypatch.setattr(tools_module.requests, "get", _raise)

    result = fetch_repo_overview_text("octocat/hello-world")

    assert result.startswith("Error")


def test_fetch_repo_overview_text_missing_readme_still_returns_metadata(mock_github_get):
    mock_github_get(
        [
            FakeResponse(200, json_data={"full_name": "octocat/hello-world"}),
            FakeResponse(404),
        ]
    )

    result = fetch_repo_overview_text("octocat/hello-world")

    assert not result.startswith("Error")
    assert "octocat/hello-world" in result
    assert "README 없음" in result


# ---------------------------------------------------------------------------
# fetch_repo_source_sample_text: 대표 소스 파일 발췌
# ---------------------------------------------------------------------------


def test_fetch_repo_source_sample_text_success(mock_github_get):
    mock_github_get(
        [
            FakeResponse(200, json_data={"default_branch": "main"}),
            FakeResponse(
                200,
                json_data={
                    "tree": [
                        {"path": "main.py", "type": "blob"},
                        {"path": "README.md", "type": "blob"},
                        {"path": "tests/test_main.py", "type": "blob"},
                        {"path": "src", "type": "tree"},
                    ]
                },
            ),
            FakeResponse(200, text="print('hello')"),
        ]
    )

    result = fetch_repo_source_sample_text("octocat/hello-world")

    assert "main.py" in result
    assert "print('hello')" in result
    # README.md는 소스 확장자가 아니고, tests/ 하위는 스킵 대상이라 후보에서 빠진다.
    assert "README.md" not in result
    assert "test_main.py" not in result


def test_fetch_repo_source_sample_text_no_matching_files(mock_github_get):
    mock_github_get(
        [
            FakeResponse(200, json_data={"default_branch": "main"}),
            FakeResponse(200, json_data={"tree": [{"path": "README.md", "type": "blob"}]}),
        ]
    )

    result = fetch_repo_source_sample_text("octocat/hello-world")

    assert "찾지 못했습니다" in result


def test_fetch_repo_source_sample_text_repo_not_found(mock_github_get):
    mock_github_get([FakeResponse(404)])

    result = fetch_repo_source_sample_text("octocat/does-not-exist")

    assert result.startswith("Error")


def test_fetch_repo_source_sample_text_tree_request_fails(mock_github_get):
    mock_github_get(
        [
            FakeResponse(200, json_data={"default_branch": "main"}),
            FakeResponse(500),
        ]
    )

    result = fetch_repo_source_sample_text("octocat/hello-world")

    assert result.startswith("Error")


# ---------------------------------------------------------------------------
# @tool 래퍼: .invoke()로도 정상 동작해야 한다
# ---------------------------------------------------------------------------


def test_fetch_repo_overview_tool_invoke(mock_github_get):
    mock_github_get([FakeResponse(404)])

    result = fetch_repo_overview.invoke({"repo": "octocat/does-not-exist"})

    assert result.startswith("Error")


def test_fetch_repo_source_sample_tool_invoke(mock_github_get):
    mock_github_get([FakeResponse(404)])

    result = fetch_repo_source_sample.invoke({"repo": "octocat/does-not-exist"})

    assert result.startswith("Error")
