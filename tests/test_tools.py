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


def test_fetch_repo_source_sample_text_prefers_primary_language_over_shallower_file(
    mock_github_get,
):
    # 저장소의 대표 언어는 Python이다. app.js가 경로는 더 얕지만(루트),
    # 주 언어와 일치하는 nested/pkg/util.py가 우선 선택되어야 한다 —
    # 다른 언어로 섞여 들어온 설정/빌드 스크립트보다 실제 코드가 리뷰
    # 대상으로서 더 대표성이 있다.
    mock_github_get(
        [
            FakeResponse(200, json_data={"default_branch": "main", "language": "Python"}),
            FakeResponse(
                200,
                json_data={
                    "tree": [
                        {"path": "app.js", "type": "blob", "size": 500},
                        {"path": "nested/pkg/util.py", "type": "blob", "size": 500},
                    ]
                },
            ),
            FakeResponse(200, text="def util(): pass"),
            FakeResponse(200, text="console.log('hi')"),
        ]
    )

    result = fetch_repo_source_sample_text("octocat/hello-world")

    assert result.startswith("### nested/pkg/util.py")


def test_fetch_repo_source_sample_text_prefers_entrypoint_filename(mock_github_get):
    # 네 파일 모두 같은 언어·같은 깊이지만, main.py만 진입점 파일명이다.
    # _MAX_SOURCE_FILES(3)이라 넷 중 하나는 밀려나야 하는데, 점수가 가장
    # 낮은 일반 파일(알파벳 역순으로 utils.py)이 밀려나고 main.py는 항상
    # 포함되며 맨 먼저 나와야 한다.
    mock_github_get(
        [
            FakeResponse(200, json_data={"default_branch": "main", "language": "Python"}),
            FakeResponse(
                200,
                json_data={
                    "tree": [
                        {"path": "utils.py", "type": "blob", "size": 500},
                        {"path": "helpers.py", "type": "blob", "size": 500},
                        {"path": "config.py", "type": "blob", "size": 500},
                        {"path": "main.py", "type": "blob", "size": 500},
                    ]
                },
            ),
            FakeResponse(200, text="config"),
            FakeResponse(200, text="helpers"),
            FakeResponse(200, text="main"),
        ]
    )

    result = fetch_repo_source_sample_text("octocat/hello-world")

    assert result.startswith("### main.py")
    assert "utils.py" not in result  # 알파벳순 동점 최하위라 3개 슬롯에서 밀려남


def test_fetch_repo_source_sample_text_excludes_examples_directory(mock_github_get):
    # examples/ 하위 파일은 흔히 index.js 같은 진입점 파일명을 그대로 쓴다.
    # 그대로 두면 진입점 가산점 때문에 실제 라이브러리 구현(lib/)보다
    # examples가 먼저 뽑히는 문제가 실제로 있었다(expressjs/express로
    # 수동 검증 중 발견) — examples/를 스킵 목록에 추가해서 고쳤다.
    mock_github_get(
        [
            FakeResponse(200, json_data={"default_branch": "main", "language": "JavaScript"}),
            FakeResponse(
                200,
                json_data={
                    "tree": [
                        {"path": "examples/auth/index.js", "type": "blob", "size": 500},
                        {"path": "lib/application.js", "type": "blob", "size": 500},
                    ]
                },
            ),
            FakeResponse(200, text="app"),
        ]
    )

    result = fetch_repo_source_sample_text("octocat/hello-world")

    assert "examples/" not in result
    assert "lib/application.js" in result


def test_fetch_repo_source_sample_text_excludes_empty_files(mock_github_get):
    # empty.py(size=0)는 사실상 배제되어, 나머지 3개(a/b/c.py)가 선택돼야 한다.
    mock_github_get(
        [
            FakeResponse(200, json_data={"default_branch": "main", "language": "Python"}),
            FakeResponse(
                200,
                json_data={
                    "tree": [
                        {"path": "empty.py", "type": "blob", "size": 0},
                        {"path": "a.py", "type": "blob", "size": 500},
                        {"path": "b.py", "type": "blob", "size": 500},
                        {"path": "c.py", "type": "blob", "size": 500},
                    ]
                },
            ),
            FakeResponse(200, text="a"),
            FakeResponse(200, text="b"),
            FakeResponse(200, text="c"),
        ]
    )

    result = fetch_repo_source_sample_text("octocat/hello-world")

    assert "empty.py" not in result
    assert "a.py" in result and "b.py" in result and "c.py" in result


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


# ---------------------------------------------------------------------------
# GITHUB_TOKEN 인증: 있으면 Authorization 헤더를 붙이고, 없으면 그대로
# 비인증으로 호출한다(기존 동작 유지).
# ---------------------------------------------------------------------------


def test_fetch_repo_overview_text_adds_auth_header_when_token_set(mock_github_get, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "test-token-123")
    fake = mock_github_get([FakeResponse(404)])

    fetch_repo_overview_text("octocat/does-not-exist")

    _, kwargs = fake.calls[0]
    assert kwargs["headers"]["Authorization"] == "Bearer test-token-123"


def test_fetch_repo_overview_text_omits_auth_header_when_token_unset(mock_github_get, monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    fake = mock_github_get([FakeResponse(404)])

    fetch_repo_overview_text("octocat/does-not-exist")

    _, kwargs = fake.calls[0]
    assert "Authorization" not in kwargs["headers"]


def test_fetch_repo_source_sample_text_adds_auth_header_to_every_call_when_token_set(
    mock_github_get, monkeypatch
):
    monkeypatch.setenv("GITHUB_TOKEN", "test-token-123")
    fake = mock_github_get(
        [
            FakeResponse(200, json_data={"default_branch": "main"}),
            FakeResponse(200, json_data={"tree": [{"path": "main.py", "type": "blob"}]}),
            FakeResponse(200, text="print('hi')"),
        ]
    )

    fetch_repo_source_sample_text("octocat/hello-world")

    assert len(fake.calls) == 3
    for _, kwargs in fake.calls:
        assert kwargs["headers"]["Authorization"] == "Bearer test-token-123"


def test_fetch_repo_overview_text_readme_request_keeps_accept_header_with_token(
    mock_github_get, monkeypatch
):
    # Authorization 헤더를 추가하면서 기존 Accept 헤더(raw 콘텐츠 요청용)를
    # 실수로 덮어쓰지 않는지 확인한다.
    monkeypatch.setenv("GITHUB_TOKEN", "test-token-123")
    fake = mock_github_get(
        [
            FakeResponse(200, json_data={"full_name": "octocat/hello-world"}),
            FakeResponse(200, text="README"),
        ]
    )

    fetch_repo_overview_text("octocat/hello-world")

    _, readme_kwargs = fake.calls[1]
    assert readme_kwargs["headers"]["Accept"] == "application/vnd.github.v3.raw"
    assert readme_kwargs["headers"]["Authorization"] == "Bearer test-token-123"
