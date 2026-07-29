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


def _file_entry(path: str, size: int = 500) -> dict:
    """GitHub contents API가 파일 하나에 대해 반환하는 항목 모양."""
    return {"path": path, "type": "file", "size": size}


def _dir_entry(path: str) -> dict:
    """GitHub contents API가 디렉토리 하나에 대해 반환하는 항목 모양."""
    return {"path": path, "type": "dir", "size": 0}

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
                json_data=[
                    _file_entry("main.py"),
                    _file_entry("README.md"),  # 소스 확장자가 아니라 후보에서 빠진다.
                    _dir_entry("tests"),  # 스킵 대상이라 아예 탐색하지 않는다.
                ],
            ),
            FakeResponse(200, text="print('hello')"),
        ]
    )

    result = fetch_repo_source_sample_text("octocat/hello-world")

    assert "main.py" in result
    assert "print('hello')" in result
    assert "README.md" not in result


def test_fetch_repo_source_sample_text_finds_deeply_nested_code_before_shallow_siblings(
    mock_github_get,
):
    # kubernetes/kubernetes로 수동 검증하다 발견한 실제 시나리오를 그대로
    # 재현한다: 루트 바로 아래에는 코드가 없는 얕은 디렉토리가 여럿 있고
    # (문서/설정용, 힌트 아님), 실제 코드는 힌트 디렉토리(pkg) 몇 단계
    # 안쪽에 있다. 너비 우선(BFS)이었다면 얕은 형제 디렉토리들을 옆으로
    # 훑다가 호출 예산을 다 써버려서 아무것도 못 찾았을 상황 — 깊이
    # 우선(DFS) + 힌트 우선 탐색이라야 이 구조에서도 파일을 찾아낸다.
    mock_github_get(
        [
            FakeResponse(200, json_data={"default_branch": "main", "language": "Go"}),
            FakeResponse(
                200,
                json_data=[
                    _dir_entry("docs"),
                    _dir_entry("hack"),
                    _dir_entry("third_party"),
                    _dir_entry("pkg"),  # 힌트 — 나머지보다 먼저 파고든다.
                ],
            ),  # 루트
            FakeResponse(200, json_data=[_dir_entry("pkg/controller")]),  # pkg/
            FakeResponse(200, json_data=[_file_entry("pkg/controller/foo.go")]),  # pkg/controller/
            # pkg 쪽에서 파일을 찾은 뒤에도 스택에 남아있던 얕은 형제
            # 디렉토리들(third_party/hack/docs, LIFO라 이 순서로 방문)은
            # 계속 방문된다 — 전부 빈 디렉토리라 후보를 더 보태진 않는다.
            FakeResponse(200, json_data=[]),  # third_party/
            FakeResponse(200, json_data=[]),  # hack/
            FakeResponse(200, json_data=[]),  # docs/
            FakeResponse(200, text="package controller"),
        ]
    )

    result = fetch_repo_source_sample_text("octocat/hello-world")

    assert "pkg/controller/foo.go" in result
    assert "package controller" in result


def test_fetch_repo_source_sample_text_excludes_test_file_naming_conventions(mock_github_get):
    # tests/ 같은 디렉토리 관례와 달리, Go는 같은 디렉토리에 foo.go와
    # foo_test.go를 나란히 둔다 — kubernetes/kubernetes로 수동 검증하다
    # metrics_block_linux_test.go가 걸러지지 않는 걸 발견하고 고쳤다.
    # 디렉토리 스킵 목록이 아니라 파일명 접미사(_test/.spec 등)로 판단한다.
    mock_github_get(
        [
            FakeResponse(200, json_data={"default_branch": "main", "language": "Go"}),
            FakeResponse(
                200,
                json_data=[
                    _file_entry("metrics_block.go"),
                    _file_entry("metrics_block_linux_test.go"),
                    _file_entry("component.spec.ts"),
                    _file_entry("test_helpers.go"),
                ],
            ),
            FakeResponse(200, text="package volume"),
        ]
    )

    result = fetch_repo_source_sample_text("octocat/hello-world")

    assert "metrics_block.go" in result
    assert "metrics_block_linux_test.go" not in result
    assert "component.spec.ts" not in result
    assert "test_helpers.go" not in result


def test_fetch_repo_source_sample_text_prefers_primary_language_over_shallower_file(
    mock_github_get,
):
    # 저장소의 대표 언어는 Python이다. app.js가 경로는 더 얕지만(루트),
    # 주 언어와 일치하는 pkg/util.py가 우선 선택되어야 한다 — 다른 언어로
    # 섞여 들어온 설정/빌드 스크립트보다 실제 코드가 리뷰 대상으로서 더
    # 대표성이 있다. pkg/는 소스 디렉토리 힌트라 루트 다음으로 탐색된다.
    mock_github_get(
        [
            FakeResponse(200, json_data={"default_branch": "main", "language": "Python"}),
            FakeResponse(200, json_data=[_file_entry("app.js"), _dir_entry("pkg")]),  # 루트
            FakeResponse(200, json_data=[_file_entry("pkg/util.py")]),  # pkg/
            FakeResponse(200, text="def util(): pass"),
            FakeResponse(200, text="console.log('hi')"),
        ]
    )

    result = fetch_repo_source_sample_text("octocat/hello-world")

    assert result.startswith("### pkg/util.py")


def test_fetch_repo_source_sample_text_prefers_entrypoint_filename(mock_github_get):
    # 네 파일 모두 같은 언어·같은 깊이(루트)지만, main.py만 진입점
    # 파일명이다. _MAX_SOURCE_FILES(3)이라 넷 중 하나는 밀려나야 하는데,
    # 점수가 가장 낮은 일반 파일(알파벳 역순으로 utils.py)이 밀려나고
    # main.py는 항상 포함되며 맨 먼저 나와야 한다.
    mock_github_get(
        [
            FakeResponse(200, json_data={"default_branch": "main", "language": "Python"}),
            FakeResponse(
                200,
                json_data=[
                    _file_entry("utils.py"),
                    _file_entry("helpers.py"),
                    _file_entry("config.py"),
                    _file_entry("main.py"),
                ],
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
    # 실제 라이브러리 구현(lib/)보다 examples가 먼저 뽑히는 문제가 실제로
    # 있었다(expressjs/express로 수동 검증 중 발견) — examples를 스킵
    # 목록에 추가해서, 디렉토리 나열 단계에서부터(그 안을 들여다보지도
    # 않고) 제외한다. lib/는 소스 디렉토리 힌트라 탐색된다.
    mock_github_get(
        [
            FakeResponse(200, json_data={"default_branch": "main", "language": "JavaScript"}),
            FakeResponse(
                200, json_data=[_dir_entry("examples"), _dir_entry("lib")]
            ),  # 루트 — examples는 스킵 대상이라 큐에 안 들어간다.
            FakeResponse(200, json_data=[_file_entry("lib/application.js")]),  # lib/
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
                json_data=[
                    _file_entry("empty.py", size=0),
                    _file_entry("a.py"),
                    _file_entry("b.py"),
                    _file_entry("c.py"),
                ],
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
            FakeResponse(200, json_data=[_file_entry("README.md")]),
        ]
    )

    result = fetch_repo_source_sample_text("octocat/hello-world")

    assert "찾지 못했습니다" in result


def test_fetch_repo_source_sample_text_repo_not_found(mock_github_get):
    mock_github_get([FakeResponse(404)])

    result = fetch_repo_source_sample_text("octocat/does-not-exist")

    assert result.startswith("Error")


def test_fetch_repo_source_sample_text_root_listing_fails(mock_github_get):
    mock_github_get(
        [
            FakeResponse(200, json_data={"default_branch": "main"}),
            FakeResponse(500),
        ]
    )

    result = fetch_repo_source_sample_text("octocat/hello-world")

    assert result.startswith("Error")


def test_fetch_repo_source_sample_text_tolerates_subdirectory_listing_failure(
    mock_github_get,
):
    # 루트 나열은 성공했지만 그 다음으로 탐색한 하위 디렉토리 하나가
    # 실패하는(네트워크 오류/일시적 장애) 경우, 전체 탐색을 포기하지 않고
    # 루트에서 이미 찾은 후보로 계속 진행해야 한다.
    mock_github_get(
        [
            FakeResponse(200, json_data={"default_branch": "main", "language": "Python"}),
            FakeResponse(200, json_data=[_file_entry("main.py"), _dir_entry("lib")]),  # 루트
            FakeResponse(500),  # lib/ 나열 실패
            FakeResponse(200, text="print('main')"),
        ]
    )

    result = fetch_repo_source_sample_text("octocat/hello-world")

    assert result.startswith("### main.py")


def test_discover_source_candidates_never_calls_recursive_tree_endpoint(mock_github_get):
    # 예전에는 /git/trees/{branch}?recursive=1로 저장소 전체를 한 번에
    # 받았는데, 큰 저장소에서 응답이 수십 MB에 달해(실측: kubernetes
    # 10MB/37,390개 항목, linux 17.6MB/71,798개 항목이면서도 truncated=True)
    # 리뷰 목적에 비해 지나치게 무거웠다. 디렉토리 단위 나열(/contents)로
    # 바꾼 뒤 이 무거운 엔드포인트를 다시는 호출하지 않는지 회귀 고정한다.
    fake = mock_github_get(
        [
            FakeResponse(200, json_data={"default_branch": "main", "language": "Python"}),
            FakeResponse(200, json_data=[_file_entry("main.py")]),
            FakeResponse(200, text="print('hi')"),
        ]
    )

    fetch_repo_source_sample_text("octocat/hello-world")

    assert not any("git/trees" in url for url, _ in fake.calls)


def test_discover_source_candidates_bounds_directory_listing_calls(mock_github_get):
    # 디렉토리마다 그 다음 단계로 이어지는 하위 디렉토리 하나씩만 두면,
    # 힌트를 계속 따라가는 한 무한정 깊어질 수 있는 저장소 구조를 흉내낼
    # 수 있다. 디렉토리 나열 호출 횟수는 _MAX_DIR_LISTING_CALLS(8)를
    # 넘지 않아야 한다 — 그래야 대형 저장소에서도 호출이 무한정 늘어나지
    # 않는다. 레벨 0(루트)~7까지 8번만 응답을 준비해뒀으므로, 코드가 실수로
    # 9번째를 호출하면 QueuedGet이 즉시 AssertionError로 잡아낸다.
    levels = [
        "d1", "d1/d2", "d1/d2/d3", "d1/d2/d3/d4",
        "d1/d2/d3/d4/d5", "d1/d2/d3/d4/d5/d6",
        "d1/d2/d3/d4/d5/d6/d7", "d1/d2/d3/d4/d5/d6/d7/d8",
    ]
    responses = [FakeResponse(200, json_data={"default_branch": "main", "language": "Python"})]
    responses += [FakeResponse(200, json_data=[_dir_entry(path)]) for path in levels]

    fake = mock_github_get(responses)

    result = fetch_repo_source_sample_text("octocat/hello-world")

    listing_calls = [c for c in fake.calls if "/contents" in c[0]]
    assert len(listing_calls) == 8
    assert "찾지 못했습니다" in result


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
            FakeResponse(200, json_data=[_file_entry("main.py")]),
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
