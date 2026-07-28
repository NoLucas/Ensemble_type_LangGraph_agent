"""
에이전트가 사용할 도구(tool) 모음.

GitHub 공개 REST API(비인증, 시간당 60회 제한)만 사용해서 두 도구를 제공한다:
1. fetch_repo_overview: 저장소 메타데이터(설명/언어/스타 수)와 README 발췌.
   "요약 리포트"의 재료가 된다.
2. fetch_repo_source_sample: 저장소 트리에서 대표 소스 파일 몇 개를 골라
   내용을 가져온다. "코드 리뷰"의 재료가 된다.

두 도구 모두 예외를 던지지 않고 "Error: ..." 문자열을 반환한다. 그래프 안에서
ToolNode가 도구를 호출하는데, 여기서 예외가 새면 그래프 실행 전체가 죽기
때문에 실패(저장소 없음, rate limit, 네트워크 오류)도 반드시 정상적인
반환값으로 표현해야 한다.
"""

import re
from pathlib import Path

import requests
from langchain_core.tools import tool

GITHUB_API_BASE = "https://api.github.com"
_REQUEST_TIMEOUT = 10

# "owner/repo" 형식만 허용한다. LLM이 URL 전체나 잘못된 문자열을 넘길 수
# 있으므로, API를 호출하기 전에 걸러낸다.
_REPO_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")

# 코드 리뷰 대상으로 삼을 확장자 화이트리스트. 설정/데이터 파일은 제외한다.
_SOURCE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs",
    ".java", ".kt", ".rb", ".c", ".cpp", ".cs",
}
# 테스트/벤더/빌드 산출물 경로는 "저장소의 진짜 코드"가 아니므로 후보에서 뺀다.
_SKIP_PATH_PARTS = {"test", "tests", "vendor", "node_modules", "dist", "build", ".github"}

_MAX_README_CHARS = 3000
_MAX_SOURCE_FILES = 3
_MAX_FILE_CHARS = 1500


def _validate_repo(repo: str) -> str | None:
    """repo가 'owner/repo' 형식이 아니면 에러 문자열을, 맞으면 None을 반환한다."""
    if not repo or not _REPO_PATTERN.match(repo):
        return f"Error: 저장소는 'owner/repo' 형식이어야 합니다 ({repo!r})."
    return None


def _fetch_repo_metadata(repo: str):
    """저장소 메타데이터를 가져온다. 성공 시 (dict, None), 실패 시 (None, 에러문자열)."""
    try:
        response = requests.get(f"{GITHUB_API_BASE}/repos/{repo}", timeout=_REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        return None, f"Error: GitHub API 요청에 실패했습니다 ({exc})"

    if response.status_code == 404:
        return None, f"Error: 저장소를 찾을 수 없습니다 ({repo})."
    if response.status_code == 403:
        return None, "Error: GitHub API 요청 한도를 초과했습니다 (비인증 시 시간당 60회)."
    if response.status_code != 200:
        return None, f"Error: GitHub API가 예상치 못한 응답을 반환했습니다 (status={response.status_code})."
    return response.json(), None


def fetch_repo_overview_text(repo: str) -> str:
    """저장소 메타데이터 + README 발췌를 사람이 읽을 텍스트로 조립한다."""
    invalid = _validate_repo(repo)
    if invalid:
        return invalid

    meta, error = _fetch_repo_metadata(repo)
    if error:
        return error

    lines = [
        f"# {meta.get('full_name', repo)}",
        f"설명: {meta.get('description') or '(없음)'}",
        f"언어: {meta.get('language') or '(알 수 없음)'}",
        f"stars: {meta.get('stargazers_count', 0)}, forks: {meta.get('forks_count', 0)}",
        f"토픽: {', '.join(meta.get('topics') or []) or '(없음)'}",
    ]

    try:
        readme_response = requests.get(
            f"{GITHUB_API_BASE}/repos/{repo}/readme",
            headers={"Accept": "application/vnd.github.v3.raw"},
            timeout=_REQUEST_TIMEOUT,
        )
    except requests.RequestException:
        lines.append("\n(README를 가져오지 못했습니다.)")
        return "\n".join(lines)

    if readme_response.status_code == 200:
        lines.append("\n## README 발췌\n" + readme_response.text[:_MAX_README_CHARS])
    else:
        lines.append("\n(README 없음)")

    return "\n".join(lines)


def fetch_repo_source_sample_text(repo: str) -> str:
    """저장소 트리에서 대표 소스 파일 몇 개를 골라 내용을 이어붙인다."""
    invalid = _validate_repo(repo)
    if invalid:
        return invalid

    meta, error = _fetch_repo_metadata(repo)
    if error:
        return error
    default_branch = meta.get("default_branch") or "main"

    try:
        tree_response = requests.get(
            f"{GITHUB_API_BASE}/repos/{repo}/git/trees/{default_branch}",
            params={"recursive": "1"},
            timeout=_REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        return f"Error: 파일 목록을 가져오지 못했습니다 ({exc})"
    if tree_response.status_code != 200:
        return f"Error: 파일 목록을 가져오지 못했습니다 (status={tree_response.status_code})."

    candidates = [
        item["path"]
        for item in tree_response.json().get("tree", [])
        if item.get("type") == "blob"
        and Path(item["path"]).suffix in _SOURCE_EXTENSIONS
        and not (_SKIP_PATH_PARTS & set(Path(item["path"]).parts))
    ]
    # 경로 깊이가 얕은(=저장소의 핵심에 가까운) 파일부터, 동률이면 이름순으로.
    candidates.sort(key=lambda p: (p.count("/"), p))
    selected = candidates[:_MAX_SOURCE_FILES]

    if not selected:
        return f"({repo}에서 리뷰할 소스 파일을 찾지 못했습니다.)"

    sections = []
    for path in selected:
        try:
            file_response = requests.get(
                f"{GITHUB_API_BASE}/repos/{repo}/contents/{path}",
                headers={"Accept": "application/vnd.github.v3.raw"},
                timeout=_REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            sections.append(f"### {path}\n(가져오기 실패: {exc})")
            continue
        if file_response.status_code != 200:
            sections.append(f"### {path}\n(가져오기 실패: status={file_response.status_code})")
            continue
        sections.append(f"### {path}\n```\n{file_response.text[:_MAX_FILE_CHARS]}\n```")

    return "\n\n".join(sections)


@tool
def fetch_repo_overview(repo: str) -> str:
    """GitHub 저장소의 메타데이터(설명/언어/스타 수)와 README 발췌를 가져온다.
    repo는 'owner/repo' 형식이어야 한다 (예: 'langchain-ai/langgraph')."""
    return fetch_repo_overview_text(repo)


@tool
def fetch_repo_source_sample(repo: str) -> str:
    """GitHub 저장소에서 대표적인 소스 파일 몇 개의 내용을 가져와 코드 리뷰에 쓴다.
    repo는 'owner/repo' 형식이어야 한다 (예: 'langchain-ai/langgraph')."""
    return fetch_repo_source_sample_text(repo)
