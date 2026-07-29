"""
에이전트가 사용할 도구(tool) 모음.

GitHub 공개 REST API를 사용해서 네 도구를 제공한다:
1. fetch_repo_overview: 저장소 메타데이터(설명/언어/스타 수)와 README 발췌.
   "요약 리포트"의 재료가 된다.
2. fetch_repo_source_sample: 저장소 트리에서 대표 소스 파일 몇 개를 골라
   전체 내용을 가져온다. 정식 "코드 리뷰"의 재료가 된다.
3. fetch_repo_structure: 저장소를 fetch_repo_source_sample보다 훨씬 넓게
   훑되, 파일 전체 본문이 아니라 함수/클래스 시그니처만 추출한다. 큰
   저장소를 "가볍게 훑어보고 공부"하려는 스터디 모드의 1단계 재료다 —
   같은 토큰 예산으로 fetch_repo_source_sample보다 훨씬 많은 파일의
   구조를 보여준다.
4. fetch_repo_file: 사용자가 콕 집은 파일 하나의 전체 내용을 가져온다.
   fetch_repo_structure로 구조를 먼저 훑고 나서, 관심 가는 파일 하나만
   깊게 파고들 때(드릴다운) 쓰는 2단계 재료다.

기본은 비인증 호출(시간당 60회 제한)이다. 환경변수 GITHUB_TOKEN이 설정돼
있으면 자동으로 Authorization 헤더를 붙여 시간당 5000회로 늘어난다 —
토큰이 없어도 동작은 그대로라 기존 사용자에게 아무 영향이 없다.

네 도구 모두 예외를 던지지 않고 "Error: ..." 문자열을 반환한다. 그래프 안에서
ToolNode가 도구를 호출하는데, 여기서 예외가 새면 그래프 실행 전체가 죽기
때문에 실패(저장소 없음, rate limit, 네트워크 오류)도 반드시 정상적인
반환값으로 표현해야 한다.
"""

import os
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
# 테스트/벤더/빌드 산출물/예제 경로는 "저장소의 진짜 구현 코드"가 아니므로
# 후보에서 뺀다. examples/demo는 특히 중요한 제외 대상이다 — 흔히
# index.js/main.py 같은 진입점 파일명을 그대로 쓰기 때문에, 진입점 가산점
# 때문에 실제 라이브러리 구현(lib/, src/)보다 먼저 뽑히는 경우가 실제로
# 있었다(psf/requests 등으로 수동 검증 중 expressjs/express에서 발견).
_SKIP_PATH_PARTS = {
    "test", "tests", "vendor", "node_modules", "dist", "build", ".github",
    "example", "examples", "demo", "demos", "sample", "samples",
}


def _looks_like_test_file(path: str) -> bool:
    """디렉토리명이 아니라 파일명 자체가 테스트 파일 명명 관례를 따르는지
    본다. `tests/` 같은 디렉토리 관례(_SKIP_PATH_PARTS)와 달리, Go는
    같은 디렉토리에 `foo.go`/`foo_test.go`를 나란히 두는 관례를 쓴다 —
    kubernetes/kubernetes로 수동 검증하다가 `metrics_block_linux_test.go`
    가 걸러지지 않는 걸 발견하고 추가했다."""
    stem = Path(path).stem.lower()
    return stem.endswith("_test") or stem.endswith(".test") or stem.endswith(".spec") or stem.startswith("test_")

# GitHub 저장소 메타데이터의 language 필드(예: "Python") -> 그 언어의 대표
# 확장자. 소스 파일 후보를 고를 때 "저장소의 실제 주 언어와 일치하는 파일"에
# 가산점을 준다 — 화이트리스트에 여러 언어 확장자가 섞여 있다 보니, 예를 들어
# Python 저장소인데 우연히 섞여 있는 .js 설정 스크립트가 뽑히는 걸 막는다.
_LANGUAGE_EXTENSIONS = {
    "python": {".py"},
    "javascript": {".js", ".jsx"},
    "typescript": {".ts", ".tsx"},
    "go": {".go"},
    "rust": {".rs"},
    "java": {".java"},
    "kotlin": {".kt"},
    "ruby": {".rb"},
    "c++": {".cpp"},
    "c": {".c"},
    "c#": {".cs"},
}
# 파일명(확장자 제외)이 이 목록에 있으면 "저장소의 진입점"일 가능성이 높다고
# 보고 가산점을 준다.
_ENTRYPOINT_STEMS = {"main", "__main__", "__init__", "index", "app", "cli", "server", "run"}

# 이 범위 밖의 파일 크기는 리뷰 가치가 낮다고 본다: 너무 작으면(스텁/빈
# 파일) 볼 코드가 없고, 너무 크면 생성된 코드·데이터·번들일 가능성이 높다
# (또 어차피 _MAX_FILE_CHARS로 잘려서 앞부분만 봐도 대표성이 떨어진다).
_MIN_USEFUL_FILE_SIZE = 50
_MAX_USEFUL_FILE_SIZE = 20000

# 소스 파일이 있을 법한 디렉토리 이름 — 탐색 큐에서 이 이름과 일치하는
# 디렉토리를 먼저 방문한다.
_SOURCE_DIR_HINTS = {"src", "lib", "app", "pkg", "cmd", "core", "source"}
# 디렉토리 나열(비재귀) API를 몇 번까지 호출할지의 상한. 저장소 전체를
# recursive=1로 한 번에 받는 대신 디렉토리 단위로 필요한 만큼만 탐색하기
# 위한 예산이다 — 아래 _discover_source_candidates() 참고. DFS라 유망한
# 경로 하나가 몇 단계 안쪽에서야 파일을 내놓는 저장소(kubernetes/kubernetes
# 등 대형 모노레포)도 감안해서 여유를 좀 둔다.
_MAX_DIR_LISTING_CALLS = 8
# 이만큼 후보가 모이면 더 탐색하지 않고 채점 단계로 넘어간다.
_MAX_CANDIDATES_BEFORE_STOP = 15

_MAX_README_CHARS = 3000
_MAX_SOURCE_FILES = 3
_MAX_FILE_CHARS = 1500

# fetch_repo_structure는 전체 본문 대신 시그니처만 보내므로 파일당 토큰이
# 훨씬 적다 — 그만큼 더 많은 파일을 훑어서 "구조 지도"의 커버리지를 넓힌다.
_MAX_STRUCTURE_FILES = 10
_MAX_SIGNATURE_CHARS_PER_FILE = 800

# fetch_repo_file은 사용자가 콕 집은 파일 하나만 보는 것이므로, 대표
# 발췌(_MAX_FILE_CHARS)보다 훨씬 넉넉하게 허용한다.
_MAX_SINGLE_FILE_CHARS = 8000

# 확장자 그룹별로 "함수/클래스 선언부로 보이는 줄"을 잡아내는 정규식.
# 정밀한 파서가 아니라 언어별 관례를 이용한 휴리스틱이다 — 본문 없이
# 시그니처만 뽑아서 구조 지도의 토큰을 줄이는 게 목적이므로, 완벽할
# 필요는 없고 "대충 구조가 보이는" 정도면 충분하다.
_SIGNATURE_PATTERNS: list[tuple[frozenset[str], re.Pattern]] = [
    (frozenset({".py"}), re.compile(r"^\s{0,4}(async\s+def\s|def\s|class\s)")),
    (
        frozenset({".js", ".jsx", ".ts", ".tsx"}),
        re.compile(
            r"^\s{0,4}(export\s+)?(default\s+)?(async\s+)?"
            r"(function\b|class\b|const\s+\w+\s*=\s*(\([^)]*\)|\w+)\s*=>)"
        ),
    ),
    (frozenset({".go"}), re.compile(r"^(func\b|type\s+\w+\s+(struct|interface)\b)")),
    (
        frozenset({".rs"}),
        re.compile(r"^\s{0,4}(pub\s+)?(async\s+)?(fn\s|struct\s|enum\s|trait\s|impl\b)"),
    ),
    (
        frozenset({".java", ".kt"}),
        re.compile(
            r"^\s{0,4}(public|private|protected)?\s*(static\s+)?(final\s+)?"
            r"(class|interface|enum|fun)\b"
        ),
    ),
    (frozenset({".rb"}), re.compile(r"^\s{0,4}(def\s|class\s|module\s)")),
]


def _extract_signatures(content: str, suffix: str) -> str:
    """소스 코드 본문에서 함수/클래스 선언부(시그니처)로 보이는 줄만 뽑는다.

    정밀한 AST 파서가 아니라 언어별 관례(들여쓰기 얕은 def/class/function
    등으로 시작하는 줄)에 기대는 정규식 휴리스틱이다 — "본문 없이도 대충
    구조가 보이는" 정도가 목적이라 완벽한 정확도는 필요 없다. 화이트리스트에
    없는 확장자(.c/.cpp/.cs 등, 시그니처가 여러 줄에 걸치기 쉬워 한 줄
    정규식으로는 신뢰도가 낮다)는 빈 문자열을 반환한다 — 호출부가 이 경우
    해당 파일을 통째로 건너뛴다.
    """
    pattern = next((regex for exts, regex in _SIGNATURE_PATTERNS if suffix in exts), None)
    if pattern is None:
        return ""
    lines = [line.rstrip() for line in content.splitlines() if pattern.match(line)]
    return "\n".join(lines)


def _score_source_candidate(path: str, size: int, primary_extensions: set[str]) -> int:
    """소스 파일 후보 하나의 "리뷰 대표성" 점수를 매긴다. 높을수록 먼저 뽑힌다.

    결정론적 정렬을 위한 점수이지 정밀한 랭킹 알고리즘이 아니다 — 몇 가지
    뚜렷한 신호(주 언어 일치, 진입점 파일명, 얕은 경로, 적당한 크기)만
    단순 가산/감산으로 조합한다.
    """
    p = Path(path)
    score = 0
    if p.suffix in primary_extensions:
        score += 10  # 저장소의 대표 언어와 일치하는 파일을 최우선시한다.
    if p.stem.lower() in _ENTRYPOINT_STEMS:
        score += 5  # 진입점으로 보이는 파일명.
    score -= path.count("/")  # 얕은 경로일수록(핵심에 가까울수록) 가산.
    if size == 0:
        score -= 100  # 빈 파일은 리뷰할 내용이 없으므로 사실상 배제한다.
    elif _MIN_USEFUL_FILE_SIZE <= size <= _MAX_USEFUL_FILE_SIZE:
        score += 2  # 스텁도 생성 코드 덤프도 아닌, 읽을 만한 크기.
    return score


def _auth_headers(extra: dict | None = None) -> dict:
    """GITHUB_TOKEN 환경변수가 있으면 Authorization 헤더를 추가한다.

    매 호출 시점에 os.environ을 읽는다(모듈 임포트 시점에 한 번만 읽지
    않는다) — 그래야 테스트에서 monkeypatch.setenv로 토큰 유무를 바꿔가며
    검증할 수 있고, 실제로도 프로세스가 떠 있는 동안 .env가 다시
    로드되는 시나리오를 자연스럽게 반영한다.
    """
    headers = dict(extra or {})
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _validate_repo(repo: str) -> str | None:
    """repo가 'owner/repo' 형식이 아니면 에러 문자열을, 맞으면 None을 반환한다."""
    if not repo or not _REPO_PATTERN.match(repo):
        return f"Error: 저장소는 'owner/repo' 형식이어야 합니다 ({repo!r})."
    return None


def _fetch_repo_metadata(repo: str):
    """저장소 메타데이터를 가져온다. 성공 시 (dict, None), 실패 시 (None, 에러문자열)."""
    try:
        response = requests.get(
            f"{GITHUB_API_BASE}/repos/{repo}", headers=_auth_headers(), timeout=_REQUEST_TIMEOUT
        )
    except requests.RequestException as exc:
        return None, f"Error: GitHub API 요청에 실패했습니다 ({exc})"

    if response.status_code == 404:
        return None, f"Error: 저장소를 찾을 수 없습니다 ({repo})."
    if response.status_code == 403:
        return None, (
            "Error: GitHub API 요청 한도를 초과했습니다 "
            "(비인증 시간당 60회 / GITHUB_TOKEN 설정 시 5000회)."
        )
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
            headers=_auth_headers({"Accept": "application/vnd.github.v3.raw"}),
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


def _list_directory(repo: str, path: str, branch: str):
    """path 바로 아래 항목만(비재귀) 나열한다.

    성공 시 (entries, None), 실패 시 (None, 에러 문자열)을 반환한다.
    `path`가 빈 문자열이면 저장소 루트를 나열한다.
    """
    url = f"{GITHUB_API_BASE}/repos/{repo}/contents/{path}" if path else f"{GITHUB_API_BASE}/repos/{repo}/contents"
    try:
        response = requests.get(
            url, params={"ref": branch}, headers=_auth_headers(), timeout=_REQUEST_TIMEOUT
        )
    except requests.RequestException as exc:
        return None, f"Error: 파일 목록을 가져오지 못했습니다 ({exc})"
    if response.status_code != 200:
        return None, f"Error: 파일 목록을 가져오지 못했습니다 (status={response.status_code})."
    data = response.json()
    # contents API는 path가 디렉토리면 항목 리스트를, 파일이면 단일 객체를
    # 반환한다 — 후자는 우리가 기대한 모양이 아니므로 빈 목록으로 취급한다.
    return (data if isinstance(data, list) else []), None


def _discover_source_candidates(repo: str, branch: str):
    """디렉토리를 하나씩(비재귀) 나열하며 소스 파일 후보를 모은다.

    예전에는 /git/trees/{branch}?recursive=1로 저장소 전체 파일 목록을
    한 번에 받았는데, 이러면 큰 저장소(예: kubernetes/kubernetes)에서도
    응답이 수 MB~수십 MB에 달하고(실측: kubernetes 10MB/37,390개 항목,
    linux 17.6MB/71,798개 항목이면서 truncated=True로 일부 누락) 리뷰
    목적(파일 몇 개만 보면 됨)에 비해 지나치게 무겁고 느리다.

    대신 저장소를 **깊이 우선(DFS)**으로 탐색한다 — 스택에서 마지막에
    넣은 항목을 먼저 꺼내되, src/lib처럼 소스가 있을 법한 디렉토리를
    나중에 넣어서(=먼저 꺼내져서) 우선 파고들게 한다. 너비 우선(BFS)으로
    먼저 만들어봤다가 kubernetes/kubernetes로 실제 검증 중 문제를
    발견했다: 그 저장소는 실제 .go 코드가 `pkg/`, `cmd/` 몇 단계 더
    안쪽에 있는데, 루트 바로 아래에는 컴포넌트별 하위 디렉토리(26개+)만
    잔뜩 있어서 BFS는 그 얕은 디렉토리들을 옆으로 훑다가 호출 예산을
    다 써버리고 파일을 하나도 못 찾았다. DFS는 유망해 보이는 경로 하나를
    바닥까지 파고든 뒤에야 옆 가지로 넘어가므로, "코드가 몇 단계 안쪽에
    몰려 있는" 실제 저장소 구조에 훨씬 잘 맞는다.

    호출 횟수(_MAX_DIR_LISTING_CALLS)나 모인 후보 수
    (_MAX_CANDIDATES_BEFORE_STOP)가 충분해지면 멈춘다.

    반환: (candidates, error). candidates는 (path, size) 튜플 리스트.
    루트 디렉토리 조회 자체가 실패하면 error를 반환한다. 그 이후
    탐색에서 개별 하위 디렉토리 조회가 실패하는 것은(권한/일시적 오류 등)
    전체 탐색을 포기할 이유가 아니므로 그 디렉토리만 건너뛴다.
    """
    candidates: list[tuple[str, int]] = []
    stack = [""]
    calls = 0
    is_root = True

    while stack and calls < _MAX_DIR_LISTING_CALLS and len(candidates) < _MAX_CANDIDATES_BEFORE_STOP:
        current_dir = stack.pop()
        entries, error = _list_directory(repo, current_dir, branch)
        calls += 1
        if error:
            if is_root:
                return [], error
            entries = []
        is_root = False

        subdirs = []
        for entry in entries:
            path = entry.get("path", "")
            if entry.get("type") == "file":
                if (
                    Path(path).suffix in _SOURCE_EXTENSIONS
                    and not (_SKIP_PATH_PARTS & set(Path(path).parts))
                    and not _looks_like_test_file(path)
                ):
                    candidates.append((path, entry.get("size", 0)))
            elif entry.get("type") == "dir":
                if Path(path).name.lower() not in _SKIP_PATH_PARTS:
                    subdirs.append(path)
        # 힌트가 아닌 디렉토리를 먼저 스택에 넣고, 힌트 디렉토리를 나중에
        # 넣는다 — 스택은 LIFO라 나중에 넣은(힌트) 쪽이 먼저 꺼내진다.
        subdirs.sort(key=lambda p: Path(p).name.lower() in _SOURCE_DIR_HINTS)
        stack.extend(subdirs)

    return candidates, None


def _fetch_file_raw(repo: str, path: str):
    """path의 원본(raw) 텍스트 내용을 가져온다.

    성공 시 (text, None), 실패 시 (None, 에러 문자열)을 반환한다.
    fetch_repo_source_sample_text/fetch_repo_structure_text/
    fetch_repo_file_text가 공통으로 쓰는 "파일 하나의 raw 내용 조회"
    로직이다.
    """
    try:
        response = requests.get(
            f"{GITHUB_API_BASE}/repos/{repo}/contents/{path}",
            headers=_auth_headers({"Accept": "application/vnd.github.v3.raw"}),
            timeout=_REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        return None, f"(가져오기 실패: {exc})"
    if response.status_code == 404:
        return None, f"(가져오기 실패: 파일을 찾을 수 없습니다 — {path})"
    if response.status_code != 200:
        return None, f"(가져오기 실패: status={response.status_code})"
    return response.text, None


def fetch_repo_source_sample_text(repo: str) -> str:
    """저장소를 디렉토리 단위로 탐색해서 대표 소스 파일 몇 개를 골라 내용을 이어붙인다."""
    invalid = _validate_repo(repo)
    if invalid:
        return invalid

    meta, error = _fetch_repo_metadata(repo)
    if error:
        return error
    default_branch = meta.get("default_branch") or "main"

    candidates, error = _discover_source_candidates(repo, default_branch)
    if error:
        return error
    if not candidates:
        return f"({repo}에서 리뷰할 소스 파일을 찾지 못했습니다.)"

    primary_extensions = _LANGUAGE_EXTENSIONS.get((meta.get("language") or "").lower(), set())
    # 점수 높은 순(주 언어 일치 > 진입점 파일명 > 얕은 경로 > 적당한 크기),
    # 동점이면 경로 이름순으로 결정론적으로 정렬한다.
    candidates.sort(
        key=lambda c: (-_score_source_candidate(c[0], c[1], primary_extensions), c[0])
    )
    selected = [path for path, _size in candidates[:_MAX_SOURCE_FILES]]

    sections = []
    for path in selected:
        text, error = _fetch_file_raw(repo, path)
        if error:
            sections.append(f"### {path}\n{error}")
            continue
        sections.append(f"### {path}\n```\n{text[:_MAX_FILE_CHARS]}\n```")

    return "\n\n".join(sections)


def fetch_repo_structure_text(repo: str) -> str:
    """저장소를 fetch_repo_source_sample보다 훨씬 넓게 훑되, 파일 전체
    본문이 아니라 함수/클래스 시그니처만 추출해서 "구조 지도"를 만든다.

    스터디 모드의 1단계 재료다 — 같은 토큰 예산으로 fetch_repo_source_sample
    (전체 본문, 파일 3개)보다 훨씬 많은 파일(_MAX_STRUCTURE_FILES)의
    구조를 보여줘서, 코드를 한 줄도 안 읽고도 "이 저장소가 대략 어떻게
    생겼는지" 감을 잡을 수 있게 한다. 시그니처 추출이 지원되지 않는
    확장자(.c/.cpp/.cs 등)의 파일은 건너뛴다.
    """
    invalid = _validate_repo(repo)
    if invalid:
        return invalid

    meta, error = _fetch_repo_metadata(repo)
    if error:
        return error
    default_branch = meta.get("default_branch") or "main"

    candidates, error = _discover_source_candidates(repo, default_branch)
    if error:
        return error
    if not candidates:
        return f"({repo}에서 구조를 파악할 소스 파일을 찾지 못했습니다.)"

    primary_extensions = _LANGUAGE_EXTENSIONS.get((meta.get("language") or "").lower(), set())
    candidates.sort(
        key=lambda c: (-_score_source_candidate(c[0], c[1], primary_extensions), c[0])
    )
    selected = [path for path, _size in candidates[:_MAX_STRUCTURE_FILES]]

    sections = []
    for path in selected:
        text, error = _fetch_file_raw(repo, path)
        if error:
            continue  # 구조 지도는 개요 성격이라, 파일 하나가 실패해도 조용히 건너뛴다.
        signatures = _extract_signatures(text, Path(path).suffix)
        if signatures:
            sections.append(f"### {path}\n```\n{signatures[:_MAX_SIGNATURE_CHARS_PER_FILE]}\n```")

    if not sections:
        return f"({repo}에서 시그니처를 추출할 수 있는 파일을 찾지 못했습니다.)"
    return "\n\n".join(sections)


def fetch_repo_file_text(repo: str, path: str) -> str:
    """저장소의 특정 파일 하나의 전체 내용을 가져온다.

    스터디 모드의 2단계 재료다 — fetch_repo_structure로 구조 지도를 먼저
    훑은 뒤, 관심 가는 파일 하나만 콕 집어 드릴다운할 때 쓴다. 대표
    발췌(fetch_repo_source_sample)와 달리 어떤 파일을 볼지 사용자가
    직접 지정하므로 탐색 없이 바로 조회한다.
    """
    invalid = _validate_repo(repo)
    if invalid:
        return invalid
    if not path:
        return "Error: 파일 경로(path)를 지정해야 합니다."

    text, error = _fetch_file_raw(repo, path)
    if error:
        return f"Error: {path} {error}"
    return f"### {path}\n```\n{text[:_MAX_SINGLE_FILE_CHARS]}\n```"


@tool
def fetch_repo_overview(repo: str) -> str:
    """GitHub 저장소의 메타데이터(설명/언어/스타 수)와 README 발췌를 가져온다.
    repo는 'owner/repo' 형식이어야 한다 (예: 'langchain-ai/langgraph')."""
    return fetch_repo_overview_text(repo)


@tool
def fetch_repo_source_sample(repo: str) -> str:
    """GitHub 저장소에서 대표적인 소스 파일 몇 개의 전체 내용을 가져와 정식
    코드 리뷰에 쓴다. 토큰을 꽤 쓰므로, 사용자가 가볍게 구조만 훑어보고
    싶어하면 대신 fetch_repo_structure를 쓴다.
    repo는 'owner/repo' 형식이어야 한다 (예: 'langchain-ai/langgraph')."""
    return fetch_repo_source_sample_text(repo)


@tool
def fetch_repo_structure(repo: str) -> str:
    """GitHub 저장소를 넓게 훑어서 함수/클래스 시그니처만 담은 구조 지도를
    가져온다(전체 코드 본문은 포함하지 않아 토큰을 훨씬 적게 쓴다). 사용자가
    "구조만 보고 싶다"/"가볍게 훑어보고 싶다"/"공부하고 싶다"처럼 저장소를
    처음 파악하려 할 때 fetch_repo_source_sample 대신 우선 사용한다.
    repo는 'owner/repo' 형식이어야 한다 (예: 'langchain-ai/langgraph')."""
    return fetch_repo_structure_text(repo)


@tool
def fetch_repo_file(repo: str, path: str) -> str:
    """GitHub 저장소의 특정 파일 하나의 전체 내용을 가져온다. 사용자가
    fetch_repo_structure로 본 구조 지도에서 특정 파일을 콕 집어 자세히
    보고 싶어할 때(드릴다운) 사용한다.
    repo는 'owner/repo' 형식, path는 저장소 루트 기준 파일 경로다
    (예: 'src/requests/__init__.py')."""
    return fetch_repo_file_text(repo, path)
