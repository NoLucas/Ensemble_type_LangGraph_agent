"""
도구(tools) 단위 테스트.

각 도구는 (1) 정상 케이스와 (2) 거부/실패 케이스를 반드시 가진다.
LLM이 임의의 문자열을 도구에 넘길 수 있다고 가정하고, 위험한 입력이
와도 예외로 프로세스를 죽이지 않고 에러 문자열을 반환하는지를 검증한다
(도구는 에이전트 그래프 안에서 호출되므로, 여기서 예외가 새면 그래프
전체가 죽는다).
"""

import pytest

from agent.tools import calculate, read_text_file, write_text_file


# ---------------------------------------------------------------------------
# calculate: 안전한 산술 계산 (eval을 쓰지 않고 ast로 화이트리스트 검증)
# ---------------------------------------------------------------------------


def test_calculate_evaluates_basic_arithmetic():
    assert calculate.invoke({"expression": "1 + 2 * 3"}) == "7"


def test_calculate_supports_parentheses_and_negative_numbers():
    assert calculate.invoke({"expression": "-(2 + 3) * 4"}) == "-20"


def test_calculate_supports_division():
    assert calculate.invoke({"expression": "7 / 2"}) == "3.5"


def test_calculate_rejects_function_calls():
    result = calculate.invoke({"expression": "__import__('os').system('echo hi')"})
    assert result.startswith("Error")


def test_calculate_rejects_non_arithmetic_names():
    result = calculate.invoke({"expression": "os.system('echo hi')"})
    assert result.startswith("Error")


def test_calculate_rejects_malformed_expression():
    result = calculate.invoke({"expression": "1 + "})
    assert result.startswith("Error")


def test_calculate_rejects_division_by_zero():
    result = calculate.invoke({"expression": "1 / 0"})
    assert result.startswith("Error")


# ---------------------------------------------------------------------------
# read_text_file: 샌드박스 디렉토리 밖은 절대 읽을 수 없어야 한다
# ---------------------------------------------------------------------------


def test_read_text_file_returns_contents(sandbox_dir):
    target = sandbox_dir / "notes.txt"
    target.write_text("hello sandbox", encoding="utf-8")

    result = read_text_file("notes.txt", base_dir=sandbox_dir)

    assert result == "hello sandbox"


def test_read_text_file_rejects_path_traversal(sandbox_dir, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret", encoding="utf-8")

    result = read_text_file("../secret.txt", base_dir=sandbox_dir)

    assert result.startswith("Error")
    assert "top secret" not in result


def test_read_text_file_rejects_absolute_path(sandbox_dir, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret", encoding="utf-8")

    result = read_text_file(str(secret), base_dir=sandbox_dir)

    assert result.startswith("Error")
    assert "top secret" not in result


def test_read_text_file_missing_file_returns_error_not_exception(sandbox_dir):
    result = read_text_file("does_not_exist.txt", base_dir=sandbox_dir)

    assert result.startswith("Error")


# ---------------------------------------------------------------------------
# write_text_file: 샌드박스 디렉토리 밖에는 절대 쓸 수 없어야 한다
# ---------------------------------------------------------------------------


def test_write_text_file_creates_new_file(sandbox_dir):
    result = write_text_file("output.txt", "hello world", base_dir=sandbox_dir)

    assert result.startswith("OK")
    assert (sandbox_dir / "output.txt").read_text(encoding="utf-8") == "hello world"


def test_write_text_file_overwrites_existing_file(sandbox_dir):
    target = sandbox_dir / "existing.txt"
    target.write_text("old content", encoding="utf-8")

    result = write_text_file("existing.txt", "new content", base_dir=sandbox_dir)

    assert result.startswith("OK")
    assert target.read_text(encoding="utf-8") == "new content"


def test_write_text_file_rejects_path_traversal(sandbox_dir, tmp_path):
    result = write_text_file("../escape.txt", "malicious", base_dir=sandbox_dir)

    assert result.startswith("Error")
    assert not (tmp_path / "escape.txt").exists()


def test_write_text_file_rejects_absolute_path(sandbox_dir, tmp_path):
    outside = tmp_path / "outside.txt"

    result = write_text_file(str(outside), "malicious", base_dir=sandbox_dir)

    assert result.startswith("Error")
    assert not outside.exists()


def test_write_text_file_rejects_writing_outside_via_nested_traversal(sandbox_dir, tmp_path):
    nested = sandbox_dir / "sub"
    nested.mkdir()

    result = write_text_file("sub/../../escape.txt", "malicious", base_dir=sandbox_dir)

    assert result.startswith("Error")
    assert not (tmp_path / "escape.txt").exists()
