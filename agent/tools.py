"""
에이전트가 사용할 도구(tool) 모음.

두 가지 원칙을 지킨다:
1. eval()/exec()로 임의 코드를 실행하지 않는다. calculate는 ast를 직접 순회하며
   허용된 연산자(+ - * / 단항부호, 괄호)만 계산하는 화이트리스트 방식이다.
2. 파일 접근은 지정된 base_dir 밖으로 절대 나갈 수 없다. "../" 상대경로 탈출과
   절대경로 지정을 모두 resolve() 후 base_dir 하위인지 검사해서 막는다.

도구는 예외를 던지지 않고 "Error: ..." 문자열을 반환한다. 그래프 안에서
ToolNode가 도구를 호출하는데, 여기서 예외가 새면 그래프 실행 전체가 죽기
때문에 실패도 반드시 정상적인 반환값으로 표현해야 한다.
"""

import ast
import operator
from pathlib import Path

from langchain_core.tools import tool

# calculate가 허용하는 연산자만 화이트리스트로 등록한다.
# 여기 없는 ast 노드(Call, Name, Attribute 등)는 전부 거부된다.
_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_ast_node(node: ast.AST):
    if isinstance(node, ast.Expression):
        return _eval_ast_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _eval_ast_node(node.left)
        right = _eval_ast_node(node.right)
        return _ALLOWED_BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        operand = _eval_ast_node(node.operand)
        return _ALLOWED_UNARYOPS[type(node.op)](operand)
    raise ValueError(f"허용되지 않은 표현식입니다: {ast.dump(node)}")


def _format_number(value: float | int) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


@tool
def calculate(expression: str) -> str:
    """사칙연산(+, -, *, /)과 괄호로 이루어진 산술식을 계산한다.
    변수, 함수 호출, import 등은 지원하지 않으며 순수 숫자 계산에만 사용한다.
    """
    try:
        parsed = ast.parse(expression, mode="eval")
        result = _eval_ast_node(parsed)
    except ZeroDivisionError:
        return "Error: 0으로 나눌 수 없습니다."
    except (SyntaxError, ValueError, TypeError) as exc:
        return f"Error: 계산할 수 없는 표현식입니다 ({exc})"
    return _format_number(result)


def read_text_file(filename: str, base_dir: Path) -> str:
    """base_dir 하위의 텍스트 파일만 읽는다. 경로 탈출은 전부 거부한다."""
    base_dir = Path(base_dir).resolve()
    target = (base_dir / filename).resolve()

    if base_dir not in target.parents and target != base_dir:
        return "Error: 허용된 디렉토리 밖의 경로는 읽을 수 없습니다."
    if not target.is_file():
        return f"Error: 파일을 찾을 수 없습니다 ({filename})."

    try:
        return target.read_text(encoding="utf-8")
    except OSError as exc:
        return f"Error: 파일을 읽는 중 오류가 발생했습니다 ({exc})"


def make_read_text_file_tool(base_dir: Path):
    """지정된 base_dir에 고정된 read_text_file @tool을 만든다.

    read_text_file 자체는 base_dir을 인자로 받는 순수 함수라 테스트하기
    쉽지만, LLM이 호출하는 도구는 filename만 인자로 받아야 하므로(도구
    스키마가 LLM에게 그대로 노출된다) base_dir을 클로저로 고정한 래퍼가
    필요하다.
    """

    @tool
    def read_sandbox_file(filename: str) -> str:
        """샌드박스 디렉토리 안에 있는 텍스트 파일의 내용을 읽는다."""
        return read_text_file(filename, base_dir)

    return read_sandbox_file
