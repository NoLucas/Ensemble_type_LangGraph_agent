"""
코드/데이터 작업 에이전트의 StateGraph 조립.

model 노드가 tool_calls를 반환하면 tools 노드로, 아니면 END로 가는
단순한 model <-> tools 순환 구조다 (langgraph.org의 표준 ReAct 패턴).
반복 상한은 route_after_model(agent/nodes.py)이 강제한다.

llm은 항상 함수 인자로 주입한다. build_graph() 안에서 전역으로 실제
ChatModel을 인스턴스화하지 않기 때문에, 테스트에서는 FakeChatModel을,
운영 코드(main.py/app.py 등)에서는 ChatAnthropic을 그대로 넣어 재사용할
수 있다.
"""

from functools import partial
from pathlib import Path

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.nodes import call_model, route_after_model
from agent.state import AgentState
from agent.tools import calculate, make_read_text_file_tool

DEFAULT_SANDBOX_DIR = Path(__file__).parent / "sandbox_data"


def build_graph(llm, sandbox_dir: Path = DEFAULT_SANDBOX_DIR):
    tools = [calculate, make_read_text_file_tool(sandbox_dir)]
    bound_llm = llm.bind_tools(tools)

    builder = StateGraph(AgentState)
    builder.add_node("model", partial(call_model, llm=bound_llm))
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", route_after_model, ["tools", END])
    builder.add_edge("tools", "model")

    return builder.compile()
