"""
코드/데이터 작업 에이전트의 StateGraph 조립 (팬아웃/팬인 구조).

        START
          |
          v
        model
          |
   (tool_call 있음)
          |
   ┌──────┴──────────────┐
   v                      v
calculate_node      read_file_node        (병렬 팬아웃)
   └──────┬──────────────┘
          v
        model                             (팬인)
          |
     (tool_call 없음)
          v
         END

model이 tool_call을 반환하면 calculate_node와 read_file_node를 항상 함께
팬아웃한다. 각 노드는 자기 담당 tool_call이 없으면 조용히 통과하고, 결과는
add_edge(["calculate_node", "read_file_node"], "model")로 다시 model로
팬인한다. tool_call이 하나뿐이어도 두 노드를 함께 깨우는 이유는, 이 팬인
조인이 "나열된 두 노드가 이번 라운드에 모두 실행됐는지"로 트리거되기
때문이다 — 한쪽만 조건부로 건너뛰면 조인이 영원히 기다리게 된다.

llm은 항상 함수 인자로 주입한다. build_graph() 안에서 전역으로 실제
ChatModel을 인스턴스화하지 않기 때문에, 테스트에서는 FakeChatModel을,
운영 코드(main.py/app.py 등)에서는 ChatAnthropic을 그대로 넣어 재사용할
수 있다.
"""

from functools import partial
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from agent.nodes import calculate_node, call_model, read_file_node, route_after_model
from agent.state import AgentState
from agent.tools import calculate, make_read_text_file_tool

DEFAULT_SANDBOX_DIR = Path(__file__).parent / "sandbox_data"


def build_graph(llm, sandbox_dir: Path = DEFAULT_SANDBOX_DIR):
    tools = [calculate, make_read_text_file_tool(sandbox_dir)]
    bound_llm = llm.bind_tools(tools)

    builder = StateGraph(AgentState)
    builder.add_node("model", partial(call_model, llm=bound_llm))
    builder.add_node("calculate_node", calculate_node)
    builder.add_node("read_file_node", read_file_node(sandbox_dir))

    builder.add_edge(START, "model")
    builder.add_conditional_edges(
        "model", route_after_model, ["calculate_node", "read_file_node", END]
    )
    # 팬인 조인: calculate_node와 read_file_node가 둘 다 실행된 뒤에만 model이
    # 다시 실행된다. route_after_model이 둘을 항상 함께 팬아웃하므로 성립한다.
    builder.add_edge(["calculate_node", "read_file_node"], "model")

    return builder.compile()
