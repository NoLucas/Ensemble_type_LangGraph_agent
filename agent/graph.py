"""
코드/데이터 작업 에이전트의 StateGraph 조립 (dispatcher/팬아웃/reporter 구조).

        START
          |
          v
      dispatcher                          (첫 입력: 효율적으로 도구 지시)
          |
   (tool_call 있음)          (tool_call 없음)
          |                        |
   ┌──────┼───────────────┐        v
   v      v               v       END    (도구가 필요 없으면 dispatcher의
calculate_node   read_file_node  write_file_node   답변이 곧 최종 답변)
   └──────┼───────────────┴──────┘        (병렬 팬아웃)
          v
       reporter                           (팬인: 결과를 사용자에게 보고)
          |
          v
         END

model 노드 하나를 쓰던 이전 구조와 달리, 첫 입력을 처리하는 dispatcher와
도구 실행 결과를 사용자에게 보고하는 reporter를 분리했다. 역할이 다르므로
llm도 다르게 바인딩한다:

- dispatcher: tools가 bind_tools된 llm. 여러 도구를 한 번에 요청할 수
  있어야 팬아웃이 실제로 병렬 이득을 본다.
- reporter: tools가 바인딩되지 않은 llm. "도구 다시 부르지 마"를 프롬프트
  로만 당부하는 대신, 도구 스키마 자체를 모르게 만들어 구조적으로
  tool_call을 낼 수 없게 한다. 그 결과 dispatcher -> 팬아웃 -> reporter ->
  END는 사이클이 없는 DAG이 되어, 이전에 필요했던 반복 상한(MAX_ITERATIONS)
  같은 무한 루프 방지 장치가 아예 필요 없어진다.

llm은 항상 함수 인자로 주입한다. build_graph() 안에서 전역으로 실제
ChatModel을 인스턴스화하지 않기 때문에, 테스트에서는 FakeChatModel을,
운영 코드(main.py/app.py 등)에서는 ChatAnthropic을 그대로 넣어 재사용할
수 있다.
"""

from functools import partial
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    FANOUT_TOOL_NODES,
    calculate_node,
    call_dispatcher_model,
    call_reporter_model,
    read_file_node,
    route_after_dispatcher,
    write_file_node,
)
from agent.state import AgentState
from agent.tools import calculate, make_read_text_file_tool, make_write_text_file_tool

DEFAULT_SANDBOX_DIR = Path(__file__).parent / "sandbox_data"


def build_graph(llm, sandbox_dir: Path = DEFAULT_SANDBOX_DIR):
    tools = [
        calculate,
        make_read_text_file_tool(sandbox_dir),
        make_write_text_file_tool(sandbox_dir),
    ]
    dispatcher_llm = llm.bind_tools(tools)
    # reporter는 의도적으로 bind_tools를 거치지 않은 llm을 그대로 쓴다.
    reporter_llm = llm

    builder = StateGraph(AgentState)
    builder.add_node("dispatcher", partial(call_dispatcher_model, llm=dispatcher_llm))
    builder.add_node("calculate_node", calculate_node)
    builder.add_node("read_file_node", read_file_node(sandbox_dir))
    builder.add_node("write_file_node", write_file_node(sandbox_dir))
    builder.add_node("reporter", partial(call_reporter_model, llm=reporter_llm))

    builder.add_edge(START, "dispatcher")
    builder.add_conditional_edges(
        "dispatcher", route_after_dispatcher, FANOUT_TOOL_NODES + [END]
    )
    # 팬인 조인: 세 도구 노드가 모두 실행된 뒤에만 reporter가 실행된다.
    # route_after_dispatcher가 셋을 항상 함께 팬아웃하므로 성립한다.
    builder.add_edge(FANOUT_TOOL_NODES, "reporter")
    builder.add_edge("reporter", END)

    return builder.compile()
