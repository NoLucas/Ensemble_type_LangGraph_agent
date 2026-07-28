"""
코드/데이터 작업 에이전트의 StateGraph 조립
(dispatcher / 도구 팬아웃 / 3-way 보고서 앙상블 / 결정론적 종합 구조).

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
   └──────┼───────────────┴──────┘        (도구 팬아웃)
          v  (도구 팬인, 그대로 3-way로 재팬아웃)
   ┌──────┼───────────────┐
   v      v               v
draft_concise   draft_detailed   draft_action        (보고서 앙상블 팬아웃)
   └──────┼───────────────┴──────┘
          v
   aggregate_reports                      (팬인: LLM 재호출 없이 결정론적 종합)
          |
          v
         END

model 노드 하나 + reporter 노드 하나였던 이전 구조에서, reporter를 3-way
앙상블로 교체했다:

- dispatcher: tools가 bind_tools된 llm. 여러 도구를 한 번에 요청할 수
  있어야 도구 팬아웃이 실제로 병렬 이득을 본다.
- draft_concise/draft_detailed/draft_action: 도구가 바인딩되지 않은 llm을
  공유해서 쓴다. 같은 도구 실행 결과를 세 가지 다른 관점(REPORT_ANGLES)으로
  각자 독립적으로 서술한다 — 정답이 하나뿐인 사실(계산값, 파일 내용)을
  다루므로 무작위성(temperature)이 아니라 관점(프롬프트)으로만 다양성을 준다.
- aggregate_reports: LLM을 전혀 호출하지 않는 순수 함수. 세 draft를 고정된
  순서로 이어붙이기만 한다. "종합" 단계에 LLM을 또 넣으면 그 단계 자체가
  draft에 없던 내용을 지어내는 환각 진입점이 되므로, 아예 LLM을 빼서
  구조적으로 그 위험을 없앴다.

두 팬인 조인(add_edge(FANOUT_TOOL_NODES, draft_node), add_edge(
FANOUT_REPORT_DRAFT_NODES, "aggregate_reports"))은 모두 "나열된 노드가
이번 라운드에 모두 실행됐는지"로 트리거되므로, 도구 노드/draft 노드 모두
tool_call이 하나만 왔어도 항상 셋 다 함께 깨운다.

llm은 항상 함수 인자로 주입한다. build_graph() 안에서 전역으로 실제
ChatModel을 인스턴스화하지 않기 때문에, 테스트에서는 FakeChatModel을,
운영 코드(main.py/app.py 등)에서는 ChatAnthropic을 그대로 넣어 재사용할
수 있다.
"""

from functools import partial
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    FANOUT_REPORT_DRAFT_NODES,
    FANOUT_TOOL_NODES,
    REPORT_ANGLES,
    aggregate_reports_node,
    calculate_node,
    call_dispatcher_model,
    call_report_draft_model,
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
    # draft 노드들은 의도적으로 bind_tools를 거치지 않은 llm을 그대로 쓴다.
    draft_llm = llm

    builder = StateGraph(AgentState)
    builder.add_node("dispatcher", partial(call_dispatcher_model, llm=dispatcher_llm))
    builder.add_node("calculate_node", calculate_node)
    builder.add_node("read_file_node", read_file_node(sandbox_dir))
    builder.add_node("write_file_node", write_file_node(sandbox_dir))
    for angle in REPORT_ANGLES:
        builder.add_node(
            angle["node_name"],
            partial(
                call_report_draft_model,
                llm=draft_llm,
                system_prompt=angle["system_prompt"],
                label=angle["key"],
            ),
        )
    builder.add_node("aggregate_reports", aggregate_reports_node)

    builder.add_edge(START, "dispatcher")
    builder.add_conditional_edges(
        "dispatcher", route_after_dispatcher, FANOUT_TOOL_NODES + [END]
    )
    # 도구 팬인 -> 보고서 팬아웃: 세 도구 노드가 모두 끝난 뒤에만 각 draft
    # 노드가 실행된다. 같은 소스 리스트로 3번 독립 조인을 걸면, 세 draft가
    # 서로 기다리지 않고 함께(병렬로) 트리거된다.
    for node_name in FANOUT_REPORT_DRAFT_NODES:
        builder.add_edge(FANOUT_TOOL_NODES, node_name)
    # 보고서 팬인: 세 draft가 모두 끝난 뒤에만 aggregate_reports가 실행된다.
    builder.add_edge(FANOUT_REPORT_DRAFT_NODES, "aggregate_reports")
    builder.add_edge("aggregate_reports", END)

    return builder.compile()
