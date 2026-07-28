"""
코드/데이터 작업 에이전트의 StateGraph 조립
(dispatcher / 도구 팬아웃 / 투표형 3-way 앙상블 / 결정론적 다수결 구조).

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
voter_1_node   voter_2_node   voter_3_node        (투표형 앙상블 팬아웃 —
   └──────┼───────────────┴──────┘                 셋 다 같은 프롬프트로
          v                                        독립 시도)
   vote_for_best_report                    (팬인: LLM 재호출 없이 결정론적
          |                                 다수결 — 도구 결과를 정확히
          v                                 담은 candidate를 선택)
         END

model 노드 하나 + reporter 노드 하나였던 이전 구조에서, reporter를 3-way
투표형 앙상블로 교체했다:

- dispatcher: tools가 bind_tools된 llm. 여러 도구를 한 번에 요청할 수
  있어야 도구 팬아웃이 실제로 병렬 이득을 본다.
- voter_1/voter_2/voter_3: 도구가 바인딩되지 않은 llm을 공유해서 쓴다.
  **같은 VOTER_SYSTEM_PROMPT로 같은 과제를 독립적으로 3번 시도**한다 —
  이전 버전(관점별 앙상블)과 달리 역할을 나누지 않는다. 다양성은 프롬프트가
  아니라 모델 샘플링 자체의 변동성에서 나온다.
- vote_for_best_report: LLM을 전혀 호출하지 않는 순수 함수. 세 candidate
  중 도구 실행 결과(ToolMessage)를 실제로 정확히 포함한 것을 골라 최종
  답변으로 쓴다. "어느 게 더 그럴듯한가"를 LLM 판사에게 다시 묻는 대신,
  이미 알고 있는 정답과의 문자열 일치로 결정론적으로 판정해서 종합 단계의
  환각 위험을 구조적으로 없앴다.

두 팬인 조인(add_edge(FANOUT_TOOL_NODES, voter_node), add_edge(
FANOUT_VOTE_NODES, "vote_for_best_report"))은 모두 "나열된 노드가 이번
라운드에 모두 실행됐는지"로 트리거되므로, 도구 노드/voter 노드 모두
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
    FANOUT_TOOL_NODES,
    FANOUT_VOTE_NODES,
    VOTER_LABELS,
    VOTER_SYSTEM_PROMPT,
    calculate_node,
    call_dispatcher_model,
    call_report_draft_model,
    read_file_node,
    route_after_dispatcher,
    vote_for_best_report_node,
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
    # voter들은 의도적으로 bind_tools를 거치지 않은 llm을 그대로 쓴다.
    voter_llm = llm

    builder = StateGraph(AgentState)
    builder.add_node("dispatcher", partial(call_dispatcher_model, llm=dispatcher_llm))
    builder.add_node("calculate_node", calculate_node)
    builder.add_node("read_file_node", read_file_node(sandbox_dir))
    builder.add_node("write_file_node", write_file_node(sandbox_dir))
    for label, node_name in zip(VOTER_LABELS, FANOUT_VOTE_NODES):
        builder.add_node(
            node_name,
            partial(
                call_report_draft_model,
                llm=voter_llm,
                system_prompt=VOTER_SYSTEM_PROMPT,
                label=label,
            ),
        )
    builder.add_node("vote_for_best_report", vote_for_best_report_node)

    builder.add_edge(START, "dispatcher")
    builder.add_conditional_edges(
        "dispatcher", route_after_dispatcher, FANOUT_TOOL_NODES + [END]
    )
    # 도구 팬인 -> 투표 팬아웃: 세 도구 노드가 모두 끝난 뒤에만 각 voter
    # 노드가 실행된다. 같은 소스 리스트로 3번 독립 조인을 걸면, 세 voter가
    # 서로 기다리지 않고 함께(병렬로) 트리거된다.
    for node_name in FANOUT_VOTE_NODES:
        builder.add_edge(FANOUT_TOOL_NODES, node_name)
    # 투표 팬인: 세 voter가 모두 끝난 뒤에만 vote_for_best_report가 실행된다.
    builder.add_edge(FANOUT_VOTE_NODES, "vote_for_best_report")
    builder.add_edge("vote_for_best_report", END)

    return builder.compile()
