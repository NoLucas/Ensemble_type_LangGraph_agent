"""
GitHub 저장소 리뷰/학습 에이전트의 StateGraph 조립
(dispatcher / 도구 팬아웃 / 투표형 N-way 앙상블 / 결정론적 다수결 구조).

        START
          |
          v
      dispatcher                          (첫 입력: 언급된 저장소에 맞는
          |                                 도구를 효율적으로 지시)
   (tool_call 있음)          (tool_call 없음)
          |                        |
   ┌──────┼──────┬──────┐         v
   v      v      v      v        END    (도구가 필요 없으면 dispatcher의
repo_overview  repo_source  repo_structure  repo_file    답변이 곧 최종 답변)
_node          _node        _node           _node
   └──────┼──────┴──────┘        (도구 팬아웃, 4-way)
          v  (도구 팬인, 그대로 N-way로 재팬아웃 — 기본 3, 스터디 모드는 1)
   ┌──────┼───────────────┐
   v      v               v
voter_1_node   voter_2_node   voter_3_node        (투표형 앙상블 팬아웃 —
   └──────┼───────────────┴──────┘                 다 같은 프롬프트로
          v                                        독립 시도)
   vote_for_best_report                    (팬인: LLM 재호출 없이 결정론적
          |                                 다수결 — 도구 결과를 정확히
          v                                 담은 candidate를 선택)
         END

- dispatcher: tools가 bind_tools된 llm. 사용자가 언급한 저장소마다 필요한
  도구를 한 번에 요청할 수 있어야 도구 팬아웃이 실제로 병렬 이득을 본다.
  도구 네 개(개요/대표 소스/구조 지도/특정 파일) 중 사용자 의도("정식
  리뷰"인지 "가볍게 훑어보기+드릴다운"인지)에 맞는 조합을 고른다.
- voter_1..voter_N (기본 N=3, num_voters로 조절): 도구가 바인딩되지 않은
  llm을 공유해서 쓴다. **같은 VOTER_SYSTEM_PROMPT로 같은 과제를 독립적으로
  N번 시도**한다. 다양성은 프롬프트가 아니라 모델 샘플링 자체의 변동성에서
  나온다. num_voters=1이면 다수결 없이 그 한 번의 시도가 그대로 채택되고,
  LLM 호출이 4번(dispatcher+voter 3)에서 2번(dispatcher+voter 1)으로
  줄어든다 — "스터디 모드": 무료 티어처럼 토큰이 빠듯할 때 정확도보다
  비용을 우선하는 선택지다.
- vote_for_best_report: LLM을 전혀 호출하지 않는 순수 함수. candidate 중
  도구 실행 결과(ToolMessage: 저장소 개요/구조/소스/특정 파일)를 실제로
  정확히 포함한 것을 골라 최종 답변으로 쓴다. "어느 게 더 그럴듯한가"를
  LLM 판사에게 다시 묻는 대신, 이미 알고 있는 정답에서 뽑아낸 토큰과의
  겹침으로 결정론적으로 판정해서 종합 단계의 환각 위험을 구조적으로
  없앴다. num_voters=1이어도 그대로 동작한다(단일 candidate를 그대로
  채택).

두 팬인 조인(add_edge(FANOUT_TOOL_NODES, voter_node), add_edge(
voter_fanout_nodes, "vote_for_best_report"))은 모두 "나열된 노드가 이번
라운드에 모두 실행됐는지"로 트리거되므로, 도구 노드/voter 노드 모두
tool_call이 하나만 왔어도 항상 함께 깨운다.

llm은 항상 함수 인자로 주입한다. build_graph() 안에서 전역으로 실제
ChatModel을 인스턴스화하지 않기 때문에, 테스트에서는 FakeChatModel을,
운영 코드(main.py/app.py 등)에서는 ChatAnthropic을 그대로 넣어 재사용할
수 있다.
"""

from functools import partial

from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    FANOUT_TOOL_NODES,
    VOTER_LABELS,
    VOTER_SYSTEM_PROMPT,
    call_dispatcher_model,
    call_report_draft_model,
    repo_file_node,
    repo_overview_node,
    repo_source_node,
    repo_structure_node,
    route_after_dispatcher,
    vote_for_best_report_node,
)
from agent.state import AgentState
from agent.tools import (
    fetch_repo_file,
    fetch_repo_overview,
    fetch_repo_source_sample,
    fetch_repo_structure,
)


def build_graph(llm, num_voters: int = 3):
    """그래프를 조립한다.

    num_voters: 투표형 앙상블에 쓸 voter 개수(1~len(VOTER_LABELS), 기본 3).
    1로 낮추면 다수결 검증 없이 단일 시도를 그대로 채택하는 대신 LLM 호출이
    거의 절반으로 줄어든다 — 무료 티어처럼 토큰이 빠듯한 "스터디 모드"용
    설정이다.
    """
    if not 1 <= num_voters <= len(VOTER_LABELS):
        raise ValueError(f"num_voters는 1~{len(VOTER_LABELS)} 사이여야 합니다 (받은 값: {num_voters}).")

    voter_labels = VOTER_LABELS[:num_voters]
    voter_fanout_nodes = [f"{label}_node" for label in voter_labels]

    tools = [fetch_repo_overview, fetch_repo_source_sample, fetch_repo_structure, fetch_repo_file]
    dispatcher_llm = llm.bind_tools(tools)
    # voter들은 의도적으로 bind_tools를 거치지 않은 llm을 그대로 쓴다.
    voter_llm = llm

    builder = StateGraph(AgentState)
    builder.add_node("dispatcher", partial(call_dispatcher_model, llm=dispatcher_llm))
    builder.add_node("repo_overview_node", repo_overview_node)
    builder.add_node("repo_source_node", repo_source_node)
    builder.add_node("repo_structure_node", repo_structure_node)
    builder.add_node("repo_file_node", repo_file_node)
    for label, node_name in zip(voter_labels, voter_fanout_nodes):
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
    # 도구 팬인 -> 투표 팬아웃: 도구 노드 네 개가 모두 끝난 뒤에만 각 voter
    # 노드가 실행된다. 같은 소스 리스트로 N번 독립 조인을 걸면, voter들이
    # 서로 기다리지 않고 함께(병렬로) 트리거된다.
    for node_name in voter_fanout_nodes:
        builder.add_edge(FANOUT_TOOL_NODES, node_name)
    # 투표 팬인: voter가 모두 끝난 뒤에만 vote_for_best_report가 실행된다.
    builder.add_edge(voter_fanout_nodes, "vote_for_best_report")
    builder.add_edge("vote_for_best_report", END)

    return builder.compile()
