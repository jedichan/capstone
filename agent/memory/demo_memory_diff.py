"""
Visual comparison: agent with vs without short-term memory.

Simulates 3 conversational turns and shows exactly what the agent
"sees" as input (messages list + summary_node prompt) in both modes.
"""

import os
import sys
from textwrap import indent

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage, AIMessage


SEP = "=" * 72
THIN_SEP = "-" * 72


def _make_box(title: str, content: str, color: str = "") -> str:
    """Wrap content in a titled box."""
    inner = indent(content, "  | ")
    return f"  +-- {title}\n{inner}\n  +{'--' * 35}"


def show_memory_module_state(flag_enabled: bool):
    """Show memory module internal state."""
    import memory.short_term as mem
    thread_id, user_id = "demo-thread", "demo-user"
    mem.clear_session(thread_id, user_id)

    if not flag_enabled:
        os.environ["AGENT_ENABLE_SHORT_TERM_MEMORY"] = "0"
    else:
        os.environ["AGENT_ENABLE_SHORT_TERM_MEMORY"] = "1"

    print(f"\n  +-- Short-Term Memory Module State")
    print(f"  |  AGENT_ENABLE_SHORT_TERM_MEMORY = {flag_enabled}")
    print(f"  |  AGENT_MEMORY_MAX_TURNS = 5")
    print(f"  |  thread_id = {thread_id}")
    print(f"  +{'--' * 35}")

    return thread_id, user_id


def simulate_turn(
    thread_id: str, user_id: str, turn_num: int, question: str, answer: str, flag_enabled: bool
) -> None:
    """Simulate one conversation turn. Show what the agent sees."""
    import memory.short_term as mem

    if not flag_enabled:
        os.environ["AGENT_ENABLE_SHORT_TERM_MEMORY"] = "0"
    else:
        os.environ["AGENT_ENABLE_SHORT_TERM_MEMORY"] = "1"

    # --- 1. What the graph receives as input ---
    history_turns = mem.get_history(thread_id, user_id)
    raw_messages = []
    for turn in history_turns:
        raw_messages.append(HumanMessage(content=str(turn["query"])))
        raw_messages.append(AIMessage(content=str(turn["answer"])))
    raw_messages.append(HumanMessage(content=question))

    print(f"\n  +-- Turn {turn_num} -- Graph Input (messages list)")
    print(f"  |  Memory mode: {'ON  (check)' if flag_enabled else 'OFF (x)'}")
    print(f"  |  Previous turns in history: {len(history_turns)}")
    print(f"  |  Messages sent to graph.stream():")
    for i, msg in enumerate(raw_messages):
        role = "USER " if isinstance(msg, HumanMessage) else "ASSISTANT"
        prefix = "     [HISTORY]" if i < len(raw_messages) - 1 else "     [NOW]    "
        print(f"  |  {prefix} [{role}] {msg.content}")
    print(f"  +{'--' * 35}")

    # --- 2. What the summary_node LLM prompt looks like ---
    history_block = mem.format_history_for_prompt(thread_id, user_id) if flag_enabled else ""

    # Build the same prompt structure as summary_node would
    prompt_lines = [
        "You are the final response summarizer for a retrieval assistant.",
        "Return a short factual answer that matches the reference-answer style used in evaluation.",
        "Use only the single strongest result. Do not merge multiple videos.",
        "Even if the evidence is partial, summarize using the strongest result.",
        "Do not return 'No matching clip is expected.' when results are provided.",
        "Do not add extra scene details, explanations, or a sources section.",
    ]
    if flag_enabled and history_block:
        prompt_lines.extend(["", history_block])
    prompt_lines.extend([
        "",
        f"Original user query: {question}",
        "Rewritten retrieval query: (rewritten by self_query_node)",
        "Retrieved result count: 1",
        "Top results: [{'video_id': 'demo', 'summary': '...'}]",
        f"Preferred fallback answer: {answer}",
        "Draft answer: (draft from answer_node)",
    ])

    print(f"\n  +-- Turn {turn_num} -- Summary Node LLM Prompt")
    print(f"  |  Memory mode: {'ON  (check)' if flag_enabled else 'OFF (x)'}")
    if flag_enabled and history_block:
        print(f"  |  (up) Conversation history block INJECTED into prompt")
    else:
        print(f"  |  (x) NO conversation history in prompt")
    for line in prompt_lines:
        print(f"  |  {line}")
    print(f"  +{'--' * 35}")

    # --- 3. Store this turn if memory is enabled ---
    mem.add_turn(thread_id, user_id, query=question, answer=answer)


def main():
    # Simulated conversation: 3 related questions about a video
    turns = [
        ("Turn 1: What time did the car enter the parking lot?", "Yes. The relevant clip is in video001, around 0:00:05 - 0:00:12."),
        ("Turn 2: What color was it?", "The most relevant clip is in video001, around 0:00:05 - 0:00:12."),
        ("Turn 3: Was it moving?", "Yes. The relevant clip is in video001, around 0:00:05 - 0:00:12."),
    ]

    print()
    print(("=" * 72))
    print("  SHORT-TERM MEMORY -- Visual Comparison: ON vs OFF")
    print("  Scenario: 3 related conversational queries about a video")
    print(("=" * 72))

    # ========== MODE 1: WITHOUT MEMORY ==========
    print(f"\n{SEP}")
    print(f"  MODE 1: MEMORY DISABLED  (AGENT_ENABLE_SHORT_TERM_MEMORY=0)")
    print(f"  Each query is independent -- the agent has NO context from prior turns")
    print(f"{SEP}")

    tid1, uid1 = show_memory_module_state(flag_enabled=False)
    for i, (q, a) in enumerate(turns, 1):
        simulate_turn(tid1, uid1, i, q, a, flag_enabled=False)
        if i < len(turns):
            print(f"\n  {'·' * 40}  (next query -- state RESET, no history)  {'·' * 20}")

    # ========== MODE 2: WITH MEMORY ==========
    print(f"\n\n{SEP}")
    print(f"  MODE 2: MEMORY ENABLED  (AGENT_ENABLE_SHORT_TERM_MEMORY=1)")
    print(f"  Previous Q&A pairs are prepended to each new query's context")
    print(f"{SEP}")

    tid2, uid2 = show_memory_module_state(flag_enabled=True)
    for i, (q, a) in enumerate(turns, 1):
        simulate_turn(tid2, uid2, i, q, a, flag_enabled=True)
        if i < len(turns):
            print(f"\n  {'·' * 20}  (history ACCUMULATES -- next turn has {i} prior turn(s))  {'·' * 20}")

    # ========== SIDE-BY-SIDE SUMMARY TABLE ==========
    print(f"\n\n{SEP}")
    print(f"  SIDE-BY-SIDE SUMMARY")
    print(f"{SEP}")

    print("""
  +----------------------------------+----------------------------------+
  |   WITHOUT MEMORY (flag=0)        |   WITH MEMORY (flag=1)           |
  +----------------------------------+----------------------------------+
  |  Each graph.stream() gets        |  Each graph.stream() gets        |
  |  exactly 1 HumanMessage:         |  N history turns + 1 new msg:    |
  |                                  |                                  |
  |  Turn 1: [Human] Q1              |  Turn 1: [Human] Q1              |
  |  Turn 2: [Human] Q2              |  Turn 2: [Human] Q1, [AI] A1,   |
  |  Turn 3: [Human] Q3              |           [Human] Q2              |
  |                                  |  Turn 3: [Human] Q1, [AI] A1,   |
  |                                  |           [Human] Q2, [AI] A2,   |
  |                                  |           [Human] Q3              |
  +----------------------------------+----------------------------------+
  |  Summary Node LLM Prompt:        |  Summary Node LLM Prompt:        |
  |  - NO conversation history       |  - INCLUDES "Conversation        |
  |  - Only sees current query       |    history (previous turns)"     |
  |  - Cannot reference prior Q&A    |  - Can reference prior answers   |
  |    ("What color was it?" goes    |    ("What color was it?" goes    |
  |     no idea what "it" is)       |     knows "it" = the car)       |
  +----------------------------------+----------------------------------+
""")

    print(f"  KEY TAKEAWAY:")
    print(f"  - Memory OFF: agent has no context, each query is isolated")
    print(f"  - Memory ON:  agent sees full conversation, can disambiguate pronouns,")
    print(f"    follow up on previous answers, and maintain coherent dialogue")
    print(f"  - Tests are safe: AGENT_ENABLE_SHORT_TERM_MEMORY defaults to 0")
    print(f"  - All intermediate retrieval results (sql/hybrid/rerank) are discarded")
    print(f"  - Only final summary_node output is stored per turn")
    print()


if __name__ == "__main__":
    main()
