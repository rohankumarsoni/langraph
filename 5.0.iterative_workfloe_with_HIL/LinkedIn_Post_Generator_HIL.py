"""
LinkedIn Post Generator: a writer + reviewer agent with a HUMAN IN THE LOOP.

WHAT IT DOES
------------
You give it a topic. A *writer* LLM drafts a LinkedIn post (searching the web
with Tavily if it needs fresh facts). A separate *reviewer* LLM grades the
draft and the writer retries until the AI reviewer is happy (or MAX_ATTEMPTS
is reached).

THEN THE WORKFLOW PAUSES and shows the draft to YOU (the human editor).
You can:
    approve -> the post is final
    revise  -> you type feedback, the writer rewrites, the AI reviewer checks it,
               and you get to look at it again
    quit    -> stop now; the latest draft is returned but NOT marked approved

Nothing is "published" without a human decision. That is the point of the pause.

WORKFLOW (read this first, the code below follows it top to bottom)
-------------------------------------------------------------------

    START
      │
      ▼
 ┌─────────┐   wants to search?   ┌───────┐
 │ writer  │ ───────── yes ─────► │ tools │  (Tavily web search)
 │         │ ◄──────────────────── │       │  results go back to the writer
 └─────────┘                       └───────┘
      │ no (writer produced final text)
      ▼
 ┌───────────────┐
 │ extract_draft │  copies the writer's final text into state["draft"]
 └───────────────┘
      │
      ▼
 ┌──────────┐   AI rejected and attempts remain
 │ reviewer │ ──────────────────────────────────► back to writer
 └──────────┘
      │ AI approved, or out of attempts
      ▼
 ┌──────────────┐   approve ─────────────────────► END
 │ human_review │   quit    ─────────────────────► END
 │  (PAUSES)    │   revise + your feedback ──────► back to writer
 └──────────────┘

HOW THE "PAUSE" WORKS (LangGraph human-in-the-loop)
---------------------------------------------------
1. Inside `human_review_node` we call `interrupt(...)`. LangGraph SAVES the
   whole state with a *checkpointer* and stops the run.
2. `app.invoke(...)` returns, with the question for the human under the
   "__interrupt__" key.
3. We collect the human's answer and call `app.invoke(Command(resume=answer))`
   with the SAME thread_id. LangGraph reloads the saved state and continues.

TERMINOLOGY
-----------
* "attempt"     = one full draft (writer -> [tools -> writer ...] -> reviewer).
* "tool round"  = one trip writer -> tools -> writer inside a single attempt.
* "thread_id"   = the ID of one saved run, so a paused run can be resumed.

SETUP
-----
    pip install "langgraph>=0.4" langchain-openai langchain-groq langchain-tavily python-dotenv pydantic

    # .env file (never commit this file)
    OPENAI_API_KEY=...
    GROQ_API_KEY=...
    TAVILY_API_KEY=...

"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from functools import lru_cache
from typing import Annotated, Any, Callable, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

# =============================================================================
# 1. CONFIGURATION
#    Every tunable value lives here so nothing is "hidden" deep in the code.
# =============================================================================

WRITER_MODEL = "gpt-4o-mini"            # drafts the post (OpenAI)
REVIEWER_MODEL = "openai/gpt-oss-120b"  # grades the post (Groq)

WRITER_TEMPERATURE = 0.7    # higher = more creative writing
REVIEWER_TEMPERATURE = 0.2  # lower  = more consistent, strict grading

# How many drafts the AI may write before it must hand over to the human.
# Every time the human asks for a revision, the AI gets this many NEW attempts.
MAX_ATTEMPTS = 3
SEARCH_MAX_RESULTS = 3      # web results Tavily returns per search

LLM_TIMEOUT_SECONDS = 60    # fail instead of hanging forever on a slow API
LLM_MAX_RETRIES = 2         # automatic retries on transient API errors

# Safety net: LangGraph raises an error if one run takes more steps than this.
# It protects us from infinite loops (e.g. the writer searching forever).
# Rough maths: 1 attempt = ~3 steps + 2 per tool round. 40 is generous.
RECURSION_LIMIT = 40

# Keys that must exist in the environment (or in .env) before we start.
REQUIRED_ENV_VARS = ("OPENAI_API_KEY", "GROQ_API_KEY", "TAVILY_API_KEY")

logger = logging.getLogger("linkedin_agent")


# =============================================================================
# 2. STATE
#    The "shared notebook" every node reads from and writes to. Each node
#    returns a dict with only the keys it wants to UPDATE; LangGraph merges it.
# =============================================================================

class State(TypedDict):
    topic: str                                # what the post is about (input)
    messages: Annotated[list, add_messages]   # writer conversation; add_messages
                                              # APPENDS instead of overwriting
    draft: str                                # latest post text
    review_feedback: str                      # AI reviewer's explanation
    is_approved: bool                         # did the AI reviewer approve?
    attempt: int                              # how many drafts started so far
    attempt_limit: int                        # AI must hand over to the human when
                                              # attempt reaches this number
    human_feedback: str                       # latest change request from the human
    human_action: str                         # "", "approve", "revise" or "quit"


def make_initial_state(topic: str) -> State:
    """Builds the starting state for a fresh run."""
    return {
        "topic": topic,
        "messages": [],
        "draft": "",
        "review_feedback": "",
        "is_approved": False,
        "attempt": 0,
        "attempt_limit": MAX_ATTEMPTS,
        "human_feedback": "",
        "human_action": "",
    }


class ReviewResult(BaseModel):
    """
    Schema the reviewer LLM must answer with.

    Using structured output (instead of parsing free text for the word
    "APPROVED") means a verdict can't be misread, e.g. when the model repeats
    "APPROVED or REJECTED" from the instructions.
    """

    verdict: Literal["APPROVED", "REJECTED"] = Field(
        description="APPROVED only if every criterion is met, otherwise REJECTED."
    )
    feedback: str = Field(
        description="One short paragraph explaining the verdict, with concrete fixes."
    )


# =============================================================================
# 3. PROMPTS
# =============================================================================

WRITER_SYSTEM_PROMPT = (
    "You are an expert LinkedIn content writer. Your job is to write "
    "engaging, professional LinkedIn posts about the given topic. "
    "If the topic requires up-to-date information, statistics, or "
    "current trends, use the web search tool to gather fresh context "
    "before writing. If you have already received feedback on a "
    "previous draft, carefully address every point in the new draft. "
    "Feedback from the human editor always has priority over the AI "
    "reviewer's feedback. "
    "Rules for good LinkedIn posts: strong hook in the first line, "
    "1 clear takeaway, easy to skim (short paragraphs), around "
    "150–200 words, ends with a question or call-to-action to invite "
    "engagement. Do not use hashtags. "
    "Reply with the post text only, no preamble or commentary."
)

# The criteria list below is the single source of truth for what "good" means.
REVIEWER_SYSTEM_PROMPT = (
    "You are a strict LinkedIn content reviewer. You judge whether a "
    "post is publish-ready. Evaluate against these criteria:\n"
    "1. Strong hook in the first line\n"
    "2. One clear, valuable takeaway\n"
    "3. Easy to skim — uses short paragraphs\n"
    "4. Roughly 150-200 words\n"
    "5. Ends with an engaging question or CTA\n"
    "6. Professional but human tone (not corporate-robotic)\n"
    "7. No hashtags\n\n"
    "Give a verdict (APPROVED or REJECTED) and one short paragraph of "
    "feedback explaining why.\n\n"
    "Be strict but fair. Approve only if the post genuinely meets all "
    "criteria. Reject if even one criterion is clearly missing."
)


# =============================================================================

# 4. TOOLS & MODELS
#    Created lazily (on first use) and cached. This means importing this file
#    never crashes because an API key is missing, and each client is built once.

# lru_cache makes a function remember its answer.
# First call: the function runs and gives the answer.
# Next calls: it skips the work and gives back the saved answer.
# python
# @lru_cache(maxsize=1)
# def get_writer_llm():
#     return ChatOpenAI(...)   # built only once

# In this code, the writer runs many times. Without lru_cache, a new OpenAI client would be created every time. With it, the client is created once and reused.
# maxsize=1 just means "remember one answer", which is all we need because the function takes no inputs.

# =============================================================================

@lru_cache(maxsize=1)
def get_tools() -> list:
    """Tools the writer is allowed to call. Add more tools here if needed."""
    return [TavilySearch(max_results=SEARCH_MAX_RESULTS)]


@lru_cache(maxsize=1)
def get_writer_llm():
    """Writer model, with the search tool attached so it CAN call it."""
    llm = ChatOpenAI(
        model=WRITER_MODEL,
        temperature=WRITER_TEMPERATURE,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=LLM_MAX_RETRIES,
    )
    return llm.bind_tools(get_tools())


@lru_cache(maxsize=1)
def get_reviewer_llm():
    """Reviewer model, forced to answer in the ReviewResult schema."""
    llm = ChatGroq(
        model=REVIEWER_MODEL,
        temperature=REVIEWER_TEMPERATURE,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=LLM_MAX_RETRIES,
    )
    return llm.with_structured_output(ReviewResult)


# =============================================================================
# 5. NODES
#    A node = a plain function: takes the State, returns a dict of updates.
# =============================================================================

def _message_text(message: BaseMessage) -> str:
    """
    Returns a message's content as a plain string.

    Most models return a string, but some return a list of content blocks
    (e.g. [{"type": "text", "text": "..."}]). This handles both.
    """
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") if isinstance(block, dict) else str(block)
        for block in content
    )


def _build_writer_prompt(state: State) -> str:
    """
    Builds the instruction for the start of a new attempt.

    * No draft yet      -> "write the first post".
    * A draft exists    -> "improve it", showing the previous draft plus
                           whatever feedback exists (human and/or AI reviewer).
                           The writer MUST see the old draft, otherwise it
                           cannot know what to change.
    """
    topic = state["topic"]

    if not state["draft"]:
        return (
            f"Write a LinkedIn post on this topic: {topic}\n"
            "If you need current info, search the web first."
        )

    parts = [
        f"Your previous draft on '{topic}' needs changes.",
        f"Previous draft:\n\n{state['draft']}",
    ]
    if state["human_feedback"]:
        parts.append(
            "Feedback from the human editor (highest priority):\n"
            f"{state['human_feedback']}"
        )
    if state["review_feedback"]:
        parts.append(f"Feedback from the AI reviewer:\n{state['review_feedback']}")
    parts.append(
        "Write a new, improved draft that fixes every issue mentioned. "
        "Do not repeat the same mistakes."
    )
    return "\n\n".join(parts)


def writer_node(state: State) -> dict[str, Any]:
    """
    Writes (or rewrites) the LinkedIn post. May ask to run a web search first.

    This node runs in two situations:

    A) START OF A NEW ATTEMPT  (last message is NOT a tool result)
       Build a fresh prompt (first draft, or "fix this feedback") and bump the
       attempt counter.

    B) RETURNING FROM A TOOL    (last message IS a ToolMessage)
       The search finished. Hand the results back to the writer so it can
       finish the post. This is NOT a new attempt, so the counter stays put.
    """
    messages = state["messages"]
    returning_from_tool = bool(messages) and isinstance(messages[-1], ToolMessage)

    # ---- Case B: continue the current attempt after a tool call -------------
    if returning_from_tool:
        # Only send the CURRENT attempt's messages (from its last human prompt
        # onward). Older attempts would just waste tokens and confuse the model.
        last_human_idx = max(
            i for i, m in enumerate(messages) if isinstance(m, HumanMessage)
        )
        conversation = messages[last_human_idx:]
        response = get_writer_llm().invoke(
            [SystemMessage(WRITER_SYSTEM_PROMPT), *conversation]
        )
        return {"messages": [response]}

    # ---- Case A: start a new attempt ----------------------------------------
    attempt = state["attempt"] + 1
    logger.info("Starting attempt %d (AI limit %d)", attempt, state["attempt_limit"])

    human_message = HumanMessage(_build_writer_prompt(state))
    response = get_writer_llm().invoke(
        [SystemMessage(WRITER_SYSTEM_PROMPT), human_message]
    )

    # Store both messages so a later "returning from tool" call can find them.
    return {"messages": [human_message, response], "attempt": attempt}


def tool_node_factory() -> ToolNode:
    """
    Executes whatever tool calls the writer requested (here: Tavily search).

    handle_tool_errors=True means a failed search is sent back to the writer
    as an error message, instead of crashing the whole run.
    """
    return ToolNode(get_tools(), handle_tool_errors=True)


def extract_draft_node(state: State) -> dict[str, Any]:
    """
    Saves the writer's final text into state["draft"].

    We only get here when the writer's last message has NO tool calls, i.e. it
    is the finished post.
    """
    last_message = state["messages"][-1]
    draft = _message_text(last_message).strip()
    logger.info("Draft ready (%d words)", len(draft.split()))
    return {"draft": draft}


def reviewer_node(state: State) -> dict[str, Any]:
    """AI reviewer: approve, or reject with feedback for the next attempt."""
    draft = state["draft"]

    # Guard: never spend an LLM call grading an empty draft.
    if not draft:
        return {
            "review_feedback": "The draft was empty. Write a complete LinkedIn post.",
            "is_approved": False,
        }

    prompt = f"Review this LinkedIn post draft:\n\n{draft}"
    result: ReviewResult = get_reviewer_llm().invoke(
        [SystemMessage(REVIEWER_SYSTEM_PROMPT), HumanMessage(prompt)]
    )

    logger.info("AI reviewer verdict: %s", result.verdict)
    return {
        "review_feedback": result.feedback.strip(),
        "is_approved": result.verdict == "APPROVED",
    }


def human_review_node(state: State) -> dict[str, Any]:
    """
    HUMAN IN THE LOOP: pauses the workflow and waits for a human decision.

    How `interrupt()` behaves (important to understand):
      1st run of this node : interrupt() saves the state and STOPS the graph.
                             It hands the dict we pass to whoever called invoke().
      After the human answers : the graph resumes and this node runs AGAIN FROM
                             THE TOP. This time interrupt() does not stop; it
                             simply RETURNS the human's answer.

    Because the node re-runs, keep everything BEFORE interrupt() free of side
    effects (no emails, no database writes, no API calls that cost money).

    The human's answer must be a dict:
        {"action": "approve"}
        {"action": "quit"}
        {"action": "revise", "feedback": "make the hook punchier"}
    """
    decision = interrupt(
        {
            "draft": state["draft"],
            "attempt": state["attempt"],
            "ai_approved": state["is_approved"],
            "ai_feedback": state["review_feedback"],
        }
    )

    action = decision.get("action")

    if action == "approve":
        return {"human_action": "approve"}

    if action == "quit":
        return {"human_action": "quit"}

    if action == "revise":
        feedback = (decision.get("feedback") or "").strip()
        if not feedback:
            raise ValueError("A 'revise' decision needs non-empty feedback.")
        return {
            "human_action": "revise",
            "human_feedback": feedback,
            "review_feedback": "",      # old AI feedback is outdated now
            "is_approved": False,
            # Give the AI a fresh budget of MAX_ATTEMPTS drafts for this round.
            "attempt_limit": state["attempt"] + MAX_ATTEMPTS,
        }

    raise ValueError(f"Unknown human action: {action!r}")


# =============================================================================
# 6. ROUTERS
#    A router looks at the State and returns the NAME of the next node.
#    These are what turn a straight pipeline into a loop.
# =============================================================================

def route_after_writer(state: State) -> Literal["tools", "extract_draft"]:
    """Did the writer ask to use a tool, or did it finish the post?"""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"          # writer wants a web search first
    return "extract_draft"      # writer is done, pull out the text


def route_after_review(state: State) -> Literal["writer", "human_review"]:
    """AI is happy or out of attempts -> ask the human. Otherwise rewrite."""
    if state["is_approved"]:
        logger.info("AI reviewer approved, handing over to the human")
        return "human_review"
    if state["attempt"] >= state["attempt_limit"]:
        logger.warning("AI attempts used up, handing over to the human")
        return "human_review"
    return "writer"             # try again, writer will see the feedback


def route_after_human(state: State) -> Literal["writer", "__end__"]:
    """Human asked for changes -> rewrite. Approved or quit -> finish."""
    if state["human_action"] == "revise":
        return "writer"
    return END


# =============================================================================
# 7. GRAPH ASSEMBLY
#    Here we wire the nodes and routers into the workflow diagram at the top.
# =============================================================================

def build_graph(checkpointer=None):
    """
    Builds and compiles the LangGraph workflow.

    A checkpointer is REQUIRED for human-in-the-loop: it is where LangGraph
    stores the paused run. MemorySaver keeps it in RAM, which is fine for a
    script. In a real web service use a persistent one (SqliteSaver or
    PostgresSaver) so a paused run survives a restart.
    """
    graph = StateGraph(State)

    # --- Nodes (the boxes in the diagram) ---
    graph.add_node("writer", writer_node)
    graph.add_node("tools", tool_node_factory())
    graph.add_node("extract_draft", extract_draft_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("human_review", human_review_node)

    # --- Edges (the arrows) ---
    graph.add_edge(START, "writer")

    # writer -> tools OR extract_draft (decided by route_after_writer)
    graph.add_conditional_edges(
        "writer",
        route_after_writer,
        {"tools": "tools", "extract_draft": "extract_draft"},
    )

    # After a search, ALWAYS go back to the writer so it can use the results.
    graph.add_edge("tools", "writer")

    # A finished draft always goes to the AI reviewer.
    graph.add_edge("extract_draft", "reviewer")

    # reviewer -> writer (AI retry) OR human_review (ask the human)
    graph.add_conditional_edges(
        "reviewer",
        route_after_review,
        {"writer": "writer", "human_review": "human_review"},
    )

    # human_review -> writer (human wants changes) OR END (approved / quit)
    graph.add_conditional_edges(
        "human_review",
        route_after_human,
        {"writer": "writer", END: END},
    )

    return graph.compile(checkpointer=checkpointer or MemorySaver())


# =============================================================================
# 8. PUBLIC API
#    Import and call `generate_post("topic", ask_human)` from other code.
# =============================================================================

# A function that receives the review request (draft, AI verdict...) and returns
# the human's decision dict. A terminal, a web form or a Slack bot can all
# provide one. That keeps the workflow independent from HOW we ask the human.
HumanDecisionFn = Callable[[dict[str, Any]], dict[str, Any]]


def check_environment() -> None:
    """Fails fast with a clear message if any API key is missing."""
    missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]
    if missing:
        raise EnvironmentError(
            f"Missing environment variables: {', '.join(missing)}. "
            "Add them to your .env file."
        )


def generate_post(topic: str, ask_human: HumanDecisionFn) -> dict[str, Any]:
    """
    Runs the full workflow for one topic and returns the final State.

    `ask_human` is called every time the workflow pauses for a human decision.

    Useful keys in the result:
      draft         -> the latest post text
      human_action  -> "approve" (human accepted) or "quit" (human stopped)
      is_approved   -> what the AI reviewer decided on the last draft
      attempt       -> how many drafts were written in total
    """
    topic = topic.strip()
    if not topic:
        raise ValueError("Topic must not be empty.")

    load_dotenv()
    check_environment()

    app = build_graph()

    # thread_id names this run. Resuming with the same id continues the same
    # saved state instead of starting a new one.
    config = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": RECURSION_LIMIT,
    }

    # First run: goes until the graph finishes OR pauses at interrupt().
    result = app.invoke(make_initial_state(topic), config=config)

    # While the graph is paused, ask the human and resume with their answer.
    while result.get("__interrupt__"):
        review_request = result["__interrupt__"][0].value
        decision = ask_human(review_request)
        result = app.invoke(Command(resume=decision), config=config)

    return result


# =============================================================================
# 9. COMMAND-LINE INTERFACE
# =============================================================================

def ask_human_in_terminal(review_request: dict[str, Any]) -> dict[str, Any]:
    """Shows the draft in the terminal and asks the human what to do next."""
    print("\n" + "=" * 55)
    print(f"HUMAN REVIEW (draft #{review_request['attempt']})")
    print("=" * 55)
    print(review_request["draft"])
    print("-" * 55)

    ai_verdict = "APPROVED" if review_request["ai_approved"] else "REJECTED"
    print(f"AI reviewer verdict: {ai_verdict}")
    if review_request["ai_feedback"]:
        print(f"AI reviewer feedback: {review_request['ai_feedback']}")
    print("=" * 55)

    # Keep asking until we get a valid answer.
    while True:
        choice = input("[a]pprove, [r]evise or [q]uit? > ").strip().lower()

        if choice in ("a", "approve"):
            return {"action": "approve"}

        if choice in ("q", "quit"):
            return {"action": "quit"}

        if choice in ("r", "revise"):
            feedback = input("What should change? > ").strip()
            if feedback:
                return {"action": "revise", "feedback": feedback}
            print("Feedback can't be empty.")
            continue

        print("Please type a, r or q.")


def main() -> int:
    """Interactive CLI. Returns a process exit code (0 = success)."""
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "WARNING").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    print("=" * 55)
    print("Welcome to the LinkedIn Post Generator")
    print("=" * 55)
    print("\nThis tool drafts a LinkedIn post, has an AI reviewer check it,")
    print("and then asks YOU for the final decision.")
    print("=" * 55)

    topic = input("\nWhat topic do you want a LinkedIn post about?\n> ").strip()
    if not topic:
        print("\nNo topic given. Exiting.")
        return 1

    print("\nStarting generation (this can take a moment)...\n")

    try:
        final_state = generate_post(topic, ask_human=ask_human_in_terminal)
    except (EnvironmentError, ValueError) as exc:
        print(f"\nConfiguration error: {exc}")
        return 1
    except GraphRecursionError:
        print("\nThe agent exceeded its step limit and was stopped.")
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        return 130
    except Exception:
        # Full traceback goes to the log; the user gets a short message.
        logger.exception("Post generation failed")
        print("\nSomething went wrong while generating the post. "
              "Set LOG_LEVEL=INFO for details.")
        return 1

    approved_by_human = final_state["human_action"] == "approve"

    print("\n" + "=" * 55)
    print("FINAL LINKEDIN POST" if approved_by_human else "LAST DRAFT (NOT APPROVED)")
    print("=" * 55)
    print(final_state["draft"])
    print("=" * 55)
    print(f"Total drafts written: {final_state['attempt']}")
    print(f"Approved by you: {approved_by_human}")

    return 0


if __name__ == "__main__":
    sys.exit(main())