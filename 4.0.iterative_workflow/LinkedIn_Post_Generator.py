"""
LinkedIn Post Generator: a writer + reviewer agent built with LangGraph.

WHAT IT DOES
------------
You give it a topic. A *writer* LLM drafts a LinkedIn post (searching the web
with Tavily if it needs fresh facts). A separate *reviewer* LLM then grades the
draft. If the reviewer rejects it, the writer sees the feedback and tries again,
up to MAX_ATTEMPTS drafts in total.

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
 ┌──────────┐   approved, or out of attempts   ┌─────┐
 │ reviewer │ ───────────────────────────────► │ END │
 └──────────┘                                  └─────┘
      │ rejected and attempts remain
      └──────────► back to writer (with the reviewer's feedback)

TERMINOLOGY
-----------
* "attempt"     = one full draft (writer -> [tools -> writer ...] -> reviewer).
* "tool round"  = one trip writer -> tools -> writer inside a single attempt.

SETUP
-----
    pip install langgraph langchain-openai langchain-groq langchain-tavily python-dotenv pydantic

    # .env file (never commit this file)
    OPENAI_API_KEY=...
    GROQ_API_KEY=...
    TAVILY_API_KEY=...

"""

from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache
from typing import Annotated, Any, Literal, TypedDict

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
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

# =============================================================================
# 1. CONFIGURATION
#    Every tunable value lives here so nothing is "hidden" deep in the code.
# =============================================================================

WRITER_MODEL = "gpt-4o-mini"                # drafts the post (OpenAI)
REVIEWER_MODEL = "openai/gpt-oss-120b"  # grades the post (Groq)

WRITER_TEMPERATURE = 0.7    # higher = more creative writing
REVIEWER_TEMPERATURE = 0.2  # lower  = more consistent, strict grading

MAX_ATTEMPTS = 10            # max number of drafts before we give up
SEARCH_MAX_RESULTS = 3      # web results Tavily returns per search

LLM_TIMEOUT_SECONDS = 60    # fail instead of hanging forever on a slow API
LLM_MAX_RETRIES = 2         # automatic retries on transient API errors

# Safety net: LangGraph raises an error if the graph takes more steps than
# this. It protects us from infinite loops (e.g. the writer searching forever).
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
    review_feedback: str                      # reviewer's explanation
    is_approved: bool                         # did the reviewer approve?
    attempt: int                              # how many drafts started so far


def make_initial_state(topic: str) -> State:
    """Builds the starting state for a fresh run."""
    return {
        "topic": topic,
        "messages": [],
        "draft": "",
        "review_feedback": "",
        "is_approved": False,
        "attempt": 0,
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
    topic = state["topic"]

    if attempt == 1:
        user_message = (
            f"Write a LinkedIn post on this topic: {topic}\n"
            "If you need current info, search the web first."
        )
    else:
        user_message = (
            f"Your previous draft on '{topic}' was rejected.\n"
            f"Here is the reviewer's feedback:\n\n{state['review_feedback']}\n\n"
            "Write a new, improved draft that fixes every issue mentioned. "
            "Do not repeat the same mistakes."
        )

    logger.info("Starting attempt %d/%d", attempt, MAX_ATTEMPTS)

    human_message = HumanMessage(user_message)
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
    """Grades the draft: approve, or reject with feedback for the next attempt."""
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

    logger.info("Reviewer verdict: %s", result.verdict)
    return {
        "review_feedback": result.feedback.strip(),
        "is_approved": result.verdict == "APPROVED",
    }


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


def route_after_review(state: State) -> Literal["writer", "__end__"]:
    """Stop if approved or out of attempts, otherwise rewrite."""
    if state["is_approved"]:
        logger.info("Post approved")
        return END
    if state["attempt"] >= MAX_ATTEMPTS:
        logger.warning("Reached max attempts (%d) without approval", MAX_ATTEMPTS)
        return END
    return "writer"             # try again, writer will see the feedback


# =============================================================================
# 7. GRAPH ASSEMBLY
#    Here we wire the nodes and routers into the workflow diagram at the top.
# =============================================================================

def build_graph():
    """Builds and compiles the LangGraph workflow."""
    graph = StateGraph(State)

    # --- Nodes (the boxes in the diagram) ---
    graph.add_node("writer", writer_node)
    graph.add_node("tools", tool_node_factory())
    graph.add_node("extract_draft", extract_draft_node)
    graph.add_node("reviewer", reviewer_node)

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

    # A finished draft always goes to the reviewer.
    graph.add_edge("extract_draft", "reviewer")

    # reviewer -> END (done) OR writer (retry), decided by route_after_review
    graph.add_conditional_edges(
        "reviewer",
        route_after_review,
        {"writer": "writer", END: END},
    )

    return graph.compile()


# =============================================================================
# 8. PUBLIC API
#    Import and call `generate_post("topic")` from other code (API, notebook...).
# =============================================================================

def check_environment() -> None:
    """Fails fast with a clear message if any API key is missing."""
    missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]
    if missing:
        raise EnvironmentError(
            f"Missing environment variables: {', '.join(missing)}. "
            "Add them to your .env file."
        )


def generate_post(topic: str) -> State:
    """
    Runs the full workflow for one topic and returns the final State.

    Useful keys in the result:
      draft        -> the final post text
      is_approved  -> True if the reviewer approved it
      attempt      -> how many drafts were written
    """
    topic = topic.strip()
    if not topic:
        raise ValueError("Topic must not be empty.")

    load_dotenv()
    check_environment()

    app = build_graph()
    return app.invoke(
        make_initial_state(topic),
        config={"recursion_limit": RECURSION_LIMIT},
    )


# =============================================================================
# 9. COMMAND-LINE INTERFACE
# =============================================================================

def main() -> int:
    """Interactive CLI. Returns a process exit code (0 = success)."""
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "WARNING").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    print("=" * 55)
    print("Welcome to the LinkedIn Post Generator")
    print("=" * 55)
    print("\nThis tool will draft a LinkedIn post for you, review it")
    print("itself, and iterate until it's publish-ready.")
    print("=" * 55)

    topic = input("\nWhat topic do you want a LinkedIn post about?\n> ").strip()
    if not topic:
        print("\nNo topic given. Exiting.")
        return 1

    print("\nStarting generation...\n")

    try:
        final_state = generate_post(topic)
    except (EnvironmentError, ValueError) as exc:
        print(f"\nConfiguration error: {exc}")
        return 1
    except GraphRecursionError:
        print("\nThe agent exceeded its step limit and was stopped.")
        return 1
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130
    except Exception:
        # Full traceback goes to the log; the user gets a short message.
        logger.exception("Post generation failed")
        print("\nSomething went wrong while generating the post. "
              "Set LOG_LEVEL=INFO for details.")
        return 1

    print("\n" + "=" * 55)
    print("FINAL LINKEDIN POST")
    print("=" * 55)
    print(final_state["draft"])
    print("=" * 55)
    print(f"Total attempts: {final_state['attempt']}")
    print(f"Approved: {final_state['is_approved']}")

    if not final_state["is_approved"]:
        print("\nNote: the reviewer never approved this post. It is the "
              "last draft, so review it yourself before publishing.")
        print(f"Last reviewer feedback: {final_state['review_feedback']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())