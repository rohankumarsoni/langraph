# =============================================================================
# PROJECT: Multi-Stage Content Pipeline (LangGraph Sequential Workflow)
# =============================================================================
#
# WHY I BUILT THIS
# I'm a content creator and my raw scripts never come out as polished as I want.
# Instead of rewriting them by hand, I split the job into small stages, each
# handled by one LLM "specialist", and chain them into a pipeline.
#
# WHAT IT DOES (3 stages, run strictly one after another)
#   1. Editor        -> fixes grammar/spelling and smooths the flow of the raw text
#   2. Scriptwriter  -> turns the clean text into an engaging, YouTube-style script
#   3. Translator    -> converts the script into natural Hinglish for an Indian audience
#
# FLOW
#   START -> editor -> scriptwriter -> translator -> END
#
# HOW LANGGRAPH FITS IN
#   - STATE : one shared dictionary that travels through the whole pipeline.
#   - NODES : Python functions. Each reads from the state, calls the LLM, and
#             returns ONLY the key(s) it wants to update.
#   - EDGES : connect the nodes and decide the order of execution.
#   LangGraph merges each node's returned dict back into the state, so every
#   stage can see the outputs of the stages before it.
#
# REQUIREMENTS
#   - pip install langgraph langchain-openai python-dotenv
#   - A .env file containing OPENAI_API_KEY
# =============================================================================


import os
from typing import TypedDict

# -----------------------------------------------------------------------------
# STATE: the shared "notebook" passed between all nodes.
# Each field is filled in by a different stage:
#   raw_input   -> provided by me at the start
#   edited_text -> written by editor_node
#   script_text -> written by scriptwriter_node
#   final_output-> written by translator_node (the finished Hinglish script)
# -----------------------------------------------------------------------------
class pipelinestate(TypedDict):

    raw_input : str
    edited_text : str
    script_text : str
    final_output : str 

from dotenv import load_dotenv 
from langchain_openai import ChatOpenAI

# Load environment variables (e.g. OPENAI_API_KEY) from the .env file
load_dotenv()

# llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

# The single LLM instance shared by all three nodes
llm = ChatOpenAI()


# -----------------------------------------------------------------------------
# NODE 1 - EDITOR
# Reads : state['raw_input']
# Writes: state['edited_text']
# Goal  : clean the text without changing its meaning.
# -----------------------------------------------------------------------------
def editor_node(state :pipelinestate) -> dict:
    """Stage 1: Cleans up grammar, removes typos, and refines the tone."""

    # The prompt gives the LLM a role + clear rules + the text to work on
    prompt = (
        "You are an expert copyeditor. Clean up the following raw text. "
        "Fix any grammatical errors, spelling mistakes, and smooth out the transition flow "
        "while keeping the core message intact. Return only the edited text.\n\n"
        f"Text:\n{state['raw_input']}"
    )
    response = llm.invoke(prompt)

    # Return only the key this node is responsible for;
    # .strip() removes stray whitespace/newlines around the LLM reply
    return {"edited_text" : response.content.strip()}


# -----------------------------------------------------------------------------
# NODE 2 - SCRIPTWRITER
# Reads : state['edited_text']   (output of Stage 1)
# Writes: state['script_text']
# Goal  : make the clean text sound like a real creator speaking on camera.
# -----------------------------------------------------------------------------
def scriptwriter_node(state: pipelinestate) -> dict:
    """Stage 2: Formats the clean text into an engaging video script style."""
    print("\n--- [Stage 2] Executing Scriptwriter Node ---")
    
    prompt = (
        "You are a charismatic YouTube content creator. Take this edited text and transform "
        "it into a highly engaging, punchy, conversational video script hook. Make it sound "
        "like a real person speaking passionately. Return only the script content.\n\n"
        f"Edited Text:\n{state['edited_text']}"
    )
    
    response = llm.invoke(prompt)
    return {"script_text": response.content.strip()}


# -----------------------------------------------------------------------------
# NODE 3 - TRANSLATOR
# Reads : state['script_text']   (output of Stage 2)
# Writes: state['final_output']
# Goal  : localize the script into natural Hinglish (mix of Hindi + English),
#         not a word-for-word translation.
# -----------------------------------------------------------------------------
def translator_node(state: pipelinestate) -> dict:
    """Stage 3: Translates the script into natural flowing Hinglish."""
    print("\n--- [Stage 3] Executing Hinglish Translator Node ---")
    
    prompt = (
        "You are an expert content localizer for the Indian market. Take the following script "
        "and convert it into natural, flowing 'Hinglish'. Do not simply translate it sentence-by-sentence "
        "or repeat information. Alternating comfortably between Hindi and English phrases just like "
        "an intellectual tech educator would speak naturally on a live stream. Keep the energy high! "
        "Return only the final Hinglish text.\n\n"
        f"Script:\n{state['script_text']}"
    )
    
    response = llm.invoke(prompt)
    return {"final_output": response.content.strip()}


# -----------------------------------------------------------------------------
# BUILDING THE GRAPH
# The state and nodes are ready. Now we wire them together.
# Edges are what turn separate functions into a workflow: they define which
# node runs after which.
# -----------------------------------------------------------------------------

from langgraph.graph import StateGraph , START , END 

# Create the graph, telling it what shape of state it will carry
graph = StateGraph(pipelinestate)

# Register the nodes: ("name used in the graph", function to run)
graph.add_node("editor",editor_node)
graph.add_node("scriptwriter",scriptwriter_node)
graph.add_node("translator",translator_node)

# Add edges (sequential - one after another):
# START -> editor -> scriptwriter -> translator -> END
graph.add_edge(START,"editor")
graph.add_edge('editor',"scriptwriter")
graph.add_edge('scriptwriter',"translator")
graph.add_edge('translator',END)

# compile() validates the graph and turns it into a runnable app
app = graph.compile()

# -----------------------------------------------------------------------------
# RUN THE PIPELINE
# invoke() takes the initial state (only raw_input is needed to start),
# runs all three stages in order, and returns the final state dictionary.
# -----------------------------------------------------------------------------
result = app.invoke({
    "raw_input" :"AI agents are the future of tech. They can think, plan, and act on their own. LangGraph helps you build these agents with proper control and memory."
})

# OUTPUT: the finished Hinglish script. The intermediate results are also
# available in result['edited_text'] and result['script_text'] if you want to debug a stage.
print("your result are : - \n\n")
print(result['final_output'])