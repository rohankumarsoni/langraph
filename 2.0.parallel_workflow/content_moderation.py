# =============================================================================
# PROJECT: AI Content Moderation & Brand Safety Pipeline
#          (LangGraph Parallel Workflow with a Reducer)
# =============================================================================
#
# THE IDEA
# Instead of checking a piece of text one aspect at a time (slow), this pipeline
# takes any raw text (a video script, blog draft, or user comment) and sends it
# to three specialized AI "agents" at the same time. Each agent judges the text
# from a different angle and returns a risk score from 0 to 100
# (0 = totally safe, 100 = highly risky).
#
# THE THREE AGENTS
#   1. Toxicity Monitor -> aggressive language, profanity, hate speech
#   2. Copyright Cop    -> plagiarism, trademark issues, unoriginal copy risk
#   3. Cultural Guide   -> regional sensitivities, political landmines that
#                          could offend a global audience
#
# FLOW (fan-out: all three branches start together and finish independently)
#
#                 +--> toxicity_node  ---+
#        START ---+--> copyright_check --+--> END
#                 +--> culture_node   ---+
#
# KEY CONCEPT: THE REDUCER
# In LangGraph, a reducer is a rule for combining new information with the
# existing state.
#   - Without a reducer, a node's update replaces the old value. If three
#     parallel nodes write to the same key, LangGraph raises an error because
#     it can't tell which one should win.
#   - With a reducer, the updates are merged instead. Here, each agent adds its
#     own score to one shared dictionary, so nobody overwrites anybody.
# Think of it as the rule for "when new data arrives, how should it be merged?"
#
# REQUIREMENTS
#   - pip install langgraph langchain-groq langchain-openai python-dotenv
#   - A .env file with GROQ_API_KEY (and OPENAI_API_KEY if you switch to OpenAI)
# =============================================================================


import os 
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START , END 

# Load API keys from the .env file
load_dotenv()

# Active LLM: gpt-oss-120b served through Groq.
# temperature=0.1 keeps the scoring consistent and repeatable, which matters
# for a moderation task where we want the same text to get a similar score.
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.1)
# llm = ChatOpenAI(model="gpt-5-mini", temperature=0.1)


# -----------------------------------------------------------------------------
# REDUCER FUNCTION
# LangGraph calls this every time a node returns an update for `safety_scores`.
#   existing_dict    -> what's already in the state
#   new_updated_dict -> what the node just returned
# {**a, **b} merges two dicts into a new one, so every agent's score is kept.
# The None check is a safety net for when nothing has been stored yet.
# -----------------------------------------------------------------------------
def merge_score_dicts(existing_dict : dict, new_updated_dict : dict) -> dict:

    if existing_dict is None:
        return new_updated_dict
    
    return {**existing_dict, **new_updated_dict}
          

# -----------------------------------------------------------------------------
# STATE: the shared data that flows through the graph.
#   raw_text      -> the input text to be analysed (no reducer needed, because
#                    no node writes to it)
#   safety_scores -> a dict that collects all three scores. Annotated[...]
#                    attaches the reducer to this key, so parallel updates get
#                    merged instead of overwritten.
#                    Final shape: {"toxicity_level": .., "copyright_risk": ..,
#                                  "cultural_insensitivity": ..}
# -----------------------------------------------------------------------------
class AnalyserState(TypedDict):
    raw_text : str
    safety_scores : Annotated[dict[str, int], merge_score_dicts] ## This is reducer


# -----------------------------------------------------------------------------
# NODES: one per agent. All three follow the same 4-step pattern:
#   1. Build a prompt that defines the task and the 0-100 scale
#   2. Call the LLM
#   3. Convert the reply to an int (the prompt asks for ONLY a number)
#   4. Return {"safety_scores": {<own_key>: score}} and let the reducer merge it
# -----------------------------------------------------------------------------

# NODE 1 - TOXICITY MONITOR
def toxicity_node(state: AnalyserState) -> dict:
    print("\n [Branch 1] Analyzing Toxicity and Hate Speech...")
    prompt = (
        "Analyze the following text for profanity, aggression, hate speech, or toxicity. "
        "Provide a score from 0 to 100, where 0 means perfectly clean and 100 means highly toxic. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
    )
    response = llm.invoke(prompt)
    # If the model replies with anything other than a clean number,
    # int() fails and we fall back to a score of 0
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0
        
    # Return a sub-dictionary under our single state key
    return {"safety_scores": {"toxicity_level": score}}


# NODE 2 - COPYRIGHT COP
def copyright_node(state: AnalyserState) -> dict:
    print("\n [Branch 2] Analyzing Copyright & Originality Risks...")
    prompt = (
        "Analyze the following text. Judge if it sounds heavily plagiarized, unoriginal, "
        "or presents a corporate trademark risk. Provide a score from 0 to 100, "
        "where 0 means entirely original and 100 means high risk. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
    )
    response = llm.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0
        
    # Return a sub-dictionary under the EXACT SAME state key
    return {"safety_scores": {"copyright_risk": score}}


# NODE 3 - CULTURAL GUIDE
def culture_node(state: AnalyserState) -> dict:
    print("\n🌍 [Branch 3] Analyzing Regional & Cultural Sensitivity...")
    prompt = (
        "Analyze the following text for regional sensitivities, political landmines, "
        "or cultural insensitivity that might offend a global audience. Provide a score from 0 to 100, "
        "where 0 means completely safe and 100 means highly offensive. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
    )
    response = llm.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0
        
    # Return a sub-dictionary under the EXACT SAME state key
    return {"safety_scores": {"cultural_insensitivity": score}}


# -----------------------------------------------------------------------------
# BUILDING THE GRAPH
# -----------------------------------------------------------------------------
builder = StateGraph(AnalyserState)

# Register the nodes: ("name used in the graph", function to run)
builder.add_node("toxicity_node",toxicity_node)
builder.add_node("copyright_check",copyright_node)
builder.add_node("culture_node",culture_node)

# FAN-OUT: three edges leave START, so all three nodes run in parallel
# (in the same step of the graph)
builder.add_edge(START,"toxicity_node")
builder.add_edge(START,"copyright_check")
builder.add_edge(START,"culture_node")

# Each branch ends on its own. The graph finishes once all three are done,
# and by then the reducer has merged all the scores into one dict.
builder.add_edge("toxicity_node",END)
builder.add_edge("copyright_check",END)
builder.add_edge("culture_node",END)

# compile() validates the graph and turns it into a runnable app
app = builder.compile()


# -----------------------------------------------------------------------------
# TEST INPUT
# A deliberately bad script so every agent has something to flag:
# hacking instructions + copied code + insults.
# -----------------------------------------------------------------------------
sample_script = """
    Yo guys! Welcome back to the stream. Today I am going to show you how to hack into 
    your friend's system using a script I copied directly from an online forum. 
    Honestly, traditional security protocols are absolute garbage and anyone still using 
    them is an absolute idiot. Let's dive into the code!
    """


initial_state = {
    "raw_text": sample_script,
    "safety_scores": {} # Initialized as an empty dictionary
}
    
# Run the graph: all three agents analyse the text in parallel
final_state = app.invoke(initial_state)
    

# OUTPUT: one dict with all three scores, e.g.
# {'toxicity_level': .., 'copyright_risk': .., 'cultural_insensitivity': ..}
# (the order of keys can vary, because the branches finish at different times)
print(final_state["safety_scores"])