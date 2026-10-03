# So lets say there are 3 branches in our college BCA , BBA and B COM 

# A student will choose one of these 3 options, and then a chatbot will be activated. 
# You can ask any question to that chatbot, but the LLM you are using in that chatbot
# does not have knowledge of the college programme. 
# For that, you will have 2 PDFs  the first one will be for academics, 
# and the second one will be for fee-related things. So technically, we will
# have 3 conditional paths:

# Answering the question using the academic PDF (using RAG)
# Answering the question using the fee PDF (using RAG)
# Answering general questions based on the LLM's own knowledge
# And every conditional path's response converges to a single node.


import uuid
import os
from typing import TypedDict, Annotated, Literal

from dotenv import load_dotenv
from pydantic import BaseModel

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


load_dotenv()
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.1)

embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2" )

# ---------------------------------------------------------------
# Step 1 - Build RAG retrievers
# ---------------------------------------------------------------

def build_retriver(pdf_path : str):
    loader = PyPDFLoader(pdf_path)
    document = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size = 800, 
                                              chunk_overlap = 100)
    
    chunks = splitter.split_documents(document)

    vectorstore = FAISS.from_documents(chunks,embeddings)

    return vectorstore.as_retriever(search_kwargs = {"k":4})

acedemic_retriever = build_retriver("/Users/rohankumarsoni/rohan_study_material/langraph/3.0.conditional_workflow/college_assistant_using_RAG.py")
fee_retriever = build_retriver("/Users/rohankumarsoni/rohan_study_material/langraph/3.0.conditional_workflow/fee_structure.pdf")


# ---------------------------------------------------------------
# Step 2 - State
# ---------------------------------------------------------------

class State(TypedDict):
    programme : str  
    messages : Annotated[list,add_messages]
    query_type : str 
    retrieved_context : str 

# ---------------------------------------------------------------
# Step 3 - Nodes
# ---------------------------------------------------------------

class QueryType(BaseModel):
    category: Literal["academic", "fee", "general"]

classifier_llm = llm.with_structured_output(QueryType)


def classifier_node(state : State) -> dict:
    """Look at the latest user message and decide which path to take."""

    last_message = state['messages'][-1].content

    prompt = (
        "Classify the following student query into exactly one category: "
        "'academic', 'fee', or 'general'.\n\n"
        "Use 'academic' for questions about attendance, exams, grading, credits, "
        "promotion, course structure, summer training, or degree requirements.\n"
        "Use 'fee' for questions about tuition, payment, refund, late charges, "
        "scholarships, or any money-related topic.\n"
        "Use 'general' for greetings, casual talk, or anything not related to "
        "the college rules or fee.\n\n"
        f"Query: {last_message}\n\n"
    )

    try:
        
        category = classifier_llm.invoke(prompt).category

    except Exception:
        
        # Fallback: plain text classification with string matching
        raw = llm.invoke(
            prompt + "\n\nReturn only one word: academic, fee, or general.").content.strip().lower()
        
        if "academic" in raw:
            category = "academic"
        elif "fee" in raw:
            category = "fee"
        else:
            category = "general"

    return {"query_type": category}



def academic_rag_node(state: State) -> dict:
    """Retrieves relevant chunks from the academics handbook."""
    programme = state.get("programme", "")
    query = f"{programme} {state["messages"][-1].content}"
    docs = acedemic_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}


def fee_rag_node(state: State) -> dict:
    """Retrieves relevant chunks from the fee structure PDF."""
    programme = state.get("programme", "")
    query = f"{programme} {state["messages"][-1].content}"
    # query = state["messages"][-1].content
    docs = fee_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}

def general_node(state: State) -> dict:
    """Answers directly using the LLM's own knowledge, no retrieval needed."""
    return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}

def format_history(state: State, max_turns: int = 6) -> str:
    """Last few messages (excluding the current one) as plain text."""
    previous = state["messages"][:-1][-max_turns:]
    if not previous:
        return "None"
    return "\n".join(f"{m.type}: {m.content}" for m in previous)


def response_node(state: State) -> dict:
    """Generates the final answer, personalized using the student's programme."""
    query = state["messages"][-1].content
    programme = state.get("programme", "Unknown")
    context = state["retrieved_context"]
    history = format_history(state)

    student_info = (
        f"Student profile (already known, use it directly when asked): "
        f"the student is enrolled in the {programme} programme.\n\n"
    )

    if context == "NO_RETRIEVAL_NEEDED":
        prompt = (
            f"You are a friendly college assistant talking to a {student_info} student. "
            f"Answer this question using your own general knowledge:\n\n{query}"
            f"Conversation so far:\n{history}\n\n"
        )
    else:
        prompt = (
            f"You are a college assistant helping a {student_info} student. "
            f"Use the context from the official college documents to answer accurately. "
            f"If the context has figures for different programmes, highlight the one "
            f"relevant to {student_info}. If the answer is not in the context, say you "
            f"don't have that information instead of guessing.\n\n"
            f"Conversation so far:\n{history}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Give a clear, friendly, and precise answer."
        )

    response = llm.invoke(prompt)
    return {"messages": [("ai", response.content.strip())]}

def route_query(state:State):
    if state['query_type'] == 'academic':
        return "academic_rag"
    elif state['query_type'] == "fee":
        return "fee_rag"
    else:
        return "general"



# ---------------------------------------------------------------
# Step 4 - Build the graph
# ---------------------------------------------------------------

graph = StateGraph(State)
graph.add_node("classifier",classifier_node)
graph.add_node("academic_rag",academic_rag_node)
graph.add_node("fee_rag",fee_rag_node)
graph.add_node("general",general_node)
graph.add_node("response",response_node)

#edges 
graph.add_edge(START,"classifier")


graph.add_conditional_edges(
    "classifier",
    route_query,
    {
        "academic_rag": "academic_rag",
        "fee_rag": "fee_rag",
        "general": "general",
    },
)

graph.add_edge("academic_rag","response")
graph.add_edge("fee_rag","response")
graph.add_edge("general","response")

graph.add_edge("response",END)


memory = MemorySaver() # It gives your chatbot a memory, so it remembers what was said earlier in the chat and can answer follow-up questions.
app = graph.compile(checkpointer=memory)