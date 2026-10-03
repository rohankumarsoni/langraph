
# LangGraph State Patterns Example
#
# This file demonstrates different ways to define state for a LangGraph application.
# In LangGraph, the state holds the data passed between nodes in a workflow, such as
# user input, summaries, scores, messages, and metadata.
#
# 1) TypedDict
#    - Lightweight and simple
#    - Best for dict-like state structures
#    - Commonly used when you want static typing without extra runtime validation
#
# 2) Pydantic BaseModel
#    - Adds runtime validation and data constraints
#    - Useful when fields must be validated before processing
#    - Example: ensuring score is not negative
#
# 3) dataclass
#    - Python-native class-based approach
#    - Good for structured object-oriented state definitions
#
# 4) MessagesState
#    - Built-in LangGraph state type for chat/message workflows
#    - Automatically supports message history with the add_messages reducer
#
# These examples show how state can store values such as:
# topic, summary, score, user_name, language, and conversation messages.
#
# In a real project, you usually choose only one of these state patterns
# depending on your needs and workflow design.

import os 
from typing import TypedDict

#1) typed DICT (Most common approch)

class State(TypedDict):
    topic : str
    summaray : str
    score : str


#2) pydantic approch
# it is good at data validation and type checking ar runtime 

from pydantic import BaseModel, field_validator

class State(BaseModel):
    topic : str 
    score :int 
    summary : str = ""

    @field_validator
    def score_positive(cls,v):
        if v < 0:
            raise ValueError("score must be positive")


#python dataclaseess 
#standard python dataclass but it is used very rarelty 
from dataclasses import dataclass, field 

@dataclass
class State:
    topic : str = ""
    summary  : str = ""
    messages : list = field(default_factory=list)

from langgraph.graph import MessagesState
class State(MessagesState):
    # messages field is already included with add_messages reducer
    # just add your extra fields
    user_name: str
    language: str