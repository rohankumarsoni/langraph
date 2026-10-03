# LangGraph Workflows

A hands-on learning repository that builds LLM workflows with [LangGraph](https://langchain-ai.github.io/langgraph/). Each folder introduces one workflow pattern, starting from the basics of state and moving towards loops and human-in-the-loop control. Every script is commented so it can be read top to bottom as a tutorial.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Core Concepts](#core-concepts)
- [Projects](#projects)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Tech Stack](#tech-stack)
- [Roadmap](#roadmap)
- [Acknowledgements](#acknowledgements)
- [Author](#author)

## Overview

LangGraph models an LLM application as a graph:

- **State** is a shared data structure that travels through the whole workflow.
- **Nodes** are Python functions that read the state, do some work (usually an LLM call), and return the fields they want to update.
- **Edges** connect the nodes and decide the order of execution.

This repository walks through the main ways of wiring those pieces together: sequential, parallel, conditional, iterative, and iterative with a human reviewer.

## Repository Structure

```
langraph/
├── 0.0.States/                      # State definitions and basics
├── 1.0.sequential_workflow/         # Stages run one after another
├── 2.0.parallel_workflow/           # Branches run at the same time, merged with a reducer
├── 3.0.conditional_workflow/        # Routing based on the current state
├── 4.0.iterative_workflow/          # Loops that refine output over several passes
├── 5.0.iterative_workfloe_with_HIL/ # Iterative workflow with a human in the loop
├── requirements.txt
└── README.md
```

## Core Concepts

| Concept | Description |
| --- | --- |
| State | A `TypedDict` that holds all data shared between nodes. |
| Node | A function that takes the state and returns a partial update. |
| Edge | A connection that defines which node runs next. |
| Reducer | A rule for merging updates when several nodes write to the same state key. |
| Compile | `graph.compile()` validates the graph and returns a runnable app. |
| Invoke | `app.invoke(initial_state)` runs the graph and returns the final state. |

## Projects

### 1.0 Sequential Workflow: Content Creation Pipeline

A three-stage pipeline that turns a rough piece of writing into a finished Hinglish video script.

```
START -> editor -> scriptwriter -> translator -> END
```

| Stage | Node | Responsibility |
| --- | --- | --- |
| 1 | `editor_node` | Fixes grammar and spelling and smooths the flow without changing the meaning. |
| 2 | `scriptwriter_node` | Rewrites the clean text as an engaging, conversational YouTube-style script. |
| 3 | `translator_node` | Localizes the script into natural Hinglish for an Indian audience. |

The state carries four fields: `raw_input`, `edited_text`, `script_text`, and `final_output`. Each node reads the output of the previous stage and writes only its own field.

### 2.0 Parallel Workflow: Content Moderation and Brand Safety

A pipeline that sends one piece of text to three specialised agents at the same time. Each agent returns a risk score from 0 to 100, where 0 is safe and 100 is high risk.

```
                 +--> toxicity_node  ---+
        START ---+--> copyright_check --+--> END
                 +--> culture_node   ---+
```

| Agent | Checks for |
| --- | --- |
| Toxicity Monitor | Profanity, aggression, and hate speech. |
| Copyright Cop | Plagiarism, trademark issues, and unoriginal content. |
| Cultural Guide | Regional sensitivities and political topics that could offend a global audience. |

Because all three nodes write to the same `safety_scores` key, the project uses a **reducer** (`merge_score_dicts`) attached through `Annotated`. The reducer merges each agent's result into one dictionary instead of letting one overwrite another.

Example output:

```python
{'toxicity_level': 70, 'copyright_risk': 85, 'cultural_insensitivity': 20}
```

The exact values depend on the model, and key order may vary because the branches finish at different times.

### 0.0, 3.0, 4.0, and 5.0

- **0.0 States** covers how state is defined and shared between nodes.
- **3.0 Conditional Workflow** covers routing the flow to different nodes depending on the current state.
- **4.0 Iterative Workflow** covers loops where output is evaluated and refined over several passes.
- **5.0 Iterative Workflow with HIL** extends the loop with a human-in-the-loop step, where a person can review the output before the workflow continues.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- An API key for at least one supported LLM provider (see [Configuration](#configuration))

### Installation

1. Clone the repository.

   ```bash
   git clone https://github.com/rohankumarsoni/langraph.git
   cd langraph
   ```

2. Create and activate a virtual environment.

   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS / Linux
   venv\Scripts\activate           # Windows
   ```

3. Install the dependencies.

   ```bash
   pip install -r requirements.txt
   ```

   The sequential workflow and the optional OpenAI line in the parallel workflow use `langchain-openai`, which is not listed in `requirements.txt`. Install it as well if you plan to use OpenAI models.

   ```bash
   pip install langchain-openai
   ```

### Running a Project

Run any script directly from its folder. For example:

```bash
python 1.0.sequential_workflow/<script_name>.py
```

Replace `<script_name>` with the name of the script inside the folder.

## Configuration

API keys are loaded from a `.env` file in the project root using `python-dotenv`. Create the file and add the keys for the provider you want to use.

```env
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
```

| Provider | Used in | Environment variable |
| --- | --- | --- |
| OpenAI (`ChatOpenAI`) | Sequential workflow | `OPENAI_API_KEY` |
| Groq (`ChatGroq`) | Parallel workflow | `GROQ_API_KEY` |

Do not commit the `.env` file. Add it to `.gitignore`.

## Tech Stack

| Package | Purpose |
| --- | --- |
| `langgraph` | Graph-based workflow orchestration |
| `langchain` | LLM abstractions and prompt handling |
| `langchain-groq` | Groq model integration |
| `langchain-community` | Community integrations |
| `langchain-text-splitters` | Text chunking utilities |
| `langchain-huggingface` | Hugging Face integration |
| `sentence-transformers` | Embedding models |
| `faiss-cpu` | Vector similarity search |
| `pypdf` | PDF loading |
| `pydantic` | Data validation and structured output |
| `python-dotenv` | Loading environment variables from `.env` |

## Roadmap

- Add a final decision node to the moderation pipeline that approves, flags for review, or rejects the text based on the three scores.
- Improve score parsing so a failed parse is not treated as a score of zero.
- Add persistence and checkpointing to the human-in-the-loop workflow.
- Add example inputs and sample outputs for each project.

## Acknowledgements

This repository was built while following the LangGraph tutorials on the YouTube channel **Sheryians AI School**. Many thanks to the team for the clear explanations of LangGraph concepts and workflow patterns. The code and comments in this repository are my own practice implementations and notes.

## Author

**Rohan Kumar Soni**

GitHub: [@rohankumarsoni](https://github.com/rohankumarsoni)
