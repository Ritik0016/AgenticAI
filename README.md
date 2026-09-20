# 🤖 Agentic AI Workflows with LangGraph

[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-blue.svg)](https://github.com/langchain-ai/langgraph)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-green.svg)](https://github.com/langchain-ai/langchain)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade repository demonstrating production-ready **Agentic AI Architecture Patterns** built using **LangGraph**, **LangChain**, **Groq**, **Google Gemini**, **FAISS**, and **Tavily Web Search**.

---

## 🚀 Key Agentic Patterns Implemented

This project showcases four core paradigms of modern AI Agent design:

| Workflow Pattern | File | Core Concept | Technologies Used |
| :--- | :--- | :--- | :--- |
| **1. Sequential Workflow** | [`sequential_workflow.py`](file:///c:/Users/ritik/OneDrive/Desktop/AgenticAI/sequential_workflow.py) | Linear multi-stage pipeline for content refinement & translation | LangGraph, ChatGroq / Mistral |
| **2. Parallel Workflow** | [`parallel_workflow.py`](file:///c:/Users/ritik/OneDrive/Desktop/AgenticAI/parallel_workflow.py) | Fan-out concurrent content moderation with state aggregation | LangGraph, Reducers (`Annotated`) |
| **3. Conditional (RAG) Workflow** | [`conditional_workflow.py`](file:///c:/Users/ritik/OneDrive/Desktop/AgenticAI/conditional_workflow.py) | Intent classification, RAG PDF search & dynamic node routing | FAISS, PyPDFLoader, HuggingFace |
| **4. Iterative (Reflection) Workflow** | [`iterative_workflow.py`](file:///c:/Users/ritik/OneDrive/Desktop/AgenticAI/iterative_workflow.py) | Writer-Reviewer feedback loop with web search tools | Gemini 3.8, Tavily Search, Groq |

---

## 🛠️ Detailed Workflow Architecture

### 1. Sequential Workflow (`sequential_workflow.py`)
A multi-stage pipeline designed for automated content transformation and localization:
* **Editor Node**: Standardizes raw text, fixes grammatical errors, and improves clarity.
* **Scriptwriter Node**: Re-architects polished text into a hook-driven YouTube script.
* **Translator Node**: Translates the script into natural Hinglish targeted at tech audiences.

```
[START] ──► Editor Node ──► Scriptwriter Node ──► Translator Node ──► [END]
```

---

### 2. Parallel Workflow (`parallel_workflow.py`)
A concurrent content moderation pipeline performing multi-perspective safety evaluation:
* **Toxicity Node**: Evaluates text for profanity, hostility, and toxicity (0–100).
* **Copyright Node**: Scans for plagiarism, unoriginal text, and trademark issues (0–100).
* **Cultural Node**: Checks for regional, political, and cultural sensitivities (0–100).
* **State Aggregation**: Merges parallel branch outcomes into a unified `safety_score` dictionary using a custom state reducer (`Annotated[dict, merge_dict]`).

```
                  ┌──► Toxicity Node  ──┐
                  │                     │
[START] ──────────┼──► Copyright Node ──┼──► State Aggregation ──► [END]
                  │                     │
                  └──► Cultural Node  ──┘
```

---

### 3. Conditional / RAG Routing Workflow (`conditional_workflow.py`)
An intelligent document-answering assistant for college students utilizing Retrieval-Augmented Generation (RAG):
* **Classifier Node**: Evaluates user queries and classifies intent into `academic`, `fee`, or `general`.
* **Router Edge**: Dynamically routes execution based on classified query intent.
* **Academic Node (RAG)**: Retrieves relevant context from [`academics_handbook.pdf`](file:///c:/Users/ritik/OneDrive/Desktop/AgenticAI/academics_handbook.pdf) via FAISS vector database and HuggingFace embeddings.
* **Fee Node (RAG)**: Retrieves fee structures from [`fee_structure.pdf`](file:///c:/Users/ritik/OneDrive/Desktop/AgenticAI/fee_structure.pdf).
* **General Node**: Handles general conversation without vector retrieval.
* **Response Node**: Formulates personalized answers customized to the student's selected academic program (BCA, BBA, B.Com).

```
                                 ┌──► Academic Node (RAG) ──┐
                                 │                          │
[START] ──► Classifier ──► Router┼──► Fee Node (RAG) ───────┼──► Response Node ──► [END]
                                 │                          │
                                 └──► General Node ─────────┘
```

---

### 4. Iterative / Reflection Loop Workflow (`iterative_workflow.py`)
An autonomous content generator featuring self-correction, web search tool usage, and reviewer reflection loops:
* **Writer Node**: Powered by Google Gemini (`gemini-3.8-flash`). Drafts or rewrites LinkedIn posts based on feedback or initial topic.
* **Tool Node**: Uses Tavily Web Search API to fetch live trends, statistics, and up-to-date data when needed.
* **Reviewer Node**: Powered by Groq (`groq/compound`). Strict quality controller evaluating hook, takeaway, readability, word count, CTA, and hashtag constraints.
* **Conditional Feedback Loop**: Routes back to the **Writer Node** with detailed critique if rejected, or terminates at **`END`** upon approval or hitting max retry attempt limit (3 attempts max).

```
                         ┌─────────────┐
                         ▼             │
[START] ──► Writer Node ──► Reviewer Node ──► Approved? ──► [YES] ──► [END]
                │                   │
             (Need Info)           [NO]
                │                   │
                ▼                   │
            Tool Node ──────────────┘
```

---

## 📂 Project Structure

```directory
AgenticAI/
├── sequential_workflow.py    # Sequential agent chain implementation
├── parallel_workflow.py      # Parallel fan-out/fan-in agent graph
├── conditional_workflow.py   # Intent classifier + RAG PDF router graph
├── iterative_workflow.py     # Writer-Reviewer loop with Tavily web search
├── academics_handbook.pdf    # Source PDF document for Academic RAG
├── fee_structure.pdf         # Source PDF document for Fee Structure RAG
├── requirement.txt           # Project dependencies
├── .env                      # API Key configuration
└── README.md                 # Project documentation
```

---

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Ritik0016/AgenticAI.git
cd AgenticAI
```

### 2. Set up Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirement.txt
```

### 4. Environment Configuration
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_gemini_api_key_here
TAVILY_API_KEY=your_tavily_search_api_key_here
MISTRAL_API_KEY=your_mistral_api_key_here
```

---

## ⚡ Running the Workflows

Run any workflow script directly using Python:

```bash
# 1. Run Sequential Content Creation Workflow
python sequential_workflow.py

# 2. Run Parallel Content Moderation Workflow
python parallel_workflow.py

# 3. Run Conditional RAG College Assistant Workflow
python conditional_workflow.py

# 4. Run Iterative LinkedIn Post Generator Workflow
python iterative_workflow.py
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Ritik0016/AgenticAI/issues).

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

