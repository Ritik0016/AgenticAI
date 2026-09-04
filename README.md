# Agentic AI Workflows with LangGraph

This repository demonstrates modern **Agentic AI** patterns and workflow orchestration using **LangGraph**, **LangChain**, and LLMs (powered by Groq / Mistral).

---

## 🚀 Overview

Agentic AI systems move beyond single-prompt interactions by structuring complex tasks into state-driven graphs. This project highlights key architecture patterns:

1. **Sequential Workflows**: Linear execution pipelines where outputs from one agent node serve as input context for subsequent specialized agent nodes.
2. **Parallel Workflows**: Fan-out execution pipelines where independent analysis nodes execute concurrently and aggregate their results into a unified state schema.

---

## 🛠️ Workflows Included

### 1. Sequential Workflow (`sequential_workflow.py`)
A multi-stage pipeline designed for content transformation and localization:
- **Editor Node**: Cleans up raw text, fixes grammar/spelling, and refines clarity.
- **Scriptwriter Node**: Formats the polished text into an engaging, conversational YouTube script hook.
- **Translator Node**: Converts the script into natural, energetic Hinglish suitable for Indian tech audiences.

```
[START] ──> Editor Node ──> Scriptwriter Node ──> Translator Node ──> [END]
```

### 2. Parallel Workflow (`parallel_workflow.py`)
A concurrent content moderation and safety evaluation pipeline:
- **Toxicity Node**: Evaluates text for profanity, aggression, and toxicity (0–100 score).
- **Copyright Node**: Evaluates plagiarism, unoriginal content, and trademark risks (0–100 score).
- **Cultural Node**: Evaluates regional, political, and cultural sensitivities (0–100 score).
- **State Aggregation**: Uses custom state reducers (`Annotated[dict, merge_dict]`) to merge branch evaluations into a unified `safety_score` dictionary.

```
                  ┌──> Toxicity Node  ──┐
                  │                     │
[START] ──────────┼──> Copyright Node ──┼──> [END]
                  │                     │
                  └──> Cultural Node  ──┘
```

---

## 📦 Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Ritik0016/AgenticAI.git
   cd AgenticAI
   ```

2. **Set up Virtual Environment**:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/macOS
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirement.txt
   ```

4. **Configure Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   MISTRAL_API_KEY=your_mistral_api_key_here
   ```

---

## ⚡ Usage

Run the sequential content creation workflow:
```bash
python sequential_workflow.py
```

Run the parallel content moderation workflow:
```bash
python parallel_workflow.py
```

---

## 📄 License

MIT License
