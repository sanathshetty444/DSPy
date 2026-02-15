# DSPy — Small Retrieval-Augmented Generation Example

This repository demonstrates a minimal RAG (Retrieval-Augmented Generation) pipeline built on top of the DSPy primitives. It shows how to wire a retriever (RM), a language model (LM), and a simple generator Signature. The code includes a tiny mock retriever so you can run the example locally without external search or vector DBs.

## What this project contains

- `index.py` — small entrypoint that instantiates the RAG module and runs a single query.
- `module.py` — defines the `RAG` module which composes a `Retrieve` and a `ChainOfThought(GenerateAnswer)` generator.
- `config.py` — sets up DSPy configuration: LM, RM (a `MockRM` is provided), and environment loading.
- `signature.py` — defines the `GenerateAnswer` signature used by the generator.

## Prerequisites

- Python 3.9+ (this project uses a virtual environment at `.venv` in the example).
- `pip` and virtualenv (or venv).
- If you want to use a real OpenAI model, set `OPENAI_API_KEY` in a `.env` file or your environment. The example `config.py` already shows a `MockRM` so you can run without keys.

## Install and run (local, using the mock retriever)

1. Create and activate a virtual environment (zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies (adjust if you have a `requirements.txt` — otherwise install DSPy and python-dotenv):

```bash
pip install python-dotenv dspy
```

3. (Optional) Create a `.env` file in the project root to add your OpenAI key:

```env
# .env
OPENAI_API_KEY=sk-...
```

4. Run the example:

```bash
python3 index.py
```

You should see the printed Question, Answer, and the Context used (if the mock retriever is active).

## How the code executes (step-by-step)

1. `index.py` imports the `RAG` module from `module.py` and the global configuration in `config.py` is executed on import.
2. `config.py` loads environment variables (via `dotenv`) and configures DSPy with an LM and RM. The included `MockRM` returns a small list of `SimpleNamespace` objects that expose a `.long_text` attribute.
3. `index.py` creates `uncompiled_rag = RAG()` and calls `uncompiled_rag(question=...)`.
4. `RAG.forward` (in `module.py`) calls `self.retrieve(question)` to fetch relevant passages from the configured RM (via DSPy’s `Retrieve` wrapper).
5. The retrieved `passages` are then passed to the generator (`ChainOfThought(GenerateAnswer)`), which produces a `prediction` with an `answer` attribute.
6. `RAG.forward` returns a `dspy.Prediction` with the selected `context` and `answer`.
7. `index.py` prints the results.

## File summaries

- `config.py` — Example configuration. Key points:
  - It loads `.env` and configures `dspy.configure(lm=lm, rm=rm)`.
  - The provided `MockRM` returns `[SimpleNamespace(long_text=text)]` so DSPy’s internal `Retrieve` can extract `long_text` safely.

- `signature.py` — Defines `GenerateAnswer` signature expected by the generator. Fields:
  - `context: str` (the retrieved facts)
  - `question: str` (input)
  - `answer: str` (output)

- `module.py` — Shows a `RAG` class that composes retrieval + generation.
