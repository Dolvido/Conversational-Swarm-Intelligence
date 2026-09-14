# Conversational Swarm Intelligence

A Python experiment with two local language-model agents, separate retrieval stores, and a simple comparison of their responses.

**Status:** Historical prototype. This repository preserves an early multi-agent exploration. Dependency compatibility and end-to-end execution have not been revalidated.

## What is implemented

The repository contains one script: [conversation.py](conversation.py).

- An `Agent` class wraps a local Ollama model, defaulting to `llama3`.
- Each agent has a named Chroma collection with Ollama embeddings.
- Knowledge entries include topic and confidence metadata.
- Agents retrieve one relevant entry to generate an opinion.
- A conversation method runs a fixed number of exchanges and records them in memory.
- A collective-response method filters responses by TF-IDF cosine similarity and appends a VADER sentiment label.

This is an exploratory response-comparison heuristic. Similar wording or sentiment does not establish that an answer is correct or that agents have reached a reliable consensus.

## Included demonstration

The script creates two agents, seeds each with a short description of AI, prints their initial opinions, and runs three conversation rounds about blockchain. It then prints the agents' AI opinions again.

The demonstration runs at module load time; importing the file also starts it.

## Reproducing the experiment

The repository has no dependency manifest or pinned environment. Imports require compatible versions of LangChain, LangChain Community, LangChain Core, Chroma, NumPy, scikit-learn, and NLTK.

The script expects a running Ollama service, the `llama3` generation model, and the embedding model selected by the installed `OllamaEmbeddings` defaults. Agent construction also calls NLTK's VADER lexicon downloader.

After restoring that environment, the entry point is:

```bash
python conversation.py
```

This command identifies the source entry point; it is not a verified setup recipe.

## Limitations

- Agents run sequentially in one process.
- Conversation history is stored but is not supplied to the generation prompts.
- Knowledge-update support exists, but the demonstration does not call it.
- The similarity filter compares other responses with the first response using a fixed 0.5 threshold.
- Confidence values are supplied metadata, not calibrated model confidence.
- Model choices, seed knowledge, and demonstration topics are hardcoded.
- There are no automated tests, benchmarks, or durable-storage configuration in this snapshot.

## Built with

[LangChain](https://github.com/langchain-ai/langchain), [Ollama](https://github.com/ollama/ollama), [Chroma](https://github.com/chroma-core/chroma), [scikit-learn](https://github.com/scikit-learn/scikit-learn), and [NLTK](https://github.com/nltk/nltk).
