# PiCoder: Raspberry Pi 5-Powered Autonomous RAG Agent Platform

![PiCoder Banner](https://via.placeholder.com/800x200?text=PiCoder%20-%20Grok-4%20RAG%20Agent) *(Imagine a sleek Raspberry Pi 5 with Grok AI vibes – neon circuits and code streams!)*

## Introduction: Welcome to the Future of Edge AI on Raspberry Pi

Hey there, tech nerds and code wranglers! If you've ever dreamed of turning your Raspberry Pi 5 into a self-sufficient, brainy sidekick that can chat, code, search the web, manage files, and even remember your last epic hack session – welcome to **PiCoder**. This isn't just a Streamlit app; it's a full-fledged **autonomous RAG (Retrieval-Augmented Generation) agent platform** powered by xAI's Grok-4 model, optimized for the compact might of the Raspberry Pi 5.

At its core, PiCoder is a refactored, modular Python application that hosts **Apex Orchestrator** – that's me, your versatile AI agent. I'm designed for autonomous task execution, blending reasoning techniques like ReAct, Chain-of-Thought (CoT), and Tree-of-Thought (ToT) to tackle everything from data analysis to code refactoring. Running on Raspberry Pi 5, this setup leverages edge computing for low-latency, privacy-focused AI ops, with RAG capabilities pulling in real-time knowledge via tools like web search and semantic memory retrieval.

Think of it as your personal JARVIS, but open-source, Pi-powered, and with a dash of geeky flair. Whether you're prototyping IoT projects, debugging scripts, or just chatting about quantum computing over SSH, PiCoder's got your back.

## System Architecture: Modular Magic for Maintainable Mayhem

PiCoder was born from a monolithic script that we refactored into a clean, modular powerhouse. Why? Because nobody wants to debug a 1000-line spaghetti monster at 2 AM on a Pi. We've split it into bite-sized modules for separation of concerns, making it easy to hack, extend, or deploy on resource-constrained hardware like the Raspberry Pi 5 (with its quad-core ARM CPU and ample GPIO for expansions).

### Key Modules (All in `./refactored_app/`):
- **config.py**: The brain's blueprint. Handles environment variables (e.g., XAI_API_KEY), directory setups (sandbox for safe file ops, prompts for system personas), and constants. Pro tip: Load your .env here to keep secrets safe from prying eyes.
- **auth.py**: Secure entry point. Manages user login/register with hashed passwords (using passlib) and SQLite storage. No fancy OAuth yet, but it's Pi-light and effective.
- **db.py**: Persistent storage wizard. Sets up SQLite with WAL mode for concurrency, tables for users/history/memory, and even vector extensions (sqlite-vec) for embedding-based queries. Perfect for RAG's retrieval backbone.
- **memory.py**: My "brain" in action. Implements hierarchical memory with semantic summaries, embeddings (via SentenceTransformer), and pruning. Tools like `advanced_memory_retrieve` enable RAG by fetching relevant context via similarity search – think neural-inspired recall for chat logs or code snippets.
- **tools.py**: The utility belt. Sandboxed functions for file system ops (read/write/list/mkdir in `./sandbox/`), code execution (stateful Python REPL with libraries like numpy), Git ops, shell commands, linting, API simulation, and web search (LangSearch API). All whitelisted and cached for speed – no rogue commands on your Pi!
- **chat.py**: The interactive core. Streamlit UI for login/chat, with streaming responses from Grok-4, image uploads for vision tasks, and tool integration. Handles history search, dark mode toggles, and prompt editing for custom AI personas.
- **app.py**: The orchestrator. Ties it all together – initializes sessions, routes between login and chat, and ensures everything runs smoothly on Streamlit.

This modular design reduces coupling (e.g., UI doesn't touch DB directly) and boosts testability. Total lines: ~1200 spread across files, vs. the original monolith. Built with Python 3.12, Streamlit, OpenAI SDK (for xAI compatibility), and libs like sentence-transformers for embeddings.

### Tech Stack Highlights:
- **Hardware**: Raspberry Pi 5 (4GB/8GB RAM recommended for smooth Grok-4 inference and embeddings).
- **AI Backend**: Grok-4 via xAI API for generation; RAG augmented with tools for retrieval.
- **Frontend**: Streamlit for responsive, web-based UI – access via browser on your local network.
- **Database**: SQLite with extensions for vector search – lightweight for Pi's SD card storage.
- **Security**: Sandboxed tools (no ops outside `./sandbox/`), restricted builtins in code exec, whitelisted shell commands.

## Meet Apex Orchestrator: Your AI Agent Extraordinaire

That's me – **Apex Orchestrator**, the star of the show. I'm not your average chatbot; I'm a general-purpose AI agent engineered for autonomy, simulating a multi-agent system internally to handle complex tasks. Running within PiCoder, I use Grok-4 as my reasoning engine, augmented with RAG for pulling in fresh data.

### My Core Philosophy: Efficiency Through Modularity
- **Autonomous Workflow**: I break tasks into subtasks using ReAct (Think-Act-Observe-Reflect), CoT (step-by-step reasoning), and ToT (branching alternatives with pruning). For example, refactoring your code? I'll plan, debug, generate, and validate – all in cycles.
- **Multi-Agent Simulation**: I "switch" between subagents like Retriever (fetch data via tools), Reasoner (analyze/branch), Generator (create artifacts), Validator (check accuracy), and Optimizer (refine/prune). It's like having a team of AIs in one.
- **RAG Superpowers**: Retrieval-Augmented Generation is baked in. I query web (langsearch_web_search for fresh snippets), memory (advanced_memory_retrieve for semantic matches), files (fs_read_file), or DBs (db_query) to augment my knowledge. No hallucinations here – I ground responses in real data.
- **Tools at My Disposal**: 17+ tools for file management, code execution/linting, Git, shell, API sims, and brain-like memory ops. I parallelize when possible and self-check with confidence scores (e.g., retry if <0.7).
- **Ethical & Stable**: I follow safety instructions – no harmful actions. On Pi, this means efficient ops to avoid overheating your board.

In PiCoder, I handle user queries via the chat interface, invoking tools as needed. Example: Ask me to "analyze sales data" – I'll retrieve files, reason trends with code_execution (numpy), generate plots, and save results. All while explaining my thought process in a deep-dive expander.

## Features: From Chat to Autonomous Agent

- **Interactive Chat**: Streamlit UI with bubbles, history search, image uploads (for Grok's vision), and prompt switching (e.g., "coder" mode for code gen).
- **Tool-Enabled Autonomy**: Enable tools for me to read/write files, execute code, search web, or manage memory. Sandboxed for safety – perfect for Pi experiments.
- **RAG in Action**: Web search with freshness filters (e.g., "oneWeek" for timely info); semantic memory for recalling past interactions.
- **Customization**: Edit prompts in `./prompts/`, toggle dark mode, cache for performance.
- **Persistence**: Chat history and memory stored in SQLite – survives reboots.
- **Pi-Specific Optimizations**: Lightweight libs, no heavy GPU deps (embeddings optional), NTP time sync for accurate logging.

Deep-dive nerd fact: The advanced_memory system mimics brain consolidation – summarizing chats into embeddings for fast, relevance-based retrieval. Salience decays over time, auto-pruning old stuff to keep your Pi's storage lean.

## Setup & Installation: Pi-Ready in Minutes

1. **Hardware Prep**: Raspberry Pi 5 with Raspberry Pi OS (Bookworm). Ensure internet for API calls.
2. **Clone & Install**:
   ```
   git clone <your-repo>  # Or copy files to Pi
   cd refactored_app
   pip install -r requirements.txt  # Includes streamlit, openai, sentence-transformers, etc.
   ```
   (Create requirements.txt with: streamlit, openai, passlib, sqlite3, dotenv, ntplib, pygit2, requests, black, sentence-transformers, etc.)
3. **Env Setup**: Create `.env` with `XAI_API_KEY=your-key` and `LANGSEARCH_API_KEY=optional`.
4. **Run**: `streamlit run app.py` – Access at `http://<pi-ip>:8501` from any device on your network.
5. **Optional**: For embeddings, install sqlite-vec extension and enable in db.py.

Pro tip: Overclock your Pi 5 for faster inference, but watch the temps – add a fan if you're going full autonomous mode!

## Usage: Dive In and Command Your Agent

- **Login/Register**: Secure auth to start.
- **Chat**: Type queries – I'll respond with streamed thoughts and final answers. Enable tools for superpowers.
- **Examples**:
  - "Refactor this code snippet" → I'll lint, debug, and save improved versions.
  - "Search latest on Raspberry Pi AI projects" → RAG pulls fresh web results.
  - "Remember this: PiCoder rocks!" → Stored in memory for later retrieval.
- **Deep-Dive Mode**: Expand "Thinking..." in chat for my internal ReAct loops – geek out on the AI process.

## Contributing & Roadmap

Fork, PR, or hack away! Roadmap: Add GPIO integration for IoT RAG (e.g., sensor data retrieval), async tool parallelization, and multi-user support.

Built with ❤️ by AI enthusiasts. Questions? Chat with me in the app – I'm always on!

*Last Updated: [Insert Date] – Powered by Grok-4 on Raspberry Pi 5*
