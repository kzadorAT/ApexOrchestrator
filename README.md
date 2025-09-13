# Apex Orchestrator: Raspberry Pi 5-Powered RAG AI Agent

[![GitHub stars](https://img.shields.io/github/stars/yourusername/apex-orchestrator?style=social)](https://github.com/yourusername/apex-orchestrator) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/) [![Streamlit](https://img.shields.io/badge/Streamlit-1.38.0-orange.svg)](https://streamlit.io/)

> *Because who needs a data center when your pocket-sized Pi can summon AI overlords? Apex Orchestrator turns your Raspberry Pi 5 into a self-contained RAG (Retrieval-Augmented Generation) powerhouse—chat, code, search, and orchestrate like a boss. Expandable, sandboxed, and nerd-approved. 🚀*

Apex Orchestrator is a standalone AI agent platform hosted on Raspberry Pi 5, leveraging xAI's Grok models for intelligent, tool-augmented conversations. It's built for tinkerers, coders, and mad scientists who want local-first AI with global smarts. No Kubernetes drama—just pure Pi magic.

Key highlights:
- **RAG-Infused Memory**: Semantic search over chat history and docs via ChromaDB embeddings.
- **ReAct Multi-Agent Simulation**: Orchestrate sub-agents (Retriever, Reasoner, Generator) for robust task handling.
- **Sandboxed Tools**: File ops, code execution (with NumPy/SymPy/Torch), Git, web search—all confined to your Pi.
- **Pi-Optimized**: Runs buttery-smooth on Pi 5's 8GB RAM; WAL-mode SQLite for concurrency without hiccups.
- **Nerd Perks**: Vision support, code linting for 10+ languages, NTP-synced timestamps (because accuracy > approximations).

## 🚀 Quick Start

### Prerequisites
- Raspberry Pi 5 (8GB recommended for embeddings).
- Python 3.12+.
- xAI API key (free tier works for Grok-3-mini).
- Optional: LangSearch API key for web tools.

### Installation
1. Clone the repo:
   ```
   git clone https://github.com/yourusername/apex-orchestrator.git
   cd apex-orchestrator
   ```

2. Set up virtual env and deps (Pi-friendly; no heavy installs):
   ```
   python -m venv venv
   source venv/bin/activate  # On Pi: use bash
   pip install -r requirements.txt
   ```
   *requirements.txt* (auto-generated from script):
   ```
   streamlit==1.38.0
   openai==1.51.0
   passlib[bcrypt]==1.7.4
   python-dotenv==1.0.1
   ntplib==0.6.5
   pygit2==1.14.1
   requests==2.32.3
   black==24.8.0
   numpy==1.26.4
   sentence-transformers==3.1.1
   torch==2.4.1  # CPU-only for Pi
   jsbeautifier==1.15.1
   pyyaml==6.0.2
   sqlparse==0.5.1
   beautifulsoup4==4.12.3
   chromadb==0.5.11
   ```

3. Config (.env):
   ```
   XAI_API_KEY=your_xai_key_here
   LANGSEARCH_API_KEY=your_langsearch_key_here  # Optional for web search
   ```

4. Fire it up:
   ```
   streamlit run app.py --server.port 8501
   ```
   Access at `http://raspberrypi.local:8501` (or Pi's IP).

### First Run
- Register/login (SQLite-backed users).
- Select a prompt (e.g., "tools-enabled.txt" for agent mode).
- Chat away! Enable tools in sidebar for file/code wizardry.

Pro Tip: On Pi, overclock to 2.7GHz for Grok-4 dreams (but watch thermals—your Pi's not a volcano).

## 🎯 Features

| Feature | Description | Nerd Factor |
|---------|-------------|-------------|
| **Streaming Chat** | Real-time Grok responses with tool loops (max 3 iterations to dodge infinite recursion). | Handles vision uploads—analyze Pi cam snaps mid-convo. |
| **Advanced Memory (EAMS)** | Hierarchical storage: Episodic (raw logs) + Semantic (Grok-summarized) via ChromaDB. Prune low-salience entries like a digital Marie Kondo. | Embeddings with all-MiniLM-L6-v2; cosine sim for relevance boosts. |
| **Tool Arsenal** | 15+ sandboxed tools: FS ops, REPL code exec, Git, DB queries, linting, mock APIs, web search. | Stateful REPL persists vars—build a NumPy sim across turns. |
| **Multi-Agent Orchestrator** | Simulate Apex Orchestrator: ReAct/CoT/ToT with sub-agents (Retriever, Reasoner, etc.). Self-checks confidence scores. | ToT branching prunes paths; memory as shared state—feels like a mini-LangChain on steroids. |
| **UI Polish** | Neon-gradient theme, dark mode toggle, chat bubbles, history search. | Custom CSS for Pi's tiny screen; expander for "deep thoughts" (tool traces). |
| **Expandability** | Add prompts to `./prompts/`, extend TOOLS list, hook custom sub-agents. | YAML/JSON configs; plugin-like tool schema for easy swaps. |

Humor Alert: If you ask it to "commit" bad code, it'll Git it done... with a diff roast. 😏

## 🏗️ Architecture & Flows

Apex Orchestrator's brain is a RAG-enhanced ReAct loop, all local except API calls. Here's the magic in diagrams.

### RAG Flow: Augmenting Generation with Memory
```mermaid
graph TD
    A[User Query] --> B{Embed Query?}
    B -->|Yes| C[SentenceTransformer Encode]
    C --> D["ChromaDB Query\nTop-K Similar Memories"]
    D --> E[Augment Prompt w/ Retrieved Docs]
    B -->|No| E
    E --> F["Grok API Call\nw/ Tools if Enabled"]
    F --> G["Stream Response\nw/ Tool Feedback Loop"]
    G --> H["Consolidate & Embed\nNew Memory Entry"]
    H --> I[Prune Low-Salience\nvia Decay Factor]
    style A fill:#f9f,stroke:#333
    style H fill:#bbf,stroke:#333
```

- **Why RAG?** Fights hallucinations by pulling from your chat history/docs. Semantic summaries keep it snappy.

### ReAct Flow: Reasoning + Acting in Agent Mode
```mermaid
flowchart TD
    A["User Query"] --> B["Task Initialization\n(Main Agent - ToT Planning)"]
    
    B --> B1["Parse Query: Goal, Constraints, Domain"]
    B1 --> B2["Decompose into 3-5 Subtasks\n(e.g., Retrieve → Reason → Generate → Validate)"]
    B2 --> B3["ToT Branching: Generate 2-3 Plans\n(Quick/Deep/Balanced)\nEvaluate & Prune Best Plan"]
    B3 --> B4["Assign Subtasks to Subagents\n(Core: 1-3; Optional: 4-5 if Complex)"]
    B4 --> B5["Self-Check Confidence ≥0.8?\nIf <0.8: Reprompt with Examples"]
    B5 -->|Yes| B6["Output Internal Plan as JSON\nMemory Insert State Key"]
    B5 -->|No| B5
    
    B6 --> C["Subtask Execution\n(Simulate Subagents via ReAct Loops)\nParallel where Possible\nReport Outputs to State via Memory"]
    
    subgraph Subagents ["Subagent Simulation"]
        C --> D1["Subagent 1: Retriever\n(Always Active)\nReAct: Think (Refine Query) → Act (Memory Retrieve → Web Search → FS Read) → Observe (Parse) → Reflect (Relevance Check)\nSelf-Check: Gaps? Fallback\nOutput: Data with Confidence & Metrics"]
        
        C --> D2["Subagent 2: Reasoner\n(Always Active)\nReAct: Think (ToT Branches) → Act (Code Exec → DB Query → Shell/Git) → Observe (Log/Handle Errors) → Reflect (Cross-Verify, Prune Branches)\nSelf-Check: Hallucination Detect\nOutput: Analysis with Confidence"]
        
        C --> D3["Subagent 3: Generator\n(Always Active)\nReAct: Think (CoT Outline) → Act (FS Write → Code Lint → Memory Insert) → Observe (Review) → Reflect (Citations)\nSelf-Check: Completeness Score\nOutput: Artifacts (Text/Code/Files)"]
        
        D1 --> E1{"High-Stakes?"}
        D2 --> E2{"High-Stakes?"}
        D3 --> E3{"High-Stakes?"}
        E1 -->|Yes e.g., Code/Research| D4["Subagent 4: Validator\nReAct: Think (Checks List) → Act (Memory Retrieve → Code Tests → Web Fact-Check) → Observe/Reflect (Error Rate <10%)\nSelf-Check: <0.7? Loop to Reasoner\nOutput: Fixes & Delta Score"]
        E2 -->|Yes| D4
        E3 -->|Yes| D4
        
        E1 -->|Iterative/Meta?| D5["Subagent 5: Optimizer\nReAct: Think (ToT Analyze Logs) → Act (Memory Prune → FS List/Cleanup) → Observe/Reflect (Update Plan)\nSelf-Check: Post-Task Only\nOutput: Refinements & Meta Learn Log"]
        E2 -->|Yes e.g., Long Sessions| D5
        E3 -->|Yes| D5
    end
    
    D1 --> F
    D2 --> F
    D3 --> F
    D4 --> F
    D5 --> F
    
    F["Memory Update: Sub-Outputs to State\n(e.g., {'agent': 'Retriever', 'output': ..., 'confidence': 0.9})"] --> G["Aggregation & Iteration\n(Main Agent - Global ReAct)"]
    
    G --> G1["Query State via Memory\nMerge Outputs (Weighted by Confidence)"]
    G1 --> G2["Global ReAct:\nThink (Assess Progress)\nAct (Route to Subagent or Terminate)\nObserve (Update State)\nReflect (End-to-End Score)"]
    G2 --> G3{"Progress Complete?\nGlobal Confidence ≥0.7?"}
    G3 -->|No, <5 Cycles| G["Iterate: Invoke Next Subagent"]
    G3 -->|No, ≥5 Cycles| G4["Abort: 'Insufficient Data;\nSuggest Query Refinement'"]
    G3 -->|Yes| H["Finalization & Output"]
    G4 --> H
    
    H --> H1["Polish Response: Structured\n(Summary, Key Outputs, Evidence w/ Citations, Next Steps)"]
    H1 --> H2["Cleanup: Run Optimizer Subagent\nMemory Insert Final Summary\nPrune State"]
    H2 --> H3["Output to User\n(Concise, Actionable;\nNote Generated Files)"]
    
    style A fill:#e1f5fe
    style H3 fill:#c8e6c9
    style G4 fill:#ffcdd2
```

- **ReAct Loop**: Cycles Think-Act-Observe-Reflect per sub-agent. Caps at 5 cycles—because even AIs need coffee breaks.
- **Multi-Agent Sim**: No extra processes; all in one Grok call via structured tools. Scalable to 5 sub-agents for epic quests.

## 🛠️ Customization & Expansion

- **Prompts**: Drop .txt files in `./prompts/` (e.g., "coder.txt" for dev mode). Edit/save via UI.
- **Tools**: Extend `TOOLS` list in script—add schemas for new functions (e.g., Pi GPIO integration).
- **Memory**: Tune ChromaDB path; add salience decay for long-running bots.
- **Pi Tweaks**: For headless: `streamlit run app.py --server.headless true`. Monitor with `htop`—Grok-3-mini sips ~500MB.

Want to fork for IoT? Hook `code_execution` to RPi.GPIO. The sandbox awaits.

## 🤝 Contributing

1. Fork & PR.
2. Lint: `black . && isort .`.
3. Test: `pytest` (add tests dir for REPL mocks).
4. Pi-Test: Run on real hardware—emulators lie.

Issues? Open one. Stars? Fuel the Pi. <3

## 📄 License

MIT—use it, tweak it, Pi it.

*Built with ❤️ on a Pi 5. Questions? Ping @yourhandle. May your embeddings cluster tightly and your loops never infinite.*
