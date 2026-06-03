# Self-Evolving AI Agent System — Discovery & Implementation Plan

## Executive Summary

The current workspace ("Locate") is a well-structured FastAPI service that wraps NVIDIA's LocateAnything-3B vision-language model behind an OpenAI-compatible API. The goal is to extend this into a **self-evolving AI agent system** — a framework where agents autonomously improve their own capabilities through reflection, memory, tool use, and prompt evolution.

This plan adds ~15 new files across 4 new sub-packages within `src/`, modifies 3 existing files, and introduces 8 new dependencies. Estimated effort: **3–4 weeks for a single engineer** (phased delivery).

---

## 1. Current State Analysis

### 1.1 What Exists Today

| File / Package | Purpose | Maturity |
|---|---|---|
| `src/locateanything_api/config.py` | Pydantic-settings config (env-driven) | Production-ready |
| `src/locateanything_api/schemas.py` | Request/response models (Chat, Responses, Models) | Production-ready |
| `src/locateanything_api/backend.py` | Model loading, prompt prep, generation, bbox parsing | Production-ready |
| `src/locateanything_api/app.py` | FastAPI app factory with streaming support | Production-ready |
| `src/locateanything_api/server.py` | Uvicorn runner entry point | Production-ready |
| `server.py` (root) | Legacy inline server (unused in prod) | Deprecated |
| `main.py` | Placeholder `Hello from locate!` | Unused |
| `src/decord/` | Stub for video loading on macOS ARM64 | Stub only |
| `Dockerfile` | GPU-capable production container | Production-ready |

### 1.2 Existing Dependencies

```
accelerate, fastapi, httpx, torch, torchvision, transformers,
peft, pillow, pydantic-settings, python-multipart, uvicorn[standard],
opencv-python-headless, lmdb, numpy
```

### 1.3 Architecture Strengths

- **Clean separation**: config → schemas → backend → app → server
- **Thread-safe model loading** with locks
- **OpenAI-compatible API** (streaming + non-streaming)
- **Environment-driven configuration** (LOCATE_* prefix)
- **Production Docker setup** with health checks

### 1.4 Gaps for Self-Evolving Agents

| Gap | Impact |
|---|---|
| No memory/persistence layer | Agents can't learn across sessions |
| No tool-execution framework | Agents can't act on the world |
| No reflection/evaluation loop | Agents can't self-improve |
| No agent lifecycle management | No spawning, monitoring, or retirement |
| No multi-agent coordination | Agents can't collaborate |
| No prompt mutation system | No evolution mechanism |
| No safety/alignment constraints | Unbounded self-modification risk |

---

## 2. Proposed Architecture: Self-Evolving Agent Framework

### 2.1 Core Design Principles

1. **Evolution through reflection** — Agents periodically review their own performance and mutate their system prompts, tool selections, and strategies.
2. **Memory-first design** — Short-term (working), long-term (episodic), and semantic memory stores enable learning.
3. **Bounded autonomy** — Evolution is constrained by safety policies; agents cannot modify their own safety constraints.
4. **Composability** — Each component (memory, tools, evolution, orchestration) is an independent module.
5. **Integration, not replacement** — The existing LocateAnything backend becomes the "vision brain"; new components add cognitive capabilities.

### 2.2 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     API Layer (FastAPI)                      │
│  /v1/agents/*   /v1/chat/*   /v1/tools/*   /v1/evolve/*    │
└──────────┬──────────────┬───────────────┬───────────────────┘
           │              │               │
┌──────────▼──────────┐ ┌─▼─────────────┐ │ ┌──────────────────┐
│   Agent Orchestrator │ │ LocateAnything│ │ │  Evolution Engine │
│   (lifecycle mgmt)  │ │   Backend     │ │ │  (reflection,     │
│                     │ │ (vision LLM)  │ │ │   mutation, eval)  │
└──┬───────────┬──────┘ └───────────────┘ │ └────────┬─────────┘
   │           │                          │          │
┌──▼────┐ ┌───▼──────────┐ ┌─────────────▼──┐ ┌─────▼────────┐
│Memory  │ │ Tool Registry│ │  Safety Layer  │ │ Prompt Store │
│ Store  │ │ & Executor   │ │  & Guardrails  │ │ (versioned)  │
└──┬─────┘ └──────────────┘ └────────────────┘ └──────────────┘
   │
┌──▼──────────────┐
│  Storage Backend │
│  (SQLite + JSON) │
└──────────────────┘
```

### 2.3 Component Breakdown

#### Component 1: Memory System (`src/agent/memory/`)

| Module | Responsibility |
|---|---|
| `working_memory.py` | In-context window (current conversation), capped at N turns |
| `episodic_memory.py` | Past interaction logs (what happened, outcomes, rewards) |
| `semantic_memory.py` | Accumulated knowledge (facts, patterns, learned strategies) |
| `vector_store.py` | Embedding-based retrieval for long-term memory recall |
| `memory_manager.py` | Unified interface: store, recall, consolidate, forget |

**Storage**: SQLite for structured data + JSON files for episodic logs. Optional: LMDB for vector embeddings (already a dependency).

#### Component 2: Tool System (`src/agent/tools/`)

| Module | Responsibility |
|---|---|
| `base.py` | Abstract `Tool` class with `name`, `description`, `execute()`, `schema` |
| `registry.py` | Dynamic tool registration, discovery, and listing |
| `executor.py` | Sandboxed tool execution with timeout and error handling |
| `builtin/` | Built-in tools: web_search, code_exec, file_read, file_write, shell_exec |
| `llm_tool.py` | Wraps LocateAnything as a callable tool for other agents |

Tools expose JSON Schema for function-calling, enabling agents to select and invoke them via the LLM's tool-use capability.

#### Component 3: Evolution Engine (`src/agent/evolution/`)

| Module | Responsibility |
|---|---|
| `prompt_store.py` | Versioned system prompts with diffs and metadata |
| `mutator.py` | Strategies for prompt mutation (add/remove instructions, rephrase, specialize) |
| `evaluator.py` | Runs agent on benchmark tasks, scores performance |
| `selector.py` | Selection strategy: tournament, elitist, random restart |
| `evolution_loop.py` | Main loop: evaluate → mutate → evaluate → select → repeat |
| `fitness.py` | Fitness functions: accuracy, latency, token efficiency, user satisfaction |

**Key insight**: The evolution engine itself does NOT modify Python code. It evolves **prompts and tool configurations** — safe, reversible, auditable changes.

#### Component 4: Agent Core (`src/agent/core/`)

| Module | Responsibility |
|---|---|
| `agent.py` | `Agent` class: prompt + memory + tools + config |
| `agent_config.py` | Pydantic model for agent configuration (versioned) |
| `orchestrator.py` | Spawns, monitors, and manages agent lifecycles |
| `conversation.py` | Manages multi-turn conversation with memory integration |
| `reflection.py` | Post-interaction self-evaluation and insight extraction |

#### Component 5: Safety Layer (`src/agent/safety/`)

| Module | Responsibility |
|---|---|
| `guardrails.py` | Hard constraints on agent behavior (token limits, tool restrictions) |
| `audit_log.py` | Immutable log of all agent actions and mutations |
| `rate_limiter.py` | Per-agent rate limiting on actions and API calls |
| `content_filter.py` | Input/output content safety filtering |

#### Component 6: API Extensions (`src/agent/api/`)

| Endpoint | Method | Purpose |
|---|---|---|
| `/v1/agents` | GET | List all agents |
| `/v1/agents` | POST | Create a new agent |
| `/v1/agents/{id}` | GET | Get agent details + config |
| `/v1/agents/{id}` | PATCH | Update agent config |
| `/v1/agents/{id}` | DELETE | Retire an agent |
| `/v1/agents/{id}/chat` | POST | Chat with an agent (auto-injects memory) |
| `/v1/agents/{id}/memory` | GET | Query agent's memory |
| `/v1/agents/{id}/evolve` | POST | Trigger evolution cycle |
| `/v1/agents/{id}/history` | GET | Evolution history (prompts, scores) |
| `/v1/tools` | GET | List available tools |
| `/v1/tools` | POST | Register a new tool |

---

## 3. New Dependencies

```toml
# Add to pyproject.toml
dependencies = [
    # ... existing ...
    "aiosqlite>=0.20.0",       # async SQLite for memory stores
    "sentence-transformers>=3.0.0",  # embeddings for vector memory
    "pydantic>=2.0",           # already indirect, make explicit
    "tenacity>=8.0",           # retry logic for LLM calls
    "structlog>=24.0",         # structured logging for audit trail
    "rich>=13.0",              # CLI agent management
]
```

---

## 4. Detailed Implementation Plan

### Phase 1: Foundation (Week 1)

**Goal**: Memory system + Agent core — agents can remember across sessions.

| Step | Task | Files Created/Modified | Est. Hours |
|---|---|---|---|
| 1.1 | Create `src/agent/` package skeleton | `src/agent/__init__.py` | 0.5 |
| 1.2 | Implement Memory Store (SQLite-backed) | `src/agent/memory/__init__.py`, `working_memory.py`, `episodic_memory.py`, `semantic_memory.py`, `memory_manager.py` | 6 |
| 1.3 | Implement Agent core class | `src/agent/core/__init__.py`, `agent.py`, `agent_config.py`, `conversation.py` | 4 |
| 1.4 | Create agent API endpoints | `src/agent/api/__init__.py`, `routes.py` | 3 |
| 1.5 | Wire into existing FastAPI app | Modify `src/locateanything_api/app.py` | 1 |
| 1.6 | Add memory consolidation background task | `src/agent/memory/consolidation.py` | 2 |

**Exit Criteria**: Can create an agent, chat with it, and it remembers context across multiple API calls.

### Phase 2: Tool System (Week 2)

**Goal**: Agents can execute tools and interact with the outside world.

| Step | Task | Files Created/Modified | Est. Hours |
|---|---|---|---|
| 2.1 | Tool base class and registry | `src/agent/tools/__init__.py`, `base.py`, `registry.py`, `executor.py` | 4 |
| 2.2 | Built-in tools (web search, code exec, file ops) | `src/agent/tools/builtin/__init__.py`, `web_search.py`, `code_exec.py`, `file_ops.py` | 5 |
| 2.3 | Tool API endpoints | Extend `src/agent/api/routes.py` | 2 |
| 2.4 | Integrate tool calling into agent conversation loop | Modify `src/agent/core/conversation.py` | 3 |
| 2.5 | Add safety rate limiting | `src/agent/safety/__init__.py`, `rate_limiter.py`, `audit_log.py` | 3 |

**Exit Criteria**: Agent can select and invoke tools during conversation; all actions are audited.

### Phase 3: Evolution Engine (Week 3)

**Goal**: Agents can self-improve through prompt evolution.

| Step | Task | Files Created/Modified | Est. Hours |
|---|---|---|---|
| 3.1 | Versioned prompt store | `src/agent/evolution/__init__.py`, `prompt_store.py` | 3 |
| 3.2 | Prompt mutator with multiple strategies | `src/agent/evolution/mutator.py` | 4 |
| 3.3 | Fitness evaluation framework | `src/agent/evolution/evaluator.py`, `fitness.py` | 5 |
| 3.4 | Evolution loop orchestrator | `src/agent/evolution/evolution_loop.py`, `selector.py` | 4 |
| 3.5 | Reflection module for post-interaction learning | `src/agent/core/reflection.py` | 3 |
| 3.6 | Evolution API endpoints | Extend `src/agent/api/routes.py` | 2 |

**Exit Criteria**: Agent can be triggered to run an evolution cycle; prompt versions are tracked with scores.

### Phase 4: Multi-Agent & Polish (Week 4)

**Goal**: Multiple agents can coordinate; production hardening.

| Step | Task | Files Created/Modified | Est. Hours |
|---|---|---|---|
| 4.1 | Agent orchestrator (multi-agent lifecycle) | `src/agent/core/orchestrator.py` | 4 |
| 4.2 | Safety guardrails and content filtering | `src/agent/safety/guardrails.py`, `content_filter.py` | 3 |
| 4.3 | CLI for agent management | `src/agent/cli.py` | 3 |
| 4.4 | Integration tests | `tests/test_memory.py`, `test_tools.py`, `test_evolution.py`, `test_api.py` | 6 |
| 4.5 | Documentation and README update | `README.md`, inline docs | 2 |

**Exit Criteria**: Multiple agents run concurrently; full test suite passes; documented.

---

## 5. Key Design Decisions

### 5.1 Evolution = Prompt Mutation (Not Code Mutation)

**Decision**: Agents evolve by mutating their system prompts, tool configurations, and behavioral parameters — NOT by modifying their own Python source code.

**Rationale**:
- **Safety**: Code self-modification is extremely dangerous and hard to audit
- **Reversibility**: Prompt mutations can be rolled back trivially
- **Observability**: Every mutation is a diff against a versioned prompt store
- **Effectiveness**: Research shows prompt engineering is the highest-leverage optimization for LLM agents

### 5.2 Memory Architecture

```
┌──────────────────────────────────────────┐
│              Memory Hierarchy             │
├──────────────┬───────────────────────────┤
│  Working     │ Current conversation      │
│  (volatile)  │ Last N turns, in context  │
├──────────────┼───────────────────────────┤
│  Episodic    │ Past interactions stored  │
│  (persistent)│ with outcomes & rewards   │
├──────────────┼───────────────────────────┤
│  Semantic    │ Consolidated knowledge    │
│  (persistent)│ Patterns, facts, strategies│
└──────────────┴───────────────────────────┘
```

Consolidation runs periodically: working → episodic → semantic, with summarization via the LLM.

### 5.3 Integration with Existing LocateAnything Backend

The existing backend becomes a **vision tool** that agents can invoke:

```python
class LocateAnythingTool(Tool):
    name = "locate_anything"
    description = "Analyze an image and detect/describe objects with bounding boxes"
    
    async def execute(self, image_url: str, prompt: str) -> str:
        # Delegates to existing LocateAnythingBackend.complete()
        ...
```

### 5.4 Configuration Extension

Extend the existing `Settings` class with agent-specific env vars:

```python
class Settings(BaseSettings):
    # ... existing fields ...
    
    # Agent settings
    agent_memory_db: str = "agent_memory.db"
    agent_max_working_turns: int = 20
    agent_tool_timeout_seconds: float = 30.0
    agent_evolution_enabled: bool = False
    agent_evolution_generations: int = 5
    agent_evolution_population: int = 4
    agent_safety_max_actions_per_hour: int = 100
    agent_embedding_model: str = "all-MiniLM-L6-v2"
```

---

## 6. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Evolution produces degenerate prompts | Medium | High | Fitness threshold must improve or revert; elitist selection preserves best |
| Memory store grows unbounded | Medium | Medium | TTL-based expiry; consolidation with summarization; storage quotas |
| Tool execution introduces security holes | Low | Critical | Sandboxed execution; allowlist of tools; rate limiting; audit log |
| Agent bypasses safety constraints | Low | Critical | Safety layer is external, not agent-modifiable; immutable audit trail |
| Performance degradation from memory recall | Medium | Medium | Vector index with caching; async recall; configurable depth |
| Scope creep delays delivery | High | Medium | Strict phased approach; Phase 1 delivers standalone value |

---

## 7. Success Criteria

| Metric | Target |
|---|---|
| Agent creation to first chat | < 2 seconds |
| Memory recall latency | < 200ms (p95) |
| Tool execution latency | < 5 seconds (p95) |
| Evolution cycle completion | < 5 minutes for 5 generations |
| Test coverage | > 80% for new modules |
| API backward compatibility | 100% — all existing endpoints unchanged |

---

## 8. File Tree (New + Modified)

```
src/
├── locateanything_api/          # EXISTING — minimal changes
│   ├── __init__.py              # (no change)
│   ├── app.py                   # MODIFY: register agent API routes
│   ├── backend.py               # (no change)
│   ├── config.py                # MODIFY: add agent settings
│   ├── schemas.py               # (no change)
│   └── server.py                # (no change)
├── agent/                       # NEW — all agent code
│   ├── __init__.py
│   ├── cli.py                   # CLI management tool
│   ├── core/
│   │   ├── __init__.py
│   │   ├── agent.py             # Agent class
│   │   ├── agent_config.py      # Agent configuration model
│   │   ├── conversation.py      # Multi-turn conversation manager
│   │   ├── orchestrator.py      # Multi-agent lifecycle
│   │   └── reflection.py        # Post-interaction self-evaluation
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── working_memory.py    # In-context memory
│   │   ├── episodic_memory.py   # Interaction logs
│   │   ├── semantic_memory.py   # Consolidated knowledge
│   │   ├── memory_manager.py    # Unified memory interface
│   │   └── consolidation.py     # Background memory consolidation
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract Tool class
│   │   ├── registry.py          # Tool registration & discovery
│   │   ├── executor.py          # Sandboxed execution
│   │   └── builtin/
│   │       ├── __init__.py
│   │       ├── web_search.py
│   │       ├── code_exec.py
│   │       └── file_ops.py
│   ├── evolution/
│   │   ├── __init__.py
│   │   ├── prompt_store.py      # Versioned prompt storage
│   │   ├── mutator.py           # Prompt mutation strategies
│   │   ├── evaluator.py         # Performance evaluation
│   │   ├── fitness.py           # Fitness functions
│   │   ├── selector.py          # Selection strategies
│   │   └── evolution_loop.py    # Main evolution orchestrator
│   ├── safety/
│   │   ├── __init__.py
│   │   ├── guardrails.py        # Hard behavioral constraints
│   │   ├── audit_log.py         # Immutable action logging
│   │   ├── rate_limiter.py      # Per-agent rate limiting
│   │   └── content_filter.py    # Input/output filtering
│   └── api/
│       ├── __init__.py
│       └── routes.py            # All /v1/agents/* and /v1/tools/* routes
└── decord/                      # EXISTING — no change
    └── __init__.py

# Root files
main.py                          # MODIFY: optional CLI entry point
Makefile                         # MODIFY: add agent-related targets
pyproject.toml                   # MODIFY: new dependencies + entry point
README.md                        # MODIFY: document agent system
```

**New files**: ~30  
**Modified files**: 5  
**Estimated total new code**: ~3,500–4,500 lines

---

## 9. Rollback Strategy

1. **Feature flag**: `LOCATE_AGENT_ENABLED=false` disables all agent routes
2. **Database isolation**: Agent memory uses separate SQLite DB; deleting it removes all agent state
3. **Evolution rollback**: Every prompt mutation is versioned; `git revert` + prompt store reset restores baseline
4. **API backward compatibility**: No existing endpoint signatures change; new endpoints are purely additive

---

## 10. Next Steps

Before implementation begins, the following questions should be resolved:

1. **Evolution target**: Should agents evolve toward a specific benchmark/task, or is open-ended "improvement" acceptable?
2. **Multi-agent**: Is multi-agent coordination a hard requirement for Phase 1, or can it wait for Phase 4?
3. **Deployment**: Will this run locally (single machine) or in a distributed setup? This affects memory backend choice.
4. **Budget constraints**: `sentence-transformers` adds ~500MB; are there constraints on image/dependency size?
5. **Data retention**: How long should agent memory be retained? (Regulatory/compliance implications)
