# Vibe Coding Philosophy — Agent-Driven Knowledge Architecture

> The complete development methodology for this project. This document defines how agents should work with the knowledge system to ensure reliability, consistency, and zero information loss across sessions.

## Overview

This project uses an **agent-driven knowledge base + key files architecture** for vibe coding. The core insight is that AI agents working on complex codebases need a structured memory system that persists across sessions, survives context resets, and prevents information distortion from summarization or compression.

## The Problem

When an AI agent works on a codebase:
1. **Context is finite** — LLM context windows have limits
2. **Context is transient** — new sessions start with no memory of prior work
3. **Summarization loses information** — compression inevitably drops details
4. **Drift accumulates** — small deviations compound over multiple sessions

Without a structured memory system, agents will:
- Repeat work already done
- Miss critical context from prior sessions
- Deviate from original user intent
- Hallucinate solutions that contradict existing decisions

## The Solution: Knowledge + State Architecture

```
AGENTS.md (entry point)
    │
    ├── .agent/                    ← Cross-session memory & task state
    │   ├── GOAL.md                   ← Direction (where are we going)
    │   ├── PLAN.md                   ← Path (how do we get there)
    │   ├── CHECKLIST.md              ← Quality bar (how to verify)
    │   ├── CHANGELIST.md             ← History (what changed and why)
    │   ├── DESIGN.md                 ← Design and docs (The design plan desired by the user)
    │   ├── NEXT_STEPS.md             ← User's plan guide (What does the user want to do next)
    │   ├── reference/                ← Methodology & norms (English)
    │   │   └── vibe-coding-philosophy.md  ← This file
    │   └── raw/                      ← Original intent (preserved verbatim)
    │
    └── *                          ← current project source code
```

## Design Principles

### 1. Information Must Persist

**Any valuable information must be written to files. Do not rely on context memory.**

- Context is temporary; files are permanent
- If you discover something important, write it down immediately
- Never assume "the next session will figure it out"
- Never assume "the agent can infer from context"

### 2. Raw Information Must Be Preserved Verbatim

**Original user input, ideas, and design intent must be stored without compression, summarization, or rewriting.**

- Store raw information in `.agent/raw/`
- Keep the original language (Chinese ideas in Chinese, English in English)
- Do not translate, rewrite, or "improve the expression"
- Summaries can exist in other files, but raw files must remain untouched
- Raw files are the **truth source**; everything else is derivative

### 3. Documents Are Memory

**The `.agent/` directory is the project's long-term memory system. Agents should rely on them as they would rely on their own memory.**

| File | Memory Type | Persistence |
|------|------------|-------------|
| `GOAL.md` | Long-term goal memory | Rarely changes |
| `PLAN.md` | Implementation path memory | Changes per phase |
| `CHECKLIST.md` | Quality review memory | Grows over time |
| `CHANGELIST.md` | Historical change memory | Append-only |
| `DESIGN.md` | Design documents and index | Rarely changes, user write |
| `NEXT_STEPS.md` | User's plan guide | User update |
| `raw/` | Original intent memory | Append-only, never modify |

### 4. Self-Reference Principle

**The principle "information must persist" must itself be persisted.**

This document exists because of this principle. A principle that is not recorded cannot be followed.

### 5. Progressive Disclosure

**Knowledge should be organized from macro to micro, from concept to detail.**

- New agents/developers start with overview, then drill down
- Direct jumps to details cause tunnel vision
- Progressive disclosure reduces cognitive load

## File Responsibilities

### AGENTS.md — Entry Point

**What it is**: The first file every agent reads. A compact orientation guide.

**What it contains**:
- Knowledge architecture overview
- Quick project facts (tech stack, commands, directories)
- Architecture summary (one paragraph)
- Current project goal (brief)
- Conventions and gotchas
- Links to deeper documentation

**What it does NOT contain**:
- Detailed code explanations (→ `raw/docs`)
- Full goal specification (→ `.agent/GOAL.md`)
- Original user intent (→ `.agent/raw/`)

**Update frequency**: Rarely. Only when project fundamentals change.

### .agent/GOAL.md — Direction

**What it is**: The source of truth for project direction.

**What it contains**:
- Overall objectives
- Goal breakdown (G1, G2, G3...)
- Current status of each goal
- Constraints and success criteria

**Agent behavior**:
- READ FIRST before making any changes
- Use it to validate that your work aligns with goals
- Do not work on anything not connected to a stated goal

### .agent/PLAN.md — Path

**What it is**: Phased implementation plan.

**What it contains**:
- Phase breakdown (Phase 0, 1, 2...)
- Tasks per phase
- Completion criteria per phase
- Risks and mitigations

**Agent behavior**:
- Check current phase before starting work
- Do not skip ahead to future phases
- Update phase status when completing work

### .agent/CHECKLIST.md — Quality Bar

**What it is**: Independent quality review criteria.

**What it contains**:
- Goal review checks
- Plan review checks
- Code review checks
- Feature verification checks
- Performance checks

**Agent behavior**:
- Use before marking tasks complete
- Acts as an independent reviewer (third-party perspective)
- Add new checks when new quality concerns arise

### .agent/CHANGELIST.md — History

**What it is**: Append-only change log.

**What it contains**:
- Date, type, scope of each change
- What changed and why
- Key decisions made
- Impact assessment

**Agent behavior**:
- Append an entry after every significant change
- Never delete entries
- Be specific about what changed

### .agent/raw/ — Original Intent

**What it is**: Verbatim preservation of user's original thoughts.

**What it contains**:
- `core-principles.md` — Core methodology principles (in original language)
- `initial-vision.md` — User's original vision and ideas (in original language)

**Agent behavior**:
- READ when you need to understand original intent
- NEVER modify these files
- These are the truth source for user intent
- If current work seems to conflict with intent, check here first

### .agent/reference/ — Methodology

**What it is**: Development methodology and agent behavior norms.

**What it contains**:
- This file (`vibe-coding-philosophy.md`)
- Future reference documents as needed

**Agent behavior**:
- READ to understand how to work with this project
- These are the rules of engagement

## Agent Workflow

### Starting a New Session

1. Read `AGENTS.md` → understand the project at a glance
2. Read `.agent/GOAL.md` → understand current objectives
3. Read `.agent/PLAN.md` → know how to achieve goals
3. Read `.agent/NEXT_STEPS.md` → understand what the user is going to do next
5. If confused about intent → read `.agent/raw/`

### During Work

1. Consult `.agent/DESIGN.md` for technical details
2. Consult `.agent/PLAN.md` for phase context
3. If making significant decisions → check `CHECKLIST.md`
4. If discovering new information → write it to appropriate file
5. If original intent is unclear → read `.agent/raw/`

### Completing Work

1. Verify against `CHECKLIST.md`
2. Append to `CHANGELIST.md`
3. Update relevant plan status `NEXT_STEPS.md` and `PLAN.md` if plan status changed
4. Update `GOAL.md` status if goal progressed

## Why This Works for Vibe Coding

**Vibe coding** relies on AI agents understanding context deeply and making good decisions autonomously. This system enables that by:

1. **Eliminating context loss** — everything important is on disk
2. **Preserving intent fidelity** — raw files prevent drift from original vision
3. **Providing decision framework** — GOAL → PLAN → TODO gives clear direction
4. **Enabling quality control** — CHECKLIST acts as independent reviewer
5. **Supporting progressive learning** — docs/ structure allows agents to learn the codebase systematically
6. **Ensuring continuity** — new sessions pick up exactly where old ones left off

## Ultimate Goal

The knowledge system exists to ensure this goal is achieved reliably, without drift, and without losing the original vision across the many agent sessions it will take to build it.
