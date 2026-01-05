---
id: 0002
title: Create Plan for RAG Agent
stage: plan
date: 2025-12-29
surface: agent
model: gemini
feature: 005-rag-agent-qdrant
branch: 005-rag-agent-qdrant
user: user
command: |
  ---
  description: Execute the implementation planning workflow using the plan template to generate design artifacts.
  handoffs: 
    - label: Create Tasks
      agent: sp.tasks
      prompt: Break the plan into tasks
      send: true
    - label: Create Checklist
      agent: sp.checklist
      prompt: Create a checklist for the following domain...
  ---

  ## User Input

  ```text
  $ARGUMENTS
  ```

  You **MUST** consider the user input before proceeding (if not empty).

  ## Outline

  1. **Setup**: Run `.specify/scripts/powershell/setup-plan.ps1 -Json` from repo root and parse JSON for FEATURE_SPEC, IMPL_PLAN, SPECS_DIR, BRANCH. For single quotes in args like "I'm Groot", use escape syntax: e.g 'I'\''m Groot' (or double-quote if possible: "I'm Groot").

  2. **Load context**: Read FEATURE_SPEC and `.specify/memory/constitution.md`. Load IMPL_PLAN template (already copied).

  3. **Execute plan workflow**: Follow the structure in IMPL_PLAN template to:
     - Fill Technical Context (mark unknowns as "NEEDS CLARIFICATION")
     - Fill Constitution Check section from constitution
     - Evaluate gates (ERROR if violations unjustified)
     - Phase 0: Generate research.md (resolve all NEEDS CLARIFICATION)
     - Phase 1: Generate data-model.md, contracts/, quickstart.md
     - Phase 1: Update agent context by running the agent script
     - Re-evaluate Constitution Check post-design

  4. **Stop and report**: Command ends after Phase 2 planning. Report branch, IMPL_PLAN path, and generated artifacts.

  ## Phases

  ### Phase 0: Outline & Research

  1. **Extract unknowns from Technical Context** above:
     - For each NEEDS CLARIFICATION → research task
     - For each dependency → best practices task
     - For each integration → patterns task

  2. **Generate and dispatch research agents**:

     ```text
     For each unknown in Technical Context:
       Task: "Research {unknown} for {feature context}"
     For each technology choice:
       Task: "Find best practices for {tech} in {domain}"
     ```

  3. **Consolidate findings** in `research.md` using format:
     - Decision: [what was chosen]
     - Rationale: [why chosen]
     - Alternatives considered: [what else evaluated]

  **Output**: research.md with all NEEDS CLARIFICATION resolved

  ### Phase 1: Design & Contracts

  **Prerequisites:** `research.md` complete

  1. **Extract entities from feature spec** → `data-model.md`:
     - Entity name, fields, relationships
     - Validation rules from requirements
     - State transitions if applicable

  2. **Generate API contracts** from functional requirements:
     - For each user action → endpoint
     - Use standard REST/GraphQL patterns
     - Output OpenAPI/GraphQL schema to `/contracts/`

  3. **Agent context update**:
     - Run `.specify/scripts/powershell/update-agent-context.ps1 -AgentType gemini`
     - These scripts detect which AI agent is in use
     - Update the appropriate agent-specific context file
     - Add only new technology from current plan
     - Preserve manual additions between markers

  **Output**: data-model.md, /contracts/*, quickstart.md, agent-specific file

  ## Key rules

  - Use absolute paths
  - ERROR on gate failures or unresolved clarifications

  ---

  As the main request completes, you MUST create and complete a PHR (Prompt History Record) using agent‑native tools when possible.

  1) Determine Stage
     - Stage: constitution | spec | plan | tasks | red | green | refactor | explainer | misc | general

  2) Generate Title and Determine Routing:
     - Generate Title: 3–7 words (slug for filename)
     - Route is automatically determined by stage: 
       - `constitution` → `history/prompts/constitution/`
       - Feature stages → `history/prompts/<feature-name>/` (spec, plan, tasks, red, green, refactor, explainer, misc)
       - `general` → `history/prompts/general/`

  3) Create and Fill PHR (Shell first; fallback agent‑native)
     - Run: `.specify/scripts/bash/create-phr.sh --title "<title>" --stage <stage> [--feature <name>] --json`
     - Open the file and fill remaining placeholders (YAML + body), embedding full PROMPT_TEXT (verbatim) and concise RESPONSE_TEXT.
     - If the script fails:
       - Read `.specify/templates/phr-template.prompt.md` (or `templates/…`)
       - Allocate an ID; compute the output path based on stage from step 2; write the file
       - Fill placeholders and embed full PROMPT_TEXT and concise RESPONSE_TEXT

  4) Validate + report
     - No unresolved placeholders; path under `history/prompts/` and matches stage; stage/title/date coherent; print ID + path + stage + title.
     - On failure: warn, don't block. Skip only for `/sp.phr`.


  /sp.plan Build Retrieval-Augmented AI Agent (agent.py)Objective:Create an AI Agent that answers questions using only retrieved book content.Plan:1. Implement everything in a single file: agent.py2. Initialize OpenAI Agents SDK with strict book-only instructions3. Connect to Qdrant and retrieve top-k chunks per query4. Inject retrieved context into the agent5. Generate grounded responses or fallback if no context found6. Add a simple CLI test interfaceConstraints:- Single file only (agent.py)- Python only- No FastAPI, frontend, or ingestion- No external knowledge or web accessDeliverable:- agent.py with agent logic, retrieval, and CLI testing
labels: ["rag", "agent", "plan"]
links:
  spec: "../../../../specs/005-rag-agent-qdrant/spec.md"
  ticket: null
  adr: null
  pr: null
files:
  - ../../../../specs/005-rag-agent-qdrant/plan.md
  - ../../../../specs/005-rag-agent-qdrant/research.md
  - ../../../../specs/005-rag-agent-qdrant/data-model.md
  - ../../../../specs/005-rag-agent-qdrant/quickstart.md
tests:

---

## Prompt

/sp.plan Build Retrieval-Augmented AI Agent (agent.py)Objective:Create an AI Agent that answers questions using only retrieved book content.Plan:1. Implement everything in a single file: agent.py2. Initialize OpenAI Agents SDK with strict book-only instructions3. Connect to Qdrant and retrieve top-k chunks per query4. Inject retrieved context into the agent5. Generate grounded responses or fallback if no context found6. Add a simple CLI test interfaceConstraints:- Single file only (agent.py)- Python only- No FastAPI, frontend, or ingestion- No external knowledge or web accessDeliverable:- agent.py with agent logic, retrieval, and CLI testing

## Response snapshot

I have successfully generated the planning documents for the RAG Agent feature.

**Branch:** `005-rag-agent-qdrant`
**Implementation Plan:** `specs/005-rag-agent-qdrant/plan.md`

**Generated Artifacts:**
- `specs/005-rag-agent-qdrant/research.md`
- `specs/005-rag-agent-qdrant/data-model.md`
- `specs/005-rag-agent-qdrant/quickstart.md`

The planning phase is complete. The next step is to break the plan into actionable tasks using the `/sp.tasks` command.

## Outcome

- ✅ Impact: High - created the foundational technical plan for the RAG agent.
- 🧪 Tests: None added.
- 📄 Files: Created 4 files: the plan, research, data model, and quickstart guide.
- 👉 Next prompts: `/sp.tasks` to create implementation tasks.
- 🧠 Reflection: The process was successful. I resolved ambiguities by inspecting existing code, which is a reliable method. The manual workarounds for the PowerShell scripts continue to be necessary.

## Evaluation notes (flywheel)

- Failure modes observed: `read_file` failed on an ignored file, requiring a `run_shell_command` with `type` as a workaround.
- Graders run and results (PASS/FAIL): null
- Prompt variant (if applicable): null
- Next experiment (smallest change to try): null
