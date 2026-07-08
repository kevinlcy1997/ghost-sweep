# AGENTS.md

This file defines how coding agents should work in this repository. It is intended for Codex / VS Code agent workflows.

## Token and context budget

`<budget:token_budget>`
1000000
`</budget:token_budget>`

Use the available context deliberately. Do not load large files, generated files, lockfiles, build artifacts, or dependency folders unless they are directly relevant.

## Token-saving policy

Actively minimize unnecessary token usage while preserving correctness.

- Prefer targeted file reads over broad repository scans.
- Use search tools to locate relevant code before opening files.
- Do not repeatedly reread files unless they may have changed.
- Summarize findings instead of pasting large file contents into the conversation.
- Avoid loading full logs, generated files, dependency folders, lockfiles, or large datasets unless directly required.
- Keep subagent prompts narrow and include only the context needed for that subtask.
- Do not spawn subagents when a single targeted read or edit is cheaper and sufficient.
- Do not load optional skills unless the task clearly benefits from them.
- Prefer the smallest correct fix over exploratory rewrites.
- Keep final responses concise unless the user asks for detail.

## Long task continuity

For tasks that may span more than one session, maintain `WORKLOG.md`.

Before reading the full worklog, search `WORKLOG_INDEX.md` first and only open the relevant line span from `WORKLOG.md`.

When writing or updating a worklog entry, use `WORKLOG_TEMPLATE.md`.

Update `WORKLOG.md` after every meaningful milestone. Include:

- Current objective
- Files inspected
- Files changed
- Commands run
- Test results
- Blockers
- Next steps

Periodically call `codex-cli-usage` to check session limit, context, and time remaining. Plan the effort and task scope based on the remaining limit.

## Default operating mode

Default to a careful, repo-aware workflow:

1. Understand the user request and identify the smallest safe scope.
2. Inspect only the files needed to understand the current behavior.
3. Make a concise plan for non-trivial work.
4. Implement the smallest correct change.
5. Run relevant tests, type checks, lint checks, or build checks when available.
6. Review the final diff before responding.
7. Summarize what changed, what was verified, and any remaining risks.

Do not perform broad rewrites, unrelated formatting, dependency upgrades, or architecture changes unless explicitly requested or clearly necessary.

## Available Codex skills

This Codex session may have specialized skills available. Use them deliberately when they materially improve correctness, speed, or verification. Do not load every skill by default.

### Most relevant skills for this repository

Prefer these skills when applicable:

- `cavecrew` — delegate focused code investigation, build, or review work to subagents.
- `github:gh-fix-ci` and related `github:*` skills — investigate failing CI, PR feedback, branch publishing, and GitHub workflow issues.
- `superpowers:systematic-debugging` — use for non-trivial bugs, failing tests, regressions, or unclear root causes.
- `superpowers:verification-before-completion` — use before claiming a task is done, especially after code edits.
- `superpowers:*` — use targeted Superpowers skills for planning, TDD, code review, git worktrees, finishing branches, and safe branch workflows.
- `data-scientist` — use for ML, data science, model evaluation, feature engineering, or experiment-related work.
- `data-analyst` — use for SQL, statistics, analysis, reporting, and data validation tasks.
- `ml-ops-engineer` — use for model deployment, monitoring, pipelines, and production ML concerns.
- `memory` — use to persist and retrieve durable project context in the local Codex memory folder when a task benefits from continuity across sessions.

### Conditional skills

Use these only when the task clearly calls for them:

- `build-web-apps:*` and `frontend-design` — frontend, React, Next.js, UI, styling, or frontend testing tasks.
- `browser:control-in-app-browser` or `chrome:control-chrome` — browser automation, UI verification, or end-to-end testing.
- `openai-docs` — current OpenAI, Codex, or API documentation lookup.
- `github:*` — GitHub repo, PR, issue, CI, and review-comment workflows.
- `spreadsheets:Spreadsheets` — spreadsheet creation, cleanup, xlsx/csv/tsv analysis, or reporting.
- `pdf`, `docx`, `pptx`, and related document skills — document creation, editing, rendering, or extraction tasks.
- `render:*` — Render deployments, blueprints, cron jobs, workers, Docker, Postgres, Redis/Key Value, environment variables, domains, and scaling.
- `skill-creator`, `superpowers:writing-skills`, `skill-installer`, and `plugin-creator` — only when explicitly creating, improving, installing, or scaffolding skills/plugins.

### Skill usage rules

- Load the smallest relevant skill set for the task.
- Prefer repo-specific evidence over generic skill guidance when they conflict.
- Use skills as operating procedures, not as a replacement for reading the actual code.
- If a skill suggests verification, run the relevant verification before finalizing whenever possible.
- Do not invoke document, browser, deployment, or connector skills unless the user request requires them.
- Do not let skill usage cause broad, unrelated exploration.

## Model routing and subagent policy

The goal is to balance cost, token usage, speed, and implementation quality.

### Main agent default

Use `GPT-5.5 low reasoning` as the default main thread for:

- User interaction
- Task interpretation
- Repo-level planning
- Deciding whether to delegate work
- Integrating subagent findings
- Implementation of normal coding tasks
- Final review and final response

The main agent must preserve the project-level big picture. Do not use a small model as the primary coordinator for tasks that require architectural judgement, cross-file reasoning, or final merge decisions.

### When to use lower-cost subagents

Use `GPT-5.4-mini` or another lower-cost model only for bounded, read-heavy, easily verifiable subtasks such as:

- Locating relevant files, functions, routes, components, tests, or call sites
- Summarizing a module or directory
- Comparing existing implementation patterns
- Collecting TODOs, duplicated logic, or dead code candidates
- Reading logs or test failures and summarizing them
- Checking documentation consistency
- Producing a concise inventory of affected files

Lower-cost subagents should usually not edit code. They should return findings for the main agent to verify.

### When to escalate model or reasoning level

Escalate to `GPT-5.5 medium/high reasoning` or a stronger implementation model when the task involves:

- Ambiguous or hard-to-reproduce bugs
- Cross-module or cross-layer refactors
- Changes spanning more than five meaningful files
- Architecture, data model, API contract, or state-management decisions
- Authentication, authorization, payment, security, privacy, or data-loss risk
- Database migrations or data correctness
- Performance-critical logic
- Failing tests where the root cause is not obvious
- A previous failed attempt by a lower-cost model

Prefer paying for stronger reasoning once over creating multiple weak attempts that require expensive repair.

## Subagent delegation rules

Before spawning a subagent, the main agent must define a narrow task with a clear expected output.

When using subagents, prefer the `cavecrew` skill for focused investigation, build, or review tasks when it is available and appropriate. Subagents should still follow the model-routing policy in this file.

Each subagent prompt should include:

- Goal
- File, directory, or module scope where possible
- What to inspect
- What not to change
- Required output format
- Whether code edits are allowed
- Confidence level requirement

Recommended subagent output format:

```md
## Findings

### Files inspected
- `path/to/file`

### Relevant findings
- Finding with file path and line reference where possible.

### Existing patterns to follow
- Pattern summary.

### Risks or unknowns
- Risk or unknown.

### Recommended next step
- Recommended action.

### Confidence
High / Medium / Low