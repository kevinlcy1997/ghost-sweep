# Worklog Writing Standard Design

**Date:** 2026-07-08  
**Status:** Drafted for review  
**Scope:** Standardize how agents write `WORKLOG.md` entries without adding a validator yet

## Goal

Make `WORKLOG.md` entries predictable enough that humans and the `/worklog` page can scan them quickly, while keeping the writing method lightweight.

## User decisions

The user explicitly chose:

1. **template + section order only**
2. no heavy writing-rule system
3. a **reusable template file** as the rollout mechanism

## Current state

The repo already says `WORKLOG.md` must include:

- Current objective
- Files inspected
- Files changed
- Commands run
- Test results
- Blockers
- Next steps

But that requirement lives in instruction text, not in one reusable source file. The result is section drift, optional headings appearing in different places, and entries that are harder to parse consistently.

## Recommendation

Standardize `WORKLOG.md` with:

1. one canonical template file in the repo
2. one fixed required section order
3. a small `AGENTS.md` instruction pointing agents to that template

Do **not** add a validator yet.

## Non-goals

This design does **not**:

- add a machine-enforced linter or validator
- require strict prose rules like max bullet length
- change the `/worklog` API or page
- replace `WORKLOG.md` with JSON/YAML
- forbid optional sections entirely

## Proposed standard

### Entry structure

Each milestone entry uses one `##` heading followed by this required section order:

1. `Current objective`
2. `Files inspected`
3. `Files changed`
4. `Commands run`
5. `Test results`
6. `Blockers`
7. `Next steps`

Optional sections remain allowed, but they always appear **after** the required core sections.

### Writing method

Keep the method lightweight:

- every required section is a bullet list
- if a section is empty, write `- None.`
- file paths go in backticks
- commands go in backticks
- one milestone = one `##` entry

This is enough structure for consistency without turning worklog writing into a style-policing exercise.

## Rollout

### Canonical template file

Add a reusable template file to the repo as the single source of truth for worklog writing.

The template should show:

- the required heading order
- the expected bullet-list structure
- a minimal example of `- None.`

### `AGENTS.md`

Update `AGENTS.md` so it tells agents to use the template file when writing or updating `WORKLOG.md`.

That moves the standard from “remember this paragraph” to “copy this exact structure.”

## File boundaries

### New template file

Add one dedicated file for the reusable worklog template.

Recommended responsibility:

- define the canonical entry shape only

### `AGENTS.md`

Only reference the template and required usage.

Recommended responsibility:

- tell agents when to update `WORKLOG.md`
- tell agents to use the template

### `WORKLOG.md`

Existing entries stay as historical records.

Only new or updated entries need to follow the standard immediately.

## Error handling

- If an agent has no content for a required section, use `- None.` instead of omitting the section.
- If an entry needs extra detail such as `Resume instructions`, place it after the required sections.
- If older entries do not match the new standard, do not rewrite history unless there is a specific cleanup task.

## Acceptance criteria

The standard is in place when:

1. the repo has one canonical worklog template file
2. `AGENTS.md` points agents to that template
3. the required section order is explicit
4. new entries can be written without guessing heading order
5. no validator is required for the first rollout

## Why this is the right size

- It solves the current drift with the smallest possible process change.
- It gives agents one copy-paste source of truth.
- It keeps the writing method human-friendly.
- It leaves room for a validator later only if the lightweight method proves insufficient.
