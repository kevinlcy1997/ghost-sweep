# Worklog Writing Standard Design

**Date:** 2026-07-08  
**Status:** Drafted for review  
**Scope:** Standardize how agents write `WORKLOG.md` entries and how they look up past entries without adding a validator yet

## Goal

Make worklog history predictable enough that:

- humans can scan entries quickly
- the `/worklog` page can display entries consistently
- agents can search an index first instead of reading the entire `WORKLOG.md`

## User decisions

The user explicitly chose:

1. **template + section order only**
2. no heavy writing-rule system
3. a **reusable template file** as the rollout mechanism
4. a separate **`WORKLOG_INDEX.md`** file containing title, brief description, and line range for each worklog entry

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

There is also no lightweight lookup layer, so an agent that wants historical context may need to read too much of `WORKLOG.md` just to find the right section.

## Recommendation

Standardize `WORKLOG.md` with:

1. one canonical template file in the repo
2. one fixed required section order
3. one `WORKLOG_INDEX.md` lookup file
4. a small `AGENTS.md` instruction pointing agents to the template and index

Do **not** add a validator yet.

## Non-goals

This design does **not**:

- add a machine-enforced linter or validator
- require strict prose rules like max bullet length
- change the `/worklog` API or page
- replace `WORKLOG.md` with JSON/YAML
- forbid optional sections entirely
- require rewriting all historical entries immediately

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

### Index structure

Add a separate `WORKLOG_INDEX.md` file.

Each index item represents one `##` entry in `WORKLOG.md` and contains:

1. the entry title
2. a brief one-line description
3. the line range for that entry in `WORKLOG.md`

Example shape:

```md
# Worklog Index

- `2026-07-08 Worklog Markdown Render` — Rendered `/worklog` markdown while preserving the richer shell. `WORKLOG.md` lines 210-268
- `2026-07-08 Worklog Writing Standard` — Standardized entry structure and added a reusable template + lookup index. `WORKLOG.md` lines 269-320
```

### Agent lookup method

When an agent needs historical worklog context, the standard lookup flow becomes:

1. search `WORKLOG_INDEX.md` first
2. identify the relevant title/description match
3. read only the referenced line span from `WORKLOG.md`
4. avoid scanning the entire worklog unless the index is missing or clearly stale

## Rollout

### Canonical template file

Add a reusable template file to the repo as the single source of truth for worklog writing.

The template should show:

- the required heading order
- the expected bullet-list structure
- a minimal example of `- None.`

### `WORKLOG_INDEX.md`

Add one maintained index file as the canonical lookup layer for historical worklog entries.

The index should be updated whenever a new top-level `##` worklog entry is added.

### `AGENTS.md`

Update `AGENTS.md` so it tells agents to:

1. use the template file when writing or updating `WORKLOG.md`
2. consult `WORKLOG_INDEX.md` before reading the full worklog

That moves the standard from “remember this paragraph” to “use these two files: template for writing, index for lookup.”

## File boundaries

### New template file

Add one dedicated file for the reusable worklog template.

Recommended responsibility:

- define the canonical entry shape only

### `WORKLOG_INDEX.md`

Add one dedicated file for worklog lookup.

Recommended responsibility:

- list entry titles
- give a one-line description per entry
- map each entry to its `WORKLOG.md` line range
- act as the first search target for agents

### `AGENTS.md`

Only reference the template and required usage.

Recommended responsibility:

- tell agents when to update `WORKLOG.md`
- tell agents to use the template
- tell agents to search the index before the full worklog

### `WORKLOG.md`

Existing entries stay as historical records.

Only new or updated entries need to follow the standard immediately.

The full worklog remains the source of detail; the index is only the lookup layer.

## Error handling

- If an agent has no content for a required section, use `- None.` instead of omitting the section.
- If an entry needs extra detail such as `Resume instructions`, place it after the required sections.
- If older entries do not match the new standard, do not rewrite history unless there is a specific cleanup task.
- If `WORKLOG_INDEX.md` is temporarily stale, agents may fall back to direct `WORKLOG.md` reading, but that should be treated as an exception, not the default method.

## Acceptance criteria

The standard is in place when:

1. the repo has one canonical worklog template file
2. the repo has one `WORKLOG_INDEX.md` file with title, brief description, and line range per entry
3. `AGENTS.md` points agents to the template and the index
4. the required section order is explicit
5. agents can find relevant history from the index before opening the full worklog
6. no validator is required for the first rollout

## Why this is the right size

- It solves the current drift with the smallest possible process change.
- It gives agents one copy-paste source of truth for writing and one lookup layer for reading.
- It keeps the writing method human-friendly.
- It saves tokens by avoiding full-log scans for routine lookups.
- It leaves room for a validator later only if the lightweight method proves insufficient.
