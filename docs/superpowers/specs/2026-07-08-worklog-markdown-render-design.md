# Worklog Markdown Render Design

**Date:** 2026-07-08  
**Status:** Drafted for review  
**Scope:** Render `WORKLOG.md` as real markdown in the `/worklog` page while keeping the current summary cards and live refresh behavior

## Goal

Make the **full worklog pane** on `/worklog` read like rendered markdown instead of raw source text.

## User decisions

The user explicitly chose:

1. keep the current summary cards
2. render only the **full worklog pane** as markdown
3. support richer markdown features
4. use a **server-side Python markdown package**, not a browser-side renderer

## Current state

Today the page already has the desired high-level layout:

- summary cards at the top
- latest-entry summary on the left
- full worklog pane on the right
- polling every 5 seconds from `/api/worklog`

The missing piece is that the full worklog content is still inserted as raw text into a `<pre>` block.

## Recommendation

Use a **server-side Python markdown package** to convert `WORKLOG.md` into HTML inside `api_worklog()`, then render that HTML inside a styled markdown container on `/worklog`.

This is the smallest change that satisfies the user’s richer-rendering requirement while preserving:

- the existing `/worklog` structure
- the current polling model
- the existing summary-card behavior

## Non-goals

This change should **not**:

- replace the page with markdown-only content
- remove the summary cards
- move markdown rendering into browser-side JavaScript
- add a full CMS/editor workflow
- change `WORKLOG.md` structure
- add syntax-highlighting infrastructure unless the chosen markdown package already makes that trivial without extra dependencies

## Proposed behavior

### Page behavior

Keep the current page shell and cards as-is.

Change only the full worklog pane:

- replace the raw `<pre>` presentation with rendered markdown HTML
- keep the existing auto-refresh interval
- keep the existing manual refresh button
- continue showing the latest file update timestamp

### API behavior

Keep the existing `/api/worklog` payload shape and add one new field:

- `html` — rendered markdown HTML for the full worklog pane

Retain `text` in the payload so the raw source is still available for debugging and tests.

## Package choice

Use a small Python markdown package server-side.

Expected configuration:

- base markdown rendering
- table support
- fenced code block support
- sane list behavior

The preferred implementation is the Python `markdown` package with only the extensions needed for this page.

## Safety model

The renderer must not blindly trust raw HTML embedded in `WORKLOG.md`.

Planned rule:

1. escape raw HTML input first
2. run markdown conversion on the escaped text
3. only emit the HTML produced by the markdown renderer

This keeps markdown features such as headings, links, emphasis, lists, code blocks, and tables while preventing embedded HTML/script content from executing in the page.

## Components and file boundaries

### `requirements.txt`

Add the chosen markdown package.

### `analysis/dashboard_service.py`

Expected changes:

- import and configure the markdown renderer
- add a helper that converts worklog text to rendered HTML
- extend `api_worklog()` to return `html`
- update the `/worklog` page script to inject `payload.html` into the full worklog pane
- add markdown-specific CSS for:
  - headings
  - paragraphs
  - bullet lists
  - links
  - inline code
  - fenced code blocks
  - tables

### `tests/test_dashboard_service.py`

Add focused tests for:

- `/api/worklog` returning rendered HTML
- markdown links/emphasis/tables rendering as expected
- `/worklog` using the rendered HTML field instead of raw text-only rendering

## Data flow

1. `/api/worklog` reads `WORKLOG.md`
2. the service extracts:
   - latest entry title
   - summary sections
   - raw `text`
3. the service converts the full markdown text into `html`
4. the browser fetches `/api/worklog`
5. summary cards keep using the structured summary fields
6. the full pane renders `html`

## Error handling

- If `WORKLOG.md` is missing, keep the current explicit “not found” behavior.
- If the markdown package is missing from the environment, fail explicitly rather than silently downgrading to raw text rendering.
- If markdown content contains unsupported features, let the package render what it supports and leave the rest as normal text according to package behavior.

## Testing strategy

### Focused tests

Run the focused dashboard-service tests that cover:

- `/api/worklog`
- `/worklog`
- favicon/page shell behavior

### New assertions

Add tests proving:

1. headings become HTML headings
2. emphasis becomes rendered emphasis
3. markdown links become anchor tags
4. pipe tables render as HTML tables
5. the `/worklog` page script consumes the rendered HTML field

## Acceptance criteria

The change is accepted when all of the following are true:

1. `/worklog` still shows the current summary cards
2. the full worklog pane displays rendered markdown instead of raw source
3. links, emphasis, tables, and fenced code blocks display correctly
4. live refresh still works
5. focused dashboard-service tests pass

## Why this is the right size

- It satisfies the richer-rendering request directly.
- It avoids a browser-side rendering stack.
- It keeps the existing page structure that already tested well.
- It adds one dependency instead of a custom parser plus long-term maintenance burden.
