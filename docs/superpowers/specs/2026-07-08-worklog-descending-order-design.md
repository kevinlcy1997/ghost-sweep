# Worklog Descending Order Design

**Date:** 2026-07-08  
**Status:** Drafted for review  
**Scope:** Show the detailed worklog pane on `/worklog` in descending order while keeping the summary cards tied to the newest entry

## Goal

Make the **full worklog pane** on `/worklog` read newest-to-oldest without changing how the summary cards determine the latest entry.

## User decisions

The user explicitly chose:

1. keep the visual change scoped to the detailed full-worklog pane
2. keep the summary cards driven by the newest entry
3. avoid a browser-based visual companion for this design step

## Current state

Today the dashboard has two different worklog views with different purposes:

- the summary panel uses the latest parsed `##` entry for `Current objective`, `Blockers`, `Test results`, and `Next steps`
- the full worklog pane renders the entire `WORKLOG.md` file as sanitized markdown HTML in source order

That means the detailed pane currently reads oldest-to-newest even though the summary panel already emphasizes the newest entry.

## Recommendation

Keep the existing summary behavior unchanged and add a **second rendered HTML field** specifically for the detailed pane.

Build that new field by:

1. splitting the worklog into `##` entries with the existing parser
2. reversing those entries
3. reassembling them into markdown
4. rendering and sanitizing that reversed markdown the same way the existing full-log HTML is rendered

This is the smallest safe change because it preserves the current summary contract and avoids fragile client-side DOM rewriting.

## Non-goals

This change should **not**:

- reverse the summary-card logic
- rewrite `WORKLOG.md` on disk
- reorder content that appears outside the `##` entry structure
- replace the current markdown rendering pipeline
- introduce client-side markdown parsing or post-render DOM transforms

## Proposed behavior

### API behavior

Keep the existing `/api/worklog` payload fields intact, including:

- `latest_title`
- `current_objective`
- `test_results`
- `blockers`
- `next_steps`
- `html`

Add one new field:

- `detail_html` — rendered markdown HTML for the descending-order detailed pane

Field responsibilities:

- `html` remains the canonical full-file rendering in source order for compatibility and debugging
- `detail_html` becomes the dashboard-specific newest-first rendering for the detailed pane

### Page behavior

Keep the current `/worklog` page shell, summary cards, refresh cadence, and status labels unchanged.

Change only the detailed pane binding:

- replace `payload.html` with `payload.detail_html` for `#worklogHtml`
- keep the existing fallback empty-state message if the worklog file is missing

The resulting page behavior is:

- left-side summary still reflects the newest entry
- right-side detailed pane shows entries from newest to oldest

## Parsing and rendering model

Use the existing `split_worklog_entries(text)` helper as the ordering boundary.

That keeps the reversal constrained to top-level worklog entries rather than raw text lines. The API should not attempt to reverse arbitrary markdown blocks or a whole-file string because that would break structure such as headings, code fences, and introductory content.

If the file has no `##` entries, the safest behavior is to reuse the current rendered full-log HTML instead of inventing a reversal rule for non-standard content.

## Components and file boundaries

### `analysis/dashboard_service.py`

Expected changes:

- add a small helper that assembles descending-order markdown from parsed entries
- extend `api_worklog()` to return `detail_html`
- switch the `/worklog` page script so `#worklogHtml` uses `payload.detail_html`

### `tests/test_dashboard_service.py`

Add focused tests for:

- `/api/worklog` returning descending-order `detail_html` while preserving latest-entry summary fields
- `/worklog` binding the detailed pane to `payload.detail_html`

## Data flow

1. `/api/worklog` reads `WORKLOG.md`
2. the service parses `##` entries in source order
3. the summary fields continue using the last entry as the newest entry
4. the service builds `detail_html` from reversed parsed entries
5. the browser fetches `/api/worklog`
6. the summary cards keep using structured latest-entry fields
7. the detailed pane renders `detail_html`

## Error handling

- If `WORKLOG.md` is missing, keep the current explicit missing-file behavior.
- If the file has no parseable `##` entries, fall back to the existing rendered `html` for `detail_html`.
- If markdown rendering fails because its dependency is unavailable, fail explicitly the same way the current renderer would fail.

## Testing strategy

### Focused tests

Run the existing focused dashboard-service tests covering:

- `/api/worklog`
- `/worklog`

### New assertions

Add tests proving:

1. `detail_html` renders a newer entry before an older entry
2. `latest_title` and the structured summary lists still come from the newest entry
3. the `/worklog` page script consumes `payload.detail_html`

## Acceptance criteria

The change is accepted when all of the following are true:

1. the summary cards still describe the newest parsed worklog entry
2. the detailed worklog pane displays entries newest-first
3. the page keeps its current live refresh behavior
4. focused dashboard-service tests pass

## Why this is the right size

- It satisfies the user request without destabilizing the already-sensitive summary logic.
- It reuses the existing parser and markdown sanitizer.
- It avoids brittle raw-text reversal and brittle client-side DOM surgery.
- It keeps the dashboard contract explicit by separating source-order HTML from dashboard-order HTML.
