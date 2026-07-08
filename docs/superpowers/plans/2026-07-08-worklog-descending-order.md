# Worklog Descending Order Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `/worklog` detailed pane display worklog entries newest-first while keeping the summary cards tied to the newest parsed entry.

**Architecture:** Keep the summary contract unchanged and add one dashboard-specific API field for descending-order rendered HTML. Reuse the existing `split_worklog_entries()` parser and `render_worklog_markdown()` sanitizer, then switch the page binding from the source-order HTML field to the new descending-order field.

**Tech Stack:** Python, standard library, existing `markdown` package, inline browser JavaScript, pytest

---

## File structure

- **Modify:** `analysis/dashboard_service.py`
  - Add one helper that builds descending-order markdown from parsed `##` entries.
  - Extend `api_worklog()` to return `detail_html`.
  - Update the `/worklog` page script so `#worklogHtml` uses the new field.
- **Modify:** `tests/test_dashboard_service.py`
  - Add one API test for descending detail order.
  - Update the page contract test to require `payload.detail_html`.
- **Modify:** `WORKLOG.md`
  - Append one milestone entry after the feature is verified.
- **Modify:** `WORKLOG_INDEX.md`
  - Add the new worklog entry title, brief description, and exact line span after the `WORKLOG.md` update.

### Task 1: Add descending-order API output

**Files:**
- Modify: `tests/test_dashboard_service.py:300-360`
- Modify: `analysis/dashboard_service.py:412-548`

- [ ] **Step 1: Write the failing API test**

Add this test below the existing latest-entry API test:

```python
def test_worklog_endpoint_returns_descending_detail_html(tmp_path):
    worklog = tmp_path / "WORKLOG.md"
    worklog.write_text(
        "\n".join(
            [
                "# Worklog",
                "",
                "## 2026-07-07 Older entry",
                "",
                "Current objective:",
                "- Finish the first pass.",
                "",
                "## 2026-07-08 Newer entry",
                "",
                "Current objective:",
                "- Ship the newest change.",
            ]
        ),
        encoding="utf-8",
    )
    original = dict(service.PATHS)
    service.PATHS["worklog"] = worklog
    try:
        status, headers, body = service.dispatch("GET", "/api/worklog")
    finally:
        service.PATHS.clear()
        service.PATHS.update(original)

    assert status == 200
    assert headers["Content-Type"] == "application/json; charset=utf-8"
    payload = json.loads(body)
    assert payload["latest_title"] == "2026-07-08 Newer entry"
    assert payload["current_objective"] == ["Ship the newest change."]
    assert payload["html"].index("2026-07-07 Older entry") < payload["html"].index("2026-07-08 Newer entry")
    assert payload["detail_html"].index("2026-07-08 Newer entry") < payload["detail_html"].index("2026-07-07 Older entry")
```

- [ ] **Step 2: Run the targeted API test and confirm it fails**

Run:

```bash
.venv\Scripts\python.exe -m pytest tests/test_dashboard_service.py::test_worklog_endpoint_returns_descending_detail_html -q
```

Expected:

- `FAIL`
- a `KeyError` or assertion failure because `detail_html` does not exist yet

- [ ] **Step 3: Write the minimal API implementation**

Add this helper near `render_worklog_markdown()`:

```python
def render_worklog_detail_markdown(text: str) -> str:
    entries = split_worklog_entries(text)
    if not entries:
        return render_worklog_markdown(text)
    reversed_markdown = "\n\n".join(
        f"## {title}\n\n{body}".strip() if body else f"## {title}"
        for title, body in reversed(entries)
    )
    return render_worklog_markdown(reversed_markdown)
```

Then extend the return payload in `api_worklog()`:

```python
    return {
        "exists": exists,
        "path": display_path(path),
        "entry_count": len(entries),
        "modified_at": modified_at,
        "latest_title": latest_title,
        "current_objective": worklog_section_items(latest_body, "Current objective"),
        "test_results": worklog_section_items(latest_body, "Test results"),
        "blockers": worklog_section_items(latest_body, "Blockers"),
        "next_steps": worklog_section_items(latest_body, "Next steps"),
        "raw_markdown": raw_markdown,
        "text": text,
        "html": render_worklog_markdown(text) if exists else "",
        "detail_html": render_worklog_detail_markdown(text) if exists else "",
    }
```

- [ ] **Step 4: Run the focused API tests and confirm they pass**

Run:

```bash
.venv\Scripts\python.exe -m pytest tests/test_dashboard_service.py -k "worklog_endpoint" -q
```

Expected:

- `PASS`
- the new descending-order API test passes without breaking the existing markdown and sanitization tests

- [ ] **Step 5: Commit the API change**

Run:

```bash
git add tests/test_dashboard_service.py analysis/dashboard_service.py
git commit -m "feat: add descending worklog detail html" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

### Task 2: Bind the detailed pane to the descending-order field

**Files:**
- Modify: `tests/test_dashboard_service.py:375-410`
- Modify: `analysis/dashboard_service.py:1160-1188`

- [ ] **Step 1: Update the page contract test first**

Change the existing page-binding test so it requires the descending-order field:

```python
def test_worklog_page_uses_descending_detail_html_field():
    status, headers, body = service.dispatch("GET", "/worklog")

    assert status == 200
    assert headers["Content-Type"] == "text/html; charset=utf-8"
    assert "<h2>Full worklog</h2>" in body
    assert 'class="logWrap"' in body
    assert 'id="worklogHtml"' in body
    assert "payload.detail_html" in body
    assert "innerHTML = payload.detail_html" in body or "payload.detail_html || payload.html" in body
    assert ".markdown-body table" in body
    assert ".markdown-body ul" in body
    assert ".markdown-body ol" in body
```

Also update the rich-shell contract test assertion from:

```python
assert "payload.html" in body
```

to:

```python
assert "payload.detail_html" in body
```

- [ ] **Step 2: Run the targeted page tests and confirm they fail**

Run:

```bash
.venv\Scripts\python.exe -m pytest tests/test_dashboard_service.py -k "worklog_page" -q
```

Expected:

- `FAIL`
- the page still references `payload.html`

- [ ] **Step 3: Make the minimal page binding change**

Replace the current detailed-pane assignment:

```javascript
  $('worklogHtml').innerHTML = payload.html || '<p class="empty">WORKLOG.md not found.</p>';
```

with:

```javascript
  $('worklogHtml').innerHTML =
    payload.detail_html || payload.html || '<p class="empty">WORKLOG.md not found.</p>';
```

This keeps backward compatibility for any intermediate payloads while making the page prefer the descending-order field.

- [ ] **Step 4: Run the focused page tests and confirm they pass**

Run:

```bash
.venv\Scripts\python.exe -m pytest tests/test_dashboard_service.py -k "worklog_page" -q
```

Expected:

- `PASS`
- the page now binds to `payload.detail_html`

- [ ] **Step 5: Commit the page binding change**

Run:

```bash
git add tests/test_dashboard_service.py analysis/dashboard_service.py
git commit -m "feat: show worklog detail newest first" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

### Task 3: Verify the feature and log the milestone

**Files:**
- Modify: `WORKLOG.md`
- Modify: `WORKLOG_INDEX.md`

- [ ] **Step 1: Run the focused end-to-end worklog test slice**

Run:

```bash
.venv\Scripts\python.exe -m pytest tests/test_dashboard_service.py -k "worklog" -q
```

Expected:

- `PASS`
- all `/api/worklog` and `/worklog` tests pass together

- [ ] **Step 2: Smoke-test the live page locally**

Run:

```bash
$env:PYTHONPATH='.'; .venv\Scripts\python.exe -c "from analysis.dashboard_service import run; run(host='127.0.0.1', port=8766)"
```

Then in a second shell run:

```bash
curl http://127.0.0.1:8766/worklog
```

Expected:

- the server starts without import errors
- `curl` returns HTTP 200 and the page HTML contains `payload.detail_html`

- [ ] **Step 3: Append the worklog milestone entry**

Append this exact entry to `WORKLOG.md`:

```md
## 2026-07-08 Worklog descending detail order

Current objective:
- Display the dashboard worklog detail pane in descending order without changing latest-entry summary behavior.

Files inspected:
- `analysis/dashboard_service.py`
- `tests/test_dashboard_service.py`

Files changed:
- `analysis/dashboard_service.py`
- `tests/test_dashboard_service.py`

Commands run:
- `.venv\Scripts\python.exe -m pytest tests/test_dashboard_service.py -k "worklog" -q`
- `curl http://127.0.0.1:8766/worklog`

Test results:
- Focused worklog dashboard tests passed.
- Local `/worklog` smoke test returned HTTP 200 with `payload.detail_html` in the page shell.

Blockers:
- None.

Next steps:
- Keep the descending-order detail pane and latest-entry summary behavior aligned in future `/worklog` changes.
```

- [ ] **Step 4: Update `WORKLOG_INDEX.md` with the computed line span**

Run this command to print the exact index line after the new worklog entry has been appended:

```bash
@'
from pathlib import Path
lines = Path("WORKLOG.md").read_text(encoding="utf-8").splitlines()
title = "## 2026-07-08 Worklog descending detail order"
start = next(i for i, line in enumerate(lines, start=1) if line == title)
end = len(lines)
print(f"- `2026-07-08 Worklog descending detail order` — Switched the dashboard detail pane to newest-first while preserving latest-entry summary cards. `WORKLOG.md` lines {start}-{end}")
'@ | .venv\Scripts\python.exe -
```

Then copy that exact printed line into `WORKLOG_INDEX.md` after the current last index item.

- [ ] **Step 5: Commit the verification and worklog updates**

Run:

```bash
git add WORKLOG.md WORKLOG_INDEX.md
git commit -m "docs: log worklog detail ordering change" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```
