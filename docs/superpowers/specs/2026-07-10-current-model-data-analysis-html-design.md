# Current Model Data Analysis HTML Design

**Date:** 2026-07-10  
**Status:** Drafted for review  
**Scope:** Create a static HTML page for the current-model data analysis and store it in a dated reports folder

## Goal

Turn the completed current-model data analysis into a single static HTML page that is easy to open locally, archive by date, and share as a snapshot.

## User decisions

The user explicitly chose:

1. text-only design review with no browser visual companion
2. a combined layout: dashboard summary first, detailed write-up below
3. a static snapshot instead of a live page
4. a dated folder layout under `analysis\reports\YYYY-MM-DD\`

## Recommendation

Build one self-contained HTML file at:

- `analysis\reports\2026-07-10\current-model-data-analysis.html`

This should contain embedded CSS and any tiny helper JavaScript inline so the file opens directly from disk with no server and no external dependencies.

This is the smallest useful format because it preserves the analysis snapshot exactly as reviewed, avoids a multi-file site, and sets a clean naming convention for future dated reports.

## Non-goals

This change should **not**:

- create a live dashboard that rereads current files on load
- depend on a web server, CDN, or local API
- split the report into multiple HTML files
- rerun model training as part of page generation
- invent new metrics beyond the analysis already completed

## Content structure

The page should be organized in three layers.

### 1. Header

Include:

- page title
- generation date
- source scope note saying the report is a static snapshot of the current analysis
- short subtitle naming the accepted model design: Stage 1 city-activity gating plus Stage 2 spatial ranking

### 2. Dashboard summary

Put the highest-signal numbers first as KPI cards and compact visuals.

Recommended KPI cards:

- raw alert count
- usable enriched events
- active H3 zones
- Stage 1 `30m` positive rate
- Stage 2 `30m` positive rate
- Stage 2 sampled negative-to-positive ratio

Recommended summary sections:

- raw feed field completeness
- recent daily alert counts
- reconstructed region / district distribution
- Stage 1 label balance by horizon
- Stage 2 label balance by horizon
- Stage 2 feature sparsity / quality highlights

### 3. Full write-up

Below the dashboard summary, render the full narrative in report form:

- Executive read
- Raw feed analysis
- Reconstructed geography analysis
- Current model design
- Stage 1 analysis
- Stage 2 analysis
- Feature quality and sparsity
- Feature separation
- Risks and caveats
- What / So What / Now What

## Data source contract

The page should be generated from the already-computed analysis results, not by scraping text out of chat.

Use the same analysis inputs that produced the current findings:

- `ghost_alerts.json`
- reconstructed geo/admin enrichment from `ghost_ranking_features.py`
- Stage 1 activity-table analysis from `ghost_activity_features.py`
- Stage 2 spatial-table analysis from `ghost_ranking_features.py`

The generation path can rerun lightweight profiling code to embed the report data, but it should stay within the current accepted model design and not add new experiment logic.

## Visualization approach

Keep the page fully static and portable.

Recommended visual treatment:

- KPI cards for the headline numbers
- simple inline SVG or pure HTML bars for distributions
- compact tables for horizon balance and top regions/districts
- highlighted callout boxes for key findings and caveats

Avoid charting libraries. The page only needs lightweight visuals that are stable when opened directly from disk.

## Styling

Use a clean report style:

- light background
- dark text
- muted accent colors
- responsive single-column narrative with a compact card grid near the top
- readable tables with sticky or emphasized headers if easy

The page should feel like a static analytical report, not an application dashboard.

## File and folder behavior

Expected output convention:

- parent folder: `analysis\reports`
- dated child folder: `analysis\reports\YYYY-MM-DD`
- report file: a stable descriptive name, starting with `current-model-data-analysis.html`

If the dated folder does not exist, create it.

If the report file already exists for the same date, overwrite it with the new snapshot rather than creating versioned duplicates. The date folder already provides the archive boundary.

## Components and file boundaries

### New report generator

Add one focused generator path that:

1. assembles the analysis data
2. formats the dashboard summary sections
3. formats the full write-up
4. writes the final HTML file

Keep this in one file unless an obvious existing helper already fits. This is a one-report output, not a framework.

### Existing analysis code

Reuse existing project functions for:

- zone enrichment
- Stage 1 table construction
- Stage 2 table construction
- target naming helpers

Do not duplicate feature logic in the HTML generator.

## Error handling

Fail explicitly if:

- `ghost_alerts.json` is missing
- the alert payload cannot be parsed
- the analysis tables come back empty

If a specific section cannot be computed, surface that section as unavailable in the HTML rather than silently dropping it.

## Testing strategy

The smallest useful verification is:

1. generate the HTML file
2. confirm the file exists at the dated path
3. confirm key strings are present in the HTML:
   - report title
   - at least one KPI label
   - at least one Stage 1 metric
   - at least one Stage 2 metric
   - the `What / So What / Now What` section

This can be one focused test or one small script-level check. No browser automation needed.

## Acceptance criteria

The change is accepted when all of the following are true:

1. the report is written to `analysis\reports\2026-07-10\current-model-data-analysis.html`
2. the page opens as a standalone local HTML file
3. the page begins with dashboard-style summary content
4. the page includes the full written analysis below the summary
5. the page reflects the current accepted model design and current analysis numbers
6. no external dependencies are required to view it

## Why this is the right size

- It gives the user the requested deliverable without building a reporting system.
- It keeps the result portable and archivable.
- It reuses the existing analysis code instead of forking the model logic.
- It creates a clear convention for future dated HTML snapshots under `analysis\reports`.
