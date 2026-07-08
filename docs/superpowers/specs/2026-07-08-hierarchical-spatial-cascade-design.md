# Hierarchical Spatial Cascade Design

**Date:** 2026-07-08  
**Status:** Drafted for review  
**Scope:** Larger-step follow-on experiment after the rejected near-miss ranker, rejected cold-zone backfill, and rejected district-aware dispatch policy

## Goal

Test whether a **coarse-to-fine H3 hierarchy** can materially improve operational spatial ranking by reducing fine-cell label sparsity before the final dispatch ordering step.

The target outcome is not a prettier probability story. The target outcome is a better **top-50 dispatch list across 30m, 1h, and 2h**.

## Context

The current accepted baseline is still:

- the existing activity stage
- the existing fine-cell spatial stage
- the accepted recent-prior blend
- the accepted dispatch quota reranker

Recent smaller experiments have already failed:

- near-miss ranker target
- hard-negative oversampling
- score-semantic changes
- cold-zone district backfill
- per-(target_time, district) dispatch quota

That failure pattern suggests the next improvement area should not be another minor feature tweak or another small reranking rule. The likely deeper problem is that **resolution-9 exact labels are too sparse**, so the model is being asked to rank too many fine cells too early.

The earlier error analysis already pointed at this possibility: a coarser H3 objective may reduce sparsity enough to learn stable top-k ordering while keeping resolution 9 as the final action surface.

## Recommendation

Run the next experiment as a **hierarchical spatial cascade**, using **H3 coarse parents above the current res9 prediction cells**.

The recommended experiment structure is:

1. **Phase A: parent-size sweep**
   - baseline
   - soft cascade with **res8** parent
   - soft cascade with **res7** parent

2. **Phase B: architecture bake-off on the winning parent size**
   - baseline
   - soft cascade
   - full hierarchical probability tree
   - candidate-generation cascade

This keeps the comparison large in effect size but bounded in scope.

## Why H3 parent layers instead of district

The parent layer should be geometric, not administrative.

Why:

- district-aware reranking already failed
- district-relative features already exist in the fine model
- H3 parent cells attack label sparsity directly
- H3 parents preserve local spatial geometry better than district boundaries

The first parent comparison should therefore be:

- **res8 parent**: one level above res9, usually 7 res9 children per parent
- **res7 parent**: one additional level coarser, to test whether a larger aggregation unlocks materially better routing

## Non-goals

This experiment must **not**:

- rewrite the activity stage
- introduce a district-first hierarchy
- change the accepted baseline before comparison
- silently replace `probability` semantics in the baseline path
- mix parent-size selection and architecture selection into one uncontrolled matrix
- bundle unrelated feature-engineering experiments into the same pass

## The three hierarchy variants

### 1. Soft cascade

This is the recommended first real hierarchy variant.

Behavior:

- predict coarse parent mass at res8 or res7
- keep multiple parents alive for each target time
- score child res9 cells only inside those live parents
- create a dispatch-oriented final score from:
  - activity-stage score
  - parent routing mass
  - child local score

This is a **routing and ranking system first**, not a strict probability tree.

### 2. Full hierarchical probability tree

Behavior:

- predict parent probability
- predict conditional child probability inside each parent
- multiply downward into a top-down normalized hierarchy

This is the cleanest probability story, but it also carries the largest calibration burden and the highest implementation risk.

### 3. Candidate-generation cascade

Behavior:

- use the parent stage only to shortlist viable child res9 cells
- pass surviving children to a fine scorer that stays close to the current baseline

This is the lowest-risk hierarchy arm, but also the least transformative.

## Experiment flow

### Shared flow

1. Keep the activity-stage model and split logic unchanged.
2. Build coarse parent labels from the existing fine-cell truth.
3. Run the hierarchy variant.
4. Produce a final dispatch-oriented ranking over res9 cells.
5. Compare against the accepted baseline on the same horizons and holdout structure.

### Phase A: parent-size sweep

Use only the **soft cascade** here.

Purpose:

- answer whether hierarchy helps at all
- answer whether **res8** or **res7** is the better parent size

Why only soft cascade in Phase A:

- avoids a 2 parent sizes x 3 hierarchy types matrix explosion
- isolates the coarsening question first
- keeps the first comparison cheap enough to reject quickly

### Phase B: architecture bake-off

Use the winning parent size from Phase A.

Purpose:

- compare the three hierarchy styles directly
- determine whether full normalization is worth the extra complexity

## Score and probability semantics

The experiment should separate **operational dispatch ranking** from **probability calibration** unless a variant explicitly proves it can do both.

That means:

- **soft cascade:** may use a dispatch score built from parent mass x child score
- **candidate-generation cascade:** may stay ranking-first
- **full probability tree:** must own the top-down normalized probability story

Do **not** force all three arms into identical probability semantics just for symmetry.

The baseline failure pattern already showed that changing score semantics too early can destroy working ranking behavior. This experiment should preserve that lesson.

## Diagnostics required

The current pipeline mostly shows the final miss. A hierarchy needs intermediate diagnostics so failures can be localized.

Each hierarchy arm should emit:

1. **Parent recall / parent mass coverage**
   - did the true parent survive?

2. **Child candidate recall before final ranking**
   - once parent routing happened, did the true res9 cell still remain available?

3. **Final dispatch precision@50**
   - did the actual operational top-50 improve?

4. **Exact artifact precision@50**
   - did the strict exact-cell artifact view hold up?

This lets the experiment answer:

- if parent routing fails, hierarchy is the wrong lever
- if parent routing succeeds but child recall fails, the candidate gate is too hard
- if parent and child recall succeed but final dispatch still fails, the fine scorer remains the bottleneck

## Acceptance gate

The hierarchy must improve the **whole stack**, not only 2h.

### Primary metric

- **dispatch precision@50** across `30m`, `1h`, and `2h`

### Guardrails

- exact artifact precision@50
- no material regression in `30m` or `1h`

### Decision rule

Keep a hierarchy variant only if:

1. it beats the accepted baseline on the primary metric in a way that improves the overall `30m` / `1h` / `2h` stack
2. it does not buy that win by materially damaging the shorter horizons

A 2h-only win does **not** count as success if 30m or 1h degrade.

## File boundaries

Likely production-code areas:

- `analysis/run_two_stage_experiment.py`
- `ghost_ranking_features.py`
- possibly a small helper module if parent/child mapping logic becomes noisy

Likely test areas:

- `tests/test_two_stage_experiment.py`
- `tests/test_ghost_ranking_features.py`
- focused diagnostics tests only where needed

The implementation should prefer **small additions around the current two-stage seam** rather than introducing a new parallel experiment framework.

## Testing strategy

### Unit tests

Add focused tests for:

- correct mapping from res9 cells to res8 and res7 parents
- parent-label construction
- candidate retention behavior in soft/candidate cascades
- probability mass conservation in the full tree
- no target-time leakage across hierarchy stages

### Experiment validation

For each phase:

1. run focused tests
2. rerun the experiment pipeline
3. rerun the spatial error analysis
4. compare against the accepted baseline

## Rollback rule

Reject the hierarchy line immediately if Phase A shows that neither res8 nor res7 improves the whole-stack gate against baseline.

If one parent size survives but Phase B shows none of the hierarchy variants beat baseline cleanly, reject the hierarchy architecture change and keep the current accepted baseline.

## Expected upside

If this works, it is a larger win than another local reranking tweak because it changes **when** the model is forced to distinguish among sparse fine cells.

Instead of asking the model to solve all discrimination at resolution 9 from the start, it can first learn:

- which coarse area is live
- then which fine child is best within that area

That is the first proposed change in this project that directly attacks the sparsity problem rather than working around it with another local heuristic.
