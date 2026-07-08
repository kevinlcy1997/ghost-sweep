# Coarse-Layer Feature Screening Design

**Date:** 2026-07-08  
**Status:** Drafted for review  
**Scope:** Analysis-first follow-on experiment after the rejected hierarchical spatial cascade

## Goal

Test whether **coarse spatial context** adds useful signal to the current accepted `res9` ranking baseline **without** reintroducing a top-down routing hierarchy.

The target outcome is a cheap screen for whether `police_zone`, `district`, or `res8` should be kept as extra **features or priors** on top of the current fine-cell model.

## Context

The rejected hierarchy experiment still taught something useful:

- coarse `res8` structure helped `2h`
- `res7` was weaker than `res8`
- hierarchical routing damaged the shorter horizons that matter most for acceptance

That means the useful question is no longer “should prediction route through a hierarchy?” The useful question is “can coarse geography help **as context** while preserving the current fine-cell ordering?”

The current accepted baseline remains:

- existing activity stage
- existing fine-cell spatial stage
- accepted recent-prior blend
- accepted dispatch quota reranker

This experiment should stay close to that baseline and only test whether coarse layers add incremental signal.

## Recommendation

Run the next experiment as a **two-phase coarse-layer feature screen**:

1. **Phase A: analysis pass**
   - attach `police_zone`, `district`, and `res8` context to each `res9` training row
   - measure whether each coarse layer shows useful independent signal
   - stop immediately if the signal is weak

2. **Phase B: minimal feature ablations**
   - `baseline + res8`
   - `baseline + police_zone`
   - `baseline + district`
   - `baseline + all coarse layers`

Only run Phase B if Phase A shows meaningful short-horizon signal.

## Why this is better than another hierarchy

This is the lazy version that still answers the question.

Why:

- keeps the current `res9` action surface intact
- keeps the full child candidate set intact
- avoids parent gating and parent-to-child normalization
- lets `2h` borrow coarse structure without forcing `30m` and `1h` through a routing layer
- costs much less than another full hierarchy branch

The rejected hierarchy already showed that coarse signal can exist while coarse routing still fails. This design tests the signal directly.

## Non-goals

This experiment must **not**:

- add another parent-routing or tree-normalization path
- replace the current accepted baseline before comparison
- turn `district` into a hard grouping gate
- bundle unrelated model-family changes into the same pass
- expand into a broad feature-engineering refactor

## Coarse layers to test

The design will screen exactly these context layers:

- **`police_zone`** — operational boundary that may align better with enforcement behavior than district
- **`district`** — already known to have broad geographic effects, but lower priority because district-aware routing already failed
- **`res8`** — best surviving coarse geometric layer from the rejected hierarchy

`res7` is intentionally excluded from this pass. The hierarchy experiment already showed `res8` is the better coarse geometric candidate, so testing `res7` again here would add cost without enough upside.

## Feature types

Each coarse layer should only contribute a small, bounded set of context features.

### 1. Recent activity features

Per layer, derive recent counts and rates such as:

- event count in `1h`, `3h`, `24h`, and `7d`
- same-hour historical activity where available
- active-cell or active-subarea counts where relevant

These are the cheapest way to test whether the coarse layer carries useful short-term signal.

### 2. Relative-position features

Per layer, derive features that describe how a candidate `res9` cell sits inside the coarser parent:

- share of parent activity
- rank within parent
- percentile within parent for same-hour or recent activity

These are more likely to help than raw parent identity because they preserve local ordering.

### 3. Categorical identity features

Where practical, keep simple identity columns for:

- `police_zone`
- `district`
- `res8`

These can be used directly by the current tabular model stack, but they should stay secondary to the recent-activity and relative-position features.

## Phase A: analysis pass

Phase A is a **screening step**, not a training step.

For each horizon (`30m`, `1h`, `2h`) and each coarse layer:

1. join the coarse layer onto the current `res9` training frame
2. compute simple signal diagnostics
3. compare those diagnostics against the current baseline intuition

Required diagnostics:

- target-rate lift by coarse bucket
- mutual information or equivalent univariate signal score
- within-`target_time` separability, so the analysis tests ranking usefulness instead of only global correlation

The key question is not whether a layer correlates with the target globally. The key question is whether it helps separate positives from negatives **inside the same target-time scoring window**.

## Phase A gate

Only continue to Phase B if at least one coarse layer shows useful **short-horizon** signal.

That means:

- evidence at `30m` or `1h` matters most
- a `2h`-only improvement is not enough to unlock the ablation phase

This is deliberate. The rejected hierarchy already proved that a `2h` lift alone can coexist with a worse operational outcome overall.

## Phase B: minimal feature ablations

If the gate passes, run the smallest useful ablation set:

1. **`baseline + res8`**
2. **`baseline + police_zone`**
3. **`baseline + district`**
4. **`baseline + all coarse layers`**

Do not add extra combinations unless one of these four wins clearly and a follow-up question remains unanswered.

This keeps the result easy to interpret:

- if `res8` alone wins, the geometry signal is likely the real lever
- if `police_zone` alone wins, operational boundaries matter more than district
- if `all coarse layers` wins, the layers likely contribute complementary signal
- if none win, the coarse context is probably not worth the complexity right now

## Evaluation

### Phase A outputs

Produce compact analysis artifacts per horizon that show:

- layer name
- signal statistic
- within-`target_time` ranking relevance
- short note on whether the layer clears the screen

### Phase B outputs

If unlocked, compare the ablations against the current accepted baseline using the same operational metrics already trusted in this repo:

- dispatch precision@50
- exact spatial precision@50
- group recall / grouped ranking quality where already tracked

## Acceptance rule

This experiment is only worth keeping if it improves the current baseline **without** repeating the short-horizon failure mode.

### Analysis phase acceptance

Phase A passes only if at least one coarse layer shows credible `30m` or `1h` signal.

### Ablation phase acceptance

A feature variant is kept only if it improves the overall `30m` / `1h` / `2h` stack relative to baseline.

If a coarse layer helps only `2h`, record that finding and reject the feature stack for now.

## File boundaries

Likely code areas:

- `ghost_ranking_features.py` for new coarse-layer context derivations
- `analysis/run_two_stage_experiment.py` for bounded analysis/ablation wiring
- `tests/test_ghost_ranking_features.py`
- `tests/test_two_stage_experiment.py`

Likely output artifacts:

- one analysis CSV per horizon
- one coarse-layer ablation comparison CSV if Phase B is unlocked
- regenerated summary artifacts only if an ablation actually runs

## Decision summary

The next step should **not** be another multistep prediction chain.

The next step should be:

- analysis first
- gate on short-horizon signal
- run the smallest useful coarse-feature ablations only if the gate passes

That gives a fast yes/no answer on whether multilevel geography belongs in the model as **context**, not as a hierarchy.
