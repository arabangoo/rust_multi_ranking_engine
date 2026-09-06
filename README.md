# rust_multi_ranking_engine

A Rust library for combining scores from multiple sources and selecting a constrained subset of candidates, with evidence for every selection and rejection.

Use it between retrieval or scoring and the component that consumes the selected items. Candidates can be documents, events, products, or any application type with a stable identifier. The engine supplies fusion, bounded candidate admission, set selection, and result accounting. Your application supplies the data and scorers.

| Property | Current source tree |
| --- | --- |
| Package version | `0.1.0`, alpha |
| Rust | Edition 2021; manifest declares Rust 1.71 for the core |
| Python | Python 3.9 or later; the `python` feature uses PyO3 0.29 and requires Rust 1.83 or later |
| Default dependencies | `thiserror`; no model, inference runtime, network client, or foreign function interface |
| Optional features | `serde`, `parallel`, `python` |
| License | Apache-2.0 |

This guide describes the current checkout. A published package or an older tag may not contain the same result-contract fixes. Minimum Rust versions are declared requirements, not a claim that every optional dependency resolution has been tested on those versions.

## Research foundations

These references explain algorithms and assumptions behind the implementation. They do not certify every combination of callbacks and constraints accepted by the API.

| Reference | Connection to this library |
| --- | --- |
| Nemhauser, Wolsey, and Fisher, [An analysis of approximations for maximizing submodular set functions I](https://link.springer.com/article/10.1007/BF01588971), 1978 | Cardinality-constrained monotone submodular maximization; `GUARANTEE_CARDINALITY = 1 - 1/e` |
| Fisher, Nemhauser, and Wolsey, [An analysis of approximations for maximizing submodular set functions II](https://link.springer.com/chapter/10.1007/BFb0121195), 1978 | Matroid-constrained greedy approximation; `GUARANTEE_MATROID = 1/2` |
| Leskovec et al., [Cost-effective outbreak detection in networks](https://www.cs.cmu.edu/~jure/pubs/detect-kdd07.pdf), 2007 | Cost-aware selection; `GUARANTEE_KNAPSACK_SUBMODULAR = (1 - 1/e) / 2` |
| Cormack, Clarke, and Buettcher, [Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf), 2009 | Reciprocal-rank fusion and conventional `k = 60` |
| Clauset, Shalizi, and Newman, [Power-law distributions in empirical data](https://arxiv.org/abs/0706.1062), 2009 | Motivation for likelihood-based fitting and distribution-distance checks; the engine uses a simpler finite Zipf procedure |

`GUARANTEE_KNAPSACK_MODULAR = 1/2` corresponds to standard modified-greedy comparison with the best singleton. Every factor remains conditional on the assumptions in [Constraints, objectives, and guarantees](#8-constraints-objectives-and-guarantees).

## Contents

1. [Capabilities and scope](#1-capabilities-and-scope)
2. [Quick start](#2-quick-start)
3. [Installation and features](#3-installation-and-features)
4. [Execution model](#4-execution-model)
5. [Core types and callback contracts](#5-core-types-and-callback-contracts)
6. [Rust API reference](#6-rust-api-reference)
7. [Fusion, scales, and missing scores](#7-fusion-scales-and-missing-scores)
8. [Constraints, objectives, and guarantees](#8-constraints-objectives-and-guarantees)
9. [Selection budgets](#9-selection-budgets)
10. [Results and audit evidence](#10-results-and-audit-evidence)
11. [Python binding reference](#11-python-binding-reference)
12. [Pipeline integration](#12-pipeline-integration)
13. [Extension points](#13-extension-points)
14. [Build and test](#14-build-and-test)
15. [Performance, limits, and troubleshooting](#15-performance-limits-and-troubleshooting)
16. [Repository layout](#16-repository-layout)
17. [License](#17-license)

## 1. Capabilities and scope

- Combine heterogeneous scores with Reciprocal Rank Fusion (RRF), or explicitly normalize scores for weighted value fusion.
- Evaluate cheap scorers across the input and expensive scorers only on a bounded pool.
- Apply candidate filters, group caps, total-count caps, and cost constraints.
- Add a coverage objective whose marginal gain depends on the items already selected.
- Choose a fixed count, derive a count from a fitted score distribution, or select within an integer cost budget.
- Require a minimum number of matching candidates and return an error when repair cannot satisfy the requirements.
- Return axis scores, fusion terms, rejection reasons, selection metadata, and execution counters.
- Use Rust traits or the Python dictionary interface without linking a model into the core.

A single score with an unconstrained top-K requirement usually needs only sorting or a heap. This library is useful when several scoring sources, set constraints, or auditable decisions must work together. It does not retrieve candidates, deduplicate identities, train models, tokenize text, or solve arbitrary constrained optimization exactly.

## 2. Quick start

### Rust

After adding the dependency described in [Installation and features](#3-installation-and-features), place this in `src/main.rs`. It selects two documents while allowing at most one per source.

```rust
use rust_multi_ranking_engine::{
    constraint, Budget, Candidate, CandidateId, Engine, Fusion,
    ScoreScale, Scorer, ScorerCost, ScorerId,
};

struct Doc {
    id: u64,
    source: &'static str,
    relevance: f32,
}

impl Candidate for Doc {
    fn id(&self) -> CandidateId { CandidateId::Num(self.id) }
}

struct Relevance;
impl Scorer<Doc> for Relevance {
    fn id(&self) -> ScorerId { ScorerId::new("relevance") }
    fn scale(&self) -> ScoreScale { ScoreScale::Unit }
    fn cost(&self) -> ScorerCost { ScorerCost::Cheap }
    fn score(&self, doc: &Doc) -> Option<f32> { Some(doc.relevance) }
}

fn main() -> rust_multi_ranking_engine::Result<()> {
    let engine = Engine::new()
        .scorer(Relevance)
        .fuse(Fusion::weighted_sum())
        .set_constraint(constraint::max_per_group(
            "one_per_source", 1, |doc: &Doc| doc.source,
        ))
        .budget(Budget::TopK(2));
    engine.validate()?;

    let out = engine.run([
        Doc { id: 1, source: "manual", relevance: 0.9 },
        Doc { id: 2, source: "manual", relevance: 0.8 },
        Doc { id: 3, source: "wiki", relevance: 0.7 },
    ])?;

    let ids: Vec<_> = out.ranked.iter().map(|r| r.candidate.id).collect();
    assert_eq!(ids, vec![1, 3]);
    assert!(out.is_complete());
    assert_eq!(out.rejected_counts.set_constraint, 1);
    for row in &out.ranked {
        assert_eq!(row.fusion.recompute(), row.fused);
        println!("{}", row.to_json(&row.candidate.id()));
    }
    Ok(())
}
```

### Python

Install from source first. Python setters mutate the engine and return `None`; do not chain them.

```python
from rust_multi_ranking_engine import Engine

engine = Engine()
engine.scorer("relevance", scale="unit")
engine.scorer("authority", scale="unit")
engine.fuse("weighted_sum", weights={"relevance": 0.8, "authority": 0.2})
engine.max_per_group("source", 1, id="one_per_source")
engine.budget_top_k(2)

out = engine.run([
    {"id": "a", "scores": {"relevance": 0.9, "authority": 0.8},
     "groups": {"source": "manual"}, "text": "Install the client."},
    {"id": "b", "scores": {"relevance": 0.8, "authority": 0.9},
     "groups": {"source": "manual"}, "text": "Configure the client."},
    {"id": "c", "scores": {"relevance": 0.7, "authority": 0.7},
     "groups": {"source": "wiki"}, "text": "Troubleshooting notes."},
])

assert [row.id for row in out.ranked] == ["a", "c"]
assert out.is_complete()
assert out.rejected_counts["set_constraint"] == 1
print(out.selection)
for row in out.ranked:
    print(row.rank, row.candidate["text"], row.to_json())
```

The result contains the selected set in selection order. A higher fused score can lose to a group cap or to a candidate with greater coverage gain.

## 3. Installation and features

### Rust dependency

For development against this checkout, use a path dependency. This example assumes your application and the engine are sibling directories; adjust the relative path to your layout.

```toml
[dependencies]
rust_multi_ranking_engine = { path = "../rust_multi_ranking_engine" }
```

Enable optional features in the same declaration when needed:

```toml
[dependencies]
rust_multi_ranking_engine = { path = "../rust_multi_ranking_engine", features = ["parallel", "serde"] }
```

| Feature | Effect | Boundary |
| --- | --- | --- |
| Default, no features | Ranking, fusion, constraints, objectives, budgets, and audit JSON | Pure Rust; no model downloads |
| `serde` | Serialization and deserialization for supported value types | Does not make `Engine`, `Outcome<C>`, or every generic result serializable |
| `parallel` | Rayon execution of expensive scoring batches | Callbacks must tolerate concurrent calls and different batch partitions |
| `python` | PyO3 extension using the stable Python application binary interface, `abi3-py39` | Source builds need a compatible Rust toolchain and Python |

Audit JavaScript Object Notation (JSON) output is available without `serde`. The standard Python build enables `python` only. Keep `parallel` out of Python callback builds: worker callbacks must acquire Python's interpreter lock, and that combination is not the supported binding configuration.

### Python source installation

From the repository root, with Rust and a native linker installed:

```shell
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Or on Linux and macOS:

```bash
source .venv/bin/activate
```

Build and install through the backend declared in `pyproject.toml`:

```shell
python -m pip install .
python -c "import rust_multi_ranking_engine as r; print(r.__version__, r.__status__)"
```

For binding development or wheel creation:

```shell
python -m pip install "maturin>=1.5,<2.0"
maturin develop --release
maturin build --release
```

Build dependencies may need network access on first use. The `abi3` setting permits a compatible wheel to serve multiple supported CPython versions on the same platform and architecture; wheels remain platform-specific. Registry installation depends on a compatible release being available there. Source installation is the route for the current behavior documented here.

## 4. Execution model

Let `N` be the input count, `M` the pool capacity, and `K` the requested or derived count.

| Stage | Work | Candidates involved |
| --- | --- | --- |
| Validate | Check scorers, scales, admission, and budget configuration | Before consuming input |
| Filter and admit | Apply unary filters, compute cheap scores, retain the best admission keys in a bounded heap | Input stream |
| Refine | Run expensive scorers, handle missing scores, fuse axes, apply the fused-score threshold | Retained pool |
| Select | Derive the budget when needed and perform constrained greedy selection | Eligible pool |
| Repair | Satisfy lower-bound requirements with feasible replacements | Selected set and eligible alternatives |
| Report | Return selected rows, rejection counts and retained details, and traces | Every input item is accounted for on success |

### Pool sizing and admission

The default pool multiplier is `32`. Capacity uses saturating arithmetic and is at least one.

| Budget | Pool capacity before saturation |
| --- | --- |
| `TopK(k)` | `k * pool_multiplier` |
| `TailMass { fallback_k, .. }` | `max(fallback_k, 1) * pool_multiplier` |
| `Tokens { max }` | `max * pool_multiplier` |

For RRF, admission uses the explicitly named cheap scorer or the first registered cheap scorer. At least one cheap scorer is required because ranks cannot be computed over a stream before the pool exists.

For weighted and maximum fusion, admission uses the explicitly named cheap scorer, or fusion over the cheap axes. With no cheap axes, admission keys are all negative infinity and identifiers decide pool membership. Register a useful cheap scorer even when an expensive scorer supplies the final signal.

Pool pruning precedes expensive scoring and coverage optimization. A discarded candidate cannot later satisfy a requirement or improve the selected set. Increase the multiplier when admission is a weak proxy for the final objective, and measure quality against a larger pool.

### Determinism

Ties use `CandidateId`: numeric identifiers sort before text identifiers, then each variant sorts ascending. Supply unique, stable identifiers and callbacks that return the same values for the same inputs.

Floating-point fusion follows scorer registration order. Keep that order fixed. Parallel equivalence additionally requires scores to be independent of batch size and partitioning. Timing, arrival order of rejection details, and a first-N rejection sample are not deterministic ranking guarantees. Identifier uniqueness is a caller contract; the engine does not deduplicate or reject duplicates.

## 5. Core types and callback contracts

| Type or trait | Purpose |
| --- | --- |
| `Candidate` | Implements `id(&self) -> CandidateId` for an application type |
| `CandidateId` | `Num(u64)` or `Text(Box<str>)`; `CandidateId::text(...)` constructs text IDs |
| `ScorerId`, `ConstraintId` | Named identifiers with `new(...)` and `as_str()` |
| `Scorer<C>` | Declares identity, scale, cost, and candidate or batch scoring |
| `ScoreScale` | `Unit`, `Unbounded`, or `Rank` |
| `ScorerCost` | `Cheap` for stage one; `Expensive` for stage two |
| `ScoreSet` | Registered axis values, including missing values |
| `UnaryConstraint<C>` | Tests one candidate through `allows` |
| `SetConstraint<C>` | Tests addition to a selected set through `admits` |
| `SetObjective<C>` | Supplies an additional set-dependent marginal gain |
| `Requirement<C>` | Minimum matching count enforced after initial selection |
| `Outcome<C>` | Selected rows, rejection information, selection metadata, and trace |

`Engine<C>` requires `C: Candidate + Sync`. Registered scorers, constraints, and objectives are stored through trait objects and must satisfy their `Sync` and registration-time `'static` bounds. `run` borrows the engine and consumes an `IntoIterator<Item = C>`, so an engine can be reused with fresh inputs.

### Scorer contract

- `id`, `scale`, and `cost` must be stable. The engine caches them at registration.
- `score(&C)` returns `Option<f32>`. `None` means unavailable, distinct from a measured zero.
- `score_batch(&[&C])` must return exactly one value per input, in the same order. Its default implementation calls `score` for each candidate.
- Expensive scorers use `score_batch`. Cheap scorers use per-candidate scoring during admission.
- A length mismatch returns `Error::BatchLengthMismatch`. Each parallel chunk is checked independently, so an extra result in one chunk cannot cancel a missing result in another.
- The engine cannot detect reordered batch values. Preserve the positional mapping.
- `Normalized<S>` forwards the underlying batch call and normalizes returned values, preserving `None` entries and allowing malformed lengths to be detected.

The engine treats a returned `NaN` as missing after scoring. It does not generally reject infinities or validate that a declared `Unit` score lies in `[0, 1]`. Normalizers transform values before the engine observes them. Validate raw model outputs in the adapter when strict numerical validity matters.

`ScoreSet::get(&id)` distinguishes an absent axis (`None`) from a registered axis with no value (`Some(None)`). `iter()` returns axes in registration order. Result scores include any explicit normalization; they need not be the original model logits.

## 6. Rust API reference

The application programming interface (API) uses builder methods that consume and return `Self`. `validate` and `run` borrow `&self`.

| Method | Meaning or default |
| --- | --- |
| `Engine::new()` | Empty configuration; register at least one scorer |
| `.scorer(scorer)` | Append a scorer with a unique identifier |
| `.fuse(fusion)` | Default: `Fusion::rrf()` with `k = 60` |
| `.missing(policy)` | Default: `MissingPolicy::Skip` |
| `.unary(constraint)` | Append a filter evaluated before scoring |
| `.set_constraint(constraint)` | Append an upper-bound or other set constraint |
| `.objective(objective)` | Set the additional objective; default: none |
| `.require(requirement)` | Append a lower-bound requirement |
| `.budget(budget)` | Default: `Budget::TopK(10)` |
| `.cost(fn(&C) -> u32)` | Required with `Budget::Tokens`; closure must be `Sync + 'static` |
| `.pool_multiplier(u32)` | Default: `32`; zero is invalid |
| `.admission(scorer_id)` | Use a registered cheap axis for stage-one admission |
| `.threshold(f32)` | Drop candidates below this fused score; default: no threshold |
| `.rejections(policy)` | Default: `Rejections::Keep` |
| `.min_fit(f32)` | Default: `0.9` for tail-mass fitting |
| `.validate()` | Check configuration without consuming candidates |
| `.run(candidates)` | Validate, execute, and return `Result<Outcome<C>>` |

### Errors

Match variants instead of parsing display strings. Current core display messages are in Korean.

| Error | Cause and action |
| --- | --- |
| `NoScorers` | Register at least one scorer |
| `DuplicateScorer` | Give each axis a unique name |
| `IncompatibleScale` | Normalize non-unit axes or use RRF |
| `UnknownWeight` | Correct a weight naming an unregistered axis |
| `UnknownAdmissionScorer` | Register or correct the admission axis |
| `ExpensiveAdmissionScorer` | Choose a cheap admission axis |
| `NoAdmissionScorer` | Add a cheap axis for RRF |
| `InvalidBudget` | Use positive `TopK`, valid tail epsilon, or provide a token cost function |
| `InvalidPoolMultiplier` | Set the multiplier to at least one |
| `BatchLengthMismatch { scorer, expected, got }` | Fix the scorer's positional batch output |
| `InfeasibleRequirement { id, needed, available }` | Inspect pool coverage, conflicts, and repair limits |

Validation checks declared configuration, not arbitrary callback semantics. It does not prove mathematical declarations, validate all floating-point parameters, or catch Rust callback panics. A failed run returns an error rather than a partial `Outcome`; an iterator may already have been consumed when a runtime error occurs.

## 7. Fusion, scales, and missing scores

### Scale compatibility

| Fusion | `Unit` | `Unbounded` | `Rank` |
| --- | --- | --- | --- |
| `Fusion::weighted_sum()` or `Fusion::WeightedSum` | Accepted | Rejected | Rejected |
| `Fusion::Max` | Accepted | Rejected | Rejected |
| `Fusion::rrf()` or `Fusion::Rrf` | Accepted | Accepted | Accepted |

**Every axis is ordered higher-first, including `Rank`.** If a retriever returns rank 1 as best, convert it to a decreasing utility such as `1 / (rank + 1)`. Negate a distance when smaller distances should rank higher. A scale declaration does not reverse ordering.

### Reciprocal Rank Fusion

For each participating axis, the engine ranks surviving candidates by score and adds `1 / (k + rank)`, with one-based rank. `Fusion::rrf()` uses `k = 60`; `Fusion::Rrf { k }` supplies another constant.

RRF uses ordering rather than magnitude. It reranks the retained pool, so upstream rank gaps and ranks of discarded candidates are not preserved. Use a finite nonnegative `k`; arbitrary values are not fully guarded by validation.

### Weighted and maximum fusion

Despite its name, `WeightedSum` computes a **weighted mean** over participating axes:

`fused = sum(weight_i * value_i) / sum(weight_i)`.

An empty weight list assigns each axis weight one. With an explicit nonempty list, omitted axes have weight zero. Under `Skip`, missing axes do not contribute to the denominator. Avoid duplicate weight entries; the current lookup uses the first match. Supply finite nonnegative weights and a positive total for participating axes. These numerical conditions are caller responsibilities.

`Fusion::Max` takes the largest participating value. With no participating values, fusion yields zero. A `Unit` declaration makes axes structurally compatible; it does not establish probability calibration or comparable business meaning.

### Explicit normalization

Import `ScorerExt` and wrap a Rust scorer with `.normalized(normalizer)`.

| Normalizer | Transformation |
| --- | --- |
| `Normalizer::Sigmoid` | `1 / (1 + exp(-x))` |
| `Normalizer::Clamp01` | Clamp to `[0, 1]` |
| `Normalizer::MinMax { min, max }` | Map a fixed interval to `[0, 1]`, then clamp; return zero when `max <= min` |

Min-max bounds are supplied externally. Estimating them from an incoming stream would make earlier and later candidates incomparable. Sigmoid maps a logit into a unit interval but does not itself calibrate a model.

### Missing scores

| Policy | Behavior |
| --- | --- |
| `MissingPolicy::Skip` | Omit the axis; weighted fusion renormalizes remaining weights |
| `MissingPolicy::Impute(value)` | Use a fallback value and record an imputed term |
| `MissingPolicy::Reject` | Reject with `NotScored(scorer_id)` |

A missing cheap score under `Reject` removes a candidate before admission. A missing expensive score removes it during refinement, without refilling the pool from candidates already discarded in stage one.

## 8. Constraints, objectives, and guarantees

### Built-in constraints

| Constructor | Behavior | Mathematical declaration |
| --- | --- | --- |
| `constraint::predicate(id, fn(&C) -> bool)` | Candidate filter | Unary filter |
| `constraint::max_per_group(id, max, key)` | At most `max` candidates per equal key | Partition matroid |
| `constraint::max_total(id, max)` | At most `max` selected candidates | Uniform matroid |
| `constraint::cost_budget(id, limit, fn(&C) -> f64)` | Total floating-point cost at most `limit` | General non-matroid constraint |
| `Requirement::at_least(id, n, predicate)` | At least `n` matching selected candidates | Separate lower-bound repair |

`constraint::group_counts(items, key)` helps inspect a selected set. Upper bounds compose by intersection. A generic `cost_budget` enforces an additional ceiling but does not activate the dedicated knapsack algorithm used by `Budget::Tokens`.

### Coverage and custom objectives

Without a set objective, the objective is the sum of fused scores. With one, each selection step uses:

`gain(S, candidate) = fused(candidate) + objective.marginal_gain(S, candidate)`.

`Coverage::new(|candidate| candidate_topics)` rewards newly covered keys. Duplicate topics within a candidate or across selected candidates count once. `.weighted(|key| weight)` assigns each unique key a value. Use finite nonnegative weights when relying on monotone submodular guarantees. Choose the weight scale deliberately: one coverage unit can dominate a small RRF score.

### Lower-bound requirements

Requirements run after greedy selection. Repair searches for feasible replacements, preserves requirements satisfied earlier, and checks every requirement before returning success. For deterministic predicates and valid constraint implementations, a successful result satisfies all registered lower bounds and the enforced upper bounds.

Repair is a single-replacement heuristic. Requirement order can affect success. `InfeasibleRequirement` means the procedure failed in the eligible pool; a feasible set may still exist and require multiple coordinated swaps. Its `available` field counts candidates matching the predicate before accounting for other constraints. It is not the number of feasible replacements.

If repair changes the selection, the result reports `exact = false` and `guarantee = None`. Requirements already satisfied do not by themselves downgrade the metadata.

### Approximation metadata

A matroid is a family of feasible sets with hereditary and exchange properties. Submodularity means diminishing marginal gains. Custom implementations declare these properties through `is_matroid()` and `is_submodular()`; the engine trusts those declarations.

The following table describes metadata dispatch for the **eligible pool**, subject to mathematical assumptions. Bounds require a normalized, nonnegative, monotone objective, valid constraint declarations, and appropriate nonnegative costs. The engine does not prove these conditions. Submodularity alone does not establish monotonicity, and negative scores or marginal gains can invalidate a bound.

Here `p` counts registered matroid set constraints; the cardinality budget is handled separately.

| Objective | Constraints | `exact` | `guarantee` |
| --- | --- | --- | --- |
| Modular, no additional objective | Cardinality with zero or one matroid | `true` | `None` |
| Modular | Cardinality with `p > 1` matroids | `false` | `1 / p` |
| Monotone submodular | Cardinality only | `false` | `1 - 1/e`, about `0.632` |
| Monotone submodular | Cardinality with one matroid | `false` | `0.5` |
| Monotone submodular | Cardinality with `p > 1` matroids | `false` | `1 / (p + 1)` |
| Modular | `Budget::Tokens`, no additional set constraints | `false` | `0.5` |
| Monotone submodular | `Budget::Tokens`, no additional set constraints | `false` | `(1 - 1/e) / 2`, about `0.316` |
| Any | Generic non-matroid constraint, including `CostBudget` | `false` | `None` |
| Any | Token budget combined with matroid constraints | `false` | `None` |
| Undeclared objective, or selection changed by repair | Any | `false` | `None` |

`None` means no factor is reported for that configuration; it does not measure quality. `exact = true` is conditional metadata, not proof that admission preserved the globally optimal set from the original input.

## 9. Selection budgets

### Fixed count

`Budget::TopK(k)` selects up to `k` candidates; `k` must be positive. Filters, constraints, or a small pool can leave fewer than `k`. Inspect the selected count and rejection evidence.

### Tail-mass count

`Budget::tail_mass(epsilon, fallback_k)` fits a finite Zipf distribution to descending fused scores after refinement and thresholding, then derives a count intended to leave at most `epsilon` of fitted mass outside the prefix.

The validator accepts `0 <= epsilon < 1`. Use positive `fallback_k` for a useful fallback. The implementation fits an exponent in `[0.01, 6]` through bounded numerical search. Fit quality is `1 - D`, where `D` is the Kolmogorov-Smirnov distance; the default acceptance threshold is `0.9`.

This quality score is not a statistical p-value, and the procedure is not the full bootstrapped power-law test described in the cited literature. It operates on the pool, not the entire input. Constraints and coverage can also change the selected items, so epsilon does not guarantee whole-corpus relevance recall or the actual selected set's omitted mass.

| Fallback reason | Trigger |
| --- | --- |
| `TooFewSamples` | Fewer than ten eligible score samples |
| `NotAMassDistribution` | Invalid mass values, including negative, nonfinite, or all-zero scores |
| `PoorFit` | Fit quality misses the configured threshold |

`BudgetTrace` records `s`, `fit_quality`, `fallback`, `reason`, and `derived_k`. Inspect it when adaptive selection is enabled. `tail_mass(s, k, v)` separately computes finite Zipf mass beyond the first `k` ranks among `v` ranks.

### Integer cost budget

`Budget::Tokens { max }` requires `.cost(|candidate| candidate_cost)` in Rust. Costs are `u32` and accumulated with a wider integer. Tokens are a common use case, but any consistently measured nonnegative integer cost can be used.

The selector compares gain-per-cost greedy selection, raw-gain greedy selection, and the best feasible singleton, then keeps the highest objective value. This is the dedicated path associated with the knapsack factors above.

Zero-cost candidates are allowed, so a token budget does not necessarily bound item count. It also sizes the pool from the numeric budget: a large token limit can create a large pool. A generic floating-point `CostBudget` remains an additional feasibility check with no claimed knapsack factor.

## 10. Results and audit evidence

### Outcome contract

| Field or method | Meaning |
| --- | --- |
| `ranked: Vec<Ranked<C>>` | Selected candidates in one-based selection order |
| `rejected: Vec<Rejected<C>>` | Rejection details retained by policy |
| `rejected_counts: RejectCounts` | Exact counts for all rejection categories |
| `selection: Selection` | Conditional optimality metadata and diagnostics |
| `trace: RunTrace` | Input count, pool capacity, admission axis, scoring counters, and optional budget fit |
| `is_complete()` | Checks `ranked.len() + rejected_counts.total() == trace.input_count` |

`is_complete()` verifies accounting. It does not certify that K was filled, IDs were unique, callbacks were correct, or selection was globally optimal.

`Ranked<C>` contains `candidate`, `rank`, `fused`, `scores`, `fusion`, and passed set-constraint identifiers. Rank is selection order, including positions affected by repair, and need not follow descending fused score. `fused` excludes additional set-objective gain.

`FusionTrace` contains the method, missing policy, optional RRF constant, and per-axis terms. Each term records the scorer, input kind (`Value`, `Rank`, `Imputed`, or `Skipped`), effective weight, and contribution. `recompute()` reconstructs the fused score from contributions; `used_axes()` counts participating terms. This reconstructs fusion, not the history of objective gains and replacement decisions.

### Rejections and retention

| Reason | Meaning |
| --- | --- |
| `NotScored(scorer)` | A score was missing under `Reject` |
| `UnaryConstraint(id)` | A candidate filter failed |
| `SetConstraint(id)` | A registered set constraint blocks addition to the final selected set |
| `BelowThreshold` | Fused score was below the threshold |
| `Outranked` | Unselected without an identified blocking generic set constraint |
| `OutOfPool` | Admission discarded the candidate, including heap eviction |

A rejection row contains the candidate, reason, and optional fused score. Stage-one rejections have no fused score. If multiple generic constraints block a remaining candidate, the first matching constraint supplies the reason. The dedicated token budget has no separate variant, so budget-limited candidates can appear as `Outranked`.

| Retention policy | Stored details | Counts |
| --- | --- | --- |
| `Rejections::Keep`, default | All rejected candidates | All candidates |
| `Rejections::Count` | None | All candidates |
| `Rejections::Sample(n)` | First `n` encountered rejections, not a random sample | All candidates |

For large streams, use `Count` or a bounded `Sample`. Default `Keep` retains rejected payloads and can use memory proportional to the entire input.

### Diagnostics

| Field | Interpretation |
| --- | --- |
| `selection.exact`, `selection.guarantee` | Conditional pool-level metadata described above |
| `selection.pool_size` | Stage-one pool size, before expensive-score rejection and thresholding |
| `selection.pool_exhausted` | For cardinality selection, the requested count could not be filled; inspect filters and constraints before assuming a larger pool will help |
| `selection.cut_margin` | Local difference at the final boundary, using marginal gain for cardinality and gain per cost for tokens |
| `trace.input_count`, `trace.pool_capacity` | Consumed input size and admission capacity |
| `trace.admission_scorer` | Named admission axis, or `None` for cheap value fusion |
| `trace.scorers` | Registration-ordered records: `scorer`, `calls`, `missing`, `elapsed_nanos` |
| `trace.budget` | Tail-mass fit information when that budget is used |

`ScorerTrace.calls` counts candidate evaluations, not batch callback invocations. Timing varies between runs.

Treat `cut_margin` as a local diagnostic. It can be absent when no finite comparison exists or the singleton branch wins. For tokens, it remains based on gain per cost even when the raw-gain branch wins. After repair it is not an optimality certificate. The token path's `pool_exhausted` follows internal greedy bookkeeping; do not use it as proof of feasibility, recall, or remaining budget.

### JSON output

Rust selected and rejected rows expose `to_json(&candidate_id)`. Python selected rows expose `to_json()`; Python rejected rows expose structured attributes instead. This selected-row example combines equally weighted values `1.0` and `0.5` into `0.75`:

```json
{
  "candidate": "doc-a",
  "rank": 1,
  "fused": 0.75,
  "scores": {"relevance": 1.0, "authority": 0.5},
  "missing_policy": "skip",
  "fusion": {
    "method": "weighted_sum",
    "terms": [
      {"scorer": "relevance", "input": {"kind": "value", "value": 1.0}, "weight": 0.5, "contribution": 0.5},
      {"scorer": "authority", "input": {"kind": "value", "value": 0.5}, "weight": 0.5, "contribution": 0.25}
    ]
  },
  "constraints": {}
}
```

Nonfinite numbers become JSON `null`. IDs render as strings, so numeric `7` and text `"7"` can produce the same audit ID. Use one identifier convention when exporting. Rows do not serialize the candidate payload or provide a complete run-level schema. Store configuration, row evidence, and run metadata in an application-owned envelope when durable replay is required.

## 11. Python binding reference

The module exports `Engine`, result classes, `EngineError`, `tail_mass`, `__version__`, and `__status__`. Setters mutate the engine. Validation occurs in `run`; the binding does not expose Rust's `validate()`.

### Candidate dictionaries

| Key | Accepted data | Default or behavior |
| --- | --- | --- |
| `id` | Prefer a nonnegative integer fitting `u64`, or a string | Required; other values fall back to string conversion |
| `scores` | Dictionary keyed by scorer name, or sequence in registration order | Omitted scores are missing; extra sequence entries are ignored |
| `groups` | Dictionary of group keys and values | Values convert to strings; absent or `None` values are missing |
| `cost` | Nonnegative integer fitting `u32` | Used for tokens; missing or `None` becomes zero |
| `cover` | Sequence of topic keys | Keys convert to strings; absent or `None` means no coverage |
| Other keys | Application payload | Preserved through the original candidate object |

Use `None` for a missing score entry, not `scores=None`. For a positive group cap, candidates with no value for that group are exempt; a cap of zero rejects all candidates through that constraint. Validate group fields upstream if missing groups must not bypass a limit.

### Configuration methods

| Method | Accepted settings |
| --- | --- |
| `scorer(name, *, scale="unit", cost="cheap", normalize=None, normalize_range=None, fn=None)` | Scales: `unit`, `unbounded`, `rank`; costs: `cheap`, `expensive` |
| `fuse(method, *, k=None, weights=None)` | `rrf`, `weighted_sum`, `max`; weights as dictionary or sequence of name/weight pairs |
| `missing(policy, *, value=None)` | `skip`, `impute`, `reject`; supply `value` for imputation |
| `max_per_group(key, max, *, id=None)` | Cap each nonmissing group value |
| `max_total(max, *, id=None)` | Configure a total-count cap |
| `require_at_least(key, value, n, *, id=None)` | Require `n` matches of the group's string value |
| `unary_min(axis, min, *, id=None)` | Filter by precomputed raw score before normalization |
| `coverage_objective()` | Enable coverage from candidate `cover` sequences |
| `budget_top_k(k)` | Fixed positive count |
| `budget_tail_mass(epsilon, fallback_k)` | Adaptive count with fallback |
| `budget_tokens(max)` | Integer budget using candidate `cost` |
| `admission(axis)` | Choose a registered cheap axis |
| `pool_multiplier(multiplier)` | Positive multiplier; default `32` |
| `threshold(value)` | Minimum fused score after refinement |
| `min_fit(value)` | Tail-mass acceptance threshold; default `0.9` |
| `rejections(policy, *, n=None)` | `keep`, `count`, `sample`; supply `n` for sampling |
| `run(candidates)` | Consume an iterable of dictionaries and return an `Outcome` |

Normalizer names are `sigmoid`, `clamp01`, and `minmax`; the last requires `normalize_range=(min, max)`. An expensive scorer without `fn` still reads precomputed scores. Declaring cost does not invoke a model.

A callback supplied through `fn` must belong to an expensive scorer. It receives a list of original candidate objects and returns an equal-length sequence of numbers or `None` in the same order. In the standard serial Python build, each expensive scorer receives the pool as a batch. Support empty batches too. Input conversion still processes every candidate; batching does not eliminate all interpreter work.

Configuration errors generally surface as `ValueError`; execution failures such as malformed batch length or failed repair use `EngineError`. Callback exceptions can propagate from the adapter. Match exception types rather than parsing localized strings.

### Result objects

| Object | Attributes and methods |
| --- | --- |
| `Outcome` | `ranked`, `rejected`, `rejected_counts`, `selection`, `trace`, `is_complete()` |
| `Ranked` | String `id`, `rank`, `fused`, score dictionary, fusion dictionary, constraint names, original `candidate`, `to_json()` |
| `Rejected` | String `id`, reason name, optional scorer or constraint `detail`, optional `fused`, original `candidate` |

`selection`, `trace`, and `rejected_counts` are dictionaries. Candidate objects remain references to originals, while selected-row JSON is a snapshot from result construction. Avoid mutating returned dictionaries or candidates used as audit evidence. Scores use Rust `f32`; compare with a tolerance such as `math.isclose` rather than exact decimal equality.

## 12. Pipeline integration

### Retrieval-augmented generation

In retrieval-augmented generation (RAG), retrieve broadly, attach cheap retrieval scores, rerank the admitted pool, select under source and context limits, and pass `row.candidate` to the context builder. Token counts must come from the tokenizer used by the consuming model.

This runnable example demonstrates the callback boundary with fixed illustrative logits. It performs no model inference.

```python
from rust_multi_ranking_engine import Engine

seen_batches = []

def rerank_batch(candidates):
    seen_batches.append(len(candidates))
    # Replace this lookup with a model's batch inference adapter.
    return [candidate["rerank_logit"] for candidate in candidates]

engine = Engine()
engine.scorer("retrieval", scale="unit")
engine.scorer("rerank", scale="unbounded", cost="expensive",
              normalize="sigmoid", fn=rerank_batch)
engine.fuse("weighted_sum", weights={"retrieval": 0.3, "rerank": 0.7})
engine.admission("retrieval")
engine.max_per_group("source", 1)
engine.coverage_objective()
engine.budget_tokens(300)
engine.rejections("sample", n=20)

out = engine.run([
    {"id": "setup", "scores": {"retrieval": 0.9}, "rerank_logit": 2.0,
     "groups": {"source": "manual"}, "cover": ["setup"], "cost": 120,
     "text": "Installation procedure."},
    {"id": "config", "scores": {"retrieval": 0.8}, "rerank_logit": 1.0,
     "groups": {"source": "manual"}, "cover": ["setup"], "cost": 100,
     "text": "Configuration procedure."},
    {"id": "repair", "scores": {"retrieval": 0.7}, "rerank_logit": 1.5,
     "groups": {"source": "wiki"}, "cover": ["repair"], "cost": 140,
     "text": "Recovery procedure."},
])

assert seen_batches == [3]
assert out.is_complete()
assert sum(row.candidate["cost"] for row in out.ranked) <= 300
assert {row.id for row in out.ranked} == {"setup", "repair"}
# Tokens combined with a group cap have no reported approximation factor.
assert out.selection["guarantee"] is None
context = "\n\n".join(row.candidate["text"] for row in out.ranked)
print(context)
```

The Rust counterpart in [examples/rag_selection.rs](examples/rag_selection.rs) combines normalized logits, source caps, coverage, and a token budget. Its logits are also supplied data, not a bundled cross-encoder.

### Multiple retrieval sources

Merge retrieval outputs by an application-owned canonical ID before running the engine. Preserve one score axis per source, mark absent scores as missing, and use RRF when magnitudes are incomparable. Convert lower-is-better ranks or distances first. Set admission deliberately: the first cheap axis otherwise controls the RRF pool.

The engine does not merge duplicate IDs. If admission depends on several retrievers, construct a suitable cheap admission score upstream and register it explicitly.

### Cached or unavailable model scores

A scorer adapter can read cached values and return `None` when unavailable. Choose `Skip`, `Impute`, or `Reject` according to application semantics. Include content version, query, model version, and preprocessing settings in cache keys; the core has no knowledge of them.

### Audit storage

For full per-candidate records, use `Keep` and export after the run. For large inputs, use `Count` or `Sample` and store aggregate counts with selected evidence. The core has no streaming audit sink callback. If full audit retention and bounded process memory are both required, design external storage or input partitioning with its effect on global selection made explicit.

## 13. Extension points

Implement existing traits in an adapter crate or application module. The core need not depend on your model runtime, service client, or domain types.

| Extension | Methods | Contract |
| --- | --- | --- |
| Candidate | `Candidate::id` | Stable, unique identity |
| Scorer | `id`, `scale`, `cost`, `score`; optionally `score_batch` | Stable metadata; deterministic positional scores |
| Candidate filter | `UnaryConstraint::id`, `allows` | Pure predicate on one candidate |
| Set constraint | `SetConstraint::id`, `admits`, `is_matroid` | Pure feasibility test; claim a matroid only when justified |
| Set objective | `SetObjective::marginal_gain`, `is_submodular` | Consistent increments, independent of selected-slice ordering |

This complete Rust example wraps a batch scorer without losing its batch implementation, then adds a lower-bound requirement:

```rust
use rust_multi_ranking_engine::{
    Budget, Candidate, CandidateId, Engine, Fusion, Normalizer, Requirement,
    ScoreScale, Scorer, ScorerCost, ScorerExt, ScorerId,
};

struct Item { id: u64, cheap: f32, logit: f32, required: bool }
impl Candidate for Item {
    fn id(&self) -> CandidateId { CandidateId::Num(self.id) }
}

struct Cheap;
impl Scorer<Item> for Cheap {
    fn id(&self) -> ScorerId { "cheap".into() }
    fn scale(&self) -> ScoreScale { ScoreScale::Unit }
    fn cost(&self) -> ScorerCost { ScorerCost::Cheap }
    fn score(&self, c: &Item) -> Option<f32> { Some(c.cheap) }
}

struct BatchLogits;
impl Scorer<Item> for BatchLogits {
    fn id(&self) -> ScorerId { "logit".into() }
    fn scale(&self) -> ScoreScale { ScoreScale::Unbounded }
    fn cost(&self) -> ScorerCost { ScorerCost::Expensive }
    fn score(&self, c: &Item) -> Option<f32> { Some(c.logit) }
    fn score_batch(&self, candidates: &[&Item]) -> Vec<Option<f32>> {
        // Replace with batch inference, preserving length and order.
        candidates.iter().map(|c| Some(c.logit)).collect()
    }
}

fn main() -> rust_multi_ranking_engine::Result<()> {
    let out = Engine::new()
        .scorer(Cheap)
        .scorer(BatchLogits.normalized(Normalizer::Sigmoid))
        .fuse(Fusion::weighted_sum())
        .require(Requirement::at_least("required_kind", 1, |c: &Item| c.required))
        .budget(Budget::TopK(1))
        .run([
            Item { id: 1, cheap: 0.9, logit: 3.0, required: false },
            Item { id: 2, cheap: 0.5, logit: 0.0, required: true },
        ])?;
    assert_eq!(out.ranked[0].candidate.id, 2);
    assert!(!out.selection.exact);
    assert_eq!(out.selection.guarantee, None);
    assert!(out.is_complete());
    Ok(())
}
```

Return `false` for a mathematical declaration unless the property is established. The engine can still run and will withhold the corresponding claim. `admits` must describe feasibility of adding the candidate to the current set; hidden mutable state can invalidate selection and repair checks.

## 14. Build and test

Run from the repository root:

```shell
cargo fmt --all -- --check
cargo test
cargo test --features parallel,serde
cargo doc --no-deps
cargo run --example rag_selection
cargo run --release --example throughput
```

Rust tests use no separate testing dependency. They include deterministic generated cases, exhaustive comparisons on small instances, configuration checks, audit invariants, and result-contract regressions.

| Test file | Coverage |
| --- | --- |
| [tests/configuration.rs](tests/configuration.rs) | Invalid configuration, admission, and expensive batching |
| [tests/invariants.rs](tests/invariants.rs) | Determinism, accounting, constraints, and fusion evidence |
| [tests/optimality.rs](tests/optimality.rs) | Small-instance comparisons and selection guarantees |
| [tests/budget_policy.rs](tests/budget_policy.rs) | Adaptive fitting and fallback behavior |
| [tests/scholar_fusion.rs](tests/scholar_fusion.rs) | Multiple-source fusion |
| [tests/result_contracts.rs](tests/result_contracts.rs) | Requirements, generic-cost metadata, normalized batches, and per-chunk length checks |
| [tests/test_python_binding.py](tests/test_python_binding.py) | Python inputs, callbacks, errors, results, and binding behavior |

For Python, build the supported configuration in an activated virtual environment, then run the binding suite:

```shell
python -m pip install pytest "maturin>=1.5,<2.0"
maturin develop --release
python -m pytest tests/test_python_binding.py
```

Test Rust parallel behavior separately from the standard Python build. `--all-features` combines Python and Rayon and is not a substitute for those configurations. Unit tests do not establish model accuracy, production throughput, or minimum-toolchain compatibility.

## 15. Performance, limits, and troubleshooting

### Cost model

Stage-one heap admission takes approximately `O(N log M)` comparisons, plus unary predicates and cheap scoring. This is not the complexity of the entire run.

Expensive scoring processes at most the pool per axis. Cardinality greedy selection scans the pool repeatedly, roughly `O(KM)` before callback costs. Token selection can perform roughly `O(M²)` scanning. Set constraints may scan the selected set, and built-in coverage rebuilds covered-key sets during gain evaluation. Requirement repair adds its own search cost.

Pool score storage grows with `M` times the number of axes, plus payloads, selection state, and rejection details. `Keep` can add `O(N)` retained payloads. `Count` and bounded `Sample` bound rejection retention, but cannot remove memory already allocated by a caller passing a materialized list. Python conversion adds overhead too.

[examples/throughput.rs](examples/throughput.rs) is a reproducible synthetic workload. Its expensive scorer simulates work; it does not measure neural inference. Record the command, processor, compiler, profile, features, input size, pool capacity, scorer workload, and retention policy when reporting timings. Historical figures without those conditions are not portable performance promises.

### Known limits

- Admission can discard candidates needed for the best constrained set; there is no automatic full-input retry.
- Lower-bound repair can fail even when a feasible set exists.
- Numerical inputs and mathematical declarations are not fully validated.
- Duplicate IDs are not detected; audit strings can collapse numeric and text IDs.
- Fusion evidence does not record the full objective or repair trajectory.
- Token diagnostics have the limitations described in [Results and audit evidence](#10-results-and-audit-evidence).
- Python exposes built-in constraints and coverage, not arbitrary Python implementations of Rust constraint and objective traits.
- The library is alpha. Durable output schemas and cross-version compatibility remain application integration responsibilities.

### Troubleshooting

| Symptom | Check |
| --- | --- |
| `IncompatibleScale` | Use RRF or normalize each non-unit scorer |
| Rank 1 appears worse than rank 10 | Convert lower-is-better upstream values before scoring |
| Fewer than K results | Inspect filters, missing scores, threshold and set constraints, then pool capacity |
| A requirement fails despite matching input items | Inspect eligible-pool membership, conflicts, upper bounds, and multi-swap needs |
| Memory grows with input size | Change rejection retention and avoid materializing input upstream |
| Expensive scores attach to the wrong items | Verify batch order as well as length |
| `exact` is true but the full-input optimum differs | Metadata applies only to the eligible pool |
| No guarantee after adding a cost constraint | Generic `CostBudget` and token-plus-matroid combinations have no reported factor |
| Results change with parallel scoring | Check callback state, batch-dependent normalization, unique IDs, and scorer order |
| Python source installation fails | Check Rust version, native linker, Python environment, and dependency access |

## 16. Repository layout

| Path | Responsibility |
| --- | --- |
| [Cargo.toml](Cargo.toml) | Package, dependencies, and features |
| [pyproject.toml](pyproject.toml) | Python metadata and Maturin build configuration |
| [src/lib.rs](src/lib.rs) | Public modules, exports, and crate examples |
| [src/candidate.rs](src/candidate.rs) | Candidate trait and identifiers |
| [src/score.rs](src/score.rs) | Scorers, normalization, and score sets |
| [src/fuse.rs](src/fuse.rs) | Fusion policies and evidence types |
| [src/constraint.rs](src/constraint.rs) | Constraints and requirements |
| [src/objective.rs](src/objective.rs) | Objectives and coverage |
| [src/budget.rs](src/budget.rs) | Budgets, Zipf fitting, and fallback traces |
| [src/select.rs](src/select.rs) | Internal selection, repair, and guarantee dispatch |
| [src/engine.rs](src/engine.rs) | Validation and execution pipeline |
| [src/evidence.rs](src/evidence.rs) | Outcomes, accounting, diagnostics, and JSON |
| [src/error.rs](src/error.rs) | Typed errors |
| [src/python.rs](src/python.rs) | Python adapters and results |
| [examples/rag_selection.rs](examples/rag_selection.rs) | Constrained document selection |
| [examples/throughput.rs](examples/throughput.rs) | Synthetic throughput workload |
| [tests](tests) | Integration and Python tests |

## 17. License

Licensed under [Apache-2.0](LICENSE). See [Cargo.toml](Cargo.toml) for package metadata and the [repository](https://github.com/arabangoo/rust_multi_ranking_engine) for source.
