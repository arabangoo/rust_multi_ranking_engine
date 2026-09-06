//! 빌더와 오케스트레이션.
//!
//! 사용자가 만나는 표면은 여기 하나다. 채점기와 제약을 등록하고 융합·예산을 고른 뒤
//! [`Engine::run`] 을 부르면 두 단계가 순서대로 돈다.

use std::time::Instant;

use crate::budget::{derive_budget, Budget, BudgetTrace, DEFAULT_MIN_FIT};
use crate::candidate::{Candidate, CandidateId};
use crate::constraint::{ConstraintId, Requirement, SetConstraint, UnaryConstraint};
use crate::error::{Error, Result};
use crate::evidence::{
    Outcome, Ranked, RejectCounts, RejectReason, Rejected, Rejections, RunTrace, ScorerTrace,
    Selection,
};
use crate::fuse::{Fusion, FusionInput, FusionTerm, FusionTrace, MissingPolicy};
use crate::objective::SetObjective;
use crate::score::{ScoreScale, ScoreSet, Scorer, ScorerCost, ScorerId};
use crate::select::{guarantee_of, Admission, BoundedHeap, PoolEntry, Selector, Structure};

/// 1단계 풀 배수의 기본값. `M = K * 32`.
pub const DEFAULT_POOL_MULTIPLIER: u32 = 32;

/// 후보별 비용 함수. [`Budget::Tokens`] 예산이 쓴다.
type CostFn<C> = Box<dyn Fn(&C) -> u32 + Sync>;

struct ScorerMeta {
    id: ScorerId,
    scale: ScoreScale,
    cost: ScorerCost,
}

/// 1단계를 통과해 힙에 앉은 후보.
struct Staged<C> {
    candidate: C,
    id: CandidateId,
    scores: Vec<Option<f32>>,
}

/// 1단계 절단 기준.
enum AdmissionKey {
    /// 지정되거나 자동으로 고른 채점기 하나의 값.
    Scorer(usize),
    /// 싼 축들만으로 계산한 융합 값.
    StreamingFusion,
}

/// 다축 점수 융합과 제약 아래 상위 K 선택 엔진.
///
/// ```
/// use rust_multi_ranking_engine::{
///     Budget, Candidate, CandidateId, Engine, Fusion, ScoreScale, Scorer, ScorerCost, ScorerId,
/// };
///
/// struct Doc { id: u64, hits: f32 }
/// impl Candidate for Doc {
///     fn id(&self) -> CandidateId { CandidateId::num(self.id) }
/// }
///
/// struct Relevance;
/// impl Scorer<Doc> for Relevance {
///     fn id(&self) -> ScorerId { ScorerId::new("relevance") }
///     fn scale(&self) -> ScoreScale { ScoreScale::Unit }
///     fn cost(&self) -> ScorerCost { ScorerCost::Cheap }
///     fn score(&self, d: &Doc) -> Option<f32> { Some(d.hits) }
/// }
///
/// let out = Engine::new()
///     .scorer(Relevance)
///     .fuse(Fusion::weighted_sum())
///     .budget(Budget::TopK(2))
///     .run(vec![
///         Doc { id: 1, hits: 0.2 },
///         Doc { id: 2, hits: 0.9 },
///         Doc { id: 3, hits: 0.5 },
///     ])
///     .unwrap();
///
/// assert_eq!(out.ranked.len(), 2);
/// assert_eq!(out.ranked[0].candidate.id, 2);
/// assert!(out.is_complete());
/// ```
pub struct Engine<C> {
    scorers: Vec<Box<dyn Scorer<C>>>,
    meta: Vec<ScorerMeta>,
    unary: Vec<Box<dyn UnaryConstraint<C>>>,
    set_constraints: Vec<Box<dyn SetConstraint<C>>>,
    objective: Option<Box<dyn SetObjective<C>>>,
    requirements: Vec<Requirement<C>>,
    cost: Option<CostFn<C>>,
    fusion: Fusion,
    missing: MissingPolicy,
    budget: Budget,
    pool_multiplier: u32,
    admission: Option<ScorerId>,
    rejections: Rejections,
    threshold: Option<f32>,
    min_fit: f32,
}

impl<C> Default for Engine<C> {
    fn default() -> Self {
        Engine {
            scorers: Vec::new(),
            meta: Vec::new(),
            unary: Vec::new(),
            set_constraints: Vec::new(),
            objective: None,
            requirements: Vec::new(),
            cost: None,
            fusion: Fusion::default(),
            missing: MissingPolicy::default(),
            budget: Budget::default(),
            pool_multiplier: DEFAULT_POOL_MULTIPLIER,
            admission: None,
            rejections: Rejections::default(),
            threshold: None,
            min_fit: DEFAULT_MIN_FIT,
        }
    }
}

impl<C: Candidate + Sync> Engine<C> {
    /// 빈 엔진을 만든다.
    pub fn new() -> Self {
        Engine::default()
    }

    /// 채점기를 등록한다. 등록 순서가 융합 기록과 감사 출력의 축 순서다.
    pub fn scorer<S: Scorer<C> + 'static>(mut self, scorer: S) -> Self {
        self.meta.push(ScorerMeta {
            id: scorer.id(),
            scale: scorer.scale(),
            cost: scorer.cost(),
        });
        self.scorers.push(Box::new(scorer));
        self
    }

    /// 융합 방식을 고른다. 기본값은 관례 상수의 순위 융합이다.
    pub fn fuse(mut self, fusion: Fusion) -> Self {
        self.fusion = fusion;
        self
    }

    /// 값이 없는 축을 어떻게 다룰지 고른다. 기본값은 건너뛰기다.
    pub fn missing(mut self, policy: MissingPolicy) -> Self {
        self.missing = policy;
        self
    }

    /// 단항 제약을 건다. 1단계에서 채점보다 먼저 돈다.
    pub fn unary<U: UnaryConstraint<C> + 'static>(mut self, constraint: U) -> Self {
        self.unary.push(Box::new(constraint));
        self
    }

    /// 집합 제약을 건다.
    pub fn set_constraint<S: SetConstraint<C> + 'static>(mut self, constraint: S) -> Self {
        self.set_constraints.push(Box::new(constraint));
        self
    }

    /// 집합 목적함수를 건다. 총 이득은 융합 점수에 이 함수의 한계 이득을 더한 값이다.
    pub fn objective<O: SetObjective<C> + 'static>(mut self, objective: O) -> Self {
        self.objective = Some(Box::new(objective));
        self
    }

    /// 최종 집합이 만족해야 할 하한 조건을 건다.
    pub fn require(mut self, requirement: Requirement<C>) -> Self {
        self.requirements.push(requirement);
        self
    }

    /// 몇 개를 고를지 정한다.
    pub fn budget(mut self, budget: Budget) -> Self {
        self.budget = budget;
        self
    }

    /// 후보별 비용. [`Budget::Tokens`] 를 쓸 때 필요하다.
    pub fn cost<F: Fn(&C) -> u32 + Sync + 'static>(mut self, cost: F) -> Self {
        self.cost = Some(Box::new(cost));
        self
    }

    /// 1단계 풀 배수. `M = K * multiplier` 다. 기본값은 32.
    pub fn pool_multiplier(mut self, multiplier: u32) -> Self {
        self.pool_multiplier = multiplier;
        self
    }

    /// 1단계 절단에 쓸 승인 채점기를 지정한다.
    ///
    /// 순위 융합은 순위를 입력으로 쓰고 순위는 풀이 있어야 나오므로, 1단계가 상위 M 개를
    /// 무엇으로 자를지 따로 정해야 한다. 지정하지 않으면 등록된 첫 싼 채점기가 쓰인다.
    ///
    /// 값 기반 융합(가중합·최댓값)에서는 지정하지 않으면 싼 축들만으로 계산한 융합 값이
    /// 절단 기준이 된다.
    pub fn admission(mut self, scorer: impl Into<ScorerId>) -> Self {
        self.admission = Some(scorer.into());
        self
    }

    /// 융합 점수의 하한. 이 아래는 [`RejectReason::BelowThreshold`] 로 떨어진다.
    pub fn threshold(mut self, value: f32) -> Self {
        self.threshold = Some(value);
        self
    }

    /// 탈락 후보를 얼마나 보관할지. 개수는 정책과 무관하게 언제나 정확하다.
    pub fn rejections(mut self, policy: Rejections) -> Self {
        self.rejections = policy;
        self
    }

    /// 꼬리 질량 예산의 적합도 문턱. 기본값은 [`DEFAULT_MIN_FIT`].
    pub fn min_fit(mut self, value: f32) -> Self {
        self.min_fit = value;
        self
    }

    /// 설정이 성립하는지 검사한다. [`run`](Self::run) 이 후보를 한 건도 읽기 전에 이것을 먼저 부른다.
    pub fn validate(&self) -> Result<()> {
        if self.scorers.is_empty() {
            return Err(Error::NoScorers);
        }
        for (i, m) in self.meta.iter().enumerate() {
            if self.meta[..i].iter().any(|other| other.id == m.id) {
                return Err(Error::DuplicateScorer(m.id.clone()));
            }
            if !self.fusion.accepts(m.scale) {
                return Err(Error::IncompatibleScale {
                    scorer: m.id.clone(),
                    scale: m.scale,
                    fusion: self.fusion.name(),
                });
            }
        }

        if let Fusion::WeightedSum { weights } = &self.fusion {
            for (id, _) in weights {
                if !self.meta.iter().any(|m| &m.id == id) {
                    return Err(Error::UnknownWeight(id.clone()));
                }
            }
        }

        if let Some(id) = &self.admission {
            match self.meta.iter().position(|m| &m.id == id) {
                None => return Err(Error::UnknownAdmissionScorer(id.clone())),
                Some(i) if self.meta[i].cost == ScorerCost::Expensive => {
                    return Err(Error::ExpensiveAdmissionScorer(id.clone()))
                }
                Some(_) => {}
            }
        } else if matches!(self.fusion, Fusion::Rrf { .. })
            && !self.meta.iter().any(|m| m.cost == ScorerCost::Cheap)
        {
            return Err(Error::NoAdmissionScorer);
        }

        if self.pool_multiplier == 0 {
            return Err(Error::InvalidPoolMultiplier);
        }

        match self.budget {
            Budget::TopK(0) => return Err(Error::InvalidBudget("TopK 는 1 이상이어야 한다")),
            Budget::TailMass { epsilon, .. } if !(0.0..1.0).contains(&epsilon) => {
                return Err(Error::InvalidBudget("epsilon 은 0 이상 1 미만이어야 한다"))
            }
            Budget::Tokens { .. } if self.cost.is_none() => {
                return Err(Error::InvalidBudget(
                    "Tokens 예산은 Engine::cost 로 후보별 비용을 함께 받아야 한다",
                ))
            }
            _ => {}
        }

        Ok(())
    }

    /// 후보를 훑어 고른다.
    pub fn run(&self, candidates: impl IntoIterator<Item = C>) -> Result<Outcome<C>> {
        self.validate()?;

        let admission_key = self.resolve_admission();
        let admission_scorer = match admission_key {
            AdmissionKey::Scorer(i) => Some(self.meta[i].id.clone()),
            AdmissionKey::StreamingFusion => None,
        };

        let mut stats: Vec<ScorerTrace> = self
            .meta
            .iter()
            .map(|m| ScorerTrace {
                scorer: m.id.clone(),
                calls: 0,
                missing: 0,
                elapsed_nanos: 0,
            })
            .collect();

        let mut counts = RejectCounts::default();
        let mut rejected: Vec<Rejected<C>> = Vec::new();
        let mut input_count: u64 = 0;

        // ── 1단계: 스트리밍 ────────────────────────────────────────
        let capacity = self.pool_capacity();
        let mut heap = BoundedHeap::new(capacity);
        let mut slab: Vec<Option<Staged<C>>> = Vec::new();
        let mut free: Vec<usize> = Vec::new();

        'next: for candidate in candidates {
            input_count += 1;

            for c in &self.unary {
                if !c.allows(&candidate) {
                    let reason = RejectReason::UnaryConstraint(c.id());
                    self.record(&mut rejected, &mut counts, candidate, reason, None);
                    continue 'next;
                }
            }

            let mut scores: Vec<Option<f32>> = vec![None; self.scorers.len()];
            for (i, m) in self.meta.iter().enumerate() {
                if m.cost != ScorerCost::Cheap {
                    continue;
                }
                let start = Instant::now();
                let v = sanitize(self.scorers[i].score(&candidate));
                stats[i].elapsed_nanos += start.elapsed().as_nanos();
                stats[i].calls += 1;
                if v.is_none() {
                    stats[i].missing += 1;
                    if self.missing == MissingPolicy::Reject {
                        let reason = RejectReason::NotScored(m.id.clone());
                        self.record(&mut rejected, &mut counts, candidate, reason, None);
                        continue 'next;
                    }
                }
                scores[i] = v;
            }

            let key = self.admission_value(&admission_key, &scores);
            let id = candidate.id();
            let staged = Staged {
                candidate,
                id: id.clone(),
                scores,
            };

            let idx = match free.pop() {
                Some(i) => {
                    slab[i] = Some(staged);
                    i
                }
                None => {
                    slab.push(Some(staged));
                    slab.len() - 1
                }
            };

            match heap.push(key, id, idx) {
                Admission::Accepted(None) => {}
                Admission::Accepted(Some(evicted)) => {
                    let s = slab[evicted].take().expect("힙이 가리키는 자리는 차 있다");
                    free.push(evicted);
                    self.record(
                        &mut rejected,
                        &mut counts,
                        s.candidate,
                        RejectReason::OutOfPool,
                        None,
                    );
                }
                Admission::Rejected => {
                    let s = slab[idx].take().expect("방금 넣은 자리는 차 있다");
                    free.push(idx);
                    self.record(
                        &mut rejected,
                        &mut counts,
                        s.candidate,
                        RejectReason::OutOfPool,
                        None,
                    );
                }
            }
        }

        let pool_size = heap.len() as u32;
        let order = heap.into_sorted_indices();
        let mut staged: Vec<Staged<C>> = order
            .into_iter()
            .map(|i| slab[i].take().expect("힙이 가리키는 자리는 차 있다"))
            .collect();

        // ── 2단계: 풀 위에서 ──────────────────────────────────────
        self.score_expensive(&mut staged, &mut stats)?;

        let mut survivors: Vec<Staged<C>> = Vec::with_capacity(staged.len());
        for s in staged {
            let mut failed: Option<ScorerId> = None;
            if self.missing == MissingPolicy::Reject {
                for (i, v) in s.scores.iter().enumerate() {
                    if v.is_none() {
                        failed = Some(self.meta[i].id.clone());
                        break;
                    }
                }
            }
            match failed {
                Some(id) => self.record(
                    &mut rejected,
                    &mut counts,
                    s.candidate,
                    RejectReason::NotScored(id),
                    None,
                ),
                None => survivors.push(s),
            }
        }

        let traces = self.fuse_all(&survivors);
        let mut pool: Vec<PoolEntry<C>> = survivors
            .into_iter()
            .zip(traces)
            .map(|(s, trace)| PoolEntry {
                fused: trace.recompute(),
                candidate: s.candidate,
                id: s.id,
                scores: s.scores,
                trace,
            })
            .collect();

        if let Some(threshold) = self.threshold {
            let mut kept = Vec::with_capacity(pool.len());
            for e in pool {
                if e.fused < threshold {
                    self.record(
                        &mut rejected,
                        &mut counts,
                        e.candidate,
                        RejectReason::BelowThreshold,
                        Some(e.fused),
                    );
                } else {
                    kept.push(e);
                }
            }
            pool = kept;
        }

        // 융합 점수 내림차순, 동점은 식별자 오름차순. 탐욕의 동점 처리도 이 순서를 딛는다.
        pool.sort_by(|a, b| b.fused.total_cmp(&a.fused).then_with(|| a.id.cmp(&b.id)));

        let budget_trace = self.resolve_budget(&pool);
        let k = match self.budget {
            Budget::Tokens { .. } => None,
            _ => Some(budget_trace.as_ref().map_or_else(
                || match self.budget {
                    Budget::TopK(k) => k as usize,
                    _ => pool.len(),
                },
                |t| t.derived_k as usize,
            )),
        };

        let selector = Selector {
            set_constraints: &self.set_constraints,
            objective: self.objective.as_deref(),
            requirements: &self.requirements,
            k,
            token_budget: match self.budget {
                Budget::Tokens { max } => Some(max),
                _ => None,
            },
            cost: self.cost.as_deref(),
        };
        let chosen = selector.run(&pool)?;

        // ── 결과 조립 ─────────────────────────────────────────────
        let constraint_ids: Vec<ConstraintId> =
            self.set_constraints.iter().map(|c| c.id()).collect();
        let mut ranked: Vec<Ranked<C>> = Vec::with_capacity(chosen.selected.len());
        let mut taken = vec![false; pool.len()];
        for i in &chosen.selected {
            taken[*i] = true;
        }

        let cut_margin = chosen.cut_margin.filter(|m| m.is_finite());

        let mut kept: Vec<(usize, PoolEntry<C>)> = Vec::new();
        for (i, entry) in pool.into_iter().enumerate() {
            if taken[i] {
                kept.push((i, entry));
            } else {
                let reason = match &chosen.blocked_by[i] {
                    Some(id) => RejectReason::SetConstraint(id.clone()),
                    None => RejectReason::Outranked,
                };
                let fused = entry.fused;
                self.record(
                    &mut rejected,
                    &mut counts,
                    entry.candidate,
                    reason,
                    Some(fused),
                );
            }
        }

        for (rank, index) in chosen.selected.iter().enumerate() {
            let pos = kept
                .iter()
                .position(|(i, _)| i == index)
                .expect("선택된 색인은 남겨 둔 목록에 있다");
            let (_, entry) = kept.remove(pos);
            ranked.push(Ranked {
                rank: rank as u32 + 1,
                fused: entry.fused,
                scores: ScoreSet::new(
                    self.meta
                        .iter()
                        .zip(entry.scores.iter())
                        .map(|(m, v)| (m.id.clone(), *v))
                        .collect(),
                ),
                fusion: entry.trace,
                constraints: constraint_ids.clone(),
                candidate: entry.candidate,
            });
        }

        let structure = match &self.objective {
            None => Structure::Modular,
            Some(o) if o.is_submodular() => Structure::Submodular,
            Some(_) => Structure::Unknown,
        };
        let matroids = self
            .set_constraints
            .iter()
            .filter(|c| c.is_matroid())
            .count();
        let non_matroids = self.set_constraints.len() - matroids;
        let (exact, guarantee) = guarantee_of(
            structure,
            matroids,
            non_matroids,
            self.budget.is_knapsack(),
            chosen.repaired,
        );

        Ok(Outcome {
            ranked,
            rejected,
            rejected_counts: counts,
            selection: Selection {
                exact,
                guarantee,
                pool_size,
                pool_exhausted: chosen.pool_exhausted,
                cut_margin,
            },
            trace: RunTrace {
                input_count,
                pool_capacity: capacity as u32,
                admission_scorer,
                scorers: stats,
                budget: budget_trace,
            },
        })
    }

    // ── 내부 ──────────────────────────────────────────────────────

    fn pool_capacity(&self) -> usize {
        let k = match self.budget {
            Budget::TopK(k) => k as usize,
            Budget::TailMass { fallback_k, .. } => fallback_k.max(1) as usize,
            // 배낭형은 몇 개가 들어갈지 미리 모른다. 비용이 1 인 최악을 상정한다.
            Budget::Tokens { max } => max as usize,
        };
        k.saturating_mul(self.pool_multiplier as usize).max(1)
    }

    fn resolve_admission(&self) -> AdmissionKey {
        if let Some(id) = &self.admission {
            let i = self
                .meta
                .iter()
                .position(|m| &m.id == id)
                .expect("validate 가 이미 확인했다");
            return AdmissionKey::Scorer(i);
        }
        if matches!(self.fusion, Fusion::Rrf { .. }) {
            let i = self
                .meta
                .iter()
                .position(|m| m.cost == ScorerCost::Cheap)
                .expect("validate 가 이미 확인했다");
            return AdmissionKey::Scorer(i);
        }
        AdmissionKey::StreamingFusion
    }

    /// 1단계 절단 키. 값이 없는 축은 결측 정책을 따르고, 그래도 값이 없으면
    /// 가장 나쁜 키를 준다 -- 자리가 남을 때만 들어간다는 뜻이다.
    fn admission_value(&self, key: &AdmissionKey, scores: &[Option<f32>]) -> f32 {
        match key {
            AdmissionKey::Scorer(i) => match (scores[*i], self.missing) {
                (Some(v), _) => v,
                (None, MissingPolicy::Impute(v)) => v,
                (None, _) => f32::NEG_INFINITY,
            },
            AdmissionKey::StreamingFusion => {
                let mut sum = 0.0f32;
                let mut weight_sum = 0.0f32;
                let mut max = f32::NEG_INFINITY;
                let mut used = 0usize;
                for (i, m) in self.meta.iter().enumerate() {
                    if m.cost != ScorerCost::Cheap {
                        continue;
                    }
                    let v = match (scores[i], self.missing) {
                        (Some(v), _) => v,
                        (None, MissingPolicy::Impute(v)) => v,
                        (None, _) => continue,
                    };
                    let w = self.weight_of(&m.id);
                    sum += w * v;
                    weight_sum += w;
                    max = max.max(v);
                    used += 1;
                }
                if used == 0 {
                    return f32::NEG_INFINITY;
                }
                match self.fusion {
                    Fusion::Max => max,
                    _ if weight_sum > 0.0 => sum / weight_sum,
                    _ => 0.0,
                }
            }
        }
    }

    fn weight_of(&self, id: &ScorerId) -> f32 {
        match &self.fusion {
            Fusion::WeightedSum { weights } if !weights.is_empty() => weights
                .iter()
                .find(|(k, _)| k == id)
                .map_or(0.0, |(_, w)| *w),
            _ => 1.0,
        }
    }

    /// 비싼 축을 2단계 풀에만 돌린다.
    ///
    /// 후보 하나씩이 아니라 [`Scorer::score_batch`] 로 넘긴다. 기본 구현은 그대로 하나씩
    /// 부르지만, 배치 추론을 쓰는 채점기는 이 한 번의 호출로 풀 전체를 처리할 수 있다.
    #[cfg(not(feature = "parallel"))]
    fn score_expensive(&self, staged: &mut [Staged<C>], stats: &mut [ScorerTrace]) -> Result<()> {
        // 채점을 먼저 다 받아 두고 나서 쓴다. 배치 호출이 후보를 빌려 보는 동안에는
        // 결과를 되돌려 쓸 수 없기 때문이다.
        let mut collected: Vec<(usize, Vec<Option<f32>>, u128)> = Vec::new();
        {
            let refs: Vec<&C> = staged.iter().map(|s| &s.candidate).collect();
            for (i, m) in self.meta.iter().enumerate() {
                if m.cost != ScorerCost::Expensive {
                    continue;
                }
                let start = Instant::now();
                let values = self.scorers[i].score_batch(&refs);
                self.check_batch(&m.id, refs.len(), values.len())?;
                collected.push((i, values, start.elapsed().as_nanos()));
            }
        }
        self.absorb(staged, stats, collected);
        Ok(())
    }

    /// 병렬 채점. 풀을 덩어리로 나눠 배치 호출을 동시에 돌리고 색인 순서로 다시 모으므로
    /// 결정성은 유지된다. 덩어리 하나가 곧 배치 하나라, 배치 추론을 쓰는 채점기에도 맞는다.
    #[cfg(feature = "parallel")]
    fn score_expensive(&self, staged: &mut [Staged<C>], stats: &mut [ScorerTrace]) -> Result<()> {
        use rayon::prelude::*;

        let mut collected: Vec<(usize, Vec<Option<f32>>, u128)> = Vec::new();
        {
            let refs: Vec<&C> = staged.iter().map(|s| &s.candidate).collect();
            // usize::div_ceil 은 1.73 부터라 선언한 최소 버전(1.71)에서 못 쓴다.
            let threads = rayon::current_num_threads().max(1);
            let chunk = ((refs.len() + threads - 1) / threads).max(1);

            for (i, m) in self.meta.iter().enumerate() {
                if m.cost != ScorerCost::Expensive {
                    continue;
                }
                let start = Instant::now();
                let scorer = &self.scorers[i];
                let chunks: Vec<Result<Vec<Option<f32>>>> = refs
                    .par_chunks(chunk)
                    .map(|part| {
                        let values = scorer.score_batch(part);
                        self.check_batch(&m.id, part.len(), values.len())?;
                        Ok(values)
                    })
                    .collect();
                // 오류도 입력 덩어리 순서로 확인한다. 길이 오차가 서로 상쇄되어
                // 다른 후보에게 점수가 붙는 것을 전체 길이 검사만으로는 막지 못한다.
                let mut values = Vec::with_capacity(refs.len());
                for part in chunks {
                    values.extend(part?);
                }
                collected.push((i, values, start.elapsed().as_nanos()));
            }
        }
        self.absorb(staged, stats, collected);
        Ok(())
    }

    /// 받아 둔 배치 결과를 풀에 써넣고 기록을 갱신한다.
    fn absorb(
        &self,
        staged: &mut [Staged<C>],
        stats: &mut [ScorerTrace],
        collected: Vec<(usize, Vec<Option<f32>>, u128)>,
    ) {
        for (i, values, elapsed) in collected {
            for (s, v) in staged.iter_mut().zip(values) {
                s.scores[i] = sanitize(v);
            }
            stats[i].elapsed_nanos += elapsed;
            stats[i].calls += staged.len() as u64;
            stats[i].missing += staged.iter().filter(|s| s.scores[i].is_none()).count() as u64;
        }
    }

    /// 배치 결과의 길이가 입력과 같은지 본다. 어긋나면 조용히 쓰지 않고 멈춘다.
    fn check_batch(&self, scorer: &ScorerId, expected: usize, got: usize) -> Result<()> {
        if expected == got {
            Ok(())
        } else {
            Err(Error::BatchLengthMismatch {
                scorer: scorer.clone(),
                expected,
                got,
            })
        }
    }

    /// 풀 전체를 한꺼번에 융합한다. 순위 융합은 풀이 있어야 순위가 나오므로 여기서 돈다.
    fn fuse_all(&self, staged: &[Staged<C>]) -> Vec<FusionTrace> {
        let n = staged.len();
        let axes = self.meta.len();

        // 순위 융합용 순위표. ranks[axis][entry] = 1 부터의 순위, 값 없으면 None.
        let mut ranks: Vec<Vec<Option<u32>>> = Vec::new();
        if matches!(self.fusion, Fusion::Rrf { .. }) {
            for axis in 0..axes {
                let mut order: Vec<usize> = (0..n)
                    .filter(|i| self.effective(staged[*i].scores[axis]).is_some())
                    .collect();
                order.sort_by(|a, b| {
                    let va = self.effective(staged[*a].scores[axis]).unwrap_or(0.0);
                    let vb = self.effective(staged[*b].scores[axis]).unwrap_or(0.0);
                    vb.total_cmp(&va)
                        .then_with(|| staged[*a].id.cmp(&staged[*b].id))
                });
                let mut column = vec![None; n];
                for (rank, i) in order.into_iter().enumerate() {
                    column[i] = Some(rank as u32 + 1);
                }
                ranks.push(column);
            }
        }

        (0..n)
            .map(|i| {
                let mut terms: Vec<FusionTerm> = Vec::with_capacity(axes);

                // 가중합은 참여한 축의 가중치로 재정규화한다.
                let mut weight_sum = 0.0f32;
                if matches!(self.fusion, Fusion::WeightedSum { .. }) {
                    for (axis, m) in self.meta.iter().enumerate() {
                        if self.effective(staged[i].scores[axis]).is_some() {
                            weight_sum += self.weight_of(&m.id);
                        }
                    }
                }

                for (axis, m) in self.meta.iter().enumerate() {
                    let raw = staged[i].scores[axis];
                    let effective = self.effective(raw);

                    let (input, weight, contribution) = match (&self.fusion, effective) {
                        (_, None) => (FusionInput::Skipped, 0.0, 0.0),
                        (Fusion::Rrf { k }, Some(_)) => match ranks[axis][i] {
                            Some(r) => (FusionInput::Rank(r), 1.0, 1.0 / (k + r as f32)),
                            None => (FusionInput::Skipped, 0.0, 0.0),
                        },
                        (Fusion::WeightedSum { .. }, Some(v)) => {
                            let w = if weight_sum > 0.0 {
                                self.weight_of(&m.id) / weight_sum
                            } else {
                                0.0
                            };
                            (self.input_kind(raw, v), w, w * v)
                        }
                        (Fusion::Max, Some(v)) => (self.input_kind(raw, v), 1.0, v),
                    };

                    terms.push(FusionTerm {
                        scorer: m.id.clone(),
                        input,
                        weight,
                        contribution,
                    });
                }

                FusionTrace {
                    method: self.fusion.method(),
                    k: match self.fusion {
                        Fusion::Rrf { k } => Some(k),
                        _ => None,
                    },
                    missing: self.missing,
                    terms,
                }
            })
            .collect()
    }

    /// 결측 정책까지 반영한 값.
    fn effective(&self, raw: Option<f32>) -> Option<f32> {
        match (raw, self.missing) {
            (Some(v), _) => Some(v),
            (None, MissingPolicy::Impute(v)) => Some(v),
            (None, _) => None,
        }
    }

    fn input_kind(&self, raw: Option<f32>, effective: f32) -> FusionInput {
        match raw {
            Some(_) => FusionInput::Value(effective),
            None => FusionInput::Imputed(effective),
        }
    }

    fn resolve_budget(&self, pool: &[PoolEntry<C>]) -> Option<BudgetTrace> {
        match self.budget {
            Budget::TailMass {
                epsilon,
                fallback_k,
            } => {
                let mass: Vec<f64> = pool.iter().map(|e| e.fused as f64).collect();
                Some(derive_budget(&mass, epsilon, fallback_k, self.min_fit))
            }
            _ => None,
        }
    }

    fn record(
        &self,
        rejected: &mut Vec<Rejected<C>>,
        counts: &mut RejectCounts,
        candidate: C,
        reason: RejectReason,
        fused: Option<f32>,
    ) {
        counts.record(&reason);
        let keep = match self.rejections {
            Rejections::Keep => true,
            Rejections::Count => false,
            Rejections::Sample(n) => rejected.len() < n,
        };
        if keep {
            rejected.push(Rejected {
                candidate,
                reason,
                fused,
            });
        }
    }
}

/// `NaN` 은 순서를 매길 수 없으므로 결측과 같이 다룬다.
fn sanitize(v: Option<f32>) -> Option<f32> {
    v.filter(|x| !x.is_nan())
}
