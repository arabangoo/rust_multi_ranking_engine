//! 스트리밍 유계 힙과 제약 아래 선택.
//!
//! 두 단계로 나뉘는 이유는 스트리밍과 집합 제약이 근본적으로 충돌하기 때문이다.
//! 집합 제약을 보려면 후보 풀이 있어야 하는데 스트리밍은 풀을 만들지 않는 것이 목적이다.
//!
//! ```text
//! 1단계 (스트리밍)   후보 N개 -> 단항 제약 -> 싼 채점기 -> 유계 힙 상위 M개
//!                    메모리 O(M), 시간 O(N log M)
//! 2단계 (풀 위에서)  상위 M개 -> 비싼 채점기 -> 집합 제약 아래 선택 -> 최종 K개
//! ```

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use crate::candidate::CandidateId;
use crate::constraint::{ConstraintId, Requirement, SetConstraint};
use crate::error::{Error, Result};
use crate::fuse::FusionTrace;
use crate::objective::SetObjective;

/// 서브모듈러 목적함수에 개수 제한만 걸렸을 때의 탐욕 보장 계수. `1 - 1/e`.
pub const GUARANTEE_CARDINALITY: f32 = 0.632_120_56;
/// 서브모듈러 목적함수에 매트로이드 하나가 걸렸을 때의 탐욕 보장 계수.
pub const GUARANTEE_MATROID: f32 = 0.5;
/// 모듈러 목적함수에 배낭형 하나가 걸렸을 때의 탐욕 보장 계수.
pub const GUARANTEE_KNAPSACK_MODULAR: f32 = 0.5;
/// 서브모듈러 목적함수에 배낭형 하나가 걸렸을 때의 탐욕 보장 계수. `1 - e^(-1/2)`.
pub const GUARANTEE_KNAPSACK_SUBMODULAR: f32 = 0.393_469_34;

/// 1단계 유계 힙에 들어가는 자리 하나.
///
/// 순서는 "좋음"의 순서다. 키가 클수록 좋고, 키가 같으면 식별자가 작을수록 좋다.
/// 동점에 임의 순서를 남기지 않으려는 것이고, 이것이 결정성 불변식의 바닥이다.
#[derive(Debug)]
struct Slot {
    key: f32,
    id: CandidateId,
    index: usize,
}

impl PartialEq for Slot {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Slot {}

impl PartialOrd for Slot {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Slot {
    fn cmp(&self, other: &Self) -> Ordering {
        // total_cmp 는 NaN 까지 포함하는 전순서를 준다. 엔진은 NaN 점수를 결측으로
        // 바꿔 넣으므로 여기까지 NaN 이 오지는 않지만, 순서 자체를 전순서로 두는 편이
        // 결정성을 코드로 못박는 방법이다.
        self.key
            .total_cmp(&other.key)
            .then_with(|| other.id.cmp(&self.id))
    }
}

/// 상위 M 개만 남기는 유계 힙.
///
/// 전체 정렬 `O(N log N)` 대신 `O(N log M)` 이고, 더 중요한 것은 후보를 전부 메모리에
/// 올리지 않는다는 점이다.
pub(crate) struct BoundedHeap {
    capacity: usize,
    heap: BinaryHeap<Reverse<Slot>>,
}

/// 후보 하나를 넣은 결과.
pub(crate) enum Admission {
    /// 힙에 들어갔다. 밀려난 것이 있으면 그 색인.
    Accepted(Option<usize>),
    /// 힙에 못 들어갔다.
    Rejected,
}

impl BoundedHeap {
    pub(crate) fn new(capacity: usize) -> Self {
        BoundedHeap {
            capacity,
            heap: BinaryHeap::with_capacity(capacity.min(1024)),
        }
    }

    pub(crate) fn push(&mut self, key: f32, id: CandidateId, index: usize) -> Admission {
        let slot = Slot { key, id, index };
        if self.heap.len() < self.capacity {
            self.heap.push(Reverse(slot));
            return Admission::Accepted(None);
        }
        // capacity 가 0 이면 무엇도 들어가지 않는다.
        let Some(Reverse(worst)) = self.heap.peek() else {
            return Admission::Rejected;
        };
        if slot > *worst {
            let evicted = self
                .heap
                .pop()
                .expect("peek 이 값을 줬으므로 pop 도 준다")
                .0;
            self.heap.push(Reverse(slot));
            Admission::Accepted(Some(evicted.index))
        } else {
            Admission::Rejected
        }
    }

    /// 좋은 순서대로 색인을 낸다.
    pub(crate) fn into_sorted_indices(self) -> Vec<usize> {
        let mut slots: Vec<Slot> = self.heap.into_iter().map(|Reverse(s)| s).collect();
        slots.sort_by(|a, b| b.cmp(a));
        slots.into_iter().map(|s| s.index).collect()
    }

    pub(crate) fn len(&self) -> usize {
        self.heap.len()
    }
}

/// 2단계 풀에 올라온 후보 하나.
pub(crate) struct PoolEntry<C> {
    pub candidate: C,
    pub id: CandidateId,
    pub scores: Vec<Option<f32>>,
    pub fused: f32,
    pub trace: FusionTrace,
}

/// 선택기에 넘기는 재료. 엔진이 조립해서 준다.
pub(crate) struct Selector<'a, C> {
    pub set_constraints: &'a [Box<dyn SetConstraint<C>>],
    pub objective: Option<&'a dyn SetObjective<C>>,
    pub requirements: &'a [Requirement<C>],
    /// 개수 상한. 배낭형 예산이면 `None`.
    pub k: Option<usize>,
    /// 비용 상한. 개수 예산이면 `None`.
    pub token_budget: Option<u32>,
    /// 후보별 비용. 배낭형 예산일 때만 있다.
    pub cost: Option<&'a (dyn Fn(&C) -> u32 + Sync)>,
}

/// 선택 결과.
pub(crate) struct Chosen {
    /// 고른 후보의 풀 색인. 고른 순서다.
    pub selected: Vec<usize>,
    /// 각 미선택 후보를 막은 집합 제약. 막힌 것이 없으면 `None`(= 밀려난 것).
    pub blocked_by: Vec<Option<ConstraintId>>,
    /// 풀을 다 쓰고도 K 를 못 채웠는가.
    pub pool_exhausted: bool,
    /// 요구 조건 교체가 일어났는가. 일어났으면 최적성 선언을 내린다.
    pub repaired: bool,
    /// 마지막 자리를 다툰 두 값의 차. 잣대는 선택기가 실제로 쓴 기준이다.
    pub cut_margin: Option<f32>,
}

impl<'a, C> Selector<'a, C> {
    /// 제약과 예산 아래에서 고른다.
    pub(crate) fn run(&self, pool: &[PoolEntry<C>]) -> Result<Chosen> {
        let mut chosen = if self.token_budget.is_some() {
            self.knapsack(pool)?
        } else {
            self.cardinality(pool)?
        };

        chosen.repaired = self.repair_requirements(pool, &mut chosen.selected)?;
        chosen.blocked_by = self.blocking_reasons(pool, &chosen.selected);
        if chosen.cut_margin.is_none() {
            chosen.cut_margin = self.cut_margin(pool, &chosen.selected);
        }
        Ok(chosen)
    }

    /// 개수 제한 아래의 탐욕.
    fn cardinality(&self, pool: &[PoolEntry<C>]) -> Result<Chosen> {
        // 요청한 K 와 풀에서 실제로 채울 수 있는 상한을 따로 둔다. 요청값을 풀 크기로
        // 미리 잘라 버리면 "풀을 다 쓰고도 K 를 못 채웠다"는 비교가 사라진다.
        let requested = self.k.unwrap_or(pool.len());
        let k = requested.min(pool.len());
        let mut selected: Vec<usize> = Vec::with_capacity(k);
        let mut taken = vec![false; pool.len()];
        let mut exhausted = false;

        while selected.len() < k {
            let refs = self.refs(pool, &selected);
            let mut best: Option<(usize, f32)> = None;

            for (i, entry) in pool.iter().enumerate() {
                if taken[i] || !self.admits_all(&refs, &entry.candidate) {
                    continue;
                }
                let gain = self.gain(&refs, entry);
                match best {
                    // 동점은 풀 순서(= 융합 점수 내림차순, 식별자 오름차순)가 가른다.
                    Some((_, b)) if gain <= b => {}
                    _ => best = Some((i, gain)),
                }
            }

            match best {
                Some((i, _)) => {
                    taken[i] = true;
                    selected.push(i);
                }
                None => {
                    exhausted = true;
                    break;
                }
            }
        }

        if selected.len() < requested {
            exhausted = true;
        }

        Ok(Chosen {
            selected,
            blocked_by: Vec::new(),
            pool_exhausted: exhausted,
            repaired: false,
            cut_margin: None,
        })
    }

    /// 배낭형 예산 아래의 탐욕.
    ///
    /// 비용 대비 이득으로 고른 집합과 단일 최고 항목을 비교해 나은 쪽을 쓴다. 이
    /// 두 갈래 비교가 보장 계수의 근거다 -- 비율 탐욕만 쓰면 값이 아주 큰 단일 항목을
    /// 통째로 놓치는 경우가 있어 계수가 성립하지 않는다.
    fn knapsack(&self, pool: &[PoolEntry<C>]) -> Result<Chosen> {
        let limit = self.token_budget.unwrap_or(u32::MAX) as u64;
        let cost_of = |c: &C| -> u64 {
            match self.cost {
                Some(f) => f(c) as u64,
                None => 0,
            }
        };

        // 갈래 1: 비용 대비 이득 탐욕.
        let mut selected: Vec<usize> = Vec::new();
        let mut taken = vec![false; pool.len()];
        let mut spent: u64 = 0;
        let mut value_ratio = 0.0f32;
        // 이 반복문은 더 담을 것이 없을 때만 빠져나가므로 그 자리에서 값이 정해진다.
        let exhausted: bool;

        loop {
            let refs = self.refs(pool, &selected);
            let mut best: Option<(usize, f32, f32, u64)> = None;

            for (i, entry) in pool.iter().enumerate() {
                if taken[i] {
                    continue;
                }
                let cost = cost_of(&entry.candidate);
                if spent + cost > limit || !self.admits_all(&refs, &entry.candidate) {
                    continue;
                }
                let gain = self.gain(&refs, entry);
                // 비용 0 인 후보는 공짜이므로 언제나 먼저 담는다.
                let ratio = if cost == 0 {
                    f32::INFINITY
                } else {
                    gain / cost as f32
                };
                match best {
                    Some((_, b, _, _)) if ratio <= b => {}
                    _ => best = Some((i, ratio, gain, cost)),
                }
            }

            match best {
                Some((i, _, gain, cost)) => {
                    taken[i] = true;
                    selected.push(i);
                    spent += cost;
                    value_ratio += gain;
                }
                None => {
                    // 배낭형에는 채워야 할 K 가 없다. 예산이 남았는데 더 담을 것이
                    // 없을 때만 풀이 모자랐다고 본다.
                    exhausted = selected.len() == pool.len();
                    break;
                }
            }
        }

        // 갈래 2: 단일 최고 항목.
        let empty: Vec<&C> = Vec::new();
        let mut single: Option<(usize, f32)> = None;
        for (i, entry) in pool.iter().enumerate() {
            if cost_of(&entry.candidate) > limit || !self.admits_all(&empty, &entry.candidate) {
                continue;
            }
            let gain = self.gain(&empty, entry);
            match single {
                Some((_, b)) if gain <= b => {}
                _ => single = Some((i, gain)),
            }
        }

        if let Some((i, gain)) = single {
            if gain > value_ratio {
                return Ok(Chosen {
                    selected: vec![i],
                    blocked_by: Vec::new(),
                    pool_exhausted: exhausted,
                    repaired: false,
                    // 단일 최고 항목이 이겼으면 절단선이라 부를 자리가 없다.
                    // 마지막 자리를 다툰 사건 자체가 일어나지 않았다.
                    cut_margin: Some(f32::NAN),
                });
            }
        }

        Ok(Chosen {
            selected,
            blocked_by: Vec::new(),
            pool_exhausted: exhausted,
            repaired: false,
            cut_margin: None,
        })
    }

    /// 요구 조건을 교체로 채운다. 채우려고 무엇이든 바꿨으면 참을 돌려준다.
    fn repair_requirements(
        &self,
        pool: &[PoolEntry<C>],
        selected: &mut Vec<usize>,
    ) -> Result<bool> {
        let mut repaired = false;

        for req in self.requirements {
            loop {
                let have = selected
                    .iter()
                    .filter(|&&i| req.satisfied_by(&pool[i].candidate))
                    .count();
                if have >= req.needed() {
                    break;
                }

                // 조건을 만족하는 미선택 후보를 점수 순으로 본다.
                let mut donors: Vec<usize> = (0..pool.len())
                    .filter(|i| !selected.contains(i) && req.satisfied_by(&pool[*i].candidate))
                    .collect();
                donors.sort_by(|a, b| {
                    pool[*b]
                        .fused
                        .total_cmp(&pool[*a].fused)
                        .then_with(|| pool[*a].id.cmp(&pool[*b].id))
                });

                // 조건을 만족하지 않는 선택분을 점수가 낮은 쪽부터 내보낸다.
                let mut victims: Vec<usize> = selected
                    .iter()
                    .copied()
                    .filter(|i| !req.satisfied_by(&pool[*i].candidate))
                    .collect();
                victims.sort_by(|a, b| {
                    pool[*a]
                        .fused
                        .total_cmp(&pool[*b].fused)
                        .then_with(|| pool[*b].id.cmp(&pool[*a].id))
                });

                let mut done = false;
                'outer: for donor in &donors {
                    // 자리가 남아 있으면 내보낼 것 없이 그냥 넣는다.
                    let room = match self.k {
                        Some(k) => selected.len() < k,
                        None => false,
                    };
                    if room {
                        let mut trial = selected.clone();
                        trial.push(*donor);
                        if self.set_is_feasible(pool, &trial) {
                            *selected = trial;
                            repaired = true;
                            done = true;
                            break 'outer;
                        }
                    }
                    for victim in &victims {
                        let mut trial: Vec<usize> =
                            selected.iter().copied().filter(|i| i != victim).collect();
                        trial.push(*donor);
                        if self.set_is_feasible(pool, &trial) {
                            *selected = trial;
                            repaired = true;
                            done = true;
                            break 'outer;
                        }
                    }
                }

                if !done {
                    let available = (0..pool.len())
                        .filter(|i| req.satisfied_by(&pool[*i].candidate))
                        .count();
                    return Err(Error::InfeasibleRequirement {
                        id: req.id().to_string(),
                        needed: req.needed(),
                        available,
                    });
                }
            }
        }

        Ok(repaired)
    }

    /// 집합 전체가 제약을 지키는가. 원소 하나씩 빼고 나머지에 대해 물어본다.
    fn set_is_feasible(&self, pool: &[PoolEntry<C>], set: &[usize]) -> bool {
        if let Some(k) = self.k {
            if set.len() > k {
                return false;
            }
        }
        if let (Some(limit), Some(cost)) = (self.token_budget, self.cost) {
            let spent: u64 = set.iter().map(|i| cost(&pool[*i].candidate) as u64).sum();
            if spent > limit as u64 {
                return false;
            }
        }
        for (pos, i) in set.iter().enumerate() {
            let others: Vec<&C> = set
                .iter()
                .enumerate()
                .filter(|(p, _)| *p != pos)
                .map(|(_, j)| &pool[*j].candidate)
                .collect();
            if !self.admits_all(&others, &pool[*i].candidate) {
                return false;
            }
        }
        true
    }

    /// 마지막 자리를 다툰 두 값의 차.
    ///
    /// **잣대는 선택기가 실제로 쓴 기준이다.** 개수 예산이면 총 이득, 배낭형이면 비용
    /// 대비 이득이다. 융합 점수로 재면 목적함수가 걸렸을 때 음수가 나오는데, 그 숫자는
    /// 순서가 뒤집혔다는 뜻이 아니라 잣대를 잘못 골랐다는 뜻이다.
    ///
    /// 비교 대상은 마지막 하나를 뺀 집합을 기준으로 다시 판정한다. 그 시점에 실제로
    /// 자리를 다툰 후보가 그것들이기 때문이다.
    fn cut_margin(&self, pool: &[PoolEntry<C>], selected: &[usize]) -> Option<f32> {
        let last = *selected.last()?;
        let prefix = &selected[..selected.len() - 1];
        let refs = self.refs(pool, prefix);

        let cost_of = |c: &C| -> u64 {
            match self.cost {
                Some(f) => f(c) as u64,
                None => 0,
            }
        };
        let remaining = self.token_budget.map(|limit| {
            (limit as u64).saturating_sub(prefix.iter().map(|i| cost_of(&pool[*i].candidate)).sum())
        });

        let score = |entry: &PoolEntry<C>| -> f32 {
            let gain = self.gain(&refs, entry);
            match remaining {
                None => gain,
                Some(_) => {
                    let cost = cost_of(&entry.candidate);
                    if cost == 0 {
                        f32::INFINITY
                    } else {
                        gain / cost as f32
                    }
                }
            }
        };

        // Option::is_none_or 는 1.82 부터라 선언한 최소 버전(1.71)에서 못 쓴다.
        let fits = |entry: &PoolEntry<C>| -> bool {
            match remaining {
                None => true,
                Some(r) => cost_of(&entry.candidate) <= r,
            }
        };

        let own = score(&pool[last]);
        let best_left = pool
            .iter()
            .enumerate()
            .filter(|(i, e)| {
                !selected.contains(i) && fits(e) && self.admits_all(&refs, &e.candidate)
            })
            .map(|(_, e)| score(e))
            .fold(None::<f32>, |acc, v| Some(acc.map_or(v, |a| a.max(v))))?;

        Some(own - best_left)
    }

    /// 미선택 후보마다 무엇이 막았는지 기록한다.
    ///
    /// 최종 집합을 기준으로 다시 물어본다. 어떤 집합 제약이 거부하면 그 제약이 사유이고,
    /// 전부 통과하는데도 못 들어갔으면 더 높은 후보에게 밀린 것이다.
    fn blocking_reasons(
        &self,
        pool: &[PoolEntry<C>],
        selected: &[usize],
    ) -> Vec<Option<ConstraintId>> {
        let refs = self.refs(pool, selected);
        let mut out = vec![None; pool.len()];
        for (i, entry) in pool.iter().enumerate() {
            if selected.contains(&i) {
                continue;
            }
            for c in self.set_constraints {
                if !c.admits(&refs, &entry.candidate) {
                    out[i] = Some(c.id());
                    break;
                }
            }
        }
        out
    }

    fn refs<'p>(&self, pool: &'p [PoolEntry<C>], selected: &[usize]) -> Vec<&'p C> {
        selected.iter().map(|i| &pool[*i].candidate).collect()
    }

    fn admits_all(&self, selected: &[&C], candidate: &C) -> bool {
        self.set_constraints
            .iter()
            .all(|c| c.admits(selected, candidate))
    }

    /// 총 이득 = 융합 점수 + 목적함수의 한계 이득.
    fn gain(&self, selected: &[&C], entry: &PoolEntry<C>) -> f32 {
        match self.objective {
            Some(o) => entry.fused + o.marginal_gain(selected, &entry.candidate),
            None => entry.fused,
        }
    }
}

/// 목적함수의 구조.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Structure {
    /// 목적함수가 없다. 융합 점수의 합이므로 모듈러다.
    Modular,
    /// 서브모듈러라고 선언됐다.
    Submodular,
    /// 목적함수는 있는데 서브모듈러가 아니라고 선언됐다. 아무 보장도 없다.
    Unknown,
}

/// 이 조합에서 탐욕이 무엇을 보장하는가.
///
/// 표는 [`Selection::guarantee`](crate::Selection::guarantee) 문서에 있다. 표에 없는
/// 조합에는 계수를 주지 않는다 -- 근거 없는 숫자를 결과에 싣지 않기 위해서다.
pub(crate) fn guarantee_of(
    structure: Structure,
    matroids: usize,
    non_matroids: usize,
    knapsack_budget: bool,
    repaired: bool,
) -> (bool, Option<f32>) {
    // 요구 조건 교체는 탐욕이 만든 순서를 사후에 뒤집으므로 최적성도 보장 계수도 남지 않는다.
    if repaired {
        return (false, None);
    }
    if structure == Structure::Unknown {
        return (false, None);
    }

    let q = non_matroids + usize::from(knapsack_budget);
    let p = matroids;

    match (structure, q, p) {
        (Structure::Modular, 0, 0 | 1) => (true, None),
        (Structure::Modular, 0, p) => (false, Some(1.0 / p as f32)),
        (Structure::Submodular, 0, 0) => (false, Some(GUARANTEE_CARDINALITY)),
        (Structure::Submodular, 0, 1) => (false, Some(GUARANTEE_MATROID)),
        (Structure::Submodular, 0, p) => (false, Some(1.0 / (p + 1) as f32)),
        (Structure::Modular, 1, 0) => (false, Some(GUARANTEE_KNAPSACK_MODULAR)),
        (Structure::Submodular, 1, 0) => (false, Some(GUARANTEE_KNAPSACK_SUBMODULAR)),
        _ => (false, None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key_order(pairs: &[(f32, u64)], capacity: usize) -> Vec<usize> {
        let mut heap = BoundedHeap::new(capacity);
        for (i, (k, id)) in pairs.iter().enumerate() {
            heap.push(*k, CandidateId::num(*id), i);
        }
        heap.into_sorted_indices()
    }

    #[test]
    fn the_heap_keeps_the_best_m_and_orders_them() {
        let out = key_order(&[(0.1, 1), (0.9, 2), (0.5, 3), (0.7, 4)], 2);
        assert_eq!(out, vec![1, 3]); // 0.9 다음 0.7
    }

    #[test]
    fn ties_break_on_the_smaller_identifier() {
        // 같은 점수 셋 중 둘만 남으면 식별자가 작은 둘이 남아야 한다.
        let out = key_order(&[(0.5, 30), (0.5, 10), (0.5, 20)], 2);
        assert_eq!(out, vec![1, 2]); // id 10, id 20
    }

    #[test]
    fn arrival_order_does_not_change_the_result() {
        let a = key_order(&[(0.3, 1), (0.8, 2), (0.5, 3)], 2);
        let b = key_order(&[(0.5, 3), (0.3, 1), (0.8, 2)], 2);
        // 색인이 아니라 남은 식별자로 비교한다.
        assert_eq!(a, vec![1, 2]);
        assert_eq!(b, vec![2, 0]);
    }

    #[test]
    fn a_zero_capacity_heap_admits_nothing() {
        let out = key_order(&[(0.5, 1)], 0);
        assert!(out.is_empty());
    }

    #[test]
    fn guarantees_follow_the_documented_table() {
        assert_eq!(
            guarantee_of(Structure::Modular, 0, 0, false, false),
            (true, None)
        );
        assert_eq!(
            guarantee_of(Structure::Modular, 1, 0, false, false),
            (true, None)
        );
        assert_eq!(
            guarantee_of(Structure::Modular, 3, 0, false, false),
            (false, Some(1.0 / 3.0))
        );
        assert_eq!(
            guarantee_of(Structure::Submodular, 0, 0, false, false),
            (false, Some(GUARANTEE_CARDINALITY))
        );
        assert_eq!(
            guarantee_of(Structure::Submodular, 1, 0, false, false),
            (false, Some(GUARANTEE_MATROID))
        );
        assert_eq!(
            guarantee_of(Structure::Submodular, 0, 0, true, false),
            (false, Some(GUARANTEE_KNAPSACK_SUBMODULAR))
        );
        // 표에 없는 조합에는 계수를 주지 않는다.
        assert_eq!(
            guarantee_of(Structure::Submodular, 2, 1, false, false),
            (false, None)
        );
        assert_eq!(
            guarantee_of(Structure::Unknown, 0, 0, false, false),
            (false, None)
        );
        // 교체가 일어나면 아무것도 남지 않는다.
        assert_eq!(
            guarantee_of(Structure::Modular, 0, 0, false, true),
            (false, None)
        );
    }
}
