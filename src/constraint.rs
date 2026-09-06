//! 단항 제약, 집합 제약, 매트로이드 판정, 요구 조건.
//!
//! 제약이 필터가 아닐 때가 이 엔진에서 가장 어려운 자리다. 단항 제약은 후보 하나만
//! 보면 되지만 집합 제약은 이미 고른 것들을 봐야 한다. 집합 제약이 걸리면 **상위 K 개가
//! 정답이 아니게 된다** -- 점수 1위·2위·3위가 전부 같은 출처면 그중 하나를 버리고
//! 4위를 넣어야 한다.

use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;

/// 제약 식별자. 탈락 사유와 감사 출력에 그대로 실린다.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConstraintId(Box<str>);

impl ConstraintId {
    /// 식별자를 만든다.
    pub fn new(name: impl Into<Box<str>>) -> Self {
        ConstraintId(name.into())
    }

    /// 이름을 빌려 본다.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ConstraintId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for ConstraintId {
    fn from(v: &str) -> Self {
        ConstraintId::new(v)
    }
}

/// 후보 하나만 보고 판정하는 제약.
///
/// 1단계에서 채점보다 **먼저** 돌린다. 후보가 1,000만 개일 때 떨어질 후보에 채점기를
/// 돌리는 것은 낭비이기 때문이다. 그 결과 어떤 후보가 단항 제약과 필수 축 결측에
/// 동시에 걸리면 사유는 [`UnaryConstraint`](crate::RejectReason::UnaryConstraint) 로 남는다.
pub trait UnaryConstraint<C>: Sync {
    /// 이 제약의 식별자.
    fn id(&self) -> ConstraintId;

    /// 이 후보를 통과시키는가.
    fn allows(&self, c: &C) -> bool;
}

/// 이미 고른 것들을 봐야 판정할 수 있는 제약.
///
/// # `is_matroid` 를 왜 구현자에게 맡기는가
///
/// 임의의 제약이 매트로이드인지 엔진이 자동으로 판정할 수는 없다. 그래서 선언받는다.
/// 대신 **엔진이 기본 제공하는 제약은 그 성질이 증명된 것만** 둔다.
///
/// 사용자가 직접 구현하면서 `is_matroid` 를 참으로 잘못 선언하면 보장이 깨진다. 그래서
/// 결과의 [`Selection::exact`](crate::Selection::exact) 는 엔진이 그렇게 **선언받았다**는
/// 뜻이지 엔진이 증명했다는 뜻이 아니다.
pub trait SetConstraint<C>: Sync {
    /// 이 제약의 식별자.
    fn id(&self) -> ConstraintId;

    /// 이 후보를 현재 선택에 더할 수 있는가.
    fn admits(&self, selected: &[&C], candidate: &C) -> bool;

    /// 매트로이드 구조인가. 참이면 탐욕이 최적임을 보장할 수 있다.
    fn is_matroid(&self) -> bool;
}

/// 최종 집합이 반드시 만족해야 하는 하한 조건.
///
/// # 설계서에서 달라진 점
///
/// 설계서는 `require_at_least` 를 기본 제공 [`SetConstraint`] 로 적었는데, 그 자리에
/// 넣으면 아무 일도 하지 않는다. `admits` 는 "이것을 더해도 되는가"를 묻는 술어라
/// **더하는 것을 막는 상한**만 표현할 수 있고, 하한은 항상 참을 돌려주게 되기 때문이다.
/// 그래서 별도 개념으로 분리했다.
///
/// 하한은 탐욕이 끝난 뒤 **교체**로 채운다. 조건을 만족하는 미선택 후보 중 가장 점수가
/// 높은 것을, 조건을 만족하지 않는 선택된 후보 중 가장 점수가 낮은 것과 바꾼다.
/// 앞서 충족한 하한을 보존하고 최종 집합의 모든 하한을 재검사한다.
/// 단일 교체로 찾지 못하면 오류를 반환하며, 해가 없음을 증명하지는 않는다.
/// 실제 교체가 일어나면 [`Selection::exact`](crate::Selection::exact) 는 거짓이 된다.
pub struct Requirement<C> {
    id: ConstraintId,
    predicate: Box<dyn Fn(&C) -> bool + Sync>,
    at_least: usize,
}

impl<C> Requirement<C> {
    /// 술어를 만족하는 후보가 최소 `n` 개 들어가야 한다는 조건을 만든다.
    pub fn at_least(
        id: impl Into<ConstraintId>,
        n: usize,
        predicate: impl Fn(&C) -> bool + Sync + 'static,
    ) -> Self {
        Requirement {
            id: id.into(),
            predicate: Box::new(predicate),
            at_least: n,
        }
    }

    /// 이 조건의 식별자.
    pub fn id(&self) -> ConstraintId {
        self.id.clone()
    }

    /// 최소 몇 개가 필요한가.
    pub fn needed(&self) -> usize {
        self.at_least
    }

    /// 이 후보가 조건을 만족하는가.
    pub fn satisfied_by(&self, c: &C) -> bool {
        (self.predicate)(c)
    }
}

// ── 기본 제공 제약 ────────────────────────────────────────────────
//
// | 제약 | 매트로이드 | 보장 |
// | --- | --- | --- |
// | max_per_group | 예 (분할 매트로이드) | 탐욕이 정확 |
// | max_total | 예 (균일 매트로이드) | 탐욕이 정확 |
// | cost_budget | 아니오 (배낭형) | 근사 |

/// 그룹마다 최대 `m` 개. 분할 매트로이드다.
///
/// 그룹 키를 뽑는 함수를 받는다. 같은 출처에서 3개를 넘지 말 것, 같은 유전자에서
/// 3개를 넘지 말 것 같은 조건이 여기 들어간다.
pub struct MaxPerGroup<C, K, F> {
    id: ConstraintId,
    key: F,
    max: usize,
    _marker: std::marker::PhantomData<fn(&C) -> K>,
}

impl<C, K, F> MaxPerGroup<C, K, F>
where
    K: Eq + Hash,
    F: Fn(&C) -> K + Sync,
{
    /// 그룹당 최대 개수를 정한다.
    pub fn new(id: impl Into<ConstraintId>, max: usize, key: F) -> Self {
        MaxPerGroup {
            id: id.into(),
            key,
            max,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<C, K, F> SetConstraint<C> for MaxPerGroup<C, K, F>
where
    C: Sync,
    K: Eq + Hash + Sync,
    F: Fn(&C) -> K + Sync,
{
    fn id(&self) -> ConstraintId {
        self.id.clone()
    }

    fn admits(&self, selected: &[&C], candidate: &C) -> bool {
        if self.max == 0 {
            return false;
        }
        let target = (self.key)(candidate);
        let mut count = 0usize;
        for s in selected {
            if (self.key)(s) == target {
                count += 1;
                if count >= self.max {
                    return false;
                }
            }
        }
        true
    }

    fn is_matroid(&self) -> bool {
        true
    }
}

/// 전체 최대 `k` 개. 균일 매트로이드다.
///
/// [`Budget::TopK`](crate::Budget::TopK) 가 이미 개수 상한을 걸고 있으므로 보통은 필요
/// 없다. 예산이 [`Tokens`](crate::Budget::Tokens) 인데 개수 상한도 함께 걸고 싶을 때 쓴다.
pub struct MaxTotal {
    id: ConstraintId,
    max: usize,
}

impl MaxTotal {
    /// 전체 상한을 정한다.
    pub fn new(id: impl Into<ConstraintId>, max: usize) -> Self {
        MaxTotal { id: id.into(), max }
    }
}

impl<C> SetConstraint<C> for MaxTotal {
    fn id(&self) -> ConstraintId {
        self.id.clone()
    }

    fn admits(&self, selected: &[&C], _candidate: &C) -> bool {
        selected.len() < self.max
    }

    fn is_matroid(&self) -> bool {
        true
    }
}

/// 후보별 비용의 합이 상한 이하. 배낭형이라 매트로이드가 아니다.
///
/// [`Budget::Tokens`](crate::Budget::Tokens) 와 같은 형태이지만 이쪽은 축이 여럿일 때
/// 쓴다. 예산이 하나면 `Budget::Tokens` 가 더 낫다 -- 선택기가 비용 대비 이득으로
/// 정렬하는 배낭 전용 경로를 타기 때문이다.
pub struct CostBudget<C, F> {
    id: ConstraintId,
    cost: F,
    limit: f64,
    _marker: std::marker::PhantomData<fn(&C)>,
}

impl<C, F> CostBudget<C, F>
where
    F: Fn(&C) -> f64 + Sync,
{
    /// 비용 함수와 상한을 정한다.
    pub fn new(id: impl Into<ConstraintId>, limit: f64, cost: F) -> Self {
        CostBudget {
            id: id.into(),
            cost,
            limit,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<C, F> SetConstraint<C> for CostBudget<C, F>
where
    C: Sync,
    F: Fn(&C) -> f64 + Sync,
{
    fn id(&self) -> ConstraintId {
        self.id.clone()
    }

    fn admits(&self, selected: &[&C], candidate: &C) -> bool {
        let used: f64 = selected.iter().map(|c| (self.cost)(c)).sum();
        used + (self.cost)(candidate) <= self.limit
    }

    fn is_matroid(&self) -> bool {
        false
    }
}

/// 술어 하나로 만드는 단항 제약.
///
/// `min_authority(0.3)` 처럼 후보 하나만 보면 판정되는 조건에 쓴다.
pub struct Predicate<C, F> {
    id: ConstraintId,
    predicate: F,
    _marker: std::marker::PhantomData<fn(&C)>,
}

impl<C, F> Predicate<C, F>
where
    F: Fn(&C) -> bool + Sync,
{
    /// 술어로 단항 제약을 만든다.
    pub fn new(id: impl Into<ConstraintId>, predicate: F) -> Self {
        Predicate {
            id: id.into(),
            predicate,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<C, F> UnaryConstraint<C> for Predicate<C, F>
where
    C: Sync,
    F: Fn(&C) -> bool + Sync,
{
    fn id(&self) -> ConstraintId {
        self.id.clone()
    }

    fn allows(&self, c: &C) -> bool {
        (self.predicate)(c)
    }
}

/// 그룹당 최대 개수 제약을 만든다. [`MaxPerGroup`] 의 짧은 이름.
pub fn max_per_group<C, K, F>(
    id: impl Into<ConstraintId>,
    max: usize,
    key: F,
) -> MaxPerGroup<C, K, F>
where
    K: Eq + Hash,
    F: Fn(&C) -> K + Sync,
{
    MaxPerGroup::new(id, max, key)
}

/// 전체 개수 상한 제약을 만든다. [`MaxTotal`] 의 짧은 이름.
pub fn max_total(id: impl Into<ConstraintId>, max: usize) -> MaxTotal {
    MaxTotal::new(id, max)
}

/// 비용 예산 제약을 만든다. [`CostBudget`] 의 짧은 이름.
pub fn cost_budget<C, F>(id: impl Into<ConstraintId>, limit: f64, cost: F) -> CostBudget<C, F>
where
    F: Fn(&C) -> f64 + Sync,
{
    CostBudget::new(id, limit, cost)
}

/// 술어 단항 제약을 만든다. [`Predicate`] 의 짧은 이름.
pub fn predicate<C, F>(id: impl Into<ConstraintId>, f: F) -> Predicate<C, F>
where
    F: Fn(&C) -> bool + Sync,
{
    Predicate::new(id, f)
}

/// 그룹별 개수를 센다. 제약 준수 불변식 검사와 테스트에 쓴다.
pub fn group_counts<C, K, F>(items: &[&C], key: F) -> HashMap<K, usize>
where
    K: Eq + Hash,
    F: Fn(&C) -> K,
{
    let mut counts = HashMap::new();
    for item in items {
        *counts.entry(key(item)).or_insert(0) += 1;
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct Doc {
        source: &'static str,
        tokens: f64,
        authority: f32,
    }

    fn doc(source: &'static str, tokens: f64, authority: f32) -> Doc {
        Doc {
            source,
            tokens,
            authority,
        }
    }

    #[test]
    fn max_per_group_blocks_the_fourth_from_one_source() {
        let c = max_per_group("max_per_source", 3, |d: &Doc| d.source);
        let a = doc("arxiv", 0.0, 1.0);
        let b = doc("arxiv", 0.0, 1.0);
        let d = doc("arxiv", 0.0, 1.0);
        let e = doc("blog", 0.0, 1.0);

        assert!(c.admits(&[], &a));
        assert!(c.admits(&[&a], &b));
        assert!(c.admits(&[&a, &b], &d));
        assert!(!c.admits(&[&a, &b, &d], &e_same_source()));
        assert!(c.admits(&[&a, &b, &d], &e));
        assert!(c.is_matroid());
    }

    fn e_same_source() -> Doc {
        doc("arxiv", 0.0, 1.0)
    }

    #[test]
    fn zero_max_admits_nothing() {
        let c = max_per_group("none", 0, |d: &Doc| d.source);
        assert!(!c.admits(&[], &doc("arxiv", 0.0, 1.0)));
    }

    #[test]
    fn cost_budget_is_not_a_matroid_and_counts_the_running_sum() {
        let c = cost_budget("tokens", 10.0, |d: &Doc| d.tokens);
        let a = doc("x", 6.0, 1.0);
        let b = doc("x", 3.0, 1.0);
        let d = doc("x", 2.0, 1.0);

        assert!(c.admits(&[&a], &b));
        assert!(!c.admits(&[&a, &b], &d));
        assert!(!c.is_matroid());
    }

    #[test]
    fn predicate_makes_a_unary_constraint() {
        let c = predicate("min_authority", |d: &Doc| d.authority >= 0.3);
        assert!(c.allows(&doc("x", 0.0, 0.5)));
        assert!(!c.allows(&doc("x", 0.0, 0.1)));
        assert_eq!(c.id().as_str(), "min_authority");
    }

    #[test]
    fn requirement_reports_its_predicate_and_count() {
        let r = Requirement::at_least("needs_arxiv", 2, |d: &Doc| d.source == "arxiv");
        assert_eq!(r.needed(), 2);
        assert!(r.satisfied_by(&doc("arxiv", 0.0, 1.0)));
        assert!(!r.satisfied_by(&doc("blog", 0.0, 1.0)));
    }
}
