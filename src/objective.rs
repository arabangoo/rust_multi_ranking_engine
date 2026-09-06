//! 집합 목적함수와 한계 이득.
//!
//! 후보별 독립 점수의 합이 아닌 목적함수를 지원한다. 검색 결과를 고를 때 실제 목표는
//! 보통 이런 모양이다.
//!
//! ```text
//! 목표:  관련성(S) + 포괄성(S) + 연결성(S) - 중복성(S)  을 최대화
//! ```
//!
//! 관련성과 연결성은 후보별로 계산되지만 포괄성과 중복성은 집합 전체의 함수다.
//! 세 번째 문서의 가치는 앞의 둘이 무엇이었느냐에 달려 있다.
//!
//! # 설계서에서 달라진 점: 목적함수는 대체가 아니라 덧셈이다
//!
//! 설계서는 목적함수를 주지 않으면 기본값이 모듈러(융합 점수의 합)라고 적었다.
//! 그런데 [`SetObjective::marginal_gain`] 은 `selected` 와 `candidate` 만 받고 융합
//! 점수를 못 본다. 목적함수가 모듈러 항을 **대체**하면 위 식의 관련성 항을 쓸 방법이
//! 사라진다.
//!
//! 그래서 엔진은 이렇게 합친다.
//!
//! ```text
//! 총 이득(S, c) = 융합점수(c) + marginal_gain(S, c)
//! ```
//!
//! 목적함수를 주지 않으면 `marginal_gain` 이 항상 0 인 것과 같아 총 이득이 융합 점수가
//! 되고, 설계서가 말한 "기본값은 모듈러"가 그대로 성립한다. 목적함수는 모듈러 항 위에
//! 얹는 보정이 된다.
//!
//! 보장 계수도 이 합성 위에서 성립한다. 모듈러 함수와 서브모듈러 함수의 합은
//! 서브모듈러이고, 융합 점수가 음이 아니면 단조성도 유지된다.

use std::collections::HashSet;
use std::hash::Hash;

/// 원소별 가중치 함수. [`Coverage::weighted`] 가 받는다.
type WeightFn<K> = Box<dyn Fn(&K) -> f32 + Sync>;

/// 집합 전체를 보는 목적함수.
///
/// # 계약
///
/// `marginal_gain(S, c)` 는 `S` 에 `c` 를 더했을 때 늘어나는 값이다. 같은 `(S, c)` 에
/// 대해 항상 같은 값을 돌려줘야 한다. `S` 의 원소 순서가 달라도 값이 같아야 한다 --
/// 순서에 의존하면 결정성이 탐욕의 진행 순서에 묶인다.
///
/// `is_submodular` 를 참으로 선언하는 것은 [`SetConstraint::is_matroid`](crate::SetConstraint::is_matroid)
/// 와 같은 성격의 약속이다. 엔진이 증명하지 않고 선언받는다. 잘못 선언하면 결과에
/// 실리는 보장 계수가 근거를 잃는다.
pub trait SetObjective<C>: Sync {
    /// 현재 선택에 이 후보를 더했을 때의 이득.
    fn marginal_gain(&self, selected: &[&C], candidate: &C) -> f32;

    /// 서브모듈러인가. 참이면 탐욕에 보장 계수가 붙는다.
    ///
    /// 개수 제한만 있으면 `1 - 1/e`(약 0.632), 매트로이드 하나가 걸리면 `1/2`,
    /// 추가 집합 제약 없이 `Budget::Tokens`를 쓰면 `(1 - 1/e)/2`(약 0.316)다. 자세한 표는
    /// [`Selection::guarantee`](crate::Selection::guarantee) 문서에 있다.
    fn is_submodular(&self) -> bool;
}

/// 포괄성 목적함수. 후보가 덮는 원소들의 합집합 크기를 최대화한다.
///
/// 이미 덮인 원소를 다시 덮어도 이득이 없으므로 한계 이득이 단조 감소한다. 그것이
/// 서브모듈러의 정의이고, 이 함수는 그 성질이 증명된 형태(최대 피복, maximum coverage)라
/// [`is_submodular`](SetObjective::is_submodular) 가 참이어도 근거가 있다.
///
/// 원소별 가중치를 주지 않으면 전부 1.0 이다.
pub struct Coverage<C, K, F> {
    cover: F,
    weight: Option<WeightFn<K>>,
    _marker: std::marker::PhantomData<fn(&C) -> K>,
}

impl<C, K, F> Coverage<C, K, F>
where
    K: Eq + Hash,
    F: Fn(&C) -> Vec<K> + Sync,
{
    /// 후보가 덮는 원소들을 뽑는 함수로 목적함수를 만든다.
    pub fn new(cover: F) -> Self {
        Coverage {
            cover,
            weight: None,
            _marker: std::marker::PhantomData,
        }
    }

    /// 원소별 가중치를 준다.
    pub fn weighted(mut self, weight: impl Fn(&K) -> f32 + Sync + 'static) -> Self {
        self.weight = Some(Box::new(weight));
        self
    }

    fn weight_of(&self, k: &K) -> f32 {
        match &self.weight {
            Some(f) => f(k),
            None => 1.0,
        }
    }
}

impl<C, K, F> SetObjective<C> for Coverage<C, K, F>
where
    C: Sync,
    K: Eq + Hash + Sync,
    F: Fn(&C) -> Vec<K> + Sync,
{
    fn marginal_gain(&self, selected: &[&C], candidate: &C) -> f32 {
        let mut covered: HashSet<K> = HashSet::new();
        for s in selected {
            covered.extend((self.cover)(s));
        }

        let mut gain = 0.0f32;
        let mut fresh: HashSet<K> = HashSet::new();
        for k in (self.cover)(candidate) {
            if !covered.contains(&k) && !fresh.contains(&k) {
                gain += self.weight_of(&k);
                fresh.insert(k);
            }
        }
        gain
    }

    fn is_submodular(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Doc(Vec<u32>);

    fn objective() -> Coverage<Doc, u32, impl Fn(&Doc) -> Vec<u32> + Sync> {
        Coverage::new(|d: &Doc| d.0.clone())
    }

    #[test]
    fn coverage_gain_shrinks_as_the_set_grows() {
        let obj = objective();
        let a = Doc(vec![1, 2, 3]);
        let b = Doc(vec![3, 4]);

        assert_eq!(obj.marginal_gain(&[], &a), 3.0);
        assert_eq!(obj.marginal_gain(&[&a], &b), 1.0);
        assert_eq!(obj.marginal_gain(&[&a, &b], &Doc(vec![1, 4])), 0.0);
    }

    #[test]
    fn duplicate_elements_inside_one_candidate_count_once() {
        let obj = objective();
        assert_eq!(obj.marginal_gain(&[], &Doc(vec![7, 7, 7])), 1.0);
    }

    #[test]
    fn weights_scale_the_gain() {
        let obj = Coverage::new(|d: &Doc| d.0.clone()).weighted(|k: &u32| *k as f32);
        assert_eq!(obj.marginal_gain(&[], &Doc(vec![2, 5])), 7.0);
    }

    #[test]
    fn gain_does_not_depend_on_the_order_of_the_selected_set() {
        let obj = objective();
        let a = Doc(vec![1, 2]);
        let b = Doc(vec![2, 3]);
        let c = Doc(vec![3, 4]);
        assert_eq!(
            obj.marginal_gain(&[&a, &b], &c),
            obj.marginal_gain(&[&b, &a], &c)
        );
    }
}
