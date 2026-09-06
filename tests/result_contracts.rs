//! 성공 결과의 요구조건과 보장, 배치 전달 계약을 검사한다.
mod common;

use common::{relevance, Doc, Unit};
use rust_multi_ranking_engine::*;

fn documents() -> Vec<Doc> {
    [(0, "neutral", 0.99), (1, "A", 0.8), (2, "B", 0.7)]
        .into_iter()
        .map(|(id, source, value)| Doc {
            id,
            source,
            relevance: Some(value),
            ..Doc::default()
        })
        .collect()
}

#[test]
fn later_requirements_preserve_earlier_ones() {
    for sources in [["A", "B"], ["B", "A"]] {
        let mut engine = Engine::new()
            .scorer(Unit("score", relevance))
            .fuse(Fusion::weighted_sum())
            .budget(Budget::TopK(2));
        for source in sources {
            engine = engine.require(Requirement::at_least(source, 1, move |d: &Doc| {
                d.source == source
            }));
        }
        let out = engine.run(documents()).unwrap();
        for source in sources {
            assert!(out.ranked.iter().any(|r| r.candidate.source == source));
        }
        assert!(out.is_complete());
        assert!(!out.selection.exact);
        assert_eq!(out.selection.guarantee, None);
    }
}

#[test]
fn conflicting_requirements_return_an_error() {
    let err = Engine::new()
        .scorer(Unit("score", relevance))
        .fuse(Fusion::weighted_sum())
        .budget(Budget::TopK(1))
        .require(Requirement::at_least("A", 1, |d: &Doc| d.source == "A"))
        .require(Requirement::at_least("B", 1, |d: &Doc| d.source == "B"))
        .run(documents())
        .unwrap_err();
    assert!(matches!(err, Error::InfeasibleRequirement { id, .. } if id == "B"));
}

#[test]
fn requirement_repairs_still_respect_cost_and_group_limits() {
    let out = Engine::new()
        .scorer(Unit("score", relevance))
        .fuse(Fusion::weighted_sum())
        .budget(Budget::Tokens { max: 2 })
        .cost(|d: &Doc| d.tokens)
        .set_constraint(constraint::max_per_group("source", 1, |d: &Doc| d.source))
        .require(Requirement::at_least("A", 1, |d: &Doc| d.source == "A"))
        .require(Requirement::at_least("B", 1, |d: &Doc| d.source == "B"))
        .run(documents())
        .unwrap();
    assert_eq!(out.ranked.len(), 2);
    assert!(out.ranked.iter().all(|r| r.candidate.source != "neutral"));
    assert!(out.ranked.iter().map(|r| r.candidate.tokens).sum::<u32>() <= 2);
}

#[test]
fn a_generic_cost_constraint_does_not_claim_a_knapsack_guarantee() {
    let mut docs = vec![Doc {
        id: 0,
        relevance: Some(1.0),
        tokens: 100,
        ..Doc::default()
    }];
    docs.extend((1..=100).map(|id| Doc {
        id,
        relevance: Some(0.9),
        tokens: 1,
        ..Doc::default()
    }));
    for coverage in [false, true] {
        let mut engine = Engine::new()
            .scorer(Unit("score", relevance))
            .fuse(Fusion::weighted_sum())
            .budget(Budget::TopK(101))
            .set_constraint(constraint::cost_budget("cost", 100.0, |d: &Doc| {
                d.tokens as f64
            }));
        if coverage {
            engine = engine.objective(Coverage::new(|d: &Doc| d.topics.clone()));
        }
        let out = engine.run(docs.clone()).unwrap();
        // 비용이 작은 후보 100개의 합계는 90이지만 일반 탐색은 점수 1인 후보를 고른다.
        assert_eq!(out.ranked.iter().map(|r| r.fused).sum::<f32>(), 1.0);
        assert!(!out.selection.exact);
        assert_eq!(out.selection.guarantee, None);
    }
}

struct BatchOnly {
    broken: bool,
}
impl Scorer<Doc> for BatchOnly {
    fn id(&self) -> ScorerId {
        "batch".into()
    }
    fn scale(&self) -> ScoreScale {
        ScoreScale::Unbounded
    }
    fn cost(&self) -> ScorerCost {
        ScorerCost::Expensive
    }
    fn score(&self, _: &Doc) -> Option<f32> {
        panic!("batch scorer must not call score")
    }
    fn score_batch(&self, docs: &[&Doc]) -> Vec<Option<f32>> {
        let mut values: Vec<_> = docs
            .iter()
            .map(|d| if d.id == 1 { None } else { Some(d.id as f32) })
            .collect();
        if self.broken {
            values.pop();
        }
        values
    }
}

#[test]
fn normalized_batch_preserves_order_values_and_missing_entries() {
    let out = Engine::new()
        .scorer(Unit("score", relevance))
        .scorer(BatchOnly { broken: false }.normalized(Normalizer::Sigmoid))
        .fuse(Fusion::weighted_sum())
        .budget(Budget::TopK(3))
        .run(documents())
        .unwrap();
    for row in &out.ranked {
        let expected = if row.candidate.id == 1 {
            None
        } else {
            Some(Normalizer::Sigmoid.apply(row.candidate.id as f32))
        };
        assert_eq!(row.scores.get(&ScorerId::new("batch")), Some(expected));
    }
    assert_eq!(out.trace.scorers[1].missing, 1);
}

#[test]
fn normalized_batch_preserves_length_errors() {
    let err = Engine::new()
        .scorer(Unit("score", relevance))
        .scorer(BatchOnly { broken: true }.normalized(Normalizer::Sigmoid))
        .fuse(Fusion::weighted_sum())
        .run(documents())
        .unwrap_err();
    assert!(matches!(err, Error::BatchLengthMismatch { expected, got, .. } if expected == got + 1));
}

#[cfg(feature = "parallel")]
#[test]
fn opposite_batch_length_errors_cannot_cancel_each_other() {
    struct Uneven;
    impl Scorer<Doc> for Uneven {
        fn id(&self) -> ScorerId {
            "uneven".into()
        }
        fn scale(&self) -> ScoreScale {
            ScoreScale::Unit
        }
        fn cost(&self) -> ScorerCost {
            ScorerCost::Expensive
        }
        fn score(&self, _: &Doc) -> Option<f32> {
            panic!("batch only")
        }
        fn score_batch(&self, docs: &[&Doc]) -> Vec<Option<f32>> {
            if docs[0].id == 0 {
                vec![]
            } else {
                vec![Some(0.5); 2]
            }
        }
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap()
        .install(|| {
            let err = Engine::new()
                .scorer(Unit("score", relevance))
                .scorer(Uneven)
                .fuse(Fusion::weighted_sum())
                .run(documents().into_iter().take(2))
                .unwrap_err();
            assert!(matches!(
                err,
                Error::BatchLengthMismatch {
                    expected: 1,
                    got: 0,
                    ..
                }
            ));
        });
}
