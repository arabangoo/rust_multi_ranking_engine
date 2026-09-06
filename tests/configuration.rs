//! 설정 오류는 후보를 한 건도 읽기 전에 걸린다.
//!
//! 로짓과 확률을 섞어 더하는 것은 흔한 실수다. 엔진은 그것을 문서로 경고하는 대신
//! 거부한다. 그리고 그 거부는 실행 시점이 아니라 설정 시점의 사건이라, 1,000만 후보를
//! 훑은 뒤에 알게 되는 일이 없어야 한다.

mod common;

use common::{authority, corpus, relevance, Batched, Doc, Expensive, Unbounded, Unit};
use rust_multi_ranking_engine::{
    Budget, Engine, Error, Fusion, MissingPolicy, Normalizer, ScoreScale, ScorerExt, ScorerId,
};

// ── 척도 판정 ─────────────────────────────────────────────────────

/// 무한 척도를 가중합에 넣으면 거부한다.
#[test]
fn an_unbounded_axis_cannot_enter_a_weighted_sum() {
    let err = Engine::new()
        .scorer(Unit("relevance", relevance))
        .scorer(Unbounded("logit"))
        .fuse(Fusion::weighted_sum())
        .budget(Budget::TopK(3))
        .validate()
        .unwrap_err();

    match err {
        Error::IncompatibleScale {
            scorer,
            scale,
            fusion,
        } => {
            assert_eq!(scorer, ScorerId::new("logit"));
            assert_eq!(scale, ScoreScale::Unbounded);
            assert_eq!(fusion, "weighted_sum");
        }
        other => panic!("다른 오류가 났다: {other}"),
    }
}

/// 최댓값 융합도 값을 비교하므로 같은 이유로 거부한다.
#[test]
fn an_unbounded_axis_cannot_enter_a_max_fusion() {
    let err = Engine::new()
        .scorer(Unbounded("logit"))
        .fuse(Fusion::Max)
        .validate()
        .unwrap_err();
    assert!(matches!(err, Error::IncompatibleScale { .. }), "{err}");
}

/// 순위 융합은 순서만 쓰므로 어떤 척도 조합도 받는다.
#[test]
fn rank_fusion_takes_any_mix_of_scales() {
    let out = Engine::new()
        .scorer(Unit("relevance", relevance))
        .scorer(Unbounded("logit"))
        .fuse(Fusion::rrf())
        .budget(Budget::TopK(5))
        .run(corpus(3, 100))
        .unwrap();
    assert_eq!(out.ranked.len(), 5);
}

/// 정규화기를 끼우면 같은 축이 가중합에 들어간다. 엔진이 몰래 정규화해 주지 않고
/// 사용자가 명시적으로 고르게 한다.
#[test]
fn a_normalizer_makes_the_same_axis_admissible() {
    let out = Engine::new()
        .scorer(Unit("relevance", relevance))
        .scorer(Unbounded("logit").normalized(Normalizer::Sigmoid))
        .fuse(Fusion::weighted_sum())
        .budget(Budget::TopK(5))
        .run(corpus(3, 100))
        .unwrap();

    assert_eq!(out.ranked.len(), 5);
    for r in &out.ranked {
        let v = r.scores.get(&ScorerId::new("logit")).flatten().unwrap();
        assert!((0.0..=1.0).contains(&v), "정규화를 거쳤는데 {v}");
    }
}

/// 거부는 후보를 한 건도 읽기 전에 나야 한다. 후보 반복자를 소비했는지로 확인한다.
#[test]
fn nothing_is_read_before_the_configuration_is_rejected() {
    let mut pulled = 0usize;
    let docs = corpus(1, 50);
    let counted = docs.into_iter().inspect(|_| pulled += 1);

    let err = Engine::new()
        .scorer(Unbounded("logit"))
        .fuse(Fusion::weighted_sum())
        .run(counted)
        .unwrap_err();

    assert!(matches!(err, Error::IncompatibleScale { .. }), "{err}");
    assert_eq!(pulled, 0, "후보를 {pulled} 건 읽고 나서 거부했다");
}

// ── 빌더 오류 ─────────────────────────────────────────────────────

#[test]
fn an_engine_without_scorers_is_an_error() {
    let engine: Engine<Doc> = Engine::new();
    assert!(matches!(engine.validate(), Err(Error::NoScorers)));
}

#[test]
fn a_duplicate_scorer_identifier_is_an_error() {
    let err = Engine::new()
        .scorer(Unit("relevance", relevance))
        .scorer(Unit("relevance", authority))
        .fuse(Fusion::rrf())
        .validate()
        .unwrap_err();
    assert_eq!(err, Error::DuplicateScorer(ScorerId::new("relevance")));
}

#[test]
fn a_weight_pointing_at_nothing_is_an_error() {
    let err = Engine::new()
        .scorer(Unit("relevance", relevance))
        .fuse(Fusion::WeightedSum {
            weights: vec![(ScorerId::new("ghost"), 1.0)],
        })
        .validate()
        .unwrap_err();
    assert_eq!(err, Error::UnknownWeight(ScorerId::new("ghost")));
}

/// 1단계는 후보 전부를 훑으므로 승인 채점기가 비싸면 캐스케이드의 뜻이 사라진다.
#[test]
fn an_expensive_admission_scorer_is_an_error() {
    let err = Engine::new()
        .scorer(Unit("relevance", relevance))
        .scorer(Expensive("cross", |_| Some(0.5)))
        .fuse(Fusion::rrf())
        .admission("cross")
        .validate()
        .unwrap_err();
    assert_eq!(err, Error::ExpensiveAdmissionScorer(ScorerId::new("cross")));
}

/// 순위 융합인데 싼 축이 하나도 없으면 1단계가 무엇으로 자를지 정할 수 없다.
#[test]
fn rank_fusion_without_a_cheap_axis_cannot_cut_the_pool() {
    let err = Engine::new()
        .scorer(Expensive("cross", |_| Some(0.5)))
        .fuse(Fusion::rrf())
        .validate()
        .unwrap_err();
    assert_eq!(err, Error::NoAdmissionScorer);
}

#[test]
fn a_token_budget_without_a_cost_function_is_an_error() {
    let err = Engine::new()
        .scorer(Unit("relevance", relevance))
        .fuse(Fusion::rrf())
        .budget(Budget::Tokens { max: 10 })
        .validate()
        .unwrap_err();
    assert!(matches!(err, Error::InvalidBudget(_)), "{err}");
}

#[test]
fn a_zero_pool_multiplier_is_an_error() {
    let err = Engine::new()
        .scorer(Unit("relevance", relevance))
        .fuse(Fusion::rrf())
        .pool_multiplier(0)
        .validate()
        .unwrap_err();
    assert_eq!(err, Error::InvalidPoolMultiplier);
}

// ── 승인 채점기 ───────────────────────────────────────────────────

/// 순위 융합에서 승인 채점기를 지정하지 않으면 첫 싼 축이 쓰이고 그 사실이 기록에 남는다.
#[test]
fn the_admission_scorer_is_recorded_even_when_it_was_chosen_for_you() {
    let out = Engine::new()
        .scorer(Unit("relevance", relevance))
        .scorer(Unit("authority", authority))
        .fuse(Fusion::rrf())
        .budget(Budget::TopK(3))
        .run(corpus(8, 100))
        .unwrap();

    assert_eq!(
        out.trace.admission_scorer,
        Some(ScorerId::new("relevance")),
        "자동으로 골랐어도 무엇을 골랐는지 남아야 한다"
    );
}

/// 값 기반 융합에서는 지정하지 않으면 싼 축들의 융합 값이 절단 기준이 되고,
/// 그때는 승인 채점기가 없다고 기록한다.
#[test]
fn value_fusion_cuts_on_the_streaming_fusion_by_default() {
    let out = Engine::new()
        .scorer(Unit("relevance", relevance))
        .scorer(Unit("authority", authority))
        .fuse(Fusion::weighted_sum())
        .budget(Budget::TopK(3))
        .run(corpus(8, 100))
        .unwrap();

    assert_eq!(out.trace.admission_scorer, None);
}

// ── 결측과 NaN ────────────────────────────────────────────────────

/// 대체 정책은 값을 채우고 그 사실을 기록에 남긴다. 원본 점수는 여전히 비어 있다.
#[test]
fn imputation_fills_the_value_but_keeps_the_original_hole_visible() {
    let out = Engine::new()
        .scorer(Unit("relevance", relevance))
        .scorer(Unit("authority", authority))
        .fuse(Fusion::weighted_sum())
        .missing(MissingPolicy::Impute(0.25))
        .budget(Budget::TopK(60))
        .run(corpus(2, 200))
        .unwrap();

    let filled = out
        .ranked
        .iter()
        .find(|r| r.candidate.relevance.is_none())
        .expect("코퍼스에 결측이 섞여 있다");

    assert_eq!(
        filled.scores.get(&ScorerId::new("relevance")),
        Some(None),
        "원본은 비어 있어야 한다"
    );
    let term = filled
        .fusion
        .terms
        .iter()
        .find(|t| t.scorer == ScorerId::new("relevance"))
        .unwrap();
    assert_eq!(
        term.input,
        rust_multi_ranking_engine::FusionInput::Imputed(0.25)
    );
}

/// `NaN` 은 순서를 매길 수 없으므로 결측과 같이 다룬다. 힙에 들어가면 결정성이 깨진다.
#[test]
fn a_nan_score_is_treated_as_missing() {
    let out = Engine::new()
        .scorer(Unit("broken", |_| Some(f32::NAN)))
        .scorer(Unit("authority", authority))
        .fuse(Fusion::weighted_sum())
        .budget(Budget::TopK(5))
        .run(corpus(6, 40))
        .unwrap();

    for r in &out.ranked {
        assert_eq!(r.scores.get(&ScorerId::new("broken")), Some(None));
        assert!(r.fused.is_finite(), "융합 점수가 {}", r.fused);
    }
    assert_eq!(out.trace.scorers[0].missing, 40);
}

// ── 실행 기록 ─────────────────────────────────────────────────────

/// 비싼 채점기가 몇 번 불렸는지 남는다. 그래야 비용을 되짚을 수 있다.
#[test]
fn the_trace_shows_the_cascade_actually_saved_calls() {
    let out = Engine::new()
        .scorer(Unit("relevance", relevance))
        .scorer(Expensive("cross_encoder", |d| Some(d.authority)))
        .fuse(Fusion::weighted_sum())
        .budget(Budget::TopK(4))
        .pool_multiplier(4)
        .run(corpus(9, 1000))
        .unwrap();

    let cheap = &out.trace.scorers[0];
    let expensive = &out.trace.scorers[1];

    assert_eq!(cheap.calls, 1000, "싼 축은 후보 전부에 돈다");
    assert_eq!(expensive.calls, 16, "비싼 축은 K 곱하기 4 인 풀에만 돈다");
    assert_eq!(out.trace.pool_capacity, 16);
    assert_eq!(out.trace.input_count, 1000);
}

// ── 배치 채점 ─────────────────────────────────────────────────────

/// 비싼 축은 축소된 풀을 배치로 받는다. 병렬 빌드는 풀을 나눠 전달한다.
///
/// 이것이 파이썬 콜백이 쓰는 경로이고, 교차 인코더 같은 배치 추론 모델이 실제로
/// 원하는 모양이다.
#[test]
fn an_expensive_axis_receives_only_the_pool_in_batches() {
    use std::sync::atomic::Ordering;

    let scorer = std::sync::Arc::new(Batched::new("cross_encoder"));
    let handle = std::sync::Arc::clone(&scorer);

    struct Shared(std::sync::Arc<Batched>);
    impl rust_multi_ranking_engine::Scorer<Doc> for Shared {
        fn id(&self) -> ScorerId {
            self.0.id()
        }
        fn scale(&self) -> rust_multi_ranking_engine::ScoreScale {
            self.0.scale()
        }
        fn cost(&self) -> rust_multi_ranking_engine::ScorerCost {
            self.0.cost()
        }
        fn score(&self, d: &Doc) -> Option<f32> {
            self.0.score(d)
        }
        fn score_batch(&self, cs: &[&Doc]) -> Vec<Option<f32>> {
            self.0.score_batch(cs)
        }
    }

    let out = Engine::new()
        .scorer(Unit("relevance", relevance))
        .scorer(Shared(scorer))
        .fuse(Fusion::weighted_sum())
        .budget(Budget::TopK(4))
        .pool_multiplier(4)
        .run(corpus(9, 1000))
        .unwrap();

    // 풀은 16개다. 직렬은 전체 풀, 병렬은 스레드 수에 맞춘 덩어리를 받는다.
    assert_eq!(out.trace.pool_capacity, 16);
    #[cfg(not(feature = "parallel"))]
    let width = 16;
    #[cfg(feature = "parallel")]
    let width = {
        let threads = rayon::current_num_threads().max(1);
        ((16 + threads - 1) / threads).max(1)
    };
    assert_eq!(handle.widest.load(Ordering::SeqCst), width);
    // 직렬 빌드는 한 번, 병렬 빌드는 덩어리 수만큼 부른다. 어느 쪽이든 1,000 번이 아니다.
    let calls = handle.calls.load(Ordering::SeqCst);
    assert_eq!(calls, (16 + width - 1) / width);
    assert_eq!(out.trace.scorers[1].calls, 16, "기록에는 후보 수로 남는다");
}

/// 배치 결과의 길이가 어긋나면 조용히 쓰지 않고 멈춘다.
///
/// 순수 러스트에서는 잘 나지 않지만 파이썬 콜백에서는 실제로 생기는 실패다.
#[test]
fn a_batch_result_of_the_wrong_length_is_an_error() {
    let err = Engine::new()
        .scorer(Unit("relevance", relevance))
        .scorer(Batched::broken("cross_encoder"))
        .fuse(Fusion::weighted_sum())
        .budget(Budget::TopK(2))
        .pool_multiplier(4)
        .run(corpus(9, 100))
        .unwrap_err();

    match err {
        Error::BatchLengthMismatch {
            scorer,
            expected,
            got,
        } => {
            assert_eq!(scorer, ScorerId::new("cross_encoder"));
            assert_eq!(expected, got + 1);
        }
        other => panic!("다른 오류가 났다: {other}"),
    }
}

/// 기본 구현을 쓰는 채점기는 아무것도 바뀌지 않는다.
#[test]
fn a_scorer_without_a_batch_override_behaves_as_before() {
    let out = Engine::new()
        .scorer(Unit("relevance", relevance))
        .scorer(Expensive("cross", |d| Some(d.authority)))
        .fuse(Fusion::weighted_sum())
        .budget(Budget::TopK(3))
        .pool_multiplier(4)
        .run(corpus(9, 500))
        .unwrap();

    assert_eq!(out.trace.scorers[1].calls, 12);
    assert!(out.is_complete());
}
