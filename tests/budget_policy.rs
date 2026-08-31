//! 적응형 예산. K 를 상수로 두지 않되, 가정하고 쓰지 않고 재고 나서 쓴다.

mod common;

use common::{Doc, Unit};
use rust_multi_ranking_engine::{budget::tail_mass, Budget, Engine, FallbackReason, Fusion};

/// 순위 `r` 인 항목의 점수가 `r^-s` 인 코퍼스. 진짜 멱법칙이다.
fn zipf_corpus(s: f32, n: u64) -> Vec<Doc> {
    (0..n)
        .map(|id| Doc {
            id,
            relevance: Some(((id + 1) as f32).powf(-s)),
            ..Doc::default()
        })
        .collect()
}

fn engine(epsilon: f32, fallback_k: u32) -> Engine<Doc> {
    Engine::new()
        .scorer(Unit("relevance", |d| d.relevance))
        .fuse(Fusion::weighted_sum())
        .budget(Budget::tail_mass(epsilon, fallback_k))
        .pool_multiplier(64)
}

/// 멱법칙이면 K 를 유도하고 그 근거를 기록에 남긴다.
#[test]
fn a_power_law_input_derives_its_own_k() {
    let out = engine(0.1, 99).run(zipf_corpus(1.2, 400)).unwrap();
    let trace = out.trace.budget.expect("꼬리 질량 예산은 기록을 남긴다");

    assert!(!trace.fallback, "{trace:?}");
    assert!((trace.s - 1.2).abs() < 0.15, "추정 지수 {}", trace.s);
    assert!(trace.fit_quality > 0.9, "적합도 {}", trace.fit_quality);
    assert_eq!(out.ranked.len(), trace.derived_k as usize);
    assert_ne!(trace.derived_k, 99, "고정 K 로 되돌아가면 안 된다");

    // 유도된 K 뒤에 남는 질량이 실제로 허용치 아래인지 다시 잰다.
    let left = tail_mass(
        trace.s,
        trace.derived_k as usize,
        out.selection.pool_size as usize,
    );
    assert!(left <= 0.1 + 1e-6, "남은 질량 {left}");
}

/// 꼬리가 두꺼우면 K 가 커지고 얇으면 작아진다. 고정 K 가 두 경우 모두에서 틀린 값인
/// 이유가 이것이다.
#[test]
fn a_fatter_tail_asks_for_a_bigger_k() {
    let flat = engine(0.1, 1)
        .run(zipf_corpus(0.8, 400))
        .unwrap()
        .trace
        .budget
        .unwrap();
    let steep = engine(0.1, 1)
        .run(zipf_corpus(2.0, 400))
        .unwrap()
        .trace
        .budget
        .unwrap();

    assert!(!flat.fallback && !steep.fallback);
    assert!(
        flat.derived_k > steep.derived_k,
        "완만한 쪽 {} vs 가파른 쪽 {}",
        flat.derived_k,
        steep.derived_k
    );
}

/// 허용 누락을 조이면 더 많이 고른다.
#[test]
fn a_tighter_epsilon_selects_more() {
    let loose = engine(0.3, 1).run(zipf_corpus(1.1, 400)).unwrap();
    let tight = engine(0.02, 1).run(zipf_corpus(1.1, 400)).unwrap();
    assert!(
        tight.ranked.len() > loose.ranked.len(),
        "느슨 {} vs 조임 {}",
        loose.ranked.len(),
        tight.ranked.len()
    );
}

/// 멱법칙이 아니면 유도된 K 를 버리고 고정 K 로 되돌린다.
///
/// **가정하고 쓰는 것이 아니라 재고 나서 쓴다.** 멱법칙이 아닌 분포에 멱법칙을 맞추면
/// 자신 있게 틀린 K 가 나온다.
#[test]
fn a_non_power_law_input_falls_back_and_says_so() {
    // 균등 분포. 로그 축에서 직선이 아니다.
    let flat: Vec<Doc> = (0..300)
        .map(|id| Doc {
            id,
            relevance: Some(0.5),
            ..Doc::default()
        })
        .collect();

    let out = engine(0.1, 7).min_fit(0.999).run(flat).unwrap();
    let trace = out.trace.budget.unwrap();

    assert!(trace.fallback);
    assert_eq!(trace.reason, Some(FallbackReason::PoorFit));
    assert_eq!(trace.derived_k, 7);
    assert_eq!(out.ranked.len(), 7);
}

/// 표본이 열 개도 안 되면 적합 자체가 뜻이 없다. 그때도 사유가 남는다.
#[test]
fn too_small_a_pool_falls_back_with_its_own_reason() {
    let out = engine(0.1, 3).run(zipf_corpus(1.2, 6)).unwrap();
    let trace = out.trace.budget.unwrap();
    assert_eq!(trace.reason, Some(FallbackReason::TooFewSamples));
    assert_eq!(out.ranked.len(), 3);
}

/// 고정 K 예산에는 예산 기록이 없다. 없는 근거를 지어내지 않는다.
#[test]
fn a_fixed_k_leaves_no_budget_trace() {
    let out = Engine::new()
        .scorer(Unit("relevance", |d| d.relevance))
        .fuse(Fusion::weighted_sum())
        .budget(Budget::TopK(4))
        .run(zipf_corpus(1.2, 50))
        .unwrap();
    assert!(out.trace.budget.is_none());
}

/// 배낭형 예산은 비용 대비 이득으로 고르고 상한을 지킨다.
#[test]
fn the_knapsack_budget_prefers_value_per_cost() {
    // 값은 같은데 비용이 다른 후보들. 싼 것부터 담아야 더 많이 담긴다.
    let docs: Vec<Doc> = (0..20)
        .map(|id| Doc {
            id,
            relevance: Some(0.5),
            tokens: if id < 10 { 1 } else { 9 },
            ..Doc::default()
        })
        .collect();

    let out = Engine::new()
        .scorer(Unit("relevance", |d| d.relevance))
        .fuse(Fusion::weighted_sum())
        .budget(Budget::Tokens { max: 10 })
        .cost(|d: &Doc| d.tokens)
        .pool_multiplier(4)
        .run(docs)
        .unwrap();

    let spent: u32 = out.ranked.iter().map(|r| r.candidate.tokens).sum();
    assert!(spent <= 10, "{spent} 토큰을 썼다");
    assert_eq!(out.ranked.len(), 10, "값이 같으면 싼 것을 열 개 담는다");
    assert!(out.ranked.iter().all(|r| r.candidate.tokens == 1));
    assert!(!out.selection.exact);
    assert_eq!(
        out.selection.guarantee,
        Some(rust_multi_ranking_engine::GUARANTEE_KNAPSACK_MODULAR)
    );
}

/// 비싼 하나가 싼 여럿보다 나으면 그쪽을 고른다. 비율 탐욕만으로는 놓치는 경우다.
#[test]
fn one_expensive_item_can_beat_the_ratio_greedy_set() {
    let mut docs: Vec<Doc> = (0..9)
        .map(|id| Doc {
            id,
            relevance: Some(0.10),
            tokens: 1,
            ..Doc::default()
        })
        .collect();
    // 비용 10 을 다 쓰고 값 5.0 을 내는 하나. 싼 것 아홉을 다 모아도 0.9 다.
    docs.push(Doc {
        id: 99,
        relevance: Some(5.0),
        tokens: 10,
        ..Doc::default()
    });

    let out = Engine::new()
        .scorer(Unit("relevance", |d| d.relevance))
        .fuse(Fusion::Max)
        .budget(Budget::Tokens { max: 10 })
        .cost(|d: &Doc| d.tokens)
        .pool_multiplier(8)
        .run(docs)
        .unwrap();

    let ids: Vec<u64> = out.ranked.iter().map(|r| r.candidate.id).collect();
    assert_eq!(ids, vec![99], "단일 최고 항목이 이겨야 한다");
    assert!(out.is_complete());
}
