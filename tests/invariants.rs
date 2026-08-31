//! 불변식 넷. 정확성을 문장이 아니라 검사로 못박는다.
//!
//! | 불변식 | 내용 | 검증 방법 |
//! | --- | --- | --- |
//! | 결정성 | 같은 입력이면 스레드 수와 무관하게 순위와 동점 처리까지 동일 | 지문 대조 |
//! | 완전성 | 모든 후보가 결과 또는 탈락 목록에 정확히 한 번 나타난다 | 개수 합 대조 |
//! | 제약 준수 | 반환된 집합이 선언된 집합 제약을 전부 만족한다 | 결과를 제약에 다시 통과 |
//! | 근거 재현 | 원본 점수와 융합 기록만으로 최종 점수를 다시 계산할 수 있다 | 재계산 후 비교 |

mod common;

use std::collections::{HashMap, HashSet};

use common::{authority, corpus, fingerprint, relevance, Doc, Expensive, Lcg, Unit};
use rust_multi_ranking_engine::{
    constraint, Budget, Coverage, Engine, Fusion, MissingPolicy, Outcome, Rejections,
};

/// 고정된 지문. 값을 바꾸려면 왜 바뀌어도 되는지를 먼저 설명할 수 있어야 한다.
const FINGERPRINT_7_300_8: &str = "35:0.043785;157:0.043445;50:0.042621;38:0.041100;271:0.040770;177:0.040680;241:0.040250;180:0.039774;|0/14/14/0/42/222";

/// 시험 대상 설정. 모든 불변식 검사가 이 하나를 공유한다.
fn engine(k: u32) -> Engine<Doc> {
    Engine::new()
        .scorer(Unit("relevance", relevance))
        .scorer(Unit("authority", authority))
        .scorer(Expensive("cross_encoder", |d| {
            Some((d.authority + d.tokens as f32 / 20.0).min(1.0))
        }))
        .fuse(Fusion::rrf())
        .admission("relevance")
        .unary(constraint::predicate("min_authority", |d: &Doc| {
            d.authority >= 0.05
        }))
        .set_constraint(constraint::max_per_group("max_per_source", 3, |d: &Doc| {
            d.source
        }))
        .budget(Budget::TopK(k))
        .pool_multiplier(8)
}

fn run(seed: u64, n: u64, k: u32) -> Outcome<Doc> {
    engine(k).run(corpus(seed, n)).expect("설정은 성립한다")
}

// ── 불변식 1: 결정성 ──────────────────────────────────────────────

/// 같은 입력을 여러 번 돌려도 지문이 같아야 한다.
///
/// `parallel` 기능을 켠 빌드에서도 이 파일 전체가 그대로 통과해야 한다. 융합은 후보
/// 식별자 순 고정 순서라 병렬 채점이 값을 흔들지 못하기 때문이다.
/// `cargo test --features parallel` 로 확인한다.
#[test]
fn the_same_input_gives_the_same_output_every_time() {
    let a = fingerprint(&run(11, 400, 12));
    for _ in 0..5 {
        assert_eq!(a, fingerprint(&run(11, 400, 12)));
    }
}

/// 도착 순서가 결과를 바꾸지 못한다. 힙의 동점 규칙이 식별자를 딛기 때문이다.
///
/// 풀 배수를 크게 잡아 1단계 절단이 개입하지 않게 한다. 절단이 걸리면 도착 순서가
/// 아니라 절단선 위아래가 결과를 가르는데, 그것은 결정성이 아니라 근사의 문제다.
#[test]
fn arrival_order_does_not_change_the_result() {
    let docs = corpus(23, 200);
    let forward = engine(10).pool_multiplier(64).run(docs.clone()).unwrap();

    let mut shuffled = docs;
    let mut rng = Lcg::new(99);
    for i in (1..shuffled.len()).rev() {
        shuffled.swap(i, rng.below(i as u32 + 1) as usize);
    }
    let scrambled = engine(10).pool_multiplier(64).run(shuffled).unwrap();

    let a: Vec<u64> = forward.ranked.iter().map(|r| r.candidate.id).collect();
    let b: Vec<u64> = scrambled.ranked.iter().map(|r| r.candidate.id).collect();
    assert_eq!(a, b);
}

/// 지문을 고정값으로 못박는다. 리팩터링이 조용히 순위를 바꾸면 여기서 걸린다.
#[test]
fn the_fingerprint_is_pinned() {
    let out = run(7, 300, 8);
    // 이 리터럴이 결정성 불변식의 실물이다. 기본 빌드와 `--features parallel` 빌드,
    // 그리고 RAYON_NUM_THREADS 를 1 부터 16 까지 바꾼 실행이 전부 이 값을 내야 한다.
    assert_eq!(
        fingerprint(&out),
        FINGERPRINT_7_300_8,
        "지문이 달라졌다. 순위나 동점 처리가 바뀐 것이다"
    );
    let ids: Vec<u64> = out.ranked.iter().map(|r| r.candidate.id).collect();
    assert_eq!(
        ids.len(),
        8,
        "여덟 개를 채워야 한다. 실제 지문 = {}",
        fingerprint(&out)
    );
    // 출처마다 셋까지이므로 여덟 개가 네 출처에 흩어져야 한다.
    let mut per_source: HashMap<&str, usize> = HashMap::new();
    for r in &out.ranked {
        *per_source.entry(r.candidate.source).or_default() += 1;
    }
    assert!(per_source.values().all(|c| *c <= 3), "{per_source:?}");
}

// ── 불변식 2: 완전성 ──────────────────────────────────────────────

/// 결과 수와 탈락 수의 합이 입력 수와 같다.
#[test]
fn every_candidate_lands_in_exactly_one_place() {
    for (seed, n, k) in [(1u64, 50u64, 5u32), (2, 500, 20), (3, 1000, 3)] {
        let out = run(seed, n, k);
        assert!(
            out.is_complete(),
            "seed {seed}: {} + {} != {}",
            out.ranked.len(),
            out.rejected_counts.total(),
            out.trace.input_count
        );
        assert_eq!(out.trace.input_count, n);
    }
}

/// 보관 정책이 상세를 줄여도 개수는 정확하다. 그것이 이 정책의 존재 이유다.
#[test]
fn counts_stay_exact_when_details_are_dropped() {
    let full = run(5, 400, 10);

    let counted = engine(10)
        .rejections(Rejections::Count)
        .run(corpus(5, 400))
        .unwrap();
    assert!(counted.rejected.is_empty());
    assert!(counted.is_complete());
    assert_eq!(counted.rejected_counts, full.rejected_counts);

    let sampled = engine(10)
        .rejections(Rejections::Sample(7))
        .run(corpus(5, 400))
        .unwrap();
    assert_eq!(sampled.rejected.len(), 7);
    assert!(sampled.is_complete());
    assert_eq!(sampled.rejected_counts, full.rejected_counts);
}

/// 같은 후보가 결과와 탈락 목록에 동시에 나타나지 않는다.
#[test]
fn no_candidate_appears_twice() {
    let out = run(31, 300, 9);
    let mut seen: HashSet<u64> = HashSet::new();
    for r in &out.ranked {
        assert!(
            seen.insert(r.candidate.id),
            "결과에 중복 {}",
            r.candidate.id
        );
    }
    for r in &out.rejected {
        assert!(
            seen.insert(r.candidate.id),
            "탈락에 중복 {}",
            r.candidate.id
        );
    }
    assert_eq!(seen.len(), out.trace.input_count as usize);
}

// ── 불변식 3: 제약 준수 ───────────────────────────────────────────

/// 반환된 집합을 제약에 다시 통과시킨다.
#[test]
fn the_returned_set_still_satisfies_every_set_constraint() {
    for seed in 0..8u64 {
        let out = run(seed, 400, 15);
        let picked: Vec<&Doc> = out.ranked.iter().map(|r| &r.candidate).collect();

        let counts = constraint::group_counts(&picked, |d: &Doc| d.source);
        for (source, n) in &counts {
            assert!(*n <= 3, "seed {seed}: 출처 {source} 가 {n} 개");
        }
        for r in &out.ranked {
            assert!(r.candidate.authority >= 0.05, "단항 제약이 새어 나갔다");
        }
    }
}

/// 비용 예산은 상한을 넘지 않는다.
#[test]
fn the_token_budget_is_never_exceeded() {
    for seed in 0..6u64 {
        let out = Engine::new()
            .scorer(Unit("relevance", relevance))
            .fuse(Fusion::weighted_sum())
            .budget(Budget::Tokens { max: 40 })
            .cost(|d: &Doc| d.tokens)
            .pool_multiplier(2)
            .run(corpus(seed, 200))
            .unwrap();

        let spent: u32 = out.ranked.iter().map(|r| r.candidate.tokens).sum();
        assert!(spent <= 40, "seed {seed}: {spent} 토큰을 썼다");
        assert!(out.is_complete());
    }
}

/// 하한 요구 조건이 실제로 채워진다. 채우려고 교체했으면 최적성 선언이 내려간다.
#[test]
fn a_lower_bound_requirement_is_filled_by_swapping() {
    let mut docs = corpus(17, 60);
    // 점수가 낮은 자리에만 news 를 심어 둔다. 그냥 고르면 안 들어온다.
    for d in docs.iter_mut() {
        d.source = if d.id >= 55 { "news" } else { "arxiv" };
        d.relevance = Some(1.0 - d.id as f32 / 100.0);
        d.authority = 0.9;
    }

    let plain = Engine::new()
        .scorer(Unit("relevance", relevance))
        .fuse(Fusion::weighted_sum())
        .budget(Budget::TopK(5))
        .run(docs.clone())
        .unwrap();
    assert!(plain.ranked.iter().all(|r| r.candidate.source == "arxiv"));
    assert!(plain.selection.exact);

    let required = Engine::new()
        .scorer(Unit("relevance", relevance))
        .fuse(Fusion::weighted_sum())
        .budget(Budget::TopK(5))
        .require(constraint::Requirement::at_least(
            "needs_news",
            2,
            |d: &Doc| d.source == "news",
        ))
        .run(docs)
        .unwrap();

    let news = required
        .ranked
        .iter()
        .filter(|r| r.candidate.source == "news")
        .count();
    assert_eq!(news, 2);
    assert_eq!(required.ranked.len(), 5);
    assert!(
        !required.selection.exact,
        "교체가 일어났으면 최적이라고 말하면 안 된다"
    );
    assert!(required.is_complete());
}

/// 요구 조건을 채울 후보가 아예 없으면 조용히 부족한 답을 주지 않고 오류를 낸다.
#[test]
fn an_unfillable_requirement_is_an_error_not_a_short_answer() {
    let docs = corpus(4, 40);
    let err = Engine::new()
        .scorer(Unit("relevance", relevance))
        .fuse(Fusion::weighted_sum())
        .budget(Budget::TopK(5))
        .require(constraint::Requirement::at_least(
            "impossible",
            3,
            |d: &Doc| d.source == "nowhere",
        ))
        .run(docs)
        .unwrap_err();

    assert!(
        matches!(
            err,
            rust_multi_ranking_engine::Error::InfeasibleRequirement {
                needed: 3,
                available: 0,
                ..
            }
        ),
        "{err}"
    );
}

// ── 불변식 4: 근거 재현 ───────────────────────────────────────────

/// 융합 기록만으로 최종 점수를 다시 계산할 수 있다.
#[test]
fn the_trace_alone_reproduces_the_fused_score() {
    for fusion in [Fusion::rrf(), Fusion::weighted_sum(), Fusion::Max] {
        let out = Engine::new()
            .scorer(Unit("relevance", relevance))
            .scorer(Unit("authority", authority))
            .fuse(fusion.clone())
            .admission("authority")
            .budget(Budget::TopK(20))
            .run(corpus(13, 250))
            .unwrap();

        for r in &out.ranked {
            let again = r.fusion.recompute();
            assert!(
                (again - r.fused).abs() < 1e-6,
                "{:?}: {} 를 재계산하니 {again}",
                fusion,
                r.fused
            );
        }
    }
}

/// 값이 없는 축은 사유와 정책까지 감사 출력에 남는다.
#[test]
fn a_missing_axis_is_visible_in_the_audit_output() {
    let out = Engine::new()
        .scorer(Unit("relevance", relevance))
        .scorer(Unit("authority", authority))
        .fuse(Fusion::weighted_sum())
        .missing(MissingPolicy::Skip)
        .budget(Budget::TopK(50))
        .run(corpus(2, 200))
        .unwrap();

    let with_hole = out
        .ranked
        .iter()
        .find(|r| r.candidate.relevance.is_none())
        .expect("코퍼스에 결측이 섞여 있다");

    let json = with_hole.to_json(&rust_multi_ranking_engine::CandidateId::num(
        with_hole.candidate.id,
    ));
    assert!(json.contains("\"relevance\":null"), "{json}");
    assert!(json.contains("\"missing_policy\":\"skip\""), "{json}");
    assert!(json.contains("\"kind\":\"skipped\""), "{json}");

    // 건너뛴 축은 가중치가 0 이고 남은 축의 가중치가 1 로 재정규화된다.
    let sum: f32 = with_hole.fusion.terms.iter().map(|t| t.weight).sum();
    assert!((sum - 1.0).abs() < 1e-6, "가중치 합이 {sum}");
}

/// 필수 축 정책이면 결측 후보가 사유와 함께 떨어진다. 0 점 처리와 다르다.
#[test]
fn a_required_axis_rejects_instead_of_scoring_zero() {
    let out = Engine::new()
        .scorer(Unit("relevance", relevance))
        .fuse(Fusion::weighted_sum())
        .missing(MissingPolicy::Reject)
        .budget(Budget::TopK(10))
        .run(corpus(2, 200))
        .unwrap();

    assert!(out.rejected_counts.not_scored > 0);
    assert!(out.ranked.iter().all(|r| r.candidate.relevance.is_some()));
    assert!(out.is_complete());
}

// ── 무작위 왕복 ───────────────────────────────────────────────────

/// 고정 시드 200 라운드. 설정을 바꿔 가며 네 불변식을 한꺼번에 건다.
#[test]
fn two_hundred_random_rounds_hold_all_four_invariants() {
    let mut rng = Lcg::new(0xC0FFEE);

    for round in 0..200u32 {
        let n = 20 + rng.below(400) as u64;
        let k = 1 + rng.below(15);
        let multiplier = 1 + rng.below(6);
        let per_source = 1 + rng.below(4) as usize;
        let use_objective = rng.below(2) == 0;
        let fusion = match rng.below(3) {
            0 => Fusion::rrf(),
            1 => Fusion::weighted_sum(),
            _ => Fusion::Max,
        };

        let mut engine = Engine::new()
            .scorer(Unit("relevance", relevance))
            .scorer(Unit("authority", authority))
            .fuse(fusion)
            .admission("authority")
            .set_constraint(constraint::max_per_group(
                "max_per_source",
                per_source,
                |d: &Doc| d.source,
            ))
            .budget(Budget::TopK(k))
            .pool_multiplier(multiplier);

        if use_objective {
            engine = engine.objective(Coverage::new(|d: &Doc| d.topics.clone()));
        }

        let docs = corpus(round as u64, n);
        let out = engine.run(docs).unwrap();

        // 완전성.
        assert!(out.is_complete(), "round {round}");
        // 제약 준수.
        let picked: Vec<&Doc> = out.ranked.iter().map(|r| &r.candidate).collect();
        for (source, count) in constraint::group_counts(&picked, |d: &Doc| d.source) {
            assert!(count <= per_source, "round {round}: {source} 가 {count} 개");
        }
        // 근거 재현.
        for r in &out.ranked {
            assert!(
                (r.fusion.recompute() - r.fused).abs() < 1e-6,
                "round {round}"
            );
        }
        // 개수 상한.
        assert!(out.ranked.len() <= k as usize, "round {round}");
        // 절단선 여유는 음수가 될 수 없다. 탐욕이 마지막 자리에 고른 것은 그 시점에
        // 다툰 것들 중 가장 나은 것이었기 때문이다. 목적함수가 걸려도 마찬가지다 --
        // 잣대가 선택기가 쓴 기준(총 이득)이라 같은 저울에서 비교된다.
        if let Some(margin) = out.selection.cut_margin {
            assert!(margin >= -1e-6, "round {round}: 절단선 여유 {margin}");
        }
    }
}
