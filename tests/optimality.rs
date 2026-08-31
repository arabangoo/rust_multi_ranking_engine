//! 최적성과 보장 계수. 최적해를 아는 소규모 사례와 직접 비교한다.
//!
//! 보장 계수를 결과에 싣는 것은 약속이다. 약속을 지키는지 재는 유일한 방법은 최적해를
//! 실제로 구해 보는 것이고, 그러려면 사례가 작아야 한다. 여기서는 후보 12 개에서 4 개를
//! 고르는 495 가지를 전수 조사한다.

mod common;

use std::collections::HashSet;

use common::{relevance, Doc, Unit};
use rust_multi_ranking_engine::{
    constraint, Budget, Coverage, Engine, Fusion, Outcome, GUARANTEE_CARDINALITY,
    GUARANTEE_KNAPSACK_MODULAR, GUARANTEE_KNAPSACK_SUBMODULAR, GUARANTEE_MATROID,
};

/// 결정적인 소규모 코퍼스. 무작위가 아니라 손으로 짠 것이라 실패가 언제나 같다.
fn small() -> Vec<Doc> {
    let rows: [(u64, &'static str, f32, &[u32]); 12] = [
        (0, "arxiv", 0.95, &[1, 2]),
        (1, "arxiv", 0.92, &[1, 2]),
        (2, "arxiv", 0.90, &[1, 2]),
        (3, "arxiv", 0.88, &[1]),
        (4, "blog", 0.70, &[3, 4]),
        (5, "blog", 0.65, &[4, 5]),
        (6, "blog", 0.60, &[5]),
        (7, "news", 0.55, &[6, 7]),
        (8, "news", 0.50, &[7, 8]),
        (9, "web", 0.45, &[9]),
        (10, "web", 0.40, &[9, 10]),
        (11, "web", 0.35, &[11]),
    ];
    rows.iter()
        .map(|(id, source, rel, topics)| Doc {
            id: *id,
            source,
            relevance: Some(*rel),
            authority: 1.0,
            topics: topics.to_vec(),
            ..Doc::default()
        })
        .collect()
}

/// 목적함수 값. 융합 점수의 합에 덮은 주제의 수를 더한다.
fn value(docs: &[Doc], set: &[usize], with_coverage: bool) -> f32 {
    let relevance: f32 = set.iter().map(|i| docs[*i].relevance.unwrap()).sum();
    if !with_coverage {
        return relevance;
    }
    let topics: HashSet<u32> = set.iter().flat_map(|i| docs[*i].topics.clone()).collect();
    relevance + topics.len() as f32
}

/// 크기 `k` 인 모든 부분집합 중 제약을 지키면서 값이 가장 큰 것.
///
/// 재귀 대신 사전식 조합 생성기를 쓴다. 인자를 아홉 개 넘기는 재귀 함수보다 읽기 쉽고,
/// 조합 순서가 고정이라 실패가 언제나 같은 자리에서 난다.
fn brute_force(docs: &[Doc], k: usize, per_source: Option<usize>, with_coverage: bool) -> f32 {
    let n = docs.len();
    let mut combo: Vec<usize> = (0..k).collect();
    let mut best = f32::NEG_INFINITY;

    loop {
        let allowed = match per_source {
            None => true,
            Some(limit) => {
                let picked: Vec<&Doc> = combo.iter().map(|i| &docs[*i]).collect();
                constraint::group_counts(&picked, |d: &Doc| d.source)
                    .values()
                    .all(|c| *c <= limit)
            }
        };
        if allowed {
            best = best.max(value(docs, &combo, with_coverage));
        }

        // 다음 조합. 뒤에서부터 올릴 수 있는 자리를 찾는다.
        let mut pos = k;
        while pos > 0 {
            pos -= 1;
            if combo[pos] != pos + n - k {
                combo[pos] += 1;
                for later in pos + 1..k {
                    combo[later] = combo[later - 1] + 1;
                }
                break;
            }
            if pos == 0 {
                return best;
            }
        }
        if k == 0 {
            return best;
        }
    }
}

fn picked_indices(out: &Outcome<Doc>) -> Vec<usize> {
    out.ranked.iter().map(|r| r.candidate.id as usize).collect()
}

// ── 모듈러 ────────────────────────────────────────────────────────

/// 제약이 없으면 상위 K 가 최적이다. 엔진도 그렇게 말해야 한다.
#[test]
fn top_k_is_optimal_when_nothing_constrains_it() {
    let docs = small();
    let out = Engine::new()
        .scorer(Unit("relevance", relevance))
        .fuse(Fusion::weighted_sum())
        .budget(Budget::TopK(4))
        .run(docs.clone())
        .unwrap();

    assert!(out.selection.exact);
    assert_eq!(out.selection.guarantee, None);

    let got = value(&docs, &picked_indices(&out), false);
    let best = brute_force(&docs, 4, None, false);
    assert!((got - best).abs() < 1e-5, "탐욕 {got}, 최적 {best}");
}

/// 분할 매트로이드 하나가 걸려도 모듈러면 탐욕이 정확히 최적이다.
///
/// 이것이 상위 K 가 정답이 아니게 되는 자리다. 점수 1·2·3위가 전부 arxiv 라 셋 중
/// 둘만 들어가고 blog 가 밀고 들어온다.
#[test]
fn greedy_is_exactly_optimal_under_a_partition_matroid() {
    let docs = small();
    let out = Engine::new()
        .scorer(Unit("relevance", relevance))
        .fuse(Fusion::weighted_sum())
        .set_constraint(constraint::max_per_group("max_per_source", 2, |d: &Doc| {
            d.source
        }))
        .budget(Budget::TopK(4))
        .run(docs.clone())
        .unwrap();

    assert!(out.selection.exact);
    let ids = picked_indices(&out);
    assert_eq!(ids, vec![0, 1, 4, 5], "실제 선택 {ids:?}");

    let got = value(&docs, &ids, false);
    let best = brute_force(&docs, 4, Some(2), false);
    assert!((got - best).abs() < 1e-5, "탐욕 {got}, 최적 {best}");
}

// ── 서브모듈러 ────────────────────────────────────────────────────

/// 서브모듈러 + 개수 제한. 탐욕이 `1 - 1/e` 를 지키는지 최적해와 직접 비교한다.
#[test]
fn a_submodular_objective_keeps_its_promised_ratio() {
    let docs = small();
    let out = Engine::new()
        .scorer(Unit("relevance", relevance))
        .fuse(Fusion::weighted_sum())
        .objective(Coverage::new(|d: &Doc| d.topics.clone()))
        .budget(Budget::TopK(4))
        .run(docs.clone())
        .unwrap();

    assert!(!out.selection.exact);
    assert_eq!(out.selection.guarantee, Some(GUARANTEE_CARDINALITY));

    let got = value(&docs, &picked_indices(&out), true);
    let best = brute_force(&docs, 4, None, true);
    assert!(
        got >= best * GUARANTEE_CARDINALITY,
        "탐욕 {got} 가 최적 {best} 의 {GUARANTEE_CARDINALITY} 배에 못 미친다"
    );
    // 이 사례에서는 실제로 최적과 붙는다. 보장은 하한이지 예측이 아니다.
    assert!((got - best).abs() < 1e-5, "탐욕 {got}, 최적 {best}");
}

/// 서브모듈러 + 매트로이드 하나면 계수가 1/2 로 내려간다. Nemhauser 의 `1 - 1/e` 는
/// 개수 제한에서만 성립하고 일반 매트로이드에서는 성립하지 않는다.
#[test]
fn a_matroid_lowers_the_submodular_guarantee_to_one_half() {
    let docs = small();
    let out = Engine::new()
        .scorer(Unit("relevance", relevance))
        .fuse(Fusion::weighted_sum())
        .objective(Coverage::new(|d: &Doc| d.topics.clone()))
        .set_constraint(constraint::max_per_group("max_per_source", 2, |d: &Doc| {
            d.source
        }))
        .budget(Budget::TopK(4))
        .run(docs.clone())
        .unwrap();

    assert_eq!(out.selection.guarantee, Some(GUARANTEE_MATROID));

    let got = value(&docs, &picked_indices(&out), true);
    let best = brute_force(&docs, 4, Some(2), true);
    assert!(got >= best * GUARANTEE_MATROID, "탐욕 {got}, 최적 {best}");
}

/// 서브모듈러가 아니라고 선언된 목적함수에는 계수를 주지 않는다.
#[test]
fn an_objective_that_declines_submodularity_gets_no_coefficient() {
    struct Anything;
    impl rust_multi_ranking_engine::SetObjective<Doc> for Anything {
        fn marginal_gain(&self, selected: &[&Doc], _c: &Doc) -> f32 {
            // 이미 고른 것이 많을수록 이득이 커진다. 서브모듈러의 반대다.
            selected.len() as f32 * 0.1
        }
        fn is_submodular(&self) -> bool {
            false
        }
    }

    let out = Engine::new()
        .scorer(Unit("relevance", relevance))
        .fuse(Fusion::weighted_sum())
        .objective(Anything)
        .budget(Budget::TopK(3))
        .run(small())
        .unwrap();

    assert!(!out.selection.exact);
    assert_eq!(out.selection.guarantee, None);
}

// ── 절단선과 풀 ───────────────────────────────────────────────────

/// 절단선 여유가 작으면 그 선택이 흔들린다는 뜻이다. 값이 실제로 그렇게 나오는지 본다.
#[test]
fn the_cut_margin_reports_a_knife_edge() {
    let mut docs = small();
    // 4위와 5위를 사실상 같게 만든다.
    docs[3].relevance = Some(0.880_01);
    docs[4].relevance = Some(0.880_00);

    let out = Engine::new()
        .scorer(Unit("relevance", relevance))
        .fuse(Fusion::weighted_sum())
        .budget(Budget::TopK(4))
        .run(docs)
        .unwrap();

    let margin = out.selection.cut_margin.expect("남은 후보가 있다");
    assert!((0.0..1e-3).contains(&margin), "절단선 여유 {margin}");
}

/// 풀이 모자라 K 를 못 채우면 조용히 넘어가지 않고 신호를 낸다.
#[test]
fn an_exhausted_pool_says_so() {
    let out = Engine::new()
        .scorer(Unit("relevance", relevance))
        .fuse(Fusion::weighted_sum())
        .budget(Budget::TopK(30))
        .run(small())
        .unwrap();

    assert_eq!(out.ranked.len(), 12);
    assert!(out.selection.pool_exhausted);
    assert_eq!(out.selection.cut_margin, None, "남은 후보가 없다");
}

/// 풀 배수가 작으면 1단계가 후보를 자르고 그 사실이 탈락 사유로 남는다.
#[test]
fn a_tight_pool_multiplier_shows_up_as_out_of_pool_rejections() {
    let out = Engine::new()
        .scorer(Unit("relevance", relevance))
        .fuse(Fusion::weighted_sum())
        .budget(Budget::TopK(2))
        .pool_multiplier(1)
        .run(small())
        .unwrap();

    assert_eq!(out.selection.pool_size, 2);
    assert_eq!(out.rejected_counts.out_of_pool, 10);
    assert!(out.is_complete());
}

// ── 보장 계수의 출처 ──────────────────────────────────────────────

/// 상수 넷을 출처가 말하는 값에 못박는다.
///
/// 이 테스트가 있는 이유는 실제 사고 때문이다. 서브모듈러 배낭 계수가 한때
/// `1 - e^(-1/2)`(약 0.393)로 박혀 있었는데 출처를 댈 수 없는 값이었고, 실제 보장인
/// `(1 - 1/e)/2`(약 0.316)보다 **높았다.** 보장을 실제보다 후하게 보고하는 것은
/// 근거 없는 숫자를 결과에 싣지 않는다는 이 엔진의 원칙을 정면으로 어기는 일이다.
///
/// 값을 바꾸려면 어느 정리가 그 값을 주는지 먼저 댈 수 있어야 한다.
#[test]
fn the_guarantee_constants_match_their_sources() {
    // Nemhauser, Wolsey, Fisher (1978). 서브모듈러 + 개수 제한.
    assert!(
        (GUARANTEE_CARDINALITY - (1.0 - 1.0 / std::f32::consts::E)).abs() < 1e-6,
        "1 - 1/e 가 아니다: {GUARANTEE_CARDINALITY}"
    );

    // Fisher, Nemhauser, Wolsey (1978, 두 번째 논문). 서브모듈러 + 매트로이드 하나.
    assert_eq!(GUARANTEE_MATROID, 0.5);

    // ModifiedGreedy. 모듈러 + 배낭형.
    assert_eq!(GUARANTEE_KNAPSACK_MODULAR, 0.5);

    // Leskovec et al. (2007). 서브모듈러 + 배낭형, 열거 없는 단순 탐욕.
    assert!(
        (GUARANTEE_KNAPSACK_SUBMODULAR - (1.0 - 1.0 / std::f32::consts::E) / 2.0).abs() < 1e-6,
        "(1 - 1/e)/2 가 아니다: {GUARANTEE_KNAPSACK_SUBMODULAR}"
    );
}

/// 열거 변종의 `1 - 1/e` 를 배낭형 계수로 쓰면 안 된다. 이 엔진은 그 알고리즘을 돌리지
/// 않는다. 컴파일 시점에 막으므로 이 관계가 깨지면 빌드가 안 된다.
const _: () = assert!(GUARANTEE_KNAPSACK_SUBMODULAR < GUARANTEE_CARDINALITY);

/// 예산 안에서 값이 가장 큰 부분집합. 크기를 고정하지 않는다.
fn brute_force_knapsack(docs: &[Doc], limit: u32, with_coverage: bool) -> f32 {
    let n = docs.len();
    let mut best = 0.0f32;
    for mask in 0u32..(1 << n) {
        let set: Vec<usize> = (0..n).filter(|i| mask & (1 << i) != 0).collect();
        let cost: u32 = set.iter().map(|i| docs[*i].tokens).sum();
        if cost > limit {
            continue;
        }
        best = best.max(value(docs, &set, with_coverage));
    }
    best
}

/// 서브모듈러 + 배낭형에서 약속한 계수를 실제로 지키는지 최적해와 직접 비교한다.
#[test]
fn the_knapsack_guarantee_holds_against_the_true_optimum() {
    let mut docs = small();
    // 비용을 흩뜨려 비율 탐욕과 단위비용 탐욕이 서로 다른 답을 내게 만든다.
    for (i, d) in docs.iter_mut().enumerate() {
        d.tokens = 1 + (i as u32 * 7) % 9;
    }

    let out = Engine::new()
        .scorer(Unit("relevance", relevance))
        .fuse(Fusion::weighted_sum())
        .objective(Coverage::new(|d: &Doc| d.topics.clone()))
        .budget(Budget::Tokens { max: 20 })
        .cost(|d: &Doc| d.tokens)
        .pool_multiplier(4)
        .run(docs.clone())
        .unwrap();

    assert_eq!(out.selection.guarantee, Some(GUARANTEE_KNAPSACK_SUBMODULAR));

    let ids: Vec<usize> = out.ranked.iter().map(|r| r.candidate.id as usize).collect();
    let spent: u32 = out.ranked.iter().map(|r| r.candidate.tokens).sum();
    assert!(spent <= 20, "{spent} 토큰을 썼다");

    let got = value(&docs, &ids, true);
    let best = brute_force_knapsack(&docs, 20, true);
    assert!(
        got >= best * GUARANTEE_KNAPSACK_SUBMODULAR,
        "탐욕 {got} 가 최적 {best} 의 {GUARANTEE_KNAPSACK_SUBMODULAR} 배에 못 미친다"
    );
}

/// 모듈러 + 배낭형도 같은 방식으로 잰다. 이쪽은 계수가 1/2 다.
#[test]
fn the_modular_knapsack_guarantee_holds_too() {
    let mut docs = small();
    for (i, d) in docs.iter_mut().enumerate() {
        d.tokens = 1 + (i as u32 * 5) % 11;
    }

    let out = Engine::new()
        .scorer(Unit("relevance", relevance))
        .fuse(Fusion::weighted_sum())
        .budget(Budget::Tokens { max: 18 })
        .cost(|d: &Doc| d.tokens)
        .pool_multiplier(4)
        .run(docs.clone())
        .unwrap();

    assert_eq!(out.selection.guarantee, Some(GUARANTEE_KNAPSACK_MODULAR));

    let ids: Vec<usize> = out.ranked.iter().map(|r| r.candidate.id as usize).collect();
    let got = value(&docs, &ids, false);
    let best = brute_force_knapsack(&docs, 18, false);
    assert!(
        got >= best * GUARANTEE_KNAPSACK_MODULAR,
        "탐욕 {got}, 최적 {best}"
    );
}

/// 단위비용 탐욕 갈래가 실제로 일을 한다.
///
/// 비율 탐욕은 값이 작고 싼 것을 먼저 담아 예산을 소진하고, 값이 큰 것을 못 넣는 경우가
/// 있다. 그 갈래를 더한 이유가 이것이다.
#[test]
fn the_unit_cost_branch_can_win() {
    // 비율은 낮지만 값이 큰 하나와, 비율은 높지만 값이 작은 여럿.
    let mut docs: Vec<Doc> = (0..6)
        .map(|id| Doc {
            id,
            relevance: Some(0.30),
            tokens: 2,
            ..Doc::default()
        })
        .collect();
    // 식별자를 색인과 맞춰 둔다. value 와 brute_force 가 색인으로 읽기 때문이다.
    docs.push(Doc {
        id: 6,
        relevance: Some(1.60),
        tokens: 10,
        ..Doc::default()
    });

    let out = Engine::new()
        .scorer(Unit("relevance", relevance))
        .fuse(Fusion::weighted_sum())
        .budget(Budget::Tokens { max: 10 })
        .cost(|d: &Doc| d.tokens)
        .pool_multiplier(4)
        .run(docs.clone())
        .unwrap();

    let ids: Vec<usize> = out.ranked.iter().map(|r| r.candidate.id as usize).collect();
    let got = value(&docs, &ids, false);
    let best = brute_force_knapsack(&docs, 10, false);
    assert!((got - best).abs() < 1e-5, "탐욕 {got}, 최적 {best}");
}
