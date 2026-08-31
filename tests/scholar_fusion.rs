//! 첫 검증 관문: 자매 프로젝트의 순위 융합을 이 엔진이 흡수하는가.
//!
//! `rust_scholar_transformer` 의 `fusion.rs` 에 있는 `fuse()` 가 실제로 도는 코드이고
//! 테스트가 붙어 있어 결과가 같은지 바로 검증된다. **못 하면 추상화가 틀린 것이다.**
//! 50 줄짜리 기존 함수를 깨끗이 흡수하지 못하는 범용 엔진은 더 큰 곳에서도 안 된다.
//!
//! # 관문의 범위는 설계서보다 좁다
//!
//! 실제 `fuse()` 는 순위 융합만 하지 않는다. 세 가지를 함께 한다.
//!
//! 1. 소스별 순위에서 `1 / (k + 순위)` 를 더한다  <- **이 엔진이 흡수하는 부분**
//! 2. 같은 문서를 정체성 키로 병합한다(중복 제거)
//! 3. 신선도와 출처 신뢰도로 2차 정렬한다
//!
//! 2 번은 이 엔진의 일이 아니다. 중복 제거는 후보를 만드는 단계의 일이고 엔진은 후보가
//! 무엇인지 모른다. 3 번도 마찬가지로 도메인 점수라 채점기 축으로 들어올 뿐이다.
//! 그래서 이 관문은 "완전 대체"가 아니라 **"순위 융합 층을 대체했을 때 융합 점수와
//! 그 순서가 같은가"** 로 판정한다. 그 경계를 흐리면 통과했다는 말이 뜻을 잃는다.

use rust_multi_ranking_engine::{
    Budget, Candidate, CandidateId, Engine, Fusion, ScoreScale, Scorer, ScorerCost, ScorerId,
    DEFAULT_RRF_K,
};

/// 소스 셋에서 온 문서 하나. 각 소스에서 몇 등이었는지를 들고 있다.
#[derive(Clone, Debug)]
struct Doc {
    id: u64,
    /// 소스별 0 부터의 순위. 그 소스에 없으면 `None`.
    ranks: [Option<usize>; 3],
}

impl Candidate for Doc {
    fn id(&self) -> CandidateId {
        CandidateId::num(self.id)
    }
}

/// 소스 하나를 축으로 본 채점기.
///
/// 순위가 낮을수록(앞설수록) 점수가 높아야 엔진이 매기는 순위가 원래 순위와 같아진다.
struct Source(usize);

impl Scorer<Doc> for Source {
    fn id(&self) -> ScorerId {
        ScorerId::new(format!("source_{}", self.0))
    }
    fn scale(&self) -> ScoreScale {
        // 순서만 뜻이 있다. 값끼리 더할 수 없다고 선언한다.
        ScoreScale::Rank
    }
    fn cost(&self) -> ScorerCost {
        ScorerCost::Cheap
    }
    fn score(&self, d: &Doc) -> Option<f32> {
        d.ranks[self.0].map(|r| 1.0 / (r as f32 + 1.0))
    }
}

/// 원본 `fuse()` 의 순위 융합 핵심을 그대로 옮긴 오라클.
///
/// 원문(`rust_scholar_transformer/src/fusion.rs`)은 이렇게 적혀 있다.
///
/// ```text
/// for list in per_source {
///     for (rank, mut doc) in list.into_iter().enumerate() {
///         let contrib = 1.0 / (k + rank as f64 + 1.0);
///         ...
///     }
/// }
/// ```
fn oracle(docs: &[Doc], k: f64) -> Vec<(u64, f64)> {
    let mut out: Vec<(u64, f64)> = docs
        .iter()
        .map(|d| {
            let fused: f64 = d
                .ranks
                .iter()
                .filter_map(|r| r.map(|rank| 1.0 / (k + rank as f64 + 1.0)))
                .sum();
            (d.id, fused)
        })
        .collect();
    // 원본은 융합 점수 내림차순이 1순위다. 2차 신호(신선도·발행일)는 이 관문의 밖이다.
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then_with(|| a.0.cmp(&b.0)));
    out
}

/// 소스 셋의 결과 목록에서 문서 묶음을 만든다. 겹치는 문서가 여러 소스에 나타난다.
fn corpus() -> Vec<Doc> {
    // 소스별 결과 목록. 안쪽 순서가 곧 그 소스에서의 순위다.
    let per_source: [&[u64]; 3] = [
        &[10, 11, 12, 13, 14, 15],
        &[12, 10, 20, 21, 22],
        &[30, 11, 12, 31],
    ];

    let mut ids: Vec<u64> = per_source.iter().flat_map(|s| s.iter().copied()).collect();
    ids.sort_unstable();
    ids.dedup();

    ids.into_iter()
        .map(|id| {
            let mut ranks = [None; 3];
            for (axis, list) in per_source.iter().enumerate() {
                ranks[axis] = list.iter().position(|x| *x == id);
            }
            Doc { id, ranks }
        })
        .collect()
}

fn engine(docs: Vec<Doc>, k: f32) -> Engine<Doc> {
    let _ = &docs;
    Engine::new()
        .scorer(Source(0))
        .scorer(Source(1))
        .scorer(Source(2))
        .fuse(Fusion::Rrf { k })
        // 어느 소스에도 못 든 문서가 1단계에서 잘리지 않도록 풀을 넉넉히 잡는다.
        .pool_multiplier(64)
}

/// 융합 점수가 원본과 같다.
#[test]
fn the_engine_reproduces_the_original_fused_scores() {
    let docs = corpus();
    let expected = oracle(&docs, DEFAULT_RRF_K as f64);

    let out = engine(docs.clone(), DEFAULT_RRF_K)
        .budget(Budget::TopK(docs.len() as u32))
        .run(docs.clone())
        .unwrap();

    assert_eq!(out.ranked.len(), docs.len());
    for r in &out.ranked {
        let (_, want) = expected
            .iter()
            .find(|(id, _)| *id == r.candidate.id)
            .expect("모든 문서가 오라클에 있다");
        assert!(
            (r.fused as f64 - want).abs() < 1e-7,
            "문서 {}: 엔진 {} vs 원본 {want}",
            r.candidate.id,
            r.fused
        );
    }
}

/// 순서도 원본과 같다.
#[test]
fn the_engine_reproduces_the_original_order() {
    let docs = corpus();
    let expected: Vec<u64> = oracle(&docs, DEFAULT_RRF_K as f64)
        .into_iter()
        .map(|(id, _)| id)
        .collect();

    let out = engine(docs.clone(), DEFAULT_RRF_K)
        .budget(Budget::TopK(docs.len() as u32))
        .run(docs)
        .unwrap();

    let got: Vec<u64> = out.ranked.iter().map(|r| r.candidate.id).collect();
    assert_eq!(got, expected);
}

/// 여러 소스에 걸친 문서가 위로 올라간다. 원본 테스트가 확인하던 성질 그대로다.
#[test]
fn a_document_seen_by_several_sources_rises() {
    let docs = corpus();
    let out = engine(docs.clone(), DEFAULT_RRF_K)
        .budget(Budget::TopK(3))
        .run(docs)
        .unwrap();

    // 12 는 세 소스 전부에 있다. 어느 소스에서도 1등이 아닌데 1위가 된다.
    assert_eq!(out.ranked[0].candidate.id, 12);
    assert_eq!(out.ranked[0].fusion.used_axes(), 3);
    assert!(out.ranked[0].fused > out.ranked[1].fused);
}

/// `k` 를 바꾸면 원본과 같은 방향으로 함께 움직인다.
#[test]
fn the_k_constant_moves_both_the_same_way() {
    for k in [1.0f32, 10.0, 60.0, 300.0] {
        let docs = corpus();
        let expected: Vec<u64> = oracle(&docs, k as f64)
            .into_iter()
            .map(|(id, _)| id)
            .collect();

        let out = engine(docs.clone(), k)
            .budget(Budget::TopK(docs.len() as u32))
            .run(docs)
            .unwrap();

        let got: Vec<u64> = out.ranked.iter().map(|r| r.candidate.id).collect();
        assert_eq!(got, expected, "k = {k}");
    }
}

/// 흡수하면 원본에 없던 것이 따라온다. 각 문서가 어느 소스에서 몇 등이었고 그것이
/// 최종 점수에 얼마를 넣었는지가 기록으로 남는다. 원본 `fuse()` 는 이 정보를 버린다.
#[test]
fn absorbing_it_adds_evidence_the_original_threw_away() {
    let docs = corpus();
    let out = engine(docs.clone(), DEFAULT_RRF_K)
        .budget(Budget::TopK(docs.len() as u32))
        .run(docs)
        .unwrap();

    let top = &out.ranked[0];
    assert!((top.fusion.recompute() - top.fused).abs() < 1e-7);

    let json = top.to_json(&CandidateId::num(top.candidate.id));
    assert!(json.contains("\"kind\":\"rank\""), "{json}");
    assert!(json.contains("\"method\":\"rrf\""), "{json}");
    assert!(json.contains("\"k\":60.0"), "{json}");

    // 12 는 소스 0 에서 3등, 소스 1 에서 1등, 소스 2 에서 3등이었다.
    let ranks: Vec<u32> = top
        .fusion
        .terms
        .iter()
        .filter_map(|t| match t.input {
            rust_multi_ranking_engine::FusionInput::Rank(r) => Some(r),
            _ => None,
        })
        .collect();
    assert_eq!(ranks, vec![3, 1, 3]);
}

/// 어느 소스에도 없는 문서는 점수가 0 이고, 그 사실이 축마다 기록된다.
#[test]
fn a_document_in_no_source_scores_zero_and_says_why() {
    let mut docs = corpus();
    docs.push(Doc {
        id: 999,
        ranks: [None, None, None],
    });

    let out = engine(docs.clone(), DEFAULT_RRF_K)
        .budget(Budget::TopK(docs.len() as u32))
        .run(docs)
        .unwrap();

    let orphan = out
        .ranked
        .iter()
        .find(|r| r.candidate.id == 999)
        .expect("풀이 넉넉하므로 결과에 들어온다");
    assert_eq!(orphan.fused, 0.0);
    assert_eq!(orphan.fusion.used_axes(), 0);
    assert_eq!(orphan.rank, out.ranked.len() as u32, "맨 끝이어야 한다");
}
