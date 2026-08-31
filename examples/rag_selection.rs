//! 검색 증강 생성 파이프라인에서 문서를 고르는 예제.
//!
//! ```text
//! cargo run --example rag_selection
//! ```
//!
//! 이 예제가 보이려는 것은 넷이다.
//!
//! 1. 척도가 다른 축 셋을 섞는다. 교차 인코더는 로짓이라 정규화기를 끼워야 들어간다
//! 2. 출처 편중을 집합 제약으로 막는다. 상위 K 가 정답이 아니게 되는 자리다
//! 3. 토큰 예산 안에서 고른다. 배낭형이라 근사이고 보장 계수를 함께 낸다
//! 4. 왜 이 문서가 들어갔고 저 문서가 떨어졌는지를 감사 출력으로 남긴다

use rust_multi_ranking_engine::{
    constraint, Budget, Candidate, CandidateId, Coverage, Engine, Fusion, Normalizer, ScoreScale,
    Scorer, ScorerCost, ScorerExt, ScorerId,
};

struct Chunk {
    id: &'static str,
    source: &'static str,
    /// 벡터 검색이 낸 코사인 유사도. 0 에서 1 사이다.
    similarity: f32,
    /// 교차 인코더가 낸 로짓. 경계가 없어 그대로 더하면 다른 축을 압도한다.
    cross_logit: f32,
    /// 이 조각이 다루는 주제들. 포괄성 목적함수가 쓴다.
    topics: &'static [&'static str],
    tokens: u32,
}

impl Candidate for Chunk {
    fn id(&self) -> CandidateId {
        CandidateId::text(self.id)
    }
}

struct Similarity;

impl Scorer<Chunk> for Similarity {
    fn id(&self) -> ScorerId {
        ScorerId::new("similarity")
    }
    fn scale(&self) -> ScoreScale {
        ScoreScale::Unit
    }
    fn cost(&self) -> ScorerCost {
        ScorerCost::Cheap
    }
    fn score(&self, c: &Chunk) -> Option<f32> {
        Some(c.similarity)
    }
}

struct CrossEncoder;

impl Scorer<Chunk> for CrossEncoder {
    fn id(&self) -> ScorerId {
        ScorerId::new("cross_encoder")
    }
    fn scale(&self) -> ScoreScale {
        ScoreScale::Unbounded
    }
    /// 실제 파이프라인이라면 여기서 모델을 부른다. 엔진은 그 안을 모르고, 다만
    /// 비싸다는 선언을 보고 2단계로 미룬다.
    fn cost(&self) -> ScorerCost {
        ScorerCost::Expensive
    }
    fn score(&self, c: &Chunk) -> Option<f32> {
        Some(c.cross_logit)
    }
}

fn corpus() -> Vec<Chunk> {
    vec![
        Chunk {
            id: "manual-01",
            source: "manual",
            similarity: 0.94,
            cross_logit: 3.1,
            topics: &["setup", "auth"],
            tokens: 220,
        },
        Chunk {
            id: "manual-02",
            source: "manual",
            similarity: 0.91,
            cross_logit: 2.8,
            topics: &["setup"],
            tokens: 180,
        },
        Chunk {
            id: "manual-03",
            source: "manual",
            similarity: 0.89,
            cross_logit: 2.6,
            topics: &["setup", "auth"],
            tokens: 260,
        },
        Chunk {
            id: "manual-04",
            source: "manual",
            similarity: 0.86,
            cross_logit: 2.2,
            topics: &["auth"],
            tokens: 200,
        },
        Chunk {
            id: "ticket-01",
            source: "tickets",
            similarity: 0.72,
            cross_logit: 1.4,
            topics: &["billing"],
            tokens: 140,
        },
        Chunk {
            id: "ticket-02",
            source: "tickets",
            similarity: 0.68,
            cross_logit: 1.9,
            topics: &["billing", "quota"],
            tokens: 160,
        },
        Chunk {
            id: "ticket-03",
            source: "tickets",
            similarity: 0.61,
            cross_logit: 0.7,
            topics: &["quota"],
            tokens: 120,
        },
        Chunk {
            id: "wiki-01",
            source: "wiki",
            similarity: 0.58,
            cross_logit: 2.4,
            topics: &["deploy"],
            tokens: 300,
        },
        Chunk {
            id: "wiki-02",
            source: "wiki",
            similarity: 0.45,
            cross_logit: -0.6,
            topics: &["deploy", "quota"],
            tokens: 280,
        },
        Chunk {
            id: "blog-01",
            source: "blog",
            similarity: 0.40,
            cross_logit: 0.2,
            topics: &["misc"],
            tokens: 100,
        },
    ]
}

fn main() {
    let out = Engine::new()
        .scorer(Similarity)
        // 로짓 축은 정규화기를 명시적으로 끼워야 가중합에 들어간다. 이 한 줄을 빼면
        // Error::IncompatibleScale 로 거부된다 -- 후보를 한 건도 읽기 전에.
        .scorer(CrossEncoder.normalized(Normalizer::Sigmoid))
        .fuse(Fusion::WeightedSum {
            weights: vec![
                (ScorerId::new("similarity"), 0.4),
                (ScorerId::new("cross_encoder"), 0.6),
            ],
        })
        // 같은 출처가 답을 독식하지 못하게 막는다.
        .set_constraint(constraint::max_per_group(
            "max_per_source",
            2,
            |c: &Chunk| c.source,
        ))
        // 주제를 넓게 덮을수록 좋다. 이미 덮은 주제를 다시 덮으면 이득이 줄어든다.
        .objective(Coverage::new(|c: &Chunk| c.topics.to_vec()))
        // 프롬프트에 넣을 수 있는 만큼만 고른다.
        .budget(Budget::Tokens { max: 900 })
        .cost(|c: &Chunk| c.tokens)
        .pool_multiplier(1)
        .run(corpus())
        .expect("설정이 성립한다");

    println!("== 고른 것 ==");
    let mut spent = 0;
    for r in &out.ranked {
        spent += r.candidate.tokens;
        println!(
            "  {:>2}. {:<10} {:<8} 융합 {:.4}  토큰 {:>3}  주제 {:?}",
            r.rank,
            r.candidate.id,
            r.candidate.source,
            r.fused,
            r.candidate.tokens,
            r.candidate.topics
        );
    }
    println!("  토큰 {spent} / 900");

    println!();
    println!("== 떨어진 것 ==");
    for r in &out.rejected {
        println!(
            "  {:<10} {:<18} 융합 {}",
            r.candidate.id,
            r.reason.kind(),
            r.fused.map_or("-".to_string(), |v| format!("{v:.4}"))
        );
    }

    println!();
    println!("== 선택의 성질 ==");
    println!("  풀 크기          {}", out.selection.pool_size);
    println!("  풀 소진          {}", out.selection.pool_exhausted);
    println!("  최적 선언        {}", out.selection.exact);
    match out.selection.guarantee {
        Some(g) => println!("  보장 계수        {g:.4}"),
        None => println!(
            "  보장 계수        없음  (매트로이드와 배낭형이 섞였다. 표에 없는 조합에는
                              계수를 주지 않는다 -- 근거 없는 숫자를 싣지 않기 위해서다)"
        ),
    }
    match out.selection.cut_margin {
        Some(m) => println!("  절단선 여유      {m:.4}"),
        None => println!("  절단선 여유      없음 (마지막 자리를 다툰 후보가 없다)"),
    }

    println!();
    println!("== 실행 기록 ==");
    println!("  입력 {} 건", out.trace.input_count);
    for s in &out.trace.scorers {
        println!(
            "  {:<14} 호출 {:>3}회  결측 {}회",
            s.scorer.as_str(),
            s.calls,
            s.missing
        );
    }

    println!();
    println!("== 1위의 감사 출력 ==");
    let top = &out.ranked[0];
    println!("{}", top.to_json(&top.candidate.id()));

    // 완전성 불변식. 조용히 사라진 후보가 없다.
    assert!(out.is_complete());
    println!();
    println!(
        "완전성 확인: 결과 {} + 탈락 {} = 입력 {}",
        out.ranked.len(),
        out.rejected_counts.total(),
        out.trace.input_count
    );
}
