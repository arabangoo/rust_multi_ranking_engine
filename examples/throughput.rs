//! 규모별 처리량 측정.
//!
//! ```text
//! cargo run --release --example throughput
//! ```
//!
//! # 왜 benches 가 아니라 examples 인가
//!
//! 설계서 15절은 `benches/` 를 적었다. 러스트의 표준 벤치 하네스(`#[bench]`)는 아직
//! 나이틀리 전용이고, 안정 채널에서 쓰려면 criterion 같은 개발 의존성을 들여야 한다.
//! 이 레포는 **개발 의존성도 0** 으로 두기로 했으므로(형제 레포 `rust_pii_transformer`
//! 와 같은 규약) 측정을 예제로 옮겼다. 통계적 신뢰구간이 필요해지면 그때 criterion 을
//! 개발 의존성으로 들이고 `benches/` 로 옮긴다.
//!
//! 여기서 재는 것은 셋이다.
//!
//! 1. 후보 수를 열 배씩 늘려도 메모리가 풀 크기에 묶여 있는가
//! 2. 비싼 채점기 호출이 후보 수가 아니라 풀 크기에 비례하는가
//! 3. 1단계 처리량이 후보당 얼마인가

use std::time::Instant;

use rust_multi_ranking_engine::{
    constraint, Budget, Candidate, CandidateId, Engine, Fusion, Rejections, ScoreScale, Scorer,
    ScorerCost, ScorerId,
};

struct Doc {
    id: u64,
    source: u32,
    relevance: f32,
}

impl Candidate for Doc {
    fn id(&self) -> CandidateId {
        CandidateId::num(self.id)
    }
}

struct Relevance;

impl Scorer<Doc> for Relevance {
    fn id(&self) -> ScorerId {
        ScorerId::new("relevance")
    }
    fn scale(&self) -> ScoreScale {
        ScoreScale::Unit
    }
    fn cost(&self) -> ScorerCost {
        ScorerCost::Cheap
    }
    fn score(&self, d: &Doc) -> Option<f32> {
        Some(d.relevance)
    }
}

/// 후보 전부에 돌면 비싼 축이다. 실제로는 여기서 교차 인코더를 부른다.
struct CrossEncoder;

impl Scorer<Doc> for CrossEncoder {
    fn id(&self) -> ScorerId {
        ScorerId::new("cross_encoder")
    }
    fn scale(&self) -> ScoreScale {
        ScoreScale::Unit
    }
    fn cost(&self) -> ScorerCost {
        ScorerCost::Expensive
    }
    fn score(&self, d: &Doc) -> Option<f32> {
        // 일부러 조금 무겁게. 캐스케이드가 무엇을 아끼는지 보이려는 것이다.
        let mut acc = 0.0f32;
        for i in 1..64u32 {
            acc += (d.relevance * i as f32).sin();
        }
        Some((acc / 64.0).abs().min(1.0))
    }
}

/// 고정 시드 선형 합동 생성기. 같은 규모면 언제나 같은 입력이다.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }
}

/// 반복자로 흘려보낸다. 후보 전부를 벡터에 담지 않는 것이 이 엔진의 요구다.
fn stream(n: u64) -> impl Iterator<Item = Doc> {
    let mut rng = Lcg(0xA5A5);
    (0..n).map(move |id| Doc {
        id,
        source: rng.next() % 8,
        relevance: (rng.next() % 1_000_000) as f32 / 1_000_000.0,
    })
}

fn main() {
    const K: u32 = 30;
    const MULTIPLIER: u32 = 32;

    println!(
        "{:>12} {:>10} {:>12} {:>14} {:>12}",
        "후보", "풀", "비싼 호출", "경과(ms)", "후보당(ns)"
    );
    println!("{}", "-".repeat(64));

    for n in [10_000u64, 100_000, 1_000_000] {
        let started = Instant::now();
        let out = Engine::new()
            .scorer(Relevance)
            .scorer(CrossEncoder)
            .fuse(Fusion::weighted_sum())
            .set_constraint(constraint::max_per_group("max_per_source", 5, |d: &Doc| {
                d.source
            }))
            .budget(Budget::TopK(K))
            .pool_multiplier(MULTIPLIER)
            // 1,000만 후보의 탈락 상세를 전부 보관하면 스트리밍의 뜻이 사라진다.
            // 개수는 이 정책과 무관하게 정확하다.
            .rejections(Rejections::Count)
            .run(stream(n))
            .expect("설정이 성립한다");

        let elapsed = started.elapsed();
        println!(
            "{:>12} {:>10} {:>12} {:>14.1} {:>12.0}",
            n,
            out.selection.pool_size,
            out.trace.scorers[1].calls,
            elapsed.as_secs_f64() * 1000.0,
            elapsed.as_nanos() as f64 / n as f64
        );

        assert!(out.is_complete(), "완전성은 규모와 무관하다");
        assert_eq!(out.ranked.len(), K as usize);
        assert_eq!(
            out.trace.scorers[1].calls,
            (K * MULTIPLIER) as u64,
            "비싼 축은 후보 수가 아니라 풀 크기에 비례해야 한다"
        );
    }

    println!();
    println!("비싼 채점기 호출 수가 세 줄 모두 같다면 캐스케이드가 실제로 돌고 있는 것이다.");
    println!("풀 크기가 K 곱하기 배수에 묶여 있으면 메모리도 후보 수와 무관하다.");
}
