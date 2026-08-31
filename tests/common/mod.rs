//! 불변식 테스트가 함께 쓰는 도구.
//!
//! 속성 테스트 크레이트를 쓰지 않고 고정 시드 선형 합동 생성기를 직접 넣는다. 기본
//! 빌드의 의존성 0 원칙을 개발 의존성에도 적용한 것이고, 시드가 고정이라 실패가 항상
//! 재현된다. 형제 레포 `rust_pii_transformer` 와 같은 이유다.

#![allow(dead_code)]

use std::sync::atomic::{AtomicUsize, Ordering};

use rust_multi_ranking_engine::{
    Candidate, CandidateId, Outcome, ScoreScale, Scorer, ScorerCost, ScorerId,
};

/// 시험용 후보. 도메인은 없고 축만 있다.
#[derive(Clone, Debug, PartialEq)]
pub struct Doc {
    pub id: u64,
    pub source: &'static str,
    pub relevance: Option<f32>,
    pub authority: f32,
    pub logit: f32,
    pub tokens: u32,
    pub topics: Vec<u32>,
}

impl Default for Doc {
    fn default() -> Self {
        Doc {
            id: 0,
            source: "arxiv",
            relevance: Some(0.5),
            authority: 0.5,
            logit: 0.0,
            tokens: 1,
            topics: Vec::new(),
        }
    }
}

impl Candidate for Doc {
    fn id(&self) -> CandidateId {
        CandidateId::num(self.id)
    }
}

/// 단위 척도의 싼 축.
pub struct Unit(pub &'static str, pub fn(&Doc) -> Option<f32>);

impl Scorer<Doc> for Unit {
    fn id(&self) -> ScorerId {
        ScorerId::new(self.0)
    }
    fn scale(&self) -> ScoreScale {
        ScoreScale::Unit
    }
    fn cost(&self) -> ScorerCost {
        ScorerCost::Cheap
    }
    fn score(&self, d: &Doc) -> Option<f32> {
        (self.1)(d)
    }
}

/// 단위 척도의 비싼 축. 2단계에서만 돈다.
pub struct Expensive(pub &'static str, pub fn(&Doc) -> Option<f32>);

impl Scorer<Doc> for Expensive {
    fn id(&self) -> ScorerId {
        ScorerId::new(self.0)
    }
    fn scale(&self) -> ScoreScale {
        ScoreScale::Unit
    }
    fn cost(&self) -> ScorerCost {
        ScorerCost::Expensive
    }
    fn score(&self, d: &Doc) -> Option<f32> {
        (self.1)(d)
    }
}

/// 배치로 채점하는 비싼 축. 몇 번 불렸고 한 번에 몇 개를 받았는지 센다.
///
/// 파이썬 콜백이 이 모양이다. 하나씩 960번이 아니라 한 번에 960개를 받는다.
pub struct Batched {
    pub name: &'static str,
    pub calls: AtomicUsize,
    pub widest: AtomicUsize,
    /// 참이면 일부러 길이가 어긋난 결과를 돌려준다.
    pub broken: bool,
}

impl Batched {
    pub fn new(name: &'static str) -> Self {
        Batched {
            name,
            calls: AtomicUsize::new(0),
            widest: AtomicUsize::new(0),
            broken: false,
        }
    }

    pub fn broken(name: &'static str) -> Self {
        Batched {
            broken: true,
            ..Batched::new(name)
        }
    }
}

impl Scorer<Doc> for Batched {
    fn id(&self) -> ScorerId {
        ScorerId::new(self.name)
    }
    fn scale(&self) -> ScoreScale {
        ScoreScale::Unit
    }
    fn cost(&self) -> ScorerCost {
        ScorerCost::Expensive
    }
    fn score(&self, d: &Doc) -> Option<f32> {
        Some(d.authority)
    }
    fn score_batch(&self, candidates: &[&Doc]) -> Vec<Option<f32>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.widest.fetch_max(candidates.len(), Ordering::SeqCst);
        let mut out: Vec<Option<f32>> = candidates.iter().map(|d| Some(d.authority)).collect();
        if self.broken {
            out.pop();
        }
        out
    }
}

/// 무한 척도 축. 값 기반 융합에 넣으면 거부돼야 한다.
pub struct Unbounded(pub &'static str);

impl Scorer<Doc> for Unbounded {
    fn id(&self) -> ScorerId {
        ScorerId::new(self.0)
    }
    fn scale(&self) -> ScoreScale {
        ScoreScale::Unbounded
    }
    fn cost(&self) -> ScorerCost {
        ScorerCost::Cheap
    }
    fn score(&self, d: &Doc) -> Option<f32> {
        Some(d.logit)
    }
}

pub fn relevance(d: &Doc) -> Option<f32> {
    d.relevance
}

pub fn authority(d: &Doc) -> Option<f32> {
    Some(d.authority)
}

/// 고정 시드 선형 합동 생성기. 표준 매개변수(Numerical Recipes)를 쓴다.
pub struct Lcg(u64);

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Lcg(seed)
    }

    pub fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }

    /// 0 이상 1 미만.
    pub fn unit(&mut self) -> f32 {
        (self.next_u32() % 100_000) as f32 / 100_000.0
    }

    /// 0 이상 `n` 미만.
    pub fn below(&mut self, n: u32) -> u32 {
        self.next_u32() % n.max(1)
    }
}

const SOURCES: [&str; 4] = ["arxiv", "blog", "news", "web"];

/// 무작위 후보 묶음. 같은 시드면 언제나 같은 묶음이 나온다.
pub fn corpus(seed: u64, n: u64) -> Vec<Doc> {
    let mut rng = Lcg::new(seed);
    (0..n)
        .map(|id| {
            let missing = rng.below(10) == 0;
            Doc {
                id,
                source: SOURCES[rng.below(4) as usize],
                relevance: if missing { None } else { Some(rng.unit()) },
                authority: rng.unit(),
                logit: rng.unit() * 8.0 - 4.0,
                tokens: 1 + rng.below(9),
                topics: (0..1 + rng.below(3)).map(|_| rng.below(12)).collect(),
            }
        })
        .collect()
}

/// 결과의 지문. 설정을 바꾸지 않았는데 이 값이 달라지면 결정성이 깨진 것이다.
pub fn fingerprint(out: &Outcome<Doc>) -> String {
    let mut s = String::new();
    for r in &out.ranked {
        s.push_str(&format!("{}:{:.6};", r.candidate.id, r.fused));
    }
    s.push('|');
    s.push_str(&format!(
        "{}/{}/{}/{}/{}/{}",
        out.rejected_counts.not_scored,
        out.rejected_counts.unary_constraint,
        out.rejected_counts.set_constraint,
        out.rejected_counts.below_threshold,
        out.rejected_counts.outranked,
        out.rejected_counts.out_of_pool
    ));
    s
}
