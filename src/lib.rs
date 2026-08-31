//! 여러 축의 점수를 스케일 안전하게 융합하고, 집합 제약 아래에서 상위 K 개를 고르며,
//! 탈락 사유까지 남기는 결정적 러스트 엔진.
//!
//! 후보가 수백만 개이고 고를 것이 수십 개일 때, 무엇을 고를지 정하는 계층을 담당한다.
//! 후보는 문서일 수도, 보안 이벤트일 수도, 유전자 돌연변이일 수도 있다. 엔진은 후보가
//! 무엇인지 모른다.
//!
//! # 설계 원칙
//!
//! **정렬이 아니라 선택이다.** 순위표를 만드는 것이 목적이 아니라 고르는 것이 목적이다.
//! 그래서 반환값의 중심은 선택된 집합과 그 근거다.
//!
//! **틀리기 쉬운 것을 못 하게 만든다.** 로짓과 확률을 섞어 더하는 것은 흔한 실수다.
//! 엔진은 그것을 문서로 경고하는 대신 [`Error::IncompatibleScale`] 로 거부한다.
//! 채점기가 자기 점수의 척도를 선언하게 하고, 비교 불가능한 척도에 가중합을 걸면
//! 후보를 한 건도 읽기 전에 오류가 난다.
//!
//! **조용히 사라지는 후보가 없다.** 모든 후보는 결과에 들어가거나 사유와 함께
//! 기록되거나 둘 중 하나다. 탈락 사유 출력은 부가 기능이 아니라 이 엔진이 규제
//! 도메인에서 쓰일 수 있는 최소 요건이다.
//!
//! **결정적이다.** 같은 입력이면 스레드 수와 무관하게 같은 출력이 나온다. 동점 처리
//! 순서까지 같다. 그래야 캐싱과 테스트와 감사 추적이 성립한다.
//!
//! **모델을 부르지 않는다.** 신경망을 실행하지 않는다. 채점기는 트레잇이고 그 안에서
//! 무엇을 부를지는 호출자의 몫이다. 기본 빌드는 순수 러스트이며 외부 함수
//! 인터페이스(FFI, Foreign Function Interface) 호출이 없다.
//!
//! # 두 단계
//!
//! 스트리밍과 집합 제약은 근본적으로 충돌한다. 집합 제약을 보려면 후보 풀이 있어야
//! 하는데 스트리밍은 풀을 만들지 않는 것이 목적이다. 그래서 나눈다.
//!
//! ```text
//! 1단계 (스트리밍)   후보 N개 -> 단항 제약 -> 싼 채점기 -> 유계 힙 상위 M개
//!                    M = K 곱하기 배수 (기본 32배)
//!                    메모리 O(M), 시간 O(N log M)
//!
//! 2단계 (풀 위에서)  상위 M개 -> 비싼 채점기 -> 집합 제약 아래 선택 -> 최종 K개
//!                    매트로이드면 탐욕이 정확, 아니면 근사와 보장 계수 보고
//! ```
//!
//! **이것은 근사이고 그 사실을 숨기지 않는다.** M 밖으로 밀려난 후보가 집합 제약 때문에
//! 최종 해에 들어가야 했을 수 있다. 그래서 [`Selection`] 에 `pool_size`·`pool_exhausted`·
//! `exact`·`guarantee`·`cut_margin` 을 실어 보낸다.
//!
//! # 빠른 시작
//!
//! ```
//! use rust_multi_ranking_engine::{
//!     constraint, Budget, Candidate, CandidateId, Engine, Fusion, ScoreScale, Scorer,
//!     ScorerCost, ScorerId,
//! };
//!
//! struct Doc {
//!     id: u64,
//!     source: &'static str,
//!     relevance: f32,
//!     authority: f32,
//! }
//!
//! impl Candidate for Doc {
//!     fn id(&self) -> CandidateId {
//!         CandidateId::num(self.id)
//!     }
//! }
//!
//! struct Axis(&'static str, fn(&Doc) -> f32);
//!
//! impl Scorer<Doc> for Axis {
//!     fn id(&self) -> ScorerId { ScorerId::new(self.0) }
//!     fn scale(&self) -> ScoreScale { ScoreScale::Unit }
//!     fn cost(&self) -> ScorerCost { ScorerCost::Cheap }
//!     fn score(&self, d: &Doc) -> Option<f32> { Some((self.1)(d)) }
//! }
//!
//! let docs = vec![
//!     Doc { id: 1, source: "arxiv", relevance: 0.95, authority: 0.9 },
//!     Doc { id: 2, source: "arxiv", relevance: 0.90, authority: 0.9 },
//!     Doc { id: 3, source: "blog",  relevance: 0.60, authority: 0.5 },
//!     Doc { id: 4, source: "arxiv", relevance: 0.55, authority: 0.9 },
//! ];
//!
//! let out = Engine::new()
//!     .scorer(Axis("relevance", |d| d.relevance))
//!     .scorer(Axis("authority", |d| d.authority))
//!     .fuse(Fusion::weighted_sum())
//!     .unary(constraint::predicate("min_authority", |d: &Doc| d.authority >= 0.3))
//!     .set_constraint(constraint::max_per_group("max_per_source", 2, |d: &Doc| d.source))
//!     .budget(Budget::TopK(3))
//!     .run(docs)
//!     .unwrap();
//!
//! // arxiv 는 둘까지만 들어가므로 3위는 점수가 더 낮은 blog 가 차지한다.
//! let picked: Vec<u64> = out.ranked.iter().map(|r| r.candidate.id).collect();
//! assert_eq!(picked, vec![1, 2, 3]);
//!
//! // 4번은 점수에 밀린 것이 아니라 출처 제약에 막혔다. 그 사실이 사유로 남는다.
//! assert!(out.is_complete());
//! assert_eq!(out.rejected_counts.set_constraint, 1);
//! ```
//!
//! # 이 엔진이 필요 없는 경우
//!
//! 정직하게 적어 둔다. 다음 경우에는 쓰지 않는 편이 낫다.
//!
//! - 후보 수가 K 보다 조금 많은 정도일 때. 그냥 정렬하면 된다
//! - 점수 축이 하나뿐일 때. 융합할 것이 없다
//! - 집합 제약이 없을 때. 상위 K 가 이미 최적이다
//! - 병목이 선택이 아니라 그 뒤의 생성 단계일 때
//!
//! # 참고 자료
//!
//! 실제로 근거로 쓴 것만 싣는다. 결과에 실려 나가는 보장 계수는 여기까지 되짚을 수 있다.
//!
//! 1. G. L. Nemhauser, L. A. Wolsey, M. L. Fisher, An analysis of approximations for
//!    maximizing submodular set functions I (1978).
//!    [`GUARANTEE_CARDINALITY`] = `1 - 1/e`. 서브모듈러 + 개수 제한
//! 2. M. L. Fisher, G. L. Nemhauser, L. A. Wolsey, An analysis of approximations for
//!    maximizing submodular set functions II (1978).
//!    [`GUARANTEE_MATROID`] = `1/2`. 서브모듈러 + 매트로이드
//! 3. J. Leskovec et al., Cost-effective outbreak detection in networks, KDD 2007.
//!    [`GUARANTEE_KNAPSACK_SUBMODULAR`] = `(1 - 1/e)/2`. 서브모듈러 + 배낭형
//! 4. G. V. Cormack et al., Reciprocal Rank Fusion outperforms Condorcet and individual
//!    Rank Learning Methods, SIGIR 2009. [`Fusion::Rrf`] 와 관례 상수 `k = 60`
//! 5. A. Clauset, C. R. Shalizi, M. E. J. Newman, Power-law distributions in empirical
//!    data (2009). [`Budget::TailMass`] 의 적합도 검정
//!
//! [`GUARANTEE_KNAPSACK_MODULAR`] = `1/2` 만 예외다. 비율 탐욕과 최고가 단일 항목 중
//! 나은 쪽(ModifiedGreedy)의 표준 결과라 특정 논문 하나로 귀속시키지 않았다.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod budget;
pub mod candidate;
pub mod constraint;
pub mod engine;
pub mod error;
pub mod evidence;
pub mod fuse;
pub mod objective;
pub mod score;

mod select;

#[cfg(feature = "python")]
mod python;

pub use budget::{tail_mass, Budget, BudgetTrace, FallbackReason, DEFAULT_MIN_FIT};
pub use candidate::{Candidate, CandidateId};
pub use constraint::{
    ConstraintId, CostBudget, MaxPerGroup, MaxTotal, Predicate, Requirement, SetConstraint,
    UnaryConstraint,
};
pub use engine::{Engine, DEFAULT_POOL_MULTIPLIER};
pub use error::{Error, Result};
pub use evidence::{
    Outcome, Ranked, RejectCounts, RejectReason, Rejected, Rejections, RunTrace, ScorerTrace,
    Selection,
};
pub use fuse::{
    Fusion, FusionInput, FusionMethod, FusionTerm, FusionTrace, MissingPolicy, DEFAULT_RRF_K,
};
pub use objective::{Coverage, SetObjective};
pub use score::{
    Normalized, Normalizer, ScoreScale, ScoreSet, Scorer, ScorerCost, ScorerExt, ScorerId,
};
pub use select::{
    GUARANTEE_CARDINALITY, GUARANTEE_KNAPSACK_MODULAR, GUARANTEE_KNAPSACK_SUBMODULAR,
    GUARANTEE_MATROID,
};
