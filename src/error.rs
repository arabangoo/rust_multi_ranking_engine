//! 엔진 오류.
//!
//! 오류는 대부분 설정 오류이고 후보를 한 건도 처리하기 전에 걸린다. 실행 중에 생기는
//! 오류는 하나뿐이다 -- 요구 조건이 실현 불가능한 경우.

use crate::score::{ScoreScale, ScorerId};
use thiserror::Error;

/// 엔진이 낼 수 있는 모든 오류.
#[derive(Debug, Error, PartialEq, Eq, Clone)]
pub enum Error {
    /// 채점기가 하나도 등록되지 않았다.
    #[error("채점기가 하나도 없다. Engine::scorer 로 최소 하나를 등록해야 한다")]
    NoScorers,

    /// 같은 식별자의 채점기가 두 번 등록됐다.
    #[error("채점기 식별자 '{0}' 가 중복 등록됐다")]
    DuplicateScorer(ScorerId),

    /// 비교 불가능한 척도에 값 기반 융합을 걸었다.
    ///
    /// 이것이 이 엔진이 문서로 경고하는 대신 거부하는 대표적인 실수다. 로짓과 확률을
    /// 그대로 더하면 로짓 축 하나가 나머지 전부를 압도한다.
    #[error(
        "채점기 '{scorer}' 의 척도 {scale:?} 는 융합 방식 {fusion} 과 함께 쓸 수 없다. \
         정규화기를 끼우거나(Scorer::normalized) 순위 융합(Fusion::Rrf)으로 바꿔야 한다"
    )]
    IncompatibleScale {
        /// 문제가 된 채점기.
        scorer: ScorerId,
        /// 그 채점기가 선언한 척도.
        scale: ScoreScale,
        /// 걸려 있는 융합 방식의 이름.
        fusion: &'static str,
    },

    /// 가중치가 등록되지 않은 채점기를 가리킨다.
    #[error("가중치가 가리키는 채점기 '{0}' 가 등록돼 있지 않다")]
    UnknownWeight(ScorerId),

    /// 승인 채점기로 지정된 것이 등록돼 있지 않다.
    #[error("승인 채점기 '{0}' 가 등록돼 있지 않다")]
    UnknownAdmissionScorer(ScorerId),

    /// 승인 채점기가 싼 축이 아니다.
    ///
    /// 1단계는 후보 전부를 훑으므로 승인 채점기가 비싸면 캐스케이드의 뜻이 사라진다.
    #[error("승인 채점기 '{0}' 가 비싼 축이다. 1단계는 후보 전부를 훑으므로 싼 축이어야 한다")]
    ExpensiveAdmissionScorer(ScorerId),

    /// 순위 융합인데 승인 채점기를 정할 수 없다.
    ///
    /// 순위 융합은 순위를 입력으로 쓰고 순위는 풀이 있어야 나온다. 그래서 1단계가
    /// 상위 M 개를 무엇으로 자를지 따로 정해야 한다.
    #[error(
        "순위 융합(Fusion::Rrf)은 1단계 절단 기준을 스스로 만들지 못한다. \
         싼 채점기를 하나 이상 등록하거나 Engine::admission 으로 승인 채점기를 지정해야 한다"
    )]
    NoAdmissionScorer,

    /// 예산 값이 뜻을 갖지 못한다.
    #[error("예산 설정이 올바르지 않다: {0}")]
    InvalidBudget(&'static str),

    /// 풀 배수가 0 이다.
    #[error("풀 배수는 1 이상이어야 한다")]
    InvalidPoolMultiplier,

    /// 배치 채점기가 입력과 다른 길이의 결과를 돌려줬다.
    ///
    /// 순수 러스트 구현에서는 잘 나지 않지만, 파이썬 콜백처럼 밖에서 값을 만들어 오는
    /// 채점기에서는 실제로 생기는 실패다. 조용히 어긋난 값을 쓰지 않고 여기서 멈춘다.
    #[error("채점기 '{scorer}' 의 배치 결과 길이가 {got} 인데 후보는 {expected} 개다")]
    BatchLengthMismatch {
        /// 문제가 된 채점기.
        scorer: ScorerId,
        /// 넘긴 후보 수.
        expected: usize,
        /// 돌려받은 값의 수.
        got: usize,
    },

    /// 요구 조건을 만족시킬 후보가 풀에 모자라 최종 집합을 만들 수 없다.
    ///
    /// 실행 중에 나오는 유일한 오류다. 설정만 보고는 알 수 없고 후보를 봐야 안다.
    #[error(
        "요구 조건 '{id}' 를 만족시킬 수 없다. 최소 {needed} 개가 필요한데 \
         풀에서 제약을 지키며 넣을 수 있는 것이 {available} 개다"
    )]
    InfeasibleRequirement {
        /// 요구 조건 식별자.
        id: String,
        /// 필요한 개수.
        needed: usize,
        /// 실제로 확보 가능한 개수.
        available: usize,
    },
}

/// 엔진 결과 타입.
pub type Result<T> = std::result::Result<T, Error>;
