//! 엔진 오류.
//!
//! 오류는 대부분 설정 오류이고 후보를 한 건도 처리하기 전에 걸린다. 실행 중에 생기는
//! 오류에는 요구 조건 교체 실패와 배치 결과 길이 불일치가 있다.

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

    /// 현재 풀에서 요구 조건을 충족하는 집합을 교체 절차로 만들지 못했다.
    ///
    /// 후보 부족이나 제약 충돌 외에도 단일 교체 탐색의 한계로 발생할 수 있다.
    /// 가능한 집합이 전혀 없다는 수학적 증명은 아니다.
    #[error(
        "요구 조건 '{id}' 의 교체에 실패했다. 최소 {needed} 개가 필요하고 \
         풀에서 해당 조건에 맞는 후보는 {available} 개다. 다른 제약과의 충돌도 확인해야 한다"
    )]
    InfeasibleRequirement {
        /// 요구 조건 식별자.
        id: String,
        /// 필요한 개수.
        needed: usize,
        /// 다른 제약을 적용하기 전 풀에서 해당 술어를 만족하는 후보 수.
        available: usize,
    },
}

/// 엔진 결과 타입.
pub type Result<T> = std::result::Result<T, Error>;
