//! 융합 방식과 융합 기록.
//!
//! 축이 서로 다른 곳에서 왔다면 값을 더하는 것보다 순위를 합치는 편이 안전하다. 그래서
//! [`Fusion::Rrf`] 가 기본값이고, 값 기반 융합은 척도가 맞을 때만 허용된다.
//!
//! 융합 결과만 남기면 "왜 이 순위인가"에 답할 수 없다. 그래서 모든 융합은
//! [`FusionTrace`] 를 함께 낸다. 그 기록과 융합 전 원본 점수만으로 최종 점수를 다시
//! 계산할 수 있어야 한다는 것이 근거 재현 불변식이다.

use crate::score::{ScoreScale, ScorerId};

/// 값이 없는 축을 만났을 때의 정책.
///
/// 채점 불가와 0 점은 다르다. 모델이 돌지 않았는데 0 점으로 세면 그 후보는 부당하게 죽는다.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MissingPolicy {
    /// 그 축을 빼고 나머지로 융합한다. 가중합이면 가중치를 재정규화한다.
    /// 축이 선택적일 때 쓴다.
    #[default]
    Skip,
    /// 지정한 값으로 대체한다. 결측이 의미를 갖는 경우에 쓴다.
    Impute(f32),
    /// 후보를 [`RejectReason::NotScored`](crate::RejectReason::NotScored) 로 탈락시킨다.
    /// 필수 축일 때 쓴다.
    Reject,
}

/// 순위 융합의 관례 상수.
pub const DEFAULT_RRF_K: f32 = 60.0;

/// 여러 축을 하나의 점수로 합치는 방법.
///
/// 척도 조합표. 거부는 실행 시점 오류가 아니라 설정 오류라, 후보를 한 건도 처리하기 전에 걸린다.
///
/// | 융합 방식 | `Unit` 만 | `Unbounded` 포함 | `Rank` 포함 |
/// | --- | --- | --- | --- |
/// | [`WeightedSum`](Self::WeightedSum) | 허용 | 거부 | 거부 |
/// | [`Rrf`](Self::Rrf) | 허용 | 허용 | 허용 |
/// | [`Max`](Self::Max) | 허용 | 거부 | 거부 |
#[derive(Clone, PartialEq, Debug)]
pub enum Fusion {
    /// 가중합. 값을 더하므로 모든 축이 단위 척도여야 한다.
    WeightedSum {
        /// 축별 가중치. 비어 있으면 모든 축이 1.0 이다. 등록되지 않은 채점기를
        /// 가리키면 설정 오류다.
        weights: Vec<(ScorerId, f32)>,
    },
    /// 순위 융합(Reciprocal Rank Fusion). 각 축에서의 순위 `r` 에 대해 `1 / (k + r)` 을
    /// 더한다. 값을 쓰지 않고 순서만 쓰므로 어떤 척도 조합에도 걸린다.
    Rrf {
        /// 상위 순위의 영향력을 조절한다. 값이 크면 순위 차이가 완만해진다.
        k: f32,
    },
    /// 축들 중 최댓값. 값을 비교하므로 모든 축이 단위 척도여야 한다.
    Max,
}

impl Default for Fusion {
    fn default() -> Self {
        Fusion::Rrf { k: DEFAULT_RRF_K }
    }
}

impl Fusion {
    /// 가중치 없는 가중합(모든 축 1.0).
    pub fn weighted_sum() -> Self {
        Fusion::WeightedSum {
            weights: Vec::new(),
        }
    }

    /// 관례 상수 `k = 60` 의 순위 융합.
    pub fn rrf() -> Self {
        Fusion::Rrf { k: DEFAULT_RRF_K }
    }

    /// 이 방식의 이름. 오류 메시지와 감사 출력에 쓰인다.
    pub fn name(&self) -> &'static str {
        self.method().name()
    }

    /// 이 척도를 받아들이는가.
    pub fn accepts(&self, scale: ScoreScale) -> bool {
        match self {
            Fusion::Rrf { .. } => true,
            Fusion::WeightedSum { .. } | Fusion::Max => scale == ScoreScale::Unit,
        }
    }

    pub(crate) fn method(&self) -> FusionMethod {
        match self {
            Fusion::WeightedSum { .. } => FusionMethod::WeightedSum,
            Fusion::Rrf { .. } => FusionMethod::Rrf,
            Fusion::Max => FusionMethod::Max,
        }
    }
}

/// 융합 기록에 실리는 방식 이름.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FusionMethod {
    /// 가중합.
    WeightedSum,
    /// 순위 융합.
    Rrf,
    /// 최댓값.
    Max,
}

impl FusionMethod {
    /// 감사 출력에 쓰이는 이름.
    pub fn name(&self) -> &'static str {
        match self {
            FusionMethod::WeightedSum => "weighted_sum",
            FusionMethod::Rrf => "rrf",
            FusionMethod::Max => "max",
        }
    }
}

/// 한 축이 융합에 실제로 넣은 입력.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FusionInput {
    /// 채점기가 낸 값 그대로.
    Value(f32),
    /// 풀 안에서의 순위(1 부터). 순위 융합에서만 나온다.
    Rank(u32),
    /// 값이 없어 [`MissingPolicy::Impute`] 로 대체된 값.
    Imputed(f32),
    /// 값이 없어 [`MissingPolicy::Skip`] 으로 빠진 축.
    Skipped,
}

impl FusionInput {
    /// 이 축이 융합에 실제로 참여했는가.
    pub fn is_used(&self) -> bool {
        !matches!(self, FusionInput::Skipped)
    }
}

/// 융합에 참여한 축 하나의 기록.
#[derive(Clone, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FusionTerm {
    /// 어느 축인가.
    pub scorer: ScorerId,
    /// 무엇을 넣었는가.
    pub input: FusionInput,
    /// 재정규화까지 반영된 실효 가중치. 빠진 축은 0 이다.
    pub weight: f32,
    /// 이 축이 최종 점수에 실제로 기여한 값.
    pub contribution: f32,
}

/// 어떻게 합쳤는가.
///
/// [`recompute`](Self::recompute) 가 이 기록만으로 최종 점수를 다시 만들어 낸다.
/// 근거 재현 불변식이 그 값과 [`Ranked::fused`](crate::Ranked::fused) 의 일치를 요구한다.
#[derive(Clone, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FusionTrace {
    /// 어떤 방식으로 합쳤는가.
    pub method: FusionMethod,
    /// 순위 융합의 `k`. 다른 방식에서는 `None`.
    pub k: Option<f32>,
    /// 결측을 어떻게 다뤘는가.
    pub missing: MissingPolicy,
    /// 축별 기록. 등록 순서대로다.
    pub terms: Vec<FusionTerm>,
}

impl FusionTrace {
    /// 기록만으로 최종 점수를 다시 계산한다.
    ///
    /// 가중합과 순위 융합은 기여값의 합이고, 최댓값은 기여값의 최대다. 참여한 축이
    /// 하나도 없으면 0 이다.
    pub fn recompute(&self) -> f32 {
        let used = self
            .terms
            .iter()
            .filter(|t| t.input.is_used())
            .map(|t| t.contribution);

        match self.method {
            FusionMethod::WeightedSum | FusionMethod::Rrf => used.sum(),
            FusionMethod::Max => used
                .fold(None::<f32>, |acc, v| Some(acc.map_or(v, |a| a.max(v))))
                .unwrap_or(0.0),
        }
    }

    /// 실제로 융합에 참여한 축의 수.
    pub fn used_axes(&self) -> usize {
        self.terms.iter().filter(|t| t.input.is_used()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn term(name: &str, input: FusionInput, weight: f32, contribution: f32) -> FusionTerm {
        FusionTerm {
            scorer: ScorerId::new(name),
            input,
            weight,
            contribution,
        }
    }

    #[test]
    fn weighted_sum_recomputes_from_the_trace_alone() {
        let trace = FusionTrace {
            method: FusionMethod::WeightedSum,
            k: None,
            missing: MissingPolicy::Skip,
            terms: vec![
                term("a", FusionInput::Value(0.8), 0.5, 0.4),
                term("b", FusionInput::Value(0.6), 0.5, 0.3),
                term("c", FusionInput::Skipped, 0.0, 0.0),
            ],
        };
        assert!((trace.recompute() - 0.7).abs() < 1e-6);
        assert_eq!(trace.used_axes(), 2);
    }

    #[test]
    fn max_recomputes_as_the_largest_contribution() {
        let trace = FusionTrace {
            method: FusionMethod::Max,
            k: None,
            missing: MissingPolicy::Skip,
            terms: vec![
                term("a", FusionInput::Value(0.3), 1.0, 0.3),
                term("b", FusionInput::Value(0.9), 1.0, 0.9),
            ],
        };
        assert!((trace.recompute() - 0.9).abs() < 1e-6);
    }

    #[test]
    fn an_all_skipped_trace_recomputes_to_zero() {
        let trace = FusionTrace {
            method: FusionMethod::Max,
            k: None,
            missing: MissingPolicy::Skip,
            terms: vec![term("a", FusionInput::Skipped, 0.0, 0.0)],
        };
        assert_eq!(trace.recompute(), 0.0);
    }

    #[test]
    fn rrf_accepts_every_scale_and_the_others_do_not() {
        let rrf = Fusion::rrf();
        let sum = Fusion::weighted_sum();
        for scale in [ScoreScale::Unit, ScoreScale::Unbounded, ScoreScale::Rank] {
            assert!(rrf.accepts(scale));
        }
        assert!(sum.accepts(ScoreScale::Unit));
        assert!(!sum.accepts(ScoreScale::Unbounded));
        assert!(!sum.accepts(ScoreScale::Rank));
        assert!(!Fusion::Max.accepts(ScoreScale::Rank));
    }
}
