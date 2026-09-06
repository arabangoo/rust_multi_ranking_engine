//! 결과와 탈락 사유, 그리고 감사 출력.
//!
//! 조용히 사라지는 후보가 없다. 모든 후보는 결과에 들어가거나 사유와 함께 기록되거나
//! 둘 중 하나다. 중간은 없다. 탈락 사유 출력은 부가 기능이 아니라 이 엔진이 규제
//! 도메인에서 쓰일 수 있는 최소 요건이다.

use std::fmt::Write as _;

use crate::budget::BudgetTrace;
use crate::candidate::CandidateId;
use crate::constraint::ConstraintId;
use crate::fuse::{FusionInput, FusionTrace, MissingPolicy};
use crate::score::{ScoreSet, ScorerId};

/// 최종 결과 하나. 왜 이 순위인지 재현 가능해야 한다.
#[derive(Clone, Debug)]
pub struct Ranked<C> {
    /// 골라진 후보.
    pub candidate: C,
    /// 1 부터 시작하는 순위.
    pub rank: u32,
    /// 융합 점수.
    pub fused: f32,
    /// 융합 전 원본 점수.
    pub scores: ScoreSet,
    /// 어떻게 합쳤는가.
    pub fusion: FusionTrace,
    /// 이 후보가 통과한 집합 제약들.
    ///
    /// 설계서의 `Ranked` 에는 없었지만 감사 출력 예시가
    /// `"constraints": { "max_per_source": "pass" }` 를 요구하므로 근거를 실어 둔다.
    pub constraints: Vec<ConstraintId>,
}

impl<C> Ranked<C> {
    /// 감사 출력용 JSON 한 덩어리.
    ///
    /// 의존성 없이 직접 쓴다. 유한하지 않은 값은 `null` 이 된다.
    pub fn to_json(&self, id: &CandidateId) -> String {
        let mut s = String::with_capacity(256);
        s.push('{');
        write!(s, "\"candidate\":{}", json_string(&id.to_string())).ok();
        write!(s, ",\"rank\":{}", self.rank).ok();
        write!(s, ",\"fused\":{}", json_f32(self.fused)).ok();

        s.push_str(",\"scores\":{");
        for (i, (name, value)) in self.scores.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            write!(
                s,
                "{}:{}",
                json_string(name.as_str()),
                value.map_or_else(|| "null".to_string(), json_f32)
            )
            .ok();
        }
        s.push('}');

        write!(
            s,
            ",\"missing_policy\":{}",
            json_string(missing_policy_name(self.fusion.missing))
        )
        .ok();

        write!(
            s,
            ",\"fusion\":{{\"method\":{}",
            json_string(self.fusion.method.name())
        )
        .ok();
        if let Some(k) = self.fusion.k {
            write!(s, ",\"k\":{}", json_f32(k)).ok();
        }
        s.push_str(",\"terms\":[");
        for (i, t) in self.fusion.terms.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            write!(
                s,
                "{{\"scorer\":{},\"input\":{},\"weight\":{},\"contribution\":{}}}",
                json_string(t.scorer.as_str()),
                json_fusion_input(t.input),
                json_f32(t.weight),
                json_f32(t.contribution)
            )
            .ok();
        }
        s.push_str("]}");

        s.push_str(",\"constraints\":{");
        for (i, c) in self.constraints.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            write!(s, "{}:\"pass\"", json_string(c.as_str())).ok();
        }
        s.push('}');

        s.push('}');
        s
    }
}

/// 떨어진 후보.
#[derive(Clone, Debug)]
pub struct Rejected<C> {
    /// 떨어진 후보.
    pub candidate: C,
    /// 왜 떨어졌는가.
    pub reason: RejectReason,
    /// 융합까지 갔다면 그 점수. 1단계에서 떨어졌으면 `None`.
    pub fused: Option<f32>,
}

impl<C> Rejected<C> {
    /// 감사 출력용 JSON 한 덩어리.
    pub fn to_json(&self, id: &CandidateId) -> String {
        let mut s = String::with_capacity(128);
        s.push('{');
        write!(s, "\"candidate\":{}", json_string(&id.to_string())).ok();
        write!(s, ",\"reason\":{}", self.reason.to_json()).ok();
        write!(
            s,
            ",\"fused\":{}",
            self.fused.map_or_else(|| "null".to_string(), json_f32)
        )
        .ok();
        s.push('}');
        s
    }
}

/// 후보가 떨어진 이유.
#[derive(Clone, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RejectReason {
    /// 채점기가 값을 내지 못했다. 0 점과 다르다.
    NotScored(ScorerId),
    /// 단항 제약에 걸렸다.
    UnaryConstraint(ConstraintId),
    /// 집합 제약 때문에 자리를 못 얻었다.
    SetConstraint(ConstraintId),
    /// 점수가 문턱 아래다.
    BelowThreshold,
    /// 더 높은 후보에게 밀렸다.
    Outranked,
    /// 1단계 풀에 들지 못했다.
    OutOfPool,
}

impl RejectReason {
    /// 사유의 종류 이름. 개수 집계의 키다.
    pub fn kind(&self) -> &'static str {
        match self {
            RejectReason::NotScored(_) => "not_scored",
            RejectReason::UnaryConstraint(_) => "unary_constraint",
            RejectReason::SetConstraint(_) => "set_constraint",
            RejectReason::BelowThreshold => "below_threshold",
            RejectReason::Outranked => "outranked",
            RejectReason::OutOfPool => "out_of_pool",
        }
    }

    fn to_json(&self) -> String {
        match self {
            RejectReason::NotScored(id) => format!(
                "{{\"kind\":\"not_scored\",\"scorer\":{}}}",
                json_string(id.as_str())
            ),
            RejectReason::UnaryConstraint(id) => format!(
                "{{\"kind\":\"unary_constraint\",\"constraint\":{}}}",
                json_string(id.as_str())
            ),
            RejectReason::SetConstraint(id) => format!(
                "{{\"kind\":\"set_constraint\",\"constraint\":{}}}",
                json_string(id.as_str())
            ),
            other => format!("{{\"kind\":\"{}\"}}", other.kind()),
        }
    }
}

/// 탈락 후보를 얼마나 보관할지.
///
/// # 설계서에서 달라진 점
///
/// 설계서는 "모든 후보는 결과에 들어가거나 사유와 함께 기록된다"고 적었고 다른 자리에서는
/// "후보를 전부 메모리에 올리지 않는 것"이 더 중요하다고 적었다. 후보가 1,000만이면
/// 둘은 그대로 충돌한다.
///
/// 그래서 **개수는 언제나 정확히 세고**(그래서 완전성 불변식은 규모와 무관하게
/// 성립한다) 상세를 얼마나 남길지만 고를 수 있게 했다.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Rejections {
    /// 전부 보관한다. 규제 도메인과 소규모 입력의 기본값.
    #[default]
    Keep,
    /// 개수만 센다. 후보가 아주 많을 때.
    Count,
    /// 처음 만난 `n` 건만 보관하고 나머지는 센다.
    Sample(usize),
}

/// 사유별 탈락 개수. 보관 정책과 무관하게 언제나 정확하다.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RejectCounts {
    /// 채점기가 값을 내지 못해 떨어진 수.
    pub not_scored: u64,
    /// 단항 제약에 걸린 수.
    pub unary_constraint: u64,
    /// 집합 제약 때문에 자리를 못 얻은 수.
    pub set_constraint: u64,
    /// 문턱 아래로 떨어진 수.
    pub below_threshold: u64,
    /// 더 높은 후보에게 밀린 수.
    pub outranked: u64,
    /// 1단계 풀에 못 든 수.
    pub out_of_pool: u64,
}

impl RejectCounts {
    /// 사유 하나를 더 센다.
    pub fn record(&mut self, reason: &RejectReason) {
        match reason {
            RejectReason::NotScored(_) => self.not_scored += 1,
            RejectReason::UnaryConstraint(_) => self.unary_constraint += 1,
            RejectReason::SetConstraint(_) => self.set_constraint += 1,
            RejectReason::BelowThreshold => self.below_threshold += 1,
            RejectReason::Outranked => self.outranked += 1,
            RejectReason::OutOfPool => self.out_of_pool += 1,
        }
    }

    /// 전체 탈락 수.
    pub fn total(&self) -> u64 {
        self.not_scored
            + self.unary_constraint
            + self.set_constraint
            + self.below_threshold
            + self.outranked
            + self.out_of_pool
    }
}

/// 선택이 어떤 성질을 가졌는가.
///
/// 조용히 부족한 답을 주는 대신 말한다.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Selection {
    /// 풀 위에서 탐욕이 최적이었는가.
    ///
    /// **엔진이 그렇게 선언받았다는 뜻이지 엔진이 증명했다는 뜻이 아니다.** 사용자가
    /// 직접 구현한 제약의 `is_matroid` 선언을 그대로 믿는다. 그리고 이 값은 어디까지나
    /// **풀 위에서**의 최적성이다. 1단계가 후보를 잘랐으면 전체 입력에 대한 최적성은
    /// 여전히 보장되지 않는다 -- 그쪽 신호는 `pool_size` 와 `pool_exhausted` 다.
    pub exact: bool,
    /// 근사 보장 계수. `exact` 가 거짓일 때만 값이 있다.
    ///
    /// | 목적함수 | 제약 | 계수 |
    /// | --- | --- | --- |
    /// | 모듈러 | 매트로이드 1개 이하 | 정확 (`exact = true`) |
    /// | 모듈러 | 매트로이드 `p` 개 | `1/p` |
    /// | 모듈러 | `Budget::Tokens`만 | `0.5` |
    /// | 서브모듈러 | 개수 제한만 | `1 - 1/e` (약 0.632) |
    /// | 서브모듈러 | 매트로이드 1개 | `0.5` |
    /// | 서브모듈러 | 매트로이드 `p` 개 | `1/(p+1)` |
    /// | 서브모듈러 | `Budget::Tokens`만 | `(1 - 1/e)/2` (약 0.316) |
    ///
    /// 위 표에 없는 조합(일반 비매트로이드 제약, 매트로이드와 배낭형의 혼합,
    /// 서브모듈러라고 선언되지 않은 목적함수)에는 계수를 주지 않는다. 근거 없는
    /// 숫자를 결과에 싣지 않기 위해서다.
    /// `CostBudget`을 일반 집합 제약으로 등록한 경우에도 계수를 주지 않는다.
    /// 배낭 보장은 비용 전용 탐색을 실행하는 `Budget::Tokens`에만 적용된다.
    pub guarantee: Option<f32>,
    /// 1단계가 남긴 풀의 크기.
    pub pool_size: u32,
    /// 풀을 다 쓰고도 K 를 못 채웠다. 참이면 풀 배수를 키워야 한다.
    pub pool_exhausted: bool,
    /// K 번째 점수에서 K+1 번째 점수를 뺀 값.
    ///
    /// 거의 0 이면 그 선택은 흔들린다. **34 개를 골랐다는 것과 34 번째와 35 번째가
    /// 사실상 같았다는 것은 다른 사건이다.** 계산 비용은 사실상 0 이다 -- 이미 정렬된
    /// 두 값의 차다.
    ///
    /// K+1 번째는 **그 자리를 실제로 다툰 후보** 중 가장 높은 것이다. 마지막 하나를 뺀
    /// 집합을 기준으로 다시 판정하므로, 집합 제약에 막혀 애초에 자리를 다툰 적이 없는
    /// 후보는 여기 들어오지 않는다. 제약에 막힌 쪽의 규모는
    /// [`RejectCounts::set_constraint`] 가 따로 낸다.
    ///
    /// **잣대는 선택기가 실제로 쓴 기준이다.** 개수 예산이면 총 이득(융합 점수 + 목적함수의
    /// 한계 이득)이고 배낭형이면 비용 대비 이득이다. 융합 점수로만 재면 목적함수가 걸렸을 때
    /// 음수가 나오는데, 그 숫자는 순서가 뒤집혔다는 뜻이 아니라 잣대가 틀렸다는 뜻이다.
    ///
    /// 남은 후보가 없거나, 배낭형에서 단일 최고 항목이 이겨 절단선이라 부를 자리가
    /// 없었으면 `None` 이다.
    pub cut_margin: Option<f32>,
}

/// 채점기 하나의 실행 기록.
#[derive(Clone, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ScorerTrace {
    /// 어느 채점기인가.
    pub scorer: ScorerId,
    /// 몇 번 불렸는가.
    pub calls: u64,
    /// 값을 못 낸 횟수.
    pub missing: u64,
    /// 누적 소요 나노초.
    pub elapsed_nanos: u128,
}

/// 이번 실행이 어떻게 굴러갔는가.
#[derive(Clone, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RunTrace {
    /// 1단계가 훑은 후보 수.
    pub input_count: u64,
    /// 1단계 유계 힙의 크기 상한 M.
    pub pool_capacity: u32,
    /// 1단계 절단에 쓴 승인 채점기. 값 기반 융합을 스트리밍으로 계산했으면 `None`.
    pub admission_scorer: Option<ScorerId>,
    /// 채점기별 기록. 등록 순서대로다.
    pub scorers: Vec<ScorerTrace>,
    /// 예산을 어떻게 정했는가. 고정 K 였으면 `None`.
    pub budget: Option<BudgetTrace>,
}

/// 실행 결과 전부.
#[derive(Debug)]
pub struct Outcome<C> {
    /// 골라진 것들. 순위 오름차순이다.
    pub ranked: Vec<Ranked<C>>,
    /// 떨어진 것들. 보관 정책이 정한 만큼만 들어 있다.
    pub rejected: Vec<Rejected<C>>,
    /// 사유별 탈락 개수. 보관 정책과 무관하게 언제나 정확하다.
    pub rejected_counts: RejectCounts,
    /// 선택의 성질.
    pub selection: Selection,
    /// 실행 기록.
    pub trace: RunTrace,
}

impl<C> Outcome<C> {
    /// 완전성 불변식. 결과 수와 탈락 수의 합이 입력 수와 같은가.
    pub fn is_complete(&self) -> bool {
        self.ranked.len() as u64 + self.rejected_counts.total() == self.trace.input_count
    }
}

fn missing_policy_name(p: MissingPolicy) -> &'static str {
    match p {
        MissingPolicy::Skip => "skip",
        MissingPolicy::Impute(_) => "impute",
        MissingPolicy::Reject => "reject",
    }
}

fn json_fusion_input(input: FusionInput) -> String {
    match input {
        FusionInput::Value(v) => format!("{{\"kind\":\"value\",\"value\":{}}}", json_f32(v)),
        FusionInput::Rank(r) => format!("{{\"kind\":\"rank\",\"rank\":{r}}}"),
        FusionInput::Imputed(v) => format!("{{\"kind\":\"imputed\",\"value\":{}}}", json_f32(v)),
        FusionInput::Skipped => "{\"kind\":\"skipped\"}".to_string(),
    }
}

/// 유한하지 않은 값은 JSON 이 표현하지 못하므로 `null` 로 낸다.
fn json_f32(v: f32) -> String {
    if v.is_finite() {
        let s = format!("{v}");
        // 정수로 찍히면 JSON 에서 정수형으로 읽히므로 소수점을 붙여 둔다.
        if s.contains('.') || s.contains('e') || s.contains('E') {
            s
        } else {
            format!("{s}.0")
        }
    } else {
        "null".to_string()
    }
}

/// JSON 문자열 리터럴로 감싼다. 제어문자까지 이스케이프한다.
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                write!(out, "\\u{:04x}", c as u32).ok();
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fuse::{FusionMethod, FusionTerm};

    #[test]
    fn json_escapes_quotes_and_control_characters() {
        assert_eq!(json_string("a\"b"), "\"a\\\"b\"");
        assert_eq!(json_string("a\nb"), "\"a\\nb\"");
        assert_eq!(json_string("a\u{1}b"), "\"a\\u0001b\"");
    }

    #[test]
    fn non_finite_scores_become_null() {
        assert_eq!(json_f32(f32::NAN), "null");
        assert_eq!(json_f32(f32::INFINITY), "null");
        assert_eq!(json_f32(1.0), "1.0");
        assert_eq!(json_f32(0.5), "0.5");
    }

    #[test]
    fn a_ranked_row_carries_the_missing_axis_and_its_policy() {
        let ranked = Ranked {
            candidate: (),
            rank: 3,
            fused: 0.913,
            scores: ScoreSet::new(vec![
                (ScorerId::new("semantic"), Some(0.94)),
                (ScorerId::new("recency"), None),
            ]),
            fusion: FusionTrace {
                method: FusionMethod::Rrf,
                k: Some(60.0),
                missing: MissingPolicy::Skip,
                terms: vec![FusionTerm {
                    scorer: ScorerId::new("semantic"),
                    input: FusionInput::Rank(1),
                    weight: 1.0,
                    contribution: 0.016_393,
                }],
            },
            constraints: vec![ConstraintId::new("max_per_source")],
        };

        let json = ranked.to_json(&CandidateId::text("doc-3141"));
        assert!(json.contains("\"candidate\":\"doc-3141\""));
        assert!(json.contains("\"recency\":null"));
        assert!(json.contains("\"missing_policy\":\"skip\""));
        assert!(json.contains("\"method\":\"rrf\""));
        assert!(json.contains("\"max_per_source\":\"pass\""));
    }

    #[test]
    fn counts_stay_exact_whatever_the_retention_policy_is() {
        let mut counts = RejectCounts::default();
        counts.record(&RejectReason::OutOfPool);
        counts.record(&RejectReason::OutOfPool);
        counts.record(&RejectReason::Outranked);
        assert_eq!(counts.total(), 3);
        assert_eq!(counts.out_of_pool, 2);
    }
}
