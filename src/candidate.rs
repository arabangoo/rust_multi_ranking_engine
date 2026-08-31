//! 후보와 후보 식별자.
//!
//! 엔진은 후보가 무엇인지 모른다. 아는 것은 하나뿐이다 -- 후보에게는 순서를 매길 수 있는
//! 식별자가 있다는 것. 그 순서가 결정성 불변식의 바닥이다. 점수가 같을 때 무엇을 앞에
//! 둘지를 식별자가 정하므로, 동점 처리에 임의 순서가 남지 않는다.

use std::fmt;

/// 후보 식별자.
///
/// 두 표현을 갖는 이유는 비용과 표현력이 부딪히기 때문이다. 후보가 1,000만 개인
/// 스트리밍에서는 정수 하나가 맞고, 감사 출력에서 `doc-3141` 같은 이름을 그대로 보여야
/// 하는 자리에서는 문자열이 맞다.
///
/// 정렬 순서는 유도된 순서다 -- [`Num`](Self::Num) 이 전부 [`Text`](Self::Text) 앞에 오고,
/// 같은 변형 안에서는 값 순이다. 둘을 섞어 써도 순서는 여전히 전순서이므로 결정성은
/// 깨지지 않는다. 다만 섞어 쓰면 사람이 읽는 순서와 어긋나므로 한 종류로 통일하는 편이 낫다.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CandidateId {
    /// 정수 식별자. 복제 비용이 없어 스트리밍 경로에 적합하다.
    Num(u64),
    /// 문자열 식별자. 복제할 때마다 할당이 일어난다.
    Text(Box<str>),
}

impl CandidateId {
    /// 정수 식별자를 만든다.
    pub fn num(v: u64) -> Self {
        CandidateId::Num(v)
    }

    /// 문자열 식별자를 만든다.
    pub fn text(v: impl Into<Box<str>>) -> Self {
        CandidateId::Text(v.into())
    }
}

impl fmt::Display for CandidateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CandidateId::Num(v) => write!(f, "{v}"),
            CandidateId::Text(v) => write!(f, "{v}"),
        }
    }
}

impl From<u64> for CandidateId {
    fn from(v: u64) -> Self {
        CandidateId::Num(v)
    }
}

impl From<&str> for CandidateId {
    fn from(v: &str) -> Self {
        CandidateId::Text(v.into())
    }
}

impl From<String> for CandidateId {
    fn from(v: String) -> Self {
        CandidateId::Text(v.into_boxed_str())
    }
}

/// 순위화 대상.
///
/// 사용자가 구현하는 두 트레잇 중 하나다. 다른 하나는 [`Scorer`](crate::Scorer) 다.
///
/// # 계약
///
/// `id` 는 같은 후보에 대해 항상 같은 값을 돌려줘야 하고, 한 실행 안에서 서로 다른 후보가
/// 같은 식별자를 갖지 않아야 한다. 식별자가 겹치면 동점 처리 순서가 정해지지 않아
/// 결정성 불변식이 깨진다. 엔진은 이것을 검증하지 않는다 -- 1,000만 후보에 대해
/// 중복을 검사하려면 후보 전부를 메모리에 올려야 하고, 그것이 이 엔진이 피하려는 바로
/// 그 비용이기 때문이다.
pub trait Candidate {
    /// 이 후보의 식별자.
    fn id(&self) -> CandidateId;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn num_precedes_text_and_orders_within_variant() {
        let mut ids = vec![
            CandidateId::text("b"),
            CandidateId::num(9),
            CandidateId::text("a"),
            CandidateId::num(2),
        ];
        ids.sort();
        assert_eq!(
            ids,
            vec![
                CandidateId::num(2),
                CandidateId::num(9),
                CandidateId::text("a"),
                CandidateId::text("b"),
            ]
        );
    }

    #[test]
    fn display_drops_the_variant() {
        assert_eq!(CandidateId::num(7).to_string(), "7");
        assert_eq!(CandidateId::text("doc-3141").to_string(), "doc-3141");
    }
}
