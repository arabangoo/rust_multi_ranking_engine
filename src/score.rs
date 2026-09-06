//! 채점기, 척도, 점수 집합.
//!
//! 이 모듈이 강제하는 것 하나가 이 엔진의 존재 이유에 가깝다 -- **채점기는 자기 점수가
//! 어떤 척도인지 선언해야 한다.** 로짓과 확률을 섞어 더하는 것은 흔한 실수인데, 엔진은
//! 그것을 문서로 경고하는 대신 [`Error::IncompatibleScale`](crate::Error::IncompatibleScale)
//! 로 거부한다.

use std::fmt;

/// 채점기 식별자.
///
/// 융합 기록·감사 출력·탈락 사유에 그대로 실리므로 사람이 읽을 수 있는 이름을 쓴다.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ScorerId(Box<str>);

impl ScorerId {
    /// 식별자를 만든다.
    pub fn new(name: impl Into<Box<str>>) -> Self {
        ScorerId(name.into())
    }

    /// 이름을 빌려 본다.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ScorerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for ScorerId {
    fn from(v: &str) -> Self {
        ScorerId::new(v)
    }
}

impl From<String> for ScorerId {
    fn from(v: String) -> Self {
        ScorerId::new(v.into_boxed_str())
    }
}

/// 점수의 척도.
///
/// 융합 가능성 판정에 쓰인다. 자세한 조합표는 [`Fusion`](crate::Fusion) 문서에 있다.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ScoreScale {
    /// 0 이상 1 이하. 확률이나 정규화된 값. 서로 더해도 된다.
    Unit,
    /// 로짓이나 거리. 정규화 없이 더하면 틀린다.
    Unbounded,
    /// 순서만 의미가 있다. 값끼리 더할 수 없다.
    Rank,
}

/// 채점기의 상대 비용.
///
/// 캐스케이드 단계 배치에 쓰인다. 싼 축은 후보 전부에 돌고, 비싼 축은 1단계가 남긴
/// 풀에만 돈다. 절대 시간이 아니라 서로에 대한 상대값이므로 두 값이면 충분하다.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ScorerCost {
    /// 후보 전부에 돌려도 되는 축. 1단계에서 실행된다.
    Cheap,
    /// 풀에만 돌려야 하는 축. 2단계에서 실행된다.
    Expensive,
}

/// 점수를 매기는 방법.
///
/// 사용자가 구현하는 두 트레잇 중 하나다. 다른 하나는 [`Candidate`](crate::Candidate) 다.
///
/// # 계약
///
/// `id`·`scale`·`cost` 는 상수여야 한다. 엔진은 등록 시점에 한 번만 읽고 그 뒤로 캐시한다.
///
/// `score` 는 같은 후보에 대해 같은 값을 돌려줘야 한다. 값을 낼 수 없으면 `None` 을
/// 돌려준다 -- **0 점과 다르다.** 모델이 돌지 않았는데 0 점으로 세면 그 후보는 부당하게
/// 죽는다. 결측을 어떻게 다룰지는 [`MissingPolicy`](crate::MissingPolicy) 가 정한다.
///
/// `NaN` 을 돌려주면 엔진은 그것을 `None` 과 같이 다룬다. 순서를 매길 수 없는 값이
/// 힙에 들어가면 결정성이 깨지기 때문이다.
pub trait Scorer<C>: Sync {
    /// 이 채점기의 식별자.
    fn id(&self) -> ScorerId;

    /// 이 점수가 어떤 척도인가. 융합 가능성 판정에 쓰인다.
    fn scale(&self) -> ScoreScale;

    /// 얼마나 비싼가. 캐스케이드 단계 배치에 쓰인다.
    fn cost(&self) -> ScorerCost;

    /// 후보에게 점수를 매긴다. 값을 낼 수 없으면 `None`.
    fn score(&self, c: &C) -> Option<f32>;

    /// 여러 후보를 한 번에 채점한다.
    ///
    /// 기본 구현은 [`score`](Self::score) 를 차례로 부르므로 대개 구현할 필요가 없다.
    /// **비싼 축을 배치 추론으로 돌릴 때만 재정의한다.** 교차 인코더나 재순위 모델은
    /// 질의와 문서 쌍 960개를 한 번에 넣는 편이 하나씩 960번 부르는 것보다 훨씬 빠르다.
    /// 2단계는 유계 힙이 남긴 풀에만 도므로 여기 들어오는 크기가 풀 크기로 묶여 있다.
    ///
    /// # 계약
    ///
    /// **돌려주는 벡터의 길이가 입력 길이와 같아야 하고 순서도 같아야 한다.** 길이가
    /// 어긋나면 엔진이 [`Error::BatchLengthMismatch`](crate::Error::BatchLengthMismatch)
    /// 를 낸다. 순서가 어긋나면 엔진이 알아챌 방법이 없으므로 조용히 틀린 순위가 나온다.
    ///
    /// `parallel` 기능을 켜면 엔진이 풀을 여러 덩어리로 나눠 이 메서드를 동시에 부른다.
    /// 덩어리별 결과를 색인 순서로 다시 모으므로 결정성은 유지된다.
    fn score_batch(&self, candidates: &[&C]) -> Vec<Option<f32>> {
        candidates.iter().map(|c| self.score(c)).collect()
    }
}

/// 후보 하나가 받은 점수 전부. 융합 전 원본을 보관한다.
///
/// 융합 결과만 남기면 "왜 이 순위인가"에 답할 수 없다. 근거 재현 불변식이 이 원본과
/// [`FusionTrace`](crate::FusionTrace) 만으로 최종 점수를 다시 계산할 것을 요구한다.
#[derive(Clone, PartialEq, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ScoreSet {
    values: Vec<(ScorerId, Option<f32>)>,
}

impl ScoreSet {
    /// 등록 순서대로 나열된 (채점기, 점수) 쌍으로 만든다.
    pub fn new(values: Vec<(ScorerId, Option<f32>)>) -> Self {
        ScoreSet { values }
    }

    /// 한 축의 점수를 찾는다. 축이 없으면 `None`, 축은 있는데 값이 없으면 `Some(None)`.
    pub fn get(&self, id: &ScorerId) -> Option<Option<f32>> {
        self.values.iter().find(|(k, _)| k == id).map(|(_, v)| *v)
    }

    /// 등록 순서대로 훑는다.
    pub fn iter(&self) -> impl Iterator<Item = (&ScorerId, Option<f32>)> {
        self.values.iter().map(|(k, v)| (k, *v))
    }

    /// 축의 개수.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// 축이 하나도 없는가.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// 무한 척도를 단위 척도로 옮기는 방법.
///
/// 전부 후보 하나만 보고 계산된다. 스트리밍 중에 관측한 최소·최대로 정규화하면 같은
/// 입력이라도 도착 순서에 따라 값이 달라져 결정성이 깨지므로,
/// [`MinMax`](Self::MinMax) 도 경계를 밖에서 받는다.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Normalizer {
    /// 로지스틱 함수. 실수 전체를 (0, 1) 로 옮긴다.
    Sigmoid,
    /// 0 미만은 0 으로, 1 초과는 1 로 자른다.
    Clamp01,
    /// 선언된 구간을 0 에서 1 로 선형 사상한다. 구간 밖은 잘린다.
    MinMax {
        /// 구간의 아래끝.
        min: f32,
        /// 구간의 위끝. `min` 과 같으면 모든 값이 0 이 된다.
        max: f32,
    },
}

impl Normalizer {
    /// 값 하나를 옮긴다.
    pub fn apply(&self, v: f32) -> f32 {
        match *self {
            Normalizer::Sigmoid => 1.0 / (1.0 + (-v).exp()),
            Normalizer::Clamp01 => v.clamp(0.0, 1.0),
            Normalizer::MinMax { min, max } => {
                if max <= min {
                    0.0
                } else {
                    ((v - min) / (max - min)).clamp(0.0, 1.0)
                }
            }
        }
    }
}

/// 정규화기를 끼운 채점기. [`ScorerExt::normalized`] 가 만든다.
pub struct Normalized<S> {
    inner: S,
    normalizer: Normalizer,
}

impl<C, S: Scorer<C>> Scorer<C> for Normalized<S> {
    fn id(&self) -> ScorerId {
        self.inner.id()
    }

    /// 정규화를 거쳤으므로 언제나 단위 척도다.
    fn scale(&self) -> ScoreScale {
        ScoreScale::Unit
    }

    fn cost(&self) -> ScorerCost {
        self.inner.cost()
    }

    fn score(&self, c: &C) -> Option<f32> {
        self.inner.score(c).map(|v| self.normalizer.apply(v))
    }

    fn score_batch(&self, candidates: &[&C]) -> Vec<Option<f32>> {
        self.inner
            .score_batch(candidates)
            .into_iter()
            .map(|v| v.map(|x| self.normalizer.apply(x)))
            .collect()
    }
}

/// 채점기에 붙는 편의 메서드.
pub trait ScorerExt<C>: Scorer<C> + Sized {
    /// 정규화기를 끼워 단위 척도로 만든다.
    ///
    /// 무한 척도 축을 가중합에 넣으려면 이 단계를 명시적으로 거쳐야 한다. 엔진이
    /// 몰래 정규화해 주지 않는 이유는, 어떤 정규화를 쓸지가 도메인 지식이기 때문이다.
    fn normalized(self, normalizer: Normalizer) -> Normalized<Self> {
        Normalized {
            inner: self,
            normalizer,
        }
    }
}

impl<C, S: Scorer<C>> ScorerExt<C> for S {}

#[cfg(test)]
mod tests {
    use super::*;

    struct Logit(f32);

    impl Scorer<()> for Logit {
        fn id(&self) -> ScorerId {
            ScorerId::new("logit")
        }
        fn scale(&self) -> ScoreScale {
            ScoreScale::Unbounded
        }
        fn cost(&self) -> ScorerCost {
            ScorerCost::Cheap
        }
        fn score(&self, _c: &()) -> Option<f32> {
            Some(self.0)
        }
    }

    #[test]
    fn normalizing_changes_the_declared_scale() {
        let raw = Logit(2.0);
        assert_eq!(Scorer::<()>::scale(&raw), ScoreScale::Unbounded);

        let wrapped = raw.normalized(Normalizer::Sigmoid);
        assert_eq!(Scorer::<()>::scale(&wrapped), ScoreScale::Unit);

        let v = wrapped.score(&()).unwrap();
        assert!((v - 0.880_797).abs() < 1e-5, "sigmoid(2) = {v}");
    }

    #[test]
    fn minmax_with_empty_range_does_not_divide_by_zero() {
        let n = Normalizer::MinMax { min: 1.0, max: 1.0 };
        assert_eq!(n.apply(5.0), 0.0);
    }

    #[test]
    fn score_set_tells_missing_axis_from_missing_value() {
        let s = ScoreSet::new(vec![
            (ScorerId::new("a"), Some(0.5)),
            (ScorerId::new("b"), None),
        ]);
        assert_eq!(s.get(&ScorerId::new("a")), Some(Some(0.5)));
        assert_eq!(s.get(&ScorerId::new("b")), Some(None));
        assert_eq!(s.get(&ScorerId::new("c")), None);
    }
}
