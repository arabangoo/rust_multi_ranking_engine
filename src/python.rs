//! PyO3 바인딩.
//!
//! `feature = "python"` 을 켰을 때만 빌드된다. abi3(안정 이진 인터페이스) 확장 모듈이라
//! Python 3.9 이상에서 휠 하나로 돈다.
//!
//! # 코어를 오염시키지 않는다
//!
//! 코어 타입에 `#[pyclass]` 를 직접 달지 않고 이 모듈 안에 래퍼를 둔다. 그래야 기본
//! 빌드가 PyO3 를 전혀 모르는 순수 러스트로 남는다.
//!
//! # 왜 러스트 API 와 모양이 다른가
//!
//! 러스트 쪽에서 [`Scorer`](crate::Scorer) 는 사용자가 구현하는 트레잇이다. 그대로
//! 옮기면 후보 하나마다, 축마다 파이썬을 다시 불러야 한다. 후보가 100만이면 파이썬
//! 호출이 200만 번이고, 전역 인터프리터 잠금(GIL, Global Interpreter Lock) 때문에
//! 병렬 채점도 무의미해진다. 캐스케이드가 아껴 둔 이득이 그 자리에서 사라진다.
//!
//! 그래서 표면을 둘로 가른다.
//!
//! - **싼 축은 데이터로 받는다.** 후보마다 점수를 미리 담아 넘긴다. 벡터 검색 결과가
//!   원래 그 모양이라 억지가 아니다. 1단계 스트리밍에 파이썬이 끼어들지 않는다.
//! - **비싼 축만 파이썬 함수로 받는다.** 2단계 풀에만 도는데다
//!   [`Scorer::score_batch`](crate::Scorer::score_batch) 로 넘어가므로 **풀 전체가 한 번의
//!   호출**로 처리된다. 교차 인코더처럼 배치 추론을 쓰는 모델이 원하는 모양 그대로다.
//!
//! ```python
//! import rust_multi_ranking_engine as rmre
//!
//! engine = rmre.Engine()
//! engine.scorer("similarity")
//! engine.scorer("cross", scale="unbounded", cost="expensive",
//!               normalize="sigmoid", fn=lambda pool: model.score(pool))
//! engine.fuse("weighted_sum", weights={"similarity": 0.4, "cross": 0.6})
//! engine.max_per_group("source", 2)
//! engine.budget_tokens(900)
//!
//! out = engine.run([
//!     {"id": "doc-1", "scores": {"similarity": 0.94},
//!      "groups": {"source": "manual"}, "cost": 220},
//! ])
//!
//! for r in out.ranked:
//!     print(r.rank, r.id, r.fused)
//! ```
//!
//! # 점수는 32비트 실수다
//!
//! 코어가 `f32` 로 계산하므로 파이썬으로 넘어온 값은 64비트로 넓혀진 32비트 값이다.
//! `0.4` 를 넣으면 `0.4000000059604645` 가 돌아온다. 순위와 비교는 그대로 성립하지만
//! **값을 등호로 견주지 말고 허용 오차를 두고 비교한다.**

use std::sync::Mutex;

use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule, PySequence};

use crate::budget::Budget;
use crate::candidate::{Candidate, CandidateId};
use crate::constraint::{self, ConstraintId, Requirement, SetConstraint, UnaryConstraint};
use crate::engine::{Engine as CoreEngine, DEFAULT_POOL_MULTIPLIER};
use crate::error::Error;
use crate::evidence::{RejectReason, Rejections};
use crate::fuse::{Fusion, FusionInput, MissingPolicy, DEFAULT_RRF_K};
use crate::objective::Coverage;
use crate::score::{Normalizer, ScoreScale, Scorer, ScorerCost, ScorerId};

create_exception!(
    rust_multi_ranking_engine,
    EngineError,
    PyException,
    "엔진 설정이 성립하지 않거나 요구 조건을 채울 수 없을 때 난다."
);

fn to_py_err(e: Error) -> PyErr {
    match e {
        Error::InfeasibleRequirement { .. } | Error::BatchLengthMismatch { .. } => {
            EngineError::new_err(e.to_string())
        }
        other => PyValueError::new_err(other.to_string()),
    }
}

// ── 후보 ──────────────────────────────────────────────────────────

/// 파이썬에서 넘어온 후보 하나.
///
/// 원본 파이썬 객체를 함께 들고 있다. 비싼 축 콜백이 점수를 매기려면 식별자만으로는
/// 모자라기 때문이다. 1단계 유계 힙이 풀 크기만 살려 두므로 동시에 살아 있는 참조는
/// `M` 개를 넘지 않는다.
struct PyCandidate {
    id: CandidateId,
    scores: Vec<Option<f32>>,
    groups: Vec<Option<Box<str>>>,
    cost: u32,
    cover: Vec<Box<str>>,
    raw: Py<PyAny>,
}

impl Candidate for PyCandidate {
    fn id(&self) -> CandidateId {
        self.id.clone()
    }
}

// ── 채점기 ────────────────────────────────────────────────────────

/// 미리 계산돼 넘어온 점수를 읽기만 하는 축.
struct IndexScorer {
    id: ScorerId,
    scale: ScoreScale,
    cost: ScorerCost,
    index: usize,
    normalizer: Option<Normalizer>,
}

impl Scorer<PyCandidate> for IndexScorer {
    fn id(&self) -> ScorerId {
        self.id.clone()
    }
    fn scale(&self) -> ScoreScale {
        // 정규화기를 끼웠으면 단위 척도가 된다. 러스트 쪽 Normalized 래퍼와 같은 규칙이다.
        if self.normalizer.is_some() {
            ScoreScale::Unit
        } else {
            self.scale
        }
    }
    fn cost(&self) -> ScorerCost {
        self.cost
    }
    fn score(&self, c: &PyCandidate) -> Option<f32> {
        let raw = c.scores[self.index]?;
        Some(match self.normalizer {
            Some(n) => n.apply(raw),
            None => raw,
        })
    }
}

/// 파이썬 함수를 부르는 비싼 축.
///
/// 풀 전체를 한 번에 넘긴다. 파이썬 쪽 함수는 후보 목록을 받아 같은 길이의 점수 목록을
/// 돌려줘야 한다. 값을 낼 수 없는 자리에는 `None` 을 넣는다.
struct CallbackScorer {
    id: ScorerId,
    scale: ScoreScale,
    normalizer: Option<Normalizer>,
    func: Py<PyAny>,
    /// 콜백 안에서 난 파이썬 예외. 트레잇이 결과 타입을 못 돌려주므로 여기 담아 두고
    /// 실행이 끝난 뒤 그대로 다시 올린다. 삼키지 않는다.
    failure: Mutex<Option<PyErr>>,
}

impl CallbackScorer {
    fn call(&self, candidates: &[&PyCandidate]) -> PyResult<Vec<Option<f32>>> {
        Python::attach(|py| {
            let items = PyList::new(py, candidates.iter().map(|c| c.raw.bind(py)))?;
            let out = self.func.call1(py, (items,))?;
            let seq = out.bind(py);
            let seq: &Bound<'_, PySequence> = seq.cast::<PySequence>().map_err(|_| {
                PyValueError::new_err(format!(
                    "채점기 '{}' 의 콜백이 목록이 아닌 것을 돌려줬다",
                    self.id
                ))
            })?;

            let n = seq.len()?;
            let mut values = Vec::with_capacity(n);
            for i in 0..n {
                let item = seq.get_item(i)?;
                if item.is_none() {
                    values.push(None);
                } else {
                    values.push(Some(item.extract::<f32>()?));
                }
            }
            Ok(values)
        })
    }
}

impl Scorer<PyCandidate> for CallbackScorer {
    fn id(&self) -> ScorerId {
        self.id.clone()
    }
    fn scale(&self) -> ScoreScale {
        if self.normalizer.is_some() {
            ScoreScale::Unit
        } else {
            self.scale
        }
    }
    fn cost(&self) -> ScorerCost {
        ScorerCost::Expensive
    }

    fn score(&self, c: &PyCandidate) -> Option<f32> {
        self.score_batch(&[c]).into_iter().next().flatten()
    }

    fn score_batch(&self, candidates: &[&PyCandidate]) -> Vec<Option<f32>> {
        match self.call(candidates) {
            Ok(values) => match self.normalizer {
                Some(n) => values.into_iter().map(|v| v.map(|x| n.apply(x))).collect(),
                None => values,
            },
            Err(e) => {
                // 첫 실패만 남긴다. 길이를 맞춰 돌려주면 엔진이 결측으로 다루지만,
                // 실행이 끝난 뒤 이 예외가 그대로 다시 올라간다.
                let mut slot = self.failure.lock().expect("잠금이 깨지지 않는다");
                if slot.is_none() {
                    *slot = Some(e);
                }
                vec![None; candidates.len()]
            }
        }
    }
}

// ── 제약 ──────────────────────────────────────────────────────────

/// 후보의 그룹 칸 하나를 기준으로 세는 상한.
struct GroupLimit {
    id: ConstraintId,
    slot: usize,
    max: usize,
}

impl SetConstraint<PyCandidate> for GroupLimit {
    fn id(&self) -> ConstraintId {
        self.id.clone()
    }
    fn admits(&self, selected: &[&PyCandidate], candidate: &PyCandidate) -> bool {
        if self.max == 0 {
            return false;
        }
        let Some(target) = candidate.groups[self.slot].as_deref() else {
            // 그룹 값이 없는 후보는 이 상한의 대상이 아니다.
            return true;
        };
        let mut count = 0usize;
        for s in selected {
            if s.groups[self.slot].as_deref() == Some(target) {
                count += 1;
                if count >= self.max {
                    return false;
                }
            }
        }
        true
    }
    fn is_matroid(&self) -> bool {
        true
    }
}

/// 한 축의 점수가 문턱 이상인 후보만 통과시키는 단항 제약.
struct MinAxis {
    id: ConstraintId,
    index: usize,
    min: f32,
}

impl UnaryConstraint<PyCandidate> for MinAxis {
    fn id(&self) -> ConstraintId {
        self.id.clone()
    }
    fn allows(&self, c: &PyCandidate) -> bool {
        match c.scores[self.index] {
            Some(v) => v >= self.min,
            // 값이 없으면 통과시키지 않는다. 문턱을 걸었다는 것은 그 축을 본다는 뜻이다.
            None => false,
        }
    }
}

// ── 설정 ──────────────────────────────────────────────────────────

struct AxisSpec {
    name: String,
    scale: ScoreScale,
    cost: ScorerCost,
    normalizer: Option<Normalizer>,
    func: Option<Py<PyAny>>,
}

#[derive(Clone)]
struct GroupSpec {
    key: String,
    id: String,
    max: usize,
}

#[derive(Clone)]
struct RequireSpec {
    key: String,
    value: String,
    n: usize,
    id: String,
}

fn parse_scale(s: &str) -> PyResult<ScoreScale> {
    match s {
        "unit" => Ok(ScoreScale::Unit),
        "unbounded" => Ok(ScoreScale::Unbounded),
        "rank" => Ok(ScoreScale::Rank),
        other => Err(PyValueError::new_err(format!(
            "척도는 'unit', 'unbounded', 'rank' 중 하나여야 한다. 받은 값 '{other}'"
        ))),
    }
}

fn parse_cost(s: &str) -> PyResult<ScorerCost> {
    match s {
        "cheap" => Ok(ScorerCost::Cheap),
        "expensive" => Ok(ScorerCost::Expensive),
        other => Err(PyValueError::new_err(format!(
            "비용은 'cheap' 또는 'expensive' 여야 한다. 받은 값 '{other}'"
        ))),
    }
}

/// 가중치를 사전으로도 목록으로도 받는다. 파이썬에서 자연스러운 쪽은 사전이다.
fn parse_weights(weights: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<(ScorerId, f32)>> {
    let Some(w) = weights else {
        return Ok(Vec::new());
    };
    if let Ok(map) = w.cast::<PyDict>() {
        let mut out = Vec::with_capacity(map.len());
        for (k, v) in map.iter() {
            out.push((
                ScorerId::new(k.str()?.extract::<String>()?),
                v.extract::<f32>()?,
            ));
        }
        return Ok(out);
    }
    let pairs: Vec<(String, f32)> = w.extract().map_err(|_| {
        PyValueError::new_err(
            "weights 는 {'축이름': 가중치} 사전이거나 [('축이름', 가중치)] 목록이어야 한다",
        )
    })?;
    Ok(pairs
        .into_iter()
        .map(|(n, v)| (ScorerId::new(n), v))
        .collect())
}

fn parse_normalizer(name: &str, range: Option<(f32, f32)>) -> PyResult<Normalizer> {
    match name {
        "sigmoid" => Ok(Normalizer::Sigmoid),
        "clamp01" => Ok(Normalizer::Clamp01),
        "minmax" => match range {
            Some((min, max)) => Ok(Normalizer::MinMax { min, max }),
            None => Err(PyValueError::new_err(
                "normalize='minmax' 는 normalize_range=(min, max) 를 함께 받아야 한다",
            )),
        },
        other => Err(PyValueError::new_err(format!(
            "정규화기는 'sigmoid', 'clamp01', 'minmax' 중 하나여야 한다. 받은 값 '{other}'"
        ))),
    }
}

// ── 엔진 ──────────────────────────────────────────────────────────

/// 다축 점수 융합과 제약 아래 상위 K 선택 엔진.
///
/// 러스트 쪽 빌더와 달리 메서드가 자기 자신을 바꾼다. 파이썬에서 더 읽기 쉬운 모양이다.
#[pyclass(name = "Engine", module = "rust_multi_ranking_engine")]
pub struct PyEngine {
    axes: Vec<AxisSpec>,
    groups: Vec<GroupSpec>,
    requires: Vec<RequireSpec>,
    unary_min: Vec<(String, String, f32)>,
    max_total: Option<(String, usize)>,
    fusion: Fusion,
    missing: MissingPolicy,
    budget: Budget,
    coverage: bool,
    admission: Option<String>,
    pool_multiplier: u32,
    threshold: Option<f32>,
    rejections: Rejections,
    min_fit: f32,
}

#[pymethods]
impl PyEngine {
    #[new]
    fn new() -> Self {
        PyEngine {
            axes: Vec::new(),
            groups: Vec::new(),
            requires: Vec::new(),
            unary_min: Vec::new(),
            max_total: None,
            fusion: Fusion::default(),
            missing: MissingPolicy::default(),
            budget: Budget::default(),
            coverage: false,
            admission: None,
            pool_multiplier: DEFAULT_POOL_MULTIPLIER,
            threshold: None,
            rejections: Rejections::default(),
            min_fit: crate::budget::DEFAULT_MIN_FIT,
        }
    }

    /// 점수 축을 등록한다. 등록 순서가 감사 출력의 축 순서다.
    ///
    /// `cost="expensive"` 이면 `fn` 을 함께 줄 수 있다. 그 함수는 **2단계 풀 전체를 한 번에**
    /// 받아 같은 길이의 점수 목록을 돌려줘야 한다. `fn` 없이 비싼 축으로 두면 후보에
    /// 담겨 온 값을 그대로 읽는다.
    #[pyo3(signature = (name, *, scale="unit", cost="cheap", normalize=None, normalize_range=None, r#fn=None))]
    #[allow(clippy::too_many_arguments)]
    fn scorer(
        &mut self,
        name: &str,
        scale: &str,
        cost: &str,
        normalize: Option<&str>,
        normalize_range: Option<(f32, f32)>,
        r#fn: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        let cost = parse_cost(cost)?;
        if r#fn.is_some() && cost != ScorerCost::Expensive {
            return Err(PyValueError::new_err(
                "fn 을 주는 축은 cost='expensive' 여야 한다. 싼 축은 후보 전부에 도므로 \
                 파이썬을 부르면 스트리밍의 뜻이 사라진다",
            ));
        }
        let normalizer = match normalize {
            Some(n) => Some(parse_normalizer(n, normalize_range)?),
            None => None,
        };
        self.axes.push(AxisSpec {
            name: name.to_string(),
            scale: parse_scale(scale)?,
            cost,
            normalizer,
            func: r#fn,
        });
        Ok(())
    }

    /// 융합 방식을 고른다. `"rrf"`, `"weighted_sum"`, `"max"` 중 하나.
    ///
    /// `weights` 는 `{"축이름": 가중치}` 사전이거나 `[("축이름", 가중치)]` 목록이다.
    /// 주지 않으면 모든 축이 1.0 이다.
    #[pyo3(signature = (method, *, k=None, weights=None))]
    fn fuse(
        &mut self,
        method: &str,
        k: Option<f32>,
        weights: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        self.fusion = match method {
            "rrf" => Fusion::Rrf {
                k: k.unwrap_or(DEFAULT_RRF_K),
            },
            "weighted_sum" => Fusion::WeightedSum {
                weights: parse_weights(weights)?,
            },
            "max" => Fusion::Max,
            other => {
                return Err(PyValueError::new_err(format!(
                    "융합은 'rrf', 'weighted_sum', 'max' 중 하나여야 한다. 받은 값 '{other}'"
                )))
            }
        };
        Ok(())
    }

    /// 값이 없는 축을 어떻게 다룰지 고른다. `"skip"`, `"impute"`, `"reject"`.
    #[pyo3(signature = (policy, *, value=None))]
    fn missing(&mut self, policy: &str, value: Option<f32>) -> PyResult<()> {
        self.missing = match policy {
            "skip" => MissingPolicy::Skip,
            "reject" => MissingPolicy::Reject,
            "impute" => MissingPolicy::Impute(value.ok_or_else(|| {
                PyValueError::new_err("missing='impute' 는 value 를 함께 받아야 한다")
            })?),
            other => {
                return Err(PyValueError::new_err(format!(
                    "결측 정책은 'skip', 'impute', 'reject' 중 하나여야 한다. 받은 값 '{other}'"
                )))
            }
        };
        Ok(())
    }

    /// 후보의 `groups[key]` 값마다 최대 `max` 개까지만 고른다. 분할 매트로이드다.
    #[pyo3(signature = (key, max, *, id=None))]
    fn max_per_group(&mut self, key: &str, max: usize, id: Option<&str>) {
        self.groups.push(GroupSpec {
            key: key.to_string(),
            id: id.unwrap_or(key).to_string(),
            max,
        });
    }

    /// 전체 상한. 균일 매트로이드다.
    #[pyo3(signature = (max, *, id=None))]
    fn max_total(&mut self, max: usize, id: Option<&str>) {
        self.max_total = Some((id.unwrap_or("max_total").to_string(), max));
    }

    /// `groups[key] == value` 인 후보가 최소 `n` 개 들어가야 한다는 하한 조건.
    ///
    /// 채우려고 교체가 일어나면 결과의 `exact` 가 거짓이 된다. 탐욕이 만든 순서를 사후에
    /// 뒤집기 때문이다.
    #[pyo3(signature = (key, value, n, *, id=None))]
    fn require_at_least(&mut self, key: &str, value: &str, n: usize, id: Option<&str>) {
        self.requires.push(RequireSpec {
            key: key.to_string(),
            value: value.to_string(),
            n,
            id: id.unwrap_or(key).to_string(),
        });
    }

    /// 한 축의 점수가 문턱 미만인 후보를 1단계에서 걸러 낸다.
    #[pyo3(signature = (axis, min, *, id=None))]
    fn unary_min(&mut self, axis: &str, min: f32, id: Option<&str>) {
        self.unary_min
            .push((id.unwrap_or(axis).to_string(), axis.to_string(), min));
    }

    /// 후보의 `cover` 목록으로 포괄성을 최대화한다. 서브모듈러다.
    fn coverage_objective(&mut self) {
        self.coverage = true;
    }

    /// 고정 개수를 고른다.
    fn budget_top_k(&mut self, k: u32) {
        self.budget = Budget::TopK(k);
    }

    /// 누락 허용 질량으로 K 를 유도한다. 적합이 나쁘면 `fallback_k` 로 되돌린다.
    fn budget_tail_mass(&mut self, epsilon: f32, fallback_k: u32) {
        self.budget = Budget::tail_mass(epsilon, fallback_k);
    }

    /// 후보의 `cost` 합이 상한 이하가 되도록 고른다. 배낭형이라 근사다.
    fn budget_tokens(&mut self, max: u32) {
        self.budget = Budget::Tokens { max };
    }

    /// 1단계 절단에 쓸 승인 채점기를 지정한다.
    fn admission(&mut self, axis: &str) {
        self.admission = Some(axis.to_string());
    }

    /// 1단계 풀 배수. `M = K * multiplier` 다.
    fn pool_multiplier(&mut self, multiplier: u32) {
        self.pool_multiplier = multiplier;
    }

    /// 융합 점수의 하한.
    fn threshold(&mut self, value: f32) {
        self.threshold = Some(value);
    }

    /// 꼬리 질량 예산의 적합도 문턱.
    fn min_fit(&mut self, value: f32) {
        self.min_fit = value;
    }

    /// 탈락 후보를 얼마나 보관할지. `"keep"`, `"count"`, `"sample"`.
    ///
    /// 개수는 이 정책과 무관하게 언제나 정확하다. 후보가 아주 많으면 `"count"` 를 쓴다.
    #[pyo3(signature = (policy, *, n=None))]
    fn rejections(&mut self, policy: &str, n: Option<usize>) -> PyResult<()> {
        self.rejections = match policy {
            "keep" => Rejections::Keep,
            "count" => Rejections::Count,
            "sample" => Rejections::Sample(n.ok_or_else(|| {
                PyValueError::new_err("rejections='sample' 은 n 을 함께 받아야 한다")
            })?),
            other => {
                return Err(PyValueError::new_err(format!(
                    "보관 정책은 'keep', 'count', 'sample' 중 하나여야 한다. 받은 값 '{other}'"
                )))
            }
        };
        Ok(())
    }

    /// 후보를 훑어 고른다.
    ///
    /// `candidates` 는 반복 가능한 것이면 무엇이든 된다. 목록을 통째로 만들지 않고
    /// 생성기(generator)를 넘기면 1단계가 스트리밍으로 처리한다.
    fn run(&self, py: Python<'_>, candidates: &Bound<'_, PyAny>) -> PyResult<PyOutcome> {
        self.build_and_run(py, candidates)
    }

    fn __repr__(&self) -> String {
        format!(
            "Engine(axes={}, fusion={}, budget={:?})",
            self.axes.len(),
            self.fusion.name(),
            self.budget
        )
    }
}

impl PyEngine {
    fn axis_index(&self, name: &str) -> PyResult<usize> {
        self.axes
            .iter()
            .position(|a| a.name == name)
            .ok_or_else(|| PyValueError::new_err(format!("등록되지 않은 축 '{name}'")))
    }

    fn build_and_run(&self, py: Python<'_>, candidates: &Bound<'_, PyAny>) -> PyResult<PyOutcome> {
        if self.axes.is_empty() {
            return Err(PyValueError::new_err(
                "축이 하나도 없다. Engine.scorer 로 최소 하나를 등록해야 한다",
            ));
        }

        let group_keys: Vec<String> = self
            .groups
            .iter()
            .map(|g| g.key.clone())
            .chain(self.requires.iter().map(|r| r.key.clone()))
            .collect();

        let mut engine: CoreEngine<PyCandidate> = CoreEngine::new()
            .fuse(self.fusion.clone())
            .missing(self.missing)
            .budget(self.budget)
            .pool_multiplier(self.pool_multiplier)
            .rejections(self.rejections)
            .min_fit(self.min_fit);

        if let Some(t) = self.threshold {
            engine = engine.threshold(t);
        }
        if let Some(a) = &self.admission {
            engine = engine.admission(a.as_str());
        }
        if self.budget.is_knapsack() {
            engine = engine.cost(|c: &PyCandidate| c.cost);
        }
        if self.coverage {
            engine = engine.objective(Coverage::new(|c: &PyCandidate| c.cover.clone()));
        }

        // 콜백 축의 실패를 실행 뒤에 다시 올리려면 참조를 들고 있어야 한다.
        let mut callbacks: Vec<std::sync::Arc<CallbackScorer>> = Vec::new();

        for (index, axis) in self.axes.iter().enumerate() {
            match &axis.func {
                Some(f) => {
                    let scorer = std::sync::Arc::new(CallbackScorer {
                        id: ScorerId::new(axis.name.clone()),
                        scale: axis.scale,
                        normalizer: axis.normalizer,
                        func: f.clone_ref(py),
                        failure: Mutex::new(None),
                    });
                    callbacks.push(std::sync::Arc::clone(&scorer));
                    engine = engine.scorer(SharedCallback(scorer));
                }
                None => {
                    engine = engine.scorer(IndexScorer {
                        id: ScorerId::new(axis.name.clone()),
                        scale: axis.scale,
                        cost: axis.cost,
                        index,
                        normalizer: axis.normalizer,
                    });
                }
            }
        }

        for (id, axis, min) in &self.unary_min {
            engine = engine.unary(MinAxis {
                id: ConstraintId::new(id.clone()),
                index: self.axis_index(axis)?,
                min: *min,
            });
        }
        for (slot, g) in self.groups.iter().enumerate() {
            engine = engine.set_constraint(GroupLimit {
                id: ConstraintId::new(g.id.clone()),
                slot,
                max: g.max,
            });
        }
        if let Some((id, max)) = &self.max_total {
            engine = engine.set_constraint(constraint::max_total(id.as_str(), *max));
        }
        for (offset, r) in self.requires.iter().enumerate() {
            let slot = self.groups.len() + offset;
            let want = r.value.clone();
            engine = engine.require(Requirement::at_least(
                r.id.as_str(),
                r.n,
                move |c: &PyCandidate| c.groups[slot].as_deref() == Some(want.as_str()),
            ));
        }

        engine.validate().map_err(to_py_err)?;

        // 후보를 반복자로 흘려보낸다. 목록으로 만들어 두지 않는 것이 이 엔진의 요구다.
        let mut reader = Reader {
            axes: &self.axes,
            group_keys: &group_keys,
            want_cost: self.budget.is_knapsack(),
            want_cover: self.coverage,
            error: None,
        };
        let iter = candidates.try_iter()?;
        let stream = iter.filter_map(|item| match item {
            Ok(obj) => match reader.read(&obj) {
                Ok(c) => Some(c),
                Err(e) => {
                    reader.error.get_or_insert(e);
                    None
                }
            },
            Err(e) => {
                reader.error.get_or_insert(e);
                None
            }
        });

        let outcome = engine.run(stream).map_err(to_py_err)?;

        // 읽기 오류와 콜백 예외는 삼키지 않는다. 결과보다 먼저 올린다.
        if let Some(e) = reader.error {
            return Err(e);
        }
        for cb in &callbacks {
            let mut slot = cb.failure.lock().expect("잠금이 깨지지 않는다");
            if let Some(e) = slot.take() {
                return Err(e);
            }
        }

        PyOutcome::build(py, outcome, &self.axes)
    }
}

/// `Arc` 로 공유되는 콜백 채점기. 실행 뒤 예외를 꺼내 보려고 밖에서도 참조를 든다.
struct SharedCallback(std::sync::Arc<CallbackScorer>);

impl Scorer<PyCandidate> for SharedCallback {
    fn id(&self) -> ScorerId {
        self.0.id()
    }
    fn scale(&self) -> ScoreScale {
        Scorer::<PyCandidate>::scale(&*self.0)
    }
    fn cost(&self) -> ScorerCost {
        ScorerCost::Expensive
    }
    fn score(&self, c: &PyCandidate) -> Option<f32> {
        self.0.score(c)
    }
    fn score_batch(&self, cs: &[&PyCandidate]) -> Vec<Option<f32>> {
        self.0.score_batch(cs)
    }
}

// ── 후보 읽기 ─────────────────────────────────────────────────────

struct Reader<'a> {
    axes: &'a [AxisSpec],
    group_keys: &'a [String],
    want_cost: bool,
    want_cover: bool,
    error: Option<PyErr>,
}

impl Reader<'_> {
    fn read(&self, obj: &Bound<'_, PyAny>) -> PyResult<PyCandidate> {
        let dict: &Bound<'_, PyDict> = obj.cast::<PyDict>().map_err(|_| {
            PyValueError::new_err("후보는 사전(dict)이어야 한다. 'id' 와 'scores' 가 필요하다")
        })?;

        let id_obj = dict
            .get_item("id")?
            .ok_or_else(|| PyValueError::new_err("후보에 'id' 가 없다"))?;
        let id = if let Ok(n) = id_obj.extract::<u64>() {
            CandidateId::Num(n)
        } else {
            CandidateId::text(id_obj.str()?.extract::<String>()?)
        };

        let mut scores = vec![None; self.axes.len()];
        if let Some(raw) = dict.get_item("scores")? {
            if let Ok(map) = raw.cast::<PyDict>() {
                for (i, axis) in self.axes.iter().enumerate() {
                    if let Some(v) = map.get_item(axis.name.as_str())? {
                        if !v.is_none() {
                            scores[i] = Some(v.extract::<f32>()?);
                        }
                    }
                }
            } else {
                let seq: &Bound<'_, PySequence> = raw.cast::<PySequence>().map_err(|_| {
                    PyValueError::new_err("'scores' 는 사전이거나 순서열이어야 한다")
                })?;
                let n = seq.len()?.min(self.axes.len());
                for (i, slot) in scores.iter_mut().enumerate().take(n) {
                    let v = seq.get_item(i)?;
                    if !v.is_none() {
                        *slot = Some(v.extract::<f32>()?);
                    }
                }
            }
        }

        let mut groups = vec![None; self.group_keys.len()];
        if !self.group_keys.is_empty() {
            if let Some(raw) = dict.get_item("groups")? {
                let map: &Bound<'_, PyDict> = raw
                    .cast::<PyDict>()
                    .map_err(|_| PyValueError::new_err("'groups' 는 사전이어야 한다"))?;
                for (i, key) in self.group_keys.iter().enumerate() {
                    if let Some(v) = map.get_item(key.as_str())? {
                        if !v.is_none() {
                            groups[i] = Some(v.str()?.extract::<String>()?.into_boxed_str());
                        }
                    }
                }
            }
        }

        let cost = if self.want_cost {
            match dict.get_item("cost")? {
                Some(v) if !v.is_none() => v.extract::<u32>()?,
                _ => 0,
            }
        } else {
            0
        };

        let cover = if self.want_cover {
            match dict.get_item("cover")? {
                Some(v) if !v.is_none() => {
                    let seq: &Bound<'_, PySequence> = v
                        .cast::<PySequence>()
                        .map_err(|_| PyValueError::new_err("'cover' 는 순서열이어야 한다"))?;
                    let n = seq.len()?;
                    let mut out = Vec::with_capacity(n);
                    for i in 0..n {
                        out.push(
                            seq.get_item(i)?
                                .str()?
                                .extract::<String>()?
                                .into_boxed_str(),
                        );
                    }
                    out
                }
                _ => Vec::new(),
            }
        } else {
            Vec::new()
        };

        Ok(PyCandidate {
            id,
            scores,
            groups,
            cost,
            cover,
            raw: obj.clone().unbind(),
        })
    }
}

// ── 결과 ──────────────────────────────────────────────────────────

/// 골라진 후보 하나.
#[pyclass(name = "Ranked", module = "rust_multi_ranking_engine", frozen)]
pub struct PyRanked {
    /// 후보 식별자.
    #[pyo3(get)]
    id: String,
    /// 1 부터 시작하는 순위.
    #[pyo3(get)]
    rank: u32,
    /// 융합 점수.
    #[pyo3(get)]
    fused: f32,
    /// 융합 전 원본 점수. 값이 없던 축은 `None` 이다.
    #[pyo3(get)]
    scores: Py<PyDict>,
    /// 어떻게 합쳤는가.
    #[pyo3(get)]
    fusion: Py<PyDict>,
    /// 이 후보가 통과한 집합 제약들.
    #[pyo3(get)]
    constraints: Vec<String>,
    /// 넘겼던 파이썬 객체 그대로.
    #[pyo3(get)]
    candidate: Py<PyAny>,
    json: String,
}

#[pymethods]
impl PyRanked {
    /// 감사 출력용 JSON 한 덩어리.
    fn to_json(&self) -> &str {
        &self.json
    }

    fn __repr__(&self) -> String {
        format!(
            "Ranked(rank={}, id={}, fused={})",
            self.rank, self.id, self.fused
        )
    }
}

/// 떨어진 후보 하나.
#[pyclass(name = "Rejected", module = "rust_multi_ranking_engine", frozen)]
pub struct PyRejected {
    /// 후보 식별자.
    #[pyo3(get)]
    id: String,
    /// 사유의 종류. `not_scored`, `unary_constraint`, `set_constraint`,
    /// `below_threshold`, `outranked`, `out_of_pool` 중 하나.
    #[pyo3(get)]
    reason: String,
    /// 사유가 가리키는 축이나 제약의 이름. 없으면 `None`.
    #[pyo3(get)]
    detail: Option<String>,
    /// 융합까지 갔다면 그 점수.
    #[pyo3(get)]
    fused: Option<f32>,
    /// 넘겼던 파이썬 객체 그대로.
    #[pyo3(get)]
    candidate: Py<PyAny>,
}

#[pymethods]
impl PyRejected {
    fn __repr__(&self) -> String {
        format!("Rejected(id={}, reason={})", self.id, self.reason)
    }
}

/// 실행 결과 전부.
#[pyclass(name = "Outcome", module = "rust_multi_ranking_engine", frozen)]
pub struct PyOutcome {
    /// 골라진 것들. 순위 오름차순이다.
    #[pyo3(get)]
    ranked: Vec<Py<PyRanked>>,
    /// 떨어진 것들. 보관 정책이 정한 만큼만 들어 있다.
    #[pyo3(get)]
    rejected: Vec<Py<PyRejected>>,
    /// 사유별 탈락 개수. 보관 정책과 무관하게 언제나 정확하다.
    #[pyo3(get)]
    rejected_counts: Py<PyDict>,
    /// 선택의 성질. `exact`, `guarantee`, `pool_size`, `pool_exhausted`, `cut_margin`.
    #[pyo3(get)]
    selection: Py<PyDict>,
    /// 실행 기록. 입력 수, 풀 크기, 승인 채점기, 채점기별 호출 횟수, 예산 근거.
    #[pyo3(get)]
    trace: Py<PyDict>,
}

#[pymethods]
impl PyOutcome {
    /// 완전성 불변식. 결과 수와 탈락 수의 합이 입력 수와 같은가.
    fn is_complete(&self, py: Python<'_>) -> PyResult<bool> {
        let counts = self.rejected_counts.bind(py);
        let mut total = 0u64;
        for (_, v) in counts.iter() {
            total += v.extract::<u64>()?;
        }
        let input: u64 = self
            .trace
            .bind(py)
            .get_item("input_count")?
            .ok_or_else(|| EngineError::new_err("기록에 input_count 가 없다"))?
            .extract()?;
        Ok(self.ranked.len() as u64 + total == input)
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "Outcome(ranked={}, rejected={}, exact={:?})",
            self.ranked.len(),
            self.rejected.len(),
            self.selection
                .bind(py)
                .get_item("exact")
                .ok()
                .flatten()
                .and_then(|v| v.extract::<bool>().ok())
        )
    }
}

impl PyOutcome {
    fn build(
        py: Python<'_>,
        out: crate::evidence::Outcome<PyCandidate>,
        axes: &[AxisSpec],
    ) -> PyResult<Self> {
        let mut ranked = Vec::with_capacity(out.ranked.len());
        for r in out.ranked {
            let id = r.candidate.id();
            let json = r.to_json(&id);

            let scores = PyDict::new(py);
            for (name, value) in r.scores.iter() {
                scores.set_item(name.as_str(), value)?;
            }

            let fusion = PyDict::new(py);
            fusion.set_item("method", r.fusion.method.name())?;
            if let Some(k) = r.fusion.k {
                fusion.set_item("k", k)?;
            }
            fusion.set_item("missing_policy", missing_name(r.fusion.missing))?;
            let terms = PyList::empty(py);
            for t in &r.fusion.terms {
                let d = PyDict::new(py);
                d.set_item("scorer", t.scorer.as_str())?;
                let (kind, value) = match t.input {
                    FusionInput::Value(v) => ("value", Some(v)),
                    FusionInput::Rank(n) => ("rank", Some(n as f32)),
                    FusionInput::Imputed(v) => ("imputed", Some(v)),
                    FusionInput::Skipped => ("skipped", None),
                };
                d.set_item("input", kind)?;
                d.set_item("value", value)?;
                d.set_item("weight", t.weight)?;
                d.set_item("contribution", t.contribution)?;
                terms.append(d)?;
            }
            fusion.set_item("terms", terms)?;

            ranked.push(Py::new(
                py,
                PyRanked {
                    id: id.to_string(),
                    rank: r.rank,
                    fused: r.fused,
                    scores: scores.unbind(),
                    fusion: fusion.unbind(),
                    constraints: r.constraints.iter().map(|c| c.to_string()).collect(),
                    candidate: r.candidate.raw,
                    json,
                },
            )?);
        }

        let mut rejected = Vec::with_capacity(out.rejected.len());
        for r in out.rejected {
            let detail = match &r.reason {
                RejectReason::NotScored(s) => Some(s.to_string()),
                RejectReason::UnaryConstraint(c) | RejectReason::SetConstraint(c) => {
                    Some(c.to_string())
                }
                _ => None,
            };
            rejected.push(Py::new(
                py,
                PyRejected {
                    id: r.candidate.id().to_string(),
                    reason: r.reason.kind().to_string(),
                    detail,
                    fused: r.fused,
                    candidate: r.candidate.raw,
                },
            )?);
        }

        let counts = PyDict::new(py);
        counts.set_item("not_scored", out.rejected_counts.not_scored)?;
        counts.set_item("unary_constraint", out.rejected_counts.unary_constraint)?;
        counts.set_item("set_constraint", out.rejected_counts.set_constraint)?;
        counts.set_item("below_threshold", out.rejected_counts.below_threshold)?;
        counts.set_item("outranked", out.rejected_counts.outranked)?;
        counts.set_item("out_of_pool", out.rejected_counts.out_of_pool)?;

        let selection = PyDict::new(py);
        selection.set_item("exact", out.selection.exact)?;
        selection.set_item("guarantee", out.selection.guarantee)?;
        selection.set_item("pool_size", out.selection.pool_size)?;
        selection.set_item("pool_exhausted", out.selection.pool_exhausted)?;
        selection.set_item("cut_margin", out.selection.cut_margin)?;

        let trace = PyDict::new(py);
        trace.set_item("input_count", out.trace.input_count)?;
        trace.set_item("pool_capacity", out.trace.pool_capacity)?;
        trace.set_item(
            "admission_scorer",
            out.trace.admission_scorer.map(|s| s.to_string()),
        )?;
        let scorers = PyList::empty(py);
        for (s, axis) in out.trace.scorers.iter().zip(axes) {
            let d = PyDict::new(py);
            d.set_item("scorer", s.scorer.as_str())?;
            d.set_item("calls", s.calls)?;
            d.set_item("missing", s.missing)?;
            d.set_item("elapsed_nanos", s.elapsed_nanos)?;
            d.set_item("cost", cost_name(axis.cost))?;
            scorers.append(d)?;
        }
        trace.set_item("scorers", scorers)?;
        match out.trace.budget {
            Some(b) => {
                let d = PyDict::new(py);
                d.set_item("s", b.s)?;
                d.set_item("fit_quality", b.fit_quality)?;
                d.set_item("fallback", b.fallback)?;
                d.set_item("reason", b.reason.map(fallback_name))?;
                d.set_item("derived_k", b.derived_k)?;
                trace.set_item("budget", d)?;
            }
            None => trace.set_item("budget", py.None())?,
        }

        Ok(PyOutcome {
            ranked,
            rejected,
            rejected_counts: counts.unbind(),
            selection: selection.unbind(),
            trace: trace.unbind(),
        })
    }
}

fn missing_name(p: MissingPolicy) -> &'static str {
    match p {
        MissingPolicy::Skip => "skip",
        MissingPolicy::Impute(_) => "impute",
        MissingPolicy::Reject => "reject",
    }
}

fn cost_name(c: ScorerCost) -> &'static str {
    match c {
        ScorerCost::Cheap => "cheap",
        ScorerCost::Expensive => "expensive",
    }
}

fn fallback_name(r: crate::budget::FallbackReason) -> &'static str {
    use crate::budget::FallbackReason as R;
    match r {
        R::PoorFit => "poor_fit",
        R::TooFewSamples => "too_few_samples",
        R::NotAMassDistribution => "not_a_mass_distribution",
    }
}

/// 상위 `k` 개 뒤에 남는 질량.
///
/// 엔진과 따로 쓸 수 있다. 같은 값이 캐시 크기 결정에도 쓰인다. 적중률은 이 식의 여집합이다.
#[pyfunction]
fn tail_mass(s: f32, k: usize, v: usize) -> f32 {
    crate::budget::tail_mass(s, k, v)
}

#[pymodule]
fn rust_multi_ranking_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    // 지금 실제로 도는 범위를 런타임에 확인할 수 있게 남긴다. 껍데기는 만들지 않는다.
    m.add(
        "__status__",
        "fusion, constraints, set objective, adaptive budget, and evidence are available",
    )?;
    m.add("EngineError", m.py().get_type::<EngineError>())?;
    m.add_class::<PyEngine>()?;
    m.add_class::<PyRanked>()?;
    m.add_class::<PyRejected>()?;
    m.add_class::<PyOutcome>()?;
    m.add_function(wrap_pyfunction!(tail_mass, m)?)?;
    Ok(())
}
