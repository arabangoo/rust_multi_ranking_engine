//! 선택 예산과 적합도 검정.
//!
//! K 를 상수로 두지 않는다. 정보가 몇 군데 몰려 있으면 K 는 작아야 하고 넓게 퍼져
//! 있으면 커야 한다. 고정 K 는 두 경우 모두에서 틀린 값이다.
//!
//! 다만 [`Budget::TailMass`] 는 분포 가정 위에 서 있으므로, 가정하고 쓰는 것이 아니라
//! **재고 나서** 쓴다. 적합이 문턱에 못 미치면 유도된 K 를 버리고 고정 K 로 되돌리며
//! 그 사실을 [`BudgetTrace`] 에 기록한다.

/// 적합도 문턱의 기본값. 콜모고로프-스미르노프 거리 0.1 에 해당한다.
pub const DEFAULT_MIN_FIT: f32 = 0.9;

/// 지수 탐색 구간. 이 밖의 지수는 실무에서 의미가 없다.
const S_LO: f64 = 0.01;
const S_HI: f64 = 6.0;

/// 몇 개를 고를지 정하는 방법.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Budget {
    /// 고정 개수.
    TopK(u32),
    /// 누락 허용 질량으로 K 를 유도한다.
    ///
    /// 적합도가 [`DEFAULT_MIN_FIT`] 에 못 미치면 `fallback_k` 로 되돌린다.
    ///
    /// # 설계서에서 달라진 점
    ///
    /// 설계서의 `TailMass { epsilon }` 에는 되돌아갈 K 가 없었다. "고정 K 로
    /// 되돌린다"는 문장이 성립하려면 그 K 를 받아야 하므로 `fallback_k` 를 더했다.
    TailMass {
        /// 허용 누락 질량. 0 초과 1 미만.
        epsilon: f32,
        /// 적합이 나쁠 때 되돌아갈 고정 개수.
        fallback_k: u32,
    },
    /// 비용 예산. 배낭형 제약이 된다.
    ///
    /// 후보별 비용은 [`Engine::cost`](crate::Engine::cost) 로 따로 준다. 비용 함수 없이
    /// 이 예산을 걸면 설정 오류다.
    Tokens {
        /// 총 비용 상한.
        max: u32,
    },
}

impl Default for Budget {
    fn default() -> Self {
        Budget::TopK(10)
    }
}

impl Budget {
    /// 꼬리 질량 예산을 만든다.
    pub fn tail_mass(epsilon: f32, fallback_k: u32) -> Self {
        Budget::TailMass {
            epsilon,
            fallback_k,
        }
    }

    /// 이 예산이 배낭형인가.
    pub fn is_knapsack(&self) -> bool {
        matches!(self, Budget::Tokens { .. })
    }
}

/// 고정 K 로 되돌린 이유.
///
/// 설계서의 `BudgetTrace` 에는 `fallback` 참·거짓만 있었다. 참인데 왜인지 모르면
/// 진단이 안 되므로 사유를 함께 남긴다.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FallbackReason {
    /// 적합도가 문턱에 못 미쳤다. 멱법칙이 아닌 분포에 멱법칙을 맞추면 자신 있게
    /// 틀린 K 가 나온다.
    PoorFit,
    /// 표본이 너무 적어 적합 자체가 뜻을 갖지 못한다.
    TooFewSamples,
    /// 질량이 음수이거나 전부 0 이라 확률로 정규화할 수 없다.
    NotAMassDistribution,
}

/// 예산을 어떻게 정했는가.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BudgetTrace {
    /// 추정한 멱법칙 지수.
    pub s: f32,
    /// 적합도. `1 - D` 이고 `D` 는 콜모고로프-스미르노프 거리다.
    pub fit_quality: f32,
    /// 적합이 나빠 고정 K 로 되돌렸는가.
    pub fallback: bool,
    /// 되돌렸다면 왜인가.
    pub reason: Option<FallbackReason>,
    /// 최종적으로 쓴 K.
    pub derived_k: u32,
}

/// 상위 `k` 개 뒤에 남는 질량.
///
/// 후보 수가 유한하므로 무한합이 아니라 **절단 조화합**을 쓴다. 지수가 1 에 가까우면
/// 무한합이 발산하기 때문이다.
///
/// 같은 값이 캐시 크기 결정에도 쓰인다. 적중률은 이 식의 여집합이다. 추정한 지수
/// 하나가 검색 예산과 캐시 크기를 동시에 정한다.
///
/// ```
/// use rust_multi_ranking_engine::budget::tail_mass;
/// // 지수가 크면 앞쪽에 질량이 몰려 꼬리가 얇다.
/// assert!(tail_mass(2.0, 10, 1000) < tail_mass(0.5, 10, 1000));
/// ```
pub fn tail_mass(s: f32, k: usize, v: usize) -> f32 {
    if v == 0 || k >= v {
        return 0.0;
    }
    let s = s as f64;
    let head: f64 = (1..=k).map(|r| (r as f64).powf(-s)).sum();
    let all: f64 = (1..=v).map(|r| (r as f64).powf(-s)).sum();
    if all <= 0.0 {
        return 0.0;
    }
    (1.0 - head / all) as f32
}

/// 절단 조화합. `H(s, n) = sum_{r=1..n} r^-s`.
fn harmonic(s: f64, n: usize) -> f64 {
    (1..=n).map(|r| (r as f64).powf(-s)).sum()
}

/// 관측된 질량으로 멱법칙 지수를 최대가능도 추정한다.
///
/// # 왜 로그-로그 회귀를 쓰지 않는가
///
/// 로그 축에서 직선처럼 보이는 것과 실제로 멱법칙인 것은 다르다. 로그-로그 회귀의
/// 결정계수는 멱법칙이 아닌 분포에서도 쉽게 0.99 가 나와서 적합도 지표로 못 쓴다.
/// 그래서 지수는 최대가능도로 추정하고 적합도는 콜모고로프-스미르노프 거리로 잰다
/// (Clauset, Shalizi, Newman 2009).
///
/// 입력은 순위 순으로 내림차순 정렬된 질량이어야 한다.
fn estimate_exponent(mass: &[f64]) -> f64 {
    let v = mass.len();
    let total: f64 = mass.iter().sum();
    // 관측 쪽 통계량: sum p_r * ln r.
    let observed: f64 = mass
        .iter()
        .enumerate()
        .map(|(i, m)| (m / total) * ((i + 1) as f64).ln())
        .sum();

    // 모형 쪽 통계량은 s 에 대해 단조 감소한다. 이분법으로 맞춘다.
    let model = |s: f64| -> f64 {
        let h = harmonic(s, v);
        if h <= 0.0 {
            return 0.0;
        }
        let num: f64 = (1..=v)
            .map(|r| {
                let r = r as f64;
                r.powf(-s) * r.ln()
            })
            .sum();
        num / h
    };

    let (mut lo, mut hi) = (S_LO, S_HI);
    if model(lo) < observed {
        return lo;
    }
    if model(hi) > observed {
        return hi;
    }
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if model(mid) > observed {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// 추정한 지수의 적합도. 콜모고로프-스미르노프 거리 `D` 를 재서 `1 - D` 를 돌려준다.
fn fit_quality(mass: &[f64], s: f64) -> f64 {
    let v = mass.len();
    let total: f64 = mass.iter().sum();
    let h_all = harmonic(s, v);
    if h_all <= 0.0 {
        return 0.0;
    }

    let mut empirical = 0.0;
    let mut model = 0.0;
    let mut d: f64 = 0.0;
    for (i, m) in mass.iter().enumerate() {
        empirical += m / total;
        model += ((i + 1) as f64).powf(-s) / h_all;
        d = d.max((empirical - model).abs());
    }
    (1.0 - d).clamp(0.0, 1.0)
}

/// 허용 누락 `epsilon` 을 만족하는 최소 K 를 찾는다.
fn derive_k(s: f64, v: usize, epsilon: f64) -> u32 {
    let all = harmonic(s, v);
    if all <= 0.0 {
        return v as u32;
    }
    let mut head = 0.0;
    for k in 1..=v {
        head += (k as f64).powf(-s);
        if 1.0 - head / all <= epsilon {
            return k as u32;
        }
    }
    v as u32
}

/// 관측 질량에서 K 를 유도한다. 적합이 나쁘면 고정 K 로 되돌린다.
///
/// `mass` 는 순위 순으로 내림차순 정렬된 값이어야 하고 음수가 없어야 한다.
pub fn derive_budget(mass: &[f64], epsilon: f32, fallback_k: u32, min_fit: f32) -> BudgetTrace {
    let bail = |reason: FallbackReason| BudgetTrace {
        s: 0.0,
        fit_quality: 0.0,
        fallback: true,
        reason: Some(reason),
        derived_k: fallback_k,
    };

    // 표본이 너무 적으면 적합 자체가 뜻이 없다. 지수 하나를 추정하는 데
    // 열 점은 있어야 한다는 실무 눈금을 쓴다.
    if mass.len() < 10 {
        return bail(FallbackReason::TooFewSamples);
    }
    if mass.iter().any(|m| !m.is_finite() || *m < 0.0) {
        return bail(FallbackReason::NotAMassDistribution);
    }
    let total: f64 = mass.iter().sum();
    if total <= 0.0 {
        return bail(FallbackReason::NotAMassDistribution);
    }

    let s = estimate_exponent(mass);
    let fit = fit_quality(mass, s);
    if (fit as f32) < min_fit {
        return BudgetTrace {
            s: s as f32,
            fit_quality: fit as f32,
            fallback: true,
            reason: Some(FallbackReason::PoorFit),
            derived_k: fallback_k,
        };
    }

    BudgetTrace {
        s: s as f32,
        fit_quality: fit as f32,
        fallback: false,
        reason: None,
        derived_k: derive_k(s, mass.len(), epsilon as f64),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zipf(s: f64, v: usize) -> Vec<f64> {
        (1..=v).map(|r| (r as f64).powf(-s)).collect()
    }

    #[test]
    fn tail_mass_falls_as_k_grows() {
        let a = tail_mass(1.0, 10, 1000);
        let b = tail_mass(1.0, 100, 1000);
        assert!(b < a, "{b} should be below {a}");
        assert_eq!(tail_mass(1.0, 1000, 1000), 0.0);
    }

    #[test]
    fn a_steeper_exponent_makes_a_thinner_tail() {
        assert!(tail_mass(2.0, 10, 1000) < tail_mass(0.5, 10, 1000));
    }

    #[test]
    fn the_estimator_recovers_the_exponent_it_was_given() {
        for truth in [0.7_f64, 1.0, 1.5, 2.2] {
            let est = estimate_exponent(&zipf(truth, 500));
            assert!((est - truth).abs() < 0.05, "truth {truth}, estimated {est}");
        }
    }

    #[test]
    fn a_true_power_law_passes_the_fit_gate() {
        let trace = derive_budget(&zipf(1.2, 500), 0.1, 99, DEFAULT_MIN_FIT);
        assert!(!trace.fallback, "trace = {trace:?}");
        assert!(trace.fit_quality > 0.95, "fit = {}", trace.fit_quality);
        assert!(trace.derived_k > 0 && trace.derived_k < 500);
        assert!(tail_mass(trace.s, trace.derived_k as usize, 500) <= 0.1 + 1e-6);
    }

    #[test]
    fn a_uniform_distribution_falls_back_instead_of_inventing_a_k() {
        // 균등 분포는 멱법칙이 아니다. 지수 0 으로 맞춰지지만 이 구현의 탐색 하한이
        // 0.01 이라 꼬리가 두꺼운 쪽으로 눌린다. 문턱이 그것을 잡아야 한다.
        let flat = vec![1.0_f64; 500];
        let trace = derive_budget(&flat, 0.1, 42, 0.999);
        assert!(trace.fallback);
        assert_eq!(trace.derived_k, 42);
        assert_eq!(trace.reason, Some(FallbackReason::PoorFit));
    }

    #[test]
    fn too_few_samples_falls_back_with_its_own_reason() {
        let trace = derive_budget(&zipf(1.2, 5), 0.1, 3, DEFAULT_MIN_FIT);
        assert!(trace.fallback);
        assert_eq!(trace.reason, Some(FallbackReason::TooFewSamples));
        assert_eq!(trace.derived_k, 3);
    }

    #[test]
    fn negative_mass_is_not_a_distribution() {
        let mut m = zipf(1.0, 20);
        m[3] = -0.5;
        let trace = derive_budget(&m, 0.1, 7, DEFAULT_MIN_FIT);
        assert_eq!(trace.reason, Some(FallbackReason::NotAMassDistribution));
    }

    #[test]
    fn a_smaller_epsilon_never_asks_for_fewer_items() {
        let m = zipf(1.1, 800);
        let loose = derive_budget(&m, 0.2, 1, DEFAULT_MIN_FIT).derived_k;
        let tight = derive_budget(&m, 0.05, 1, DEFAULT_MIN_FIT).derived_k;
        assert!(tight >= loose, "tight {tight}, loose {loose}");
    }
}
