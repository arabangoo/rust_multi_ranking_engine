# rust_multi_ranking_engine

> **여러 축의 점수를 스케일 안전하게 융합하고, 집합 제약 아래에서 상위 K개를 고르며,
> 탈락 사유까지 남기는 결정적 러스트 엔진.**

후보가 수백만 개이고 고를 것이 수십 개일 때, 무엇을 고를지 정하는 계층을 담당한다.
후보는 문서일 수도, 보안 이벤트일 수도, 유전자 돌연변이일 수도 있다. 엔진은 후보가
무엇인지 모른다.

| 항목 | 값 |
| --- | --- |
| 버전 | 0.1.0 |
| 언어 | Rust edition 2021. 최소 1.71 (`python` 기능만 1.83) |
| 라이선스 | Apache-2.0 |
| 기본 빌드 의존성 | `thiserror` 하나. 외부 함수 인터페이스(FFI) 0, 모델 0, 네트워크 0 |
| Python | PyO3 abi3 확장 모듈. 3.9 이상에서 휠 하나 |

핵심 참고 자료는 셋이다. 첫 번째가 제약 아래 선택의 근거이고, 두 번째가 점수 융합의
근거이며, 세 번째는 적응형 예산에 반드시 붙여야 하는 안전장치의 근거다.

1. G. L. Nemhauser, L. A. Wolsey, M. L. Fisher, [An analysis of approximations for maximizing submodular set functions](https://link.springer.com/article/10.1007/BF01588971) (1978). 서브모듈러 목적함수에서 탐욕 선택이 `1 - 1/e` 를 보장한다는 고전 결과
2. G. V. Cormack et al., [Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf), SIGIR 2009
3. A. Clauset, C. R. Shalizi, M. E. J. Newman, [Power-law distributions in empirical data](https://arxiv.org/abs/0706.1062) (2009). 멱법칙 적합이 왜 틀리기 쉬운지, 적합도 검정이 왜 필요한지

---

## 목차

1. [핵심 특징](#1-핵심-특징)
2. [빠른 시작](#2-빠른-시작)
3. [설치와 Cargo Feature](#3-설치와-cargo-feature)
4. [아키텍처](#4-아키텍처)
5. [공통 타입](#5-공통-타입)
6. [공개 API](#6-공개-api)
7. [융합 방식과 척도 판정](#7-융합-방식과-척도-판정)
8. [제약과 목적함수와 보장 계수](#8-제약과-목적함수와-보장-계수)
9. [선택 예산](#9-선택-예산)
10. [근거와 감사 출력](#10-근거와-감사-출력)
11. [Python 바인딩](#11-python-바인딩)
12. [파이프라인 통합](#12-파이프라인-통합)
13. [새 채점기와 제약과 목적함수 추가](#13-새-채점기와-제약과-목적함수-추가)
14. [빌드와 테스트](#14-빌드와-테스트)
15. [성능과 알려진 한계](#15-성능과-알려진-한계)
16. [디렉토리 구조](#16-디렉토리-구조)
17. [라이선스](#17-라이선스)

---

## 1. 핵심 특징

### 정렬이 아니라 선택이다

순위표를 만드는 것이 목적이 아니라 **고르는 것**이 목적이다. 그래서 이름은 랭킹이지만
반환값의 중심은 선택된 집합과 그 근거다.

### 틀리기 쉬운 것을 못 하게 만든다

모델 A는 로짓을 내고 B는 확률을 내고 C는 0에서 100을 낸다. 단순 가중합은 여기서 조용히
틀린다. 로짓 하나가 다른 축 전부를 압도하기 때문이다.

엔진은 그것을 문서로 경고하는 대신 **거부한다.** 채점기가 자기 점수의 척도를 선언하게
하고, 비교 불가능한 척도에 가중합을 걸면 후보를 한 건도 읽기 전에 오류가 난다.

### 조용히 사라지는 후보가 없다

모든 후보는 결과에 들어가거나 사유와 함께 기록되거나 둘 중 하나다. 중간은 없다. 탈락
사유 출력은 부가 기능이 아니라 이 엔진이 규제 도메인에서 쓰일 수 있는 최소 요건이다.

### 규모가 커져도 메모리와 비싼 호출이 늘지 않는다

전체 정렬 `O(N log N)` 대신 유계 힙으로 `O(N log M)` 이고, 더 중요한 것은 **후보를 전부
메모리에 올리지 않는 것**이다. 비싼 채점기는 1단계가 남긴 풀에만 돈다.

| 후보 수 | 풀 크기 | 비싼 채점기 호출 | 경과 |
| ---: | ---: | ---: | ---: |
| 10,000 | 960 | 960 | 2.6ms |
| 100,000 | 960 | 960 | 15.4ms |
| 1,000,000 | 960 | 960 | 97.8ms |

릴리스 빌드 실측(`examples/throughput.rs`). 후보가 100배 늘어도 비싼 채점기 호출은 그대로다.

### 근사일 때는 근사라고 말한다

집합 제약 아래 부분집합 선택은 일반형이 NP-hard 다. 이 엔진의 일은 그것을 푸는 것이
아니라 **어느 경우인지 판별해서 맞는 알고리즘을 고르고, 근사일 때는 얼마나 근사인지를
결과에 실어 보내는 것**이다.

### 결정적이다

같은 입력이면 스레드 수와 무관하게 같은 출력이 나온다. 동점 처리 순서까지 같다. 그래야
캐싱과 테스트와 감사 추적이 성립한다.

### 모델을 부르지 않는다

신경망을 실행하지 않는다. 채점기는 트레잇이고 그 안에서 무엇을 부를지는 호출자의 몫이다.
기본 빌드는 순수 러스트이며 외부 함수 인터페이스 호출이 없어 폐쇄망에서 그대로 돈다.

---

## 2. 빠른 시작

### Rust

```rust
use rust_multi_ranking_engine::{
    constraint, Budget, Candidate, CandidateId, Engine, Fusion, ScoreScale, Scorer,
    ScorerCost, ScorerId,
};

struct Doc {
    id: u64,
    source: &'static str,
    relevance: f32,
    authority: f32,
}

impl Candidate for Doc {
    fn id(&self) -> CandidateId {
        CandidateId::num(self.id)
    }
}

struct Axis(&'static str, fn(&Doc) -> f32);

impl Scorer<Doc> for Axis {
    fn id(&self) -> ScorerId { ScorerId::new(self.0) }
    fn scale(&self) -> ScoreScale { ScoreScale::Unit }
    fn cost(&self) -> ScorerCost { ScorerCost::Cheap }
    fn score(&self, d: &Doc) -> Option<f32> { Some((self.1)(d)) }
}

let docs = vec![
    Doc { id: 1, source: "arxiv", relevance: 0.95, authority: 0.9 },
    Doc { id: 2, source: "arxiv", relevance: 0.90, authority: 0.9 },
    Doc { id: 3, source: "blog",  relevance: 0.60, authority: 0.5 },
    Doc { id: 4, source: "arxiv", relevance: 0.55, authority: 0.9 },
];

let out = Engine::new()
    .scorer(Axis("relevance", |d| d.relevance))
    .scorer(Axis("authority", |d| d.authority))
    .fuse(Fusion::weighted_sum())
    .unary(constraint::predicate("min_authority", |d: &Doc| d.authority >= 0.3))
    .set_constraint(constraint::max_per_group("max_per_source", 2, |d: &Doc| d.source))
    .budget(Budget::TopK(3))
    .run(docs)
    .unwrap();

// arxiv 는 둘까지만 들어가므로 3위는 점수가 더 낮은 blog 가 차지한다.
let picked: Vec<u64> = out.ranked.iter().map(|r| r.candidate.id).collect();
assert_eq!(picked, vec![1, 2, 3]);

// 4번은 점수에 밀린 것이 아니라 출처 제약에 막혔다. 그 사실이 사유로 남는다.
assert_eq!(out.rejected_counts.set_constraint, 1);
assert!(out.is_complete());
```

### Python

```python
import rust_multi_ranking_engine as rmre

engine = rmre.Engine()
engine.scorer("relevance")
engine.scorer("authority")
engine.fuse("weighted_sum")
engine.max_per_group("source", 2)
engine.budget_top_k(3)

out = engine.run([
    {"id": 1, "scores": {"relevance": 0.95, "authority": 0.9}, "groups": {"source": "arxiv"}},
    {"id": 2, "scores": {"relevance": 0.90, "authority": 0.9}, "groups": {"source": "arxiv"}},
    {"id": 3, "scores": {"relevance": 0.60, "authority": 0.5}, "groups": {"source": "blog"}},
    {"id": 4, "scores": {"relevance": 0.55, "authority": 0.9}, "groups": {"source": "arxiv"}},
])

assert [r.id for r in out.ranked] == ["1", "2", "3"]
assert out.rejected_counts["set_constraint"] == 1
assert out.is_complete()
```

---

## 3. 설치와 Cargo Feature

### Rust

crates.io 에는 아직 게시하지 않았다. 태그를 지정해 저장소에서 받는다.

```toml
[dependencies]
rust_multi_ranking_engine = { git = "https://github.com/arabangoo/rust_multi_ranking_engine", tag = "v0.1.0" }
```

### Python

```bash
pip install rust_multi_ranking_engine
```

소스에서 빌드하려면 maturin 을 쓴다.

```bash
maturin develop          # 개발용 설치
maturin build --release  # 휠 생성
```

### Feature

| Feature | 기본 | 켜면 생기는 것 | 추가 의존성 |
| --- | --- | --- | --- |
| (없음) | 켜짐 | 타입, 트레잇, 융합, 제약, 목적함수, 예산, 선택, 감사 출력 전부 | `thiserror` |
| `serde` | 꺼짐 | 공개 타입에 `Serialize`·`Deserialize` 유도 | `serde` |
| `parallel` | 꺼짐 | 2단계 풀 위에서 비싼 채점기를 병렬 실행 | `rayon` |
| `python` | 꺼짐 | PyO3 abi3 확장 모듈 | `pyo3` |

**코어는 언제나 빌드된다.** 감사 출력용 JSON 도 의존성 없이 직접 쓰므로 `serde` 는 소비자
편의용 옵트인이다.

**최소 러스트 버전은 1.71 이고 `python` 기능만 1.83 이다.** PyO3 0.29 가 요구하는 값이라
옵트인 경로 하나 때문에 코어 소비자 전체를 끌어올리지 않는다.

**`parallel` 을 켜도 결정성은 그대로다.** 융합은 후보 식별자 순 고정 순서이고 병렬 채점
결과를 색인 순서로 다시 모으기 때문이다.

---

## 4. 아키텍처

### 두 단계로 나뉘는 이유

스트리밍과 집합 제약은 근본적으로 충돌한다. 집합 제약을 보려면 후보 풀이 있어야 하는데
스트리밍은 풀을 만들지 않는 것이 목적이다. 그래서 나눈다.

```text
1단계 (스트리밍)
  후보 N개 → 단항 제약 → 싼 채점기 → 유계 힙 상위 M개
             M = K 곱하기 배수 (기본 32배, 조정 가능)
             메모리 O(M), 시간 O(N log M)

2단계 (풀 위에서)
  상위 M개 → 비싼 채점기 → 융합 → 집합 제약 아래 선택 → 최종 K개
             매트로이드면 탐욕이 정확, 아니면 근사와 보장 계수 보고
```

**단항 제약이 채점보다 먼저 돈다.** 후보가 1,000만 개일 때 떨어질 후보에 채점기를 돌리는
것은 낭비다. 그 결과 어떤 후보가 단항 제약과 필수 축 결측에 동시에 걸리면 사유는 단항
제약으로 남는다.

### 이것은 근사이고 그 사실을 숨기지 않는다

M 밖으로 밀려난 후보가 집합 제약 때문에 최종 해에 들어가야 했을 수 있다. 그래서 결과에
싣는다.

| 필드 | 의미 | 위험 신호 |
| --- | --- | --- |
| `pool_size` | 1단계가 남긴 M | |
| `pool_exhausted` | 풀을 다 쓰고도 K를 못 채웠다 | 참이면 풀 배수를 키워야 한다 |
| `exact` | 선언받은 구조상 탐욕이 풀 위에서 최적이었다 | |
| `guarantee` | 근사 보장 계수 | `exact` 가 거짓일 때만 |
| `cut_margin` | 마지막 자리를 다툰 두 값의 차 | 0 에 가까우면 그 선택이 흔들린다 |

조용히 부족한 답을 주는 대신 말한다.

### 채점기 비용에 따라 단계를 나눈다

채점기 하나가 다른 것보다 1,000배 비싸면 모든 후보에 다 돌리는 것은 설계 결함이다.
검색 시스템에서 캐스케이드 순위화라 부르는 방식을 쓴다. 비싼 채점기가 몇 번 불렸는지도
감사 기록에 남으므로 비용을 되짚을 수 있다.

### 엔진의 경계

특징 계산은 엔진 밖이다. 그래프 라플라시안 고유벡터나 임베딩은 색인 시점에 미리 구해
두고, 엔진은 그 좌표를 점수 축 하나로 받아 쓸 뿐이다.

| 엔진 밖 (색인 시점) | 엔진 안 (질의 시점) |
| --- | --- |
| 임베딩 생성 | 받은 축들을 융합 |
| 그래프 중심성, 스펙트럴 좌표 | 제약 적용 |
| 커뮤니티 탐지 | 예산 안에서 선택 |
| 모델 학습 | 근거 기록 |
| 중복 제거 | |

---

## 5. 공통 타입

### 후보 식별자

```rust
pub enum CandidateId {
    Num(u64),          // 복제 비용 없음. 스트리밍 경로에 적합
    Text(Box<str>),    // 복제할 때마다 할당
}
```

두 표현을 갖는 이유는 비용과 표현력이 부딪히기 때문이다. 후보가 1,000만 개인 스트리밍에서는
정수 하나가 맞고, 감사 출력에서 `doc-3141` 같은 이름을 그대로 보여야 하는 자리에서는
문자열이 맞다.

정렬 순서는 `Num` 이 전부 `Text` 앞에 오고 같은 변형 안에서는 값 순이다. 둘을 섞어 써도
전순서라 결정성은 깨지지 않지만, 사람이 읽는 순서와 어긋나므로 한 종류로 통일하는 편이 낫다.

**같은 실행 안에서 서로 다른 후보가 같은 식별자를 가지면 안 된다.** 엔진은 이것을 검증하지
않는다. 1,000만 후보의 중복을 검사하려면 후보 전부를 메모리에 올려야 하고, 그것이 이
엔진이 피하려는 바로 그 비용이기 때문이다.

### 점수 집합과 결과

```rust
/// 후보 하나가 받은 점수 전부. 융합 전 원본을 보관한다.
pub struct ScoreSet { /* (ScorerId, Option<f32>) 쌍들 */ }

/// 최종 결과. 왜 이 순위인지 재현 가능해야 한다.
pub struct Ranked<C> {
    pub candidate: C,
    pub rank: u32,
    pub fused: f32,
    pub scores: ScoreSet,                // 융합 전 원본
    pub fusion: FusionTrace,             // 어떻게 합쳤는가
    pub constraints: Vec<ConstraintId>,  // 통과한 집합 제약
}

/// 떨어진 후보.
pub struct Rejected<C> {
    pub candidate: C,
    pub reason: RejectReason,
    pub fused: Option<f32>,
}

pub enum RejectReason {
    NotScored(ScorerId),           // 채점기가 값을 내지 못했다. 0점과 다르다
    UnaryConstraint(ConstraintId),
    SetConstraint(ConstraintId),   // 집합 제약 때문에 자리를 못 얻었다
    BelowThreshold,
    Outranked,                     // 더 높은 후보에게 밀렸다
    OutOfPool,                     // 1단계 풀에 들지 못했다
}
```

### 실행 결과 전부

```rust
pub struct Outcome<C> {
    pub ranked: Vec<Ranked<C>>,
    pub rejected: Vec<Rejected<C>>,      // 보관 정책이 정한 만큼만
    pub rejected_counts: RejectCounts,   // 정책과 무관하게 언제나 정확
    pub selection: Selection,
    pub trace: RunTrace,
}

impl<C> Outcome<C> {
    /// 완전성 불변식. 결과 수와 탈락 수의 합이 입력 수와 같은가.
    pub fn is_complete(&self) -> bool;
}
```

### 채점 불가와 0점은 다르다

`Option<f32>` 가 중요하다. 모델이 돌지 않았는데 0점으로 세면 그 후보는 부당하게 죽는다.
융합 단계는 값이 없는 축을 만나면 결측 정책 세 가지 중 하나를 택하고 어느 것을 택했는지
기록한다.

**`NaN` 은 결측과 같이 다룬다.** 순서를 매길 수 없는 값이 힙에 들어가면 결정성이 깨진다.

---

## 6. 공개 API

### 사용자가 구현하는 트레잇은 둘

```rust
pub trait Candidate {
    fn id(&self) -> CandidateId;
}

pub trait Scorer<C>: Sync {
    fn id(&self) -> ScorerId;

    /// 이 점수가 어떤 척도인가. 융합 가능성 판정에 쓰인다.
    fn scale(&self) -> ScoreScale;

    /// 얼마나 비싼가. 캐스케이드 단계 배치에 쓰인다.
    fn cost(&self) -> ScorerCost;

    fn score(&self, c: &C) -> Option<f32>;

    /// 여러 후보를 한 번에. 기본 구현은 위를 차례로 부르므로 대개 건드리지 않는다.
    /// 배치 추론을 쓰는 비싼 축만 재정의한다.
    fn score_batch(&self, candidates: &[&C]) -> Vec<Option<f32>> {
        candidates.iter().map(|c| self.score(c)).collect()
    }
}
```

`id`·`scale`·`cost` 는 상수여야 한다. 엔진은 등록 시점에 한 번만 읽고 그 뒤로 캐시한다.

융합 방식, 선택 알고리즘, 근거 기록은 엔진의 일이므로 트레잇으로 열지 않는다.

### 빌더

```rust
let out = Engine::new()
    .scorer(semantic_relevance)          // 싼 축
    .scorer(recency)                     // 싼 축
    .scorer(cross_encoder)               // 비싼 축. 2단계로 밀린다
    .fuse(Fusion::Rrf { k: 60.0 })
    .missing(MissingPolicy::Skip)
    .admission("semantic")               // 1단계 절단 기준
    .unary(constraint::predicate("min_authority", |d: &Doc| d.authority >= 0.3))
    .set_constraint(constraint::max_per_group("max_per_source", 3, |d: &Doc| d.source))
    .objective(Coverage::new(|d: &Doc| d.topics.clone()))
    .require(Requirement::at_least("needs_recent", 2, |d: &Doc| d.is_recent))
    .budget(Budget::TopK(30))
    .pool_multiplier(32)
    .threshold(0.1)
    .rejections(Rejections::Keep)
    .run(candidates)?;
```

| 메서드 | 하는 일 | 기본값 |
| --- | --- | --- |
| `scorer(s)` | 점수 축 등록. 등록 순서가 감사 출력의 축 순서 | 없음(필수) |
| `fuse(f)` | 융합 방식 | `Rrf { k: 60.0 }` |
| `missing(p)` | 값 없는 축을 어떻게 다룰지 | `Skip` |
| `unary(c)` | 단항 제약. 1단계에서 채점보다 먼저 돈다 | 없음 |
| `set_constraint(c)` | 집합 제약 | 없음 |
| `objective(o)` | 집합 목적함수 | 없음(모듈러) |
| `require(r)` | 최종 집합의 하한 조건 | 없음 |
| `budget(b)` | 몇 개를 고를지 | `TopK(10)` |
| `cost(f)` | 후보별 비용. `Budget::Tokens` 에 필요 | 없음 |
| `pool_multiplier(n)` | `M = K * n` | 32 |
| `admission(id)` | 1단계 절단에 쓸 승인 채점기 | 자동 선택 |
| `threshold(v)` | 융합 점수의 하한 | 없음 |
| `rejections(p)` | 탈락 후보 보관 정책 | `Keep` |
| `min_fit(v)` | 꼬리 질량 예산의 적합도 문턱 | 0.9 |
| `validate()` | 설정 검사만 한다 | |
| `run(iter)` | 후보를 훑어 고른다 | |

`run` 은 반복 가능한 것이면 무엇이든 받는다. 목록을 통째로 만들지 않고 반복자를 넘기면
1단계가 스트리밍으로 처리한다.

### 승인 채점기가 무엇인가

**순위 융합은 순위를 입력으로 쓰는데 순위는 풀이 있어야 나온다.** 그래서 1단계가 상위 M개를
무엇으로 자를지 따로 정해야 한다. 그것이 승인 채점기다.

| 융합 방식 | `admission` 지정 | 1단계 절단 기준 |
| --- | --- | --- |
| `Rrf` | 함 | 그 채점기의 값 |
| `Rrf` | 안 함 | 등록된 첫 싼 채점기. 무엇을 골랐는지 기록에 남는다 |
| `WeightedSum`·`Max` | 함 | 그 채점기의 값 |
| `WeightedSum`·`Max` | 안 함 | 싼 축들만으로 계산한 융합 값 |

승인 채점기는 후보 전부를 훑으므로 **싼 축이어야 한다.** 비싼 축을 지정하면 설정 오류다.

### 오류

설정 오류는 후보를 한 건도 읽기 전에 걸린다. 실행 중에 나는 것은 아래 둘뿐이다.

| 오류 | 언제 |
| --- | --- |
| `NoScorers` | 채점기가 하나도 없다 |
| `DuplicateScorer` | 같은 식별자가 두 번 등록됐다 |
| `IncompatibleScale` | 비교 불가능한 척도에 값 기반 융합을 걸었다 |
| `UnknownWeight` | 가중치가 등록되지 않은 채점기를 가리킨다 |
| `UnknownAdmissionScorer` | 승인 채점기가 등록돼 있지 않다 |
| `ExpensiveAdmissionScorer` | 승인 채점기가 비싼 축이다 |
| `NoAdmissionScorer` | 순위 융합인데 싼 축이 하나도 없다 |
| `InvalidBudget` | 예산 값이 뜻을 갖지 못한다 |
| `InvalidPoolMultiplier` | 풀 배수가 0 이다 |
| `BatchLengthMismatch` | 배치 채점기가 입력과 다른 길이를 돌려줬다 (실행 중) |
| `InfeasibleRequirement` | 요구 조건을 채울 후보가 풀에 모자라다 (실행 중) |

---

## 7. 융합 방식과 척도 판정

### 척도를 선언하게 한다

```rust
pub enum ScoreScale {
    Unit,        // 0 이상 1 이하. 확률이나 정규화된 값. 서로 더해도 된다
    Unbounded,   // 로짓이나 거리. 정규화 없이 더하면 틀린다
    Rank,        // 순서만 의미가 있다. 값끼리 더할 수 없다
}
```

엔진은 이 선언을 근거로 융합 가능성을 판정한다.

| 융합 방식 | `Unit` 만 | `Unbounded` 포함 | `Rank` 포함 |
| --- | --- | --- | --- |
| `WeightedSum` | 허용 | 거부 | 거부 |
| `Rrf` | 허용 | 허용 | 허용 |
| `Max` | 허용 | 거부 | 거부 |

거부는 실행 시점 오류가 아니라 **설정 오류**다.

### 정규화기

무한 척도 축을 가중합에 넣으려면 정규화기를 명시적으로 끼워야 한다. 엔진이 몰래
정규화해 주지 않는 이유는 어떤 정규화를 쓸지가 도메인 지식이기 때문이다.

```rust
use rust_multi_ranking_engine::{Normalizer, ScorerExt};

.scorer(logit_model.normalized(Normalizer::Sigmoid))
```

| 정규화기 | 동작 |
| --- | --- |
| `Sigmoid` | 로지스틱 함수. 실수 전체를 0에서 1 사이로 |
| `Clamp01` | 0 미만은 0, 1 초과는 1 로 자른다 |
| `MinMax { min, max }` | 선언된 구간을 0에서 1 로 선형 사상 |

**`MinMax` 도 경계를 밖에서 받는다.** 스트리밍 중에 관측한 최소·최대로 정규화하면 같은
입력이라도 도착 순서에 따라 값이 달라져 결정성이 깨진다.

### 순위 융합을 기본값으로 둔다

축이 서로 다른 곳에서 왔다면 값을 더하는 것보다 순위를 합치는 편이 안전하다.

```text
RRF(d) = 각 소스에 대해 1 / (k + 순위(d)) 를 더한 값
```

`k` 는 상위 순위의 영향력을 조절한다. 값이 크면 순위 차이가 완만해진다. 관례 상수는 60 이다.

### 결측 정책

| 정책 | 동작 | 언제 |
| --- | --- | --- |
| `Skip` | 그 축을 빼고 나머지로 융합. 가중합이면 가중치를 재정규화 | 축이 선택적일 때 |
| `Impute(v)` | 지정한 값으로 대체 | 결측이 의미를 갖는 경우 |
| `Reject` | 후보를 `NotScored` 로 탈락 | 필수 축일 때 |

어느 정책을 썼는지가 융합 기록과 감사 출력에 남는다.

### 결정성을 지키는 방법

채점은 병렬로 하되 **융합은 후보 식별자 순으로 고정 순서 합산**한다. 부동소수 덧셈은
결합법칙이 성립하지 않으므로, 합산 순서가 스레드 스케줄에 따라 달라지면 값이 미세하게
갈리고 동점 처리가 뒤집힌다. 이것을 지키지 않은 병렬 순위화 코드가 흔하다.

동점일 때는 후보 식별자 오름차순으로 정렬한다. 임의 순서를 남기지 않는다.

---

## 8. 제약과 목적함수와 보장 계수

### 두 종류의 제약

집합 제약이 걸리면 **상위 K개가 정답이 아니게 된다.** 점수 1위, 2위, 3위가 전부 같은
출처면 그중 하나를 버리고 4위를 넣어야 한다.

```rust
pub trait UnaryConstraint<C>: Sync {
    fn id(&self) -> ConstraintId;
    fn allows(&self, c: &C) -> bool;
}

pub trait SetConstraint<C>: Sync {
    fn id(&self) -> ConstraintId;
    /// 이 후보를 현재 선택에 더할 수 있는가.
    fn admits(&self, selected: &[&C], candidate: &C) -> bool;
    /// 매트로이드 구조인가. 참이면 탐욕이 최적임을 보장할 수 있다.
    fn is_matroid(&self) -> bool;
}
```

### 기본 제공 제약

| 함수 | 성질 | 보장 |
| --- | --- | --- |
| `constraint::max_per_group(id, m, key)` | 분할 매트로이드 | 탐욕이 정확 |
| `constraint::max_total(id, k)` | 균일 매트로이드 | 탐욕이 정확 |
| `constraint::cost_budget(id, limit, cost)` | 배낭형 | 근사 |
| `constraint::predicate(id, f)` | 단항 | 필터 |

`is_matroid` 를 구현자가 선언하게 하는 것은 위험해 보이지만 의도적이다. 임의의 제약이
매트로이드인지 엔진이 자동으로 판정할 수는 없다. 대신 **엔진이 기본 제공하는 제약은 그
성질이 증명된 것만** 둔다.

**결과의 `exact` 는 엔진이 그렇게 선언받았다는 뜻이지 엔진이 증명했다는 뜻이 아니다.**

### 하한 요구 조건

`admits` 는 "이것을 더해도 되는가"를 묻는 술어라 **상한만** 표현할 수 있다. 하한은 별도
개념으로 둔다.

```rust
.require(Requirement::at_least("needs_news", 2, |d: &Doc| d.source == "news"))
```

하한은 탐욕이 끝난 뒤 **교체**로 채운다. 조건을 만족하는 미선택 후보 중 가장 점수가 높은
것을, 조건을 만족하지 않는 선택된 후보 중 가장 점수가 낮은 것과 바꾼다. 이 교체는 탐욕의
최적성을 깨므로 **요구 조건이 하나라도 있으면 `exact` 는 거짓이 된다.**

채울 후보가 아예 없으면 조용히 부족한 답을 주지 않고 `InfeasibleRequirement` 를 낸다.

### 집합 목적함수

후보별 독립 점수의 합이 아닌 목적함수를 지원한다. 검색 결과를 고를 때 실제 목표는 보통
이런 모양이다.

```text
목표:  관련성(S) + 포괄성(S) + 연결성(S) - 중복성(S)  을 최대화
       S 는 고를 후보 집합
제약:  토큰비용(S) <= 예산
```

관련성과 연결성은 후보별로 계산되지만 **포괄성과 중복성은 집합 전체의 함수**다. 세 번째
문서의 가치는 앞의 둘이 무엇이었느냐에 달려 있다.

```rust
pub trait SetObjective<C>: Sync {
    /// 현재 선택에 이 후보를 더했을 때의 이득.
    fn marginal_gain(&self, selected: &[&C], candidate: &C) -> f32;
    /// 서브모듈러인가. 참이면 탐욕에 보장 계수가 붙는다.
    fn is_submodular(&self) -> bool;
}
```

**총 이득은 융합 점수에 이 함수의 한계 이득을 더한 값이다.**

```text
총 이득(S, c) = 융합점수(c) + marginal_gain(S, c)
```

목적함수를 주지 않으면 `marginal_gain` 이 항상 0 인 것과 같아 총 이득이 융합 점수가 되고,
그때는 상위 K가 최적이다. 목적함수는 모듈러 항 위에 얹는 보정이다.

`marginal_gain` 은 `selected` 의 **원소 순서가 달라도 같은 값**을 돌려줘야 한다. 순서에
의존하면 결정성이 탐욕의 진행 순서에 묶인다.

### 기본 제공 목적함수

`Coverage` 하나뿐이다. 후보가 덮는 원소들의 합집합 크기를 최대화한다.

```rust
.objective(Coverage::new(|d: &Doc| d.topics.clone()))
.objective(Coverage::new(|d: &Doc| d.topics.clone()).weighted(|t: &Topic| t.importance))
```

이미 덮인 원소를 다시 덮어도 이득이 없으므로 한계 이득이 단조 감소한다. 그것이 서브모듈러의
정의이고, 최대 피복(maximum coverage)은 그 성질이 증명된 형태라 `is_submodular` 가 참이어도
근거가 있다.

**최대 한계 관련성(MMR) 류를 기본 제공하지 않는 이유가 여기 있다.** 서브모듈러라는 근거를
댈 수 없어서, 넣으면 `is_submodular` 를 함부로 선언하지 말라는 이 엔진의 원칙을 스스로
어기게 된다.

### 보장 계수

| 목적함수 | 제약 | 계수 | 상수 |
| --- | --- | --- | --- |
| 모듈러 | 매트로이드 1개 이하 | 정확 | `exact = true` |
| 모듈러 | 매트로이드 `p` 개 | `1/p` | |
| 모듈러 | 배낭형 1개 | `0.5` | `GUARANTEE_KNAPSACK_MODULAR` |
| 서브모듈러 | 개수 제한만 | `1 - 1/e` (약 0.632) | `GUARANTEE_CARDINALITY` |
| 서브모듈러 | 매트로이드 1개 | `0.5` | `GUARANTEE_MATROID` |
| 서브모듈러 | 매트로이드 `p` 개 | `1/(p+1)` | |
| 서브모듈러 | 배낭형 1개 | `1 - e^(-1/2)` (약 0.393) | `GUARANTEE_KNAPSACK_SUBMODULAR` |

**`1 - 1/e` 는 개수 제한에서만 성립한다.** 일반 매트로이드에서는 `1/2` 로 내려간다.

**위 표에 없는 조합에는 계수를 주지 않는다.** 비매트로이드 제약이 둘 이상이거나, 매트로이드와
배낭형이 섞였거나, 서브모듈러라고 선언되지 않은 목적함수가 걸렸을 때다. 근거 없는 숫자를
결과에 싣지 않기 위해서다.

---

## 9. 선택 예산

### K를 상수로 두지 않는다

정보가 몇 군데 몰려 있으면 K는 작아야 하고 넓게 퍼져 있으면 커야 한다. 고정 K는 두 경우
모두에서 틀린 값이다.

```rust
pub enum Budget {
    TopK(u32),
    TailMass { epsilon: f32, fallback_k: u32 },
    Tokens { max: u32 },
}
```

### 꼬리 질량으로 K를 유도하는 방법

접근 빈도가 긴 꼬리 분포를 보이는 경우가 많다. 순위 `r` 인 항목의 확률을 `r` 의 `-s` 승으로
근사하면 상위 K개 뒤에 남는 질량을 계산할 수 있고, 허용 누락 `epsilon` 을 만족하는 최소 K를
찾을 수 있다.

후보 수가 유한하므로 무한합이 아니라 **절단 조화합**을 쓴다. 지수가 1에 가까우면 무한합이
발산하기 때문이다.

```rust
use rust_multi_ranking_engine::tail_mass;

// 지수가 크면 앞쪽에 질량이 몰려 꼬리가 얇다.
assert!(tail_mass(2.0, 10, 1000) < tail_mass(0.5, 10, 1000));
```

같은 값이 **캐시 크기 결정에도 쓰인다.** 적중률은 이 식의 여집합이다. 추정한 지수 하나가
검색 예산과 캐시 크기를 동시에 정한다.

### 적합도를 재지 않으면 쓰지 않는다

이 방식 전체가 지수 `s` 의 추정에 걸려 있다. 멱법칙 적합은 틀리기 쉽기로 유명하다. 로그
축에서 직선처럼 보이는 것과 실제로 멱법칙인 것은 다르다. 멱법칙이 아닌 분포에 멱법칙을
맞추면 **자신 있게 틀린 K** 가 나온다.

**지수는 최대가능도로 추정하고 적합도는 콜모고로프-스미르노프 거리로 잰다.** 로그-로그
회귀의 결정계수는 멱법칙이 아닌 분포에서도 쉽게 0.99 가 나와 적합도 지표로 쓸 수 없다.

```rust
pub struct BudgetTrace {
    pub s: f32,                        // 추정한 지수
    pub fit_quality: f32,              // 1 - D. D 는 콜모고로프-스미르노프 거리
    pub fallback: bool,                // 고정 K로 되돌렸는가
    pub reason: Option<FallbackReason>,
    pub derived_k: u32,
}
```

| 되돌린 사유 | 뜻 |
| --- | --- |
| `PoorFit` | 적합도가 문턱(기본 0.9)에 못 미쳤다 |
| `TooFewSamples` | 표본이 열 개도 안 돼 적합 자체가 뜻이 없다 |
| `NotAMassDistribution` | 질량이 음수이거나 전부 0 이다 |

가정하고 쓰는 것이 아니라 재고 나서 쓴다.

### 배낭형 예산

`Budget::Tokens` 는 `Engine::cost` 로 후보별 비용을 함께 받는다. 비용 대비 이득으로 고른
집합과 단일 최고 항목을 비교해 나은 쪽을 쓴다. 이 두 갈래 비교가 보장 계수의 근거다.
비율 탐욕만 쓰면 값이 아주 큰 단일 항목을 통째로 놓치는 경우가 있어 계수가 성립하지 않는다.

---

## 10. 근거와 감사 출력

### 불변식 넷

정확성을 문장이 아니라 검사로 못박는다. `tests/invariants.rs` 가 이것을 강제한다.

| 불변식 | 내용 | 검증 방법 |
| --- | --- | --- |
| **결정성** | 같은 입력이면 스레드 수와 무관하게 순위와 동점 처리까지 동일 | 고정 지문 리터럴 대조 |
| **완전성** | 모든 후보가 결과 또는 탈락 목록에 정확히 한 번 나타난다 | 개수 합 대조 |
| **제약 준수** | 반환된 집합이 선언된 집합 제약을 전부 만족한다 | 결과를 제약에 다시 통과 |
| **근거 재현** | 원본 점수와 융합 기록만으로 최종 점수를 다시 계산할 수 있다 | `FusionTrace::recompute` |

### 절단선의 여유

K번째와 K+1번째의 차가 거의 0이면 그 선택은 흔들린다. 씨앗이나 부동소수 오차 하나로 순위가
뒤집힌다. **34개를 골랐다는 것과 34번째와 35번째가 사실상 같았다는 것은 다른 사건이다.**

K+1번째는 **그 자리를 실제로 다툰 후보** 중 가장 높은 것이다. 마지막 하나를 뺀 집합을
기준으로 다시 판정하므로, 집합 제약에 막혀 애초에 자리를 다툰 적이 없는 후보는 들어오지
않는다.

**잣대는 선택기가 실제로 쓴 기준이다.** 개수 예산이면 총 이득이고 배낭형이면 비용 대비
이득이다. 융합 점수로만 재면 목적함수가 걸렸을 때 음수가 나오는데, 그 숫자는 순서가
뒤집혔다는 뜻이 아니라 잣대가 틀렸다는 뜻이다.

### 탈락 후보 보관 정책

"모든 후보를 기록한다"와 "후보를 전부 메모리에 올리지 않는다"는 후보가 1,000만이면 그대로
충돌한다. 그래서 **개수는 언제나 정확히 세고** 상세를 얼마나 남길지만 고른다.

| 정책 | 동작 | 언제 |
| --- | --- | --- |
| `Keep` | 전부 보관 (기본) | 규제 도메인, 소규모 입력 |
| `Count` | 개수만 센다 | 후보가 아주 많을 때 |
| `Sample(n)` | 처음 만난 n건만 보관 | 표본만 필요할 때 |

완전성 불변식은 개수로 성립하므로 규모와 무관하다.

### 감사 출력

결과 하나를 JSON 으로 낸다. 의존성 없이 직접 쓰므로 `serde` 없이도 나온다.

```rust
let json = ranked.to_json(&ranked.candidate.id());
```

```json
{
  "candidate": "doc-3141",
  "rank": 3,
  "fused": 0.913,
  "scores": { "semantic": 0.94, "centrality": 0.87, "recency": null },
  "missing_policy": "skip",
  "fusion": {
    "method": "rrf",
    "k": 60.0,
    "terms": [
      { "scorer": "semantic", "input": { "kind": "rank", "rank": 1 },
        "weight": 1.0, "contribution": 0.016393 }
    ]
  },
  "constraints": { "max_per_source": "pass" }
}
```

`recency` 가 값 없음이고 정책이 건너뛰기였다는 것까지 남는다. 왜 3위인가에 답할 수 있어야
규제 도메인에서 쓸 수 있다.

### 실행 기록

```rust
pub struct RunTrace {
    pub input_count: u64,
    pub pool_capacity: u32,
    pub admission_scorer: Option<ScorerId>,   // 자동 선택됐어도 무엇인지 남는다
    pub scorers: Vec<ScorerTrace>,            // 축별 호출 횟수, 결측 수, 소요 시간
    pub budget: Option<BudgetTrace>,          // 고정 K였으면 None
}
```

---

## 11. Python 바인딩

`feature = "python"` 을 켜면 PyO3 확장 모듈이 된다. abi3 휠이라 Python 3.9 이상에서 휠
하나로 돈다.

### 러스트 API 와 모양이 다른 이유

러스트 쪽 `Scorer` 는 사용자가 구현하는 트레잇이다. 그대로 옮기면 후보 하나마다, 축마다
파이썬을 다시 불러야 한다. 후보가 100만이면 파이썬 호출이 200만 번이고, 전역 인터프리터
잠금(GIL, Global Interpreter Lock) 때문에 병렬 채점도 무의미해진다. 캐스케이드가 아껴 둔
이득이 그 자리에서 사라진다.

그래서 표면을 둘로 가른다.

| 축 | 넘기는 방법 | 도는 자리 |
| --- | --- | --- |
| 싼 축 | 후보에 점수를 담아 데이터로 | 1단계. 파이썬이 끼어들지 않는다 |
| 비싼 축 | 파이썬 함수 하나 | 2단계 풀. **풀 전체가 한 번의 호출** |

벡터 검색 결과가 원래 점수를 들고 오는 모양이라 억지가 아니다. 그리고 비싼 축은
`score_batch` 로 넘어가므로 교차 인코더처럼 배치 추론을 쓰는 모델이 원하는 모양 그대로다.
후보 100만 건에서도 콜백은 **한 번** 불리고 960건을 받는다.

### 후보 표현

사전이다. `id` 와 `scores` 만 필수이고 나머지는 걸어 둔 기능이 쓸 때만 읽는다.

```python
{"id": "doc-1",
 "scores": {"similarity": 0.94},   # 사전이나 등록 순서에 맞춘 목록
 "groups": {"source": "manual"},   # max_per_group, require_at_least 가 본다
 "cost": 220,                      # budget_tokens 가 본다
 "cover": ["setup", "auth"]}       # coverage_objective 가 본다
```

### Engine 메서드

| 메서드 | 하는 일 |
| --- | --- |
| `scorer(name, *, scale, cost, normalize, normalize_range, fn)` | 축 등록 |
| `fuse(method, *, k, weights)` | `"rrf"`, `"weighted_sum"`, `"max"` |
| `missing(policy, *, value)` | `"skip"`, `"impute"`, `"reject"` |
| `unary_min(axis, min, *, id)` | 한 축의 문턱으로 1단계에서 거른다 |
| `max_per_group(key, max, *, id)` | `groups[key]` 마다 상한 |
| `max_total(max, *, id)` | 전체 상한 |
| `require_at_least(key, value, n, *, id)` | 하한 조건 |
| `coverage_objective()` | `cover` 로 포괄성 최대화 |
| `budget_top_k(k)` | 고정 개수 |
| `budget_tail_mass(epsilon, fallback_k)` | 꼬리 질량으로 유도 |
| `budget_tokens(max)` | `cost` 합의 상한 |
| `admission(axis)` | 1단계 절단 기준 |
| `pool_multiplier(n)` | `M = K * n` |
| `threshold(v)` | 융합 점수 하한 |
| `min_fit(v)` | 적합도 문턱 |
| `rejections(policy, *, n)` | `"keep"`, `"count"`, `"sample"` |
| `run(candidates)` | 반복 가능한 것이면 무엇이든. 생성기면 스트리밍 |

### 결과

```python
out = engine.run(chunks)

for r in out.ranked:
    r.id, r.rank, r.fused
    r.scores        # {"축이름": 값 또는 None}
    r.fusion        # {"method", "k", "missing_policy", "terms"}
    r.constraints   # 통과한 집합 제약 이름들
    r.candidate     # 넘겼던 파이썬 객체 그대로
    r.to_json()

for r in out.rejected:
    r.id, r.reason, r.detail, r.fused, r.candidate

out.rejected_counts   # 사유별 개수. 보관 정책과 무관하게 정확
out.selection         # exact, guarantee, pool_size, pool_exhausted, cut_margin
out.trace             # input_count, pool_capacity, admission_scorer, scorers, budget
out.is_complete()
```

모듈에는 `__version__`, `__status__`, `EngineError`, `tail_mass` 도 있다.

### 비싼 축 콜백의 계약

풀 전체를 목록으로 받아 **같은 길이의 점수 목록**을 돌려준다. 값을 낼 수 없는 자리에는
`None` 을 넣는다.

```python
def cross_encoder(pool):
    pairs = [(query, c["text"]) for c in pool]
    return model.predict(pairs).tolist()   # len(pool) 개

engine.scorer("cross", scale="unbounded", cost="expensive",
              normalize="sigmoid", fn=cross_encoder)
```

길이가 어긋나면 조용히 쓰지 않고 `EngineError` 를 낸다. 콜백 안에서 난 예외는 삼키지 않고
그대로 다시 올라온다.

### 알아 둘 것

- **점수는 32비트 실수다.** 코어가 `f32` 로 계산하므로 `0.4` 를 넣으면 `0.4000000059604645`
  가 돌아온다. 순위와 비교는 그대로 성립하지만 값을 등호로 견주지 말고 허용 오차를 둔다.
- **휠에 `parallel` 을 넣지 않았다.** 콜백이 파이썬 함수라 여러 스레드에서 불러도 전역
  인터프리터 잠금이 직렬화한다. 얻는 것 없이 복잡도만 는다.
- 단항 제약은 `unary_min` 하나만 낸다. 그 밖의 걸러 내기는 파이썬 쪽에서 하는 편이 싸다.
  다만 그렇게 거른 후보는 **탈락 사유에 남지 않는다.**

---

## 12. 파이프라인 통합

### 검색 증강 생성에서 문서 고르기

토큰 예산 안에서 출처 편중을 막고 주제를 넓게 덮는 조각을 고른다.

```python
engine = rmre.Engine()
engine.scorer("similarity")                                    # 벡터 검색이 낸 값
engine.scorer("cross", scale="unbounded", cost="expensive",
              normalize="sigmoid", fn=cross_encoder)           # 재순위 모델
engine.fuse("weighted_sum", weights={"similarity": 0.4, "cross": 0.6})
engine.max_per_group("source", 2)                              # 한 출처가 독식하지 못하게
engine.coverage_objective()                                    # 주제를 넓게
engine.budget_tokens(4000)                                     # 프롬프트에 들어갈 만큼만

out = engine.run(chunks)
context = "\n\n".join(r.candidate["text"] for r in out.ranked)
```

`out.selection["cut_margin"]` 이 작으면 그 선택이 흔들린다는 뜻이므로 로그에 남길 값이다.

### 여러 검색 소스의 결과 합치기

소스마다 점수 체계가 달라 값을 더할 수 없으면 순위만 융합한다. 소스별 순위를 그대로 값으로
바꿔 넣으면 된다.

```python
engine = rmre.Engine()
for name in ("vector", "bm25", "graph"):
    engine.scorer(name, scale="rank")
engine.fuse("rrf", k=60.0)
engine.budget_top_k(20)

out = engine.run([
    {"id": doc_id, "scores": {name: 1.0 / (rank + 1) for name, rank in ranks.items()}}
    for doc_id, ranks in merged.items()
])
```

**중복 제거는 이 엔진의 밖이다.** 같은 문서를 정체성 키로 묶는 것은 후보를 만드는 단계의
일이고, 엔진은 후보가 무엇인지 모른다.

### 캐시 크기 정하기

`tail_mass` 를 엔진과 따로 부를 수 있다. 추정한 지수 하나가 검색 예산과 캐시 크기를 동시에
정한다.

```python
miss_rate = rmre.tail_mass(s, cache_size, total_items)
hit_rate = 1.0 - miss_rate
```

### 감사 로그 남기기

```python
for r in out.ranked:
    audit.write(r.to_json() + "\n")
for r in out.rejected:
    audit.write(json.dumps({"id": r.id, "reason": r.reason, "detail": r.detail}) + "\n")
```

후보가 아주 많으면 `rejections("count")` 로 상세를 끄되 개수는 그대로 정확하다.

---

## 13. 새 채점기와 제약과 목적함수 추가

### 채점기

`scale` 과 `cost` 를 정직하게 선언하는 것이 전부다.

```rust
struct CrossEncoder { model: Model }

impl Scorer<Doc> for CrossEncoder {
    fn id(&self) -> ScorerId { ScorerId::new("cross_encoder") }
    fn scale(&self) -> ScoreScale { ScoreScale::Unbounded }   // 로짓이다
    fn cost(&self) -> ScorerCost { ScorerCost::Expensive }    // 2단계로 밀린다
    fn score(&self, d: &Doc) -> Option<f32> {
        self.model.predict(&d.text).ok()
    }

    // 배치 추론을 쓴다면 이쪽을 재정의한다. 풀 전체가 한 번에 들어온다.
    fn score_batch(&self, docs: &[&Doc]) -> Vec<Option<f32>> {
        let texts: Vec<&str> = docs.iter().map(|d| d.text.as_str()).collect();
        self.model.predict_batch(&texts)
    }
}
```

`score_batch` 는 **입력과 같은 길이를 같은 순서로** 돌려줘야 한다. 길이가 어긋나면 엔진이
`BatchLengthMismatch` 를 낸다. 순서가 어긋나면 엔진이 알아챌 방법이 없으므로 조용히 틀린
순위가 나온다.

### 집합 제약

`is_matroid` 를 참으로 선언하는 것은 약속이다. 확신이 없으면 거짓으로 둔다. 거짓으로 두면
계수가 낮게 잡힐 뿐이지만, 참으로 잘못 두면 결과의 `exact` 가 근거를 잃는다.

```rust
struct AtMostOnePerAuthor;

impl SetConstraint<Doc> for AtMostOnePerAuthor {
    fn id(&self) -> ConstraintId { ConstraintId::new("one_per_author") }
    fn admits(&self, selected: &[&Doc], c: &Doc) -> bool {
        !selected.iter().any(|s| s.author == c.author)
    }
    fn is_matroid(&self) -> bool { true }   // 분할 매트로이드다
}
```

`admits` 는 선택이 커질 때마다 불리므로 `selected` 를 훑는 비용이 그대로 곱해진다. 풀이
수천 건이면 그 안에서 조기 반환을 넣는다.

### 집합 목적함수

`is_submodular` 도 같은 성격의 약속이다.

```rust
struct Redundancy<F> { similarity: F }

impl<F: Fn(&Doc, &Doc) -> f32 + Sync> SetObjective<Doc> for Redundancy<F> {
    fn marginal_gain(&self, selected: &[&Doc], c: &Doc) -> f32 {
        let worst = selected.iter()
            .map(|s| (self.similarity)(s, c))
            .fold(0.0f32, f32::max);
        -worst   // 이미 고른 것과 비슷할수록 손해
    }
    fn is_submodular(&self) -> bool { false }   // 근거를 못 대므로 거짓
}
```

`marginal_gain` 은 `selected` 의 원소 순서에 의존하면 안 된다.

---

## 14. 빌드와 테스트

```bash
cargo build                                     # 기본. 의존성 하나
cargo build --features serde
cargo build --features parallel
cargo test                                      # 러스트 전체
cargo test --features parallel                  # 병렬 빌드에서도 같은 지문
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --all --check
cargo run --release --example throughput        # 규모별 처리량
cargo run --example rag_selection               # 감사 출력까지 보이는 예제
```

Python 쪽은 maturin 으로 설치한 뒤 pytest 를 돌린다.

```bash
maturin develop
pytest tests/test_python_binding.py
```

### 테스트 구성

| 파일 | 무엇을 검사하는가 |
| --- | --- |
| `tests/invariants.rs` | 불변식 넷. 무작위 200라운드 포함 |
| `tests/configuration.rs` | 척도 판정, 빌더 오류, 배치 채점 |
| `tests/optimality.rs` | 495가지 전수 조사한 최적해와의 비교 |
| `tests/budget_policy.rs` | 적응형 예산과 배낭형 선택 |
| `tests/scholar_fusion.rs` | 자매 프로젝트 순위 융합 흡수 관문 |
| `tests/test_python_binding.py` | 파이썬 표면 회귀 |

**개발 의존성이 0이다.** 무작위 불변식 검사는 속성 테스트 크레이트 대신 고정 시드 선형
합동 생성기를 직접 넣었다. 기본 빌드의 의존성 원칙을 개발 의존성에도 적용한 것이고, 시드가
고정이라 실패가 항상 재현된다.

**결정성은 고정 지문 리터럴로 못박혀 있다.** 자기 일관성 확인이 아니라 구성 간 대조다.

```bash
cargo test --test invariants
RAYON_NUM_THREADS=1  cargo test --features parallel --test invariants
RAYON_NUM_THREADS=16 cargo test --features parallel --test invariants
```

셋이 같은 지문을 내야 한다.

---

## 15. 성능과 알려진 한계

### 처리량

릴리스 빌드, 후보당 싼 축 하나와 비싼 축 하나, K는 30, 풀 배수 32.

| 후보 | 풀 | 비싼 호출 | 경과 | 후보당 |
| ---: | ---: | ---: | ---: | ---: |
| 10,000 | 960 | 960 | 2.6ms | 261ns |
| 100,000 | 960 | 960 | 15.4ms | 154ns |
| 1,000,000 | 960 | 960 | 97.8ms | 98ns |

파이썬 표면에서는 같은 조건에 617ms 다. 차이는 엔진이 아니라 후보 사전을 만드는 파이썬 쪽
비용이고, 비싼 축 콜백은 **한 번** 불려 960건을 받는다.

이 값은 단일 실행값이라 신뢰구간이 없다. 추세를 읽는 용도로만 쓴다.

### 구조적으로 남는 것

**두 단계 분할은 근사다.** 1단계 풀 밖으로 밀려난 후보가 집합 제약 때문에 최종 해에
들어가야 했을 수 있다. `pool_exhausted` 로 신호를 보내지만 신호가 없다고 최적이 보장되지는
않는다.

**매트로이드 선언은 신뢰에 의존한다.** 사용자가 직접 구현한 제약의 성질을 엔진이 증명할 수
없다. 잘못 선언하면 `exact` 가 참인데 실제로는 최적이 아닐 수 있다. 기본 제공 제약만 쓰면
이 위험이 없다.

**적응형 예산은 분포 가정에 걸려 있다.** 접근 패턴이 멱법칙이 아니면 유도된 K가 틀린다.
적합도 검정으로 걸러내지만 검정 자체도 표본이 적으면 신뢰도가 떨어진다.

**식별자 중복을 검사하지 않는다.** 같은 실행에서 두 후보가 같은 식별자를 가지면 동점 처리
순서가 정해지지 않아 결정성이 깨진다.

**검증은 전부 합성 입력과 자매 함수 대조다.** 실세계 도메인 데이터로 품질을 재지 않았고,
지금까지 Windows 에서만 실측됐다.

### 하지 않는 것

| 하지 않음 | 이유 |
| --- | --- |
| 신경망 추론 런타임 내장 | 대표적인 후보인 ONNX Runtime 바인딩은 C++ 라이브러리를 링크하므로 외부 함수 인터페이스 호출이 0이라는 원칙이 깨진다. 채점기는 트레잇이므로 호출자가 그 안에서 부르면 된다 |
| 모델 학습과 재학습 | 학습은 다른 생태계의 일이다. 엔진은 결정 경로만 맡는다 |
| 특징 계산 (임베딩, 고유값 분해) | 색인 시점에 미리 계산해 두는 것이 맞다. 질의 시점에 하면 오히려 느려진다 |
| 중복 제거 | 후보를 만드는 단계의 일이다. 엔진은 후보가 무엇인지 모른다 |
| 피드백 저장소와 재학습 루프 | 예측 기록을 내보내는 것까지가 엔진이다 |

첫 줄이 가장 중요하다. 이것을 지키면 기본 빌드가 순수 러스트와 최소 의존성으로 유지되어
폐쇄망에서 그대로 돈다.

### 이 엔진이 필요 없는 경우

정직하게 적어 둔다. 다음 경우에는 쓰지 않는 편이 낫다.

- 후보 수가 K보다 조금 많은 정도일 때. 그냥 정렬하면 된다
- 점수 축이 하나뿐일 때. 융합할 것이 없다
- 집합 제약이 없을 때. 상위 K가 이미 최적이다
- 병목이 선택이 아니라 그 뒤의 생성 단계일 때

---

## 16. 디렉토리 구조

- `README.md` 이 문서
- `Cargo.toml` 패키지와 기능 플래그
- `pyproject.toml` maturin 빌드 설정과 PyPI 메타데이터
- `LICENSE`
- `.github/workflows/release.yml` 태그 push 시 전 플랫폼 휠 빌드와 PyPI 게시
- `src/`
  - `lib.rs` 공개 API 진입점
  - `candidate.rs` 후보와 후보 식별자
  - `score.rs` 채점기 트레잇, 척도, 점수 집합, 정규화기
  - `fuse.rs` 융합 방식과 융합 기록
  - `constraint.rs` 단항 제약, 집합 제약, 매트로이드 판정, 요구 조건
  - `objective.rs` 집합 목적함수와 한계 이득, 포괄성 목적함수
  - `budget.rs` 예산 정책과 적합도 검정
  - `select.rs` 스트리밍 유계 힙, 제약 아래 선택, 보장 계수 판정
  - `evidence.rs` 결과와 탈락 사유, 감사 출력
  - `engine.rs` 빌더와 오케스트레이션
  - `error.rs` 설정 오류와 실현 불가능 판정
  - `python.rs` PyO3 바인딩 (`feature = "python"` 에서만 빌드된다)
- `tests/`
  - `common/mod.rs` 시험용 후보와 고정 시드 난수 생성기
  - `invariants.rs` 결정성, 완전성, 제약 준수, 근거 재현
  - `configuration.rs` 척도 판정, 빌더 오류, 배치 채점
  - `optimality.rs` 전수 조사한 최적해와의 비교
  - `budget_policy.rs` 적응형 예산과 배낭형 선택
  - `scholar_fusion.rs` 자매 프로젝트 순위 융합 흡수 관문
  - `test_python_binding.py` 파이썬 표면 회귀 테스트
- `examples/`
  - `rag_selection.rs` 검색 증강 생성 파이프라인의 문서 선택
  - `throughput.rs` 규모별 처리량 측정

`benches/` 는 두지 않았다. 러스트의 표준 벤치 하네스는 아직 나이틀리 전용이고, 안정 채널에서
쓰려면 criterion 같은 개발 의존성을 들여야 한다. 개발 의존성 0을 지키려고 측정을
`examples/throughput.rs` 로 옮겼다.

단일 크레이트다. 크레이트를 미리 쪼개면 서로를 부르는 배관만 늘어난다.

---

## 17. 라이선스

Apache License 2.0
