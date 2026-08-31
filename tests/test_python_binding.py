"""파이썬 바인딩 회귀 테스트.

러스트 쪽 불변식 넷을 파이썬 표면에서도 그대로 확인한다. 표면이 다르다고 성질이
달라지면 안 되기 때문이다.

    maturin develop --features python
    pytest tests/test_python_binding.py
"""

import math

import pytest

import rust_multi_ranking_engine as rmre


# ── 도구 ──────────────────────────────────────────────────────────


def doc(i, sim, source="arxiv", cost=1, cover=None, cross=None):
    d = {
        "id": i,
        "scores": {"similarity": sim},
        "groups": {"source": source},
        "cost": cost,
    }
    if cover is not None:
        d["cover"] = cover
    if cross is not None:
        d["scores"]["cross"] = cross
    return d


def simple_engine(k=3):
    e = rmre.Engine()
    e.scorer("similarity")
    e.fuse("weighted_sum")
    e.budget_top_k(k)
    return e


# ── 모듈 표면 ─────────────────────────────────────────────────────


def test_module_reports_its_version_and_scope():
    assert isinstance(rmre.__version__, str)
    assert rmre.__version__.count(".") >= 2
    assert "fusion" in rmre.__status__
    assert issubclass(rmre.EngineError, Exception)


def test_tail_mass_is_callable_on_its_own():
    # 지수가 크면 앞쪽에 질량이 몰려 꼬리가 얇다.
    assert rmre.tail_mass(2.0, 10, 1000) < rmre.tail_mass(0.5, 10, 1000)
    assert rmre.tail_mass(1.0, 1000, 1000) == 0.0


# ── 기본 동작 ─────────────────────────────────────────────────────


def test_top_k_picks_the_highest_scores():
    out = simple_engine(2).run([doc(1, 0.2), doc(2, 0.9), doc(3, 0.5)])
    assert [r.id for r in out.ranked] == ["2", "3"]
    assert out.ranked[0].rank == 1
    assert out.ranked[0].fused == pytest.approx(0.9)


def test_every_candidate_lands_in_exactly_one_place():
    out = simple_engine(2).run([doc(i, i / 10) for i in range(1, 8)])
    assert out.is_complete()
    total = sum(out.rejected_counts.values())
    assert len(out.ranked) + total == out.trace["input_count"] == 7


def test_the_original_python_object_comes_back():
    d = doc(1, 0.9)
    out = simple_engine(1).run([d])
    assert out.ranked[0].candidate is d


def test_a_generator_is_accepted_without_building_a_list():
    out = simple_engine(2).run(doc(i, i / 100) for i in range(1, 50))
    assert len(out.ranked) == 2
    assert out.trace["input_count"] == 49


def test_scores_may_be_given_as_a_sequence():
    e = rmre.Engine()
    e.scorer("a")
    e.scorer("b")
    e.fuse("weighted_sum")
    e.budget_top_k(1)
    out = e.run([{"id": 1, "scores": [0.4, 0.6]}])
    # 코어가 32비트 실수로 계산하므로 등호가 아니라 허용 오차로 견준다.
    scores = out.ranked[0].scores
    assert set(scores) == {"a", "b"}
    assert scores["a"] == pytest.approx(0.4)
    assert scores["b"] == pytest.approx(0.6)


# ── 척도 판정 ─────────────────────────────────────────────────────


def test_an_unbounded_axis_is_refused_by_a_weighted_sum():
    e = rmre.Engine()
    e.scorer("logit", scale="unbounded")
    e.fuse("weighted_sum")
    with pytest.raises(ValueError, match="척도"):
        e.run([doc(1, 0.5)])


def test_a_normalizer_makes_the_same_axis_admissible():
    e = rmre.Engine()
    e.scorer("logit", scale="unbounded", normalize="sigmoid")
    e.fuse("weighted_sum")
    e.budget_top_k(1)
    out = e.run([{"id": 1, "scores": {"logit": 2.0}}])
    assert out.ranked[0].fused == pytest.approx(1 / (1 + math.exp(-2.0)), abs=1e-6)


def test_rank_fusion_takes_any_mix_of_scales():
    e = rmre.Engine()
    e.scorer("logit", scale="unbounded")
    e.scorer("prob", scale="unit")
    e.fuse("rrf")
    e.budget_top_k(2)
    out = e.run(
        [
            {"id": 1, "scores": {"logit": 8.0, "prob": 0.1}},
            {"id": 2, "scores": {"logit": -3.0, "prob": 0.9}},
        ]
    )
    assert len(out.ranked) == 2
    assert out.ranked[0].fusion["method"] == "rrf"


def test_minmax_needs_a_range():
    e = rmre.Engine()
    with pytest.raises(ValueError, match="normalize_range"):
        e.scorer("x", normalize="minmax")


# ── 결측 ──────────────────────────────────────────────────────────


def test_a_missing_axis_is_skipped_and_stays_visible():
    e = rmre.Engine()
    e.scorer("a")
    e.scorer("b")
    e.fuse("weighted_sum")
    e.budget_top_k(1)
    out = e.run([{"id": 1, "scores": {"a": 0.8}}])

    r = out.ranked[0]
    assert r.scores["a"] == pytest.approx(0.8)
    assert r.scores["b"] is None
    assert r.fused == pytest.approx(0.8)  # 남은 축으로 재정규화된다
    kinds = {t["scorer"]: t["input"] for t in r.fusion["terms"]}
    assert kinds == {"a": "value", "b": "skipped"}


def test_a_required_axis_rejects_instead_of_scoring_zero():
    e = rmre.Engine()
    e.scorer("a")
    e.fuse("weighted_sum")
    e.missing("reject")
    e.budget_top_k(5)
    out = e.run([{"id": 1, "scores": {"a": 0.5}}, {"id": 2, "scores": {}}])

    assert [r.id for r in out.ranked] == ["1"]
    assert out.rejected_counts["not_scored"] == 1
    assert out.rejected[0].reason == "not_scored"
    assert out.rejected[0].detail == "a"
    assert out.is_complete()


def test_imputation_fills_the_value_but_keeps_the_hole_visible():
    e = rmre.Engine()
    e.scorer("a")
    e.fuse("weighted_sum")
    e.missing("impute", value=0.25)
    e.budget_top_k(1)
    out = e.run([{"id": 1, "scores": {}}])

    r = out.ranked[0]
    assert r.scores == {"a": None}
    assert r.fused == pytest.approx(0.25)
    assert r.fusion["terms"][0]["input"] == "imputed"


# ── 집합 제약 ─────────────────────────────────────────────────────


def test_a_group_limit_lets_a_lower_score_in():
    """상위 K 가 정답이 아니게 되는 자리. 점수 1·2·3위가 전부 같은 출처다."""
    out = None
    e = rmre.Engine()
    e.scorer("similarity")
    e.fuse("weighted_sum")
    e.max_per_group("source", 2)
    e.budget_top_k(3)
    out = e.run(
        [
            doc(1, 0.95, "arxiv"),
            doc(2, 0.92, "arxiv"),
            doc(3, 0.90, "arxiv"),
            doc(4, 0.60, "blog"),
        ]
    )

    assert [r.id for r in out.ranked] == ["1", "2", "4"]
    assert out.selection["exact"] is True
    assert out.rejected_counts["set_constraint"] == 1
    assert out.rejected[0].reason == "set_constraint"
    assert out.rejected[0].detail == "source"
    assert out.ranked[0].constraints == ["source"]


def test_a_lower_bound_requirement_is_filled_by_swapping():
    e = rmre.Engine()
    e.scorer("similarity")
    e.fuse("weighted_sum")
    e.require_at_least("source", "news", 1)
    e.budget_top_k(2)
    out = e.run([doc(1, 0.9), doc(2, 0.8), doc(3, 0.1, "news")])

    sources = {r.candidate["groups"]["source"] for r in out.ranked}
    assert "news" in sources
    assert len(out.ranked) == 2
    # 교체가 일어났으면 최적이라고 말하지 않는다.
    assert out.selection["exact"] is False


def test_an_unfillable_requirement_raises():
    e = rmre.Engine()
    e.scorer("similarity")
    e.fuse("weighted_sum")
    e.require_at_least("source", "nowhere", 2)
    e.budget_top_k(2)
    with pytest.raises(rmre.EngineError, match="요구 조건"):
        e.run([doc(1, 0.9), doc(2, 0.8)])


def test_a_unary_minimum_filters_before_scoring():
    e = rmre.Engine()
    e.scorer("similarity")
    e.fuse("weighted_sum")
    e.unary_min("similarity", 0.5, id="min_similarity")
    e.budget_top_k(5)
    out = e.run([doc(1, 0.9), doc(2, 0.1)])

    assert [r.id for r in out.ranked] == ["1"]
    assert out.rejected[0].reason == "unary_constraint"
    assert out.rejected[0].detail == "min_similarity"


# ── 목적함수와 예산 ───────────────────────────────────────────────


def test_the_coverage_objective_spreads_the_selection():
    e = rmre.Engine()
    e.scorer("similarity")
    e.fuse("weighted_sum")
    e.coverage_objective()
    e.budget_top_k(2)
    out = e.run(
        [
            doc(1, 0.90, cover=["a", "b"]),
            doc(2, 0.89, cover=["a", "b"]),
            doc(3, 0.50, cover=["c", "d"]),
        ]
    )

    # 2번은 1번과 같은 주제만 덮으므로 한계 이득이 0 이다. 점수가 낮아도 3번이 이긴다.
    assert [r.id for r in out.ranked] == ["1", "3"]
    assert out.selection["guarantee"] == pytest.approx(1 - 1 / math.e, abs=1e-6)


def test_the_token_budget_is_never_exceeded():
    e = rmre.Engine()
    e.scorer("similarity")
    e.fuse("weighted_sum")
    e.budget_tokens(10)
    out = e.run([doc(i, 0.5, cost=4) for i in range(1, 10)])

    spent = sum(r.candidate["cost"] for r in out.ranked)
    assert spent <= 10
    assert out.selection["exact"] is False
    assert out.selection["guarantee"] == pytest.approx(0.5)


def test_tail_mass_budget_derives_its_own_k_and_shows_the_fit():
    e = rmre.Engine()
    e.scorer("similarity")
    e.fuse("weighted_sum")
    e.budget_tail_mass(0.1, 99)
    e.pool_multiplier(64)
    out = e.run([{"id": i, "scores": {"similarity": (i + 1) ** -1.2}} for i in range(400)])

    budget = out.trace["budget"]
    assert budget["fallback"] is False
    assert budget["s"] == pytest.approx(1.2, abs=0.15)
    assert budget["derived_k"] != 99
    assert len(out.ranked) == budget["derived_k"]


def test_a_non_power_law_falls_back_and_says_why():
    e = rmre.Engine()
    e.scorer("similarity")
    e.fuse("weighted_sum")
    e.budget_tail_mass(0.1, 7)
    e.min_fit(0.999)
    out = e.run([{"id": i, "scores": {"similarity": 0.5}} for i in range(300)])

    assert out.trace["budget"]["fallback"] is True
    assert out.trace["budget"]["reason"] == "poor_fit"
    assert len(out.ranked) == 7


# ── 비싼 축 콜백 ──────────────────────────────────────────────────


def test_the_expensive_callback_gets_the_whole_pool_in_one_call():
    """캐스케이드의 존재 이유가 파이썬 표면에서도 살아 있는지 본다."""
    calls = []

    def cross(pool):
        calls.append(len(pool))
        return [1.0 - i / len(pool) for i, _ in enumerate(pool)]

    e = rmre.Engine()
    e.scorer("similarity")
    e.scorer("cross", cost="expensive", fn=cross)
    e.fuse("weighted_sum")
    e.budget_top_k(2)
    e.pool_multiplier(4)
    out = e.run([doc(i, i / 1000) for i in range(1000)])

    # 후보는 1,000 건인데 콜백은 한 번, 풀 크기만큼만 받는다.
    assert calls == [8]
    assert out.trace["pool_capacity"] == 8
    assert out.trace["scorers"][1]["calls"] == 8
    assert out.trace["scorers"][1]["cost"] == "expensive"
    assert len(out.ranked) == 2


def test_the_callback_receives_the_original_objects():
    seen = []

    def cross(pool):
        seen.extend(p["id"] for p in pool)
        return [0.5] * len(pool)

    e = rmre.Engine()
    e.scorer("similarity")
    e.scorer("cross", cost="expensive", fn=cross)
    e.fuse("weighted_sum")
    e.budget_top_k(1)
    e.pool_multiplier(2)
    e.run([doc(7, 0.9), doc(8, 0.8)])

    assert set(seen) == {7, 8}


def test_a_callback_may_return_none_for_unscorable_items():
    e = rmre.Engine()
    e.scorer("similarity")
    e.scorer("cross", cost="expensive", fn=lambda pool: [None] * len(pool))
    e.fuse("weighted_sum")
    e.budget_top_k(1)
    out = e.run([doc(1, 0.8)])

    assert out.ranked[0].scores["cross"] is None
    assert out.ranked[0].fused == pytest.approx(0.8)


def test_a_callback_of_the_wrong_length_is_an_error_not_a_silent_wrong_answer():
    e = rmre.Engine()
    e.scorer("similarity")
    e.scorer("cross", cost="expensive", fn=lambda pool: [0.5])
    e.fuse("weighted_sum")
    e.budget_top_k(2)
    e.pool_multiplier(4)
    with pytest.raises(rmre.EngineError, match="배치 결과 길이"):
        e.run([doc(1, 0.9), doc(2, 0.8), doc(3, 0.7)])


def test_an_exception_inside_the_callback_is_not_swallowed():
    def broken(pool):
        raise KeyError("모델이 없다")

    e = rmre.Engine()
    e.scorer("similarity")
    e.scorer("cross", cost="expensive", fn=broken)
    e.fuse("weighted_sum")
    e.budget_top_k(1)
    with pytest.raises(KeyError, match="모델이 없다"):
        e.run([doc(1, 0.9)])


def test_a_cheap_axis_cannot_take_a_callback():
    e = rmre.Engine()
    with pytest.raises(ValueError, match="expensive"):
        e.scorer("x", fn=lambda pool: [])


# ── 근거 ──────────────────────────────────────────────────────────


def test_the_trace_alone_reproduces_the_fused_score():
    e = rmre.Engine()
    e.scorer("a")
    e.scorer("b")
    e.fuse("weighted_sum", weights={"a": 0.25, "b": 0.75})
    e.budget_top_k(3)
    out = e.run([{"id": i, "scores": {"a": i / 10, "b": 1 - i / 10}} for i in range(1, 4)])

    for r in out.ranked:
        again = sum(t["contribution"] for t in r.fusion["terms"] if t["input"] != "skipped")
        assert again == pytest.approx(r.fused, abs=1e-6)


def test_the_audit_json_carries_the_missing_axis_and_its_policy():
    e = rmre.Engine()
    e.scorer("a")
    e.scorer("b")
    e.fuse("weighted_sum")
    e.budget_top_k(1)
    out = e.run([{"id": "doc-3141", "scores": {"a": 0.94}}])

    import json

    parsed = json.loads(out.ranked[0].to_json())
    assert parsed["candidate"] == "doc-3141"
    assert parsed["scores"]["a"] == pytest.approx(0.94)
    assert parsed["scores"]["b"] is None
    assert parsed["missing_policy"] == "skip"


def test_out_of_pool_rejections_are_reported():
    e = simple_engine(2)
    e.pool_multiplier(1)
    out = e.run([doc(i, i / 100) for i in range(20)])

    assert out.selection["pool_size"] == 2
    assert out.rejected_counts["out_of_pool"] == 18
    assert out.is_complete()


def test_counts_stay_exact_when_details_are_dropped():
    full = simple_engine(2).run([doc(i, i / 100) for i in range(50)])

    e = simple_engine(2)
    e.rejections("count")
    counted = e.run([doc(i, i / 100) for i in range(50)])

    assert counted.rejected == []
    assert counted.rejected_counts == full.rejected_counts
    assert counted.is_complete()

    e = simple_engine(2)
    e.rejections("sample", n=3)
    sampled = e.run([doc(i, i / 100) for i in range(50)])
    assert len(sampled.rejected) == 3
    assert sampled.rejected_counts == full.rejected_counts


# ── 결정성 ────────────────────────────────────────────────────────


def test_the_same_input_gives_the_same_output_every_time():
    docs = [doc(i, (i * 37 % 101) / 101, source=["a", "b", "c"][i % 3]) for i in range(200)]

    def fingerprint():
        e = rmre.Engine()
        e.scorer("similarity")
        e.fuse("weighted_sum")
        e.max_per_group("source", 3)
        e.budget_top_k(8)
        out = e.run(docs)
        return [(r.id, round(r.fused, 6)) for r in out.ranked]

    first = fingerprint()
    for _ in range(5):
        assert fingerprint() == first


def test_ties_break_on_the_identifier_not_on_arrival_order():
    docs = [doc(i, 0.5) for i in (30, 10, 20)]
    e = simple_engine(2)
    e.pool_multiplier(64)
    out = e.run(docs)
    assert [r.id for r in out.ranked] == ["10", "20"]


# ── 입력 오류 ─────────────────────────────────────────────────────


def test_a_candidate_without_an_id_is_an_error():
    with pytest.raises(ValueError, match="'id'"):
        simple_engine().run([{"scores": {"similarity": 0.5}}])


def test_a_non_dict_candidate_is_an_error():
    with pytest.raises(ValueError, match="사전"):
        simple_engine().run([("doc", 0.5)])


def test_an_engine_without_scorers_is_an_error():
    with pytest.raises(ValueError, match="축이 하나도 없다"):
        rmre.Engine().run([])


def test_an_unknown_axis_in_unary_min_is_an_error():
    e = simple_engine()
    e.unary_min("ghost", 0.5)
    with pytest.raises(ValueError, match="ghost"):
        e.run([doc(1, 0.5)])
