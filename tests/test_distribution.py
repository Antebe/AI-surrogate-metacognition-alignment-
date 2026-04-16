import numpy as np
import pytest

from distribution import (
    category_distribution,
    wasserstein_4cat,
    js_divergence_4cat,
    CATEGORIES,
)


def test_category_distribution_normalizes_to_one():
    cats = ["Correct", "Correct", "Lure", "Unsure", "Other"]
    dist = category_distribution(cats)
    assert set(dist) == set(CATEGORIES)
    assert dist["Correct"] == pytest.approx(0.4)
    assert dist["Lure"] == pytest.approx(0.2)
    assert sum(dist.values()) == pytest.approx(1.0)


def test_category_distribution_handles_missing_categories():
    dist = category_distribution(["Correct"] * 3)
    assert dist["Correct"] == 1.0
    assert dist["Lure"] == 0.0


def test_category_distribution_handles_empty_input():
    dist = category_distribution([])
    for c in CATEGORIES:
        assert dist[c] == 0.0


def test_wasserstein_identical_is_zero():
    a = {"Correct": .5, "Lure": .3, "Unsure": .1, "Other": .1}
    assert wasserstein_4cat(a, a) == pytest.approx(0.0)


def test_wasserstein_adjacent_shift_is_one():
    a = {"Correct": 1.0, "Lure": 0.0, "Unsure": 0.0, "Other": 0.0}
    b = {"Correct": 0.0, "Lure": 1.0, "Unsure": 0.0, "Other": 0.0}
    assert wasserstein_4cat(a, b) == pytest.approx(1.0)


def test_wasserstein_max_spread_is_three():
    a = {"Correct": 1.0, "Lure": 0.0, "Unsure": 0.0, "Other": 0.0}
    b = {"Correct": 0.0, "Lure": 0.0, "Unsure": 0.0, "Other": 1.0}
    assert wasserstein_4cat(a, b) == pytest.approx(3.0)


def test_js_divergence_identical_is_zero():
    a = {"Correct": .5, "Lure": .3, "Unsure": .1, "Other": .1}
    assert js_divergence_4cat(a, a) == pytest.approx(0.0, abs=1e-9)


def test_js_divergence_symmetric():
    a = {"Correct": .5, "Lure": .3, "Unsure": .1, "Other": .1}
    b = {"Correct": .25, "Lure": .25, "Unsure": .25, "Other": .25}
    ab = js_divergence_4cat(a, b)
    ba = js_divergence_4cat(b, a)
    assert ab == pytest.approx(ba, abs=1e-12)
