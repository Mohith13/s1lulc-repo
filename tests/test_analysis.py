"""Tests for the prototype's calculations, checked against hand-computed values."""

import numpy as np
import pytest

from s1lulc.analysis.area import adjusted_share_series, error_adjusted_area
from s1lulc.analysis.metrics import (
    confusion_matrix,
    mean_iou,
    overall_accuracy,
    per_class_f1,
    per_class_iou,
)
from s1lulc.analysis.significance import fit_noise_floor, flag_significant
from s1lulc.analysis.stats import (
    change_frequency,
    class_shares,
    common_valid_mask,
    gross_net,
    month_to_month,
    total_change,
    transition_matrix,
)
from s1lulc.analysis.temporal import entropy, labels_from_probs, smooth_probabilities

IGN = 255


# --- shares and changes ------------------------------------------------------------------
def three_months():
    # 2x2 maps, classes 0..2. Pixel (1,1) is ignore in month 2 -> excluded everywhere.
    return np.array([
        [[0, 0], [1, 2]],
        [[0, 1], [1, IGN]],
        [[1, 1], [1, 2]],
    ], dtype=np.uint8)


def test_common_mask_excludes_pixel_invalid_in_any_month():
    valid = common_valid_mask(three_months(), IGN)
    assert valid.tolist() == [[True, True], [True, False]]


def test_shares_changes_by_hand():
    maps = three_months()
    valid = common_valid_mask(maps, IGN)          # 3 valid pixels
    shares = class_shares(maps, valid, 3)
    # month1: {0,0,1} -> 66.67/33.33/0 ; month2: {0,1,1} ; month3: {1,1,1}
    np.testing.assert_allclose(shares, [[200 / 3, 100 / 3, 0], [100 / 3, 200 / 3, 0],
                                        [0, 100, 0]])
    np.testing.assert_allclose(shares.sum(1), 100)
    np.testing.assert_allclose(month_to_month(shares), [[-100 / 3, 100 / 3, 0],
                                                        [-100 / 3, 100 / 3, 0]])
    np.testing.assert_allclose(total_change(shares), [-200 / 3, 200 / 3, 0])


def test_no_valid_pixels_raises():
    maps = np.full((2, 2, 2), IGN, np.uint8)
    with pytest.raises(ValueError):
        class_shares(maps, common_valid_mask(maps, IGN), 3)


def test_transitions_and_gross_net():
    maps = three_months()
    valid = common_valid_mask(maps, IGN)
    t = transition_matrix(maps[0], maps[-1], valid, 3)
    # pixels: 0->1, 0->1, 1->1
    np.testing.assert_allclose(t, [[0, 200 / 3, 0], [0, 100 / 3, 0], [0, 0, 0]])
    g = gross_net(t)
    np.testing.assert_allclose(g.gain, [0, 200 / 3, 0])
    np.testing.assert_allclose(g.loss, [200 / 3, 0, 0])
    np.testing.assert_allclose(g.net, total_change(class_shares(maps, valid, 3)))
    assert g.changed_total == pytest.approx(200 / 3)


def test_gross_change_hidden_by_zero_net():
    # Two pixels swap classes: net change 0 pp, but 100% of the area changed.
    first = np.array([[0, 1]], np.uint8)
    last = np.array([[1, 0]], np.uint8)
    g = gross_net(transition_matrix(first, last, np.ones((1, 2), bool), 2))
    np.testing.assert_allclose(g.net, [0, 0])
    assert g.changed_total == pytest.approx(100)


def test_change_frequency():
    maps = three_months()
    f = change_frequency(maps, common_valid_mask(maps, IGN))
    assert f.tolist() == [[1, 1], [0, -1]]


# --- error-adjusted area (Olofsson et al. 2014) -------------------------------------------
def test_perfect_classifier_returns_mapped_area():
    cm = np.diag([50, 80, 30])
    est = error_adjusted_area(cm, np.array([0.2, 0.5, 0.3]))
    np.testing.assert_allclose(est.proportion, [0.2, 0.5, 0.3])
    np.testing.assert_allclose(est.se, 0)


def test_two_class_hand_example():
    # rows = map class, cols = reference.  W = (0.6, 0.4)
    cm = np.array([[80, 20], [10, 90]])
    est = error_adjusted_area(cm, np.array([0.6, 0.4]))
    # p_.0 = 0.6*0.8 + 0.4*0.1 = 0.52 ; p_.1 = 0.48
    np.testing.assert_allclose(est.proportion, [0.52, 0.48])
    # SE = sqrt(0.36*0.8*0.2/99 + 0.16*0.1*0.9/99) = sqrt(0.072/99)
    np.testing.assert_allclose(est.se, [np.sqrt(0.072 / 99)] * 2)
    assert est.proportion.sum() == pytest.approx(1.0)


def test_accepts_percentages_and_effective_n_widens_ci():
    cm = np.array([[8000, 2000], [1000, 9000]])
    a = error_adjusted_area(cm, np.array([60.0, 40.0]))
    b = error_adjusted_area(cm, np.array([0.6, 0.4]), effective_n=100)
    np.testing.assert_allclose(a.proportion, b.proportion)
    assert np.all(b.se > a.se)
    np.testing.assert_allclose(b.se, [np.sqrt(0.072 / 99)] * 2)


def test_class_without_samples_is_left_unadjusted():
    cm = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 0]])
    est = error_adjusted_area(cm, np.array([0.3, 0.3, 0.4]))
    np.testing.assert_allclose(est.proportion, [0.3, 0.3, 0.4])


def test_adjusted_series_shape():
    cm = np.array([[80, 20], [10, 90]])
    adj, ci = adjusted_share_series(cm, np.array([[60, 40], [50, 50]]))
    assert adj.shape == ci.shape == (2, 2)
    np.testing.assert_allclose(adj.sum(1), 100)


# --- noise floor ---------------------------------------------------------------------------
def test_noise_floor_and_flags():
    stable = [np.tile([50.0, 50.0], (12, 1)) + np.array([[1, -1], [-1, 1]] * 6)]
    nf = fit_noise_floor(stable, ["a", "b"], quantile=95)
    np.testing.assert_allclose(nf.month_step_pp, [2, 2])     # every monthly step is +-2 pp
    np.testing.assert_allclose(nf.total_pp, [2, 2])          # month1=+1, month12=-1
    flags = flag_significant(np.array([1.5, -3.0]), nf.month_step_pp)
    assert flags.tolist() == [False, True]


# --- temporal smoothing and uncertainty ----------------------------------------------------
def test_smoothing_removes_one_month_flip():
    # class 0 every month, except month 3 where class 1 narrowly wins
    p = np.tile(np.array([0.8, 0.2], np.float32)[None, :, None, None], (5, 1, 1, 1))
    p[2, :, 0, 0] = [0.45, 0.55]
    raw = labels_from_probs(p, np.ones((1, 1), bool), IGN)[:, 0, 0]
    sm = labels_from_probs(smooth_probabilities(p, 3), np.ones((1, 1), bool), IGN)[:, 0, 0]
    assert raw.tolist() == [0, 0, 1, 0, 0]
    assert sm.tolist() == [0, 0, 0, 0, 0]


def test_smoothing_edges_and_window_one():
    p = np.random.default_rng(1).dirichlet([1, 1, 1], size=(4, 2, 2)).transpose(0, 3, 1, 2)
    s = smooth_probabilities(p, 3)
    np.testing.assert_allclose(s[0], (p[0] + p[1]) / 2, rtol=1e-6)
    np.testing.assert_allclose(s.sum(1), 1, rtol=1e-5)
    np.testing.assert_allclose(smooth_probabilities(p, 1), p)
    with pytest.raises(ValueError):
        smooth_probabilities(p, 2)


def test_entropy_bounds():
    p = np.zeros((3, 1, 2), np.float32)
    p[:, 0, 0] = [1, 0, 0]
    p[:, 0, 1] = [1 / 3, 1 / 3, 1 / 3]
    h = entropy(p, axis=0)
    assert h[0, 0] == pytest.approx(0, abs=1e-6) and h[0, 1] == pytest.approx(1, abs=1e-6)


# --- accuracy metrics ----------------------------------------------------------------------
def test_metrics_by_hand():
    y = np.array([0, 0, 1, 1, 2, IGN])
    p = np.array([0, 1, 1, 1, 0, 2])
    cm = confusion_matrix(y, p, 3, ignore_index=IGN)
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 0]]
    assert overall_accuracy(cm) == pytest.approx(3 / 5)
    # IoU: c0 = 1/(1+1+1)=1/3, c1 = 2/(2+1+0)=2/3, c2 = 0/(0+0+1)=0
    np.testing.assert_allclose(per_class_iou(cm), [1 / 3, 2 / 3, 0])
    np.testing.assert_allclose(per_class_f1(cm), [0.5, 0.8, 0])
    assert mean_iou(cm) == pytest.approx(1 / 3)
