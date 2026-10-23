"""F006: Delta/se estimators + O5 verdict (design sec 9). Tests first.

Trials are UNPAIRED across conditions: Delta is a difference of GROUP means,
never per-trial. All fixtures below are synthetic arrays.
"""

import math

import numpy as np
import torch

from structx import evaluate as ev
from structx.physics import DT, physics_loss

MATCHED = [1.0, 2.0, 3.0]
SHIFTED = [2.0, 3.0, 4.0]
# Hand values: means 2.0/3.0 -> Delta = 1.0;
# sample var (ddof=1) = 1.0 each side -> se = sqrt(1/3 + 1/3) = sqrt(2/3).
DELTA_HAND = 1.0
SE_HAND = math.sqrt(2.0 / 3.0)


def test_delta_m_hand_computed_groups():
    assert ev.delta_m(MATCHED, SHIFTED) == DELTA_HAND


def test_se_delta_hand_value():
    assert ev.se_delta(MATCHED, SHIFTED) == math.sqrt(2.0 / 3.0)
    assert abs(ev.se_delta(MATCHED, SHIFTED) - SE_HAND) < 1e-12


def test_delta_r_sign_flipped():
    matched_r = [0.95, 0.93, 0.97]
    shifted_r = [0.80, 0.82, 0.78]
    got = ev.delta_r(matched_r, shifted_r)
    assert got == np.mean(matched_r) - np.mean(shifted_r)
    assert got > 0  # shift hurts r -> positive gap, opposite of delta_m convention
    assert ev.delta_r(SHIFTED, MATCHED) == -ev.delta_m(SHIFTED, MATCHED)


def test_o5_boundary_wins_just_above_sigma():
    # se = 0.5 each -> sigma_R = sqrt(0.5) ~= 0.7071; R = 0.8 clears it.
    d = ev.o5_verdict(delta_base=1.0, delta_phys=0.2, se_base=0.5, se_phys=0.5)
    assert d["R"] == 0.8
    assert d["sigma_R"] == math.sqrt(0.5)
    assert d["wins"] is True
    assert "non-identifiable" not in d["statement"].lower()


def test_o5_boundary_non_identifiable_just_below_sigma():
    # Same sigma_R ~= 0.7071; R = 0.6 does not clear it -> negative counts.
    d = ev.o5_verdict(delta_base=1.0, delta_phys=0.4, se_base=0.5, se_phys=0.5)
    assert d["R"] == 0.6
    assert d["wins"] is False
    assert "non-identifiable" in d["statement"].lower()


def test_rq4_true_and_false():
    assert ev.rq4_compare(freq_delta=0.01, rmse_delta=0.05) is True
    assert ev.rq4_compare(freq_delta=0.05, rmse_delta=0.01) is False


def test_matched_degradation_harm_sign():
    harmed = ev.matched_degradation_report(
        matched_base=[0.10, 0.12, 0.11], matched_phys=[0.13, 0.14, 0.15]
    )
    assert harmed["harm"] > 0
    assert harmed["harmed"] is True
    helped = ev.matched_degradation_report(
        matched_base=[0.13, 0.14, 0.15], matched_phys=[0.10, 0.12, 0.11]
    )
    assert helped["harm"] < 0
    assert helped["harmed"] is False


def test_metric_kernels():
    assert ev.rmse([0.0, 0.0], [1.0, 1.0]) == 1.0
    assert ev.rmse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
    assert ev.mae([0.0, 0.0], [1.0, -1.0]) == 1.0
    assert ev.pearson_r([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0
    assert ev.pearson_r([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == -1.0
    assert ev.pearson_r([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == 0.0  # zero variance
    assert ev.rel_freq_error(51.0, 50.0) == 0.02


def test_summarize_trials():
    s = ev.summarize_trials([1.0, 2.0, 3.0])
    assert s == {"mean": 2.0, "std": 1.0}  # sample std, ddof=1 like se_delta


def test_l_phys_reuses_physics_stencil():
    torch.manual_seed(0)
    q = torch.sin(2.0 * math.pi * 50.0 * torch.arange(1000, dtype=torch.float32) * DT)
    m, c, k = 1.0, 0.5, (2.0 * math.pi * 50.0) ** 2
    assert ev.l_phys(q, m, c, k) == physics_loss(q, m, c, k).item()


def test_measure_latency_ms():
    ms = ev.measure_latency_ms(lambda x: x + 1, np.zeros(1000), reps=5)
    assert isinstance(ms, float) and ms >= 0.0
