"""Known-effect DGP and Monte Carlo reporting checks, without model rankings."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from econ_causal_lab.simulation import SCENARIOS, generate, monte_carlo


class SimulationTests(unittest.TestCase):
    def test_every_scenario_is_reproducible_and_exposes_only_observed_covariates(self):
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario):
                first, second = generate(1000, 17, scenario), generate(1000, 17, scenario)
                for a, b in zip(first[:3], second[:3]):
                    assert_array_equal(a, b)
                x, y, d, truth = first
                self.assertEqual(x.shape, (1000, 5))
                self.assertEqual(y.shape, (1000,))
                self.assertEqual(set(d), {0., 1.})
                self.assertTrue(np.isfinite(x).all() and np.isfinite(y).all())
                self.assertEqual(truth, 2000.)

    def test_individual_potential_outcomes_differ_by_exactly_2000(self):
        # Intervene on D only while consuming the same RNG draws. All X,
        # hidden confounding, and outcome noise stay identical under both arms.
        real_default_rng = np.random.default_rng

        class InterventionRNG:
            def __init__(self, seed, arm):
                self.rng, self.arm = real_default_rng(seed), arm

            def normal(self, *args, **kwargs):
                return self.rng.normal(*args, **kwargs)

            def binomial(self, *args, **kwargs):
                draw = self.rng.binomial(*args, **kwargs)
                return np.full_like(draw, self.arm)

        for scenario in SCENARIOS:
            samples = []
            for arm in (0, 1):
                with patch("econ_causal_lab.simulation.np.random.default_rng", side_effect=lambda seed, arm=arm: InterventionRNG(seed, arm)):
                    samples.append(generate(500, 91, scenario))
            with self.subTest(scenario=scenario):
                assert_array_equal(samples[0][0], samples[1][0])
                assert_array_equal(samples[0][2], np.zeros(500))
                assert_array_equal(samples[1][2], np.ones(500))
                assert_allclose(samples[1][1] - samples[0][1], 2000., rtol=0, atol=2e-12)

    def test_scenarios_share_observed_covariate_draws(self):
        good = generate(1000, 28, "good_overlap")
        for scenario in ("weak_overlap", "hidden_confounding"):
            altered = generate(1000, 28, scenario)
            assert_array_equal(good[0], altered[0])
            self.assertFalse(np.array_equal(good[2], altered[2]))

    def test_invalid_scenario_size_and_repetitions_are_rejected(self):
        for n, scenario in ((19, "good_overlap"), (100, "missing")):
            with self.subTest(n=n, scenario=scenario), self.assertRaises(ValueError):
                generate(n, scenario=scenario)
        for repetitions in (0, 1, -1):
            with self.subTest(repetitions=repetitions), self.assertRaises(ValueError):
                monte_carlo(repetitions=repetitions)

    def test_monte_carlo_aggregation_and_paired_seeds_are_correct(self):
        # Test aggregation using two fixed errors (-100,+300), one covered.
        # This avoids claiming one estimated method must beat another in 2 draws.
        calls = []

        def fake_generate(n, seed, scenario):
            calls.append((n, seed, scenario))
            x = np.zeros((n, 5))
            y = np.full(n, seed)
            d = np.tile([0., 1.], n // 2)
            return x, y, d, 2000.

        def fake_result(y, *args, **kwargs):
            estimate = 1900. if y[0] == 50 else 2300.
            return dict(estimate=estimate, se=100., ci_low=estimate - 196., ci_high=estimate + 196.)

        nuisance = SimpleNamespace(propensity=np.full(20, .5), m0=np.zeros(20))
        with patch("econ_causal_lab.simulation.generate", side_effect=fake_generate), \
             patch("econ_causal_lab.simulation.mean_difference", side_effect=fake_result), \
             patch("econ_causal_lab.simulation.cross_fit", return_value=nuisance) as fit, \
             patch("econ_causal_lab.simulation.aipw_att", side_effect=lambda *a, **k: (fake_result(*a, **k), np.zeros(20))):
            summary, records = monte_carlo(repetitions=2, n=20, folds=2, seed=50)
        self.assertEqual(len(summary), 9)
        self.assertEqual(len(records), 18)
        self.assertEqual(calls, [(20, seed, scenario) for scenario in SCENARIOS for seed in (50, 51)])
        self.assertEqual(fit.call_count, 12)
        for pair_start in range(0, len(fit.call_args_list), 2):
            first, second = fit.call_args_list[pair_start:pair_start + 2]
            self.assertEqual(first.kwargs, second.kwargs)
            self.assertEqual(first.kwargs["folds"], 2)
            self.assertIs(first.args[0], second.args[0])
        for row in summary:
            self.assertAlmostEqual(row["bias"], 100.)
            self.assertAlmostEqual(row["rmse"], np.sqrt(50000))
            self.assertEqual(row["coverage"], .5)
            self.assertAlmostEqual(row["coverage_mc_se"], np.sqrt(.5 * .5 / 2))
            self.assertAlmostEqual(row["mean_ci_width"], 392.)
        for scenario in SCENARIOS:
            for repetition in (0, 1):
                paired = [r for r in records if r["scenario"] == scenario and r["repetition"] == repetition]
                self.assertEqual({r["seed"] for r in paired}, {50 + repetition})
                self.assertEqual({r["truth"] for r in paired}, {2000.})


if __name__ == "__main__":
    unittest.main()
