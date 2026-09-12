"""Numerical identities and data-isolation contracts; no downloaded data."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from econ_causal_lab.estimators import (
    aipw_att, cross_fit, diagnostics, learners, mean_difference, validate,
)


class SpyClassifier(ClassifierMixin, BaseEstimator):
    """Record row IDs globally so records survive sklearn.clone."""

    records = []

    def fit(self, x, d):
        self.record_ = {"train": x[:, 0].astype(int), "target": np.array(d)}
        type(self).records.append(self.record_)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, x):
        self.record_["test"] = x[:, 0].astype(int)
        p = .2 + x[:, 0] / 100
        return np.column_stack((1 - p, p))


class SpyRegressor(RegressorMixin, BaseEstimator):
    records = []

    def fit(self, x, y):
        self.record_ = {"train": x[:, 0].astype(int), "target": np.array(y)}
        type(self).records.append(self.record_)
        return self

    def predict(self, x):
        self.record_["test"] = x[:, 0].astype(int)
        return 10 + x[:, 0]


class ATTScoreTests(unittest.TestCase):
    def test_canonical_score_and_empirical_influence_match_hand_calculation(self):
        # Control odds are 1 and 3; their sum differs from treated count 2.
        # Thus this also distinguishes the canonical score from a Hajek score.
        result, influence = aipw_att(
            [8, 12, 4, 10], [1, 1, 0, 0], [.2, .8, .5, .75], [5, 7, 3, 8],
        )
        self.assertAlmostEqual(result["estimate"], .5)
        assert_allclose(influence, [5, 9, -2, -12])
        self.assertAlmostEqual(influence.mean(), 0)
        self.assertAlmostEqual(result["se"], np.sqrt(254 / 12))
        self.assertAlmostEqual(result["ci_low"], .5 - 1.96 * np.sqrt(254 / 12))
        self.assertAlmostEqual(result["ci_high"], .5 + 1.96 * np.sqrt(254 / 12))

    def test_correct_outcome_model_recovers_att_with_wrong_propensity(self):
        # Within each stratum the residual has zero mean in both groups.
        # True propensity is .5; the supplied model is deliberately very wrong.
        d = np.tile([0, 0, 1, 1], 2)
        m0 = np.repeat([10., 30.], 4)
        noise = np.tile([-2., 2., -3., 3.], 2)
        tau = np.repeat([2., 4.], 4)
        y = m0 + noise + d * tau
        result, _ = aipw_att(y, d, np.repeat([.85, .15], 4), m0)
        self.assertAlmostEqual(result["estimate"], tau[d == 1].mean(), places=12)

    def test_correct_propensity_recovers_att_with_wrong_outcome_model(self):
        # Empirical treatment fractions are exactly .25 and .75 by stratum.
        # Treated and weighted control residuals cancel even with wrong m0.
        d = np.array([0] * 6 + [1] * 2 + [0] * 2 + [1] * 6)
        y0 = np.repeat([10., 30.], 8)
        tau = np.repeat([2., 4.], 8)
        wrong_m0 = np.repeat([-5., 100.], 8)
        result, _ = aipw_att(y0 + d * tau, d, np.repeat([.25, .75], 8), wrong_m0)
        self.assertAlmostEqual(result["estimate"], 3.5, places=12)

    def test_treated_propensities_do_not_enter_att_control_residual_weights(self):
        y, d, m0 = [8, 12, 4, 10], [1, 1, 0, 0], [5, 7, 3, 8]
        a, influence_a = aipw_att(y, d, [0, 1, .5, .75], m0)
        b, influence_b = aipw_att(y, d, [.4, .6, .5, .75], m0)
        self.assertEqual(a, b)
        assert_array_equal(influence_a, influence_b)

    def test_endpoint_propensities_are_clipped_before_forming_odds(self):
        result, influence = aipw_att([3, 3, 1, 1], [1, 1, 0, 0], [0, 1, 0, 1], [0] * 4, clip=.1)
        self.assertAlmostEqual(result["estimate"], (6 - 1 / 9 - 9) / 2)
        self.assertTrue(np.isfinite(influence).all())

    def test_outcome_translation_preserves_estimate_and_influence(self):
        y, m0 = np.array([8, 12, 4, 10]), np.array([5, 7, 3, 8])
        a, ia = aipw_att(y, [1, 1, 0, 0], [.2, .8, .5, .75], m0)
        b, ib = aipw_att(y + 10000, [1, 1, 0, 0], [.2, .8, .5, .75], m0 + 10000)
        self.assertEqual(a, b)
        assert_array_equal(ia, ib)

    def test_mean_difference_uses_groupwise_sampling_variances(self):
        result = mean_difference([2, 6, 1, 3, 5], [1, 1, 0, 0, 0])
        self.assertAlmostEqual(result["estimate"], 1)
        self.assertAlmostEqual(result["se"], np.sqrt(8 / 2 + 4 / 3))

    def test_rejects_invalid_outcomes_treatment_and_features(self):
        cases = [
            ([1, 2, 3], [0, 1, 1], None),
            ([1, 2, 3, 4], [0, 0, 1], None),
            ([[1, 2], [3, 4]], [0, 0, 1, 1], None),
            ([1, np.nan, 3, 4], [0, 0, 1, 1], None),
            ([1, 2, 3, np.inf], [0, 0, 1, 1], None),
            ([1, 2, 3, 4], [0, 0, .5, 1], None),
            ([1, 2, 3, 4], [0, 0, 0, 1], None),
            ([1, 2, 3, 4], [1, 1, 1, 1], None),
            ([1, 2, 3, 4], [0, 0, 1, 1], np.ones(4)),
            ([1, 2, 3, 4], [0, 0, 1, 1], np.ones((3, 2))),
            ([1, 2, 3, 4], [0, 0, 1, 1], np.empty((4, 0))),
            ([1, 2, 3, 4], [0, 0, 1, 1], [[1], [2], [3], [np.inf]]),
        ]
        for y, d, x in cases:
            with self.subTest(y=y, d=d, x=x), self.assertRaises(ValueError):
                validate(y, d, x)

    def test_rejects_invalid_nuisance_arrays_and_clip(self):
        y, d = [1, 2, 3, 4], [0, 0, 1, 1]
        for e in ([.5] * 3, [[.5]] * 4, [np.nan] * 4, [-.01] * 4, [1.01] * 4):
            with self.subTest(e=e), self.assertRaises(ValueError):
                aipw_att(y, d, e, [0] * 4)
        for m0 in ([0] * 3, [np.inf] * 4):
            with self.subTest(m0=m0), self.assertRaises(ValueError):
                aipw_att(y, d, [.5] * 4, m0)
        for clip in (0, -.1, .5, 1, np.nan, np.inf):
            with self.subTest(clip=clip), self.assertRaises(ValueError):
                aipw_att(y, d, [.5] * 4, [0] * 4, clip)


class CrossFitTests(unittest.TestCase):
    def setUp(self):
        SpyClassifier.records.clear()
        SpyRegressor.records.clear()
        self.x = np.column_stack((np.arange(24), np.arange(24) ** 2))
        self.d = np.tile([0., 1.], 12)
        self.y = 100 + np.arange(24) * 3

    def test_each_prediction_is_out_of_fold_and_outcome_training_has_controls_only(self):
        classifier, regressor = SpyClassifier(), SpyRegressor()
        result = cross_fit(self.x, self.y, self.d, folds=4, seed=13,
                           custom_learners=(classifier, regressor))
        self.assertFalse(hasattr(classifier, "record_"))
        self.assertFalse(hasattr(regressor, "record_"))
        self.assertEqual(len(SpyClassifier.records), 4)
        self.assertEqual(len(SpyRegressor.records), 4)
        all_test_rows = []
        for fold, (propensity, outcome) in enumerate(zip(SpyClassifier.records, SpyRegressor.records)):
            train, test = propensity["train"], propensity["test"]
            self.assertEqual(set(train) & set(test), set())
            self.assertEqual(set(train) | set(test), set(range(24)))
            assert_array_equal(outcome["test"], test)
            assert_array_equal(outcome["train"], train[self.d[train] == 0])
            assert_array_equal(outcome["target"], self.y[outcome["train"]])
            assert_array_equal(propensity["target"], self.d[train])
            assert_array_equal(result.fold[test], np.full(len(test), fold))
            all_test_rows.extend(test)
        self.assertEqual(sorted(all_test_rows), list(range(24)))
        assert_allclose(result.propensity, .2 + self.x[:, 0] / 100)
        assert_array_equal(result.m0, 10 + self.x[:, 0])

    def test_folds_and_predictions_are_reproducible(self):
        a = cross_fit(self.x, self.y, self.d, folds=3, seed=9)
        b = cross_fit(self.x, self.y, self.d, folds=3, seed=9)
        assert_array_equal(a.fold, b.fold)
        assert_allclose(a.propensity, b.propensity)
        assert_allclose(a.m0, b.m0)
        self.assertEqual(set(a.fold), {0, 1, 2})
        self.assertTrue(np.isfinite(a.m0).all())
        self.assertTrue(((a.propensity > 0) & (a.propensity < 1)).all())

    def test_invalid_folds_and_unknown_method_are_rejected(self):
        for folds in (1, 0, 2.5, 13):
            with self.subTest(folds=folds), self.assertRaises(ValueError):
                cross_fit(self.x, self.y, self.d, folds=folds)
        with self.assertRaises(ValueError):
            cross_fit(self.x, self.y, self.d, method="unknown")

    def test_standardization_belongs_to_each_linear_model_pipeline(self):
        propensity, outcome = learners("linear", 42)
        self.assertEqual(list(propensity.named_steps), ["standardscaler", "logisticregression"])
        self.assertEqual(list(outcome.named_steps), ["standardscaler", "ridge"])
        self.assertIsNot(propensity.named_steps["standardscaler"], outcome.named_steps["standardscaler"])


class DiagnosticTests(unittest.TestCase):
    def test_ess_and_smd_share_hand_calculated_unweighted_denominator(self):
        # Treated mean=6, control mean=1; variances 8 and 2; pooled SD=sqrt(5).
        # Control odds [1,3] give weighted mean=1.5 and ESS=16/10.
        result = diagnostics([[4], [8], [0], [2]], [1, 1, 0, 0], [.2, .8, .5, .75], ["x"])
        self.assertAlmostEqual(result["ess_controls"], 1.6)
        self.assertAlmostEqual(result["control_weight_sum"], 4)
        self.assertAlmostEqual(result["max_control_weight"], 3)
        self.assertAlmostEqual(result["balance"][0]["before"], 5 / np.sqrt(5))
        self.assertAlmostEqual(result["balance"][0]["after"], 4.5 / np.sqrt(5))

    def test_clipping_fractions_and_histograms_keep_group_denominators(self):
        result = diagnostics([[0], [1], [2], [3]], [1, 1, 0, 0], [0, .5, 1, .8], ["x"], clip=.1)
        self.assertEqual(result["clipped_fraction"], .5)
        self.assertEqual(result["clipped_treated_fraction"], .5)
        self.assertEqual(result["clipped_control_fraction"], .5)
        self.assertEqual(sum(row["treated_count"] for row in result["propensity_bins"]), 2)
        self.assertEqual(sum(row["control_count"] for row in result["propensity_bins"]), 2)
        self.assertEqual(result["propensity_bins"][0]["treated_count"], 1)
        self.assertEqual(result["propensity_bins"][-1]["control_count"], 1)
        self.assertAlmostEqual(result["max_control_weight"], 9)

    def test_constant_feature_imbalance_is_explicitly_undefined(self):
        result = diagnostics([[1, 5], [1, 5], [0, 5], [0, 5]], [1, 1, 0, 0], [.5] * 4, ["separated", "constant"])
        self.assertIsNone(result["balance"][0]["before"])
        self.assertIsNone(result["balance"][0]["after"])
        self.assertEqual(result["balance"][1]["before"], 0)
        self.assertEqual(result["balance"][1]["after"], 0)

    def test_invalid_diagnostic_inputs_are_rejected(self):
        for propensity, names, clip in (([.5] * 3, ["x"], .01), ([np.nan] * 4, ["x"], .01),
                                        ([1.1] * 4, ["x"], .01), ([.5] * 4, [], .01),
                                        ([.5] * 4, ["x"], .5)):
            with self.subTest(propensity=propensity, names=names, clip=clip), self.assertRaises(ValueError):
                diagnostics([[0], [1], [2], [3]], [0, 0, 1, 1], propensity, names, clip)


if __name__ == "__main__":
    unittest.main()
