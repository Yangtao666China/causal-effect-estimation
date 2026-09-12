"""Data integrity, offline operation, and explicit custom-data cleaning."""

import hashlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal

from econ_causal_lab.data import COLUMNS, load_benchmark, load_csv, verified_array


def fixture_source(name="fixture", treatment=0, rows=2):
    array = np.arange(rows * 10, dtype=float).reshape(rows, 10)
    array[:, 0] = treatment
    output = io.StringIO()
    np.savetxt(output, array)
    content = output.getvalue().encode("ascii")
    source = dict(name=name, treatment=treatment, rows=rows,
                  sha256=hashlib.sha256(content).hexdigest(), url=f"https://example.invalid/{name}.txt")
    return source, content, array


class VerifiedDataTests(unittest.TestCase):
    def test_valid_checksum_and_shape_return_source_values(self):
        source, content, expected = fixture_source()
        assert_array_equal(verified_array(content, source), expected)

    def test_tampered_bytes_fail_even_when_still_parseable(self):
        source, content, _ = fixture_source()
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            verified_array(content + b"\n", source)

    def test_checked_hash_does_not_replace_schema_validation(self):
        source, content, _ = fixture_source()
        for changed in ({**source, "rows": 3}, {**source, "treatment": 1}):
            with self.subTest(source=changed), self.assertRaises(ValueError):
                verified_array(content, changed)

    def test_nonfinite_data_are_rejected_even_with_valid_hash(self):
        source, content, _ = fixture_source()
        damaged = content.replace(b"0.000000000000000000e+00", b"nan", 1)
        source["sha256"] = hashlib.sha256(damaged).hexdigest()
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            verified_array(damaged, source)

    def test_offline_cache_is_verified_and_never_contacts_network(self):
        fixtures = [fixture_source("treated", 1), fixture_source("random_controls", 0), fixture_source("cps", 0, 3)]
        with tempfile.TemporaryDirectory() as folder:
            for source, content, _ in fixtures:
                (Path(folder) / (source["name"] + ".txt")).write_bytes(content)
            with patch("econ_causal_lab.data.SOURCES", [f[0] for f in fixtures]), \
                 patch("econ_causal_lab.data.urllib.request.urlopen") as network:
                frames = load_benchmark(folder, offline=True)
            network.assert_not_called()
            self.assertEqual(list(frames["experimental"].columns), COLUMNS)
            self.assertEqual(frames["experimental"].shape, (4, 10))
            self.assertEqual(frames["observational"].shape, (5, 10))
            assert_array_equal(frames["experimental"].iloc[:2], frames["observational"].iloc[:2])

    def test_missing_offline_cache_fails_without_network_or_writes(self):
        source, _, _ = fixture_source()
        with tempfile.TemporaryDirectory() as folder, \
             patch("econ_causal_lab.data.SOURCES", [source]), \
             patch("econ_causal_lab.data.urllib.request.urlopen") as network:
            with self.assertRaisesRegex(FileNotFoundError, "Missing cache"):
                load_benchmark(folder, offline=True)
            network.assert_not_called()
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_damaged_cache_fails_closed_without_silent_redownload(self):
        source, content, _ = fixture_source()
        with tempfile.TemporaryDirectory() as folder, \
             patch("econ_causal_lab.data.SOURCES", [source]), \
             patch("econ_causal_lab.data.urllib.request.urlopen") as network:
            path = Path(folder) / (source["name"] + ".txt")
            path.write_bytes(content + b"\n")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_benchmark(folder, offline=False)
            network.assert_not_called()
            self.assertEqual(path.read_bytes(), content + b"\n")

    def test_download_is_verified_before_it_enters_cache(self):
        source, content, _ = fixture_source()
        with tempfile.TemporaryDirectory() as folder, \
             patch("econ_causal_lab.data.SOURCES", [source]), \
             patch("econ_causal_lab.data.urllib.request.urlopen", return_value=io.BytesIO(content + b"\n")):
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_benchmark(folder)
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_verified_downloads_can_be_reused_offline(self):
        fixtures = [fixture_source("treated", 1), fixture_source("random_controls", 0), fixture_source("cps", 0)]
        responses = [io.BytesIO(item[1]) for item in fixtures]
        with tempfile.TemporaryDirectory() as folder, \
             patch("econ_causal_lab.data.SOURCES", [f[0] for f in fixtures]), \
             patch("econ_causal_lab.data.urllib.request.urlopen", side_effect=responses) as network:
            online = load_benchmark(folder)
            self.assertEqual(network.call_count, 3)
            offline = load_benchmark(folder, offline=True)
            self.assertEqual(network.call_count, 3)
            for name in online:
                assert_array_equal(online[name], offline[name])

    def test_cli_offline_missing_cache_returns_actionable_error(self):
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run(
                [sys.executable, "-m", "econ_causal_lab", "benchmark", "--offline", "--cache", folder, "--repetitions", "0"],
                cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, timeout=30,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("Missing cache", result.stderr)
            self.assertNotIn("Traceback", result.stderr)


class CustomCSVTests(unittest.TestCase):
    def load_text(self, content, **kwargs):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "custom.csv"
            path.write_text(content, encoding="utf-8-sig")
            return load_csv(path, kwargs.get("outcome", "y"), kwargs.get("treatment", "d"), kwargs.get("features", ["x"]))

    def test_bom_numeric_data_keep_requested_order_and_ignore_unselected_text(self):
        frame = self.load_text("d,x,y,note\n0,2,10,alpha\n1,3,20,beta\n0,4,30,\n1,5,40,delta\n")
        self.assertEqual(list(frame.columns), ["y", "d", "x"])
        assert_array_equal(frame.to_numpy(), [[10, 0, 2], [20, 1, 3], [30, 0, 4], [40, 1, 5]])

    def test_outcome_and_treatment_are_forbidden_as_features(self):
        for features in (["x", "y"], ["d", "x"]):
            with self.subTest(features=features), self.assertRaisesRegex(ValueError, "cannot be included"):
                self.load_text("y,d,x\n1,0,2\n", features=features)

    def test_duplicate_empty_features_and_same_target_columns_are_rejected(self):
        for kwargs in ({"features": []}, {"features": ["x", "x"]}, {"outcome": "d"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.load_text("y,d,x\n1,0,2\n", **kwargs)

    def test_missing_columns_are_named_in_error(self):
        with self.assertRaisesRegex(ValueError, "missing columns: x"):
            self.load_text("y,d\n1,0\n")

    def test_selected_categories_require_explicit_encoding(self):
        with self.assertRaisesRegex(ValueError, "must be numeric"):
            self.load_text("y,d,x\n1,0,urban\n2,1,rural\n")

    def test_missing_or_nonfinite_values_require_explicit_cleaning(self):
        for bad in ("", "NaN", "inf", "-inf"):
            with self.subTest(value=bad), self.assertRaisesRegex(ValueError, "cleaning strategy"):
                self.load_text(f"y,d,x\n1,0,2\n2,1,{bad}\n")


if __name__ == "__main__":
    unittest.main()
