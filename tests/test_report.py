"""Meaningful safety and portability checks for exported reports."""

import json
from pathlib import Path
import re
import tempfile
import unittest

from econ_causal_lab.report import render_report


class ReportTests(unittest.TestCase):
    def render(self, data):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "nested" / "report.html"
            self.assertEqual(render_report(data, path), path)
            return path.read_text(encoding="utf-8")

    def extract_data(self, html):
        return json.loads(re.search(
            r'<script id="report-data" type="application/json">(.*?)</script>',
            html, re.DOTALL,
        ).group(1))

    def test_untrusted_text_cannot_close_json_script(self):
        malicious = '</script><script>alert("xss")</script><img src=x onerror=alert(1)>&\u2028'
        data = {"notes": [malicious], "sources": [{"name": malicious, "url": "javascript:alert(1)"}]}
        rendered = self.render(data)
        self.assertNotIn(malicious, rendered)
        self.assertEqual(self.extract_data(rendered), data)
        self.assertNotIn("innerHTML", rendered)
        self.assertNotIn("document.write", rendered)
        self.assertIn('["https:","http:"].includes(url.protocol)', rendered)

    def test_nonfinite_results_become_missing_not_invalid_json(self):
        rendered = self.render({"results": [{"estimate": float("nan"), "se": float("inf")}], "notes": []})
        self.assertEqual(self.extract_data(rendered)["results"], [{"estimate": None, "se": None}])

    def test_empty_and_custom_reports_remain_self_contained(self):
        for data in ({}, {"benchmark": None, "datasets": [{"id": "custom", "label": "自有数据", "n": 3}], "simulation": []}):
            rendered = self.render(data)
            self.assertEqual(self.extract_data(rendered), data)
            self.assertNotRegex(rendered, r'<script[^>]+src=')
            self.assertNotRegex(rendered, r'<link[^>]+href=')
            self.assertNotIn("__ECON_REPORT_DATA__", rendered)


if __name__ == "__main__":
    unittest.main()
