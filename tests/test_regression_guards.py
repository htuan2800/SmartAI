import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app.py"
ENGINE_PATH = ROOT / "rag_engine.py"


class RegressionGuardsTest(unittest.TestCase):
    def test_upload_uses_sanitized_name(self):
        app_text = APP_PATH.read_text(encoding="utf-8")
        self.assertIn('file_path = config.UPLOAD_DIR / f"{raw_hash}_{clean_name}"', app_text)

    def test_benchmark_identifier_no_longer_uses_none_bytes(self):
        app_text = APP_PATH.read_text(encoding="utf-8")
        self.assertIn('f"benchmark::{st.session_state.current_file_path}"', app_text)
        self.assertNotIn('get_vector_store(\n                        chunks,\n                        st.session_state.current_file_bytes,', app_text)

    def test_multihop_supports_vietnamese_connector(self):
        engine_text = ENGINE_PATH.read_text(encoding="utf-8")
        self.assertIn('re.search(r"\\b(and|và)\\b"', engine_text)
        self.assertIn('re.split(r"\\b(?:and|và)\\b"', engine_text)

    def test_rerank_metrics_are_returned_for_ui_comparison(self):
        engine_text = ENGINE_PATH.read_text(encoding="utf-8")
        self.assertIn('"retrieval_time_ms"', engine_text)
        self.assertIn('"rerank_time_ms"', engine_text)
        self.assertIn('"pages_before_rerank"', engine_text)
        self.assertIn('"pages_after_rerank"', engine_text)


if __name__ == "__main__":
    unittest.main()
