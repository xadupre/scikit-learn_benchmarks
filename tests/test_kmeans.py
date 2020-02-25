"""
Unit tests.
"""
import unittest
from benchmarks.cluster import KMeans_bench


class TestKMeans_bench(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.bench = KMeans_bench()
        cls.bench.setup_cache()
        cls.bench.setup('dense', 'full', 1)

    def test_setup(self):
        self.assertFalse(self.bench is None)
        bench = self.bench
        self.assertFalse(bench.estimator is None)
        self.assertFalse(bench.estimator_onnx is None)
        self.assertFalse(bench.estimator_onnx_ort is None)
        self.assertFalse(bench.estimator_onnx_pyrt is None)

    def test_track_test_score(self):
        bench = self.bench
        s1 = bench.track_test_score_skl()
        s2 = bench.track_test_score_ort()
        s3 = bench.track_test_score_pyrt()
        self.assertTrue(abs(s1 - s2) < 1e-7)
        self.assertTrue(abs(s1 - s3) < 1e-7)

    def test_astype(self):
        bench = self.bench
        bench.time_astype32()
        bench.peakmem_astype32()

    def test_time_predict(self):
        bench = self.bench
        bench.time_predict_skl()
        bench.time_predict_ort()
        bench.time_predict_pyrt()

    def test_peakmem_predict(self):
        bench = self.bench
        bench.peakmem_predict_skl()
        bench.peakmem_predict_ort()
        bench.peakmem_predict_pyrt()


if __name__ == "__main__":
    unittest.main()
