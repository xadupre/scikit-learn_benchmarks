"""
Unit tests.
"""
import os
import unittest
import inspect
from importlib import import_module
import benchmarks
from benchmarks.common import Estimator, Predictor, Classifier, Transformer
from benchmarks.linear_model import LinearRegression_bench


def try_import(name):
    try:
        mod = import_module(name)
    except ImportError:
        return None
    return dict(name=name, mod=mod)


class TestABC(unittest.TestCase):

    def test_abc(self):
        self.assertTrue(inspect.isabstract(Estimator))
        self.assertTrue(inspect.isabstract(Predictor))
        self.assertTrue(inspect.isabstract(Classifier))
        self.assertTrue(inspect.isabstract(Transformer))
        self.assertFalse(inspect.isabstract(LinearRegression_bench))
        
    def test_import_all(self):
        fold = os.path.dirname(benchmarks.__file__)
        subs = os.listdir(fold)
        ok = 0
        for sub in subs:
            name, ext = os.path.splitext(sub)            
            if ext != '.py':
                continue
            name = "benchmarks.{}".format(name)
            mod = try_import(name)
            if mod is None:
                continue
            fcts = mod['mod'].__dict__
            for k, v in fcts.items():
                if k.endswith('_bench'):
                    if inspect.isabstract(v):
                        raise RuntimeError("Benchmark '{}' is considered as abstract.".format(k))
                    ok += 1
        self.assertGreater(ok, 18)
            


if __name__ == "__main__":
    unittest.main()
