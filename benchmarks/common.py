import os
from multiprocessing import cpu_count
import json
import timeit
import pickle
import itertools
import numpy as np


def get_from_config():
    current_path = os.path.dirname(os.path.realpath(__file__))

    config_path = os.path.join(current_path, 'config.json')
    with open(config_path, 'r') as config_file:
        config_file = ''.join(line for line in config_file
                              if line and '//' not in line)
        config = json.loads(config_file)

    profile = config['profile']

    n_jobs_vals = config['n_jobs_vals']
    if not n_jobs_vals:
        n_jobs_vals = list(range(1, 1 + cpu_count()))

    save_estimators = config['save_estimators']
    save_folder = os.getenv('ASV_COMMIT', 'new')[:8]

    if save_estimators:
        save_path = os.path.join(current_path, 'cache',
                                 'estimators', save_folder)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

    base_folder = config['base_folder']

    bench_predict = config['bench_predict']
    bench_predictproba = config['bench_predictproba']
    bench_transform = config['bench_transform']
    bench_onnx = config['bench_onnx']

    return (profile, n_jobs_vals, save_estimators, save_folder, base_folder,
            bench_predict, bench_predictproba, bench_transform, bench_onnx)


def get_estimator_path(benchmark, folder, params, save=False):
    folder = os.path.join('estimators', folder) if save else 'tmp'
    f_name = (benchmark.__class__.__name__[:-6]
              + '_estimator_' + '_'.join(list(map(str, params))) + '.pkl')
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'cache', folder, f_name)
    return path


def get_data_path(benchmark, params):
    f_name = (benchmark.__class__.__name__[:-6]
              + '_data_' + '_'.join(list(map(str, params))) + '.pkl')
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'cache', 'tmp', f_name)
    return path


def clear_tmp():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'cache', 'tmp')
    list(map(os.remove, (os.path.join(path, f)
                         for f in os.listdir(path) if f != '.gitignore')))


class Benchmark:
    timer = timeit.default_timer  # wall time
    processes = 1
    timeout = 500

    (profile, n_jobs_vals, save_estimators, save_folder, base_folder,
     bench_predict, bench_predictproba, bench_transform,
     bench_onnx) = get_from_config()

    if profile == 'fast':
        warmup_time = 0
        repeat = 1
        number = 1
        min_run_count = 1
        data_size = 'small'
    elif profile == 'regular':
        warmup_time = 1
        repeat = (3, 100, 30)
        data_size = 'small'
    elif profile == 'large_scale':
        warmup_time = 1
        repeat = 3
        number = 1
        data_size = 'large'


class Estimator:
    def setup_cache(self):
        clear_tmp()

        param_grid = list(itertools.product(*self.params))

        for params in param_grid:
            data, estimator = self.setup_cache_(params) or (None, None)
            if data is None:
                continue

            data_path = get_data_path(self, params)
            with open(data_path, 'wb') as f:
                pickle.dump(data, f)

            X, _, y, _ = data
            estimator.fit(X, y)

            est_path = get_estimator_path(self, Benchmark.save_folder,
                                          params, Benchmark.save_estimators)
            with open(est_path, 'wb') as f:
                pickle.dump(estimator, f)

    def setup(self, *params):
        if hasattr(self, 'setup_'):
            self.setup_(params)

        data_path = get_data_path(self, params)
        with open(data_path, 'rb') as f:
            self.X, self.X_val, self.y, self.y_val = pickle.load(f)

        est_path = get_estimator_path(self, Benchmark.save_folder,
                                      params, Benchmark.save_estimators)
        with open(est_path, 'rb') as f:
            self.estimator = pickle.load(f)

        if Benchmark.bench_onnx:
            self._setup_onnx()

        self.make_scorers()

    def _setup_onnx(self):
        from skl2onnx import to_onnx
        try:
            self.estimator_onnx = to_onnx(self.estimator, self.X[:1])
        except RuntimeError as e:
            self.estimator_onnx = None
        if self.estimator_onnx is not None:
            from onnxruntime import InferenceSession
            try:
                self.estimator_onnx_ort = InferenceSession(
                    self.estimator_onnx.SerializeToString())
            except RuntimeError as e:
                self.estimator_onnx_ort = None

            from mlprodict.onnxrt import OnnxInference
            try:
                self.estimator_onnx_pyrt = OnnxInference(
                    self.estimator_onnx)
            except RuntimeError as e:
                self.estimator_onnx_pyrt = None

    def time_fit(self, *args):
        self.estimator.fit(self.X, self.y)

    def peakmem_fit(self, *args):
        self.estimator.fit(self.X, self.y)

    def track_train_score_skl(self, *args):
        if isinstance(self, Predictor):
            y_pred = self.estimator.predict(self.X)
        else:
            y_pred = None
        return float(self.train_scorer(self.y, y_pred))

    def track_test_score_skl(self, *args):
        if isinstance(self, Predictor):
            y_val_pred = self.estimator.predict(self.X_val)
        else:
            y_val_pred = None
        return float(self.test_scorer(self.y_val, y_val_pred))

    if Benchmark.bench_onnx:

        def track_test_score_ort(self, *args):
            if (isinstance(self, Predictor) and
                    self.estimator_onnx_ort is not None):
                y_val_pred = self.estimator_onnx_ort.run(
                    None, {'X': self.X_val.astype(np.float32)})[0]
            else:
                y_val_pred = None
            return float(self.test_scorer(self.y_val, y_val_pred))

        def track_test_score_pyrt(self, *args):
            if (isinstance(self, Predictor) and
                    self.estimator_onnx_pyrt is not None):
                res = self.estimator_onnx_pyrt.run(
                    {'X': self.X_val.astype(np.float32)})
                y_val_pred = (res['variable']
                              if 'variable' in res else res['output_label'])
            else:
                y_val_pred = None
            return float(self.test_scorer(self.y_val, y_val_pred))


class Predictor:
    if Benchmark.bench_predict:
        def time_predict_skl(self, *args):
            self.estimator.predict(self.X)

        def peakmem_predict_skl(self, *args):
            self.estimator.predict(self.X)

        if Benchmark.base_folder is not None:
            def track_same_prediction_skl(self, *args):
                file_path = get_estimator_path(self, Benchmark.base_folder,
                                               args, True)
                with open(file_path, 'rb') as f:
                    estimator_base = pickle.load(f)

                y_val_pred_base = estimator_base.predict(self.X_val)
                y_val_pred = self.estimator.predict(self.X_val)

                return np.allclose(y_val_pred_base, y_val_pred)

    if Benchmark.bench_onnx:
        def time_predict_ort(self, *args):
            if self.estimator_onnx_ort is not None:
                self.estimator_onnx_ort.run(
                    None, {'X': self.X.astype(np.float32)})[0]
            else:
                raise RuntimeError("estimator_onnx_ort could not be created.")

        def peakmem_predict_ort(self, *args):
            if self.estimator_onnx_ort is not None:
                self.estimator_onnx_ort.run(
                    None, {'X': self.X.astype(np.float32)})[0]
            else:
                raise RuntimeError("estimator_onnx_ort could not be created.")

        def time_predict_pyrt(self, *args):
            if self.estimator_onnx_pyrt is not None:
                self.estimator_onnx_pyrt.run(
                    {'X': self.X.astype(np.float32)})
            else:
                raise RuntimeError("estimator_onnx_pyrt could not be created.")

        def peakmem_predict_pyrt(self, *args):
            if self.estimator_onnx_pyrt is not None:
                self.estimator_onnx_pyrt.run(
                    {'X': self.X.astype(np.float32)})
            else:
                raise RuntimeError("estimator_onnx_pyrt could not be created.")


class Classifier(Predictor):

    if Benchmark.bench_predictproba:
        def time_predictproba_skl(self, *args):
            self.estimator.predict_proba(self.X)

        def peakmem_predictproba_skl(self, *args):
            self.estimator.predict_proba(self.X)

        if Benchmark.base_folder is not None:
            def track_same_predictionproba_skl(self, *args):
                file_path = get_estimator_path(self, Benchmark.base_folder,
                                               args, True)
                with open(file_path, 'rb') as f:
                    estimator_base = pickle.load(f)

                y_val_pred_base = estimator_base.predict_proba(self.X_val)
                y_val_pred = self.estimator.predict_proba(self.X_val)

                return np.allclose(y_val_pred_base, y_val_pred)

        if Benchmark.bench_onnx:
            def time_predictproba_ort(self, *args):
                if self.estimator_onnx_ort is not None:
                    self.estimator_onnx_ort.run(
                        None, {'X': self.X.astype(np.float32)})[1]
                else:
                    raise RuntimeError(
                        "estimator_onnx_ort could not be created.")

            def peakmem_predictproba_ort(self, *args):
                if self.estimator_onnx_ort is not None:
                    self.estimator_onnx_ort.run(
                        None, {'X': self.X.astype(np.float32)})[1]
                else:
                    raise RuntimeError(
                        "estimator_onnx_ort could not be created.")

            def time_predictproba_pyrt(self, *args):
                if self.estimator_onnx_pyrt is not None:
                    self.estimator_onnx_pyrt.run(
                        {'X': self.X.astype(np.float32)})
                else:
                    raise RuntimeError(
                        "estimator_onnx_pyrt could not be created.")

            def peakmem_predictproba_pyrt(self, *args):
                if self.estimator_onnx_pyrt is not None:
                    self.estimator_onnx_pyrt.run(
                        {'X': self.X.astype(np.float32)})
                else:
                    raise RuntimeError(
                        "estimator_onnx_pyrt could not be created.")


class Transformer:
    if Benchmark.bench_transform:
        def time_transform_skl(self, *args):
            self.estimator.transform(self.X)

        def peakmem_transform_skl(self, *args):
            self.estimator.transform(self.X)

        if Benchmark.base_folder is not None:
            def track_same_transform_skl(self, *args):
                file_path = get_estimator_path(self, Benchmark.base_folder,
                                               args, True)
                with open(file_path, 'rb') as f:
                    estimator_base = pickle.load(f)

                X_val_t_base = estimator_base.transform(self.X_val)
                X_val_t = self.estimator.transform(self.X_val)

                return np.allclose(X_val_t_base, X_val_t)

    if Benchmark.bench_onnx:
        def time_transform_ort(self, *args):
            if self.estimator_onnx_ort is not None:
                self.estimator_onnx_ort.run(
                    None, {'X': self.X.astype(np.float32)})
            else:
                raise RuntimeError("estimator_onnx_ort could not be created.")

        def peakmem_transform_ort(self, *args):
            if self.estimator_onnx_ort is not None:
                self.estimator_onnx_ort.run(
                    None, {'X': self.X.astype(np.float32)})
            else:
                raise RuntimeError("estimator_onnx_ort could not be created.")

        def time_transform_pyrt(self, *args):
            if self.estimator_onnx_pyrt is not None:
                self.estimator_onnx_pyrt.run(
                    {'X': self.X.astype(np.float32)})
            else:
                raise RuntimeError("estimator_onnx_pyrt could not be created.")

        def peakmem_transform_pyrt(self, *args):
            if self.estimator_onnx_pyrt is not None:
                self.estimator_onnx_pyrt.run(
                    {'X': self.X.astype(np.float32)})
            else:
                raise RuntimeError("estimator_onnx_pyrt could not be created.")
