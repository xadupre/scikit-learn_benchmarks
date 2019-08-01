echo --CLONE--
if [ ! -d scikit-learn_benchmarks ]
then
    git clone -b onnx https://github.com/xadupre/scikit-learn_benchmarks --recursive
else
    cd scikit-learn_benchmarks
    git pull
    git submodule update --init --recursive
    cd ..
fi

cd scikit-learn_benchmarks

echo --MACHINE--
asv machine --yes || exit 1

echo --RUN-BENCHMARK--
asv run -b ElasticNet --append-samples || exit 1
asv run -b GradientBoosting --append-samples --no-pull || exit 1
asv run -b KMeans --append-samples --no-pull || exit 1
asv run -b KNeighbors --append-samples --no-pull || exit 1
asv run -b Lasso --append-samples --no-pull || exit 1
asv run -b LinearRegression --append-samples --no-pull || exit 1
asv run -b LogisticRegression --append-samples --no-pull || exit 1
asv run -b PCA --append-samples --no-pull || exit 1
asv run -b RandomForest --append-samples --no-pull || exit 1
asv run -b Ridge --append-samples --no-pull || exit 1
asv run -b SGDRegressor --append-samples --no-pull || exit 1
asv run -b SVC --append-samples --no-pull || exit 1

echo --PUBLISH-BENCHMARK--
if [ -d scikit-learn_benchmarks ]
then
    rm ./../dist/html/sklbench_results -r -f
fi
asv publish -o ./../dist/html/sklbench_results || exit 1

echo --END--
cd ..

