import math

import importlib.util
from pathlib import Path


def _load_module_from_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Ensure the module is present in sys.modules before executing code that
    # may reference its __name__ for dataclass construction
    import sys

    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore
    return module


ROOT = Path(__file__).resolve().parents[2]
split_path = str(ROOT / "examples" / "connection_layer" / "split_reversal_demo.py")
bench_path = str(
    ROOT / "examples" / "connection_layer" / "benchmarks" / "macro_coupling_benchmark.py"
)

split_module = _load_module_from_path("split_reversal_demo", split_path)
bench_module = _load_module_from_path("macro_coupling_benchmark", bench_path)


def test_split_reversal_demo_run_demo_structure():
    results = split_module.run_demo()
    assert isinstance(results, list)
    assert len(results) >= 1
    for item in results:
        assert isinstance(item, tuple)
        assert len(item) == 2
        name, value = item
        assert isinstance(name, str)
        assert isinstance(value, float)
        # Deterministic numeric values should be finite
        assert math.isfinite(value)


def test_macro_coupling_benchmark_run_benchmark_shape():
    result = bench_module.run_benchmark()
    assert isinstance(result, dict)
    # Required keys
    for key in ("elapsed_seconds", "macro_steps", "iterations"):
        assert key in result
        assert isinstance(result[key], float)


def test_smoke_integration_imports():
    # Ensure both modules load and expose expected callables without error
    demo = split_module.run_demo()
    bench = bench_module.run_benchmark()
    assert isinstance(demo, list)
    assert isinstance(bench, dict)
