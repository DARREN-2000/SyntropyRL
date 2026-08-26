#!/usr/bin/env python3
"""Run the test suite with the standard library only.

CI installs pytest and runs it for real. This script exists so the suite also
runs on a bare interpreter with no network access, which is exactly the
situation you are in when a training box has no egress. It implements the small
slice of the pytest API the tests actually use: `mark.parametrize`, `raises`,
`approx`, `fail`, and `skip`.

    python3 scripts/run_tests.py
    python3 scripts/run_tests.py -k healthy -v
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import os
import sys
import time
import traceback
import types
from typing import Any, Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
TESTS = os.path.join(ROOT, "tests")


class _Skipped(Exception):
    pass


class _Approx:
    def __init__(self, expected: Any, rel: float = 1e-6, abs: float = 1e-12) -> None:
        self.expected = expected
        self.rel = rel
        self.abs = abs

    def __eq__(self, other: Any) -> bool:
        try:
            diff = abs(float(other) - float(self.expected))
        except (TypeError, ValueError):
            return NotImplemented
        return diff <= max(self.abs, self.rel * abs(float(self.expected)))

    def __repr__(self) -> str:
        return "approx({!r})".format(self.expected)


class _Raises:
    def __init__(self, expected: Any) -> None:
        self.expected = expected
        self.value: Any = None

    def __enter__(self) -> "_Raises":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if exc_type is None:
            raise AssertionError("DID NOT RAISE {}".format(self.expected))
        if issubclass(exc_type, self.expected):
            self.value = exc
            return True
        return False


class _Mark:
    """Just enough of pytest.mark to expand parametrized cases."""

    def parametrize(self, argnames: Any, argvalues: Any, **_kwargs: Any):
        if isinstance(argnames, str):
            names = [a.strip() for a in argnames.split(",") if a.strip()]
        else:
            names = [str(a) for a in argnames]

        def decorator(fn):
            cases: List[Dict[str, Any]] = []
            for values in argvalues:
                if len(names) == 1:
                    values = (values,)
                cases.append(dict(zip(names, tuple(values))))
            existing = getattr(fn, "_rld_cases", None)
            if existing:
                merged = []
                for outer in cases:
                    for inner in existing:
                        combined = dict(inner)
                        combined.update(outer)
                        merged.append(combined)
                fn._rld_cases = merged
            else:
                fn._rld_cases = cases
            return fn

        return decorator

    def __getattr__(self, _name: str):
        # skipif, slow, xfail, ... treated as no-ops.
        def marker(*args: Any, **_kwargs: Any):
            if len(args) == 1 and callable(args[0]):
                return args[0]
            return lambda fn: fn

        return marker


def _make_pytest_shim() -> types.ModuleType:
    mod = types.ModuleType("pytest")
    mod.mark = _Mark()
    mod.raises = lambda expected, **_kw: _Raises(expected)
    mod.approx = lambda expected, rel=1e-6, abs=1e-12: _Approx(expected, rel, abs)

    def _fail(msg: str = "", **_kw: Any):
        raise AssertionError(msg or "pytest.fail()")

    def _skip(msg: str = "", **_kw: Any):
        raise _Skipped(msg)

    mod.fail = _fail
    mod.skip = _skip
    mod.xfail = _skip
    mod.fixture = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda fn: fn))
    mod.importorskip = lambda name, **k: __import__(name)
    return mod


def _load_module(path: str) -> types.ModuleType:
    name = "rld_test_" + os.path.basename(path)[:-3]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load {}".format(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _collect(module: types.ModuleType) -> List[Tuple[str, Any, Dict[str, Any]]]:
    """Return (test_id, callable, kwargs) triples, expanding parametrize."""
    found: List[Tuple[str, Any, Dict[str, Any]]] = []
    for name, obj in vars(module).items():
        if name.startswith("test_") and inspect.isfunction(obj):
            for case in getattr(obj, "_rld_cases", None) or [{}]:
                label = name
                if case:
                    label += "[{}]".format(
                        "-".join(str(v) for v in case.values()))
                found.append((label, obj, case))
        elif name.startswith("Test") and inspect.isclass(obj):
            for attr, member in vars(obj).items():
                if attr.startswith("test_") and inspect.isfunction(member):
                    for case in getattr(member, "_rld_cases", None) or [{}]:
                        label = "{}::{}".format(name, attr)
                        if case:
                            label += "[{}]".format(
                                "-".join(str(v) for v in case.values()))
                        found.append((label, (obj, member), case))
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description="run the syntropyrl test suite")
    ap.add_argument("-k", dest="pattern", default=None,
                    help="only run tests whose id contains this substring")
    ap.add_argument("-v", dest="verbose", action="store_true")
    ap.add_argument("-f", dest="failfast", action="store_true")
    args = ap.parse_args()

    for path in (SRC, TESTS, ROOT):
        if path not in sys.path:
            sys.path.insert(0, path)
    sys.modules.setdefault("pytest", _make_pytest_shim())

    files = sorted(
        os.path.join(TESTS, f) for f in os.listdir(TESTS)
        if f.startswith("test_") and f.endswith(".py")
    )
    if not files:
        print("no test files found in {}".format(TESTS))
        return 1

    passed = failed = skipped = 0
    failures: List[Tuple[str, str]] = []
    started = time.time()

    for path in files:
        try:
            module = _load_module(path)
        except Exception:
            failed += 1
            failures.append((os.path.basename(path), traceback.format_exc()))
            print("E", end="", flush=True)
            continue

        for label, target, kwargs in _collect(module):
            test_id = "{}::{}".format(os.path.basename(path), label)
            if args.pattern and args.pattern.lower() not in test_id.lower():
                continue
            try:
                if isinstance(target, tuple):
                    cls, func = target
                    func(cls(), **kwargs)
                else:
                    target(**kwargs)
            except _Skipped as exc:
                skipped += 1
                print("s" if not args.verbose else "SKIP {} {}\n".format(test_id, exc),
                      end="", flush=True)
            except Exception:
                failed += 1
                failures.append((test_id, traceback.format_exc()))
                print("F" if not args.verbose else "FAIL {}\n".format(test_id),
                      end="", flush=True)
                if args.failfast:
                    break
            else:
                passed += 1
                print("." if not args.verbose else "pass {}\n".format(test_id),
                      end="", flush=True)
        if args.failfast and failed:
            break

    elapsed = time.time() - started
    print("")
    for test_id, tb in failures:
        print("")
        print("=" * 70)
        print("FAILED {}".format(test_id))
        print("-" * 70)
        print(tb.rstrip())
    print("")
    print("{} passed, {} failed, {} skipped in {:.1f}s".format(
        passed, failed, skipped, elapsed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
