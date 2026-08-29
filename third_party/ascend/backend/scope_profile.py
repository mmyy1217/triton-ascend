import contextlib
import json
import math
import os
import statistics
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


_MODE_ENV = "TRITON_ASCEND_SCOPE_PROFILE"
_DUMP_ENV = "TRITON_ASCEND_SCOPE_PROFILE_DUMP"
_PLAN_ENV = "TRITON_ASCEND_SCOPE_PROFILE_PLAN_ID"


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def _error_payload(error):
    return {
        "type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
    }


@contextlib.contextmanager
def _profile_environment(mode, dump, plan_id=None):
    names = (_MODE_ENV, _DUMP_ENV, _PLAN_ENV, "TRITON_ASCEND_COMPILE_MODE")
    previous = {name: os.environ.get(name) for name in names}
    os.environ[_MODE_ENV] = mode
    os.environ[_DUMP_ENV] = str(Path(dump).expanduser().resolve())
    os.environ["TRITON_ASCEND_COMPILE_MODE"] = "simd_simt"
    if plan_id is None:
        os.environ.pop(_PLAN_ENV, None)
    else:
        os.environ[_PLAN_ENV] = str(plan_id)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _compile(kernel, args, grid, kwargs, mode, dump, plan_id=None):
    started = time.perf_counter()
    with _profile_environment(mode, dump, plan_id):
        compiled = kernel.warmup(*args, grid=grid, **kwargs)
    elapsed = (time.perf_counter() - started) * 1000.0
    return compiled, elapsed


def _report_dir(compiled):
    metadata = getattr(compiled, "metadata", None)
    directory = getattr(metadata, "scope_profile_report_dir", None)
    if not directory:
        raise RuntimeError("ScopeProfile compilation did not expose scope_profile_report_dir metadata.")
    return Path(directory)


def _sync():
    from triton.runtime import driver
    driver.active.get_device_interface().synchronize()


def _percentile(values, quantile):
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _benchmark(launch, prepare, warmup_ms, rep_ms):
    from triton.runtime import driver

    device_interface = driver.active.get_device_interface()
    cache = driver.active.get_empty_cache_for_benchmark()
    if prepare:
        prepare()
    launch()
    device_interface.synchronize()

    if prepare:
        prepare()
    estimate_start = device_interface.Event(enable_timing=True)
    estimate_end = device_interface.Event(enable_timing=True)
    estimate_start.record()
    launch()
    estimate_end.record()
    device_interface.synchronize()
    estimate_ms = max(float(estimate_start.elapsed_time(estimate_end)), 1e-6)
    warmup_count = max(1, int(warmup_ms / estimate_ms))
    repeat_count = max(1, int(rep_ms / estimate_ms))

    for _ in range(warmup_count):
        if prepare:
            prepare()
        launch()
    device_interface.synchronize()

    starts = [device_interface.Event(enable_timing=True) for _ in range(repeat_count)]
    ends = [device_interface.Event(enable_timing=True) for _ in range(repeat_count)]
    for index in range(repeat_count):
        driver.active.clear_cache(cache)
        if prepare:
            prepare()
        starts[index].record()
        launch()
        ends[index].record()
    device_interface.synchronize()
    samples = [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]
    return {
        "estimate_ms": estimate_ms,
        "warmup_count": warmup_count,
        "repeat_count": repeat_count,
        "median_ms": statistics.median(samples),
        "p20_ms": _percentile(samples, 0.2),
        "p80_ms": _percentile(samples, 0.8),
        "samples_ms": samples,
    }


def _run_correctness(launch, reset, check):
    if reset:
        reset()
    launch()
    _sync()
    outcome = check()
    if outcome is False:
        raise AssertionError("ScopeProfile correctness callback returned False.")
    return outcome


def _selected_plan_id(raw):
    if raw is None:
        raw = os.environ.get(_PLAN_ENV)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError("ScopeProfile plan_id must be a non-negative integer.") from error
    if value < 0:
        raise ValueError("ScopeProfile plan_id must be a non-negative integer.")
    return value


def _discover(kernel, args, grid, kwargs, dump):
    compiled, compile_ms = _compile(kernel, args, grid, kwargs, "plan", dump)
    report_dir = _report_dir(compiled)
    manifest = json.loads((report_dir / "manifest.json").read_text())
    _write_json(report_dir / "plan-compile.json", {
        "status": "valid",
        "compile_ms": compile_ms,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    return report_dir, manifest


def _apply_one(kernel, args, grid, kwargs, dump, report_dir, plan_id, reset, check, warmup_ms, rep_ms):
    plan_dir = report_dir / "plans" / str(plan_id)
    compile_started = datetime.now(timezone.utc).isoformat()
    try:
        compiled, compile_ms = _compile(kernel, args, grid, kwargs, "apply", dump, plan_id)
        actual_report_dir = _report_dir(compiled)
        if actual_report_dir != report_dir:
            raise RuntimeError(
                f"ScopeProfile report directory changed from {report_dir} to {actual_report_dir}.")
        compile_result = {
            "status": "valid",
            "compile_ms": compile_ms,
            "timestamp": compile_started,
        }
        _write_json(plan_dir / "compile.json", compile_result)
    except Exception as error:
        result = {
            "status": "invalid",
            "failed_phase": "compile",
            "error": _error_payload(error),
        }
        _write_json(plan_dir / "compile.json", result)
        _write_json(plan_dir / "result.json", result)
        return result

    with _profile_environment("apply", dump, plan_id):
        launch = lambda: kernel[grid](*args, **kwargs)
        try:
            correctness = _run_correctness(launch, reset, check)
        except Exception as error:
            result = {
                "status": "invalid",
                "failed_phase": "correctness",
                "error": _error_payload(error),
            }
            _write_json(plan_dir / "result.json", result)
            return result

        try:
            timing = _benchmark(launch, reset, warmup_ms, rep_ms)
        except Exception as error:
            result = {
                "status": "invalid",
                "failed_phase": "runtime",
                "error": _error_payload(error),
            }
            _write_json(plan_dir / "result.json", result)
            return result

    result = {
        "status": "valid",
        "correctness": "passed",
        "check_result": correctness,
        "timing": timing,
    }
    _write_json(plan_dir / "result.json", result)
    return result


def _apply_existing(kernel, args, grid, kwargs, dump, plan_id, reset, check, warmup_ms, rep_ms):
    compiled, compile_ms = _compile(kernel, args, grid, kwargs, "apply", dump, plan_id)
    report_dir = _report_dir(compiled)
    manifest = json.loads((report_dir / "manifest.json").read_text())
    if not any(int(plan["id"]) == plan_id for plan in manifest["plans"]):
        raise ValueError(f"ScopeProfile plan_id {plan_id} does not exist.")
    plan_dir = report_dir / "plans" / str(plan_id)
    _write_json(plan_dir / "compile.json", {
        "status": "valid",
        "compile_ms": compile_ms,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    with _profile_environment("apply", dump, plan_id):
        launch = lambda: kernel[grid](*args, **kwargs)
        correctness = _run_correctness(launch, reset, check)
        timing = _benchmark(launch, reset, warmup_ms, rep_ms)
    result = {
        "status": "valid",
        "correctness": "passed",
        "check_result": correctness,
        "timing": timing,
    }
    _write_json(plan_dir / "result.json", result)
    return report_dir, result


def _summary(manifest, results):
    valid = [(plan_id, result) for plan_id, result in results.items() if result["status"] == "valid"]
    best = min(valid, key=lambda item: item[1]["timing"]["median_ms"]) if valid else None
    baseline = results.get(0)
    baseline_ms = (baseline["timing"]["median_ms"]
                   if baseline and baseline.get("status") == "valid" else None)
    best_ms = best[1]["timing"]["median_ms"] if best else None
    failures = {"compile": 0, "correctness": 0, "runtime": 0}
    for result in results.values():
        phase = result.get("failed_phase")
        if phase in failures:
            failures[phase] += 1
    return {
        "schema_version": 1,
        "search_model": manifest["search_model"],
        "search_complete": len(results) == len(manifest["plans"]),
        "candidate_count": len(manifest["plans"]),
        "attempted_count": len(results),
        "valid_count": len(valid),
        "invalid_count": len(results) - len(valid),
        "failure_counts": failures,
        "baseline_plan_id": 0,
        "baseline_median_ms": baseline_ms,
        "best_plan_id": best[0] if best else None,
        "best_median_ms": best_ms,
        "speedup_over_empty": baseline_ms / best_ms if baseline_ms and best_ms else None,
    }


def run(kernel, *args, grid, reset=None, check=None, mode=None, dump=None, plan_id=None,
        warmup_ms=25, rep_ms=100, **kernel_kwargs):
    mode = (mode or os.environ.get(_MODE_ENV, "off")).strip().lower()
    if mode not in {"plan", "tune", "apply"}:
        raise ValueError("ScopeProfile runner mode must be plan, tune, or apply.")
    dump = dump or os.environ.get(_DUMP_ENV)
    if not dump:
        raise ValueError("ScopeProfile requires a dump directory.")

    if mode == "plan":
        report_dir, manifest = _discover(kernel, args, grid, kernel_kwargs, dump)
        return {"mode": mode, "report_dir": str(report_dir), "manifest": manifest}

    if mode == "apply":
        selected = _selected_plan_id(plan_id)
        if selected is None:
            raise ValueError("ScopeProfile apply mode requires plan_id.")
        report_dir, result = _apply_existing(
            kernel, args, grid, kernel_kwargs, dump, selected, reset,
            check or (lambda: None), warmup_ms, rep_ms)
        return {"mode": mode, "report_dir": str(report_dir), "plan_id": selected, "result": result}

    if check is None:
        raise ValueError("ScopeProfile tune mode requires a correctness callback.")
    report_dir, manifest = _discover(kernel, args, grid, kernel_kwargs, dump)
    results = {}
    for plan in manifest["plans"]:
        current_id = int(plan["id"])
        results[current_id] = _apply_one(
            kernel, args, grid, kernel_kwargs, dump, report_dir, current_id,
            reset, check, warmup_ms, rep_ms)
    summary = _summary(manifest, results)
    _write_json(report_dir / "summary.json", summary)
    return {"mode": mode, "report_dir": str(report_dir), "summary": summary}


__all__ = ["run"]
