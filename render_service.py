"""
render_service.py — local HTTP wrapper around the Equations Rendered engine.

Runs natively on the GPU host (NOT inside Docker/n8n's container), so it has
direct access to cupy/CUDA exactly as renderer.py already does when run from
the command line. n8n's HTTP Request node calls this service instead of
trying to execute Python inside its own container — see
n8n-pipeline-build-plan.md for the full architecture reasoning.

Renders and test runs are genuinely slow (minutes, not seconds), so every
operation is asynchronous: POST an endpoint, get a job_id back immediately,
poll GET /jobs/{job_id} until status is "complete" or "error". Nothing here
blocks an HTTP request open for the duration of a render.

Run with:  python render_service.py
Then:      GET http://localhost:8420/health   (or host.docker.internal:8420 from inside Docker)
"""

import os
import sys
import json
import uuid
import time
import glob
import shutil
import subprocess
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "tests"))

OUTPUT_DIR = os.path.join(REPO_ROOT, "output", "service_renders")
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="Equations Rendered — render service")

# Bounded thread pool: renders are GPU/CPU-bound blocking subprocess calls,
# not real Python concurrency — this just lets a few run without literally
# serializing every request through one thread. Keep this small; the GPU
# itself is the real bottleneck, not the pool size.
executor = ThreadPoolExecutor(max_workers=2)

# In-memory job store. This is a single-user local service — no need for a
# database. NOTE: job history is lost on service restart; if that ever
# matters (e.g. n8n polling a job across a service restart), swap this for
# a small SQLite file. Not needed yet.
_jobs_lock = threading.Lock()
_jobs = {}


def _sanitize_for_json(obj):
    """
    Recursively converts numpy scalar/array types to native Python
    equivalents. Needed because test_visuals.py's OCR/SSIM functions return
    numpy types directly (np.True_, np.float64, np.int64, etc. — visible
    throughout this project's actual test output, e.g. "'passed': np.True_"),
    and FastAPI's default JSON encoder cannot serialize those, producing a
    500 error on every job-status poll for a visual-check job. Confirmed via
    a real reproduction: /jobs/visual-check completed successfully
    server-side, but every subsequent GET /jobs/{job_id} crashed with
    `TypeError: 'numpy.bool' object is not iterable` inside FastAPI's
    encoder. Fixed by sanitizing job results at storage time, not just
    hoping the caller never hits a numpy type.
    """
    try:
        import numpy as np
    except ImportError:
        np = None

    if np is not None:
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return _sanitize_for_json(obj.tolist())

    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _new_job(job_type: str) -> str:
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "type": job_type,
            "status": "queued",
            "created_at": time.time(),
            "started_at": None,
            "completed_at": None,
            "result": None,
            "error": None,
        }
    return job_id


def _set_job(job_id: str, **fields):
    if "result" in fields:
        fields["result"] = _sanitize_for_json(fields["result"])
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(fields)


def _get_job(job_id: str) -> Optional[dict]:
    with _jobs_lock:
        return dict(_jobs[job_id]) if job_id in _jobs else None


# ---------------------------------------------------------------------------
# /render — wraps renderer.py exactly as it's invoked from the CLI today
# ---------------------------------------------------------------------------

class RenderRequest(BaseModel):
    simulation_name: str
    duration_override: Optional[float] = None


def _run_render_job(job_id: str, simulation_name: str, duration_override: Optional[float]):
    _set_job(job_id, status="running", started_at=time.time())
    try:
        config_path = os.path.join(REPO_ROOT, "configs", f"{simulation_name}.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No config found for simulation '{simulation_name}' at {config_path}")

        job_output_dir = os.path.join(OUTPUT_DIR, job_id)
        os.makedirs(job_output_dir, exist_ok=True)
        output_path = os.path.join(job_output_dir, f"{simulation_name}.mp4")
        baseline_dir = os.path.join(job_output_dir, "baselines")

        cmd = [
            sys.executable, os.path.join(REPO_ROOT, "renderer.py"),
            "--config", config_path,
            "--output", output_path,
            "--baseline-dir", baseline_dir,
        ]
        # NOTE: renderer.py currently takes duration from the YAML config,
        # not a CLI flag. If per-request duration overrides are needed,
        # renderer.py would need a --duration-override flag added — not
        # done yet since no caller needs it today. Documented here so it's
        # a known, deliberate gap rather than a silent one.
        if duration_override is not None:
            raise NotImplementedError(
                "duration_override requested but renderer.py has no CLI flag for "
                "this yet — would need to be added to renderer.py first."
            )

        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=1800)

        if proc.returncode != 0:
            _set_job(job_id, status="error", completed_at=time.time(),
                      error=f"renderer.py exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
            return

        provenance_path = os.path.join(job_output_dir, f"{simulation_name}_provenance.json")
        provenance = None
        if os.path.exists(provenance_path):
            with open(provenance_path) as f:
                provenance = json.load(f)

        baseline_frames = {}
        if os.path.isdir(baseline_dir):
            for f in sorted(glob.glob(os.path.join(baseline_dir, "*.png"))):
                name = os.path.basename(f)
                if "10pct" in name:
                    baseline_frames["start"] = f
                elif "50pct" in name:
                    baseline_frames["mid"] = f
                elif "90pct" in name:
                    baseline_frames["end"] = f

        _set_job(job_id, status="complete", completed_at=time.time(), result={
            "output_path": output_path,
            "provenance": provenance,
            "baseline_frames": baseline_frames,
            "stdout_tail": proc.stdout[-2000:],
        })
    except subprocess.TimeoutExpired:
        _set_job(job_id, status="error", completed_at=time.time(),
                  error="Render exceeded 30 minute timeout — likely stuck, not just slow.")
    except Exception as e:
        _set_job(job_id, status="error", completed_at=time.time(),
                  error=f"{e}\n{traceback.format_exc()}")


@app.post("/jobs/render")
def start_render(req: RenderRequest):
    job_id = _new_job("render")
    executor.submit(_run_render_job, job_id, req.simulation_name, req.duration_override)
    return {"job_id": job_id}


# ---------------------------------------------------------------------------
# /verify — physics/math correctness (TEST_SPEC categories)
# ---------------------------------------------------------------------------

class VerifyRequest(BaseModel):
    simulation_name: str


def _run_verify_job(job_id: str, simulation_name: str):
    _set_job(job_id, status="running", started_at=time.time())
    try:
        import run_physics_tests as rpt
        result = rpt.run_tests_for_module(simulation_name)
        _set_job(job_id, status="complete", completed_at=time.time(), result=result)
    except Exception as e:
        _set_job(job_id, status="error", completed_at=time.time(),
                  error=f"{e}\n{traceback.format_exc()}")


@app.post("/jobs/verify")
def start_verify(req: VerifyRequest):
    job_id = _new_job("verify")
    executor.submit(_run_verify_job, job_id, req.simulation_name)
    return {"job_id": job_id}


# ---------------------------------------------------------------------------
# /visual-check — OCR + SSIM against a rendered frame
# ---------------------------------------------------------------------------

class VisualCheckRequest(BaseModel):
    simulation_name: str
    frame_type: str  # "start" | "mid" | "end"
    frame_path: str
    baseline_dir: str = "tests/baseline_frames"


def _run_visual_check_job(job_id: str, req: VisualCheckRequest):
    _set_job(job_id, status="running", started_at=time.time())
    try:
        import test_visuals as tv
        ocr_result = tv.test_ocr_overlays(req.simulation_name, req.frame_path, configs_dir=os.path.join(REPO_ROOT, "configs"))
        consistency_result = tv.test_visual_consistency(req.simulation_name, req.frame_type, req.frame_path, baseline_dir=req.baseline_dir)
        overall_passed = bool(ocr_result.get("passed")) and bool(consistency_result.get("passed"))
        _set_job(job_id, status="complete", completed_at=time.time(), result={
            "passed": overall_passed,
            "ocr": ocr_result,
            "consistency": consistency_result,
        })
    except Exception as e:
        _set_job(job_id, status="error", completed_at=time.time(),
                  error=f"{e}\n{traceback.format_exc()}")


@app.post("/jobs/visual-check")
def start_visual_check(req: VisualCheckRequest):
    job_id = _new_job("visual_check")
    executor.submit(_run_visual_check_job, job_id, req)
    return {"job_id": job_id}


# ---------------------------------------------------------------------------
# Job status polling — same shape for every job type
# ---------------------------------------------------------------------------

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No job with id {job_id}")
    return job


@app.get("/files/{job_id}/{filename:path}")
def get_file(job_id: str, filename: str):
    """Lets n8n fetch the actual rendered video/frame files by job_id,
    e.g. for attaching the full video to the Telegram approval message."""
    path = os.path.join(OUTPUT_DIR, job_id, filename)
    if not os.path.exists(path) or not os.path.abspath(path).startswith(os.path.abspath(OUTPUT_DIR)):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)


# ---------------------------------------------------------------------------
# Health check — n8n should ping this first so a dead service fails loudly
# and immediately, rather than every downstream node timing out separately.
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    cupy_available = False
    gpu_name = None
    try:
        import cupy as cp
        cupy_available = True
        gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except Exception:
        pass
    return {
        "status": "ok",
        "cupy_available": cupy_available,
        "gpu_name": gpu_name,
        "repo_root": REPO_ROOT,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8420)
