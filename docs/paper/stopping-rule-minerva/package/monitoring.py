"""Process-tree monitoring reused from the completed atlas harness."""
import csv
import importlib.metadata
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
import time
import psutil
from common import digest, read_json, write_json

def environment():
    packages = {d.metadata["Name"].lower(): d.version for d in importlib.metadata.distributions()}
    import rustscenic
    extensions = sorted(Path(rustscenic.__file__).parent.glob("*.so"))
    cpu = ""
    if Path("/proc/cpuinfo").exists():
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), "")
    return {"python": sys.version, "packages": packages, "platform": platform.platform(),
            "host": socket.gethostname(), "cpu": cpu, "logical_cpus": psutil.cpu_count(),
            "physical_cpus": psutil.cpu_count(logical=False), "ram_bytes": psutil.virtual_memory().total,
            "extension_hashes": {p.name: digest(p) for p in extensions},
            "lsf": {k: os.environ[k] for k in ("LSB_JOBID", "LSB_HOSTS", "LSB_DJOB_NUMPROC") if k in os.environ}}

def tree_memory(pid):
    """PSS is physical proportional memory; RSS is only a diagnostic sum."""
    try:
        parent = psutil.Process(pid)
        processes = [parent] + parent.children(recursive=True)
    except psutil.NoSuchProcess:
        return {"rss_mb": 0.0, "pss_mb": None, "missing": 0, "processes": 0}
    rss = pss = 0
    missing = live = 0
    for proc in processes:
        try:
            info = proc.memory_full_info()
            rss += info.rss
            live += 1
            if hasattr(info, "pss"):
                pss += info.pss
            else:
                missing += 1
        except psutil.NoSuchProcess:
            continue  # It exited during sampling; not an unreadable live process.
        except psutil.AccessDenied:
            missing += 1
            try:
                rss += proc.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    return {"rss_mb": rss / 1e6, "pss_mb": pss / 1e6 if live and not missing else None,
            "missing": missing, "processes": live}

def kill_group(proc):
    descendants = []
    try:
        descendants = psutil.Process(proc.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        pass
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    except ProcessLookupError:
        proc.wait()
    finally:
        # A terminated parent does not prove every pool/worker has exited.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        for child in descendants:
            try:
                child.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

def monitored(command, out, timeout, config):
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    env = os.environ.copy()
    env.update({k: "1" for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]})
    env.update(RAYON_NUM_THREADS=str(config["threads"]), PYTHONUNBUFFERED="1")
    env.update(config.get("env", {}))
    status, peak_pss, peak_rss, incomplete, samples = "complete", None, 0.0, 0, 0
    analysis_peak, analysis_incomplete, analysis_samples = None, 0, 0
    with (out / "run.log").open("w") as log, (out / "memory.csv").open("w") as mem:
        writer = csv.DictWriter(mem, fieldnames=["elapsed_s", "phase", "rss_mb", "pss_mb", "missing", "processes", "node_load_1m", "node_memory_available_mb"])
        writer.writeheader()
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            while proc.poll() is None:
                elapsed = time.perf_counter() - started
                sample = tree_memory(proc.pid)
                phase_file = out / "phase.json"
                phase = read_json(phase_file)["phase"] if phase_file.exists() else "startup"
                writer.writerow({"elapsed_s": round(elapsed, 3), "phase": phase, **sample,
                                 "node_load_1m": os.getloadavg()[0],
                                 "node_memory_available_mb": psutil.virtual_memory().available / 1e6})
                mem.flush()
                samples += 1
                incomplete += int(sample["missing"] > 0)
                peak_rss = max(peak_rss, sample["rss_mb"])
                if sample["pss_mb"] is not None:
                    peak_pss = max(peak_pss or 0, sample["pss_mb"])
                if phase in {"analysis", "grn_inference", "grn_correlation"}:
                    analysis_samples += 1
                    analysis_incomplete += int(sample["missing"] > 0)
                    if sample["pss_mb"] is not None:
                        analysis_peak = max(analysis_peak or 0, sample["pss_mb"])
                if elapsed >= timeout:
                    status = "timeout"
                    kill_group(proc)
                    break
                if sample["pss_mb"] is not None and sample["pss_mb"] > config["memory_limit_mb"]:
                    status = "memory_limit"
                    kill_group(proc)
                    break
                time.sleep(config["sample_interval_seconds"])
        except BaseException:
            kill_group(proc)
            raise
        if status == "complete" and proc.returncode != 0:
            status = "failed"
    result = {"status": status, "exit_code": proc.returncode,
              "process_wall_s": time.perf_counter() - started,
              "peak_tree_pss_mb": peak_pss, "peak_summed_rss_mb": peak_rss,
              "pss_incomplete_samples": incomplete, "memory_samples": samples,
              "analysis_peak_pss_mb": analysis_peak, "analysis_pss_incomplete_samples": analysis_incomplete,
              "analysis_memory_samples": analysis_samples,
              "timeout_seconds": timeout, "command": command}
    write_json(out / "monitor.json", result)
    return result
