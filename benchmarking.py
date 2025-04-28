import os
import time
import gc
import psutil
import pandas as pd
import shutil
from tabulate import tabulate
from multiprocessing import Process, Queue
from datetime import datetime

from synkit.IO.debug import setup_logging
from synkit.IO.data_io import load_database
from partialaams.aam_expand import partial_aam_extension_from_smiles
from synkit.Graph.ITS.aam_validator import AAMValidator

# --- Run‐specific setup with timestamp ---
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
entry_times_dir = os.path.join("Data", f"entry_times_{run_id}")
final_entry_csv = os.path.join("Data", f"entry_times_{run_id}.csv")

os.makedirs(entry_times_dir, exist_ok=True)

# Setup logging
logger = setup_logging("INFO", f"./Data/log.txt")
logger.info(f"Starting benchmark run {run_id}")

def load_data(limit=None):
    logger.info("Loading database")
    data = load_database("Data/benchmark.json.gz")
    if limit:
        data = data[:limit]
    logger.info(f"Loaded {len(data)} entries")
    return data

def run_single_method(method, limit, queue):
    data = load_data(limit)
    proc = psutil.Process(os.getpid())
    gc.collect()
    base_mem = proc.memory_info().rss

    rows = []
    key = f"{method}_time_s"
    for idx, entry in enumerate(data):
        start = time.perf_counter()
        try:
            entry[method] = partial_aam_extension_from_smiles(
                entry["partial"], method=method
            )
        except Exception:
            entry[method] = None
        elapsed = time.perf_counter() - start
        rows.append({"id": idx, key: round(elapsed, 4)})

    # save per-entry CSV
    df = pd.DataFrame(rows)
    csv_path = os.path.join(entry_times_dir, f"{method}_times.csv")
    df.to_csv(csv_path, index=False)

    # summary metrics
    total_time = df[key].sum()
    peak_rss_mib = (proc.memory_info().rss - base_mem) / (1024**2)

    valid = [e for e in data if e[method]]
    validation = AAMValidator.validate_smiles(valid, "smart", [method], "ITS")
    accuracies = [res["accuracy"] for res in validation]
    accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
    success_rate = len(valid) / len(data) if data else 0.0

    summary = {
        "method": method,
        "accuracy": round(accuracy, 4),
        "success_rate": round(success_rate, 4),
        "total_time_s": round(total_time, 2),
        "avg_time_s": round(total_time / len(data), 4) if data else 0.0,
        "peak_rss_mib": round(peak_rss_mib, 2),
    }
    queue.put(summary)

def merge_entry_times(methods):
    dfs = []
    for method in methods:
        path = os.path.join(entry_times_dir, f"{method}_times.csv")
        dfs.append(pd.read_csv(path))
    from functools import reduce
    merged = reduce(lambda l, r: pd.merge(l, r, on="id"), dfs)
    merged.to_csv(final_entry_csv, index=False)
    logger.info(f"Merged entry times → {final_entry_csv}")

    # clean up per-run directory
    shutil.rmtree(entry_times_dir)
    logger.info(f"Removed temporary directory {entry_times_dir}")

def log_results_table(summaries):
    headers = ["Method","Accuracy","Success Rate","Total Time (s)",
               "Avg Time (s)","Peak RSS (MiB)"]
    table = [
        [
            s["method"],
            f"{s['accuracy']:.4f}",
            f"{s['success_rate']:.4f}",
            f"{s['total_time_s']:.2f}",
            f"{s['avg_time_s']:.4f}",
            f"{s['peak_rss_mib']:.2f}",
        ]
        for s in summaries
    ]
    logger.info("\n" + tabulate(table, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    methods = ["gm", "extend", "extend_g", "ilp", "syn"]
    limit = None  # or a small integer for quick tests

    queue = Queue()
    processes = []
    for m in methods:
        p = Process(target=run_single_method, args=(m, limit, queue))
        p.start()
        processes.append(p)

    # collect summaries
    summaries = [queue.get() for _ in methods]
    for p in processes:
        p.join()

    # sort and log
    order = {m: i for i, m in enumerate(methods)}
    summaries.sort(key=lambda s: order[s["method"]])
    log_results_table(summaries)

    # merge per-method entry times into one CSV and cleanup
    merge_entry_times(methods)
