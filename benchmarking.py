import os
import time
import gc
import psutil
import pandas as pd
from tabulate import tabulate
from multiprocessing import Process, Queue

from synkit.IO.debug import setup_logging
from synkit.IO.data_io import load_database
from partialaams.aam_expand import partial_aam_extension_from_smiles
from synkit.Graph.ITS.aam_validator import AAMValidator

# Setup logging for debugging
logger = setup_logging("INFO", "./Data/log_test.txt")

# Paths for entry times CSVs
entry_times_dir = "Data/entry_times"
final_entry_csv = "Data/entry_times.csv"

os.makedirs(entry_times_dir, exist_ok=True)

def load_data(limit=None):
    logger.info("Loading database")
    data = load_database("Data/benchmark.json.gz")
    if limit:
        data = data[:limit]
    logger.info(f"Loaded {len(data)} entries")
    return data

def run_single_method(method, limit, queue):
    """
    Load data, benchmark one method in isolation (child process),
    append per-entry timing to per-method CSV, and put summary metrics on the queue.
    """
    data = load_data(limit)
    process = psutil.Process(os.getpid())
    gc.collect()
    base_mem = process.memory_info().rss

    entry_times = []
    rows = []
    # Per-entry timing
    for idx, entry in enumerate(data):
        start_entry = time.perf_counter()
        try:
            entry[method] = partial_aam_extension_from_smiles(
                entry["partial"], method=method
            )
        except Exception:
            entry[method] = None
        elapsed = time.perf_counter() - start_entry
        key = f"{method}_time_s"
        entry[key] = round(elapsed, 4)
        entry_times.append(elapsed)
        rows.append({
            "id": idx,
            key: round(elapsed, 4)
        })

    # Write per-method CSV
    df_rows = pd.DataFrame(rows)
    csv_path = os.path.join(entry_times_dir, f"{method}_times.csv")
    df_rows.to_csv(csv_path, index=False)

    # Total timing and memory
    total_time = sum(entry_times)
    peak_rss = process.memory_info().rss
    peak_rss_mib = (peak_rss - base_mem) / (1024**2)

    # Validate
    valid = [e for e in data if e[method]]
    validation = AAMValidator.validate_smiles(valid, "smart", [method], "ITS")
    accuracies = [res["accuracy"] for res in validation]
    accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
    success_rate = len(valid) / len(data) if data else 0.0

    # Package summary
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
    """
    After all per-method CSVs are written, merge them into one table.
    """
    dfs = []
    for method in methods:
        path = os.path.join(entry_times_dir, f"{method}_times.csv")
        df = pd.read_csv(path)
        dfs.append(df)
    # Merge all on 'id'
    from functools import reduce
    merged = reduce(lambda left, right: pd.merge(left, right, on='id'), dfs)
    merged.to_csv(final_entry_csv, index=False)
    logger.info(f"Merged entry times written to {final_entry_csv}")


def log_results_table(summaries):
    """
    Log the summary metrics in a simple table.
    """
    headers = [
        "Method",
        "Accuracy",
        "Success Rate",
        "Total Time (s)",
        "Avg Time (s)",
        "Peak RSS (MiB)",
    ]
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
    limit = None  # or set to an integer for subset

    # Clear old per-method CSVs and final
    if os.path.exists(entry_times_dir):
        for f in os.listdir(entry_times_dir):
            os.remove(os.path.join(entry_times_dir, f))
    if os.path.exists(final_entry_csv):
        os.remove(final_entry_csv)

    queue = Queue()
    processes = []
    for m in methods:
        p = Process(target=run_single_method, args=(m, limit, queue))
        p.start()
        processes.append(p)

    summaries = [queue.get() for _ in methods]
    for p in processes:
        p.join()

    # Ensure consistent ordering
    order = {m: i for i, m in enumerate(methods)}
    summaries.sort(key=lambda s: order[s["method"]])

    log_results_table(summaries)
    merge_entry_times(methods)
