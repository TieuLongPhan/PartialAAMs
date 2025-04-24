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
    and put summary metrics on the queue.
    """
    data = load_data(limit)
    process = psutil.Process(os.getpid())
    gc.collect()
    base_mem = process.memory_info().rss

    # Timing and mem
    start = time.perf_counter()
    for entry in data:
        try:
            entry[method] = partial_aam_extension_from_smiles(
                entry["partial"], method=method
            )
        except Exception as e:
            entry[method] = None
    total_time = time.perf_counter() - start

    # Peak RSS
    peak_rss = process.memory_info().rss
    peak_rss_mib = (peak_rss - base_mem) / (1024**2)

    # Validate
    valid = [e for e in data if e[method]]
    validation = AAMValidator.validate_smiles(valid, "smart", [method], "ITS")
    
    # Compute overall accuracy and success rate
    accuracies = [res["accuracy"] for res in validation]
    accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
    success_rate = len(valid) / len(data) if data else 0.0

    # Package summary
    summary = {
        "method": method,
        "accuracy": round(accuracy, 4),
        "success_rate": round(success_rate, 4),
        "total_time_s": round(total_time, 2),
        "avg_time_s": round(total_time / len(data), 4),
        "peak_rss_mib": round(peak_rss_mib, 2),
    }
    queue.put(summary)

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
    # methods = ["gm", "extend", "extend_g"]
    methods = ["gm", "extend", "extend_g"]
    limit = None  # or None for full dataset

    queue = Queue()
    processes = []
    for m in methods:
        p = Process(target=run_single_method, args=(m, limit, queue))
        p.start()
        processes.append(p)

    summaries = [queue.get() for _ in methods]
    logger.info(summaries)
    for p in processes:
        p.join()

    # sort summaries into the same order as `methods`
    order = {m: i for i, m in enumerate(methods)}
    summaries.sort(key=lambda s: order[s["method"]])
    logger.info(summaries)

    log_results_table(summaries)

