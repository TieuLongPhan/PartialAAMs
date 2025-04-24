import time
from tabulate import tabulate
from synkit.IO.debug import setup_logging
from synkit.IO.data_io import load_database, save_database
from partialaams.aam_expand import partial_aam_extension_from_smiles

# Setup logging for debugging
logger = setup_logging("INFO", "./Data/gm_log.txt")
# Load only the first 10 records for demonstration
data = load_database("./Data/benchmark.json.gz")[:]


def gm_expand_run(data, key="p_partial"):
    """
    Processes each record in data using `partial_aam_extension_from_smiles`
    with the provided key and updates the record with a new key, '<key>_gm_expand'.
    """
    for value in data:
        try:
            result = partial_aam_extension_from_smiles(value[key], "gm")
        except Exception as e:
            logger.exception("Error processing value %s with key %s: %s", value, key, e)
            result = []
        value[f"{key}_gm_expand"] = result
    return data


config = ["p_partial", "r_partial", "b_partial"]
results_table = []

for conf in config:
    start_time = time.time()

    # Process the data with the given configuration
    data = gm_expand_run(data, key=conf)
    total_time = time.time() - start_time

    # Compute success rate: those records that have a non-empty expansion
    success = [value for value in data if value.get(f"{conf}_gm_expand")]
    success_rate = (len(success) / len(data)) * 100 if data else 0
    average_time = total_time / len(data) if data else 0

    redundant_total = sum(len(value.get(f"{conf}_gm_expand", [])) for value in data)
    redundant_avg = redundant_total / len(data) if data else 0
    logger.info("Redundant average for %s is %.2f", conf, redundant_avg)

    logger.info("Success rate for %s is %.2f%%", conf, success_rate)

    results_table.append(
        [
            conf,
            f"{success_rate:.2f}%",
            f"{total_time:.2f}",
            f"{average_time:.4f}",
            f"{redundant_avg:.2f}",
        ]
    )


table_headers = [
    "Config",
    "Success Rate",
    "Total Time (s)",
    "Average Time (s)",
    "Redundant Avg",
]
table_str = tabulate(results_table, headers=table_headers, tablefmt="grid")

save_database(data, "./Data/gm_expand.json.gz")

print(table_str)

logger.info("\n%s", table_str)
