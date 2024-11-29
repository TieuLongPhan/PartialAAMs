import sys
import time
from pathlib import Path
from multiprocessing import Pool, Manager
from synutility.SynIO.debug import setup_logging
from synutility.SynIO.data_type import load_database, save_database


root_dir = Path(__file__).parents[1]
sys.path.append(str(root_dir))
from partialaams.gm_expand import gm_extend_aam_from_rsmi

logger = setup_logging(
    "INFO",
    f"{root_dir}/Data/gm_log.txt",
)


def process_entry(value):
    """Function to process each entry in parallel."""
    try:
        logger.info(value["PartialAAM"])
        value["GM"] = gm_extend_aam_from_rsmi(value["PartialAAM"])

    except Exception as e:
        value["GM"] = None
        logger.error(f"Error processing entry {value}: {e}")
    return value


def main():
    logger.info("Load database")
    data = load_database(f"{root_dir}/Data/benchmark.json.gz")[19:20]
    logger.info("Start to extend")

    # Use a Manager to share data between processes
    # with Manager() as manager:
    # results = manager.list()

    # Start the timer
    start_time = time.time()

    # Set up multiprocessing Pool
    # with Pool() as pool:
    # Map the process_entry function to each data entry
    # results = pool.map(process_entry, data)
    for value in data:
        results = process_entry(value)

    total_time = time.time() - start_time  # Total processing time

    # if total_time > 1:
    #     logger.warning(
    #         f"Warning: Total processing time exceeded 1 second: {total_time:.2f} seconds"
    #     )

    average_time = total_time / len(data) if data else 0
    logger.info(f"Total processing time: {total_time:.2f} seconds")
    logger.info(f"Average time per entry: {average_time:.2f} seconds")

    # # Checking for entries where GM has more than 1 solution
    # n = [value for value in results if len(value["GM"]) > 1]
    # logger.info(f"More than 1 solution: {len(n)}")

    # Optionally save the results if needed
    # save_database(list(results), f"{root_dir}/Data/benchmark.pkl.gz")


if __name__ == "__main__":
    main()
