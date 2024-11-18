import sys
import time
from pathlib import Path
from synutility.SynIO.debug import setup_logging
from synutility.SynIO.data_type import load_from_pickle, save_to_pickle

root_dir = Path(__file__).parents[1]
sys.path.append(str(root_dir))
from partialaams.gm_expand import gm_extend_from_graph

logger = setup_logging(
    "INFO",
    f"{root_dir}/Data/gm_log.txt",
)


if __name__ == "__main__":
    logger.info("Load database")
    data = load_from_pickle(f"{root_dir}/Data/test_dataset_partial_aam.pkl.gz")[2:3]
    logger.info("Start to extend")
    start_time = time.time()  # Start the timer
    for value in data:
        try:
            r = gm_extend_from_graph(value["G"], value["H"])
            value["GM"] = r[0]
        except Exception as e:
            value["length"] = 0
            logger.error(e)

    total_time = time.time() - start_time  # Total processing time
    average_time = (
        total_time / len(data) if data else 0
    )  # Calculate average time per entry

    # Print total and average processing time
    logger.info(f"Total processing time: {total_time:.2f} seconds")
    logger.info(f"Average time per entry: {average_time:.2f} seconds")
    n = [value for value in data if len(value["GM"]) > 1]
    logger.info(f"More than 1 solution :{len(n)}")

    save_to_pickle(
        data,
        f"{root_dir}/Data/test_dataset_gm.pkl.gz",
    )
