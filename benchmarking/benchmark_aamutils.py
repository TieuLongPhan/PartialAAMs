import sys
import time
from pathlib import Path

from synutility.SynIO.debug import setup_logging
from synutility.SynIO.data_type import load_from_pickle, save_to_pickle


root_dir = Path(__file__).parents[1]
sys.path.append(str(root_dir))
from partialaams.aam_expand import extend_aam_from_graph

logger = setup_logging(
    "INFO",
    f"{root_dir}/Data//aamutils_log.txt",
)


if __name__ == "__main__":

    logger.info("Load database")
    data = load_from_pickle(f"{root_dir}/Data/test_dataset_partial_aam.pkl.gz")[2:3]
    logger.info("Start to extend")
    start_time = time.time()  # Start the timer
    for value in data:
        try:
            value["AAMUtils"] = extend_aam_from_graph(value["G"], value["H"])
        except Exception as e:
            logger.error(e)

    total_time = time.time() - start_time  # Total processing time
    average_time = total_time / len(data) if data else 0
    logger.info(f"Total processing time: {total_time:.2f} seconds")
    logger.info(f"Average time per entry: {average_time:.2f} seconds")

    save_to_pickle(
        data,
        f"{root_dir}/Data/test_dataset_aamutils.pkl.gz",
    )
