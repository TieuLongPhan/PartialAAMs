import time
import pandas as pd
from tabulate import tabulate

from synutility.SynIO.debug import setup_logging
from synutility.SynIO.data_type import load_database
from partialaams.aam_expand import partial_aam_extension_from_smiles
from synutility.SynAAM.aam_validator import AAMValidator

# Setup logging for debugging
logger = setup_logging(
    "INFO",
    "./Data/log.txt",
)


# Load the dataset
def load_data():
    logger.info("Loading database")
    data = load_database("Data/benchmark.json.gz")[0:]
    logger.info(f"Loaded {len(data)} entries")
    return data


# Perform atom mapping extension for each method and track processing time
def benchmark_extension(data, methods):
    results = []

    for method in methods:
        logger.info(f"Starting extension with method: {method}")
        start_time = time.time()

        # Iteratively apply atom mapping extension for each method
        for value in data:
            try:
                value[f"{method}"] = partial_aam_extension_from_smiles(
                    value["PartialAAM"], method=method
                )
            except Exception as e:
                value[f"AAMUtils_{method}"] = None
                logger.error(f"Error processing entry with method {method}: {e}")

        total_time = time.time() - start_time  # Total processing time for this method
        average_time = total_time / len(data) if data else 0

        # Log results for time taken for this method
        logger.info(f"Total processing time with {method}: {total_time:.2f} seconds")
        logger.info(f"Average time per entry with {method}: {average_time:.2f} seconds")

        # Validate smiles for this method
        logger.info(f"Validating results for method: {method}")
        validation_results = AAMValidator.validate_smiles(data, "RSMI", [method])

        # Process the validation results and store them along with the time
        for result in validation_results:
            result["method"] = method
            result["total_time"] = total_time
            result["average_time"] = average_time

            # Append the processed result for this method to the results list
            results.append(result)

    return results


# Save the final results to a CSV file
def save_results(results):
    logger.info("Saving results")

    # Convert results to DataFrame for easier handling and analysis
    df_results = pd.DataFrame(results)

    # Save the results to a CSV file
    df_results.to_csv("/Data/benchmark_results.csv", index=False)
    logger.info("Results saved to /Data/benchmark_results.csv")


# Log results in a table format
def log_results_table(results):
    # Prepare table for logging
    table = []
    headers = [
        "Method",
        "Accuracy",
        "Success Rate",
        "Total Time (s)",
        "Average Time (s)",
        "Validation Results",
    ]

    for result in results:
        row = [
            result["method"],
            result["accuracy"],
            result["success_rate"],
            f"{result['total_time']:.2f}",
            f"{result['average_time']:.2f}",
        ]
        table.append(row)

    # Print the table to the log file using tabulate
    logger.info("\n" + tabulate(table, headers=headers, tablefmt="grid"))


# Main execution
if __name__ == "__main__":
    # Load data
    data = load_data()

    # List of methods to benchmark
    # methods = ['ilp', 'gm', 'syn']
    methods = ["ilp", "syn"]

    # Run benchmarking and gather results
    results = benchmark_extension(data, methods)

    # Save the results to a file
    # save_results(results)

    # Log the results as a table in the log file
    log_results_table(results)

    # Optionally, save the processed data back to the database after extensions
    logger.info("Saving extended data to the database")
