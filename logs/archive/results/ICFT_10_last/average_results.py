#!/usr/bin/env python3

"""
Combines accuracy results from multiple JSON files by averaging them.

Example usage:
  python average_results.py file1.json file2.json file3.json
will compute the average for each "_accuracy" key and output it to averagedResults.json.

If no files are specified, it will process all JSON files in the current directory.

The output format will match the original file structure but with averaged accuracy values.
"""

import json
import os
import glob
import statistics

def average_results(file_paths=None, output_file="averagedResults.json"):
    """
    Averages accuracy results from multiple JSON files into a single JSON file.

    Each input JSON file is expected to be a list of objects of the form:
        [
          {
            "task": "<task_name>",
            "<some_method>_accuracy": <float>,
            ...
          },
          ...
        ]

    The output JSON will be a similar list but with averaged accuracy values for each task.
    """
    if file_paths is None:
        file_paths = glob.glob("*.json")

    if not file_paths:
        print("No JSON files found in the current directory.")
        return

    # Dictionary to store intermediate sums and counts for averaging
    results = {}

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist. Skipping.")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: {file_path} is not valid JSON or is empty. Skipping.")
                continue

            if not isinstance(data, list):
                print(f"Warning: {file_path} does not contain a list of tasks. Skipping.")
                continue

            for entry in data:
                task_name = entry.get("task")
                if not task_name:
                    continue

                # Initialize task entry if not already present
                if task_name not in results:
                    results[task_name] = {}

                # Process all keys ending with "_accuracy"
                for key, value in entry.items():
                    if key.endswith("_accuracy") and isinstance(value, (int, float)):
                        # Strip suffix numbers from accuracy keys for averaging
                        base_key = "_".join(key.split("_")[:-3]) + "_accuracy"
                        print(base_key)
                        if base_key not in results[task_name]:
                            results[task_name][base_key] = []
                        results[task_name][base_key].append(value)

    # Compute averages and format the output
    averaged_results = []
    for task_name, metrics in results.items():
        averaged_entry = {"task": task_name}
        for key, values in metrics.items():
            averaged_entry[key] = float(statistics.mean(values))
        averaged_results.append(averaged_entry)

    # Write the output file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(averaged_results, f, indent=2)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        average_results(file_paths=sys.argv[1:])
    else:
        average_results()
