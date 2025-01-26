#!/usr/bin/env python3

"""
First go to the directory this file is in
Example usage: 
  python combine_results.py
will combine all JSON files in the current directory into allResults.json.

Or: 
  python combine_results.py file1.json file2.json
to specify the input files explicitly.

Example:
cd ~/TTT/logs/current
python3 combine_results.py
"""

import json
import glob
import statistics
import os

def combine_results(file_paths=None, output_file="allResults.json"):
    """
    Combines accuracy results from multiple JSON files into a single JSON.
    
    Each input JSON file is expected to be a list of objects of the form:
        [
          {
            "task": "<task_name>",
            "<some_method>_accuracy": <float>,
            ...
          },
          ...
        ]
    Additional keys are allowed, but only those ending with '_accuracy' will be 
    collected and combined.

    The output JSON format will be:
    {
      "per_task_results": [
        {
          "task": "<task_name>",
          "<some_method>_accuracy": <float>,
          ...
        },
        ...
      ],
      "aggregated_statistics": {
        "<some_method>_accuracy": {
          "average": <float>,
          "median": <float>
        },
        ...
      }
    }
    """

    # If no file paths provided, default to all .json files in current directory.
    if file_paths is None:
        file_paths = glob.glob("*.json")

    # We'll store results in a dict keyed by task name.
    # Each value will be another dict of {accuracy_key: value}
    combined_results = {}

    for file_path in file_paths:
        # Skip the output file if it happens to exist in the same directory
        if os.path.basename(file_path) == output_file:
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: {file_path} is not valid JSON or is empty. Skipping.")
                continue
            
            # Each file is expected to be a list of task entries
            if not isinstance(data, list):
                print(f"Warning: {file_path} does not contain a list of tasks. Skipping.")
                continue
            
            for entry in data:
                # Each entry should have a "task" key
                task_name = entry.get("task")
                if not task_name:
                    # If there's no "task" key, skip
                    continue

                # If this is the first time we've seen this task, initialize it
                if task_name not in combined_results:
                    combined_results[task_name] = {}

                # Grab all keys that end with "_accuracy"
                for key, value in entry.items():
                    if key.endswith("_accuracy"):
                        combined_results[task_name][key] = value

    # Now, convert combined_results to a list suitable for "per_task_results"
    per_task_results = []
    for task_name, accuracies_dict in combined_results.items():
        entry = {"task": task_name}
        entry.update(accuracies_dict)
        per_task_results.append(entry)

    # Compute aggregated statistics: for each accuracy key, compute average & median
    # We'll gather all accuracy values in a dict keyed by accuracy method
    accuracy_values_by_method = {}
    for task_entry in per_task_results:
        for key, value in task_entry.items():
            # skip 'task' field
            if key == "task":
                continue
            if key not in accuracy_values_by_method:
                accuracy_values_by_method[key] = []
            accuracy_values_by_method[key].append(value)

    aggregated_statistics = {}
    for accuracy_key, values in accuracy_values_by_method.items():
        aggregated_statistics[accuracy_key] = {
            "average": float(statistics.mean(values)),
            "median": float(statistics.median(values))
        }

    # Final output structure
    final_output = {
        "per_task_results": per_task_results,
        "aggregated_statistics": aggregated_statistics
    }

    # Write to the output file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # user provided filenames
        combine_results(file_paths=sys.argv[1:])
    else:
        combine_results()
