import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def load_gsa_results(file_name="gsa_results_morris.csv", run_id=None):
    """
    Load the GSA results from a file.
    Defaults to the last run if run_id is not specified.

    :param file_name: Name of the file to load results from
    :param run_id: Specific run ID to load (optional)
    :return: GSA results DataFrame and the selected run ID
    """
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"The file {file_name} does not exist.")

    results_df = pd.read_csv(file_name)
    if run_id is None:
        run_id = results_df['run_id'].max()  # Default to last run

    run_results = results_df[results_df['run_id'] == run_id]
    if run_results.empty:
        raise ValueError(f"No results found for run ID {run_id}.")

    print(f"Loaded results for run ID: {run_id}")
    return run_results, run_id


def compute_elementary_effects(param_values, model_outputs, names, delta=1.0):
    """
    Compute elementary effects (EEs) manually.

    :param param_values: Array of parameter values (N x D)
    :param model_outputs: Array of model outputs (N,)
    :param names: List of parameter names
    :param delta: Step size used in Morris sampling
    :return: Dictionary of elementary effects for each parameter
    """
    N, D = param_values.shape
    EEs = {name: [] for name in names}

    for i in range(D):  # For each parameter
        for j in range(N - 1):  # Pairwise differences
            diff_output = model_outputs[j + 1] - model_outputs[j]
            diff_param = param_values[j + 1, i] - param_values[j, i]
            if diff_param != 0:
                EEs[names[i]].append(diff_output / diff_param)

    return EEs


def plot_ee_histogram(EEs, parameter, bins=20):
    """
    Plot histogram of elementary effects for a specific parameter.

    :param EEs: Dictionary of elementary effects
    :param parameter: Parameter name to plot
    :param bins: Number of histogram bins
    """
    if parameter not in EEs:
        raise ValueError(f"Parameter {parameter} not found in EEs.")

    plt.hist(EEs[parameter], bins=bins)
    plt.title(f"Histogram of Elementary Effects for {parameter}")
    plt.xlabel("Elementary Effect")
    plt.ylabel("Frequency")
    plt.show()

# Example Usage
import numpy as np
def expand_dynamic_names(names, param_values):
    """
    Expand grouped dynamic names to match the structure of param_values.

    :param names: List of parameter names (some may be grouped like 'cf_wind').
    :param param_values: Array of parameter values (N x D).
    :return: Expanded list of parameter names.
    """
    expanded_names = []
    num_dynamic_steps = param_values.shape[1] - len(names)  # Estimate dynamic steps
    for name in names:
        if name.startswith('cf_') or name == 'el_price':  # Identify dynamic groups
            expanded_names.extend([f"{name}_{i}" for i in range(num_dynamic_steps)])
        else:
            expanded_names.append(name)
    return expanded_names


# Load parameter values and model outputs
try:
    param_values = np.load("param_values.npy")  # Load parameter values
    model_outputs = np.load("model_outputs.npy")  # Load model outputs
    raw_names = list(pd.read_csv("gsa_results_morris.csv")['name'].unique())  # Raw names from GSA results
    expanded_names = expand_dynamic_names(raw_names, param_values)  # Expand names
    print(f"param_values shape: {param_values.shape}")
    print(f"Number of expanded names: {len(expanded_names)}")

except FileNotFoundError as e:
    print(f"File not found: {e}")
    exit(1)

# Compute Elementary Effects
try:
    EEs = compute_elementary_effects(param_values, model_outputs, expanded_names)
    print("Elementary Effects computed successfully.")
except Exception as e:
    print(f"Error computing Elementary Effects: {e}")

for param, ees in EEs.items():
    if "cf_wind" in param:  # Or any other dynamic parameter
        print(f"Elementary Effects for {param}: {ees}")


# Example: Inspect dynamic parameters
try:
    plot_ee_histogram(EEs, "cf_wind_0")  # Example for first time step
    plot_ee_histogram(EEs, "cf_wind_1")  # Example for second time step
except Exception as e:
    print(f"Error plotting EE histogram: {e}")
