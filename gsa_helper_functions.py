import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pickle
import os

def save_gsa_results(s1_df, st_df, s2_df, method, n_samples):
    """
    Saves S1, ST, and S2 DataFrames to timestamped CSV files and returns the file names.

    Parameters:
        s1_df (pd.DataFrame): DataFrame containing S1 results.
        st_df (pd.DataFrame): DataFrame containing ST results.
        s2_df (pd.DataFrame): DataFrame containing S2 results.
        method (str): GSA method used (e.g., "sobol").
        n_samples (int): Number of samples used in GSA.

    Returns:
        tuple: File names of the saved S1, ST, and S2 DataFrames.
    """
    # Generate timestamped file names
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    s1_filename = f"gsa_results_{method}_{n_samples}_samples_S1_{timestamp}.csv"
    st_filename = f"gsa_results_{method}_{n_samples}_samples_ST_{timestamp}.csv"
    s2_filename = f"gsa_results_{method}_{n_samples}_samples_S2_{timestamp}.csv"

    # Save DataFrames to CSV
    s1_df.to_csv(s1_filename, index=False)
    st_df.to_csv(st_filename, index=False)
    s2_df.to_csv(s2_filename, index=False)

    print(f"S1 results saved to {s1_filename}")
    print(f"ST results saved to {st_filename}")
    print(f"S2 results saved to {s2_filename}")

    # Return the file names
    return s1_filename, st_filename, s2_filename
def load_gsa_results(s1_filename, st_filename, s2_filename=None):
    """
    Loads S1, ST, and optionally S2 DataFrames from CSV files.

    Parameters:
        s1_filename (str): File name to load the S1 DataFrame.
        st_filename (str): File name to load the ST DataFrame.
        s2_filename (str): Optional. File name to load the S2 DataFrame.

    Returns:
        tuple: Loaded S1, ST, and optionally S2 DataFrames.
    """
    s1_df = pd.read_csv(s1_filename)
    st_df = pd.read_csv(st_filename)
    print(f"S1 results loaded from {s1_filename}")
    print(f"ST results loaded from {st_filename}")

    if s2_filename:
        s2_df = pd.read_csv(s2_filename)
        print(f"S2 results loaded from {s2_filename}")
        return s1_df, st_df, s2_df
    else:
        return s1_df, st_df
def plot_gsa_results(s1_df, st_df, s2_df=None):
    """
    Plots S1, ST, and optionally S2 DataFrames.

    Parameters:
        s1_df (pd.DataFrame): S1 DataFrame to plot.
        st_df (pd.DataFrame): ST DataFrame to plot.
        s2_df (pd.DataFrame): Optional. S2 DataFrame to plot.
    """
    # Plot S1 values
    plt.figure(figsize=(10, 6))
    plt.bar(s1_df['name'], s1_df['S1'], yerr=s1_df['S1_conf'], capsize=5, label="S1", alpha=0.7)
    plt.xticks(rotation=90)
    plt.xlabel("Parameters")
    plt.ylabel("S1 Values")
    plt.title("S1 Sensitivity Indices")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot ST values
    plt.figure(figsize=(10, 6))
    plt.bar(st_df['name'], st_df['ST'], yerr=st_df['ST_conf'], capsize=5, label="ST", alpha=0.7)
    plt.xticks(rotation=90)
    plt.xlabel("Parameters")
    plt.ylabel("ST Values")
    plt.title("ST Sensitivity Indices")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # If S2 data is available, show a heatmap of S2 values
    if s2_df is not None:
        plt.figure(figsize=(12, 8))
        s2_pivot = s2_df.pivot(index='Parameter 1', columns='Parameter 2', values='S2')
        plt.imshow(s2_pivot, cmap='viridis', aspect='auto')
        plt.colorbar(label="S2 Values")
        plt.xticks(range(len(s2_pivot.columns)), s2_pivot.columns, rotation=90)
        plt.yticks(range(len(s2_pivot.index)), s2_pivot.index)
        plt.title("S2 Sensitivity Indices Heatmap")
        plt.tight_layout()
        plt.show()
def condense_dyn_par_names(parameter_names = None):
    # Define dynamic groups for condensing
    dynamic_groups = {
        'cf_wind': 'cf_wind',
        'cf_solar': 'cf_solar',
        'el_price': 'el_price'
    }
    # Condense dynamic parameter names
    condensed_names = []
    if dynamic_groups:
        for name in parameter_names:
            for key, condensed_name in dynamic_groups.items():
                if key in name:
                    if condensed_name not in condensed_names:
                        condensed_names.append(condensed_name)
                    break
            else:
                condensed_names.append(name)
    else:
        condensed_names = parameter_names

    return condensed_names
def process_s1_st_results(sobol_results, param_names):
    """
    Processes Sobol S1 and ST results and aligns them with parameter names.

    Parameters:
        sobol_results (dict): The Sobol analysis results containing 'S1', 'S1_conf', 'ST', and 'ST_conf'.
        param_names (list): List of parameter names.

    Returns:
        pd.DataFrame, pd.DataFrame: DataFrames containing the S1 and ST results respectively.
    """
    # Extract S1, S1_conf, ST, and ST_conf
    s1_values = sobol_results['S1']
    s1_conf_values = sobol_results['S1_conf']
    st_values = sobol_results['ST']
    st_conf_values = sobol_results['ST_conf']

    # Check dimensions
    if len(param_names) != len(s1_values):
        raise ValueError("The length of param_names does not match the number of S1 or ST values.")

    # Create DataFrames
    s1_data = []
    for name, s1, s1_conf in zip(param_names, s1_values, s1_conf_values):
        s1_data.append({
            "name": name,
            "S1": s1,
            "S1_conf": s1_conf
        })

    st_data = []
    for name, st, st_conf in zip(param_names, st_values, st_conf_values):
        st_data.append({
            "name": name,
            "ST": st,
            "ST_conf": st_conf
        })

    # Convert to DataFrames
    s1_df = pd.DataFrame(s1_data)
    st_df = pd.DataFrame(st_data)

    return s1_df, st_df
def process_s2_results(sobol_results, s2_names):
    """
    Processes Sobol S2 results and aligns them with the corresponding names.

    Parameters:
        sobol_results (dict): The Sobol analysis results containing 'S2' and 'S2_conf' matrices.
        s2_names (list): List of names corresponding to S2 parameter pairs.

    Returns:
        pd.DataFrame: DataFrame containing the S2 results and confidence intervals, aligned with the names.
    """
    # Extract the S2 and S2_conf matrices from the Sobol results
    s2_matrix = sobol_results['S2']  # 26x26 matrix
    s2_conf_matrix = sobol_results['S2_conf']  # 26x26 matrix

    # Check matrix dimensions
    if s2_matrix.shape[0] != s2_matrix.shape[1]:
        raise ValueError("S2 matrix must be square.")

    if len(s2_names) != (s2_matrix.shape[0] * (s2_matrix.shape[0] - 1)) // 2:
        raise ValueError("The length of s2_names does not match the upper triangle size of the S2 matrix.")

    # Extract upper triangular indices (excluding diagonal)
    upper_indices = np.triu_indices(s2_matrix.shape[0], k=1)

    # Extract S2 and S2_conf values for the upper triangle
    s2_values = s2_matrix[upper_indices]
    s2_conf_values = s2_conf_matrix[upper_indices]

    # Align with s2_names
    s2_data = []
    for name, s2_value, s2_conf in zip(s2_names, s2_values, s2_conf_values):
        s2_data.append({
            "name": name,
            "S2": s2_value,
            "S2_conf": s2_conf
        })

    # Convert to a DataFrame
    s2_df = pd.DataFrame(s2_data)

    return s2_df
def generate_s2_names(parameter_names):
    """
    Generate names for second-order Sobol indices.

    :param parameter_names: List of parameter names.
    :return: List of parameter pair names for S2 interactions.
    """
    s2_names = []
    num_params = len(parameter_names)
    for i in range(num_params):
        for j in range(i + 1, num_params):  # Upper triangle only
            s2_names.append(f"{parameter_names[i]} x {parameter_names[j]}")
    return s2_names

def save_results(results, method, n_samples, num_of_samples, valid_outputs, num_valid_trajectories, num_levels = None):
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if num_levels == None: # Sobol
        file_name = f"gsa_results_{method}_{n_samples}_samples_{timestamp}.pkl"
    else:
        file_name = f'{method}_gsa_results_{n_samples}_trajectories_{num_levels}_grid_levels_{num_of_samples}_model_runs_{valid_outputs}_feasible_runs_{num_valid_trajectories}_number_valid_trajectories_{timestamp}.pkl'

    # Ensure the "names" key is explicitly included
    results_to_save = dict(results)
    with open(file_name, "wb") as f:
        pickle.dump(results_to_save, f)

    print(f"Results saved to {file_name}")
    return file_name
def load_results(file_name):
    """Load results from a file."""
    with open(file_name, 'rb') as f:
        results = pickle.load(f)
    print(f"Results loaded from {file_name}")
    return results
def plot_sobol_results_S1_ST(sobol_results):
    """
    Plot S1 and ST indices with their confidence intervals from the Sobol analysis results.

    :param sobol_results: SALib ResultDict object containing Sobol analysis results
    """
    # Extract parameter names, S1, ST, and their confidence intervals
    parameter_names = sobol_results['names']
    S1 = np.array(sobol_results['S1'])
    S1_conf = np.array(sobol_results['S1_conf'])
    ST = np.array(sobol_results['ST'])
    ST_conf = np.array(sobol_results['ST_conf'])

    # Create the plot
    x = np.arange(len(parameter_names))  # Index for each parameter
    width = 0.35  # Bar width

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot S1 indices and their confidence intervals
    ax.bar(x - width/2, S1, width, yerr=S1_conf, label='S1', capsize=4, color='blue')

    # Plot ST indices and their confidence intervals
    ax.bar(x + width/2, ST, width, yerr=ST_conf, label='ST', capsize=4, color='orange')

    # Add labels, title, and legend
    ax.set_xlabel('Parameters')
    ax.set_ylabel('Sensitivity Indices')
    ax.set_title('Sobol Sensitivity Analysis Results')
    ax.set_xticks(x)
    ax.set_xticklabels(parameter_names, rotation=45, ha='right')
    ax.legend()

    # Tight layout for better spacing
    plt.tight_layout()

    # Show the plot
    plt.show()
def plot_objective_distribution(objective_values):
    """
    Plots the distribution of objective values from the GSA samples.

    :param objective_values: List of objective values (one per sample).
    """
    # Convert values from €/t to €/kg by dividing by 1000
    objective_values_kg = [value / 1000 for value in objective_values]

    plt.figure(figsize=(10, 6))
    # Set bin width to 0.1 €/kg
    bins = np.arange(min(objective_values_kg), max(objective_values_kg) + 0.1, 0.1)
    plt.hist(objective_values_kg, bins=bins, color='blue', alpha=0.7)
    plt.xlabel('Objective Value: H2 price [€/kg]', fontsize=14)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Objective Values', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

def save_objectives(objective_values, method, n_samples, execution_time):
    """
    Saves the objective values array to a file.

    :param objective_values: List or NumPy array of objective values to save.
    :param method: GSA method name (e.g., 'sobol').
    :param n_samples: Number of samples used in GSA.
    :return: The name of the file where the objective values were saved.
    """
    if not isinstance(objective_values, np.ndarray):
        objective_values = np.array(objective_values)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"gsa_results_{method}_{n_samples}_samples_outputs_{execution_time:.2f}_minutes_{timestamp}.npy"
    np.save(file_name, objective_values)
    print(f"Objective values saved to {file_name}")
    return file_name
def load_objectives(file_name):
    """
    Loads the objective values array from a file.

    :param file_name: The name of the file to load the objectives from.
    :return: The loaded NumPy array of objective values.
    """
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"The file {file_name} does not exist.")

    objective_values = np.load(file_name)
    print(f"Objective values loaded from {file_name}")
    return objective_values



