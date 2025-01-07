import numpy as np
import os
import platform
from SALib.sample import saltelli, morris as morris_sampler
from SALib.analyze import sobol, morris
from SALib.util import ResultDict
import config
import time
from model_morris import run_model  # Import the model function you defined
import matplotlib.pyplot as plt
import pandas as pd
from gsa_helper_functions import (
    plot_objective_distribution,
    plot_sobol_results_S1_ST,
    condense_dyn_par_names,
    save_gsa_results,
    load_gsa_results,
    save_results,
    load_results,
    save_objectives,
    load_objectives,
    plot_gsa_results,
    process_s1_st_results,
    process_s2_results,
    generate_s2_names

)

# Start the timer
start_time = time.time()

# Combine global parameters from config (CAPEX, OPEX, efficiencies, max, capacities and other parameters)
CAPEX = config.CAPEX
OPEX = config.OPEX
Lifetimes = config.bounds_lifetimes
Max_capacities = config.bounds_max_capacities
bounds_efficiencies = config.bounds_efficiencies
other_parameters = config.other_parameters
# Check if the min_prod_lvl constraint is included
if config.include_min_prod_lvl_electrolyzer:
    if 'min_prod_lvl_electrolyzer' not in config.other_parameters:
            # Define the bounds for the minimum production level
            config.other_parameters['min_prod_lvl_electrolyzer'] = [
                config.min_prod_lvl_bounds_electrolyzer[0],  # Minimum bound
                config.min_prod_lvl_bounds_electrolyzer[1]   # Maximum bound
            ]
if config.include_startup_cost_electrolyzer:
    if 'startup_cost_electrolyzer' not in config.other_parameters:
            # Define the bounds for the start up cost
            config.other_parameters['startup_cost_electrolyzer'] = [
                config.startup_cost_electrolyzer[0],  # Minimum bound
                config.startup_cost_electrolyzer[1]   # Maximum bound
            ]

bounds_dynamic_parameters = config.bounds_dynamic_parameters

# Step 1: Define global parameter names, bounds, and assign each its own unique "group" label
global_param_names = (
        list(CAPEX.keys()) +
        list(OPEX.keys()) +
        list(bounds_efficiencies.keys()) +
        list(other_parameters.keys())
)

global_bounds = (
        list(CAPEX.values()) +
        list(OPEX.values()) +
        list(bounds_efficiencies.values()) +
        list(other_parameters.values())
)

# Adding lifetimes to global parameters
global_param_names += [f'lifetime_{key}' for key in Lifetimes.keys()]
global_bounds += list(Lifetimes.values())

# Adding max capacities to global parameters
global_param_names += [f'max_capacity_{key}' for key in Max_capacities.keys()]
global_bounds += list(Max_capacities.values())


# Assign each global parameter its own unique "group" label, using the parameter's name as its group (unless its lifetimes)
# Create groups for global parameters
# Group OPEX costs together into a single group
# Group Lifteimes together into a single group
global_groups = []
for name in global_param_names:
    if name.startswith("opex_"):
        global_groups.append("OPEX")  # Group all OPEX costs together
    elif name.startswith("lifetime_"):
        global_groups.append("lifetimes")  # Group all lifetimes together
    else:
        global_groups.append(name)  # Keep other parameters as individual groups


# Step 2: Define dynamic parameter names, bounds, and groups
dynamic_param_names = []
dynamic_bounds = []
dynamic_groups_flat = []
if config.use_dynamic_bounds_from_file:
    # Load dynamic bounds from file
    bounds_file = "multi_index_bounds.csv"  # Ensure this file exists and matches the structure
    try:
        bounds_df = pd.read_csv(bounds_file, header=[0, 1], index_col=0)
        print(f"Bounds File Shape: {bounds_df.shape}")
        print(f"Bounds File Columns: {bounds_df.columns}")

        # Extract bounds dynamically from the file
        for param in ['cf_wind', 'cf_solar', 'el_price']:
            dynamic_param_names.extend([f'{param}_{i}' for i in range(len(bounds_df))])
            dynamic_bounds.extend([
                (bounds_df[(param, 'min')][i], bounds_df[(param, 'max')][i]) for i in range(len(bounds_df))
            ])
            dynamic_groups_flat.extend([param] * len(bounds_df))

        # Fix zero-range bounds for dynamic parameters
        dynamic_bounds = [
            (lower, upper if upper > lower else lower + 1e-6)  # Add small range if bounds are zero
            for lower, upper in dynamic_bounds
        ]

    except Exception as e:
        print(f"Error loading bounds from file: {e}")
        exit(1)

else:
    # Use static bounds from config
    for group_name, param_bound in bounds_dynamic_parameters.items():
        dynamic_param_names.extend([f'{group_name}_{i}' for i in range(config.time_period_length)])
        dynamic_bounds.extend([param_bound] * config.time_period_length)
        dynamic_groups_flat.extend([group_name] * config.time_period_length)  # Group each dynamic parameter by its name

# Convert bounds to lists (if they are not already lists)
global_bounds = [list(bound) for bound in global_bounds]  # Ensure all bounds are lists
dynamic_bounds = [list(bound) for bound in dynamic_bounds]  # Convert dynamic bounds to lists

# Step 3: Define the complete problem configuration for SALib
problem = {
    'num_vars': len(global_param_names) + len(dynamic_param_names),
    'names': global_param_names + dynamic_param_names,
    'bounds': global_bounds + dynamic_bounds,
    'groups': global_groups + dynamic_groups_flat  # Assign each global param its own group and dynamic params as labeled groups
}

# Count unique groups
unique_groups = set(global_groups + dynamic_groups_flat)  # Extract unique group names
num_unique_groups = len(unique_groups)  # Count them
print("Unique Groups:", unique_groups)
print("Number of Unique Groups:", num_unique_groups)

# After defining the problem dictionary
print("Parameter names in the GSA problem setup:")
print(len((problem['names'])))
print(len(global_param_names))
print(problem['names'])

print("\nParameter bounds in the GSA problem setup:")
print(problem['bounds'])

print("\nParameter groups in the GSA problem setup:")
print(problem['groups'])



# Generate samples for global and dynamic parameters based on the selected GSA method (in config.py)
if config.gsa_method == 'sobol':
    param_values = saltelli.sample(problem, config.n_samples)
elif config.gsa_method == 'morris':
    param_values = morris_sampler.sample(problem, config.n_samples, num_levels=config.num_levels_morris)


def map_global_params(sample):
    """Map the sampled global parameters to the structure expected by the model."""
    mapped_params = {}
    try:
        # Mapping CAPEX values
        mapped_params['capex_wind'] = sample[global_param_names.index('capex_wind')]
        mapped_params['capex_solar'] = sample[global_param_names.index('capex_solar')]
        mapped_params['capex_battery'] = sample[global_param_names.index('capex_battery')]
        mapped_params['capex_electrolyzer'] = sample[global_param_names.index('capex_electrolyzer')]
        mapped_params['capex_H2_storage'] = sample[global_param_names.index('capex_H2_storage')]


        # Mapping OPEX values
        mapped_params['opex_wind'] = sample[global_param_names.index('opex_wind')]
        mapped_params['opex_solar'] = sample[global_param_names.index('opex_solar')]
        mapped_params['opex_battery'] = sample[global_param_names.index('opex_battery')]
        mapped_params['opex_electrolyzer'] = sample[global_param_names.index('opex_electrolyzer')]
        mapped_params['opex_H2_storage'] = sample[global_param_names.index('opex_H2_storage')]

        # Mapping Efficiency values
        mapped_params['eff_battery_charge'] = sample[global_param_names.index('eff_battery_charge')]
        mapped_params['eff_battery_discharge'] = sample[global_param_names.index('eff_battery_discharge')]
        mapped_params['eff_h2_storage'] = sample[global_param_names.index('eff_h2_storage')]
        mapped_params['eff_electrolyzer'] = sample[global_param_names.index('eff_electrolyzer')]

        # Map lifetimes
        mapped_params['lifetime_wind'] = sample[global_param_names.index('lifetime_wind')]
        mapped_params['lifetime_solar'] = sample[global_param_names.index('lifetime_solar')]
        mapped_params['lifetime_electrolyzer'] = sample[global_param_names.index('lifetime_electrolyzer')]
        mapped_params['lifetime_battery'] = sample[global_param_names.index('lifetime_battery')]
        mapped_params['lifetime_H2_storage'] = sample[global_param_names.index('lifetime_H2_storage')]
        mapped_params['lifetime_el_grid_connection'] = sample[global_param_names.index('lifetime_el_grid_connection')]

        # Map max capacities
        mapped_params['max_capacity_wind'] = sample[global_param_names.index('max_capacity_wind')]
        mapped_params['max_capacity_solar'] = sample[global_param_names.index('max_capacity_solar')]
        mapped_params['max_capacity_electrolyzer'] = sample[global_param_names.index('max_capacity_electrolyzer')]
        mapped_params['max_capacity_battery'] = sample[
            global_param_names.index('max_capacity_battery')]
        mapped_params['max_capacity_H2_storage'] = sample[global_param_names.index('max_capacity_H2_storage')]

        # Mapping Other parameters
        mapped_params['total_h2_demand'] = sample[global_param_names.index('total_h2_demand')]
        mapped_params['interest_rate'] = sample[global_param_names.index('interest_rate')]
        mapped_params['usage_fee'] = sample[global_param_names.index('usage_fee')]
        mapped_params['heat_price'] = sample[global_param_names.index('heat_price')]
        mapped_params['el_grid_connection_fee'] = sample[global_param_names.index('el_grid_connection_fee')]
        mapped_params['gas_grid_connection_fee'] = sample[global_param_names.index('gas_grid_connection_fee')]
        mapped_params['o2_price'] = sample[global_param_names.index('o2_price')]
        mapped_params['water_demand'] = sample[global_param_names.index('water_demand')]
        mapped_params['water_price'] = sample[global_param_names.index('water_price')]

        # Map minimum production level if included
        if config.include_min_prod_lvl_electrolyzer:
            mapped_params['min_prod_lvl_electrolyzer'] = sample[global_param_names.index('min_prod_lvl_electrolyzer')]

            # Map minimum production level if included
        if config.include_startup_cost_electrolyzer:
            mapped_params['startup_cost_electrolyzer'] = sample[global_param_names.index('startup_cost_electrolyzer')]
            # Map minimum production level if included
            if config.include_el_grid_connection_fee:
                mapped_params['el_grid_connection_fee'] = sample[global_param_names.index('el_grid_connection_fee')]

    except ValueError as e:
        print("Error in `map_global_params`: Missing parameter in global_param_names list.")
        print("Exception:", e)
        print("Sample causing error:", sample)
        raise

    return mapped_params

if config.enable_sample_analysis:
    # Choose the sample index you want to examine
    sample_index = 5  # Adjust this to view other samples

    # Retrieve the global and dynamic parameters for the sample
    example_params = param_values[sample_index]
    global_params_example = map_global_params(example_params)

    # Group dynamic parameters by their main names (e.g., 'cf_wind', 'cf_solar', 'el_price')
    dynamic_param_sample_example = {
        'cf_wind': [example_params[len(global_param_names) + i] for i in range(config.time_period_length)],
        'cf_solar': [example_params[len(global_param_names) + config.time_period_length + i] for i in range(config.time_period_length)],
        'el_price': [example_params[len(global_param_names) + 2 * config.time_period_length + i] for i in range(config.time_period_length)]
    }

    # Print the sample values for inspection
    print(f"\n--- Example Parameter Values for Sample {sample_index} ---")
    print("Global Parameters:")
    for k, v in global_params_example.items():
        print(f"{k}: {v}")
    print("\nDynamic Parameters:")
    for k, v in dynamic_param_sample_example.items():
        print(f"{k}: {v}")

    # Run model for the example sample and display the result
    example_output = run_model(global_params_example, dynamic_param_sample_example)
    print(f"\nModel Output for Sample {sample_index}: {example_output}")

    # Plotting the example sample data
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot cf_wind and cf_solar on the primary y-axis
    ax1.plot(dynamic_param_sample_example['cf_wind'], label='cf_wind', color='blue', linewidth=1)
    ax1.plot(dynamic_param_sample_example['cf_solar'], label='cf_solar', color='green', linewidth=1)
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Capacity Factor', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # Create a secondary y-axis for el_price
    ax2 = ax1.twinx()
    ax2.plot(dynamic_param_sample_example['el_price'], label='el_price', color='red', linestyle='--', linewidth=1)
    ax2.set_ylabel('Electricity Price', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Adding legends and showing plot
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title(f'Dynamic Parameter Values for Sample {sample_index}')
    plt.show()

#
# # Initialize lists to store valid parameter sets and model outputs
# valid_params = []
# valid_outputs = []
#
# # Run the model for each sample
# for i, params in enumerate(param_values):
#     try:
#         # Map the global parameters
#         global_params = map_global_params(params)
#
#     except Exception as e:
#         print(f"Error mapping global parameters for sample {i}")
#         continue
#
#     # Group dynamic parameters by each parameter type
#     dynamic_param_sample = {}
#     try:
#         for param in bounds_dynamic_parameters.keys():
#             dynamic_param_sample[param] = [
#                 params[len(global_param_names) + dynamic_param_names.index(f"{param}_{j}")]
#                 for j in range(config.time_period_length)
#             ]
#     except Exception as e:
#         print(f"Error mapping dynamic parameters for sample {i}: {e}")
#         continue
#
#     # Run the model and collect output
#     try:
#         output = run_model(global_params, dynamic_param_sample)
#         if output is None:
#             print(f"Model infeasible for sample {i}")
#             continue  # Skip infeasible sample
#         valid_params.append(params)  # Add valid parameter sample
#         valid_outputs.append(output)  # Add valid model output
#     except Exception as e:
#         print(f"Error running model for sample {i}: {e}")
#         continue
#
# # Convert valid parameters and outputs to NumPy arrays
# valid_params = np.array(valid_params)
# valid_outputs = np.array(valid_outputs)









# Initialize lists for valid parameter sets and outputs
valid_params = []
valid_outputs = []

# Calculate trajectory size
trajectory_size = len(set(problem["groups"])) + 1  # Number of groups + 1
num_trajectories = len(param_values) // trajectory_size

# Iterate over trajectories
for traj_idx in range(num_trajectories):
    start_idx = traj_idx * trajectory_size
    end_idx = start_idx + trajectory_size

    # Extract the trajectory
    trajectory = param_values[start_idx:end_idx]
    trajectory_outputs = []
    is_valid_trajectory = True  # Assume the trajectory is valid

    # Check each sample in the trajectory
    for params in trajectory:
        try:
            # Map parameters and run the model
            global_params = map_global_params(params)
            dynamic_param_sample = {
                param: [
                    params[len(global_param_names) + dynamic_param_names.index(f"{param}_{j}")]
                    for j in range(config.time_period_length)
                ]
                for param in bounds_dynamic_parameters.keys()
            }

            # Run the model
            output = run_model(global_params, dynamic_param_sample)
            if output is None:
                is_valid_trajectory = False  # Invalidate trajectory
                break

            trajectory_outputs.append(output)
        except Exception as e:
            print(f"Error in trajectory {traj_idx}: {e}")
            is_valid_trajectory = False  # Invalidate trajectory
            break

    # If the trajectory is valid, add it to the results
    if is_valid_trajectory:
        valid_params.extend(trajectory)
        valid_outputs.extend(trajectory_outputs)

# Convert valid data to NumPy arrays
valid_params = np.array(valid_params)
valid_outputs = np.array(valid_outputs)

# Ensure the number of valid outputs matches the expected structure
num_valid_trajectories = len(valid_outputs) // trajectory_size
print(f"Number of valid trajectories: {num_valid_trajectories}")

# Summary
print(f"Number of original parameter samples: {len(param_values)}")
print(f"Number of successful model outputs: {len(valid_outputs)}")

sample_count = len(param_values)


# Check if there are enough feasible samples
if len(valid_outputs) == 0:
    print("No feasible model outputs found. Sensitivity analysis cannot be performed.")
else:

        morris_results = morris.analyze(problem, valid_params, valid_outputs, print_to_console=True)
        #morris_results.plot()
        file_name = save_results(morris_results, method=config.gsa_method, n_samples=config.n_samples, num_of_samples=len(param_values), valid_outputs=len(valid_outputs), num_valid_trajectories=num_valid_trajectories,num_levels=config.num_levels_morris)
        morris = ResultDict(load_results(file_name))
        df = morris.to_df()
        # Assuming your DataFrame is named 'df'
        df_sorted = df.sort_values(by='mu_star', ascending=False)
        # Round all values in the DataFrame to the closest integer
        df_rounded = df_sorted.round(0)
        morris.plot()

# Stop the timer after the analysis completes
end_time = time.time()

# Calculate the elapsed time
execution_time = (end_time - start_time)/60

# Print or log the execution time
print(f"Execution time for the GSA analysis: {execution_time:.2f} minutes")

# Plot the distribution of objective values
min_objective_values = [(output) for output in valid_outputs]
objective_file = save_objectives(min_objective_values, config.gsa_method, config.n_samples,execution_time)
objectives = load_objectives(objective_file)
plot_objective_distribution(objectives)



# Check the shutdown flag
if config.shutdown_after_execution:
    print("Shutting down the system as per configuration...")
    if platform.system() == "Windows":
        os.system("shutdown /s /t 1")  # Shutdown command for Windows
    else:
        print("Shutdown not supported on this operating system.")

