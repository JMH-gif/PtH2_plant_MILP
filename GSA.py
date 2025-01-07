import numpy as np
from SALib.sample import saltelli, morris as morris_sampler
from SALib.analyze import sobol, morris
import config
from model import run_model  # Import the model function you defined
import clustering  # Import the clustering and data sampling methods

# Combine global parameters from config (CAPEX, OPEX, efficiencies, and other parameters)
CAPEX = config.CAPEX
OPEX = config.OPEX
bounds_efficiencies = config.bounds_efficiencies
other_parameters = config.other_parameters
bounds_dynamic_parameters = config.bounds_dynamic_parameters

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

# Dynamic parameter names and bounds (from bounds_dynamic_parameters in config.py)
dynamic_param_names = []
dynamic_bounds = []
dynamic_groups_flat = []

group_counter = len(global_param_names)
for param in bounds_dynamic_parameters.keys():
    dynamic_param_names.extend([f'{param}_{i}' for i in range(config.time_period_length)])
    dynamic_bounds.extend([bounds_dynamic_parameters[param]] * config.time_period_length)
    dynamic_groups_flat.extend([group_counter] * config.time_period_length)
    group_counter += 1

# Define the problem for SALIB
problem = {
    'num_vars': len(global_param_names) + len(dynamic_param_names),
    'names': global_param_names + dynamic_param_names,
    'bounds': global_bounds + dynamic_bounds,
    'groups': ([None] * len(global_param_names)) + dynamic_groups_flat
}

# Generate samples for the global parameters based on the selected GSA method (in config.py)
if config.gsa_method == 'sobol':
    param_values = saltelli.sample(problem, config.n_samples)
elif config.gsa_method == 'morris':
    param_values = morris_sampler.sample(problem, config.n_samples)


def sample_time_varying_params(method="clustering"):
    """Sample dynamic parameters based on the selected method in the configuration."""
    sampled_params = []

    if method == "clustering":
        cluster_data, scaler = clustering.get_clustered_data()
        for i, sample in enumerate(param_values):
            time_varying_sample = clustering.sample_time_varying_params(sample, cluster_data,
                                                                        config.use_structured_sampling, scaler)
            sampled_params.append(time_varying_sample)
            # Debugging print statement
            print(f"Sample {i}: Dynamic Parameters - {time_varying_sample}")

    elif method == "block_bootstrapping":
        block_size = config.block_size
        for i in range(len(param_values)):
            bootstrapped_sample = clustering.get_bootstrapped_data(block_size, config.time_period_length)
            sampled_params.append(bootstrapped_sample)
            # Debugging print statement
            print(f"Sample {i}: Bootstrapped Parameters - {bootstrapped_sample}")

    elif method == "simple_slice":
        for i in range(len(param_values)):
            simple_slice_sample = clustering.get_simple_slice_data(config.time_period_length)
            sampled_params.append(simple_slice_sample)
            # Debugging print statement
            print(f"Sample {i}: Sliced Parameters - {simple_slice_sample}")
            print(f"el_price sample: {simple_slice_sample['el_price']}")
            print(f"cf_wind sample: {simple_slice_sample['cf_wind']}")
            print(f"cf_solar sample: {simple_slice_sample['cf_solar']}")

    return sampled_params


# Sample the time-varying parameters based on the selected method
time_varying_samples = sample_time_varying_params(method=config.sampling_method)


def map_global_params(sample):
    """Map the sampled global parameters to the structure expected by the model."""
    return {
        # CAPEX values
        'capex_wind': sample[global_param_names.index('capex_wind')],
        'capex_solar': sample[global_param_names.index('capex_solar')],
        'capex_battery': sample[global_param_names.index('capex_battery')],
        'capex_electrolyzer': sample[global_param_names.index('capex_electrolyzer')],
        'capex_h2_storage': sample[global_param_names.index('capex_h2_storage')],

        # OPEX values
        'opex_wind': sample[global_param_names.index('opex_wind')],
        'opex_solar': sample[global_param_names.index('opex_solar')],
        'opex_battery': sample[global_param_names.index('opex_battery')],
        'opex_electrolyzer': sample[global_param_names.index('opex_electrolyzer')],
        'opex_h2_storage': sample[global_param_names.index('opex_h2_storage')],

        # Efficiency values
        'eff_battery_charge': sample[global_param_names.index('eff_battery_charge')],
        'eff_battery_discharge': sample[global_param_names.index('eff_battery_discharge')],
        'eff_h2_storage': sample[global_param_names.index('eff_h2_storage')],
        'eff_electrolyzer': sample[global_param_names.index('eff_electrolyzer')],

        # Other parameters
        'total_h2_demand': sample[global_param_names.index('total_h2_demand')],
        'interest_rate': sample[global_param_names.index('interest_rate')],
        'usage_fee': sample[global_param_names.index('usage_fee')],
        'connection_fee': sample[global_param_names.index('connection_fee')],
    }


# Run the model for each sample
model_outputs = []
for i, params in enumerate(param_values):
    global_params = map_global_params(params)
    dynamic_param_sample = time_varying_samples[i]
    output = run_model(global_params, dynamic_param_sample)
    if output is None:
        print(f"Model infeasible for sample {i}")
        continue
    model_outputs.append(output)
print(f"Number of parameter samples: {len(param_values)}")
print(f"Number of model outputs: {len(model_outputs)}")
# Check if there are enough feasible samples
if len(model_outputs) == 0:
    print("No feasible model outputs found. Sensitivity analysis cannot be performed.")
else:
    # Perform sensitivity analysis using the selected method
    if config.gsa_method == 'sobol':
        sobol_results = sobol.analyze(problem, np.array(model_outputs), print_to_console=True)
    elif config.gsa_method == 'morris':
        morris_results = morris.analyze(problem, param_values, np.array(model_outputs), print_to_console=True)

print(f"Number of parameter samples: {len(param_values)}")
print(f"Number of model outputs: {len(model_outputs)}")