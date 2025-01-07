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
file_name = 'morris_gsa_results_140_trajectories_12_grid_levels_4340_model_runs_4247_feasible_runs_137_number_valid_trajectories_2024-12-28_16-34-29.pkl'
objective_file = 'gsa_results_morris_140_samples_outputs_2024-12-28_16-34-30.npy'
# file_name = 'morris_gsa_results_100_trajectories_10_grid_levels_3100_model_runs_3038_feasible_runs_98_number_valid_trajectories_2025-01-05_10-34-03.pkl'
# objective_file = 'gsa_results_morris_100_samples_outputs_246.66_minutes_2025-01-05_10-34-04.npy'
morris = ResultDict(load_results(file_name))
df = morris.to_df()
# Divide values by 1000 to get €/kg
df = df / 1000
# Assuming your DataFrame is named 'df'
df_sorted = df.sort_values(by='mu_star', ascending=False)
# Round all values in the DataFrame to the closest integer
df_rounded = df_sorted.round(0)
# Display the sorted DataFrame
print(df_rounded)
# Plotting
# Remove 'gas_grid_connection_fee' from the dataframe

df = df_sorted.drop('gas_grid_connection_fee')



# Plotting with centered error bars and LaTeX formatting for both mu_star and sigma
plt.figure(figsize=(15, 8))
bar_width = 0.35
x = np.arange(len(df.index))

# Bar plot for mu_star and sigma only
plt.bar(x - bar_width / 2, df['mu_star'], width=bar_width, label=r'$\mu^*$')
plt.bar(x + bar_width / 2, df['sigma'], width=bar_width, label=r'$\sigma$')

# Add error bars for mu_star (centered over mu_star bars)
plt.errorbar(x - bar_width / 2, df['mu_star'], yerr=df['mu_star_conf'], fmt='none', ecolor='black', capsize=3)

# Labeling and formatting
plt.xlabel('Parameters', fontsize=14)
plt.ylabel('Value (€/kg)', fontsize=14)
plt.title('Morris Method Sensitivity Analysis Results (€/kg)', fontsize=16)
plt.xticks(x, df.index, rotation=90, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)

plt.tight_layout()
plt.show()

# Convert data to €/kg by dividing by 1000
data = load_objectives(objective_file)
data = data / 1000



# Define bins of size 0.1 €/kg (since data is now scaled)

bins = np.arange(min(data), max(data) + 0.1, 0.1)



# Calculate histogram

hist, bin_edges = np.histogram(data, bins=bins)



# Find the bin with the highest frequency

max_bin_index = np.argmax(hist)

most_frequent_bin = (bin_edges[max_bin_index], bin_edges[max_bin_index + 1])

most_frequent_count = hist[max_bin_index]



# Plot histogram

plt.figure(figsize=(10, 6))

plt.bar(bin_edges[:-1], hist, width=0.1, edgecolor='black', align='edge')

plt.xlabel('Hydrogen selling price (€/kg)', fontsize=14)

plt.ylabel('Frequency', fontsize=14)

plt.title('Histogram of Data (€/kg) - Bins = 0.1 €/kg', fontsize=16)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.grid(True)

plt.show()



# Display results

result = {

    "Most Frequent Bin (€/kg)": most_frequent_bin,

    "Frequency in Bin": most_frequent_count

}


