# config.py
# Central configuration for model, clustering, and GSA
num_representative_weeks = 1  # Number of representative weeks for random sampling
#                                               General settings across files
time_period_length =   168*4*num_representative_weeks # One representative week
use_condensed_weeks = True
# Toggle for generating plots and tables in the model file for analysis purposes
# General flags
generate_tables = True
generate_plots = True

# Specific table flags

generate_energy_flows_table = False
generate_cost_summary_table = False
generate_installed_capacities_table = True
generate_hydrogen_storage_table = False
generate_energy_requirements_table = False
generate_cost_revenue_table = False
generate_flh_table = False
generate_utilization_and_cycles_table = True


# Specific plot flags
generate_energy_flows_plot = False
generate_cost_summary_plot = False
generate_installed_capacities_plot = False
generate_hydrogen_storage_plot = False
generate_battery_soc_plot = False
generate_energy_requirements_plot = False
generate_cost_revenue_plot = False
generate_flh_plot = False
generate_utilization_and_cycles_plot = False
plot_energy_flows_and_grid_price = False

go_quadratic = True
analyse = False
Elec_Required_Per_Ton = 39.4  # Example electricity required per ton of hydrogen
c_rate = 0.5 # for the battery
# Toggle for including minimum production constraints for the electrolyzer
include_min_prod_lvl_electrolyzer = True
include_startup_cost_electrolyzer = True
include_el_grid_connection_fee = True
startup_cost_electrolyzer = [1500, 2600]  # Adjust as needed (€)
# Minimum production bounds as a percentage of installed capacity
min_prod_lvl_bounds_electrolyzer = [0.10, 0.20]  # 10% to 20%


#                       Parameters for the create_condensed_weeks-file for the model execution:

plot_average_week = False       # Toggle to enable or disable plotting of averaged weeks
# Config option to toggle bounds overlay on plots
plot_bounds = True
plot_parameter = 'cf_wind'    # Choose 'cf_wind', 'cf_solar', or 'el_price'
representative_week_method = 'average'  # Choose 'random' or 'average'

#                                                   Global sensitivity analysis (GSA)
shutdown_after_execution = False  # Set to True to shut down the PC after execution
MIPGAP =  0.01 # 0.001 is the default
# Toggle between using static bounds from config or dynamic bounds loaded from the file
use_dynamic_bounds_from_file = True
gsa_method = 'morris'  # Choose between 'sobol' or 'morris'
# Number of samples for the GSA
n_samples = 100 # SOBOL: N=(2+D)×Nbase
# Sampling method options: 'clustering', 'block_bootstrapping', 'simple_slice'
sampling_method = 'simple_slice'  # or 'clustering', 'block_bootstrapping'
block_bootstrapping = False
num_levels_morris = 10
# Toggle for structured or random sampling in clustering
use_structured_sampling = False
n_clusters = 168  # Number of clusters (one for each hour in the week)
random_state = 42  # Random seed for reproducibility
enable_sample_analysis = False  # Set to False to disable sample analysis
factor = 1
# CAPEX values with bounds for different technologies (€/MW or €/ton or MWh)
CAPEX = {
    'capex_wind': [867*1000, 1500*1000],  # [€/MW] Bounds for wind CAPEX lb = Gutachten, ub = WEO2024
    'capex_solar': [368*1000, 690*1000],   # [€/MW]  Bounds for solar CAPEX lb = Gutachten, ub = WEO2024
    'capex_battery': [factor*402*1000, factor*798*1000],  # [€/MWh] Bounds for battery CAPEX: lb = WEO2024, ub = GUTACHTEN [165000, 665000]
    'capex_electrolyzer': [423*1000, 850*1000],  #  [€/MW] Bounds for electrolyzer CAPEX lb, ub = GUTACHTEN lb = 400 adjusted for 5.6% inflation from Fraunhofer institute
    'capex_H2_storage': [1524*1000, 5354166],  # [€/ton] derived from H2PA, cost for public HRS
}

# OPEX values with example numbers for different technologies (€/MW or €/ton)
OPEX = {
    'opex_wind': [(867*1000)*0.02, (1500*1000)*0.02],  # [€/MW] assumed to be 2% of CAPEX WEO2024
    'opex_solar': [(368*1000)*0.02, (690*1000)*0.02], # [€/MW] original value 10 $/MWh +/- 20% WEO2024
    'opex_battery': [factor*(402*1000)*0.02, (factor*798*1000)*0.02],  # [€/MWh a] OPEX are assumed to be 2% of CAPEX [165000, 665000]
    'opex_electrolyzer': [423*1000*0.035, 850*1000*0.035],  #  [€/MW a] OPEX are assumed to be 3.5% of CAPEX GUTACHTEN [16625, 29750]
    'opex_H2_storage': [(1524*1000)*0.02, 5354166*0.02],  # [€/t a] OPEX are assumed to be 2% of CAPEX [2489583.33, 5354166.67]
}
 # Fraunhofer: 400-500 €/kW

# Efficiency bounds for technologies
bounds_efficiencies = {
    'eff_battery_charge': [0.87, 0.95],  # Efficiency of battery charging
    'eff_battery_discharge': [0.87, 0.95],  # Efficiency of battery discharging
    'eff_h2_storage': [0.8, 0.9],  # Efficiency of hydrogen storage
    'eff_electrolyzer': [0.5, 0.83],  # Efficiency of electrolyzer
}

# Maximum capacities for technologies
max_capacities = {
    'wind': 500,
    'solar': 350,
    'battery': 350,
    'electrolyzer': 350,
    'H2_storage': 200,
}

# New bounds for max capacities
bounds_max_capacities = {
    'wind': [400, 500],
    'solar': [250, 350],
    'battery': [250, 350],
    'electrolyzer': [250, 350],
    'H2_storage': [150, 250]  # Max capacity for H2 storage
}

# Lifetimes for different technologies (years)
lifetimes = {
    'wind': 25,
    'solar': 25,
    'battery': 20,
    'electrolyzer': 15,
    'H2_storage': 30,
    'gas_grid_connection': 40,
    'el_grid_connection': 30
}

# New bounds for lifetimes
bounds_lifetimes = {
    'wind': [20, 30],         # Wind Turbine Lifetime
    'solar': [20, 30],        # Solar PV Lifetime
    'electrolyzer': [10, 20], # Electrolyzer Lifetime
    'battery': [10, 20],  # Battery Lifetime
    'H2_storage': [25, 35],   # Hydrogen Storage Lifetime
    'el_grid_connection': [25, 40]  # Grid Connection Lifetime
}


# Other parameters including hydrogen demand and economic parameters
other_parameters = {
    'total_h2_demand': [18000, 22000],  #[t H2]  Total hydrogen demand
    'interest_rate': [0.04, 0.09],  # Example bounds for interest rate
    'el_grid_connection_fee': [32*1000, 196 * 1000],  # Example connection fee (€/ MW)
    'usage_fee': [240, 709],  # Example usage fee (€/ton)
    'gas_grid_connection_fee': [393900, 1181700], # in [€/ton H2]
    'heat_price': [100, 130], # €/t
    'water_price': [0.00176, 0.00264],  # €/l, ±20% bounds
    'water_demand': [12000, 17000],  # l/tons
    'o2_price': [0.032*1000, 0.048*1000]  # 0.04 €/kg H2, ±20% bounds

}
# Dynamic parameter bounds for each hour (e.g., capacity factors for wind, solar, and electricity price)
bounds_dynamic_parameters = {
    'cf_wind': [0, 1],  # Wind capacity factor bounds (already between 0 and 1)
    'cf_solar': [0, 0.748878],  # Solar capacity factor bounds (already between 0 and 1)
    #'el_price': [-90.01, 121.46]  # Electricity price bounds (standardized between -1 and 1)
    'el_price': [-50, 50]
}
