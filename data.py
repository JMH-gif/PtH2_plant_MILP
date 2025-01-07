import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set Seaborn style for all plots
sns.set_theme(style="darkgrid")


# Load the dataset without a date column
def load_data(file_path, start_date="2010-01-01"):
    try:
        # Load data without specifying a date column
        df = pd.read_excel(file_path)

        # Generate a datetime index assuming hourly data starting from `start_date`
        df.index = pd.date_range(start=start_date, periods=len(df), freq='h')

        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


# Calculate descriptive statistics
def calculate_descriptive_statistics(df):
    descriptive_stats = df.describe()
    return descriptive_stats


# Plotting functions for hourly data
def plot_hourly_data(df):
    fig, axs = plt.subplots(3, 1, figsize=(12, 18))

    # Plot Electricity Price
    axs[0].plot(df.index, df['el_price'], label='Electricity Price')
    axs[0].set_title('Hourly Electricity Price')
    axs[0].set_xlabel('Date')
    axs[0].set_ylabel('Price (EUR/MWh)')
    axs[0].legend()

    # Plot Wind Capacity Factor
    axs[1].plot(df.index, df['cf_wind'], label='Wind Capacity Factor', color='blue')
    axs[1].set_title('Hourly Wind Capacity Factor')
    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('Capacity Factor')
    axs[1].legend()

    # Plot Solar Capacity Factor
    axs[2].plot(df.index, df['cf_solar'], label='Solar Capacity Factor', color='orange')
    axs[2].set_title('Hourly Solar Capacity Factor')
    axs[2].set_xlabel('Date')
    axs[2].set_ylabel('Capacity Factor')
    axs[2].legend()

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.5)
    plt.show()


# Plotting descriptive statistics for el_price and capacity factors separately, with standard deviation included
def plot_descriptive_statistics(descriptive_stats):
    # Plot for Electricity Price with Standard Deviation
    fig, ax = plt.subplots(figsize=(10, 6))
    el_price_stats = descriptive_stats['el_price'][['mean', 'std', '50%', 'min', 'max', '25%', '75%']]
    el_price_stats.index = ['Mean', 'Std Dev', 'Median', 'Min', 'Max', '25th Percentile', '75th Percentile']
    el_price_stats.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title('Descriptive Statistics for Electricity Price')
    ax.set_ylabel('Value (EUR/MWh)')
    ax.set_xticklabels(el_price_stats.index, rotation=0)  # Set labels to horizontal
    plt.show()

    # Plot for Wind and Solar Capacity Factors with Standard Deviation
    fig, ax = plt.subplots(figsize=(10, 6))
    cf_wind_stats = descriptive_stats['cf_wind'][['mean', 'std', '50%', 'min', 'max', '25%', '75%']]
    cf_solar_stats = descriptive_stats['cf_solar'][['mean', 'std', '50%', 'min', 'max', '25%', '75%']]
    cf_stats = pd.DataFrame({
        'Wind Capacity Factor': cf_wind_stats,
        'Solar Capacity Factor': cf_solar_stats
    })
    cf_stats.index = ['Mean', 'Std Dev', 'Median', 'Min', 'Max', '25th Percentile', '75th Percentile']

    cf_stats.plot(kind='bar', ax=ax)
    ax.set_title('Descriptive Statistics for Wind and Solar Capacity Factors')
    ax.set_ylabel('Capacity Factor')
    ax.set_xticklabels(cf_stats.index, rotation=0)  # Set labels to horizontal
    ax.legend(loc='upper left')
    plt.show()


# Main function to execute the workflow
def main():
    file_path = r"C:\Users\Mika\Desktop\Master\Data/Hourly_el_price_cf_wind_cf_solar_1.xlsx"
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist in the current directory.")
        return

    # Load data
    df = load_data(file_path, start_date="2010-01-01")
    if df is None:
        return

    # Calculate and display descriptive statistics
    descriptive_stats = calculate_descriptive_statistics(df)
    print("Descriptive Statistics:")
    print(descriptive_stats)

    # Plot hourly data
    plot_hourly_data(df)

    # Plot descriptive statistics for Electricity Price and Capacity Factors separately
    plot_descriptive_statistics(descriptive_stats)




# Execute main function
if __name__ == '__main__':
    main()
    #                                   Unit COMMITMENT CONSTRAINTS  FOR BATTERY AND ELECTROLYZER

    # Comment out to check if this constraint is causing infeasibility
    # model.addConstrs(
    #     (u['electrolyzer', t] * P_min['electrolyzer'] <= x_out['electrolyzer', t] for t in T),
    #     name="ElectrolyzerMinOperation"
    # )

    # # Ensure electrolyzer operates within installed capacity when committed
    # model.addConstrs(
    #     (x_out['electrolyzer', t] <= u['electrolyzer', t] * P_invest['electrolyzer'] for t in T),
    #     name="ElectrolyzerMaxOperation"
    # )














    # # Load general configuration from config file
    # time_period_length = config.time_period_length
    # hours_per_year = 8760
    # scaling_factor = hours_per_year / time_period_length
    # T = list(range(time_period_length))
    # # Initialize CAPEX and OPEX dictionaries
    # CAPEX = {}
    # OPEX = {}
    #
    # # If standalone execution, calculate average values between lower and upper bounds from config
    # if global_params is None:
    #     CAPEX = {
    #         'wind': 1190000, # From Gutachten SA
    #         'solar': 455000, # From Gutachten SA
    #         'battery': 665000, # From Gutachten SA
    #         'electrolyzer': (config.CAPEX['capex_electrolyzer'][0] + config.CAPEX['capex_electrolyzer'][1]) / 2,
    #         'H2_storage': (config.CAPEX['capex_h2_storage'][0] + config.CAPEX['capex_h2_storage'][1]) / 2
    #     }
    #
    #     OPEX = {
    #         'wind': (config.OPEX['opex_wind'][0] + config.OPEX['opex_wind'][1]) / 2,
    #         'solar': (config.OPEX['opex_solar'][0] + config.OPEX['opex_solar'][1]) / 2,
    #         'battery': (config.OPEX['opex_battery'][0] + config.OPEX['opex_battery'][1]) / 2,
    #         'electrolyzer': (config.OPEX['opex_electrolyzer'][0] + config.OPEX['opex_electrolyzer'][1]) / 2,
    #         'H2_storage': (config.OPEX['opex_h2_storage'][0] + config.OPEX['opex_h2_storage'][1]) / 2
    #     }
    #
    #     Eff = {
    #         'battery': {
    #             'charge': (config.bounds_efficiencies['eff_battery_charge'][0] +
    #                        config.bounds_efficiencies['eff_battery_charge'][1]) / 2,
    #             'discharge': (config.bounds_efficiencies['eff_battery_discharge'][0] +
    #                           config.bounds_efficiencies['eff_battery_discharge'][1]) / 2
    #         },
    #         'H2_storage': {
    #             'storage': (config.bounds_efficiencies['eff_h2_storage'][0] +
    #                         config.bounds_efficiencies['eff_h2_storage'][1]) / 2,
    #             'discharge': (config.bounds_efficiencies['eff_h2_storage'][0] +
    #                           config.bounds_efficiencies['eff_h2_storage'][1]) / 2
    #         }
    #     }
    #
    #     Eff_electrolyzer = (config.bounds_efficiencies['eff_electrolyzer'][0] +
    #                         config.bounds_efficiencies['eff_electrolyzer'][1]) / 2
    #
    #     if include_min_prod_lvl:
    #         min_prod_lvl_electrolyzer = (config.min_prod_lvl_bounds_electrolyzer[0] + config.min_prod_lvl_bounds_electrolyzer[1]) / 2
    #
    #     water_price = (config.other_parameters['water_price'][0] + config.other_parameters['water_price'][1]) / 2
    #     water_demand = (config.other_parameters['water_demand'][0] + config.other_parameters['water_demand'][1]) / 2
    #     o2_price =  (config.other_parameters['o2_price'][0] + config.other_parameters['o2_price'][1]) / 2
    #     # Hydrogen demand calculation
    #     total_hydrogen_demand = (config.other_parameters['total_h2_demand'][0] +
    #                              config.other_parameters['total_h2_demand'][1]) / 2
    #     hourly_hydrogen_demand = total_hydrogen_demand / hours_per_year
    #
    #
    #
    #     # Transportation cost parameters
    #     el_grid_connection_fee = (config.other_parameters['el_grid_connection_fee'][0] + config.other_parameters['el_grid_connection_fee'][1])/2
    #     usage_fee = (config.other_parameters['usage_fee'][0] + config.other_parameters['usage_fee'][1]) / 2
    #     gas_grid_connection_fee = (config.other_parameters['gas_grid_connection_fee'][0]+ config.other_parameters['gas_grid_connection_fee'][1])/2
    #     Interest_rate = (config.other_parameters['interest_rate'][0] + config.other_parameters['interest_rate'][1]) / 2
    #     heat_price = (config.other_parameters['heat_price'][0] + config.other_parameters['heat_price'][1]) / 2
    #
    # # If GSA assigns data from GSA sample
    # else:
    #     CAPEX = {
    #         'wind': global_params['capex_wind'],
    #         'solar': global_params['capex_solar'],
    #         'battery': global_params['capex_battery'],
    #         'electrolyzer': global_params['capex_electrolyzer'],
    #         'H2_storage': global_params['capex_h2_storage']
    #     }
    #
    #     OPEX = {
    #         'wind': global_params['opex_wind'],
    #         'solar': global_params['opex_solar'],
    #         'battery': global_params['opex_battery'],
    #         'electrolyzer': global_params['opex_electrolyzer'],
    #         'H2_storage': global_params['opex_h2_storage']
    #     }
    #
    #     Eff = {
    #         'battery': {
    #             'charge': global_params['eff_battery_charge'],
    #             'discharge': global_params['eff_battery_discharge']
    #         },
    #         'H2_storage': {
    #             'storage': global_params['eff_h2_storage'],
    #             'discharge': global_params['eff_h2_storage']
    #         }
    #     }
    #
    #     Eff_electrolyzer = global_params['eff_electrolyzer']
    #
    #     if include_min_prod_lvl:
    #         min_prod_lvl_electrolyzer = global_params['min_prod_lvl_electrolyzer']
    #
    #     water_price = global_params['water_price']
    #     water_demand = global_params['water_demand']
    #     o2_price = global_params['o2_price']
    #
    #
    #     # Hydrogen demand calculation
    #     total_hydrogen_demand = global_params['total_h2_demand']
    #     hourly_hydrogen_demand = total_hydrogen_demand / hours_per_year
    #
    #     # Transportation cost parameters
    #     el_grid_connection_fee = global_params['el_grid_connection_fee']
    #     gas_grid_connection_fee = global_params['gas_grid_connection_fee']
    #     usage_fee = global_params['usage_fee']
    #     Interest_rate = global_params['interest_rate']
    #     heat_price = global_params['heat_price']
    #
    #
    #
    # Elec_Required_Per_Ton = config.Elec_Required_Per_Ton
    #
    # # Set up the time-varying parameters
    # if time_varying_scenario is None:
    #     wind_cf = pd.Series([config.wind_cf_constant] * time_period_length)
    #     solar_cf = pd.Series([config.solar_cf_constant] * time_period_length)
    #     grid_price = pd.Series([config.grid_price_constant] * time_period_length)
    # else:
    #     # Use the GSA scenario data for wind, solar, and grid prel_priceices
    #     wind_cf = pd.Series(time_varying_scenario['cf_wind'], index=T)
    #     solar_cf = pd.Series(time_varying_scenario['cf_solar'], index=T)
    #     grid_price = pd.Series(time_varying_scenario['el_price'], index=T)
    #
    #
    # Lifetimes = config.lifetimes
    # Annuity_factors = {
    #     tech: (Interest_rate * (1 + Interest_rate) ** Lifetimes[tech]) /
    #           ((1 + Interest_rate) ** Lifetimes[tech] - 1)
    #     for tech in Lifetimes
    # }
    #
    #
    # Max_capacities = config.max_capacities
