# output_generation.py
import seaborn as sns
import matplotlib.pyplot as plt
import config
import numpy as np


text_scaling_factor = 1
sns.set(style='darkgrid')

def plot_installed_capacities(results, max_capacities, config, grid_connection_capacity=None):
    """
    Plot installed capacities vs maximum capacities.

    Parameters:
    - results (dict): Contains the installed capacities.
    - max_capacities (dict): Maximum capacities for each technology.
    - config (Config): Configuration object with plotting preferences.
    - grid_connection_capacity (float, optional): Installed grid connection capacity if available.
    """
    if config.generate_plots and config.generate_installed_capacities_plot:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Add installed capacities
        ax.bar(results['installed_capacities'].keys(), results['installed_capacities'].values(),
               label='Installed Capacity', alpha=0.6)

        # Add maximum capacities
        ax.bar(max_capacities.keys(), max_capacities.values(), label='Max Capacity', alpha=0.3)

        # Add grid connection capacity if available
        if grid_connection_capacity is not None:
            ax.bar('Grid Connection', grid_connection_capacity, label='Grid Connection Capacity', alpha=0.6, color='orange')

        ax.set_xlabel('Technology', fontsize = 14)
        ax.set_ylabel('Capacity (MW or tons)', fontsize = 14)
        ax.set_title('Installed Capacities vs Maximum Capacities', fontsize = 16)
        ax.legend(fontsize = 12)
        # Tick font sizes
        ax.tick_params(axis='x', labelsize=12)  # X-axis ticks font size
        ax.tick_params(axis='y', labelsize=12)  # Y-axis ticks font
        plt.show()

# Plot: Energy Flows over Time
def plot_energy_flows(results, config):
    if config.generate_plots and config.generate_energy_flows_plot:
        plt.figure(figsize=(12, 6))
        plt.plot(results['time'], results['wind_prod'], label='Wind Production')
        plt.plot(results['time'], results['solar_prod'], label='Solar Production')
        plt.plot(results['time'], results['battery_charge'], label='Battery Charging')
        plt.plot(results['time'], results['battery_discharge'], label='Battery Discharging')
        plt.plot(results['time'], results['elec_sales'], label='Electricity Sold to Grid')
        plt.xlabel('Time (hours)')
        plt.ylabel('Energy (MWh)')
        plt.title('Energy Flows over Time')
        plt.legend()
        plt.grid(True)
        plt.show()
def plot_hydrogen_storage_and_production(results, config, high_cost_times=None):
    if config.generate_plots and config.generate_hydrogen_storage_plot:
        plt.figure(figsize=(12, 6))

        # Define season boundaries for a 4-week plot (assuming 168 hours per week)
        season_boundaries = [
            (0, 168, "Spring"),       # Week 1: Spring
            (168, 336, "Summer"),     # Week 2: Summer
            (336, 504, "Autumn"),     # Week 3: Autumn
            (504, 672, "Winter")      # Week 4: Winter
        ]
        season_colors = ["#98FB98", "#FFD700", "#FFB6C1", "#ADD8E6"]  # Light colors: Spring, Summer, Autumn, Winter

        # Highlight each season with background color
        for i, ((start, end, season)) in enumerate(season_boundaries):
            color = season_colors[i]  # Get the corresponding color
            plt.axvspan(start, end, color=color, alpha=0.2, label='_nolegend_')  # Exclude from legend

        # Plot hydrogen components
        plt.plot(results['time'], results['hydrogen_prod'], label='Hydrogen Production', color='green')
        plt.plot(results['time'], results['hydrogen_storage_level'], label='Hydrogen Storage Level', color='purple')
        plt.plot(results['time'], results['hydrogen_sold'], label='Hydrogen Sold', color='orange')
        plt.plot(results['time'], results['hydrogen_stored'], label='Hydrogen Stored', color='blue', linestyle='--')
        plt.plot(results['time'], results['hydrogen_from_storage'], label='Hydrogen From Storage', color='red', linestyle='--')

        # Force plot limits to update
        plt.draw()
        ylim_top = plt.gca().get_ylim()[1]  # Get updated y-axis upper limit

        # Add season labels at the top
        for (start, end, season) in season_boundaries:
            plt.text((start + end) / 2, ylim_top * 0.95, season,
                     ha='center', va='top', fontsize=12, fontweight='bold')

        # Mark times where high-cost electricity purchases were made
        if high_cost_times is not None:
            high_cost_hours = [t for t in high_cost_times]
            high_cost_storage_levels = [results['hydrogen_storage_level'][t] for t in high_cost_hours]
            plt.scatter(high_cost_hours, high_cost_storage_levels, color="red", label="High-Cost Electricity Purchased", zorder=5)

        # Labels, grid, and legend
        plt.xlabel('Time (hours)', fontsize = 14)
        plt.ylabel('Hydrogen (tons)', fontsize = 14)
        plt.title('Hydrogen Production, Storage, and Sales with High-Cost Electricity Purchases', fontsize = 16)
        plt.xticks(fontsize=12)  # Set xtick size to 12
        plt.yticks(fontsize=12)  # Set ytick size to 12
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# Table: Installed Capacities vs Maximum Capacities
def table_installed_capacities(results, max_capacities, config):
    if config.generate_tables and config.generate_installed_capacities_table:
        print("\nInstalled Capacities vs Maximum Capacities:")
        print(f"{'Technology':<15} {'Installed Capacity (MW or tons)':<30} {'Max Capacity (MW or tons)':<30}")
        for tech, value in results['installed_capacities'].items():
            max_cap = max_capacities.get(tech, 'No Limit')
            print(f"{tech:<15} {value:<30.2f} {max_cap:<30}")

# Table: Energy Flows
def table_energy_flows(results, config):
    if config.generate_tables and config.generate_energy_flows_table:
        print("\nEnergy Flows over Time:")
        print(f"{'Hour':<10} {'Wind Production (MW)':<25} {'Solar Production (MW)':<25} "
              f"{'Battery Charge (MW)':<25} {'Battery Discharge (MW)':<25} "
              f"{'Battery SOC (MWh)':<25} {'Electricity Sold (MW)':<25}")
        for t in results['time']:
            print(f"{t:<10} {results['wind_prod'][t]:<25.2f} {results['solar_prod'][t]:<25.2f} "
                  f"{results['battery_charge'][t]:<25.2f} {results['battery_discharge'][t]:<25.2f} "
                  f"{results['battery_soc'][t]:<25.2f} {results['elec_sales'][t]:<25.2f}")

# Table: Hydrogen Storage and Production
def table_hydrogen_storage_and_production(results, config):
    if config.generate_tables and config.generate_hydrogen_storage_table:
        print("\nHydrogen Production, Storage, and Sales:")
        print(f"{'Hour':<10} {'H2 Production (tons)':<25} {'H2 Storage Level (tons)':<25} "
              f"{'H2 Sold (tons)':<25} {'H2 Stored (tons)':<25} {'H2 From Storage (tons)':<25}")
        for t in results['time']:
            print(f"{t:<10} {results['hydrogen_prod'][t]:<25.2f} {results['hydrogen_storage_level'][t]:<25.2f} "
                  f"{results['hydrogen_sold'][t]:<25.2f} {results['hydrogen_stored'][t]:<25.2f} {results['hydrogen_from_storage'][t]:<25.2f}")


def plot_battery_soc(results, config):
    if  config.generate_battery_soc_plot and config.generate_plots:
        # Extract data from results
        battery_charge = results['battery_charge']
        battery_discharge = results['battery_discharge']
        battery_soc = results['battery_soc']
        time_steps = range(len(battery_charge))  # Assume sequential time steps

        # Plot battery charge, discharge, and SOC
        plt.figure(figsize=(12, 6))
        plt.plot(time_steps, battery_charge, label='Battery Charge (MW)', linestyle='--', marker='o')
        plt.plot(time_steps, battery_discharge, label='Battery Discharge (MW)', linestyle='--', marker='o')
        plt.plot(time_steps, battery_soc, label='Battery SOC (MWh)', linestyle='-', marker='x')

        plt.xlabel('Time (hours)')
        plt.ylabel('Battery Metrics')
        plt.title('Battery Charge, Discharge, and SOC over Time')
        plt.legend()
        plt.grid(True)
        plt.show()


def table_energy_requirements(results, hydrogen_demand_per_hour, wind_max_cap, solar_max_cap, wind_cf, solar_cf,
                              config):
    if config.generate_tables and config.generate_energy_requirements_table:
        print("\nEnergy Requirements and Availability for Hydrogen Production:")
        print(f"{'Hour':<10} {'Hydrogen Energy Demand (MWh)':<30} {'Wind Available Energy (MWh)':<30} "
              f"{'Solar Available Energy (MWh)':<30} {'Total Available Energy (MWh)':<30} {'Percentage Difference (%)':<30}")

        for t in range(len(results['hydrogen_prod'])):
            hydrogen_energy_demand = hydrogen_demand_per_hour
            wind_available_energy = wind_max_cap * wind_cf[t]
            solar_available_energy = solar_max_cap * solar_cf[t]
            total_available_energy = wind_available_energy + solar_available_energy
            percentage_diff = ((total_available_energy - hydrogen_energy_demand) / hydrogen_energy_demand) * 100

            print(f"{t:<10} {hydrogen_energy_demand:<30.2f} {wind_available_energy:<30.2f} "
                  f"{solar_available_energy:<30.2f} {total_available_energy:<30.2f} {percentage_diff:<30.2f}")

        # Define season boundaries
        season_boundaries = [
            (0, 168, "Spring"),
            (168, 336, "Summer"),
            (336, 504, "Autumn"),
            (504, 672, "Winter")
        ]

        # Initialize table data
        seasonal_data = []

        # Calculate cycles and energy for each season
        for start, end, season in season_boundaries:
            # Available energy per season
            wind_available_energy = sum(wind_max_cap * wind_cf[start:end])
            solar_available_energy = sum(solar_max_cap * solar_cf[start:end])
            total_available_energy = wind_available_energy + solar_available_energy

            # Append results to table
            seasonal_data.append([season,round(wind_available_energy), round(solar_available_energy),
                                  round(total_available_energy)])
            df = pd.DataFrame(seasonal_data, columns=[
             "Season",
            "Wind Energy (MWh)", "Solar Energy (MWh)", "Total Energy (MWh)"
        ])

            print(df)

def plot_energy_requirements(results, hydrogen_demand_per_hour, wind_max_cap, solar_max_cap, wind_cf, solar_cf, config,
                             high_cost_times = None):
    if config.generate_plots and config.generate_energy_requirements_plot:
        # Calculate hourly data
        hydrogen_energy_demand = [hydrogen_demand_per_hour] * len(wind_cf)
        wind_available_energy = [wind_max_cap * wind_cf[t] for t in range(len(wind_cf))]
        solar_available_energy = [solar_max_cap * solar_cf[t] for t in range(len(solar_cf))]
        total_available_energy = [wind_available_energy[i] + solar_available_energy[i] for i in range(len(wind_cf))]


        # Calculate cumulative totals
        total_energy_needed = sum(hydrogen_energy_demand)
        total_energy_available = sum(total_available_energy)
        percentage_of_available = (total_energy_needed/total_energy_available)*100

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Left subplot: Hourly energy requirements vs availability
        ax1.plot(hydrogen_energy_demand, label="Hydrogen Energy Demand (MWh)", color="blue")
        ax1.plot(total_available_energy, label="Total Available Energy (MWh)", color="green", linestyle='--')

        # Mark high-cost purchase times only if there were any
        if high_cost_times != None:
            high_cost_hours = [i for i in high_cost_times]
            high_cost_values = [total_available_energy[i] for i in high_cost_hours]
            ax1.scatter(high_cost_hours, high_cost_values, color="red", label="High-Cost Electricity Purchased",
                        zorder=5)

        ax1.set_xlabel("Hour", fontsize = 14)
        ax1.set_ylabel("Energy (MWh)", fontsize = 14)
        ax1.set_title("Hourly Energy Requirements vs Available Renewable Energy", fontsize = 16)
        ax1.legend(fontsize = 12)
        ax1.grid(True)

        # Right subplot: Cumulative energy comparison
        ax2.bar(['Total Energy Needed', 'Total Energy Available'], [total_energy_needed, total_energy_available],
                color=['blue', 'green'])
        ax2.set_title("Total Energy Comparison", fontsize = 16)
        # Add percentage on top of the "Total Energy Needed" bar
        ax2.text(0, total_energy_needed + 0.05 * total_energy_needed,
                 f"{percentage_of_available:.2f}%",
                 ha='center', va='bottom', fontsize=14, color='black')

        plt.tight_layout()
        plt.show()
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
def plot_cost_revenue_overview(financial_results, config):
    """
    Plot cost and revenue overview using financial results dictionary.

    Parameters:
    - financial_results (dict): Dictionary containing CAPEX, OPEX, and revenue data.
    - config (Config): Configuration object with plotting preferences.
    """
    if config.generate_plots and config.generate_cost_revenue_plot:
        # Extract costs and revenues from the financial results
        costs = {
            key: value for key, value in financial_results.items()
            if key.startswith('CAPEX_') or key.startswith('OPEX') or key in ['Transport', 'Water Cost']
        }
        revenues = {
            key: value for key, value in financial_results.items()
            if key in ['Hydrogen Revenue', 'Electricity Sales Revenue', 'Heat Revenue', 'O2 Revenue']
        }

        # Include Start-up cost if present
        if financial_results.get('Start_up_cost') is not None:
            costs['Start_up_cost'] = financial_results['Start_up_cost']

        # Include Grid Connection Cost if present
        if financial_results.get('el_grid_connection_cost') is not None:
            costs['el_grid_connection_cost'] = financial_results['el_grid_connection_cost']

        # Aggregate OPEX costs
        opex_sum = sum(value for key, value in costs.items() if key.startswith('OPEX'))
        costs = {key: value for key, value in costs.items() if not key.startswith('OPEX')}
        costs['Total_OPEX'] = opex_sum

        # Sort costs by value in descending order and rearrange small costs to alternate
        sorted_costs_items = sorted(costs.items(), key=lambda item: item[1], reverse=True)
        sorted_large_costs = [item for item in sorted_costs_items if item[1] >= 0.05 * sum(costs.values())]
        sorted_small_costs = [item for item in sorted_costs_items if item[1] < 0.05 * sum(costs.values())]

        # Interleave small costs with large costs to prevent clustering
        interleaved_costs = []
        small_index, large_index = 0, 0
        while large_index < len(sorted_large_costs) or small_index < len(sorted_small_costs):
            if large_index < len(sorted_large_costs):
                interleaved_costs.append(sorted_large_costs[large_index])
                large_index += 1
            if small_index < len(sorted_small_costs):
                interleaved_costs.append(sorted_small_costs[small_index])
                small_index += 1

        # Separate the interleaved costs into labels and values
        interleaved_cost_labels, interleaved_cost_values = zip(*interleaved_costs)

        # Explode smaller slices for visibility
        cost_explode = [0.1 if val < 0.05 * sum(costs.values()) else 0 for val in interleaved_cost_values]

        # Omit annotation for start-up cost
        interleaved_cost_labels = ['' if label == 'Start_up_cost' else label for label in interleaved_cost_labels]

        # Sort revenues by value in descending order and rearrange small revenues to alternate
        sorted_revenue_items = sorted(revenues.items(), key=lambda item: item[1], reverse=True)
        sorted_large_revenues = [item for item in sorted_revenue_items if item[1] >= 0.05 * sum(revenues.values())]
        sorted_small_revenues = [item for item in sorted_revenue_items if item[1] < 0.05 * sum(revenues.values())]

        # Interleave small revenues with large revenues to prevent clustering
        interleaved_revenues = []
        small_index, large_index = 0, 0
        while large_index < len(sorted_large_revenues) or small_index < len(sorted_small_revenues):
            if large_index < len(sorted_large_revenues):
                interleaved_revenues.append(sorted_large_revenues[large_index])
                large_index += 1
            if small_index < len(sorted_small_revenues):
                interleaved_revenues.append(sorted_small_revenues[small_index])
                small_index += 1

        # Separate the interleaved revenues into labels and values
        interleaved_revenue_labels, interleaved_revenue_values = zip(*interleaved_revenues)

        # Explode smaller slices for visibility
        revenue_explode = [0.1 if val < 0.05 * sum(revenues.values()) else 0 for val in interleaved_revenue_values]

        # Create the pie chart with CAPEX broken down by technology
        fig, axs = plt.subplots(1, 2, figsize=(14, 7))

        # Cost pie chart with adjusted order and no shadow
        axs[0].pie(
            interleaved_cost_values,
            labels=interleaved_cost_labels,
            autopct='%1.1f%%',
            startangle=120,  # Adjusted start angle for balance
            explode=cost_explode,
            wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
            shadow=False,  # Shadow removed
            textprops={'fontsize': 14}
        )
        axs[0].set_title("Cost Composition", fontsize=16)

        # Revenue pie chart with adjusted order and no shadow
        axs[1].pie(
            interleaved_revenue_values,
            labels=interleaved_revenue_labels,
            autopct='%1.1f%%',
            startangle=140,
            explode=revenue_explode,
            wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
            shadow=False,  # Shadow removed
            textprops={'fontsize': 14}
        )
        axs[1].set_title("Revenue Composition", fontsize=16)

        plt.suptitle("Cost and Revenue Overview", fontsize=16)
        plt.tight_layout()
        plt.show()

def table_cost_revenue_overview(financial_results, config):
    """
    Print cost and revenue overview in a readable table format, with values in thousands of €.

    Parameters:
    - financial_results (dict): Dictionary containing CAPEX, OPEX, and revenue data.
    - config (Config): Configuration object with table preferences.
    """
    if config.generate_tables and config.generate_cost_revenue_table:
        print("\nCost and Revenue Overview (in Thousands of €):")
        print("-" * 50)

        # Print CAPEX details
        capex_details = {key: value / 1_000 for key, value in financial_results.items() if key.startswith('CAPEX_')}
        total_capex = sum(capex_details.values())
        print(f"{'Category':<25} {'Amount (k€)':>20}")
        print("-" * 50)
        for tech, cost in capex_details.items():
            print(f"{tech:<25} {cost:>20.2f}")
        print(f"{'Total CAPEX':<25} {total_capex:>20.2f}")
        print("-" * 50)

        # Print OPEX details
        opex_details = {key: value / 1_000 for key, value in financial_results.items() if key.startswith('OPEX_')}
        total_opex = sum(opex_details.values())
        for tech, cost in opex_details.items():
            print(f"{tech:<25} {cost:>20.2f}")
        print(f"{'Total OPEX':<25} {total_opex:>20.2f}")
        print("-" * 50)

        # Print additional costs
        other_costs = {
            'Transport Costs': financial_results.get('Transport', 0) / 1_000,
            'Water Cost': financial_results.get('Water Cost', 0) / 1_000
        }

        if 'Start_up_cost' not in financial_results and 'el_grid_connection_cost' not in financial_results:
            for name, cost in other_costs.items():
                print(f"{name:<25} {cost:>20.2f}")

        elif 'Start_up_cost' in financial_results and 'el_grid_connection_cost' not in financial_results:
            other_costs.update({'Start_up_cost': financial_results.get('Start_up_cost')/1_000})
            print(f"Startup Cost Term: {other_costs['Start_up_cost']}")
            for name, cost in other_costs.items():
                print(f"{name:<25} {cost:>20.2f}")

        elif 'Start_up_cost' not in financial_results and 'el_grid_connection_cost' in financial_results:
            other_costs.update({'el_grid_connection_cost': financial_results.get('el_grid_connection_cost')/1_000})
            print(f"el_grid_connection_cost: {other_costs['el_grid_connection_cost']}")
            for name, cost in other_costs.items():
                print(f"{name:<25} {cost:>20.2f}")
        else:
            other_costs.update({'Start_up_cost': financial_results.get('Start_up_cost') / 1_000})
            other_costs.update({'el_grid_connection_cost': financial_results.get('el_grid_connection_cost') / 1_000})
            for name, cost in other_costs.items():
                print(f"{name:<25} {cost:>20.2f}")






        total_costs = total_capex + total_opex + sum(other_costs.values())
        print(f"{'Total Costs':<25} {total_costs:>20.2f}")
        print("-" * 50)

        # Print revenue details
        revenues = {
            'Hydrogen Revenue': financial_results.get('Hydrogen Revenue', 0) / 1_000,
            'Electricity Sales Revenue': financial_results.get('Electricity Sales Revenue', 0) / 1_000,
            'Heat Revenue': financial_results.get('Heat Revenue', 0) / 1_000,
            'O₂ Revenue': financial_results.get('O2 Revenue', 0) / 1_000
        }
        total_revenues = sum(revenues.values())
        for name, revenue in revenues.items():
            print(f"{name:<25} {revenue:>20.2f}")
        print(f"{'Total Revenue':<25} {total_revenues:>20.2f}")
        print("-" * 50)

        # Print comparison of total costs and total revenues
        print(f"{'Net Balance (Revenue - Cost)':<25} {total_revenues - total_costs:>20.2f}")

def table_full_load_hours(results, config, scaling_factor):
    if config.generate_tables and config.generate_flh_table:
        print("\nFull Load Hours per Technology:")
        print(f"{'Technology':<20} {'Full Load Hours (hours)':<25}")

        for tech, energy_usage in results['energy_usage'].items():
            # Skip inflow components for storage
            if tech == 'hydrogen_storage_inflow':
                continue  # Ignore hydrogen inflow; only consider discharge
            elif tech == 'battery_charge':
                continue  # Ignore battery inflow; only consider discharge
            elif tech == 'battery_discharge':
                # For battery, calculate FLH using discharge and installed capacity
                base_tech = 'battery'
                flh = (scaling_factor * sum(energy_usage)) / results['installed_capacities'][base_tech]
            elif tech == 'hydrogen_storage_outflow':
                # For hydrogen storage, use outflow and installed capacity
                base_tech = 'H2_storage'
                flh = (scaling_factor * sum(energy_usage)) / results['installed_capacities'][base_tech]
            else:
                # Production technologies and electrolyzer
                flh = (scaling_factor * sum(energy_usage)) / results['installed_capacities'][tech]
            print(f"{tech:<20} {flh:<25.2f}")

def plot_full_load_hours(results, config, scaling_factor):
    if config.generate_plots and config.generate_flh_plot:
        flh_values = {}

        for tech, energy_usage in results['energy_usage'].items():
            # Skip inflow components for storage
            if tech == 'hydrogen_storage_inflow':
                continue  # Ignore hydrogen inflow; only consider outflow
            elif tech == 'battery_charge':
                continue  # Ignore battery charge; only consider discharge
            elif tech == 'battery_discharge':
                continue
            elif tech == 'hydrogen_storage_outflow':
                continue
            else:
                # For all other technologies and electrolyzer
                flh_values[tech] = (scaling_factor * sum(energy_usage)) / results['installed_capacities'][tech]

        # Plotting FLH values
        plt.figure(figsize=(10, 6))
        plt.bar(flh_values.keys(), flh_values.values(), color = "#8A9A5B", alpha=0.7)
        plt.xlabel('Technology', fontsize = 14)
        plt.ylabel('Full Load Hours (hours)', fontsize = 14)
        plt.title('Full Load Hours per Technology', fontsize = 16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.savefig(r"C:\Users\Mika\Desktop\Master\Pictures_Master\flh.png", bbox_inches="tight", dpi=300)
        plt.grid(True)
        plt.show()
import pandas as pd

def plot_storage_utilization_and_cycles(results, config, scaling_factor, text_scaling_factor=text_scaling_factor):
    if config.generate_plots and config.generate_utilization_and_cycles_plot:
        # Extract relevant results
        installed_capacities = results['installed_capacities']
        battery_charge = results['battery_charge']
        battery_discharge = results['battery_discharge']
        hydrogen_stored = results['hydrogen_stored']
        hydrogen_from_storage = results['hydrogen_from_storage']

        # Calculate cycles for battery and hydrogen storage
        battery_cycles = (
            (scaling_factor * (sum(battery_discharge)) / (installed_capacities['battery']))
            if installed_capacities['battery'] > 0 else 0
        )

        hydrogen_cycles = (
            (scaling_factor * (sum(hydrogen_from_storage)) / (
                    installed_capacities['H2_storage']))
            if installed_capacities['H2_storage'] > 0 else 0
        )
        print(battery_cycles, hydrogen_cycles)
        # Define season boundaries
        season_boundaries = [
            (0, 168, "Spring"),
            (168, 336, "Summer"),
            (336, 504, "Autumn"),
            (504, 672, "Winter")
        ]

        # Initialize table data
        seasonal_cycles = []

        # Calculate cycles for each season
        for start, end, season in season_boundaries:
            # Battery cycles for the season
            seasonal_battery_cycles = (
                (scaling_factor * (sum(battery_charge[start:end]) + sum(battery_discharge[start:end])) /
                 (2 * installed_capacities['battery']))
                if installed_capacities['battery'] > 0 else 0
            )

            # Hydrogen cycles for the season
            seasonal_hydrogen_cycles = (
                (scaling_factor * (sum(hydrogen_stored[start:end]) + sum(hydrogen_from_storage[start:end])) /
                 (2 * installed_capacities['H2_storage']))
                if installed_capacities['H2_storage'] > 0 else 0
            )

            # Append results to table
            seasonal_cycles.append([season, seasonal_battery_cycles, seasonal_hydrogen_cycles])

        # Create DataFrame for better visualization
        cycles_df = pd.DataFrame(seasonal_cycles, columns=["Season", "Battery Cycles", "Hydrogen Cycles"])

        # Display table
        print(cycles_df)


        # Plotting
        fig, ax1 = plt.subplots(figsize=(6, 4))

        # Subplot for Cycles
        storage_techs = ['Battery', 'Hydrogen Storage']
        cycles = [battery_cycles, hydrogen_cycles]
        ax1.bar(storage_techs, cycles, color=["#8A9A5B", "#C3A36F"], alpha=0.7)  # Light olive green and ocker
        ax1.set_title('Cycles', fontsize=16 * text_scaling_factor)
        ax1.set_xlabel('Storage Technology', fontsize=14 * text_scaling_factor)
        ax1.set_ylabel('Cycles', fontsize=14 * text_scaling_factor)
        ax1.tick_params(axis='both', labelsize= 12* text_scaling_factor)

        # Main title
        fig.suptitle('Cycles for Battery and Hydrogen Storage', fontsize=14 * text_scaling_factor)
        plt.tight_layout()
        plt.savefig(r"C:\Users\Mika\Desktop\Master\Pictures_Master\figure_cycles.png", bbox_inches="tight", dpi=300)
        plt.show()
import matplotlib.pyplot as plt
import numpy as np

def plot_energy_flows_and_grid_price(results, config, text_scaling_factor=1.0):
    if config.generate_plots and config.plot_energy_flows_and_grid_price:
        time = results['time']
        solar_prod = np.array(results['solar_prod'])
        wind_prod = np.array(results['wind_prod'])
        battery_charge = np.array(results['battery_charge'])
        battery_discharge = np.array(results['battery_discharge'])
        electrolyzer_intake = np.array(results['electrolyzer_intake'])
        elec_sales = np.array(results['elec_sales'])
        grid_price = np.array(results['grid_price'])
        battery_max_discharge = round(results['battery_max_discharge'])
        electrolyzer_max_intake = round(results['electrolyzer_max_intake'])

        # Create a figure with two subplots
        fig, ax1 = plt.subplots(figsize=(14, 7))

        # Define season boundaries for a 4-week plot (assuming 168 hours per week)
        season_boundaries = [
            (0, 168, "Spring"),
            (168, 336, "Summer"),
            (336, 504, "Autumn"),
            (504, 672, "Winter")
        ]
        season_colors = ["#98FB98", "#FFD700", "#FFB6C1", "#ADD8E6"]

        # Highlight each season with background color
        for i, (start, end, season) in enumerate(season_boundaries):
            color = season_colors[i]
            ax1.axvspan(start, end, color=color, alpha=0.2, label='_nolegend_')

        # Plot the main energy data
        ax1.fill_between(time, solar_prod, label="Solar Production", color="gold", alpha=0.6, step='mid')
        ax1.fill_between(time, solar_prod + wind_prod, solar_prod, label="Wind Production", color="skyblue", alpha=0.6,
                         step='mid')
        ax1.step(time, battery_charge, label="Battery Charge", color="blue", alpha=0.7, where='mid', linestyle="--")
        ax1.step(time, battery_discharge, label="Battery Discharge", color="red", alpha=0.7, where='mid',
                 linestyle="--")
        ax1.step(time, elec_sales, label="Electricity Sold to Grid", color="orange", alpha=0.8, where='mid')
        ax1.step(time, electrolyzer_intake, label="Electrolyzer Intake", color="green", alpha=0.8, where='mid')

        ax1.set_xlabel("Time (hours)", fontsize=14 * text_scaling_factor)
        ax1.set_ylabel("Electricity (MWh)", fontsize=14 * text_scaling_factor)
        ax1.set_title("Energy Flows with Maximum Limits", fontsize=16 * text_scaling_factor)

        # Add y-axis markers for max battery discharge rate and max electrolyzer intake
        ax1.annotate(
            f"{battery_max_discharge}",
            xy=(-0.01, battery_max_discharge),  # Adjusted position to align with tick numbers
            xycoords=("axes fraction", "data"),
            textcoords="offset points",
            xytext=(-5, 0),  # Position directly under black tick numbers
            color="red",
            fontsize=12,  # Match size with tick numbers
            ha="center",  # Center align the text
            #arrowprops=dict(arrowstyle="-", color="red", lw=0.8)
        )

        ax1.annotate(
            f"{electrolyzer_max_intake}",
            xy=(-0.01, electrolyzer_max_intake),  # Adjusted position to align with tick numbers
            xycoords=("axes fraction", "data"),
            textcoords="offset points",
            xytext=(-8, 0),  # Position directly under black tick numbers
            color="green",
            fontsize=12,  # Match size with tick numbers
            ha="center",  # Center align the text
            #arrowprops=dict(arrowstyle="-", color="green", lw=0.8)
        )

        # Grid price on the second y-axis
        ax2 = ax1.twinx()
        ax2.plot(time, grid_price, label="Grid Price", color="black", linestyle="-", alpha=0.8)
        ax2.set_ylabel("Grid Price (€/MWh)", fontsize=14 * text_scaling_factor)
        ax2.tick_params(axis='both', labelsize=12 * text_scaling_factor)
        ax2.legend(loc="upper right", fontsize=12 * text_scaling_factor)

        # Grid price on the second y-axis
        ax2 = ax1.twinx()
        ax2.plot(time, grid_price, label="Grid Price", color="black", linestyle="-", alpha=0.8)
        ax2.set_ylabel("Grid Price (€/MWh)", fontsize=14 * text_scaling_factor)
        ax2.tick_params(axis='both', labelsize=12 * text_scaling_factor)
        ax2.legend(loc="upper right", fontsize=12 * text_scaling_factor)

        # Force plot limits to update
        plt.draw()
        ylim_top = ax1.get_ylim()[1]

        # Add season labels at the top
        for (start, end, season) in season_boundaries:
            ax1.text((start + end) / 2, ylim_top * 0.95, season,
                     ha='center', va='top', fontsize=12 * text_scaling_factor, fontweight='bold', color='black')

        ax1.legend(loc="upper left", fontsize=12 * text_scaling_factor)
        plt.tight_layout()
        plt.savefig(r"C:\Users\Mika\Desktop\Master\Pictures_Master\plot_energy_flows_and_grid_price.png", bbox_inches="tight", dpi=300)
        plt.show()
