import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import matplotlib.pyplot as plt

# Option to print tables for plots (set to True or False)
generate_tables = True
generate_plots = True
# Define the length of the time period in hours (e.g., 168 for one week)
time_period_length = 168 # This can be changed dynamically before running the model
# Number of hours in one year
hours_per_year = 8760
# Calculate the scaling factor based on the length of the time period
scaling_factor = hours_per_year / time_period_length

# DEFINE SETS
S = ['wind', 'solar', 'battery', 'electrolyzer', 'H2_storage']  # Set of technology units
# Define set for electricity-producing technologies
S_el = ['wind', 'solar']  # Set of electricity-producing technologies
S_in_out = ['battery', 'electrolyzer']  # Technologies that require x_in and x_out
T = list(range(time_period_length))  # Set of time periods for the specified length

# Define parameters (replace with actual values from the model)
C_var = {
    'electrolyzer': 50,  # Example variable operating cost for electrolyzer (€/MWh)
    'wind': 0,  # Wind and solar typically have no variable costs
    'solar': 0,
    'battery': 1,
    'H2_storage': 1,
}
Investment_cost = {
    'wind': 1000,  # Example investment cost per MW (€/MW)
    'solar': 800,  # Example investment cost per MW (€/MW)
    'battery': 500,  # Example investment cost per MW (€/MW)
    'electrolyzer': 700,  # Example investment cost per MW (€/MW)
    'H2_storage': 20000,  # Example investment cost per ton (€/ton)
}

# Define the interest rate (discount rate) and technology lifetimes
Interest_rate = 0.05  # Example discount rate of 5%

Lifetimes = {
    'wind': 20,        # Lifetime in years for wind turbines
    'solar': 25,       # Lifetime in years for solar panels
    'battery': 10,     # Lifetime in years for battery storage
    'electrolyzer': 15, # Lifetime in years for electrolyzers
    'H2_storage': 30   # Lifetime in years for hydrogen storage
}

# Calculate the annuity factor for each technology using the formula
Annuity_factors = {tech: (Interest_rate * (1 + Interest_rate) ** Lifetimes[tech]) /
                           ((1 + Interest_rate) ** Lifetimes[tech] - 1)
                   for tech in Lifetimes}

# Example output: Annuity_factors['wind'] = 0.0802 (This would represent the annuity factor for wind)
Eff = {
    'battery': {
        'charge': 1,        # Charging efficiency for battery
        'discharge': 1      # Discharging efficiency for battery
    },
    'H2_storage': {
        'storage': 0.9,     # Storage efficiency for hydrogen
        'discharge': 0.9    # Discharge efficiency for hydrogen storage
    }
}
Eff_electrolyzer = 0.7  # Example electrolyzer efficiency (70%)
P_min = {
    'electrolyzer': 10  # Example minimum operating level for electrolyzer (MW)
}
Elec_Required_Per_Ton = 33   # MWh/ton: Electricity required to produce one ton of hydrogen before efficiency is applied [LHV]
grid_price = pd.Series([5] * time_period_length)  # Example constant grid price (€/MWh) for the selected time period

# Define ceiling values for specific technologies
Max_capacities = {
    'wind': 107,  # Maximum 200 MW for wind
    'solar': 1,  # Maximum 200 MW for solar
    'battery': 20,  # Maximum 20 MWh for battery storage
    'H2_storage': 200,  # Maximum 20 tons for hydrogen storage
}

# Load capacity factor time series for the selected time period (replace with actual data)
wind_cf = pd.Series([1] * time_period_length)  # Example capacity factor for wind
solar_cf = pd.Series([1] * time_period_length)  # Example capacity factor for solar

# Hydrogen demand calculation based on the specified time period
total_hydrogen_demand = 20000 # tons
hourly_hydrogen_demand = total_hydrogen_demand / hours_per_year  # Distributed across the hours of a year (tons/hour)

# Transportation cost parameters
connection_fee = 0  # Fixed connection fee to hydrogen pipeline (€)
usage_fee = 10  # Variable usage fee per ton of hydrogen transported (€/ton)

# Create a new model
model = gp.Model("HydrogenAndBatteryInvestmentWithPriceMinimizationAndDynamicPeriod")

# Add decision variables for new capacities
#P_invest = model.addVars(S, name="P_invest")  # Investment in new capacity for each technology
# Add decision variables for new capacities with upper bounds (ceilings)
P_invest = model.addVars(S, ub=[Max_capacities.get(g, GRB.INFINITY) for g in S], name="P_invest")

# Add decision variable for electricity sales to the grid
elec_sold = model.addVars(T, name="elec_sold")  # Electricity sold to the grid

# Add decision variable for the hydrogen selling price
hydrogen_price = model.addVar(name="hydrogen_price")  # Selling price of hydrogen (€/ton)

# Modify decision variables
# Add electricity production variable for wind and solar
el_prod = model.addVars(S_el, T, name="el_prod")  # Electricity production for wind and solar
x_in = model.addVars(S_in_out, T, name="x_in")  # Input electricity consumption for battery and electrolyzer (MW)
x_out = model.addVars(S_in_out, T, name="x_out")  # Output (discharge) for battery and hydrogen production for electrolyzer (MW)
soc = model.addVars(S, T, name="soc")  # State of charge for hydrogen and battery storage (MWh or tons)
u = model.addVars(S, T, vtype=GRB.BINARY, name="u")  # Unit commitment for electrolyzers and battery operation
u_charge = model.addVars(T, vtype=GRB.BINARY, name="u_charge")
u_discharge = model.addVars(T, vtype=GRB.BINARY, name="u_discharge")
h2_sold = model.addVars(T, name="h2_sold")  # Hydrogen sold (tons)
h2_stored = model.addVars(T, name="h2_stored")  # Hydrogen stored in storage (tons)
h2_from_storage = model.addVars(T, name="h2_from_storage")  # Hydrogen taken from storage for sale (tons)

# Objective function: minimize the selling price of hydrogen
model.setObjective(
    hydrogen_price, GRB.MINIMIZE
)

#                                                     CONSTRAINTS

# Cost Coverage Constraint with Transportation Costs and Hydrogen Revenue
model.addConstr(
    scaling_factor * gp.quicksum(h2_sold[t] * (hydrogen_price - usage_fee) for t in T) >=
    scaling_factor * gp.quicksum(
        C_var[g] * gp.quicksum(x_out[g, t] for t in T) for g in S_in_out
    ) + gp.quicksum(
        P_invest[g] * Investment_cost[g] * Annuity_factors[g] for g in S
    ) - scaling_factor * gp.quicksum(elec_sold[t] * grid_price[t] for t in T) + connection_fee,
    name="CostCoverage"
)




#=                                          PRODUCTION LIMITS
#=
# Ensure non-zero investment if required
#model.addConstrs((P_invest[g] >= 1 for g in ['wind', 'solar']), name="MinInvestment")

# Ensure hydrogen production cannot exceed installed electrolyzer capacity
model.addConstrs(
    (x_out['electrolyzer', t] <= P_invest['electrolyzer'] * Eff_electrolyzer / Elec_Required_Per_Ton for t in T),
    name="HydrogenProductionCapacityLimit"
)
# Production limits based on new capacity and time-varying capacity factors for wind and solar
model.addConstrs(
    (el_prod[g, t] == P_invest[g] * wind_cf[t] for g in ['wind'] for t in T),
    name="WindProductionLimit"
)

model.addConstrs(
    (el_prod[g, t] == P_invest[g] * solar_cf[t] for g in ['solar'] for t in T),
    name="SolarProductionLimit"
)

# Limit hydrogen storage by the invested capacity in H2 storage
model.addConstrs(
    (soc['H2_storage', t] <= P_invest['H2_storage'] for t in T),
    name="HydrogenStorageCapacityLimit"
)
# Limit battery storage by the invested capacity in the battery
model.addConstrs(
    (soc['battery', t] <= P_invest['battery'] for t in T),
    name="BatteryStorageCapacityLimit"
)

#                                        STORAGE CONSTRAINTS
# Ensure the battery is initially empty
model.addConstr(soc['battery', 0] == 0, name="InitialBatterySOC")

# Set the initial hydrogen storage to zero
model.addConstr(soc['H2_storage', 0] == 0, name="InitialHydrogenSOC")

# SOC update for hydrogen storage in the first period
model.addConstr(
    soc['H2_storage', 0] == Eff['H2_storage']['storage'] * h2_stored[0] - h2_from_storage[0],
    name="InitialHydrogenSOCUpdate"
)

# Ensure battery charging and discharging are limited by installed capacity
model.addConstrs(
    (x_in['battery', t] <= u_charge[t] * P_invest['battery']  for t in T),
    name="BatteryChargeLimit"
)

model.addConstrs(
    (x_out['battery', t] <= u_discharge[t] * P_invest['battery']  for t in T),
    name="BatteryDischargeLimit"
)

# SOC update for battery storage
model.addConstrs(
    (soc['battery', t] == soc['battery', t-1] + Eff['battery']['charge'] * x_in['battery', t] - Eff['battery']['discharge'] * x_out['battery', t]
     for t in T if t > 0),
    name="SOCUpdate"
)

# Prevent simultaneous charging and discharging
model.addConstrs(
    (u_charge[t] + u_discharge[t] <= 1 for t in T),
    name="NoSimultaneousChargeDischarge"
)

#                                                   HYDROGEN CONSTRAINTS


# Hydrogen balance: Hydrogen produced is used for either current sales or storage
model.addConstrs(
    (x_out['electrolyzer', t] == h2_sold[t] + Eff['H2_storage']['storage'] * h2_stored[t] - Eff['H2_storage']['discharge'] * h2_from_storage[t]
     for t in T),
    name="HydrogenBalance"
)

# Hydrogen Production Constraint: Tie hydrogen production to the installed capacity of the electrolyzer and include efficiency
model.addConstrs(
    (x_out['electrolyzer', t] == x_in['electrolyzer', t] * Eff_electrolyzer / Elec_Required_Per_Ton for t in T),
    name="HydrogenProduction"
)
# Hydrogen stored in each period considering storage efficiency
model.addConstrs(
    (soc['H2_storage', t] == soc['H2_storage', t-1] + Eff['H2_storage']['storage'] * h2_stored[t] - Eff['H2_storage']['discharge'] * h2_from_storage[t]
     for t in T if t > 0),
    name="HydrogenSOCUpdate"
)


# Ensure electrolyzer can only use electricity generated by wind, solar, or from the battery
model.addConstrs(
    (x_in['electrolyzer', t] <= el_prod['wind', t] + el_prod['solar', t]  +  Eff['battery']['discharge'] * x_out['battery', t] for t in T),
    name="ElectrolyzerEnergySource"
)

# Hydrogen sold must equal the hourly hydrogen demand
model.addConstrs(
    (h2_sold[t] == hourly_hydrogen_demand for t in T),
    name="HydrogenDemand"
)

#                                   Unit COMMITMENT CONSTRAINTS  FOR BATTERY AND ELECTROLYZER

# Comment out to check if this constraint is causing infeasibility
# model.addConstrs(
#     (u['electrolyzer', t] * P_min['electrolyzer'] <= x_out['electrolyzer', t] for t in T),
#     name="ElectrolyzerMinOperation"
# )

# Ensure electrolyzer operates within installed capacity when committed
model.addConstrs(
    (x_out['electrolyzer', t] <= u['electrolyzer', t] * P_invest['electrolyzer'] for t in T),
    name="ElectrolyzerMaxOperation"
)



# Electricity balance: generated electricity minus consumption is either stored or sold
model.addConstrs(
    (
        el_prod['wind', t] + el_prod['solar', t] + x_out['battery', t] ==
        x_in['electrolyzer', t] + x_in['battery', t] + elec_sold[t]
        for t in T
    ), name="ElectricityBalance"
)
# Optimize the model
model.optimize()


# Check for infeasibility
if model.status == GRB.INFEASIBLE:
    print("Model is infeasible; computing IIS")
    model.computeIIS()
    model.write("model.ilp")
    for c in model.getConstrs():
        if c.IISConstr:
            print(f"Infeasible constraint: {c.constrName}")
    print("Skipping plots due to infeasibility.")
else:
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set Seaborn style
    sns.set(style='darkgrid')  # You can try other styles like 'whitegrid', 'dark', etc.
    # Functions for plots (independent of tables)
    if generate_plots:
        sns.set(style='darkgrid')

        def plot_installed_capacities(P_invest, Max_capacities):
            capacities = {g: P_invest[g].x for g in P_invest.keys() if P_invest[g].x > 0}
            max_capacities = {g: Max_capacities[g] for g in Max_capacities.keys()}

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(capacities.keys(), capacities.values(), label='Installed Capacity', alpha=0.6)
            ax.bar(max_capacities.keys(), max_capacities.values(), label='Max Capacity', alpha=0.3)
            ax.set_xlabel('Technology')
            ax.set_ylabel('Capacity (MW or tons)')
            ax.set_title('Installed Capacities vs Maximum Capacities')
            ax.legend()
            plt.show()

        def plot_energy_flows(T, el_prod, x_in, x_out, elec_sold):
            time_range = range(len(T))
            wind_prod = [el_prod['wind', t].x for t in time_range]
            solar_prod = [el_prod['solar', t].x for t in time_range]
            battery_charge = [x_in['battery', t].x for t in time_range]
            battery_discharge = [x_out['battery', t].x for t in time_range]
            elec_sales = [elec_sold[t].x for t in time_range]

            plt.figure(figsize=(12, 6))
            plt.plot(time_range, wind_prod, label='Wind Production')
            plt.plot(time_range, solar_prod, label='Solar Production')
            plt.plot(time_range, battery_charge, label='Battery Charging')
            plt.plot(time_range, battery_discharge, label='Battery Discharging')
            plt.plot(time_range, elec_sales, label='Electricity Sold to Grid')
            plt.xlabel('Time (hours)')
            plt.ylabel('Energy (MW)')
            plt.title('Energy Flows over Time')
            plt.legend()
            plt.grid(True)
            plt.show()

        def plot_hydrogen_storage_and_production(T, soc, x_out, h2_sold, h2_stored, h2_from_storage):
            time_range = range(len(T))
            hydrogen_prod = [x_out['electrolyzer', t].x for t in time_range]
            hydrogen_storage_level = [soc['H2_storage', t].x for t in time_range]
            hydrogen_sold = [h2_sold[t].x for t in time_range]
            hydrogen_stored = [h2_stored[t].x for t in time_range]
            hydrogen_from_storage = [h2_from_storage[t].x for t in time_range]

            plt.figure(figsize=(12, 6))
            plt.plot(time_range, hydrogen_prod, label='Hydrogen Production')
            plt.plot(time_range, hydrogen_storage_level, label='Hydrogen Storage Level')
            plt.plot(time_range, hydrogen_sold, label='Hydrogen Sold')
            plt.plot(time_range, hydrogen_stored, label='Hydrogen Stored')
            plt.plot(time_range, hydrogen_from_storage, label='Hydrogen From Storage')
            plt.xlabel('Time (hours)')
            plt.ylabel('Hydrogen (tons)')
            plt.title('Hydrogen Production, Storage, and Sales')
            plt.legend()
            plt.grid(True)
            plt.show()

        # Call the plot functions
        plot_installed_capacities(P_invest, Max_capacities)
        plot_energy_flows(T, el_prod, x_in, x_out, elec_sold)
        plot_hydrogen_storage_and_production(T, soc, x_out, h2_sold, h2_stored, h2_from_storage)

    # Functions for tables (independent of plots)
    if generate_tables:

        def table_installed_capacities(P_invest, Max_capacities):
            capacities = {g: P_invest[g].x for g in P_invest.keys() if P_invest[g].x > 0}

            print("\nInstalled Capacities vs Maximum Capacities:")
            print(f"{'Technology':<15} {'Installed Capacity (MW or tons)':<30} {'Max Capacity (MW or tons)':<30}")
            for tech in capacities.keys():
                max_cap = Max_capacities.get(tech,
                                             'No Limit')  # Default to 'No Limit' if the tech doesn't have an upper bound
                print(f"{tech:<15} {capacities[tech]:<30.2f} {max_cap:<30}")


        def table_energy_flows(T, el_prod, x_in, x_out, elec_sold):
            time_range = range(len(T))
            wind_prod = [el_prod['wind', t].x for t in time_range]
            solar_prod = [el_prod['solar', t].x for t in time_range]
            battery_charge = [x_in['battery', t].x for t in time_range]
            battery_discharge = [x_out['battery', t].x for t in time_range]
            elec_sales = [elec_sold[t].x for t in time_range]

            print("\nEnergy Flows over Time:")
            print(f"{'Hour':<10} {'Wind Production (MW)':<25} {'Solar Production (MW)':<25} "
                  f"{'Battery Charge (MW)':<25} {'Battery Discharge (MW)':<25} {'Electricity Sold (MW)':<25}")
            for t in time_range:
                print(f"{t:<10} {wind_prod[t]:<25.2f} {solar_prod[t]:<25.2f} "
                      f"{battery_charge[t]:<25.2f} {battery_discharge[t]:<25.2f} {elec_sales[t]:<25.2f}")


        def table_hydrogen_storage_and_production(T, soc, x_out, h2_sold, h2_stored, h2_from_storage):
            time_range = range(len(T))
            hydrogen_prod = [x_out['electrolyzer', t].x for t in time_range]
            hydrogen_storage_level = [soc['H2_storage', t].x for t in time_range]
            hydrogen_sold = [h2_sold[t].x for t in time_range]
            hydrogen_stored = [h2_stored[t].x for t in time_range]
            hydrogen_from_storage = [h2_from_storage[t].x for t in time_range]

            print("\nHydrogen Production, Storage, and Sales:")
            print(f"{'Hour':<10} {'H2 Production (tons)':<25} {'H2 Storage Level (tons)':<25} "
                  f"{'H2 Sold (tons)':<25} {'H2 Stored (tons)':<25} {'H2 From Storage (tons)':<25}")
            for t in time_range:
                print(f"{t:<10} {hydrogen_prod[t]:<25.2f} {hydrogen_storage_level[t]:<25.2f} "
                      f"{hydrogen_sold[t]:<25.2f} {hydrogen_stored[t]:<25.2f} {hydrogen_from_storage[t]:<25.2f}")


        # Call the table functions
        table_installed_capacities(P_invest, Max_capacities)
        table_energy_flows(T, el_prod, x_in, x_out, elec_sold)
        table_hydrogen_storage_and_production(T, soc, x_out, h2_sold, h2_stored, h2_from_storage)


def run_model(global_params, time_varying_scenario):
    """
    Function to run the optimization model based on given global parameters and time-varying scenarios.

    Parameters:
    global_params (dict): Dictionary containing global parameters (CAPEX, OPEX, efficiency, etc.).
    time_varying_scenario (dict): Dictionary containing time-varying data (electricity prices, capacity factors).

    Returns:
    dict: Model results such as installed capacities, costs, hydrogen production, etc.
    """

    # Extract the relevant data from the global parameters
    CAPEX = global_params['CAPEX']
    OPEX = global_params['OPEX']
    efficiency_electrolyzer = global_params['efficiency_electrolyzer']
    demand = global_params['total_hydrogen_demand']

    # Load the time-varying parameters (e.g., wind and solar capacity factors, electricity prices)
    el_prices = time_varying_scenario['el_prices']
    cf_wind = time_varying_scenario['cf_wind']
    cf_solar = time_varying_scenario['cf_solar']

    # Now, you can set up the optimization model here
    # Use CAPEX, OPEX, efficiency, el_prices, cf_wind, and cf_solar to run the optimization

    model = gp.Model("HydrogenAndBatteryInvestment")

    # ... (Define variables, constraints, and objective as before)

    model.optimize()

    if model.status == GRB.OPTIMAL:
        # Extract results and return
        installed_capacities = {g: P_invest[g].x for g in P_invest.keys() if P_invest[g].x > 0}
        hydrogen_production = [x_out['electrolyzer', t].x for t in T]
        hydrogen_sales = [h2_sold[t].x for t in T]

        return {
            'installed_capacities': installed_capacities,
            'hydrogen_production': hydrogen_production,
            'hydrogen_sales': hydrogen_sales,
        }
    else:
        return None
