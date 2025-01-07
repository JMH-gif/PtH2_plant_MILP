import gurobipy as gp
from gurobipy import GRB
import platform
import os
import time
import pandas as pd
from model_helper_functions import prepare_parameters_morris, prepare_parameters, calculate_financial_results_from_results, calculate_cost_coverage_constraint_morris
from create_condensed_weeks import get_condensed_data
import config  # Import the config file
from output_generation import (
    plot_installed_capacities, plot_energy_flows, plot_hydrogen_storage_and_production,
    table_installed_capacities, table_energy_flows, table_hydrogen_storage_and_production,
    plot_battery_soc,  # Import the new plot function
    table_energy_requirements,
    plot_energy_requirements,
    table_cost_revenue_overview, plot_cost_revenue_overview,
    plot_full_load_hours,
    table_full_load_hours,
    plot_storage_utilization_and_cycles, plot_energy_flows_and_grid_price)


def setup_model(global_params=None, time_varying_scenario=None, generate_tables=False, generate_plots=False):
    """
        Sets up the energy model, defining the constraints and objectives.

        :param global_params: GSA parameters (optional).
        :param time_varying_scenario: Time-varying parameters for wind/solar capacity factors and grid prices.
        :param generate_tables: Whether to generate result tables.
        :param generate_plots: Whether to generate plots.
        """

    analyse = config.analyse
    include_min_prod_lvl = config.include_min_prod_lvl_electrolyzer
    include_startup_cost_electrolyzer = config.include_startup_cost_electrolyzer
    include_el_grid_connection_fee = config.include_el_grid_connection_fee

    parameters = prepare_parameters_morris(config, global_params, time_varying_scenario)
    #parameters = prepare_parameters(config, global_params, time_varying_scenario)

    # Unpack parameters
    CAPEX = parameters["CAPEX"]
    OPEX = parameters["OPEX"]
    Eff = parameters["Eff"]
    Eff_electrolyzer = parameters["Eff_electrolyzer"]
    min_prod_lvl_electrolyzer = parameters["min_prod_lvl_electrolyzer"]
    startup_cost_electrolyzer = parameters["startup_cost_electrolyzer"]
    water_price = parameters["water_price"]
    water_demand = parameters["water_demand"]
    o2_price = parameters["o2_price"]
    total_hydrogen_demand = parameters["total_hydrogen_demand"]
    hourly_hydrogen_demand = parameters["hourly_hydrogen_demand"]
    el_grid_connection_fee = parameters["el_grid_connection_fee"]
    usage_fee = parameters["usage_fee"]
    gas_grid_connection_fee = parameters["gas_grid_connection_fee"]
    heat_price = parameters["heat_price"]
    Elec_Required_Per_Ton = parameters["Elec_Required_Per_Ton"]
    wind_cf = parameters["wind_cf"]
    solar_cf = parameters["solar_cf"]
    grid_price = parameters["grid_price"]
    Annuity_factors = parameters["Annuity_factors"]
    Annuity_factor_grid = parameters["Annuity_factor_grid"]
    Max_capacities = parameters["Max_capacities"]
    scaling_factor = parameters["scaling_factor"]
    T = parameters["T"]


    # Define sets
    S = ['wind', 'solar', 'battery', 'electrolyzer', 'H2_storage']
    S_el = ['wind', 'solar']
    S_in_out = ['battery', 'electrolyzer']
    # Initialize and set up the Gurobi model
    model = gp.Model("HydrogenAndBatteryInvestment")

    # Add decision variables for new capacities with upper bounds (ceilings)
    P_invest = model.addVars(S, ub=[Max_capacities.get(g, GRB.INFINITY) for g in S], name="P_invest")
    # Add decision variable for electricity sales to the grid
    elec_sold = model.addVars(T, name="elec_sold")
    # Add decision variable for the hydrogen selling price
    hydrogen_price = model.addVar(name="hydrogen_price")
    # Modify decision variables
    el_prod = model.addVars(S_el, T, name="el_prod")
    x_in = model.addVars(S_in_out, T, name="x_in")
    x_out = model.addVars(S_in_out, T, name="x_out")
    soc = model.addVars(S, T, name="soc")
    u = model.addVars(S, T, vtype=GRB.BINARY, name="u")
    u_charge = model.addVars(T, vtype=GRB.BINARY, name="u_charge")
    u_discharge = model.addVars(T, vtype=GRB.BINARY, name="u_discharge")
    h2_sold = model.addVars(T, name="h2_sold")
    h2_stored = model.addVars(T, name="h2_stored")
    h2_from_storage = model.addVars(T, name="h2_from_storage")
    # Create the auxiliary variable
    aux_battery_discharge_cap = model.addVars(T, name="aux_battery_discharge_cap")
    # Create an auxiliary variable for charging capacity
    aux_battery_charge_cap = model.addVars(T, name="aux_battery_charge_cap")
    # Add a variable for the initial state of charge (SOC) for hydrogen storage
    soc_initial_H2 = model.addVar(lb=0, name="InitialSOC_H2Storage")
    # Add a variable for the initial state of charge (SOC) for the battery
    soc_initial_battery = model.addVar(lb=0, name="InitialSOC_Battery")
    # Add auxiliary variables
    z_min = model.addVars(T, name="z_min")
    z_max = model.addVars(T, name="z_max")

    if include_el_grid_connection_fee:
        # Add decision variable for grid connection capacity
        grid_connection_capacity = model.addVar(lb=0, name="grid_connection_capacity")
        # Electricity sold to the grid must not exceed the grid connection capacity
    else:
        grid_connection_capacity = None

    if include_startup_cost_electrolyzer:
        u_electrolyzer_start = model.addVars(T, vtype=GRB.BINARY, name="u_electrolyzer_start")
    else:
        u_electrolyzer_start = None


    # Objective function: minimize the selling price of hydrogen
    model.setObjective(hydrogen_price, GRB.MINIMIZE)
    # Add a term to the objective to penalize high-cost electricity purchases

    #                                                     CONSTRAINTS

    water_cost = scaling_factor * gp.quicksum(x_out['electrolyzer', t] * water_price * water_demand for t in T)
    o2_revenue = scaling_factor * gp.quicksum(x_out['electrolyzer', t] * o2_price for t in T)
    heat_revenue = scaling_factor * gp.quicksum(x_out['electrolyzer', t] * heat_price for t in T)
    yearly_grid_sale_revenue = scaling_factor * gp.quicksum(elec_sold[t] * grid_price[t] for t in T)
    yearly_hydrogen_revenue = total_hydrogen_demand * hydrogen_price
    yearly_transportation_cost = usage_fee * total_hydrogen_demand
    yearly_CAPEX_cost = gp.quicksum(P_invest[g] * CAPEX[g] * Annuity_factors[g] for g in S)
    yearly_OPEX_fixed = gp.quicksum(OPEX[g] * P_invest[g] for g in S)  # Apply OPEX based on capacity for Battery, Electrolyzer and Hydrogen storage

    calculate_cost_coverage_constraint_morris(T, model,Annuity_factor_grid,yearly_hydrogen_revenue,o2_revenue,heat_revenue,yearly_grid_sale_revenue, yearly_transportation_cost, yearly_OPEX_fixed, yearly_CAPEX_cost,water_cost,startup_cost_electrolyzer,u_electrolyzer_start,grid_connection_capacity,el_grid_connection_fee)

    # Production limits based on new capacity and time-varying capacity factors for wind and solar
    model.addConstrs(
        (el_prod[g, t] == P_invest[g] * wind_cf[t] for g in ['wind'] for t in T),
        name="WindProductionLimit"
    )

    model.addConstrs(
        (el_prod[g, t] == P_invest[g] * solar_cf[t] for g in ['solar'] for t in T),
        name="SolarProductionLimit"
    )

    # Electricity sold to the grid cannot exceed the available supply from solar, wind, and battery
    model.addConstrs(
        elec_sold[t] <= el_prod['solar', t] + el_prod['wind', t] + x_out['battery', t] for t in T
    )
    if include_el_grid_connection_fee:
        model.addConstrs((elec_sold[t] <= grid_connection_capacity for t in T), name="GridConnectionLimit")
    #                                        STORAGE CONSTRAINTS
    #--------------------------BATTERY:
    #
    # Add a constraint to ensure the initial SOC is less than or equal to the invested battery capacity
    model.addConstr(soc_initial_battery <= P_invest['battery'], name="InitialSOC_BatteryWithinCapacity")

   # Link the initial and final SOCs to this variable for the battery
    model.addConstr(soc['battery', len(T) - 1] == soc_initial_battery, name="LinkFinalSOC_Battery")

    # Explicitly update SOC for the first hour to account for the initial SOC
    model.addConstr(
        soc['battery', 0] == soc_initial_battery - Eff['battery']['discharge'] * x_out['battery', 0] + Eff['battery']['charge'] * x_in['battery', 0],
        name="FirstHourSOCUpdate"
    )
    # SOC update for battery storage
    model.addConstrs(
        (soc['battery', t] == soc['battery', t - 1] + Eff['battery']['charge'] * x_in['battery', t] - Eff['battery'][
            'discharge'] * x_out['battery', t]
         for t in T if t > 0),
        name="SOCUpdate"
    )

    # Limit battery storage by the invested capacity in the battery
    model.addConstrs(
        (soc['battery', t] <= P_invest['battery'] for t in T),
        name="BatteryStorageCapacityLimit"
    )

    M = Max_capacities['battery']  # Upper bound on the battery capacity
    # Constraint 1: Limit discharge rate based on the C-rate and binary variable
    model.addConstr(aux_battery_discharge_cap[0] <= soc_initial_battery, name="InitialDischargeCapSOC")
    model.addConstr(aux_battery_charge_cap[0] <= P_invest['battery'] - soc_initial_battery, name="InitialChargeCapSOC")
    model.addConstrs((x_out['battery', t] <= config.c_rate * aux_battery_discharge_cap[t] for t in T),
                     name="C_Rate_DischargeLimit")
    # Constraint 1: Limit charging rate based on the C-rate and binary variable
    model.addConstrs((x_in['battery', t] <= config.c_rate * aux_battery_charge_cap[t] for t in T),
                     name="C_Rate_ChargeLimit")
    # Constraint 2: Define aux_battery_discharge_cap using Big-M to link it to battery capacity and binary variable

    model.addConstrs((aux_battery_discharge_cap[t] <= P_invest['battery'] for t in T), name="BatteryMaxCapacity")
    model.addConstrs((aux_battery_discharge_cap[t] <= M * u_discharge[t] for t in T), name="BinaryControlDischarge")
    model.addConstrs((aux_battery_charge_cap[t] <= M * u_charge[t] for t in T), name="BinaryControlCharge")
    model.addConstrs((aux_battery_charge_cap[t] <= P_invest['battery'] for t in T), name="BatteryMaxCapacity")
    # Prevent simultaneous charging and discharging
    model.addConstrs(
        (u_charge[t] + u_discharge[t] <= 1 for t in T),
        name="NoSimultaneousChargeDischarge"
    )

    #-------------------------------------HYDROGEN STORAGE:

    # Add a constraint to ensure the initial SOC is less than or equal to the invested hydrogen storage capacity
    model.addConstr(soc_initial_H2 <= P_invest['H2_storage'], name="InitialSOC_H2WithinCapacity")

    # Link the initial and final SOCs to this variable for hydrogen storage
    model.addConstr(soc['H2_storage', len(T) - 1] == soc_initial_H2, name="LinkFinalSOC_H2")

    model.addConstr(
        x_out['electrolyzer', 0] + Eff['H2_storage']['discharge'] * h2_from_storage[0] + soc['H2_storage', 0] >=
        h2_sold[0], name="HydrogenBalanceInitial")

    # Update SOC for the first time step
    model.addConstr(
        soc['H2_storage', 0] == soc_initial_H2 + Eff['H2_storage']['storage'] * h2_stored[0] - Eff['H2_storage']['discharge'] * h2_from_storage[0],
        name="InitialHydrogenSOCUpdate")

    # Hydrogen stored in each period considering storage efficiency
    model.addConstrs(
        (soc['H2_storage', t] == soc['H2_storage', t - 1] + Eff['H2_storage']['storage'] * h2_stored[t] - Eff['H2_storage']['discharge'] * h2_from_storage[t]
         for t in T if t > 0),
        name="HydrogenSOCUpdate"
    )

    # Limit hydrogen storage by the invested capacity in H2 storage
    model.addConstrs(
        (soc['H2_storage', t] <= P_invest['H2_storage'] for t in T),
        name="HydrogenStorageCapacityLimit"
    )

    #                                                   HYDROGEN CONSTRAINTS

    # = ------------------------ ELECTROLYZER

    # Hydrogen Production Constraint: Tie hydrogen production to the installed capacity of the electrolyzer and include efficiency
    model.addConstrs(
        (x_out['electrolyzer', t] == x_in['electrolyzer', t] * Eff_electrolyzer / Elec_Required_Per_Ton for t in T),
        name="HydrogenProduction"
    )


    model.addConstrs(
        (x_out['electrolyzer', t] + h2_from_storage[t] - h2_stored[t] == h2_sold[t]
         for t in T),
        name="HydrogenBalance"
    )

    # Hydrogen sold must equal the hourly hydrogen demand
    model.addConstrs(
        (h2_sold[t] == hourly_hydrogen_demand for t in T),
        name="HydrogenDemand"
    )

    if include_startup_cost_electrolyzer:

        model.addConstr(u_electrolyzer_start[0] == u['electrolyzer', 0], name="StartupFirstHour")



        BigM_startup = P_invest['electrolyzer'] * Eff_electrolyzer / Elec_Required_Per_Ton

        model.addConstrs(
            (u_electrolyzer_start[t] == u['electrolyzer', t] - u['electrolyzer', t - 1])
            for t in range(1, len(T))
        )

        # Ensures startup costs are only incurred when the unit starts up
        model.addConstrs(
            (startup_cost_electrolyzer * u_electrolyzer_start[t] <= BigM_startup)
            for t in range(1, len(T))
        )

    if include_min_prod_lvl:
        # Add constraints to define z_min and z_max
        model.addConstrs((z_min[t] <= P_invest['electrolyzer'] for t in T), name="ZMinUpperBound")
        model.addConstrs((z_min[t] <= Max_capacities['electrolyzer'] * u['electrolyzer', t] for t in T),name="ZMinBinaryLink")
        model.addConstrs((z_min[t] >= P_invest['electrolyzer'] - Max_capacities['electrolyzer'] * (1 - u['electrolyzer', t]) for t in T), name="ZMinLowerBound")

        model.addConstrs((z_max[t] <= P_invest['electrolyzer'] for t in T), name="ZMaxUpperBound")
        model.addConstrs((z_max[t] <= Max_capacities['electrolyzer'] * u['electrolyzer', t] for t in T), name="ZMaxBinaryLink")
        model.addConstrs((z_max[t] >= P_invest['electrolyzer'] - Max_capacities['electrolyzer'] * (1 - u['electrolyzer', t]) for t in T), name="ZMaxLowerBound")

        # Update production constraints
        model.addConstrs((x_out['electrolyzer', t] >= (
                    min_prod_lvl_electrolyzer * z_min[t] * Eff_electrolyzer / Elec_Required_Per_Ton) for t in T),
                         name="MinElectrolyzerProductionLinear")
        model.addConstrs((x_out['electrolyzer', t] <= (z_max[t] * Eff_electrolyzer / Elec_Required_Per_Ton) for t in T),
                         name="MaxElectrolyzerProductionLinear")




        model.addConstrs((x_out['electrolyzer', t] <= (z_max[t] * Eff_electrolyzer / Elec_Required_Per_Ton) for t in T), name="MaxElectrolyzerProductionLinear")
    else:
        model.addConstrs((z_max[t] <= P_invest['electrolyzer'] for t in T), name="ZMaxUpperBound")
        model.addConstrs((z_max[t] <= Max_capacities['electrolyzer'] * u['electrolyzer', t] for t in T),
                         name="ZMaxBinaryLink")
        model.addConstrs(
            (z_max[t] >= P_invest['electrolyzer'] - Max_capacities['electrolyzer'] * (1 - u['electrolyzer', t]) for t
             in T), name="ZMaxLowerBound")

        # Maximum production constraint
        model.addConstrs((x_out['electrolyzer', t] <= (z_max[t] * Eff_electrolyzer / Elec_Required_Per_Ton) for t in T),
                         name="MaxElectrolyzerProductionLinear")

    # Electricity balance: generated electricity plus purchased electricity minus consumption is either stored or sold
    model.addConstrs(
        (
            el_prod['wind', t] + el_prod['solar', t] + x_out['battery', t] ==
            x_in['electrolyzer', t] + x_in['battery', t] + elec_sold[t]
            for t in T
        ), name="ElectricityBalance"
    )

    # Model Parameters for analysis
    model.setParam('MIPGap', config.MIPGAP)  # 0.001 is the default

    # Optimize the model
    model.optimize()

    solve_time = model.Runtime
    print(f"Solving time: {solve_time} seconds")

    # Check if the model is feasible
    if model.status == GRB.OPTIMAL:
        if config.go_quadratic:
            if model.NumQConstrs > 0:
                print(f"Model contains {model.NumQConstrs} quadratic constraint(s):")
                qcon_names = model.getAttr("QCName", model.getQConstrs())
                for qcon_name in qcon_names:
                    print(f"Quadratic constraint: {qcon_name}")

        hydrogen_price = hydrogen_price.x  # Or any objective value or variable you want to return
        objective = hydrogen_price

        # Initialize the energy usage dictionary to track hourly usage for each technology
        energy_usage = {
            'wind': [el_prod['wind', t].x for t in T],
            'solar': [el_prod['solar', t].x for t in T],
            'battery_charge': [x_in['battery', t].x for t in T],
            'battery_discharge': [x_out['battery', t].x for t in T],
            'electrolyzer': [x_in['electrolyzer', t].x for t in T],
            'hydrogen_storage_inflow': [h2_stored[t].x for t in T],  # Hydrogen entering storage
            'hydrogen_storage_outflow': [h2_from_storage[t].x for t in T]  # Hydrogen leaving storage
        }


        # If the model succeeds, gather results
        results = {
            'installed_capacities': {g: P_invest[g].x for g in P_invest.keys()},
            'wind_prod': [el_prod['wind', t].x for t in T],
            'solar_prod': [el_prod['solar', t].x for t in T],
            'battery_charge': [x_in['battery', t].x for t in T],
            'battery_discharge': [x_out['battery', t].x for t in T],
            'battery_soc': [soc['battery', t].x for t in T],
            'hydrogen_prod': [x_out['electrolyzer', t].x for t in T],
            'hydrogen_storage_level': [soc['H2_storage', t].x for t in T],
            'hydrogen_sold': [h2_sold[t].x for t in T],
            'hydrogen_stored': [h2_stored[t].x for t in T],
            'hydrogen_from_storage': [h2_from_storage[t].x for t in T],
            'elec_sales': [elec_sold[t].x for t in T],
            'time': T,
            'hydrogen_price': hydrogen_price,
            'grid_price': [grid_price[t] for t in T],
            'energy_usage': energy_usage,
            'gas_grid_connection_fee': gas_grid_connection_fee,
            'heat_price': heat_price,
            'electrolyzer_intake': [x_in['electrolyzer', t].x for t in T],
            'electrolyzer_output': [x_out['electrolyzer', t].x for t in T],
            'battery_max_discharge': P_invest['battery'].x * config.c_rate,
            'electrolyzer_max_intake': P_invest['electrolyzer'].x
        }

        financial_results = calculate_financial_results_from_results(results, OPEX, CAPEX, Annuity_factors, scaling_factor, water_price, water_demand, o2_price, usage_fee)
        if include_startup_cost_electrolyzer:
            startup_cost_term =gp.quicksum(u_electrolyzer_start[t].x * startup_cost_electrolyzer for t in T)
            if isinstance(startup_cost_term, gp.LinExpr):
                startup_cost_term = startup_cost_term.getValue()  # Evaluate after optimization
            financial_results.update({'Start_up_cost': startup_cost_term}) # €

        if include_el_grid_connection_fee:
            el_grid_connection_cost = grid_connection_capacity.x * el_grid_connection_fee * Annuity_factor_grid
            if isinstance(el_grid_connection_cost, gp.LinExpr):
                el_grid_connection_cost= el_grid_connection_fee.getValue()  # Evaluate after optimization
            financial_results.update({'el_grid_connection_cost': el_grid_connection_cost}) # €



        # Parameters for energy requirement and availability calculations
        hydrogen_demand_per_hour = config.Elec_Required_Per_Ton * hourly_hydrogen_demand / Eff_electrolyzer
        wind_max_capacity = Max_capacities['wind']
        solar_max_capacity = Max_capacities['solar']

        # Generate table
        table_installed_capacities(results, Max_capacities, config)
        table_energy_flows(results, config)
        table_hydrogen_storage_and_production(results, config)
        table_cost_revenue_overview(financial_results, config)
        table_full_load_hours(results, config, scaling_factor)
        table_energy_requirements(results, hydrogen_demand_per_hour, wind_max_capacity, solar_max_capacity, wind_cf,
                                  solar_cf, config)

        # Generate plots
        plot_installed_capacities(results, Max_capacities, config, grid_connection_capacity.x)
        plot_energy_flows(results, config)
        plot_hydrogen_storage_and_production(results, config, None)
        plot_battery_soc(results, config)
        plot_cost_revenue_overview(financial_results, config)
        plot_full_load_hours(results, config, scaling_factor)
        plot_energy_requirements(results, hydrogen_demand_per_hour, wind_max_capacity, solar_max_capacity, wind_cf,
                                 solar_cf, config, None)
        plot_storage_utilization_and_cycles(results, config, scaling_factor)
        plot_energy_flows_and_grid_price(results, config)

        return objective

    elif model.status == GRB.INFEASIBLE:
        print("Model could not be solved to optimality for another reason.")
        if analyse:
            print("Model is infeasible; computing IIS")
            model.computeIIS()
            model.write("model.ilp")
            for c in model.getConstrs():
                if c.IISConstr:
                    print(f"Infeasible constraint: {c.constrName}")
        return None

def run_model(global_params, time_varying_scenario):
    """
    Function to run the optimization model based on given global parameters and time-varying scenarios.

    Parameters:
    global_params (dict): Dictionary containing global parameters (CAPEX, OPEX, efficiency, etc.).
    time_varying_scenario (dict): Dictionary containing time-varying data (electricity prices, capacity factors).

    Returns:
    float or None: The objective value (hydrogen price) or None if the model is infeasible.
    """

    # Call setup_model to build and solve the model
    model_results = setup_model(global_params, time_varying_scenario)
    # Check if setup_model returned None (infeasible or failed to solve)
    if model_results is None:
        print("Model infeasible or failed to solve.")
        return None
    else:
        # Return the hydrogen price or other objective result
        return model_results
import time
import pandas as pd

def evaluate_interval(file_path, start_idx=0, end_idx=None):
    """
    Load the dataset and return data for a predefined interval.

    Parameters:
    - file_path (str): Path to the dataset file.
    - start_idx (int): Start index for the interval (inclusive).
    - end_idx (int): End index for the interval (exclusive). If None, reads till the end of the dataset.

    Returns:
    - dict: A dictionary with keys 'cf_wind', 'cf_solar', and 'el_price' containing the data for the interval.
    """
    try:
        # Load the dataset
        original_data = pd.read_excel(file_path)

        # Apply the interval
        filtered_data = original_data.iloc[start_idx:end_idx]

        # Return the data as a dictionary
        return {
            'cf_wind': filtered_data['cf_wind'].tolist(),
            'cf_solar': filtered_data['cf_solar'].tolist(),
            'el_price': filtered_data['el_price'].tolist(),
        }
    except Exception as e:
        print(f"Error loading dataset or applying interval: {e}")
        return None

if __name__ == "__main__":
    # Start the timer
    start_time = time.time()

    # File path
    file_path = r"C:\Users\Mika\Desktop\Master\Data\Hourly_el_price_cf_wind_cf_solar_1.xlsx"

    # Check configuration for condensed weeks
    if config.use_condensed_weeks:
        # Unpack the returned values from create_condensed_weeks
        condensed_data, seasonal_bounds = get_condensed_data(file_path)

        if condensed_data is not None:
            time_varying_scenario = {
                'cf_wind': condensed_data['cf_wind'].tolist(),
                'cf_solar': condensed_data['cf_solar'].tolist(),
                'el_price': condensed_data['el_price'].tolist(),
            }
        else:
            print("Failed to generate condensed data for the model.")
            time_varying_scenario = None
    else:
        # Evaluate the entire year or another interval from the original dataset
        time_varying_scenario = evaluate_interval(file_path, 0, config.time_period_length)

    # Proceed to setup the model if scenario data is available
    if time_varying_scenario is not None:
        setup_model(
            None,
            time_varying_scenario,
            generate_tables=config.generate_tables,
            generate_plots=config.generate_plots
        )

        # Stop the timer after the analysis completes
        end_time = time.time()

        # Calculate the elapsed time
        execution_time = (end_time - start_time) / 60

        # Print or log the execution time
        print(f"Execution time for the model run: {execution_time:.2f} minutes")

    if config.shutdown_after_execution:
        print("Shutting down the system as per configuration...")
        if platform.system() == "Windows":
            os.system("shutdown /s /t 1")  # Shutdown command for Windows
        else:
            print("Shutdown not supported on this operating system.")

    else:
        print("Failed to generate data for the model.")
