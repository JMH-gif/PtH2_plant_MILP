import pandas as pd
import gurobipy as gp

def prepare_parameters(config, global_params=None, time_varying_scenario=None):
    """
    Prepare and calculate all parameters for the model setup, including static, time-varying,
    and derived parameters.

    :param config: Configuration object containing parameter bounds and other settings.
    :param global_params: Dictionary of GSA sample parameters (optional).
    :param time_varying_scenario: Dictionary of time-varying scenarios for GSA (optional).
    :return: A dictionary containing all computed parameters, time-varying data, and other model setup values.
    """
    # Load general configuration from config file
    time_period_length = config.time_period_length
    hours_per_year = 8760
    scaling_factor = hours_per_year / time_period_length
    T = list(range(time_period_length))
    include_min_prod_lvl = config.include_min_prod_lvl_electrolyzer
    include_startup_cost_electrolyzer = config.include_startup_cost_electrolyzer
    include_el_grid_connection_fee = config.include_el_grid_connection_fee

    # Initialize CAPEX, OPEX, and Efficiencies
    if global_params is None:
        # Standalone execution
        CAPEX = {
            'wind': 1190000,  # From Gutachten SA
            'solar': 455000,  # From Gutachten SA
            'battery': 665000,  # From Gutachten SA
            'electrolyzer': 675000,
            'H2_storage': 5354000
        }

        OPEX = {
            'wind': 1190000*0.02,
            'solar': 455000*0.02 ,
            'battery': 665000*0.02,
            'electrolyzer': 675*0.035,
            'H2_storage':  5354000*0.02
        }

        Eff = {
            'battery': {
                'charge': 0.92,
                'discharge': 0.92
            },
            'H2_storage': {
                'storage': 0.88,
                'discharge': 0.88
            }
        }

        Eff_electrolyzer = 0.7 # [70%]

        if include_min_prod_lvl:
            min_prod_lvl_electrolyzer = 0.2 # [20%]

        if config.include_el_grid_connection_fee:
            el_grid_connection_fee = 107*1000 # [€/MW]

        if config.include_startup_cost_electrolyzer:
            startup_cost_electrolyzer = 2000 #[€]

        water_price =  0.0022 # [€/l]
        water_demand = 14500 # [l/ton]
        o2_price = 0.04*1000 # [€/ton]

        total_hydrogen_demand = 20000 # [t]
        hourly_hydrogen_demand = total_hydrogen_demand / hours_per_year
        usage_fee = 320 # [€/t]
        gas_grid_connection_fee = (config.other_parameters['gas_grid_connection_fee'][0] +
                                   config.other_parameters['gas_grid_connection_fee'][1]) / 2
        Interest_rate = 0.05 #625
        heat_price = (config.other_parameters['heat_price'][0] + config.other_parameters['heat_price'][1]) / 2
    else:
        # GSA execution
        CAPEX = {key: global_params[f'capex_{key}'] for key in ['wind', 'solar', 'battery', 'electrolyzer', 'H2_storage']}
        OPEX = {key: global_params[f'opex_{key}'] for key in ['wind', 'solar', 'battery', 'electrolyzer', 'H2_storage']}

        Eff = {
            'battery': {
                'charge': global_params['eff_battery_charge'],
                'discharge': global_params['eff_battery_discharge']
            },
            'H2_storage': {
                'storage': global_params['eff_h2_storage'],
                'discharge': global_params['eff_h2_storage']
            }
        }

        Eff_electrolyzer = global_params['eff_electrolyzer']

        if include_min_prod_lvl:
            min_prod_lvl_electrolyzer = global_params['min_prod_lvl_electrolyzer']

        if config.include_el_grid_connection_fee:
            el_grid_connection_fee = global_params['el_grid_connection_fee']
        if config.include_startup_cost_electrolyzer:
            startup_cost_electrolyzer = global_params['startup_cost_electrolyzer']

        water_price = global_params['water_price']
        water_demand = global_params['water_demand']
        o2_price = global_params['o2_price']

        total_hydrogen_demand = global_params['total_h2_demand']
        hourly_hydrogen_demand = total_hydrogen_demand / hours_per_year
        gas_grid_connection_fee = global_params['gas_grid_connection_fee']
        usage_fee = global_params['usage_fee']
        Interest_rate = global_params['interest_rate']
        heat_price = global_params['heat_price']

    Elec_Required_Per_Ton = config.Elec_Required_Per_Ton

    # Time-varying parameters
    wind_cf = pd.Series(time_varying_scenario['cf_wind'], index=T)
    solar_cf = pd.Series(time_varying_scenario['cf_solar'], index=T)
    grid_price = pd.Series(time_varying_scenario['el_price'], index=T)

    # Lifetimes and annuity factors
    Lifetimes = config.lifetimes
    Max_capacities = config.max_capacities
    Annuity_factors = {
        tech: (Interest_rate * (1 + Interest_rate) ** Lifetimes[tech]) /
              ((1 + Interest_rate) ** Lifetimes[tech] - 1)
        for tech in Lifetimes
    }

    if include_el_grid_connection_fee:
        Annuity_factor_grid = (Interest_rate * (1 + Interest_rate) ** Lifetimes['el_grid_connection']) /((1 + Interest_rate) ** Lifetimes['el_grid_connection'] - 1)



    # Prepare results
    return {
        "CAPEX": CAPEX,
        "OPEX": OPEX,
        "Eff": Eff,
        "Eff_electrolyzer": Eff_electrolyzer,
        "min_prod_lvl_electrolyzer": min_prod_lvl_electrolyzer if include_min_prod_lvl else None,
        "el_grid_connection_fee": el_grid_connection_fee if include_el_grid_connection_fee else None,
        "startup_cost_electrolyzer": startup_cost_electrolyzer if include_startup_cost_electrolyzer else None,
        "water_price": water_price,
        "water_demand": water_demand,
        "o2_price": o2_price,
        "total_hydrogen_demand": total_hydrogen_demand,
        "hourly_hydrogen_demand": hourly_hydrogen_demand,
        "usage_fee": usage_fee,
        "gas_grid_connection_fee": gas_grid_connection_fee,
        "Interest_rate": Interest_rate,
        "heat_price": heat_price,
        "Elec_Required_Per_Ton": Elec_Required_Per_Ton,
        "wind_cf": wind_cf,
        "solar_cf": solar_cf,
        "grid_price": grid_price,
        "Annuity_factors": Annuity_factors,
        "Annuity_factor_grid": Annuity_factor_grid if include_el_grid_connection_fee else None,
        "Max_capacities": Max_capacities,
        "scaling_factor": scaling_factor,
        "time_varying_scenario": time_varying_scenario,
        "T": T
    }


def prepare_parameters_morris(config, global_params=None, time_varying_scenario=None):
    """
    Prepare and calculate all parameters for the model setup, including static, time-varying,
    and derived parameters.

    :param config: Configuration object containing parameter bounds and other settings.
    :param global_params: Dictionary of GSA sample parameters (optional).
    :param time_varying_scenario: Dictionary of time-varying scenarios for GSA (optional).
    :return: A dictionary containing all computed parameters, time-varying data, and other model setup values.
    """
    # Load general configuration from config file
    time_period_length = config.time_period_length
    hours_per_year = 8760
    scaling_factor = hours_per_year / time_period_length
    T = list(range(time_period_length))
    include_min_prod_lvl = config.include_min_prod_lvl_electrolyzer
    include_startup_cost_electrolyzer = config.include_startup_cost_electrolyzer
    include_el_grid_connection_fee = config.include_el_grid_connection_fee

    # Initialize CAPEX, OPEX, and Efficiencies
    if global_params is None:
        # Standalone execution
        CAPEX = {
            'wind': 1190000,  # From Gutachten SA
            'solar': 455000,  # From Gutachten SA
            'battery': 665000,  # From Gutachten SA
            'electrolyzer': 675000,
            'H2_storage': 5354000
        }

        OPEX = {
            'wind': 1190000*0.02,
            'solar': 455000*0.02 ,
            'battery': 665000*0.02,
            'electrolyzer': 675*0.035,
            'H2_storage':  5354000*0.02
        }

        Eff = {
            'battery': {
                'charge': 0.92,
                'discharge': 0.92
            },
            'H2_storage': {
                'storage': 0.88,
                'discharge': 0.88
            }
        }

        Eff_electrolyzer = 0.7 # [70%]

        if include_min_prod_lvl:
            min_prod_lvl_electrolyzer = 0.2 # [20%]

        if config.include_el_grid_connection_fee:
            el_grid_connection_fee = 107*1000 # [€/MW]

        if config.include_startup_cost_electrolyzer:
            startup_cost_electrolyzer = 2000 #[€]

        water_price =  0.0022 # [€/l]
        water_demand = 14500 # [l/ton]
        o2_price = 0.04*1000 # [€/ton]

        total_hydrogen_demand = 20000 # [kt]
        hourly_hydrogen_demand = total_hydrogen_demand / hours_per_year
        usage_fee = 320 # [€/t]
        gas_grid_connection_fee = (config.other_parameters['gas_grid_connection_fee'][0] +
                                   config.other_parameters['gas_grid_connection_fee'][1]) / 2
        Interest_rate = 0.0625
        heat_price = (config.other_parameters['heat_price'][0] + config.other_parameters['heat_price'][1]) / 2
    else:
        # GSA execution
        CAPEX = {key: global_params[f'capex_{key}'] for key in ['wind', 'solar', 'battery', 'electrolyzer', 'H2_storage']}
        OPEX = {key: global_params[f'opex_{key}'] for key in ['wind', 'solar', 'battery', 'electrolyzer', 'H2_storage']}
        # Use global parameters for GSA
        Lifetimes = {
            key: global_params[f'lifetime_{key}'] for key in config.bounds_lifetimes.keys()
        }
        Max_capacities = {
            key: global_params[f'max_capacity_{key}'] for key in config.bounds_max_capacities.keys()
        }

        Eff = {
            'battery': {
                'charge': global_params['eff_battery_charge'],
                'discharge': global_params['eff_battery_discharge']
            },
            'H2_storage': {
                'storage': global_params['eff_h2_storage'],
                'discharge': global_params['eff_h2_storage']
            }
        }

        Eff_electrolyzer = global_params['eff_electrolyzer']

        if include_min_prod_lvl:
            min_prod_lvl_electrolyzer = global_params['min_prod_lvl_electrolyzer']

        if config.include_el_grid_connection_fee:
            el_grid_connection_fee = global_params['el_grid_connection_fee']
        if config.include_startup_cost_electrolyzer:
            startup_cost_electrolyzer = global_params['startup_cost_electrolyzer']

        water_price = global_params['water_price']
        water_demand = global_params['water_demand']
        o2_price = global_params['o2_price']

        total_hydrogen_demand = global_params['total_h2_demand']
        hourly_hydrogen_demand = total_hydrogen_demand / hours_per_year
        gas_grid_connection_fee = global_params['gas_grid_connection_fee']
        usage_fee = global_params['usage_fee']
        Interest_rate = global_params['interest_rate']
        heat_price = global_params['heat_price']

    Elec_Required_Per_Ton = config.Elec_Required_Per_Ton

    # Time-varying parameters
    wind_cf = pd.Series(time_varying_scenario['cf_wind'], index=T)
    solar_cf = pd.Series(time_varying_scenario['cf_solar'], index=T)
    grid_price = pd.Series(time_varying_scenario['el_price'], index=T)

    Annuity_factors = {
        tech: (Interest_rate * (1 + Interest_rate) ** Lifetimes[tech]) /
              ((1 + Interest_rate) ** Lifetimes[tech] - 1)
        for tech in Lifetimes
    }

    if include_el_grid_connection_fee:
        Annuity_factor_grid = (Interest_rate * (1 + Interest_rate) ** Lifetimes['el_grid_connection']) /((1 + Interest_rate) ** Lifetimes['el_grid_connection'] - 1)


    # Prepare results
    return {
        "CAPEX": CAPEX,
        "OPEX": OPEX,
        "Eff": Eff,
        "Eff_electrolyzer": Eff_electrolyzer,
        "min_prod_lvl_electrolyzer": min_prod_lvl_electrolyzer if include_min_prod_lvl else None,
        "el_grid_connection_fee": el_grid_connection_fee if include_el_grid_connection_fee else None,
        "startup_cost_electrolyzer": startup_cost_electrolyzer if include_startup_cost_electrolyzer else None,
        "water_price": water_price,
        "water_demand": water_demand,
        "o2_price": o2_price,
        "total_hydrogen_demand": total_hydrogen_demand,
        "hourly_hydrogen_demand": hourly_hydrogen_demand,
        "usage_fee": usage_fee,
        "gas_grid_connection_fee": gas_grid_connection_fee,
        "Interest_rate": Interest_rate,
        "heat_price": heat_price,
        "Elec_Required_Per_Ton": Elec_Required_Per_Ton,
        "wind_cf": wind_cf,
        "solar_cf": solar_cf,
        "grid_price": grid_price,
        "Annuity_factors": Annuity_factors,
        "Annuity_factor_grid": Annuity_factor_grid if include_el_grid_connection_fee else None,
        "Max_capacities": Max_capacities,
        "scaling_factor": scaling_factor,
        "time_varying_scenario": time_varying_scenario,
        "Lifetimes": Lifetimes,
        "T": T
    }

def calculate_financial_results_from_results(
        results, OPEX, CAPEX, Annuity_factors, scaling_factor, water_price, water_demand, o2_price, usage_fee):
    """
    Calculate financial parameters including CAPEX, OPEX, and revenues,
    leveraging the existing results dictionary.

    Parameters:
    - results (dict): Dictionary containing model outputs and relevant parameters.
    - OPEX (dict): Operational expenditure rates (€/MW or €/MWh).
    - CAPEX (dict): Capital expenditure rates (€/MW or €/ton).
    - Annuity_factors (dict): Annuity factors by technology.
    - scaling_factor (float): Scaling factor to convert costs to annualized values.
    - water_price (float): Cost of water per m³.
    - water_demand (float): Water demand per ton of hydrogen produced.
    - o2_price (float): Revenue per kg of oxygen produced.

    Returns:
    dict: Dictionary with financial results (OPEX, CAPEX, revenues, costs).
    """
    financial_results = {}

    # Installed capacities
    installed_capacities = results['installed_capacities']

    # OPEX calculations
    for tech, capacity in installed_capacities.items():
        if capacity > 0:  # Only calculate for non-zero capacities
                # OPEX based on installed capacity
                financial_results[f'OPEX_{tech}'] = capacity * OPEX[tech]

    # CAPEX calculations
    for tech, capacity in installed_capacities.items():
        if capacity > 0:  # Only calculate for non-zero capacities
            financial_results[f'CAPEX_{tech}'] = (
                    capacity * CAPEX[tech] * Annuity_factors[tech]
            )

    # Transport costs
    transport_costs = usage_fee * scaling_factor * sum(results['hydrogen_sold'])
    financial_results['Transport'] = transport_costs

    # Revenues
    hydrogen_revenue = scaling_factor * sum(results['hydrogen_sold'][t] * results['hydrogen_price'] for t in range(len(results['hydrogen_sold'])))

    electricity_sales_revenue = scaling_factor * sum(results['elec_sales'][t] * results['grid_price'][t] for t in range(len(results['elec_sales'])))

    heat_revenue = scaling_factor * sum(
        results['hydrogen_prod'][t] * results['heat_price']
        for t in range(len(results['hydrogen_prod']))
    )

    # O₂ Revenue: Calculate from electrolyzer output and O₂ price
    o2_revenue = scaling_factor * sum(
        results['electrolyzer_output'][t] * o2_price for t in range(len(results['electrolyzer_output']))
    )

    # Water cost
    water_cost = scaling_factor * sum(
        results['electrolyzer_output'][t] * water_price * water_demand
        for t in range(len(results['electrolyzer_output']))
    )




    # Add revenues and other financials to the dictionary
    financial_results.update({
        'Hydrogen Revenue': hydrogen_revenue,
        'Electricity Sales Revenue': electricity_sales_revenue,
        'Heat Revenue': heat_revenue,
        'O2 Revenue': o2_revenue,
        'Water Cost': water_cost
    })

    return financial_results

def calculate_cost_coverage_constraint(T, model, Annuity_factor_grid, yearly_hydrogen_revenue, o2_revenue, heat_revenue,
                                       yearly_grid_sale_revenue, yearly_transportation_cost,
                                       yearly_OPEX_fixed, yearly_CAPEX_cost, water_cost, slack_variable_cost,
                                       startup_cost_electrolyzer = None, u_electrolyzer_start = None, grid_connection_capacity = None,
                                       el_grid_connection_fee = None):
    """
    Calculate and apply the cost coverage constraint based on the inclusion of startup cost and grid connection fee.

    Parameters:
        model (gurobipy.Model): The optimization model.
        yearly_hydrogen_revenue (float): Revenue from hydrogen sales.
        o2_revenue (float): Revenue from oxygen sales.
        heat_revenue (float): Revenue from heat sales.
        yearly_grid_sale_revenue (float): Revenue from grid electricity sales.
        yearly_transportation_cost (float): Transportation cost.
        yearly_OPEX_fixed (float): Fixed operational expenses.
        yearly_CAPEX_cost (float): Capital expenses.
        water_cost (float): Cost of water usage.
        slack_variable_cost (float): Cost of slack variables for purchased electricity.
        include_startup_cost (bool): Whether to include startup cost in the constraint.
        startup_cost_electrolyzer (float): Startup cost per event.
        u_electrolyzer_start (gurobipy.Var): Binary variable indicating startup of electrolyzer.
        T (list): List of time periods.
        include_el_grid_connection_fee (bool): Whether to include grid connection fee.
        grid_connection_capacity (gurobipy.Var): Variable for grid connection capacity.
        el_grid_connection_fee (float): CAPEX per MW for grid connection.
        annuity_factor_grid (float): Annuity factor for grid connection fee.
        grid_connection_opex (float): OPEX per MW for grid connection.

    Returns:
        None: Adds the cost coverage constraint to the model.
    """
    # Base cost coverage components
    lhs = yearly_hydrogen_revenue + o2_revenue + heat_revenue + yearly_grid_sale_revenue
    rhs = yearly_transportation_cost + yearly_OPEX_fixed + yearly_CAPEX_cost + water_cost + slack_variable_cost

    # Handle cases based on the inclusion of startup cost and grid connection fee
    if startup_cost_electrolyzer != None and el_grid_connection_fee != None:
        # Both startup cost and grid connection fee included
        startup_cost_term = gp.quicksum(u_electrolyzer_start[t] * startup_cost_electrolyzer for t in T)
        grid_connection_cost = grid_connection_capacity * el_grid_connection_fee * Annuity_factor_grid
        #grid_connection_operational_cost = grid_connection_capacity * grid_connection_opex
        rhs += startup_cost_term + grid_connection_cost #+ grid_connection_operational_cost

    elif startup_cost_electrolyzer != None and el_grid_connection_fee == None:
        # Only startup cost included
        startup_cost_term = gp.quicksum(u_electrolyzer_start[t] * startup_cost_electrolyzer for t in T)
        rhs += startup_cost_term

    elif startup_cost_electrolyzer == None and el_grid_connection_fee != None:
        # Only grid connection fee included
        grid_connection_cost = grid_connection_capacity * el_grid_connection_fee * Annuity_factor_grid
        #grid_connection_operational_cost = grid_connection_capacity * grid_connection_opex
        rhs += grid_connection_cost# + grid_connection_operational_cost

    # Add the cost coverage constraint
    model.addConstr(lhs >= rhs, name="CostCoverage")


def calculate_cost_coverage_constraint_morris(T, model, Annuity_factor_grid, yearly_hydrogen_revenue, o2_revenue, heat_revenue,
                                       yearly_grid_sale_revenue, yearly_transportation_cost,
                                       yearly_OPEX_fixed, yearly_CAPEX_cost, water_cost,
                                       startup_cost_electrolyzer = None, u_electrolyzer_start = None, grid_connection_capacity = None,
                                       el_grid_connection_fee = None):
    """
    Calculate and apply the cost coverage constraint based on the inclusion of startup cost and grid connection fee.

    Parameters:
        model (gurobipy.Model): The optimization model.
        yearly_hydrogen_revenue (float): Revenue from hydrogen sales.
        o2_revenue (float): Revenue from oxygen sales.
        heat_revenue (float): Revenue from heat sales.
        yearly_grid_sale_revenue (float): Revenue from grid electricity sales.
        yearly_transportation_cost (float): Transportation cost.
        yearly_OPEX_fixed (float): Fixed operational expenses.
        yearly_CAPEX_cost (float): Capital expenses.
        water_cost (float): Cost of water usage.
        include_startup_cost (bool): Whether to include startup cost in the constraint.
        startup_cost_electrolyzer (float): Startup cost per event.
        u_electrolyzer_start (gurobipy.Var): Binary variable indicating startup of electrolyzer.
        T (list): List of time periods.
        include_el_grid_connection_fee (bool): Whether to include grid connection fee.
        grid_connection_capacity (gurobipy.Var): Variable for grid connection capacity.
        el_grid_connection_fee (float): CAPEX per MW for grid connection.
        annuity_factor_grid (float): Annuity factor for grid connection fee.
        grid_connection_opex (float): OPEX per MW for grid connection.

    Returns:
        None: Adds the cost coverage constraint to the model.
    """
    # Base cost coverage components
    lhs = yearly_hydrogen_revenue + o2_revenue + heat_revenue + yearly_grid_sale_revenue
    rhs = yearly_transportation_cost + yearly_OPEX_fixed + yearly_CAPEX_cost + water_cost

    # Handle cases based on the inclusion of startup cost and grid connection fee
    if startup_cost_electrolyzer != None and el_grid_connection_fee != None:
        # Both startup cost and grid connection fee included
        startup_cost_term = gp.quicksum(u_electrolyzer_start[t] * startup_cost_electrolyzer for t in T)
        grid_connection_cost = grid_connection_capacity * el_grid_connection_fee * Annuity_factor_grid
        #grid_connection_operational_cost = grid_connection_capacity * grid_connection_opex
        rhs += startup_cost_term + grid_connection_cost #+ grid_connection_operational_cost

    elif startup_cost_electrolyzer != None and el_grid_connection_fee == None:
        # Only startup cost included
        startup_cost_term = gp.quicksum(u_electrolyzer_start[t] * startup_cost_electrolyzer for t in T)
        rhs += startup_cost_term

    elif startup_cost_electrolyzer == None and el_grid_connection_fee != None:
        # Only grid connection fee included
        grid_connection_cost = grid_connection_capacity * el_grid_connection_fee * Annuity_factor_grid
        #grid_connection_operational_cost = grid_connection_capacity * grid_connection_opex
        rhs += grid_connection_cost# + grid_connection_operational_cost

    # Add the cost coverage constraint
    model.addConstr(lhs >= rhs, name="CostCoverage")

