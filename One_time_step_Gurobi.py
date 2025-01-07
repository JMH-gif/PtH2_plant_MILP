import matplotlib.pyplot as plt
from gurobipy import *
import numpy as np
import pandas as pd
from Plots import cap_plot
from Plots import plot_operation_vs_time

def calculate_annuity_factor(r, n, c):
    """
    Calculate the annuity factor for a technology investment.

    :param r: Discount rate per period
    :param n: Number of periods
    :param c: Cost of technology per unit capacity
    :return: Annuity factor
    """
    return ((1 - (1 + r) ** -n) / r)*c


execute_model = True

CAPEX_wind = 1291 # €/kW
CAPEX_solar = 726 # €/kW
CAPEX_electrolyser = 750 # €/kW    # 39795 kg/h
CAPEX_Battery = 230 # €/kWh

# Import Data from the Saxony-Anhalt report
path_SA_report = r'C:\Users\Mika\Desktop\Master\20240119_Strukturen.xlsx'
cf_wind = pd.read_excel(path_SA_report, header = 0, sheet_name='EE_Ganglinien', usecols="H").values
cf_wind = [cf_wind[i][0] for i in range(len(cf_wind))]
cf_pv = pd.read_excel(path_SA_report, header = 0, sheet_name='EE_Ganglinien', usecols="C").values
cf_pv = [cf_pv[i][0] for i in range(len(cf_pv))]

# Import Data for Market price
df_el = pd.read_excel(r'C:\Users\Mika\Desktop\Master\Master_data.xlsx', header = 0, sheet_name='Day_ahead_Germany').values
x_el_2023 = [df_el[i][0] for i in range(len(df_el))] # €/MWh

h2_demand = 20000 # kg/year

def hydrogen_model(day_ahead_price:float, eta_elecrolyser:float, h2_revenue:float, P_solar = cf_pv, P_wind = cf_wind, CAPEX_wind=CAPEX_wind, CAPEX_solar=CAPEX_solar,
                   CAPEX_electrolyser = CAPEX_electrolyser,
                  ) -> float:

    stats = True
    # Declare the model
    m = Model(name='basic gurobi model')

    # define the objective function
    m.ModelSense = GRB.MAXIMIZE
    plot = True
    day_ahead_price = day_ahead_price
    P_solar = max(cf_pv)
    P_wind = max(cf_wind)
    # The energy density of hydrogen is approximately 33.3 kWh/kg
    el_required_h2 = 33.3
    # Energy required in one hour:
    El_h2 = ((el_required_h2)/eta_elecrolyser)
    h2_demand_for_model = (h2_demand / (365 * 24))

    # Declaring capacity variable for WIND, solar and electrolyser to be installed
    C_wind = m.addVar(lb=0, ub = 200*1000, name='Capacity wind', vtype=GRB.INTEGER)
    C_pv = m.addVar(lb=0, ub = 200*1000, name='Capacity solar', vtype=GRB.INTEGER)
    C_electrolyser = m.addVar(lb=0, ub = 200*1000, name='Capacity electrolyser', vtype=GRB.INTEGER)

    # Declaring Variables for hydrogen production and electricity sold in every time step
    x_el_sold = m.addVar( name ='x_el_sold_', vtype=GRB.CONTINUOUS)
    x_h2 = m.addVar(name ='x_h2_', vtype=GRB.CONTINUOUS)
    x_el_avail = m.addVar(name='x_el_avail', vtype=GRB.CONTINUOUS)

    #define the objective function: Maximizing production value
    obj_fct = x_el_sold * day_ahead_price - (CAPEX_wind * C_wind) - (CAPEX_solar * C_pv) - (CAPEX_electrolyser * C_electrolyser) + x_h2 * h2_revenue

    m.setObjective(obj_fct)



    m.addConstr(x_el_avail == (P_wind * C_wind + P_solar * C_pv))
    # Amount of hydrogen produced in each hour must be leq than capacity and efficiency
    c0 = m.addConstr(x_h2 <= C_electrolyser / El_h2, name ="h2 limit")
    # The electricity available in each hour is the ceiling for electricity sold and the hydrogen produced
    c2 = m.addConstr(El_h2 + x_el_sold <= x_el_avail, name="energy_used_stays_leq_than_energy_available")
    m.addConstr(x_h2 == h2_demand_for_model)
    solution = m.optimize()
    obj = m.getObjective()
    objective_value = obj.getValue()
    if stats == True:
       C_wind = C_wind.getAttr('X')
       C_pv = C_pv.getAttr('X')
       C_electrolyser = C_electrolyser.getAttr('X')

       df_capacity = pd.DataFrame({
           'Technology': ['Solar', 'Wind', 'Electrolyser'],
           'Capacity [kW]': [C_pv, C_wind, C_electrolyser]

       })

       x_h2 = x_h2.getAttr('X')
       df_h2 = pd.DataFrame({
           'H2': ['Produced', 'Needed'],
           'Amount [kg]': [x_h2, h2_demand_for_model]

       })
       print(f"The objective value is: {objective_value:.2f}")
       print(df_capacity, end='\n')
       print(df_h2)

    return objective_value



def wrapped_hydrogen_model(X: np.ndarray, func= hydrogen_model) -> np.ndarray:
    N, D = X.shape

    results = np.empty(N)
    for i in range(N):
        day_ahead_price, eta_elecrolyser, h2_revenue = X[i, :]
        results[i] = func(day_ahead_price, eta_elecrolyser, h2_revenue)

    return results


    #return func(eta_elecrolyser, h2_revenue)

if execute_model == True:

    obj = hydrogen_model(max(x_el_2023), 1, 100000)
#print(obj)






