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


time_interval_start = 0
time_interval_stop =  875        # 4 weeks = 671

execute_model = True
df = pd.read_csv(r'C:\Users\Mika\Desktop\Master\Data\dummy_data.csv', sep = ';', header = 0)
P_wind = df['P_wind [kW]']
P_solar = df['P_solar [kW]']


df_2 = pd.read_excel(r'C:\Users\Mika\Desktop\Master\Master_data.xlsx', header = 0, sheet_name='Electrolyser')
#print(df_2)
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

SOC_min =20 #MWh
SOC_max = 100 #MWh

h2_demand = 20000 # t/year



def hydrogen_model(day_ahead_price:list, eta_elecrolyser:float, h2_revenue:float, P_solar = cf_pv, P_wind = cf_wind, CAPEX_wind=CAPEX_wind, CAPEX_solar=CAPEX_solar,
                   CAPEX_electrolyser = CAPEX_electrolyser,
                    Int_a = time_interval_start, Int_b =time_interval_stop, CAPEX_Battery = CAPEX_Battery) -> float:

    # The energy density of hydrogen is approximately 33.3 kWh/kg
    el_required_h2 = 33.3
    # Declare the model
    m = Model(name='basic gurobi model')

    # define the objective function
    m.ModelSense = GRB.MAXIMIZE
    plot = True
    day_ahead_price = day_ahead_price[Int_a: Int_b]
    P_solar = cf_pv[Int_a: Int_b]
    P_wind = cf_wind[Int_a: Int_b]
    time = [i for i in range(len(P_solar))]
    h2_demand_for_model = (h2_demand / 365 * 24) * len(time)
    El_h2 = ((el_required_h2) / eta_elecrolyser)

    # Declaring capacity variable for WIND, solar and electrolyser to be installed
    C_wind = m.addVar(lb=0, ub = 200*1000, name='Capacity wind', vtype=GRB.CONTINUOUS)
    C_pv = m.addVar(lb=0, ub = 200*1000, name='Capacity solar', vtype=GRB.CONTINUOUS)
    C_electrolyser = m.addVar(lb=0, ub = 200*1000, name='Capacity electrolyser', vtype=GRB.CONTINUOUS)
    C_battery = m.addVar(lb=0, ub = 200*1000, name='Capacity battery', vtype=GRB.CONTINUOUS)
    # Declaring Variables for hydrogen production and electricity sold in every time step
    x_el_sold = m.addVars(time, name ='x_el_sold_', vtype=GRB.CONTINUOUS)
    x_h2 = m.addVars(time, name ='x_h2_', vtype=GRB.CONTINUOUS)
    x_el_avail = m.addVars(time, name='x_el_avail', vtype=GRB.CONTINUOUS)
    # Battery variables:
    Bat_SOC = m.addVars(time, name ='x_Bat_SOC_', vtype=GRB.CONTINUOUS)
    Bat_ch = m.addVars(time, name ='x_Bat_ch_', vtype=GRB.CONTINUOUS)
    Bat_disch = m.addVars(time, name ='x_Bat_disch_', vtype=GRB.CONTINUOUS)
    bat_switch = m.addVars(time, name ='x_Bat_ch_dis', vtype=GRB.BINARY)
    bat_power_rating = m.addVar(lb = 0, name ='"Power rating battery [kW]"', vtype=GRB.CONTINUOUS)

    #define the objective function: Maximizing production value
    obj_fct = quicksum(x_el_sold[t] * day_ahead_price[t] for t in time) - (CAPEX_wind * C_wind) - (CAPEX_solar * C_pv) - (CAPEX_electrolyser * C_electrolyser) - (CAPEX_Battery * C_battery) + quicksum(x_h2[t] * h2_revenue for t in time)

    m.setObjective(obj_fct)

    m.addConstr(bat_power_rating == 0.5 * C_battery)
    m.addConstr(quicksum(x_h2[t] for t in time) == h2_demand_for_model)
    # Defining time dependent constraints
    for t in time:
        m.addConstr(x_el_avail[t] == (P_wind[t] * C_wind + P_solar[t] * C_pv) - Bat_ch[t] + Bat_disch[t])
        # Amount of hydrogen produced in each hour must be leq than capacity and efficiency
        c0 = m.addConstr(x_h2[t] <= C_electrolyser / El_h2, name ="h2 limit")
        # The electricity available in each hour is the ceiling for electricity sold and the hydrogen produced
        c2 = m.addConstr(El_h2 + x_el_sold[t] + Bat_ch[t] - Bat_disch[t]  <=x_el_avail[t], name="energy_used_stays_leq_than_energy_available")
        #BATTERY CONSTRAINTS
        #Discharge is at least 0 or leq power rating (times switch variable for forcing either charge or discharge)
        c_bat_dis_max = m.addConstr(Bat_disch[t] <= bat_power_rating * (1 - bat_switch[t]), name = 'c_bat_dis_max')
        m.addConstr(Bat_disch[t] >= 0)
        # Same for charge
        m.addConstr(Bat_ch[t] <= bat_power_rating * bat_switch[t])
        m.addConstr(Bat_ch[t] >= 0)
        #SOC can vary between storage size limits
        m.addConstr(Bat_SOC[t] >= 0)
        m.addConstr(Bat_SOC[t] <= C_battery)



        for i in range(len(time)):
            if i == 0:
                m.addConstr(Bat_SOC[time[0]] == Bat_SOC[time[len(time) - 1]] + Bat_ch[time[0]] - Bat_disch[time[0]])
            else:
                t = time[i]
                tm = time[i - 1]
                m.addConstr(Bat_SOC[t] == Bat_SOC[tm] + Bat_ch[t] - Bat_disch[t])

    solution = m.optimize()
    obj = m.getObjective()
    objective_value = obj.getValue()
    x_h2_df = sum(x_h2[t].getAttr('X') for t in time)
    df_h2 = pd.DataFrame({
        'H2': ['Produced', 'Needed'],
        'Amount [kg]': [x_h2_df, h2_demand_for_model]

    })


    if plot == True:

        C_wind = C_wind.getAttr('X')
        C_pv = C_pv.getAttr('X')
        C_electrolyser = C_electrolyser.getAttr('X')
        C_battery = C_battery.getAttr('X')
        df_capacity = pd.DataFrame({
            'Technology': ['Solar', 'Wind', 'Electrolyser', 'Battery'],
            'Capacity [kW]': [C_pv, C_wind, C_electrolyser, C_battery]

        })
        # Plotting
        cap_plot(df_capacity)

        # Plotting dispatch
        time = pd.Series(time, index=time, name="time")
        x_el_avail = pd.Series(m.getAttr('X',x_el_avail), name="Electrcitiy_available", index=time)
        x_el_h2 = pd.Series(m.getAttr('X', x_h2), name="H2_prod", index=time)
        x_el_sold = pd.Series(m.getAttr('X',x_el_sold), name="El_sold", index=time)
        p_market = pd.Series(day_ahead_price, index = time, name="Market_price")
        df = pd.concat([time, p_market, x_el_h2, x_el_avail, x_el_sold], axis=1)
        plot_operation_vs_time(df, time_interval_start, time_interval_stop)


        Bat_SOC = pd.Series(m.getAttr('X',Bat_SOC), name="Bat_SOC", index=time)
        Bat_ch = pd.Series(m.getAttr('X', Bat_ch), name="Bat_ch", index=time)
        Bat_disch = pd.Series(m.getAttr('X', Bat_disch), name="Bat_disch", index=time)
        sol = pd.concat([Bat_SOC, Bat_ch, Bat_disch], axis=1)
        plt.plot(sol)
        plt.ylabel('Battery State (kWhr)')
        plt.xlabel('Time Period')
        plt.legend(["SOC", "Charge", "Discharge"])
        plt.axhline(y=C_battery, c='blue', linestyle='--', alpha=0.5)
        plt.show()

        print(f"The objective value is: {objective_value:.2f}")
        print(df_capacity, end='\n')
        print(df_h2)
        """
        bat_switch = pd.Series(m.getAttr('X', bat_switch), name="bat_switch", index=time)
        sol = pd.concat([bat_switch], axis=1)
        plt.plot(sol)
        plt.show()
        """


    return m.getObjective()



def wrapped_hydrogen_model(X: np.ndarray, func= hydrogen_model) -> np.ndarray:
    N, D = X.shape

    results = np.empty(N)
    for i in range(N):
        day_ahead_price, eta_elecrolyser, h2_revenue = X[i, :]
        results[i] = func(day_ahead_price, eta_elecrolyser, h2_revenue)

    return results


    #return func(eta_elecrolyser, h2_revenue)

if execute_model == True:

    obj = hydrogen_model(x_el_2023, 1, 1000)
#print(obj)






