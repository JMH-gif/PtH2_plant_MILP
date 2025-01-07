from docplex.mp.model import Model
from SALib import ProblemSpec
from gekko import GEKKO
import numpy as np
import SALib
import numpy as np
import pandas as pd
from Plots import cap_plot
from Plots import plot_operation_vs_time


execute_model = True
df = pd.read_csv(r'C:\Users\Mika\Desktop\Master\Data\dummy_data.csv', sep = ';', header = 0)
P_wind = df['P_wind [kW]']
P_solar = df['P_solar [kW]']
h2_demand = np.ones(24)

df_2 = pd.read_excel(r'C:\Users\Mika\Desktop\Master\Master_data.xlsx', header = 0, sheet_name='Electrolyser')
#print(df_2)
CAPEX_wind = 1291
CAPEX_solar = 726
CAPEX_electrolyser = 10

# Import Data from the Saxony-Anhalt report
path_SA_report = r'C:\Users\Mika\Desktop\Master\20240119_Strukturen.xlsx'
cf_wind = pd.read_excel(path_SA_report, header = 0, sheet_name='EE_Ganglinien', usecols="H").values

cf_pv = pd.read_excel(path_SA_report, header = 0, sheet_name='EE_Ganglinien', usecols="C").values

# Import Data for Market price
df_el = pd.read_excel(r'C:\Users\Mika\Desktop\Master\Master_data.xlsx', header = 0, sheet_name='Day_ahead_Germany').values
#print(df_el)
x_el_2023 = [df_el[i][0] for i in range(len(df_el))]
print(x_el_2023)
SOC_min = 0 #MWh
SOC_max = 10 #MWh

def hydrogen_model(day_ahead_price:list, eta_elecrolyser:float, h2_revenue:float, P_solar = cf_pv, P_wind = cf_wind, CAPEX_wind=CAPEX_wind, CAPEX_solar=CAPEX_solar,
                   CAPEX_electrolyser = CAPEX_electrolyser, h2_el_sold_plot=True, capacity_plot=True, Bat_cap_min=SOC_min,
                   Bat_cap_max=SOC_max) -> float:
    plot = True
    time = len(P_solar)
    h2_demand = 2


    time_for_plot = [i for i in range(1,time+1)]

    # Declaring the model
    m = Model(name='first_model')
    # Declaring capacity variable for WIND to be installed
    C_wind = m.integer_var(name="Capacity wind", lb=0, ub=10000)
    # Declaring capacity variable for SOLAR to be installed
    C_pv = m.integer_var(name="Capacity solar", lb=0, ub=30000)
    # Declaring capacity variable for ELECTROLYSER to be installed
    C_electrolyser = m.integer_var(name="Capacity electrolyser", lb=0, ub=1000)
    # Declaring Variables for hydrogen production and electricity sold in every time step
    x_el_sold = {(i): m.continuous_var(name='x_el_sold_{0}'.format(i)) for i in range(time)}
    x_h2 = {(i): m.continuous_var(name='x_h2_{0}'.format(i)) for i in range(time)}

    # Battery variables:
    Bat_SOC = {(i): m.continuous_var(name='SOC_{0}'.format(i)) for i in range(time)}
    Bat_ch = {(i): m.continuous_var(name='Bat_charge_{0}'.format(i)) for i in range(time)}
    Bat_disch = {(i): m.continuous_var(name='Bat_discharge_{0}'.format(i)) for i in range(time)}
    Bat_ch_dis = {(i): m.binary_var(name='Bat_discharge_{0}'.format(i)) for i in range(time)}
    bat_power_rating = m.integer_var(name="Power rating battery [kW]")


    #define the objective function: Maximizing production value
    obj_fct = m.sum(x_el_sold[t] * day_ahead_price[t] for t in
                     range(time)) - CAPEX_wind * C_wind - CAPEX_solar * C_pv - CAPEX_electrolyser * C_electrolyser + m.sum(x_h2[t] * h2_revenue for t in range(time))
    m.maximize(obj_fct)
    # Defining time dependent constraints
    for t in range(time):
        # Amount of hydrogen produced in each hour must be leq than capacity and efficiency
        c0 = m.add_constraint(x_h2[t] <= eta_elecrolyser * C_electrolyser, ctname ="h2 limit")
        # The electricity available in each hour is the ceiling for electricity sold and the hydrogen produced
        c2 = m.add_constraint(x_h2[t] + x_el_sold[t] - Bat_disch[t] + Bat_ch <= P_wind[t][0] * C_wind + (P_solar[t][0]) * C_pv, ctname="energy_used_stays_leq_than_energy_available")

        #BATTERY CONSTRAINTS
        #Discharge is at least 0 or leq power rating (times switch variable for forcing either charge or discharge)
        m.add_constraint(Bat_disch[t] <= bat_power_rating * (1 - Bat_ch_dis[t]))
        m.add_constraint(Bat_disch[t] >= 0)
        # Same for charge
        m.add_constraint(Bat_ch[t] <= bat_power_rating * Bat_ch_dis[t])
        m.add_constraint(Bat_ch[t] >= 0)
        #SOC can vary between storage size limits
        m.add_constraint(Bat_SOC[t] >= Bat_cap_min)
        m.add_constraint(Bat_SOC[t] <= Bat_cap_max)

        if t == 0:
            m.add_constraint(Bat_SOC[t] == (Bat_cap_min+Bat_cap_max)/2)
                             #Bat_SOC[time[-1]] + Bat_ch[time[0]] - Bat_ch_dis[time[0]])
        else:
            m.add_constraint(Bat_SOC[t] == Bat_SOC[t-1] + Bat_ch[t] - Bat_ch_dis[t])



                #m.print_information()
    solution = m.solve(log_output=True)
    solution.display()
    x_el_available = [(P_wind[t][0] * solution.get_value(C_wind) + (P_solar[t][0]) * solution.get_value(C_pv)) for t in range(time)]
    #Create Dataframe for H2 produced and el sold
    df = pd.DataFrame({'time': time_for_plot,
                       'Market_price': day_ahead_price,
                       'Electrcitiy_available': x_el_available,
                       'H2_prod': solution.get_value_list([x_h2[i] for i in range(len(x_h2))]),
                       'El_sold': solution.get_value_list([x_el_sold[i] for i in range(len(x_el_sold))]),

                       })
    print(df)
    df_capacity = pd.DataFrame({
    'Technology': ['Solar', 'Wind', 'Electrolyser'],
    'Capacity [kW]': [solution.get_value(C_pv), solution.get_value(C_wind), solution.get_value(C_electrolyser)]

    })
    #Plotting
    cap_plot(df_capacity)
    plot_operation_vs_time(df, 0, 23)


    return m.objective_value



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



