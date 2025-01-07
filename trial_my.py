from docplex.mp.model import Model
from SALib import ProblemSpec
from gekko import GEKKO
import numpy as np
import SALib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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
    return ((1 - (1 + r) ** -n) / r) * c
CAPEX_wind = 1291
CAPEX_solar = 726
CAPEX_electrolyser = 750

AF_wind = calculate_annuity_factor(0.07, 20, CAPEX_wind)/(8760-24)
print(AF_wind)

AF_solar = calculate_annuity_factor(0.07, 20, CAPEX_solar)/(8760-24)
AF_electrolyser = calculate_annuity_factor(0.07, 20, CAPEX_electrolyser)/(8760-24)



time_interval_start = 0
time_interval_stop = 23

execute_model = True
df_2 = pd.read_excel(r'C:\Users\Mika\Desktop\Master\Master_data.xlsx', header = 0, sheet_name='Electrolyser')

# Import Data from the Saxony-Anhalt report
path_SA_report = r'C:\Users\Mika\Desktop\Master\20240119_Strukturen.xlsx'
cf_wind = pd.read_excel(path_SA_report, header = 0, sheet_name='EE_Ganglinien', usecols="H").values
cf_wind = [cf_wind[i][0] for i in range(len(cf_wind))]
cf_pv = pd.read_excel(path_SA_report, header = 0, sheet_name='EE_Ganglinien', usecols="C").values
cf_pv = [cf_pv[i][0] for i in range(len(cf_pv))]
# Import Data for Market price
df_el = pd.read_excel(r'C:\Users\Mika\Desktop\Master\Master_data.xlsx', header = 0, sheet_name='Day_ahead_Germany').values
x_el_2023 = [df_el[i][0] for i in range(len(df_el))]
time = [i for i in range(len(x_el_2023))]
plt.plot(time, cf_pv)
plt.plot(time, cf_wind)
#plt.plot(time, x_el_2023)
plt.show()

SOC_min =0 #MWh
SOC_max = 100 #MWh
CAPEX_Bat_C = 0
bat_ratio = 1

def hydrogen_model(day_ahead_price:list, eta_elecrolyser:float, h2_revenue:float, P_solar = cf_pv, P_wind = cf_wind, CAPEX_wind=AF_wind, CAPEX_solar=AF_solar,
                   CAPEX_electrolyser = AF_electrolyser, Bat_cap_min=SOC_min,
                   Bat_cap_max=SOC_max, bat_ratio = bat_ratio, CAPEX_Bat_C = CAPEX_Bat_C,
                   Int_a = time_interval_start, Int_b =time_interval_stop) -> float:
    with_storage = False
    day_ahead_price = day_ahead_price[Int_a: Int_b]
    P_solar = cf_pv[Int_a: Int_b]
    P_wind = cf_pv[Int_a: Int_b]

    time = [i for i in range(len(P_solar))]

    # Declaring the model
    m = Model(name='first_model')
    m.set_time_limit(20)

    # Declaring capacity variable for WIND to be installed
    C_wind = m.integer_var(name="Capacity wind", lb=0, ub=10000)
    # Declaring capacity variable for SOLAR to be installed
    C_pv = m.integer_var(name="Capacity solar", lb=0, ub=30000)
    # Declaring capacity variable for ELECTROLYSER to be installed
    #C_electrolyser = m.integer_var(name="Capacity electrolyser", lb=0, ub=1000)
    C_electrolyser = 100
    # Declaring Variables for hydrogen production and electricity sold in every time step
    x_el_sold = m.continuous_var_dict(time, name ='x_el_sold_')
    x_h2 = m.continuous_var_dict(time, name ='x_h2_')
    Power_balance =m.continuous_var_dict(time, name ='x_h2_')

    # Battery variables:
    C_Bat = m.integer_var(name="Capacity battery [kWh]", lb=Bat_cap_min, ub=Bat_cap_max)
    Bat_SOC = m.continuous_var_dict(time, name ='x_Bat_SOC_')
    Bat_ch = m.continuous_var_dict(time, name ='x_Bat_ch_')
    Bat_disch = m.continuous_var_dict(time, name ='x_Bat_disch_')
    Bat_binary = m.binary_var_dict(time, name ='x_Bat_ch_dis')
    bat_power_rating = m.integer_var(name="Power rating battery [kW]")
    #bat_power_rating = SOC_max



    m.add_constraint(bat_power_rating <= C_Bat)
    # Defining time dependent constraints
    for t in time:
        # Amount of hydrogen produced in each hour must be leq than capacity and efficiency
        c0 = m.add_constraint(x_h2[t] <= (eta_elecrolyser * C_electrolyser)*Power_balance[t], ctname ="h2 limit")
        # The electricity available in each hour is the ceiling for electricity sold and the hydrogen produced
        #c2 = m.add_constraint(x_h2[t] + x_el_sold[t] <= P_wind[t] * C_wind + P_solar[t] * C_pv, ctname="energy_used_stays_leq_than_energy_available")
        c2 = m.add_constraint(Power_balance[t] == P_wind[t] * C_wind + P_solar[t] * C_pv - Bat_ch[t] + Bat_disch[t] , ctname="energy_used_stays_leq_than_energy_available")
        c3 = m.add_constraint(Power_balance[t] >= x_h2[t] + x_el_sold[t])

        #BATTERY CONSTRAINTS
        #Discharge is at least 0 or leq power rating (times switch variable for forcing either charge or discharge)
        #c_bat_dis_max = m.add_constraint(Bat_disch[t] <= bat_power_rating * (1 - Bat_binary[t]), ctname = 'c_bat_dis_max')
        m.add_constraint(Bat_disch[t] >= 0)
        # Same for charge
        #m.add_constraint(Bat_ch[t] <= bat_power_rating * Bat_binary[t])
        m.add_constraint(Bat_ch[t] >= 0)
        #SOC can vary between storage size limits
        m.add_constraint(Bat_SOC[t] >= 0)
        m.add_constraint(Bat_SOC[t] <= C_Bat)
        m.add_indicator(Bat_binary[t], Bat_disch[t] <= bat_power_rating, active_value=0)
        m.add_indicator(Bat_binary[t], Bat_ch[t] <= bat_power_rating, active_value=1)

        for i in range(len(time)):
            if i == 0:
                #m.add_constraint(Bat_SOC[time[0]] == Bat_SOC[time[len(time) - 1]] + Bat_ch[time[0]] - Bat_binary[time[0]])
                m.add_constraint(Bat_SOC[time[0]] == C_Bat/2)
                m.add_constraint(Bat_disch[i] == 0)
            else:
                t = time[i]
                tm = time[i - 1]
                m.add_constraint(Bat_SOC[t] == Bat_SOC[tm] + Bat_ch[t] - Bat_disch[t])





                #m.print_information()

        # define the objective function: Maximizing production value
        obj_fct = m.sum(x_el_sold[t] * day_ahead_price[t] for t in time) - (CAPEX_wind * C_wind) - (
                    CAPEX_solar * C_pv) - (CAPEX_electrolyser * C_electrolyser) \
                  - (CAPEX_Bat_C * C_Bat) + m.sum(x_h2[t] * h2_revenue for t in time)

        m.maximize(obj_fct)

    solution = m.solve(log_output=True)
    solution.display()
    x_el_available = [(P_wind[t] * solution.get_value(C_wind) + (P_solar[t]) * solution.get_value(C_pv)) for t in time]
    #Create Dataframe for H2 produced and el sold
    df = pd.DataFrame({'time': time,
                       'Market_price': day_ahead_price,
                       'Electrcitiy_available': x_el_available,
                       'El_sold': solution.get_value_list([x_el_sold[i] for i in range(len(x_el_sold))]),
                       'H2_prod': solution.get_value_list([x_h2[i] for i in range(len(x_h2))]),
                       'Bat_SOC': solution.get_value_list([Bat_SOC[i] for i in range(len(Bat_SOC))])})
    df_bat = pd.DataFrame({'time': time,
                       'Market_price': day_ahead_price,
                       'Electrcitiy_available': x_el_available,
                       'Discharge': solution.get_value_list([Bat_disch[i] for i in range(len(Bat_disch))]),
                       'Charge': solution.get_value_list([Bat_ch[i] for i in range(len(Bat_ch))]),
                       'Bat_SOC': solution.get_value_list([Bat_SOC[i] for i in range(len(Bat_SOC))])})
    print(df_bat)
    #df_capacity = pd.DataFrame({
    #'Technology': ['Solar', 'Wind', 'Electrolyser', 'Battery', 'Battery power rating [kW]'],
    #'Capacity [kW]': [solution.get_value(C_pv), solution.get_value(C_wind), solution.get_value(C_electrolyser), solution.get_value(C_Bat), solution.get_value(bat_power_rating)]

    #})
    #Plotting
    #cap_plot(df_capacity)
    plot_operation_vs_time(df, 0, 23)
    #plot_battery_vs(df, 0, 23)


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

    obj = hydrogen_model(x_el_2023, 1, 100)
#print(obj)



