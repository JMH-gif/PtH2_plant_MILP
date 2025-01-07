from SALib import ProblemSpec
from gekko import GEKKO
import numpy as np
import SALib
import numpy as np
import pandas as pd
df = pd.read_csv(r'C:\Users\Mika\Desktop\Master\Data\dummy_data.csv', sep = ';', header = 0)
P_wind = df['P_wind [kW]']
P_solar = df['P_solar [kW]']
h2_demand = np.ones(24)

print(P_solar[1])

CAPEX_wind = 1291*7200
CAPEX_solar = 726*1000



def hydrogen_model(eta_elecrolyser:float, h2_revenue:float, P_solar=P_solar[1], P_wind=P_wind[1], CAPEX_wind=CAPEX_wind, CAPEX_solar=CAPEX_solar) -> float:

    m = GEKKO(remote=False)

    #x_h2 = m.Var(lb=0, ub=5)
    CAP_wind = m.Var(lb=0, ub=10)
    CAP_pv = m.Var(lb=0, ub=10)

    x_el_sold = m.Var(lb=0)
    m.Equation(x_el_sold <= P_wind * CAP_wind + P_solar * CAP_pv)


    eta_elecrolyser = eta_elecrolyser
    h2_revenue = h2_revenue
    eMarket_revenue = 1e88
    print(eMarket_revenue)
    h2_demand = 2
    CAPEX_electrolyser = 10

    # Objective fct: maximize the revenue from plant operation while satisfying H2 production constraint
    m.Maximize(-CAPEX_wind*CAP_wind - CAPEX_solar*CAP_pv + x_el_sold*eMarket_revenue)


    # Amount of H2 produced
    #m.Equation(x_h2 <= eta_elecrolyser * x_el * CAP_electrolyser)
    # Demand of H2 needs to be fullfilled
    #m.Equation(x_h2 >= h2_demand)
    # solve the model
    m.solve(disp=False)
    objective = -m.options.objfcnval
    # print the main results to the console
    print('Optimal cost: ' + str(objective))
    #print('H2: ' + str(x_h2.value[0]))
    print('# Wind turbines: ' + str(CAP_wind.value[0]))
    print('# PV modules: ' + str(CAP_pv.value[0]))
    #print('Electrolyser Capacity: ' + str(CAP_electrolyser.value[0]))
    print('electricity sold: ' + str(x_el_sold.value[0]))

    return objective

hydrogen_model(0.7, 5)



