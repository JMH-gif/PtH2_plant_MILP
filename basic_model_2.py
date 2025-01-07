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


CAPEX_wind = 1291*7200
CAPEX_solar = 726*1000
eMarket_revenue = 1e99

m = GEKKO(remote=False)

CAP_wind = m.Var(lb=0, ub=10, integer=True)
CAP_pv = m.Var(lb=0, ub=10, integer=True)
#x_el_sold = m.Var(lb=0)
#m.Equation(x_el_sold <= P_wind[1] * CAP_wind + P_solar[1] * CAP_pv)
x_el = P_wind[1] * CAP_wind + P_solar[1] * CAP_pv
# Objective fct: maximize the revenue from plant operation while satisfying H2 production constraint
#m.Maximize(-CAPEX_wind*CAP_wind - CAPEX_solar*CAP_pv + x_el_sold * eMarket_revenue)
m.Maximize(-CAPEX_wind*CAP_wind - CAPEX_solar*CAP_pv + x_el * eMarket_revenue)
# solve the model
m.solve(disp=False)
objective = -m.options.objfcnval
# print the main results to the console
print('Optimal cost: ' + str(objective))
#print('H2: ' + str(x_h2.value[0]))
print('# Wind turbines: ' + str(CAP_wind.value[0]))
print('# PV modules: ' + str(CAP_pv.value[0]))
#print('electricity sold: ' + str(x_el_sold.value[0]))






