from docplex.mp.model import Model
from SALib import ProblemSpec
from gekko import GEKKO
import numpy as np
import SALib
import numpy as np
import pandas as pd

df = pd.read_csv(r'C:\Users\Mika\Desktop\Master\Data\dummy_data.csv', sep=';', header=0)
P_wind = df['P_wind [kW]']
P_solar = df['P_solar [kW]']
h2_demand = np.ones(24)

CAPEX_wind = 1291 * 7200
CAPEX_solar = 726 * 100


def hydrogen_model(day_ahead_price: float, eta_elecrolyser: float, h2_revenue: float, P_solar=P_solar, P_wind=P_wind,
                   CAPEX_wind=CAPEX_wind, CAPEX_solar=CAPEX_solar) -> float:
    time = len(P_solar)
    h2_demand = 2
    CAPEX_electrolyser = 10

    m = Model(name='first_model')
    CAP_wind = m.integer_var(name="Capacity wind", lb=0, ub=10)
    CAP_pv = m.integer_var(name="Capacity solar", lb=0, ub=10)
    x_el_sold = {(i): m.continuous_var(name='x_el_sold_{0}'.format(i)) for i in range(time)}
    x_h2 = {(i): m.continuous_var(name='x_h2_{0}'.format(i)) for i in range(time)}
    for t in range(time):
        c1 = m.add_constraint(x_el_sold[t] + x_h2[t] / eta_elecrolyser <= P_wind[t] * CAP_wind + P_solar[t] * CAP_pv,
                              ctname="energy_used_stays_leq_than_energy_available")
        # c2 = m.add_constraint(x_el_sold[t] <= P_wind[t] * CAP_wind + P_solar[t] * CAP_pv, ctname="const2")

    m.maximize(m.sum(x_el_sold[t] * day_ahead_price for t in range(time)) - CAPEX_wind * CAP_wind - CAPEX_solar * CAP_pv
               + m.sum(x_h2[t] * h2_revenue for t in range(time)))

    m.print_information()
    m.solve().display()
    # m.print_solution()
    return m.objective_value


def wrapped_hydrogen_model(X: np.ndarray, func=hydrogen_model) -> np.ndarray:
    N, D = X.shape

    results = np.empty(N)
    for i in range(N):
        day_ahead_price, eta_elecrolyser, h2_revenue = X[i, :]
        results[i] = func(day_ahead_price, eta_elecrolyser, h2_revenue)

    return results

    # return func(eta_elecrolyser, h2_revenue)
# obj = hydrogen_model(1, 1, 1e11)
# print(obj)



w