from SALib import ProblemSpec
from gekko import GEKKO
import numpy as np
import SALib

def hydrogen_model(eta_elecrolyser:float, h2_revenue:float) -> float:

    m = GEKKO(remote=False)

    x_h2 = m.Var(lb=0, ub=5)
    x_wind = m.Var(lb=0, ub=10)
    x_pv = m.Var(lb=0, ub=10)
    CAP_electrolyser = m.Var(lb=0)
    # Available electricity definition
    x_el = x_wind + x_pv

    eta_elecrolyser = eta_elecrolyser
    h2_revenue = h2_revenue
    eMarket_revenue = 1
    h2_demand = 2
    CAPEX_electrolyser = 10

    # Amount of H2 produced
    m.Equation(x_h2 <= eta_elecrolyser * x_el * CAP_electrolyser)
    # Demand of H2 needs to be fullfilled
    m.Equation(x_h2 >= h2_demand)
    # Objective fct: maximize the revenue from plant operation while satisfying H2 production constraint
    m.Maximize(h2_revenue * x_h2 + eMarket_revenue * x_el - CAPEX_electrolyser * CAP_electrolyser)
    # solve the model
    m.solve(disp=False)
    objective = -m.options.objfcnval
    # print the main results to the console
    print('Optimal cost: ' + str(objective))
    print('H2: ' + str(x_h2.value[0]))
    print('Wind: ' + str(x_wind.value[0]))
    print('PV: ' + str(x_pv.value[0]))
    print('Electrolyser Capacity: ' + str(CAP_electrolyser.value[0]))

    return objective

def wrapped_hydrogen_model(X: np.ndarray, func= hydrogen_model) -> np.ndarray:
    N, D = X.shape

    results = np.empty(N)
    for i in range(N):
        eta_elecrolyser, h2_revenue = X[i, :]
        results[i] = func(eta_elecrolyser, h2_revenue)

    return results


    #return func(eta_elecrolyser, h2_revenue)
