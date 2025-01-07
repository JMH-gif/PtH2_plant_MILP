import gurobipy as gp
from gurobipy import GRB
import pandas as pd

# Create a new model
model = gp.Model("HydrogenAndBatteryInvestmentWithGridSalesAndDemand")

# Define sets (replace with actual values from the model)
G = ['wind', 'solar', 'battery', 'electrolyzer', 'H2_storage']  # Set of technology units
T = list(range(8760))  # Set of time periods (one year of hourly time steps)

# Define parameters (replace with actual values from the model)
C_var = {...}  # Variable operating costs for technologies
F_in = {...}  # Fixed amount of input (e.g., electricity for electrolyzers)
F_out = {'electrolyzer': 1.0}  # Fixed amount of hydrogen output per unit of input
SOC_max = {...}  # Maximum state of charge for hydrogen and battery storage
Investment_cost = {...}  # Investment cost per unit of capacity
Eff_charge = {...}  # Charging efficiency of the battery
Eff_discharge = {...}  # Discharging efficiency of the battery
P_min = {...}  # Minimum operating level for unit commitment technologies
grid_price = pd.Series(...)  # Day-ahead power market prices (8760 values)

# Load capacity factor time series (replace with actual data)
wind_cf = pd.Series(...)  # Capacity factor time series for wind (8760 values)
solar_cf = pd.Series(...)  # Capacity factor time series for solar (8760 values)

# Hydrogen demand calculation
total_hydrogen_demand = 20000  # 20000 tons per year
hourly_hydrogen_demand = total_hydrogen_demand / len(T)  # Equally distributed across all hours

# Add decision variables for new capacities
P_invest = model.addVars(G, name="P_invest")  # Investment in new capacity for each technology

# Add decision variable for electricity sales to the grid
elec_sold = model.addVars(T, name="elec_sold")  # Electricity sold to the grid

# Modify decision variables
x_in = model.addVars(G, T, name="x_in")  # Input electricity consumption (including battery charging)
x_out = model.addVars(G, T, name="x_out")  # Hydrogen production or battery discharging
x_total = model.addVars(G, T, name="x_total")  # Total activity level
soc = model.addVars(G, T, name="soc")  # State of charge for hydrogen and battery storage
u = model.addVars(G, T, vtype=GRB.BINARY, name="u")  # Unit commitment for electrolyzers and battery operation
charge = model.addVars(G, T, name="charge")  # Battery charging
discharge = model.addVars(G, T, name="discharge")  # Battery discharging

# Objective function: maximize profit (including electricity sales) minus investment costs
model.setObjective(
    gp.quicksum(
        x_out[g, t] * F_out[g] - x_in[g, t] * F_in[g] - C_var[g] * x_out[g, t]
        for g in G for t in T
    ) - gp.quicksum(
        P_invest[g] * Investment_cost[g] for g in G
    ) + gp.quicksum(
        elec_sold[t] * grid_price[t] for t in T
    ), GRB.MAXIMIZE
)

# Constraints
# Production limits based on new capacity and time-varying capacity factors
model.addConstrs(
    (x_total['wind', t] <= P_invest['wind'] * wind_cf[t] for t in T),
    name="WindProductionLimit"
)
model.addConstrs(
    (x_total['solar', t] <= P_invest['solar'] * solar_cf[t] for t in T),
    name="SolarProductionLimit"
)

# State of charge constraints for hydrogen and battery storage
model.addConstrs(
    (soc[g, t] <= SOC_max[g] for g in G if g in ['battery', 'H2_storage'] for t in T),
    name="SOCConstraint"
)

# Battery charging/discharging constraints
model.addConstrs(
    (charge['battery', t] <= P_invest['battery'] * Eff_charge['battery'] for t in T),
    name="ChargeLimit"
)
model.addConstrs(
    (discharge['battery', t] <= P_invest['battery'] * Eff_discharge['battery'] for t in T),
    name="DischargeLimit"
)

# SOC update for battery storage
model.addConstrs(
    (soc['battery', t] == soc['battery', t-1] + Eff_charge['battery'] * charge['battery', t] - Eff_discharge['battery'] * discharge['battery', t]
     for t in T if t > 0),
    name="SOCUpdate"
)

# Prevent simultaneous charging and discharging of batteries
model.addConstrs(
    (charge['battery', t] + discharge['battery', t] <= 1 for t in T),
    name="NoSimultaneousChargeDischarge"
)

# Unit commitment constraints for electrolyzers and battery storage
model.addConstrs(
    (u[g, t] * P_min[g] <= x_total[g, t] for g in ['electrolyzer'] for t in T),
    name="UnitCommitmentMin"
)
model.addConstrs(
    (x_total[g, t] <= u[g, t] * P_invest[g] for g in ['electrolyzer'] for t in T),
    name="UnitCommitmentMax"
)

# Electricity balance: ensure that generated electricity minus consumption is either stored or sold
model.addConstrs(
    (
        x_total['wind', t] * wind_cf[t] + x_total['solar', t] * solar_cf[t]
        - x_in['electrolyzer', t] - charge['battery', t] + discharge['battery', t]
        == elec_sold[t]
        for t in T
    ), name="ElectricityBalance"
)

# Hydrogen demand satisfaction
model.addConstrs(
    (gp.quicksum(x_out['electrolyzer', t] for t in T) == total_hydrogen_demand),
    name="TotalHydrogenDemand"
)

# Hourly hydrogen demand satisfaction
model.addConstrs(
    (x_out['electrolyzer', t] >= hourly_hydrogen_demand for t in T),
    name="HourlyHydrogenDemand"
)

# Optimize the model
model.optimize()

# Print the results
if model.status == GRB.OPTIMAL:
    for v in model.getVars():
        print(f'{v.varName}: {v.x}')
    print(f'Optimal objective value: {model.objVal}')
else:
    print("No optimal solution found")
