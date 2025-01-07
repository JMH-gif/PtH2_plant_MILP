from SALib import ProblemSpec
import matplotlib.pyplot as plt
#from model_definitions import wrapped_hydrogen_model
#from Several_time_steps_CPLEX_trial import wrapped_hydrogen_model
#from Full_time_series_gurobi import wrapped_hydrogen_model
from One_time_step_Gurobi import wrapped_hydrogen_model

SOBOL = True

sp = ProblemSpec({
    'names': ['market_p', 'eff_elec', 'sell_p h2'],
    'bounds': [
        [0, 1000],
        [0.1, 1],
        [0, 100000],

    ],
    'outputs': ['Obj value']
})

############# SOBOL METHOD##########################
if SOBOL == True:
    (
        sp.sample_sobol(64)
        .evaluate(wrapped_hydrogen_model)
        .analyze_sobol()
    )
    total_Si, first_Si, second_Si = sp.to_df()




######################################         METHOD OF MORRIS ########################
(
    sp.sample_morris(100, num_levels=4)
    .evaluate(wrapped_hydrogen_model)
    .analyze_morris(conf_level=0.95,
        print_to_console=True, num_levels=4)
)



X = sp.samples
y = sp.results
S = sp.analysis



print(sp)
#print(first_Si)
#print(second_Si)
"""
axes = sp.plot()
axes[0].set_yscale('log')
fig = plt.gcf()  # get current figure
fig.set_size_inches(10, 4)
plt.tight_layout()
"""


sp.plot()
plt.show()


