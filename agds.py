import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
# change directory
os.chdir(r"C:\Users\Mika\Desktop\Master")

H2_demand = pd.read_excel("Master_data.xlsx", sheet_name=0)
plt.plot(np.linspace(0, len(H2_demand['Zeitz_Basis_2030']), len(H2_demand['Zeitz_Basis_2030'])), H2_demand['Zeitz_Basis_2030'], label = "Basis")
plt.plot(np.linspace(0, len(H2_demand['Zeitz_Cons_20230']), len(H2_demand['Zeitz_Basis_2030'])), H2_demand['Zeitz_Cons_20230'], label = "Cons")
plt.plot(np.linspace(0, len(H2_demand['Zeitz_Prog_2030']), len(H2_demand['Zeitz_Basis_2030'])), H2_demand['Zeitz_Prog_2030'], label = "Prog")
plt.legend()
# show the plot
plt.show()

from  matplotlib import pyplot as plt

I_Err = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
M0Nm_cos_phi = [0.21, 0.35, 0.55, 0.72, 0.93, 0.999, 0.85]
M0Nm_I_1 = [0.373, 0.317, 0.26, 0.204, 0.155, 0.117, 0.117]

M03Nm_cos_phi = [0.87, 0.91, 0.93, 0.96, 0.999, 0.96, 0.94]
M03Nm_I_1 = [0.377, 0.326, 0.274, 0.229, 0.202, 0.191, 0.202]

M06Nm_cos_phi = [0.995, 0.999, 0.999, 0.990, 0.97, 0.96, 0.95]
M06Nm_I_1 = [0.440, 0.378, 0.331, 0.300, 0.287, 0.283, 0.296]


fig, ax = plt.subplots()

ax.plot(I_Err, M0Nm_I_1, color = '#444444', linewidth = 2, linestyle = '--', marker = '.', label = "$M = 0 Nm, I_{1}$")
ax.plot(I_Err, M03Nm_I_1, color = '#5a7d9a', linewidth = 2, linestyle = '-', marker = '.', label = "$M = 0,3 Nm, I_{1}$")
ax.plot(I_Err, M06Nm_I_1, color = '#adad3b', linewidth = 2, linestyle = '-', marker = '.', label = "$M = 0,6 Nm, I_{1}$")
ax.set_xlabel('$I_{Err}/A$')
ax.set_ylabel('$I_{1}/A$')
ax.legend(bbox_to_anchor=(1.1, 1.1), loc="upper left")
ax2 = ax.twinx()
ax2.plot(I_Err, M0Nm_cos_phi, label = "$M = 0 Nm, cos_\Phi}$")
ax2.plot(I_Err, M03Nm_cos_phi, label = "$M = 0,3 Nm, cos_\Phi}$")
ax2.plot(I_Err, M06Nm_cos_phi, label = "$M = 0,6 Nm, cos_\Phi}$")
plt.ylabel('$cos_\Phi$')
plt.tight_layout()
ax2.legend(bbox_to_anchor=(1.1,  0.85), loc="upper left")
#plt.grid(True)
plt.show()