import ase
from ase.io.trajectory import Trajectory
import numpy as np
import pandas as pd
import os
import statsmodels.tsa.stattools as sts


d = []
f = []
atom_1 = 8
atom_2 = 13

for a in ase.io.read("isoprene_t=300_s=1.xyz", index=":"):
    dr = (a.get_positions()[atom_1, :] - a.get_positions()[atom_2, :])
    r = np.linalg.norm(dr)  # magnitude
    dru = dr / r  # unit vector
    d.append(r)

    # dru is the variable holding the unit vector between these atoms
    f.append((a.get_forces()[atom_1, :] - a.get_forces()[atom_2, :]) @ dru)

print(len(f))
f = f[len(f)//2:]
acf = np.array(sts.acf(f, nlags=len(f)))
acf_df = pd.DataFrame(data=acf, columns=["acf"])
cwd = os.getcwd()  # Gets the current working directory
acf_df.to_csv(cwd + "/ex1_acf_results_t=300_f_4.csv")
