import numpy as np
import ase
from ase.build import bulk
from ase.data.pubchem import pubchem_atoms_search
from ase.visualize import view
#
# now import the modules we need to run molecular dynamics
#
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
from ase.md.langevin import Langevin
from ase.io.trajectory import Trajectory
from ase.optimize.precon import PreconLBFGS
from ase.constraints import FixAtoms
from scipy import integrate


# Import modules to store our results
import pandas as pd
import os
import statsmodels.tsa.stattools as sts
import math
import matplotlib.pyplot as plt

# import an energy model
import sys

sys.path.insert(0, "ANI")

# Analysis of the results.
f = []  # Force
d = []  # Distance
e = []  # Internal Energy
fixed_atom_1 = 27
fixed_atom_2 = 29

count = 0
for a in ase.io.read("polyisoprene_trajectory_t=300_6.xyz", index=":"):
    dr = (a.get_positions()[fixed_atom_1, :] - a.get_positions()[fixed_atom_2, :])
    r = np.linalg.norm(dr)  # magnitude
    dru = dr / r  # unit vector

    d.append(r)
    f.append((a.get_forces()[fixed_atom_1, :] - a.get_forces()[fixed_atom_2,
                                                :]) @ dru)  # dru is the variable holding the unit vector between these atoms
    e.append(a.get_potential_energy())

    count += 1

print("Number of force measurements: ", len(f))
print("Number of distance measurements: ", len(d))
print("Number of energy measurements: ", len(e))

count = 1
temp_d = []  # Stores all distances for a set distance
temp_f = []  # Stores all forces for a set distance
temp_e = []  # Stores initernal energy
avr_d = []  # Stores the average distance for a set distance
avr_f = []  # Stores the average force for a set distance
avr_e = []  # Stores the average internal energy
std_f = []  # Stores the standard deviation of forces for a set distance
act_var_f = []
blocks_of_forces = []

# CHANGE THIS !!
num_of_samples_per_stretch = 2000  # This needs to be manually inputed from previous sim.
relax_time = 300  # This is the number of steps we will miss out of the data collection after a strech to allow the system to reach equilibrium
samples = num_of_samples_per_stretch - relax_time # The actual number of samples used

for i in range(len(f)):
    if (count <= num_of_samples_per_stretch):
        if count > relax_time:
            temp_d.append(d[i])
            temp_f.append(f[i])
            temp_e.append(e[i])
    else:
        # We have reached the end of elements at that distance
        temp_d = np.array(temp_d)
        temp_f = np.array(temp_f)
        temp_e = np.array(temp_e)

        # Calculates the average distance and force for that distance and stores in an array to plot
        avr_d.append(np.average(temp_d))
        avr_f.append(np.average(temp_f))
        avr_e.append(np.average(temp_e))
        std_f.append(np.std(temp_f) / math.sqrt(num_of_samples_per_stretch))  # Standard error of the force measurement

        # Add forces block to total
        blocks_of_forces.append(temp_f)

        # Set the temp arrays to blank
        temp_d = []
        temp_f = []
        temp_e = []

        count = 1

    count += 1

# Add in the last one
temp_d = np.array(temp_d)
temp_f = np.array(temp_f)
temp_e = np.array(temp_e)
blocks_of_forces.append(temp_f)

avr_d.append(np.average(temp_d))
avr_f.append(np.average(temp_f))
avr_e.append(np.average(temp_e))
std_f.append(np.std(temp_f) / math.sqrt(num_of_samples_per_stretch))  # Standard error of the force measurement
print(std_f)

# Make our arrays into data frame and save as .csv file to plot later
def save_results(ecf_errors):
    # Make our arrays into data frame and save as .csv file to plot later
    results_array = np.array([avr_d, avr_f, std_f, avr_e, ecf_errors])
    results_array = np.transpose(results_array)
    df = pd.DataFrame(data=results_array, columns=["average distance", "average force", "std force", "average energy",
                                                   "ecf errors"])

    cwd = os.getcwd()  # Gets the current working directory
    df.to_csv(cwd + "/polyisoprene_trajectory_t=300_6.csv", index=False)


def get_sample_forces(blocks_of_forces, stretch_number):
    # 0 is first stretch, 9 is last
    f_block = blocks_of_forces[stretch_number]

    #f_block = f_block[len(f_block)//2:] # Throw away first half of the results
    acf = np.array(sts.acf(f_block, nlags=len(f_block)))
    i = np.linspace(0, len(acf)-1, len(acf))
    acf_array = np.array([i, acf])
    acf_array = np.transpose(acf_array)
    acf_df = pd.DataFrame(data=acf_array, columns=["i", "acf"])
    # cwd = os.getcwd()  # Gets the current working directory
    # acf_df.to_csv(cwd + "/acf_results_t=300_2_1.csv")

    return acf_df

    #results_array = np.array([temp_f])
    #results_array = np.transpose(results_array)
    #df = pd.DataFrame(data=results_array, columns=["force"])

    #cwd = os.getcwd()  # Gets the current working directory
    #df.to_csv(cwd + "/poly_forces.csv")


# Calculate the errors from ACF
acf_errors = []
for i in range(len(blocks_of_forces)):
    df1 = get_sample_forces(blocks_of_forces, i)

    tau_exp_1 = 30
    tau_exp_2 = 15
    tau_exp_3 = 5

    x = np.linspace(0, len(df1['acf'])-1, len(df1['acf']))
    y = np.exp(-x/tau_exp_1)
    y2 = np.exp(-x/tau_exp_2)
    y3 = np.exp(-x/tau_exp_3)

    sample = 200

    if i < 10:
        plt.plot(df1['acf'][:sample])
        plt.plot(y[:sample])
        plt.plot(y2[:sample])
        plt.plot(y3[:sample])

        plt.xlabel("Lag")
        plt.ylabel("ACF")

        plt.legend(["ACF", "$tau$: " + str(tau_exp_1), "$tau$: " + str(tau_exp_2), "$tau$: " + str(tau_exp_3)])

        plt.show()

    m = 5
    acf = df1['acf'][:m]
    index = df1['i'][:m]

    tau_int = integrate.simpson(acf, index)
    while m < 10 * tau_int:
        acf = df1['acf'][:m]
        index = df1['i'][:m]
        tau_int = integrate.simpson(acf, index)
        m += 1

    var_f = np.var(acf)
    acf_errors.append((2*tau_int*var_f/len(acf))**0.5) # sqrt for standard deviatoin
    print("i:", i, "M:", m, "tau", tau_int)

print(acf_errors)

save_results(acf_errors)
#get_sample_forces(blocks_of_forces)
