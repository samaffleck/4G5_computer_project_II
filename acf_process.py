import ase
from ase.io.trajectory import Trajectory
import numpy as np
import pandas as pd
import os
import statsmodels.tsa.stattools as sts
import matplotlib.pyplot as plt


def fun_1():
    data = ["isoprene_t=300_s=10.xyz", "isoprene_t=600_s=10.xyz"]

    fig, axs = plt.subplots(2, 2, constrained_layout=True, sharex=True)
    plt.rc('legend', fontsize=12)

    colours = ["black", "green"]

    for count, xyz in enumerate(data):
        d = []
        f = []
        atom_1 = 8
        atom_2 = 13

        for a in ase.io.read(xyz, index=":"):
            dr = (a.get_positions()[atom_1, :] - a.get_positions()[atom_2, :])
            r = np.linalg.norm(dr)  # magnitude
            dru = dr / r  # unit vector
            d.append(r)

            # dru is the variable holding the unit vector between these atoms
            f.append((a.get_forces()[atom_1, :] - a.get_forces()[atom_2, :]) @ dru)

        print(len(f))
        acf_f = np.array(sts.acf(f, nlags=len(f)))
        i_f = np.linspace(0, len(acf_f) - 1, len(acf_f))

        acf_d = np.array(sts.acf(d, nlags=len(d)))
        i_d = np.linspace(0, len(acf_d) - 1, len(acf_d))

        #acf_array = np.array([i, acf])
        #acf_array = np.transpose(acf_array)
        #acf_df = pd.DataFrame(data=acf_array, columns=["i", "acf"])
        #cwd = os.getcwd()  # Gets the current working directory
        #acf_df.to_csv(cwd + "/ex1_acf_results_t=300_f_4.csv")

        t_step = np.linspace(0, len(f) - 1, len(f))

        axs[0, 0].plot(t_step, f, color=colours[count], linewidth=1.5, alpha=0.75)
        axs[1, 0].plot(t_step, d, color=colours[count], linewidth=1.5, alpha=0.75)

        axs[0, 1].plot(i_f, acf_f, color=colours[count], linewidth=1.5, alpha=0.75)
        axs[1, 1].plot(i_d, acf_d, color=colours[count], linewidth=1.5, alpha=0.75)

        axs[1, 0].set_xlabel("Time step")
        axs[1, 1].set_xlabel("Lag")

        axs[0, 0].set_ylabel("Force [$eV/\AA$]")
        axs[1, 0].set_ylabel("Distance [$\AA$]")
        axs[0, 1].set_ylabel("ACF of force")
        axs[1, 1].set_ylabel("ACF of distance")

        axs[0, 0].set_xlim(0, len(f))
        axs[0, 1].set_xlim(0, len(f))
        axs[1, 1].set_xlim(0, len(f))
        axs[1, 0].set_xlim(0, len(f))

        axs[0, 1].legend(["T = 300 K", "T = 600 K"])

    plt.show()


def fun_2():
    data = ["isoprene_t=300_s=1.xyz", "isoprene_t=600_s=10.xyz"]

    plt.rc('legend', fontsize=12)

    colours = ["black", "green"]

    for count, xyz in enumerate(data):
        d = []
        f = []
        atom_1 = 8
        atom_2 = 13

        for a in ase.io.read(xyz, index=":"):
            dr = (a.get_positions()[atom_1, :] - a.get_positions()[atom_2, :])
            r = np.linalg.norm(dr)  # magnitude
            dru = dr / r  # unit vector
            d.append(r)

            # dru is the variable holding the unit vector between these atoms
            f.append((a.get_forces()[atom_1, :] - a.get_forces()[atom_2, :]) @ dru)

        print(len(f))
        acf_f = np.array(sts.acf(f, nlags=len(f)))
        i_f = np.linspace(0, len(acf_f) - 1, len(acf_f))

        acf_d = np.array(sts.acf(d, nlags=len(d)))
        i_d = np.linspace(0, len(acf_d) - 1, len(acf_d))

        t_step = np.linspace(0, len(f) - 1, len(f))

        plt.plot(i_f, acf_f, color=colours[count], linewidth=1.5, alpha=0.75)

        plt.xlabel("Lag")

        plt.ylabel("ACF of force")

        plt.xlim(0, 200)

        plt.legend(["Sample frequency = 1", "Sample frequency = 10"])

    plt.show()


fun_2()
