import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import integrate
from sklearn.linear_model import LinearRegression
from scipy import stats


def display_force_distance():
    cwd = os.getcwd()  # Gets current working directory.

    # This is a list so you can pass multiple csv files to be overlayed on the same plot.
    dataframes = ["/polyisoprene_trajectory_t=300_6.csv"]

    colours = ['black', 'red', 'blue', 'green', 'yellow']  # Array of colours for the lines.

    for count, df in enumerate(dataframes):
        df1 = pd.read_csv(cwd + df)  # reads file into a dataframe.

        index_to_regress = 5 # Number of data points in the linear region
        dist = np.array(df1['average distance'])
        force = np.array(df1['average force'])
        std = np.array(df1["ecf errors"])

        dist = dist.reshape((-1, 1))
        reg = LinearRegression().fit(dist[:index_to_regress], force[:index_to_regress])
        m = reg.coef_
        c = reg.intercept_
        x = np.linspace(6, 26, 10)
        y = m*x + c

        plt.plot(df1['average distance'], df1['average force'], linestyle=' ', color=colours[count], marker='o',
                 markerfacecolor=colours[count], markeredgecolor=colours[count], markersize=5, linewidth=0)
        plt.errorbar(dist, force, yerr=std, fmt=' ', ecolor='grey', elinewidth=1.5, capsize=5, capthick=1.5)

        plt.plot(x[:index_to_regress], y[:index_to_regress], color=colours[count], linewidth=1)

        plt.ylabel('Force [$eV/\AA$]')
        plt.xlabel('Distance between two atoms [$\AA$]')

    plt.legend(['T = 300 $K$', 'T = 300 $K$', 'T = 600 $K$', 'T = 600 $K$'])
    plt.show()

    display_energy_distance()


def display_energy_distance():
    cwd = os.getcwd()  # Gets current working directory.

    # This is a list so you can pass multiple csv files to be overlayed on the same plot.
    dataframes = ["/polyisoprene_trajectory_t=300_6.csv", "/polyisoprene_trajectory_t=300_4.csv"]

    colours = ['black', 'red', 'blue', 'green', 'yellow']  # Array of colours for the lines.

    for count, df in enumerate(dataframes):
        df1 = pd.read_csv(cwd + df)  # reads file into a dataframe.

        plt.plot(df1['average distance'], df1['average energy'], linestyle=' ', color=colours[count], marker='o',
                 markerfacecolor=colours[count], markeredgecolor=colours[count], markersize=5, linewidth=0)

        plt.ylabel('Internal energy [$eV$]')
        plt.xlabel('Distance between two atoms [$\AA$]')

    plt.legend(['T = 300 $K$', 'T = 600 $K$'])
    plt.show()


def fun_1():
    data = ["/polyisoprene_trajectory_t=300_10.csv", "/polyisoprene_trajectory_t=600_10.csv"]
    cwd = os.getcwd()  # Gets current working directory.

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, constrained_layout=True)
    ax1.set_box_aspect(0.35)
    ax2.set_box_aspect(0.35)
    plt.rc('legend', fontsize=12)

    colours = ["black", "green"]

    for count, csv in enumerate(data):
        df1 = pd.read_csv(cwd + csv)  # reads file into a dataframe.

        index_to_regress = [4, 8] # Number of data points in the linear region
        dist = np.array(df1['average distance'])
        force = np.array(df1['average force'])
        energy = np.array(df1["average energy"])
        std = np.array(df1["ecf errors"])
        std_e = np.array(df1["acf energy"])

        print(dist[index_to_regress[0]:index_to_regress[1]])

        dist = dist.reshape((-1, 1))
        reg = LinearRegression().fit(dist[index_to_regress[0]:index_to_regress[1]], force[index_to_regress[0]:index_to_regress[1]])
        m = reg.coef_
        c = reg.intercept_
        x = np.linspace(6.37, 33.37, 10)
        y = m*x + c
        print(csv, " m: ", m, " +c: ", c)


        ax2.plot(df1['average distance'], df1['average force'], linestyle=' ', color=colours[count], marker='o',
                 markerfacecolor=colours[count], markeredgecolor=colours[count], markersize=5, linewidth=0)
        ax2.errorbar(dist, force, yerr=std, fmt=' ', ecolor=colours[count], elinewidth=1.5, capsize=5, capthick=1.5, label='_nolegend_')
        ax2.plot(x[index_to_regress[0]:index_to_regress[1]], y[index_to_regress[0]:index_to_regress[1]], color=colours[count], linewidth=1, label='_nolegend_')
        #ax2.plot(x[index_to_regress-1:], y[index_to_regress-1:], color=colours[count], linewidth=1, label='_nolegend_', linestyle='--')

        ax2.set_ylabel('Force [$eV/\AA$]')
        ax2.set_xlabel('Distance between two atoms [$\AA$]')

        ax1.plot(df1['average distance'], df1['average energy'], linestyle=' ', color=colours[count], marker='o',
                 markerfacecolor=colours[count], markeredgecolor=colours[count], markersize=5, linewidth=0)
        ax1.errorbar(dist, energy, yerr=std_e, fmt=' ', ecolor=colours[count], elinewidth=1.5, capsize=5, capthick=1.5, label='_nolegend_')

        N = 8
        mean_energy = np.average(df1['average energy'][:N])
        ax1.plot(df1['average distance'][:N], np.zeros(N) + mean_energy, linestyle='-',
                 color="red", linewidth=1, label='_nolegend_')
        ax1.plot(df1['average distance'][N-1:], np.zeros(len(df1['average distance'])-N+1) + mean_energy, linestyle='--',
                 color="red", linewidth=1, label='_nolegend_')

        ax1.set_ylabel('Internal energy [$eV$]')

    ax2.legend(['T = 300 $K$', 'T = 600 $K$'])
    ax1.legend(['T = 300 $K$', 'T = 600 $K$'])

    plt.show()

#display_force_distance()
fun_1()
