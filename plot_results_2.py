import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import integrate
from sklearn.linear_model import LinearRegression


def display_force_distance():
    cwd = os.getcwd()  # Gets current working directory.

    # This is a list so you can pass multiple csv files to be overlayed on the same plot.
    dataframes = ["/polyisoprene_trajectory_t=300_6.csv", "/polyisoprene_trajectory_t=300_4.csv"]

    colours = ['black', 'red', 'blue', 'green', 'yellow']  # Array of colours for the lines.

    for count, df in enumerate(dataframes):
        df1 = pd.read_csv(cwd + df)  # reads file into a dataframe.

        dist = np.array(df1['average distance'])
        force = np.array(df1['average force'])
        std = np.array(df1["ecf errors"])

        dist = dist.reshape((-1, 1))
        reg = LinearRegression().fit(dist, force)
        m = reg.coef_
        c = reg.intercept_
        x = np.linspace(6, 26, 10)
        y = m*x + c

        plt.plot(df1['average distance'], df1['average force'], linestyle=' ', color=colours[count], marker='o',
                 markerfacecolor=colours[count], markeredgecolor=colours[count], markersize=5, linewidth=0)
        plt.errorbar(dist, force, yerr=std, fmt=' ', ecolor='grey', elinewidth=1.5, capsize=5, capthick=1.5)

        plt.plot(x, y, color=colours[count])

        plt.ylabel('Force [$eV/\AA$]')
        plt.xlabel('Distance between two atoms [$\AA$]')

    plt.legend(['T = 300 $K$', 'T = 300 $K$', 'T = 600 $K$', 'T = 600 $K$'])
    plt.show()


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


display_force_distance()
display_energy_distance()