import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import integrate


def display():
    cwd = os.getcwd()  # Gets current working directory.

    # This is a list so you can pass multiple csv files to be overlayed on the same plot.
    dataframes = ["/ex1_distance_time_t=300_2.csv"]

    colours = ['black', 'red', 'blue', 'green', 'yellow']  # Array of colours for the lines.

    # Create plot and formats
    fig, axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    plt.rc('font', size=10)
    lw = 1  # Line width

    for count, df in enumerate(dataframes):
        df1 = pd.read_csv(cwd + df)  # reads file into a dataframe.

        ax1 = df1.plot(linestyle=' ', color=colours[count], marker='o', markerfacecolor=colours[count], markeredgecolor=colours[count],
                       markersize=3, linewidth=0, ax=axes, x='time', y='distance')

        ax1.set_xlim(0, 5000)

        ax1.set_xlabel('Time [$fs$]')
        ax1.set_ylabel('Distance between two atoms [$\AA$]')

        ax1.legend(['T = 100 $K$', 'T = 300 $K$', 'T = 500 $K$'])
        print("std: ", np.std(df1["distance"]))

    plt.show()


def display_acf():
    cwd = os.getcwd()  # Gets current working directory.

    # This is a list so you can pass multiple csv files to be overlayed on the same plot.
    #dataframes = ["/ex1_acf_results_t=100_2.csv", "/ex1_acf_results_t=300_2.csv", "/ex1_acf_results_t=500_2.csv"]
    dataframes = ["/acf_results_t=300_2_1.csv"]

    colours = ['black', 'red', 'blue', 'green', 'yellow']  # Array of colours for the lines.

    # Create plot and formats
    fig, axes = plt.subplots(nrows=1, ncols=1, constrained_layout=True)
    plt.rc('font', size=10)
    lw = 1  # Line width

    for count, df in enumerate(dataframes):
        df1 = pd.read_csv(cwd + df)  # reads file into a dataframe.
        ax1 = df1.plot(linestyle=' ', color=colours[count], marker='o', markerfacecolor=colours[count], markeredgecolor=colours[count],
                       markersize=2, linewidth=0, ax=axes, x='i', y='acf')

        #ax1.set_xlim(0, 5000)

        ax1.set_xlabel('Lag')
        ax1.set_ylabel('ACF')

        m = 5
        acf = df1['acf'][:m]
        i = df1['i'][:m]
        tau_int = integrate.simpson(acf, i)
        while m < 10*tau_int:
            acf = df1['acf'][:m]
            i = df1['i'][:m]
            tau_int = integrate.simpson(acf, i)
            m += 1

        print("10Tau: ", 10 * tau_int)
        print("M", m)
        print()


        ax1.legend(['T = 100 $K$', 'T = 300 $K$', 'T = 500 $K$'])

    plt.axhline(y=0, color='green', linestyle='-')

    plt.show()



#display()
display_acf()
