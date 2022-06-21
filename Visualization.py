import matplotlib.pyplot as plt
import datetime
import numpy as np
import pandas as pd


def get_versus_withpanda(node, time_1, time_2, time_3, title):
    df1 = pd.DataFrame({'node': np.array(node),
                        'time': time_1})

    df2 = pd.DataFrame({'node': np.array(node),
                        'time': time_2})

    df3 = pd.DataFrame({'node': np.array(node),
                        'time': time_3})

    # plot both time series
    plt.plot(df1.node, df1.time, color='blue', label='Pandapower Newton-Raphson', linewidth=3)
    plt.plot(df2.node, df2.time, color='red', label='Powergridmodel Newton-Raphson', linewidth=3)
    plt.plot(df3.node, df3.time, color='green', label='Powergridmodel Linear Method',
             alpha=0.3, linestyle='--', linewidth=3)

    # add title and axis labels
    plt.title(title)
    plt.xlabel('no.of nodes')
    plt.ylabel('Calculation_time')

    # add legend
    plt.legend()

    # display plot
    plt.show()


def get_versus_withoutpanda(node, time_1, time_2, title):
    df1 = pd.DataFrame({'node': np.array(node),
                        'time': time_1})

    df2 = pd.DataFrame({'node': np.array(node),
                        'time': time_2})

    # plot both time series
    plt.plot(df1.node, df1.time, color='red', label='Powergridmodel Newton-Raphson', linewidth=3)
    plt.plot(df2.node, df2.time, color='green', label='Powergridmodel Linear Method',
             alpha=0.3, linestyle='--', linewidth=3)

    # add title and axis labels
    plt.title(title)
    plt.xlabel('no.of nodes')
    plt.ylabel('Calculation_time')

    # add legend
    plt.legend()

    # display plot
    plt.show()

