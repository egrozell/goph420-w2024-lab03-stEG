import numpy as np
import matplotlib.pyplot as plt


def main():
    time, magnitude = np.loadtxt(fname="M_data.txt", unpack=True)
    plot_rawData(time, magnitude, 'Time t [hr]',
                 'Magnitude M', 'figures/raw_data')
    cut = [34, 46, 72, 96]
    plot_rawData(time, magnitude, 'Time t [hr]',
                 'Magnitude M', 'figures/cut_bars', cut)


def plot_rawData(x, y, x_label, y_label, location, vert_bar=None):
    """
    plots raw data for long sorted datasets in the x direction
    Inputs
    ______
    x: arraylike
        independent variables required to be sorted
    y: arraylike
        dependant variables
    x_label: string
        label for the x axis
    y_label: string
        label for the y axis
    vert_bar: arraylike
        places vertical bars
    location: string
        save location and name
    """
    def vert(n, j):
        if vert_bar is not None:
            for value in vert_bar:
                if value > x[window[n]-1] and value < x[window[j]-1]:
                    ax[j].vlines(value, -1.5, 1.5, color='black', linewidth=3)
                    value = None
    window = index_day(x)
    fig, ax = plt.subplots(len(window), figsize=(15, len(window)*3))
    i = 0
    ax[i].plot(x[:window[i]], y[:window[i]],
               'ro', fillstyle='none', label='Day 1')
    vert(i, i)
    ax[i].legend(loc="upper right")
    ax[i].set_ylabel(y_label)
    for i in range(len(window)-1):
        ax[i+1].plot(x[window[i]:window[i+1]],
                     y[window[i]:window[i+1]], 'ro',
                     fillstyle='none', label=f'Day {i+2}')
        vert(i, i+1)
        ax[i].legend(loc="upper right")
        ax[i].set_ylabel(y_label)
    i += 1
    vert(i, i)
    ax[i].legend(loc="upper right")
    ax[i].set_ylabel(y_label)
    ax[i].set_xlabel(x_label)
    plt.savefig(location)
    plt.close("all")


def index_day(array, value=24):
    array = np.asarray(array)
    window = []
    k = 2
    i_0 = bisection(array, value)
    window.append(i_0)
    while i_0 < len(array) and k < 100:
        i_0 = bisection(array, value*k)
        k += 1
        window.append(i_0)
    return window


def bisection(array, value):
    n = len(array)
    if (value < array[0]):
        return -1
    elif (value > array[n-1]):
        return n
    jl = 0    # Initialize lower
    ju = n-1  # and upper limits.
    while (ju-jl > 1):  # If we are not yet done,
        jm = (ju + jl) >> 1  # compute a midpoint with a bitshift
        if (value >= array[jm]):
            jl = jm  # and replace either the lower limit
        else:
            ju = jm  # or the upper limit, as appropriate.
        # Repeat until the test condition is satisfied.
    if (value == array[0]):  # edge cases at bottom
        return 0
    elif (value == array[n-1]):  # and top
        return n-1
    else:
        return jl


if __name__ == "__main__":
    main()
