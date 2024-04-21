import numpy as np
import matplotlib.pyplot as plt
from goph420_lab03.regression import multi_regress


def main():
    time, magnitude = np.loadtxt(fname="M_data.txt", unpack=True)
    plot_rawData(time, magnitude, 'Time t [hr]',
                 'Magnitude M', 'figures/raw_data')
    cut = [34, 46, 72, 96]
    plot_rawData(time, magnitude, 'Time t [hr]',
                 'Magnitude M', 'figures/cut_bars', cut)
    cut_dex = index_cut(time, cut)

    # splicing the dependant variables
    # to be later counted to create the regressions dependant variable
    y = []
    y.append(magnitude[:cut_dex[0]])
    for k in range(len(cut_dex)-1):
        y.append(magnitude[cut_dex[k]:cut_dex[k+1]])
    y.append(magnitude[cut_dex[k+1]:])

    # creating the dependent variable for the regression
    M = np.linspace(-.15, 1, 30)
    # Build a list of lists for the linear regression will inputs
    N, Z, res = gut_rich_lin(y, M)
    # Plot the linearized Guten-Richter Law model for each period
    i = 1
    for num, z, residual in zip(N, Z, res):
        plt_gut_rich(M, num, z, residual, f"figures/period_{i}")
        i += 1


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
    window = index_time(x)
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


def plt_gut_rich(M, N, Z, res, location):
    plt.figure()
    a, e, rsq = multi_regress(np.log10(N), Z)
    line = 10 ** np.matmul(Z, a)
    res = np.append(res, e)

    plt.semilogx(N, M, 'g*', label="Number of Events With Magnitudes M")

    plt.plot(line, M, ':r',
             label=f"N = 10^({a[0]:.2f} - {a[1]:.2f}M)\n$R^2$ = {rsq:.3f})")
    plt.title(f"Time Period {location[-1]}")
    plt.ylabel("Magnitudes M")
    plt.xlabel("Number of Events")
    plt.legend()
    plt.savefig(location)


def gut_rich_lin(y, M):
    """
    creates master lists of arrays for multi_regress

    Inputs
    ______
    y: arraylike
        dependant values of the raw data
    M: arraylike
        dependent values for the model

    Outputs
    _______
    N: list of arraylike elements
        list with arrays that contain the number
        of events above a certain threshold
    z: list of arraylike elements
        list with arrays that contain the Z matrix of coefficients
    res: list of arraylike elements
        list with arrays that will contain the residuals

    """
    N = []
    res = []
    z = []
    for o in range(len(y)):
        N.append([sum(1 for n in y[o] if n > M[m]) for m in range(len(M))])
        res.append(np.zeros(0))
        Z = np.ones_like(N[o]) * -M
        Z = np.column_stack((np.ones_like(N[o]), Z))
        z.append(Z)

    return (N, z, res)


def index_time(array, value=24):
    """
    finds the index with the closest value to a time interval

    Inputs
    ______
    array: arraylike
        array to search over
    Value: float
        time in hours for which to section off default 24
    Outputs
    _______
    index: list
        the index for the points of seperations
    """
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


def index_cut(array, lista):
    """
    finds the index with the closest value to values in a list

    Inputs
    ______
    array: arraylike
        array to search over
    lista: list
        list of values to search over
    Outputs
    _______
    index: list
        the index values the searched items
    """
    array = np.asarray(array)
    index = []
    for c in lista:
        i_0 = bisection(array, c)
        index.append(i_0)
    return index


def bisection(array, value):
    """
    This a search method that finds the index of a value in a sorted list
    which lists in the time domain are

    Inputs
    ______
    array: arraylike
        the sorted list that is to be searched
    value: float
        the value to search for

    Outputs
    _______
    jl: int
        index value if the index is not on either end of the array
    """
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
