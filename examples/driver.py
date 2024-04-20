import numpy as np
import matplotlib.pyplot as plt


def main():
    time, magnitude = np.loadtxt(fname="M_data.txt", unpack=True)

    window = index_day(time)
    print(window)


def index_day(array):
    array = np.asarray(array)
    window = []
    value = 24.0
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
