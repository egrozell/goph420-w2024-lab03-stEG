import numpy as np


def multi_regress(y, Z):
    """
    Parameters
    __________

    y: array_like, shape = (n, ) or (n,1)
        The vector pf dependent variable data
    Z: array_like, shape = (n,m)
        The matrix of independent variable data

    Returns
    _______
    numpy.ndarray, shape = (m, ) or (m,1)
        The vector of model coefficients
    numpy.ndarray, shape = (n, ) or (n,1)
        The vector of residuals
    float
        The coefficient of determination, r^2
    """

    ZTZ = np.matmul(np.transpose(Z), Z)  # Multiply ZTZ (Z Transpose Z)
    ZTy = np.matmul(np.transpose(Z), y)  # Multiply ZTZ (Z Transpose Y)

    e_avg = y - np.mean(y)
    # Residuals Multiply the inverse of ZTZ by ZTy
    a_coefs = np.matmul(np.linalg.inv(ZTZ), ZTy)
    Sy = np.matmul(np.transpose(e_avg), e_avg)
    e_residual = y - np.matmul(Z, a_coefs)
    Sr = np.matmul(np.transpose(e_residual), e_residual)
    R_squared = float((Sy-Sr)/Sy)

    return a_coefs, e_residual, R_squared
