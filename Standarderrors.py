import os
import pyreadr
import time
import datetime
import folium
import numpy as np
import pandas as pd
import geopandas as gpd
import scipy.sparse as sp
from scipy.linalg import solve, pinv
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib.colors as mcolors
import statsmodels.api as sm
from folium.plugins import MarkerCluster
from adjustText import adjust_text
import matplotlib.dates as mdates
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from joblib import Parallel, delayed
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
starttime = pd.Timestamp.now()
#Import the cleaned data
data = pd.read_csv('PM10_BFF.csv')
dates = pd.to_datetime(data['Date'])
Y = data.drop('Date', axis=1).values.T
Time = Y.shape[1]
n = Y.shape[0]
time_index = np.arange(Time)
#############################################################################################
def create_design_matrix_X(time_index, Time, n):
    # Sine components
    sine_func_yearly = np.sin((2 * np.pi / (365 * 24)) * time_index)
    sine_func_semi_yearly = np.sin((2 * np.pi / (365 * 24 / 2)) * time_index)
    sine_func_monthly = np.sin((2 * np.pi / (365 * 24 / 12)) * time_index)
    sine_func_daily = np.sin((2 * np.pi / 24) * time_index)

    # Cosine components
    cosine_func_yearly = np.cos((2 * np.pi / (365 * 24)) * time_index)
    cosine_func_semi_yearly = np.cos((2 * np.pi / (365 * 24 / 2)) * time_index)
    cosine_func_monthly = np.cos((2 * np.pi / (365 * 24 / 12)) * time_index)
    cosine_func_daily = np.cos((2 * np.pi / 24) * time_index)

    # Combine all into a design matrix
    X_single = np.column_stack((np.ones(Time),  # Intercept
        np.linspace(0, 1, Time),  # Time trend
        sine_func_yearly,sine_func_semi_yearly,sine_func_monthly,sine_func_daily,cosine_func_yearly,cosine_func_semi_yearly,cosine_func_monthly,cosine_func_daily))
    # Expand for all stations
    X = np.tile(X_single[np.newaxis, :, :], (n, 1, 1))
    
    return X
#######################################################################################################
X = create_design_matrix_X(time_index, Time, n)
k = X.shape[2]
logging.info("The number of hourly timepoints are %d", Time)
logging.info("The number of monitoring stations are %d", n)
logging.info("The shape of the Y: %s", Y.shape)
logging.info(f'The number of hourly timepoints are {Time} across {n} monitoring stations for {k} covariates.\nThe shape of the Y is {Y.shape} and X is {X.shape}')
######################################################################################################################### Penalized Malimum Likelihood Estimation ################################################################
def LL_unpenalized(parameters, Y, X, k, n, Time, active_indices,lambda1, lambda2,lambda3 ):
    # Reconstruct the full parameter vector, including active parameters
    full_params = np.zeros(k + n + n*(n-1) + 1)
    full_params[active_indices] = parameters
    
    # Extract individual parameter sets
    beta = full_params[:k]
    phi = full_params[k:k+n]
    W = np.zeros((n, n))
    W[np.triu_indices(n, 1)] = full_params[int(k+n):int(k+n+(0.5*n*(n-1)))]
    W[np.tril_indices(n, -1)] = full_params[int(k+n+(0.5*n*(n-1))):int(k+(n*n))]    
    sigma2_eps = full_params[-1] + 1e-6  # Ensure positivity
    
    # Compute determinant term for log-likelihood
    try:
        det_term = np.linalg.slogdet(np.eye(n) - W)[1]
    except np.linalg.LinAlgError:
        return 1e10  # Penalize infeasible region
    
    # Compute the residuals and sum of squares
    residuals_est = np.zeros(Time - 1)
    for t in range(1, Time):
        u_t = Y[:, t] - W @ Y[:, t] - (phi * Y[:, t-1]) - X[:, t] @ beta
        residuals_est[t-1] = np.sum(u_t**2 / sigma2_eps)
    
    sum_of_squares = np.sum(residuals_est)
    
    # Calculate log-likelihood
    constant = -0.5 * (Time - 1) * (np.log(2 * np.pi) + n * np.log(sigma2_eps))
    LogLik = constant + (Time - 1) * det_term - 0.5 * sum_of_squares

    return -LogLik  # Negative for minimization
#################################################################################################
#################################################################################################
# Spectral norm constraint function for W
def spectral_norm_constraint(params, n, k, active_indices):
    # Reconstruct W matrix from parameters
    full_params = np.zeros(k + n + n*(n-1) + 1)
    full_params[active_indices] = params  # Restore only active parameters
    beta = full_params[:k]
    phi = full_params[k:k+n]
    W = np.zeros((n, n))
    W[np.triu_indices(n, 1)] = full_params[int(k+n):int(k+n+(0.5*n*(n-1)))]
    W[np.tril_indices(n, -1)] = full_params[int(k+n+(0.5*n*(n-1))):int(k+(n*n))]
    
    # Compute the spectral radius (maximum absolute eigenvalue)
    try:
        eigvals = np.linalg.eigvals(W)
        spectral_radius = max(abs(eigvals))
    except np.linalg.LinAlgError:
        return 1e10  # Penalize infeasible region if matrix is singular or nearly singular
    
    return spectral_radius  # Ensure spectral norm < 1
constraints = [{'type': 'ineq', 'fun': lambda x: spectral_norm_constraint(x, n, k,active_indices)}]
# Numerical Hessian computation 
def compute_hessian_entry(func, params, epsilon, args, i, j):
    n = len(params)
    perturb = np.eye(n) * epsilon

    delta_i = perturb[i]
    delta_j = perturb[j]

    f_pp = func(params + delta_i + delta_j, *args)
    f_pm = func(params + delta_i - delta_j, *args)
    f_mp = func(params - delta_i + delta_j, *args)
    f_mm = func(params - delta_i - delta_j, *args)

    value = (f_pp - f_pm - f_mp + f_mm) / (4 * epsilon**2)
    return (i, j, value)

def compute_numerical_hessian_parallel(func, params, epsilon=1e-5, args=()):
    n = len(params)
    hessian = np.zeros((n, n))

    # Prepare all (i, j) pairs with j >= i
    pairs = [(i, j) for i in range(n) for j in range(i, n)]
    logging.info(f"Submitting {len(pairs)} Hessian entries to the process pool...")
    # Use ThreadPoolExecutor
    with ProcessPoolExecutor(max_workers=200) as executor:
        futures = {executor.submit(compute_hessian_entry, func, params, epsilon, args, i, j): (i, j) for i, j in pairs}
        for future in as_completed(futures):
            i, j = futures[future]
            try:
                _, _, value = future.result()
                hessian[i, j] = value
                hessian[j, i] = value  # Symmetry
            except Exception as e:
                logging.error(f"Error computing Hessian entry ({i}, {j}): {e}")

    return hessian
# ==============================================================================================================================
# Load optimized parameters
parameters_opt = np.loadtxt('Unpenalized_parameters.txt')
active_indices = np.where(parameters_opt != 0)[0]  # Identify active parameters
# Select only active parameters
parameters_opt = parameters_opt[active_indices]

args = (Y, X, k, n, Time, active_indices,0,0,0) # Arguments for LL_unpenalized

# Compute Hessian (parallelized)
logging.info("Hessian computation started...")
hessian = compute_numerical_hessian_parallel(LL_unpenalized, parameters_opt, epsilon=1e-5, args=args)
np.save('hessian.npy', hessian)
logging.info("Hessian computation finished and saved to disk.")
# Compute inverse Hessian
logging.info("Inverse Hessian computation started...")
inv_hessian = pinv(hessian)
logging.info("Inverse Hessian computation completed.")

# Compute standard errors and Z-values
logging.info("Calculating standard errors and z-values...")
std_errors = np.sqrt(np.diag(inv_hessian))
z_values = parameters_opt / std_errors

# Save results
df_results = pd.DataFrame({
    'Parameter': parameters_opt,
    'Standard Error': std_errors,
    'Z-Value': z_values
})
df_results.to_csv(f'Data/model_results.csv', index=False)
logging.info(f"Parameters, standard errors, and z-values saved to Data folder")


