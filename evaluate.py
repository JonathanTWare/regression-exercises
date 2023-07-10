import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def plot_residuals(y, yhat):
    """
    Creates a residual plot.

    Parameters:
    - y: Array-like, actual values
    - yhat: Array-like, predicted values
    """
    residuals = y - yhat
    plt.scatter(yhat, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.show()

def regression_errors(y, yhat):
    """
    Returns regression error metrics.

    Parameters:
    - y: Array-like, actual values
    - yhat: Array-like, predicted values

    Returns:
    - Dictionary with SSE, ESS, TSS, MSE, RMSE
    """
    sse = np.sum((y - yhat) ** 2)
    ess = np.sum((yhat - np.mean(y)) ** 2)
    tss = np.sum((y - np.mean(y)) ** 2)
    mse = mean_squared_error(y, yhat)
    rmse = np.sqrt(mse)

    return {'SSE': sse, 'ESS': ess, 'TSS': tss, 'MSE': mse, 'RMSE': rmse}

def baseline_mean_errors(y):
    """
    Computes the SSE, MSE, and RMSE for the baseline model.

    Parameters:
    - y: Array-like, actual values

    Returns:
    - Dictionary with SSE, MSE, and RMSE for the baseline model
    """
    y_mean = np.mean(y)
    yhat_baseline = np.full_like(y, y_mean)
    sse = np.sum((y - yhat_baseline) ** 2)
    mse = mean_squared_error(y, yhat_baseline)
    rmse = np.sqrt(mse)

    return {'SSE': sse, 'MSE': mse, 'RMSE': rmse}

def better_than_baseline(y, yhat):
    """
    Returns True if the model performs better than the baseline, otherwise False.

    Parameters:
    - y: Array-like, actual values
    - yhat: Array-like, predicted values

    Returns:
    - Boolean value indicating if the model performs better than the baseline
    """
    sse_model = np.sum((y - yhat) ** 2)
    sse_baseline = baseline_mean_errors(y)['SSE']

    return sse_model < sse_baseline
