from numpy import *
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import t

def data_loader(filename):
    # Load data from 'filename' and return the predictor variables as data_a and response variable as name_v
    # Replace this implementation with your actual data loading process
    data_a = loadtxt(filename,skiprows=1, usecols=(0), delimiter=',')
    name_v = loadtxt(filename,skiprows=1, usecols=(5), delimiter=',')
    return data_a, name_v

def fit_linear_regression(data_a, name_v):
    X_linear = column_stack((ones(len(data_a)), data_a))
    model = LinearRegression()
    model.fit(X_linear, name_v)
    return model

def fit_polynomial_regression(data_a, name_v, degree):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(data_a.reshape(-1, 1))
    model = LinearRegression()
    model.fit(X_poly, name_v)
    return model

def calculate_p_value(t_test_stat, dof):
    p_value = 2 * t.cdf(-abs(t_test_stat), df=dof)  # Calculate the two-tailed p-value
    return p_value

def perform_t_test(linear_model, poly_model, data_a, name_v):
    linear_residuals = name_v - linear_model.predict(column_stack((ones(len(data_a)), data_a)))
    poly_residuals = name_v - poly_model.predict(PolynomialFeatures(degree=2).fit_transform(data_a.reshape(-1, 1)))
    t_stat = (linear_residuals.T @ linear_residuals - poly_residuals.T @ poly_residuals) / len(name_v)

    dof = len(name_v) - 3  # Degrees of freedom for the t-test (number of observations - number of model parameters)
    p_value = calculate_p_value(t_stat, dof)

    return t_stat, p_value

def compute_cross_val_error(model, X, name_v):
    cv_error = mean(cross_val_score(model, X, name_v, scoring='neg_mean_squared_error', cv=5))
    return -cv_error

def plot_results(data_a, name_v, X_pred, linear_estimate, poly_estimate):
    plt.scatter(data_a, name_v, label='Data')
    plt.plot(X_pred, linear_estimate, label='Linear Regression', color='r')
    plt.plot(X_pred, poly_estimate, label='Polynomial Regression', color='g')
    plt.xlabel('Predictor X')
    plt.ylabel('Response y')
    plt.legend()
    plt.show()

def main():
    # Replace 'Auto.csv' with the actual filename of your data file
    data_a, name_v = data_loader('Auto.csv')

    # Convert 'name_v' to a numeric data type (e.g., float)
    name_v = name_v.astype(float)

    # Continue with the rest of the code as before
    # Fit Linear Regression Model
    linear_model = fit_linear_regression(data_a, name_v)

    # Fit Polynomial Regression Model
    polynomial_model = fit_polynomial_regression(data_a, name_v, degree=2)

    # Perform t-test on residuals
    t_test_stat, p_value = perform_t_test(linear_model, polynomial_model, data_a, name_v)

    # Compute cross-validation error for both models using the same data array with linear and polynomial features
    X_linear = column_stack((data_a, data_a**2))  # Combine linear and quadratic features
    linear_cv_error = compute_cross_val_error(linear_model, X_linear, name_v)

    X_poly = PolynomialFeatures(degree=2).fit_transform(data_a.reshape(-1, 1))
    poly_cv_error = compute_cross_val_error(polynomial_model, X_poly, name_v)

    # Plot the Predictor vs. Non-linear Estimate
    X_pred = linspace(data_a.min(), data_a.max(), 100)
    X_pred_reshaped = column_stack((ones(100), X_pred))  # Add a column of ones for linear regression
    linear_estimate = X_pred_reshaped @ linear_model.coef_.reshape(-1, 1)
    poly_estimate = polynomial_model.predict(PolynomialFeatures(degree=2).fit_transform(X_pred.reshape(-1, 1)))
    plot_results(data_a, name_v, X_pred, linear_estimate, poly_estimate)

    # Output t-test result and cross-validation errors
    print("T-test statistic:", t_test_stat)
    print("P-value:", p_value)
    print("Linear CV error:", linear_cv_error)
    print("Polynomial CV error:", poly_cv_error)

if __name__ == "__main__":
    main()
