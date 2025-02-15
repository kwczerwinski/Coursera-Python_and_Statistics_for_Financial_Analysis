# Linear pattern

# Quantifying association with covariance
# Population: Cov(X,Y) = (sum((X_i - mu_X) * (Y_i - mu_Y)) / N
# Sample: Cov(X,Y) = (sum((X_i - X_bar) * (Y_i - Y_bar)) / (n - 1)
# DataFrame's method for showing covariance .cov()
# Covariance is affected by variance of two random variables.

# Coefficient of correlation (CoC)
# Population: ro(X,Y) = Cov(X,Y) / (sigma_X * sigma_Y)
# Sample: r(X,Y) = Cov(X,Y) / (s_X * s_Y)
# CoC always between -1 and 1.
# DataFrame's method for showing CoC .corr()

# scatter plot
from pandas.plotting import scatter_matrix
# sm = scatter_matrix(<DataFrame>)

# Model is equation of Responce using predictors
# Simple linear model regression - only one predictor
# Assuming population is normal - responce population is also normal, with equal variance
# Linearity: The mean of y is linearly determined by predictors
# Independence: With different X, responses are independent
# Normality: The random noise and y follow normal distributions
# Equal variance: The variance y are all equal even if the values of predictors are different
# Population model: y = B0 + B1 * x + e; e ~ N(0, sigma**2)
# Mean equation: mu = B0 + B1 * x

# Assumptions needs to be validated

# Finding line closest to mean equation
# Sum of squared error: SSE_i<1,N> = sum(error_i**2)
# Best fit line - minimalization of SSE
import statsmodels.formula.api as smf
# Ordinary Least Square estimation (OLS)
# model = smf.ols(formula='<responce>~<predictor>', data=<dataframe>).fit()
# B0 = model.params[0]
# B1 = model.params[1]
# <dataframe>['BestResponse'] = B0 + B1 * <dataframe>['<predictor>']
# model.summary()
# R-squared - how good is model for predicting in %
# P>[t] on <predictor> row - p-value should be less than 0.05 for model to be accepted
# to the right we have bounds for 95% (deafult) confidence interval

# Validating assumptions for linear regression model
# Io other words - demonstrate that sample data is not against assumptions
# Linearity - make scatter plot to show it exists or not
# Independent - observed error is independent mutually, in other words, no serial correlation in errors
# - make plot of errors (differences) between real results and response, and try to find a pattern
# <data>['error'] = <data>['<responce>'] - <data>['BestResponse']
# - Durbin-Watson test - visible in model.summary(), always [0, 4],
#   0 is strong positive autocorrelation, 4 is strong negative autocorrelation, 2 is no autocorrelation
# Normality - use quantile-quantile (QQ) plot
import scipy.stats as stats
import matplotlib.pyplot as plt


# z = (<data error> - <data error>.mean()) / <data error>.std(ddof=1)  # TODO: shouldn't be 2?
# stats.probplot(z, dist="norm", plot=plt)
# plt.show()
# plot should be a line, and around it should be scattered points that follows that line
# Equal variance - plot error vs predictor
# <data>.plot(kind='scatter', x='<responce>', y='error')
# try to find pattern, like smaller variance at the end of plot

# If at least one assumption is violated, then cannot make statistical inference,
# however model accuracy and consistency doesn't rely on assumptions and still can be used.

# Multiple linear regression model (MLRM)
# Predicting SPY (S&P500)
# Different markets have different opening and close times.
# US Markets: SPY, S&P500, Nasdaq, Dji (FTSE 100 from UK is unavailable from Yahoo Finance, but it's still important)
# EU Markets: CAC40, DAXI
# Asian Markets: Aord, HSI, Nikkei
#
# Responce: SPY open price tomorrow - SPY open price today
# Predictors:
#   Group 1: US markets one day lag (open - open last day)
#   Group 2: EU markets one day lag (open - open last day) - ideally it should be, price at noon - open price
#   Group 3: Asian markets close - open prices
#
# Handling NA values:
# 1. Fill-forward method - fill missing values by using last valid value: <data>=<data>.fillna(method='ffill')
# 2. Drop first rows with NA: <data>.dropna()
# 3. Check if any NA is still remaining: <data>.isnull().sum()
#
# Saving dataframe: <df>.to_csv('<path')
#
# Training and testing model
# Split data into two parts - train and test
# Stock markets are very noisy and correlations between them are tiny.
# Building model: mlrm = smf.ols(formula='<responce>~<predictor1>+<predictor2>+...', data=<train_data>).fit()
# mlrm.summary()
# Prob (F-statistc) - overall significance of mlrm; if <0.05 then we know that some predictors are useful
# only Aord have p-value <0.05
# multicollinearity: two or more predictors are highly, linearly related (does not reduce predictive power)
#
# Making prediction
# <train_data>['predictedY'] = mlrm.predict(<train_data>)
# <test_data>['predictedY'] = mlrm.predict(<test_data>)
# Real and predicted train data should have some correlation.
#
# Model evaluation
# k - number of predictors
# RSME = (SSE / (n - k - 1)) ** 0.5
# Adjusted R^2 = 1 - (1 - R ** 2) * (n - 1) / (n - k - 1)

# RMSE - Root Mean Squared Error, Adjusted R^2
# model_k - number of predictors
# yname - name of responce
def adjusted_metric(data, model, model_k, yname):
    data['yhat'] = model.predict(data)
    SST = ((data[yname] - data[yname].mean()) ** 2).sum()
    SSR = ((data['yhat'] - data[yname].mean()) ** 2).sum()
    SSE = ((data[yname] - data['yhat']) ** 2).sum()
    r2 = SSR / SST
    adjustR2 = 1 - (1 - r2) * (data.shape[0] - 1) / (data.shape[0] - model_k - 1)
    RMSE = (SSE / (data.shape[0] - model_k - 1)) ** 0.5
    return adjustR2, RMSE


import pandas as pd


# model_k - number of predictors
# yname - name of responce
def assess_table(train, test, model, model_k, yname):
    r2train, rmse_train = adjusted_metric(train, model, model_k, yname)
    r2test, rmse_test = adjusted_metric(test, model, model_k, yname)
    assessment = pd.DataFrame(index=['R2', 'RMSE'], columns=['Train', 'Test'])
    assessment['Train'] = [r2train, rmse_train]
    assessment['Test'] = [r2test, rmse_test]
    return assessment


# Evaluate strategy
# Strategy: buy 1 share when signal is positive, sell otherwise
#
# Train
# Train['Order'] = [1 if sig>0 else -1 for sig in Train['PredictedY']]
# Train['Profit'] = Train['spy'] * Train['Order']
# Train['Wealth'] = Train['Profit'].cumsum()
# print('Total profit made in Train: ', Train['Profit'].sum())
# plt.figure(figsize=(10, 10))
# plt.title('Performance of Strategy in Train')
# plt.plot(Train['Wealth'].values, color='green', label='Signal based strategy')
# plt.plot(Train['spy'].cumsum().values, color='red', label='Buy and Hold strategy')
# plt.legend()
# plt.show()
#
# Test
# Test['Order'] = [1 if sig>0 else -1 for sig in Test['PredictedY']]
# Test['Profit'] = Test['spy'] * Test['Order']
# Test['Wealth'] = Test['Profit'].cumsum()
# print('Total profit made in Test: ', Test['Profit'].sum())
# plt.figure(figsize=(10, 10))
# plt.title('Performance of Strategy in Train')
# plt.plot(Test['Wealth'].values, color='green', label='Signal based strategy')
# plt.plot(Test['spy'].cumsum().values, color='red', label='Buy and Hold strategy')
# plt.legend()
# plt.show()
#
# Include initial investment in wealth for computing Sharpe ratio
# Train['Wealth'] = Train['Wealth'] + Train.loc[Train.index[0], 'Price']
# Test['Wealth'] = Test['Wealth'] + Test.loc[Test.index[0], 'Price']
#
# Sharpe Ratio on Train data
# Train['Return'] = np.log(Train['Wealth']) - np.log(Train['Wealth'].shift(1))
# dailyr = Train['Return'].dropna()
# print('Daily Sharpe Ratio is ', dailyr.mean()/dailyr.std(ddof=1))
# print('Yearly Sharpe Ratio is ', (252**0.5)*dailyr.mean()/dailyr.std(ddof=1))
#
# Maximum Drawdown in Train data
# Train['Peak'] = Train['Wealth'].cummax()
# Train['Drawdown'] = (Train['Peak'] - Train['Wealth'])/Train['Peak']
# print('Maximum Drawdown in Train is ', Train['Drawdown'].max())
