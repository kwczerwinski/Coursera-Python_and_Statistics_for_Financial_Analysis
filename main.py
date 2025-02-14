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

