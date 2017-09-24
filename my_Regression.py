"""
Cameron Knight
Code based on Intro and data section od the Machine learning series by Sentdex
https://www.youtube.com/channel/UCfzlCWGWYyIQ0aLC5w48gBQ
Description: regression on randomly generated data
"""

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')


# Sample Data------------------------------------------------------------------------------
# xs = np.array([1,2,3,4,5,6], dtype = np.float64) # Data x values converted to num py array
# Explicitly set the data type to 64 bit float
# ys = np.array([5,2,5,3,7,6], dtype = np.float64) # data y values converted to num py array
# Explicitly set the data type to 64 bit float


def create_dataset(hm, variance, step=2, correlation=False):
    """
    Genorates a random set of data of lenth hm with the max variance of varance and a pos correlation corespondiong to
    the slope of step if correlation  == "pos" and negative if correlation = "neg" otherwise no correlation
    """

    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if (correlation and correlation == 'pos'):
            val += step
        if (correlation and correlation == 'neg'):
            val -= step

    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def best_fit_slope_and_intercept(xs, ys):
    """
    Calculates the slope of the best fit and intecept
    """
    m = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
         ((mean(xs) ** 2) - mean(xs ** 2)))  # mean definition of linear regression

    b = mean(ys) - m * mean(xs)  # calculates the y intecept based on the slope

    return m, b


def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig) ** 2)


def coefficient_of_determ(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_reg = squared_error(ys_orig, ys_line)
    squared_error_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_reg / squared_error_mean)


xs, ys = create_dataset(40, 5, 2, correlation='pos')
m, b = best_fit_slope_and_intercept(xs, ys)

regression_Line = [m * x + b for x in xs]

predictPerc = .1  # Percent of the data out
predict_x = max(xs) * (1 + predictPerc)  # predict data predictPerc% out
predict_y = m * predict_x + b  # Predicts using regression line

r_sqr = coefficient_of_determ(ys, regression_Line)

print("r^2", r_sqr)

# Show Graph using matplotlib--------------------------------------------------------------
plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, s=50, color='g')
plt.plot(xs, regression_Line)
plt.show()
