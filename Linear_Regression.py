                        # Linear Regression (Python Implementation) : 
                        
This article discusses the basics of linear regression and its implementation in Python programming language.
Linear regression is a statistical method for modelling relationship between a dependent variable with a given set of independent variables.
Note: In this article, we refer dependent variables as response and independent variables as features for simplicity.
In order to provide a basic understanding of linear regression, we start with the most basic version of linear regression, i.e. Simple linear regression. 

            # Simple Linear Regression :
Simple linear regression is an approach for predicting a response using a single feature.
It is assumed that the two variables are linearly related. Hence, we try to find a linear function that predicts the response value(y) as accurately as possible as a function of the feature or independent variable(x).

    # Example :

import numpy as np
import matplotlib.pyplot as plt

def estimate_coef(x, y):
	# number of observations/points
	n = np.size(x)

	# mean of x and y vector
	m_x = np.mean(x)
	m_y = np.mean(y)

	# calculating cross-deviation and deviation about x
	SS_xy = np.sum(y*x) - n*m_y*m_x
	SS_xx = np.sum(x*x) - n*m_x*m_x

	# calculating regression coefficients
	b_1 = SS_xy / SS_xx
	b_0 = m_y - b_1*m_x

	return (b_0, b_1)

def plot_regression_line(x, y, b):
	# plotting the actual points as scatter plot
	plt.scatter(x, y, color = "m",
			marker = "o", s = 30)

	# predicted response vector
	y_pred = b[0] + b[1]*x

	# plotting the regression line
	plt.plot(x, y_pred, color = "g")

	# putting labels
	plt.xlabel('x')
	plt.ylabel('y')

	# function to show plot
	plt.show()

def main():
	# observations / data
	x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
	y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

	# estimating coefficients
	b = estimate_coef(x, y)
	print("Estimated coefficients:\nb_0 = {} \
		\nb_1 = {}".format(b[0], b[1]))

	# plotting regression line
	plot_regression_line(x, y, b)

if __name__ == "__main__":
	main()

    # Output of above piece of code is: 
 

Estimated coefficients:
b_0 = -0.0586206896552
b_1 = 1.45747126437

             # Multiple linear regression : 
             
Multiple linear regression attempts to model the relationship between two or more features and a response by fitting a linear 
equation to the observed data.
Clearly, it is nothing but an extension of simple linear regression.
Consider a dataset with p features(or independent variables) and one response(or dependent variable). 
Also, the dataset contains n rows/observations.


The regression line for p features is represented as: 
h(x_i) = \beta_0 + \beta_1x_{i1} + \beta_2x_{i2} + .... + \beta_px_{ip}  
where h(x_i) is predicted response value for ith observation and b_0, b_1, …, b_p are the regression coefficients.

                # Example : 

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics

# load the boston dataset
boston = datasets.load_boston(return_X_y=False)

# defining feature matrix(X) and response vector(y)
X = boston.data
y = boston.target

# splitting X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
													random_state=1)

# create linear regression object
reg = linear_model.LinearRegression()

# train the model using the training sets
reg.fit(X_train, y_train)

# regression coefficients
print('Coefficients: ', reg.coef_)

# variance score: 1 means perfect prediction
print('Variance score: {}'.format(reg.score(X_test, y_test)))

# plot for residual error

## setting plot style
plt.style.use('fivethirtyeight')

## plotting residual errors in training data
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train,
			color = "green", s = 10, label = 'Train data')

## plotting residual errors in test data
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test,
			color = "blue", s = 10, label = 'Test data')

## plotting line for zero residual error
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)

## plotting legend
plt.legend(loc = 'upper right')

## plot title
plt.title("Residual errors")

## method call for showing the plot
plt.show()

      # Output : 
      
Coefficients:
[ -8.80740828e-02   6.72507352e-02   5.10280463e-02   2.18879172e+00
-1.72283734e+01   3.62985243e+00   2.13933641e-03  -1.36531300e+00
2.88788067e-01  -1.22618657e-02  -8.36014969e-01   9.53058061e-03
-5.05036163e-01]
Variance score: 0.720898784611


                                        # Polynomial Regression : 

You can regard polynomial regression as a generalized case of linear regression. You assume the polynomial dependence between the output and inputs and, consequently, the polynomial estimated regression function.

In other words, in addition to linear terms like 𝑏₁𝑥₁, your regression function 𝑓 can include non-linear terms such as 𝑏₂𝑥₁², 𝑏₃𝑥₁³, or even 𝑏₄𝑥₁𝑥₂, 𝑏₅𝑥₁²𝑥₂, and so on.

The simplest example of polynomial regression has a single independent variable, and the estimated regression function is a polynomial of degree 2: 𝑓(𝑥) = 𝑏₀ + 𝑏₁𝑥 + 𝑏₂𝑥².


##############################################################################################################################

                    # Applications:

1. Trend lines: A trend line represents the variation in  quantitative data with passage of time (like GDP, oil prices, etc.). 
These trends usually follow a linear relationship. Hence, linear regression can be applied to predict future values. However, 
this method suffers from a lack of scientific validity in cases where other potential changes can affect the data.

2. Economics: Linear regression is the predominant empirical tool in economics. For example, it is used to predict consumption 
spending, fixed investment spending, inventory investment, purchases of a country’s exports, spending on imports, the demand to 
hold liquid assets, labour demand, and labour supply.

3. Finance: Capital price asset model uses linear regression to analyse and quantify the systematic risks of an investment.

4. Biology: Linear regression is used to model causal relationships between parameters in biological systems.

