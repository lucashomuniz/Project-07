"""
========
IMPORTS
========
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
sns.set_style('whitegrid')

# Loading the dataset
dados = pd.read_csv('dados/dataset1.csv')

# Shape
print(dados.shape)

# Info
dados.info()

# Sample
dados.sample(10)

"""
=====================
EXPLORATORY ANALYSIS
=====================
"""

# Correlation Table
print(dados.corr())

# Correlation (Graphic)
sns.pairplot(dados)
plt.show()


"""
=====================================================================
ANALYSIS 1: RELATIONSHIP BETWEEN TIME ON THE WEBSITE AND AMOUT SPENT
=====================================================================
"""

# Plot
plt.figure(figsize = (18, 12))
sns.set(font_scale = 1.1)
sns.jointplot(data = dados, x = 'tempo_total_logado_website', y = 'valor_total_gasto', color = 'blue')

# Observing the histograms, we can see that the data are normally distributed (follow a normal distribution).
# Looking at the scatter plot, there doesn't seem to be a correlation between the two variables.
# Let's confirm by calculating the correlation coefficient between them.

# Correlation
dados[['tempo_total_logado_website', 'valor_total_gasto']].corr()

# There does not seem to be a correlation between the time logged into the website and the amount spent by customers.

"""
==================================================================
ANALYSIS 2: RELATIONSHIP BETWEEN TIME ON THE APP AND AMOUNT SPENT
==================================================================
"""

# Columns
print(dados.columns)

# Plot
plt.figure(figsize = (18, 12))
sns.set(font_scale = 1.2)
sns.jointplot(data = dados, x = 'tempo_total_logado_app', y = 'valor_total_gasto', color = 'green')

# Observing the histograms, we can see that the data are normally distributed (follow a normal distribution).
# Looking at the scatter plot, there seems to be a positive correlation between the two variables.
# Let's confirm by calculating the correlation coefficient between them.

# Correlation
dados[['tempo_total_logado_app', 'valor_total_gasto']].corr()

# The data has a moderate positive correlation.
# We can deduce that the total amount spent monthly tends to increase if the customer spends more time logged into the app.

"""
==============================================================================
ANALYSIS 3: RELATIONSHIP BETWEEN REGISTRATION ON THE APP AND REGISTRATION TIME
===============================================================================
"""

# Columns
print(dados.columns)

# Plot
plt.figure(figsize = (18, 12))
sns.set(font_scale = 1.2)
sns.jointplot(data = dados, x = 'tempo_total_logado_app', y = 'tempo_cadastro_cliente', color = 'red')

# Observing the histograms, we can see that the data are normally distributed (follow a normal distribution).
# From the scatterplot we can see that the data is very dense in the middle of the plot and there is no clear correlation.
# We can deduce that there are many customers who have been members for 3-4 years and spend approximately 11.5-12.5 minutes on the app.

"""
===================================================================
ANALYSIS 4: RELATIONSHIP BETWEEN REGISTRATION TIME AND AMOUT SPENT
===================================================================
"""

# Columns
print(dados.columns)

# Plot
sns.set(font_scale = 1.1)
sns.set_style('whitegrid')
sns.lmplot(y = "valor_total_gasto", x = "tempo_cadastro_cliente", data = dados)

# Correlation
dados[['tempo_cadastro_cliente', 'valor_total_gasto']].corr()

# From the lmplot and the correlation coefficient, we can see that the registration time
# and total amount spent have a strong positive correlation i.e. as the customer becomes older
# (longer registration time), the total amount spent by customers also increases.
# Another thing to note is that the shadow around the line is very thin,
# which means that the errors between the estimate (line) and the data points are relatively small.

"""
=======================================================================================
ANALYSIS 5: RELATIONSHIP BETWEEN TIME LOGGED IN THE APP AND TIME LOGGED IN THE WEBSITE
=======================================================================================
"""

# Plot
plt.figure(figsize = (18, 12))
sns.set(font_scale = 1.2)
sns.jointplot(data = dados, x = 'tempo_total_logado_app', y = 'tempo_total_logado_website', color = 'magenta' )

# We have not detected critical issues and can move forward.
# Let's keep the predictor variable with low correlation and validate this relationship with the target variable in the final model.
# Pre-Processing Data for Building Machine Learning Models
print(dados.columns)

# Input Variables (predictor variables)
X = dados[['tempo_cadastro_cliente', 'numero_medio_cliques_por_sessao',
'tempo_total_logado_app', 'tempo_total_logado_website']]

# Output Variable (target variable or target)
y = dados['valor_total_gasto']

# Split into training and testing data
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size = 0.3, random_state = 101)
len(X_treino)
len(X_teste)

# Standardization
scaler = StandardScaler()
scaler.fit(X_treino)
X_treino = scaler.transform(X_treino)
X_teste = scaler.transform(X_teste)

"""
=======================================
MODEL 1: LINEAR REGRESSION (BENCHMARK)
=======================================
"""

# Create the model
modelo_v1 = LinearRegression()

# Training
modelo_v1.fit(X_treino, y_treino)
print('Coeficientes: \n', modelo_v1.coef_)

# Coefficients of predictor variables
df_coef = pd.DataFrame(modelo_v1.coef_, X.columns, columns = ['Coeficiente'])

# Model Evaluation
# Predictions with test data
pred_v1 = modelo_v1.predict(X_teste)

# Plot
plt.figure(figsize = (10,8))
plt.scatter(x = y_teste, y = pred_v1, color = 'skyblue', edgecolors = 'black')
plt.xlabel('Valor Real de Y')
plt.ylabel('Valor Previsto de Y');

# From the scatterplot, we can see that there is a very strong correlation between the predicted y's
# and the actual y's in the test data. This means that we have a very good model.

# Metrics
# Average amount spent by customers
dados['valor_total_gasto'].mean()

# Minimum Value
dados['valor_total_gasto'].min()

# Maximum Value
dados['valor_total_gasto'].max()

# MAE - Mean Absolute Error
mean_absolute_error(y_teste, pred_v1)

# The MAE predicts that, on average, our model's predictions (of amounts spent)
# are wrong by approximately 7.76 reais, which is a small amount compared to the average amount spent per customer.

# MSE - Mean Squared Error
mean_squared_error(y_teste, pred_v1)

# RMSE - Square Root Mean Square Error
np.sqrt(mean_squared_error(y_teste, pred_v1))

# RMSE predicts that, on average, our model's predictions (of amounts spent) are wrong
# at approximately 9.74, which is a small amount compared to the average amount spent per customer.
# R2 coefficient
r2_score(y_teste, pred_v1)

# Explained Variance
explained_variance_score(y_teste, pred_v1)

# Our model is able to explain 98% of the data variance, which is excellent.
# The R2 coefficient of 98% and the other metrics demonstrate that this is a very good model.
# Could we improve this performance?

# Waste
# Plot
plt.figure(figsize = (8,4))
ax = sns.distplot((y_teste - pred_v1), bins = 40, color = 'red', hist_kws = dict(edgecolor = 'black', linewidth = 0.3))
ax.set(xlim = (-40, 40))
ax.set(ylim = (0, 0.055));

# The residuals are approximately normally distributed, which indicates a good fit of the model.

"""
==========================
MODEL 2: RIDGE REGRESSION
==========================
"""

# Create the model
modelo_v2 = Ridge(alpha = 1.0)

# Training
modelo_v2.fit(X_treino, y_treino)
print('Coeficientes: \n', modelo_v2.coef_)

# Coefficients of predictor variables
df_coef = pd.DataFrame(modelo_v2.coef_, X.columns, columns = ['Coeficiente'])

# Predictions with test data
pred_v2 = modelo_v2.predict(X_teste)

# Plot
plt.figure(figsize = (10,8))
plt.scatter(x = y_teste, y = pred_v2, color = 'skyblue', edgecolors = 'black')
plt.xlabel('Valor Real de Y')
plt.ylabel('Valor Previsto de Y');

# MAE - Mean Absolute Error
mean_absolute_error(y_teste, pred_v2)

# MSE - Mean Squared Error
mean_squared_error(y_teste, pred_v2)

# RMSE - Square Root Mean Square Error
np.sqrt(mean_squared_error(y_teste, pred_v2))

# R2 coefficient
r2_score(y_teste, pred_v2)

# Explained Variance
explained_variance_score(y_teste, pred_v2)

# Plot
plt.figure(figsize = (8,4))
ax = sns.distplot((y_teste - pred_v2), bins = 40, color = 'red', hist_kws = dict(edgecolor = 'black', linewidth = 0.3))
ax.set(xlim = (-40, 40))
ax.set(ylim = (0, 0.055));

"""
==========================
MODEL 3: LASSO REGRESSION
==========================
"""

# Create the model
modelo_v3 = Lasso(alpha = 1.0)

# Training
modelo_v3.fit(X_treino, y_treino)
print('Coeficientes: \n', modelo_v3.coef_)

# Coefficients
df_coef = pd.DataFrame(modelo_v3.coef_, X.columns, columns = ['Coeficiente'])

# Predictions with test data
pred_v3 = modelo_v3.predict(X_teste)

# Plot
plt.figure(figsize = (10,8))
plt.scatter(x = y_teste, y = pred_v3, color = 'green', edgecolors = 'blue')
plt.xlabel('Valor Real de Y')
plt.ylabel('Valor Previsto de Y');

# MAE - Mean Absolute Error
mean_absolute_error(y_teste, pred_v3)

# MSE - Mean Squared Error
mean_squared_error(y_teste, pred_v3)

# RMSE - Square Root Mean Square Error
np.sqrt(mean_squared_error(y_teste, pred_v3))

# R2 coefficient
r2_score(y_teste, pred_v2)

# Explained Variance
explained_variance_score(y_teste, pred_v3)

# Plot
plt.figure(figsize = (8,4))
ax = sns.distplot((y_teste - pred_v3), bins = 40, color = 'red', hist_kws = dict(edgecolor = 'black', linewidth = 0.3))
ax.set(xlim = (-40, 40))
ax.set(ylim = (0, 0.055));

"""
=====================
PROJECT`S CONCLUSION
=====================
"""

# Template Selection
# Model 3 had a slightly higher error rate (RMSE) and can be discarded.
# Models 1 and 2 were very close and in this case we should choose
# the simplest model, which in our example is model 1.
dados.head()

# Coefficients
df_coef_final = pd.DataFrame(modelo_v1.coef_, X.columns, columns = ['Coeficiente'])

# Interpretations of Coefficients:
# * Keeping all other resources fixed, a 1 unit increase in player registration time
# customer is associated with an increase of BRL 63.74 in the total amount spent per customer per month.
# * Keeping all other features fixed, a 1-unit increase in average number of clicks
# per session is associated with a $26.24 increase in total amount spent per customer per month.
# * Keeping all other resources fixed, a 1 unit increase in total time logged into the app
# is associated with an increase of R$38.57 in the total amount spent per customer per month.
# * Keeping all other resources fixed, a 1 unit increase in total time logged into the website
# is associated with an increase of R$0.68 in the total amount spent per customer per month.
# We can see that it will be more profitable for the company to invest in updating its app once the
# return will be higher. In addition, it is important to create policies to retain the customer for a longer time,
# as this also leads to increased sales. The app update itself will be a way to retain the customer for longer.
# It does not pay, at this moment, to invest in updating the website as the return will be minimal.
