import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, d2_absolute_error_score, median_absolute_error

data = pd.read_csv('US_House_Price.csv')

x = data['DATE'] = pd.to_datetime(data['DATE']).dt.year
y = data['urban_population']

# Year vs Population graphical representation using matplotlib and seaborn:

# Year vs Population line-plot using matplotlib:

plt.plot(x, y)
plt.title('Population Growth with Years')
plt.xlabel('Year')
plt.ylabel('Population')
plt.show()

# Year vs Population Scatter-plot using matplotlib:

plt.scatter(x, y, color='maroon', edgecolors='green', alpha=0.6)
plt.title('Population Growth with Years', fontsize=18)
plt.xlabel('Year')
plt.xticks(rotation=40)
plt.ylabel('Population')
plt.show()

# Year vs Population Bar-plot using matplotlib:

plt.bar(x, y, color='green', width=0.4)
plt.title('Population Growth with Years', fontsize=18)
plt.xlabel('Year')
plt.xticks(rotation=40)
plt.ylabel('Population')
plt.show()

# Year vs Population line-plot using seaborn:

sns.set(style='whitegrid')
sns.lineplot(data=data, x='DATE', y='urban_population', color='red', )
plt.title('Population Growth with Years', fontsize=18)
plt.xlabel('Year')
plt.xticks(rotation=40)
plt.ylabel('Population')
plt.show()

# Year vs Population scatter-plot using seaborn:

sns.set(style='whitegrid')
sns.scatterplot(data=data, x='DATE', y='urban_population', hue='income')
plt.title('Population Growth with Years', fontsize=18)
plt.xlabel('Year')
plt.xticks(rotation=40)
plt.ylabel('Population')
plt.show()

# Year vs Delinquency rate graphical representation using matplotlib and seaborn:

y = data['delinquency_rate']

# Year vs Delinquency rate line-plot using matplotlib:

plt.plot(x, y)
plt.title('Delinquency rate with Years')
plt.xlabel('Year')
plt.ylabel('Delinquency rate')
plt.show()

# Year vs Delinquency rate Scatter-plot using matplotlib:

plt.scatter(x, y, color='red', edgecolors='blue', alpha=0.7)
plt.title('Delinquency rate with Years', fontsize=12)
plt.xlabel('Year')
plt.xticks(rotation=45)
plt.ylabel('Delinquency rate')
plt.show()

# Year vs Delinquency rate Bar-plot using matplotlib:

plt.bar(x, y, color='maroon', width=0.6)
plt.title('Delinquency rate with Years', fontsize=12)
plt.xlabel('Year')
plt.xticks(rotation=45)
plt.ylabel('Delinquency rate')
plt.show()

# Year vs Delinquency rate line-plot using seaborn:

sns.set(style='whitegrid')
sns.lineplot(data=data, x='DATE', y='delinquency_rate', color='blue', )
plt.title('Delinquency rate with Years', fontsize=12)
plt.xlabel('Year')
plt.xticks(rotation=40)
plt.ylabel('Delinquency rate')
plt.show()

# Year vs Delinquency rate scatter-plot using seaborn:

sns.set(style='whitegrid')
sns.scatterplot(data=data, x='DATE', y='delinquency_rate', hue='GDP')
plt.title('Delinquency rate with Years', fontsize=12)
plt.xlabel('Year')
plt.xticks(rotation=40)
plt.ylabel('Delinquency rate')
plt.show()

# Year vs Delinquency rate box-plot using seaborn:

sns.set(style='whitegrid')
sns.boxplot(data=data, x='DATE', y='delinquency_rate')
plt.title('Delinquency rate with Years', fontsize=12)
plt.xlabel('Year')
plt.xticks(rotation=40)
plt.ylabel('Delinquency rate')
plt.show()

# Year vs delinquency rate Joint-plot using seaborn:

sns.set(style='whitegrid')
sns.jointplot(data=data, x='DATE', y='delinquency_rate', color='purple')
plt.title('Population Growth with Years', fontsize=12)
plt.xlabel('Year')
plt.xticks(rotation=40)
plt.ylabel('Population')
plt.tight_layout()  # To overcome overlapping in graph
plt.show()

# Year vs delinquency_rate dis-plot using seaborn:

sns.set(style='whitegrid')
sns.displot(data=data, x='DATE', y='delinquency_rate', color='brown')
plt.title('Population Growth with Years', fontsize=16)
plt.xlabel('Year')
plt.xticks(rotation=40)
plt.ylabel('Population')
plt.show()

# Subplot: Time vs Population, Income, Unemployment rate, GDP

y1 = data['urban_population']
y2 = data['income']
y3 = data['unemployment_rate']
y4 = data['GDP']

plt.figure(4, figsize=(6, 12))
plt.title('Time vs Population, Income, Unemployment rate, GDP')

plt.subplot(411)
plt.scatter(x, y1, c='green', alpha=0.1)
plt.xlabel('Time')
plt.xticks(rotation=45)
plt.ylabel('Population')

plt.subplot(412)
plt.bar(x, y2, color='purple', width=0.6)
plt.xlabel('Time')
plt.xticks(rotation=45)
plt.ylabel('Income')

plt.subplot(413)
plt.plot(x, y3, color='maroon')
plt.xlabel('Time')
plt.xticks(rotation=45)
plt.ylabel('Unemployment rate')

plt.subplot(414)
plt.scatter(x, y4, c='blue', alpha=0.4)
plt.xlabel('Time')
plt.xticks(rotation=45)
plt.ylabel('GDP')

plt.tight_layout()
plt.show()

# Subplot: Time vs Cost price index, Delinquency rate, Mortgage rate, Interest rates, Subsidies

y1 = data['const_price_index']
y2 = data['delinquency_rate']
y3 = data['mortgage_rate']
y4 = data['interest_rate']
y5 = data['housing_subsidies']

plt.figure(5, figsize=(10, 10))
plt.title('Time vs Cost price index, Delinquency rate, Mortgage rate, Interest rates, Subsidies')

plt.subplot(511)
plt.scatter(x, y1, c='green', alpha=0.1)
plt.ylabel('Cost price Index')

plt.subplot(512)
plt.bar(x, y2, color='purple', width=0.6)
plt.ylabel('Delinquency rate')

plt.subplot(513)
plt.plot(x, y3, color='maroon')
plt.ylabel('Mortgage rate')

plt.subplot(514)
plt.scatter(x, y4, c='blue', alpha=0.4)
plt.ylabel('Interest rate')

plt.subplot(515)
plt.bar(x, y5, color='brown', width=0.3)
plt.ylabel('Subsidies rate')

plt.xlabel('Time')
plt.tight_layout()
plt.show()

# Liner regression on income and GDP:

x = data[['income']]
y = data[['GDP']]
print(x)
print(y)
print(x.shape)
print(y.shape)

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3)
print(xTrain.shape)
print(xTest.shape)
print(yTrain.shape)
print(yTest.shape)

lr = LinearRegression()
lr.fit(xTrain, yTrain)
yPredict = lr.predict(xTest)
print(yPredict.shape)

# To plot sns plots we need to make dataframe of this regression:

data = {}
data['yTest'] = np.reshape(yTest, (-1))
data['yPredict'] = np.reshape(np.array(yPredict), (-1))
print(data)
df = pd.DataFrame(data=data)
print(df)

# reg-plot of prediction of income and GDP:

sns.regplot(data=df, x='yTest', y='yPredict', color='brown')
plt.title('Regplot between yTest and yPredict')
plt.show()

mse = mean_squared_error(yTest, yPredict)
print(mse)

d2error = d2_absolute_error_score(yTest, yPredict)
print(d2error)

medianError = median_absolute_error(yTest, yPredict)
print(medianError)

accScore = r2_score(yTest, yPredict)
print(accScore)

# Liner regression on GDP and Urban population:

x = data[['GDP']]
y = data[['urban_population']]
print(x)
print(y)
print(x.shape)
print(y.shape)

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.35)
print(xTrain.shape)
print(xTest.shape)
print(yTrain.shape)
print(yTest.shape)

lr = LinearRegression()
lr.fit(xTrain, yTrain)
yPredict = lr.predict(xTest)
print(yPredict.shape)

# To make dataframe

data = {}
data['yTest'] = np.reshape(yTest, (-1))
data['yPredict'] = np.reshape(np.array(yPredict), (-1))
print(data)
df = pd.DataFrame(data=data)
print(df)

# reg plot of GDP and Urban population:

sns.regplot(data=df, x='yTest', y='yPredict', color='maroon')
plt.title('Regplot between yTest and yPredict')
plt.show()

mse = mean_squared_error(yTest, yPredict)
print(mse)

d2error = d2_absolute_error_score(yTest, yPredict)
print(d2error)

medianError = median_absolute_error(yTest, yPredict)
print(medianError)

accScore = r2_score(yTest, yPredict)
print(accScore)
