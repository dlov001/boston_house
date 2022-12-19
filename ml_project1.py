import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

# Import Boston House Pricing Dataset

from sklearn.datasets import load_boston

boston_df = load_boston()
boston_df.keys()

# Check the description of the dataset

print(boston_df.DESCR)
print(boston_df.data)

print(boston_df.target)
print(boston_df.feature_names)

# Create the dataset

df = pd.DataFrame(boston_df.data)
df.head()
df.columns = boston_df.feature_names
df.head()

df['Price'] = boston_df.target
df.head()

df.info()

# Summarize the data

df.describe()

# Check for missing values

for i in df.columns:
    if df[i].isna().sum() != 0:
        print(f'Missing values in {i}')
    else:
        print(f'No missing values in {i}')
        
# Exploratory Data Analysis

sb.heatmap(df.corr())
plt.show()

sb.pairplot(df)
plt.show()

# Analyzing the Correlated Features

plt.scatter(df['CRIM'], df['Price'])
plt.xlabel('Crime Rate')
plt.ylabel('Price')
plt.show()

len(df.columns)

for i in df.columns:
    sb.regplot(x = df[i], y = 'Price', data = df)
    plt.show()
    
# Split the data

x = df.drop('Price', axis = 1)
y = df['Price']

x.head()
y.head()

# Train the data

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

x_train.shape
x_test.shape

# Standardize data

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train, x_test = scaler.fit_transform(x_train),scaler.fit_transform(x_test)

import pickle
pickle.dump(scaler, open('scaling.pkl', 'wb'))

# Model Training

from sklearn.linear_model import LinearRegression

regression = LinearRegression()

lin_mod = regression.fit(x_train, y_train)

print(lin_mod.coef_)
lin_mod.get_params()

lin_pred = lin_mod.predict(x_test)

pd.crosstab(y_test, lin_pred)

# Test Assumptions

plt.scatter(y_test, lin_pred)
plt.show()

resid = y_test - lin_pred
resid.head()

sb.displot(resid, kind = 'kde')
plt.show()

# Linear and normal ffrom what we can see

# Scatter plot with respect to prediction and residuals

plt.scatter(lin_pred, resid)
plt.show()

# Measuring goodness of fit

from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

print(mean_absolute_error(y_test, lin_pred))
print(mean_squared_error(y_test, lin_pred))
print(np.sqrt(mean_squared_error(y_test, lin_pred)))

score = r2_score(y_test, lin_pred)
print(score)

# Adjusted R2

adj_r2 = 1 - (1 - score) * (len(y_test) - 1)/ (len(y_test) - x_test.shape[1] - 1)
print(adj_r2)

# New Data Prediction

new = boston_df.data[0].reshape(1,-1)

# Transform the new data

new = scaler.transform(new)

regression.predict(new)

# Pickling the Model file for Deployment

pickle.dump(regression,open('regmodel.pkl','wb'))

pickled_model = pickle.load(open('regmodel.pkl','rb'))

pickled_model.predict(new)

