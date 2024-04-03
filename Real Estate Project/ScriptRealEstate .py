#!/usr/bin/env python
# coding: utf-8

# In[50]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px



# Load the dataset
csv_file_path = '/Users/fahrkamal/Documents/Data Science Python Data Storage/realtor-data.zip.csv'
df = pd.read_csv(csv_file_path)

df.head(10)


# In[51]:


df.shape


# In[52]:


df.info()


# In[53]:


#Preprocessing

df.drop_duplicates(inplace=True)

# Mode imputation to handle missing value
df['bed'].fillna(df['bed'].mode()[0], inplace=True)
df['bath'].fillna(df['bath'].mode()[0], inplace=True)
df['acre_lot'].fillna(df['acre_lot'].mode()[0], inplace=True)
df['house_size'].fillna(df['house_size'].mode()[0], inplace=True)


# In[54]:


df = df.dropna(subset=['zip_code','city', 'price'])
df = df.drop('prev_sold_date', axis=1)


# In[55]:


df.isnull().sum()


# In[56]:


columns = ['bed', 'bath', 'acre_lot', 'house_size', 'price']
plt.boxplot(df[columns])
plt.xticks(range(1, len(columns) + 1), columns)
plt.title('Boxplot of Features Before Removing Outliers')
plt.show()
print(f'Total Rows Before Removing Outliers: {df.shape[0]}')


# In[57]:



column_num = ['bed', 'bath', 'acre_lot', 'house_size', 'price']

Q1 = df[column_num].quantile(0.25)
Q3 = df[column_num].quantile(0.75)

IQR = Q3 - Q1

outliers = ((df[column_num] < (Q1 - 1.5 * IQR)) | (df[column_num] > (Q3 + 1.5 * IQR)))

df_cleaned = df[~outliers.any(axis=1)]


# In[58]:


columns_to_plot = ['bed', 'bath', 'acre_lot', 'house_size', 'price']

plt.boxplot(df_cleaned[columns_to_plot])

plt.xticks(range(1, len(columns_to_plot) + 1), columns_to_plot)

plt.title('Boxplot of Features After Removing Outliers')

plt.show()

print(f'Total Rows Without Outliers: {df_cleaned.shape[0]}')


# In[59]:


#Distribution by Bed

plt.figure(figsize=(10,10))
ax = sns.countplot(data=df, x='bed')
for i in ax.containers:
  ax.bar_label(i,)
plt.title('Distribution of Bed', fontsize=18)
plt.show()


# In[60]:



plt.figure(figsize=(10, 10))

ax = sns.countplot(data=df, x='bath')

for bar in ax.containers:
    ax.bar_label(bar)

plt.title('Number of Bathrooms Distribution', fontsize=18)

# Display the plot
plt.show()


# In[61]:



city_counts = df['city'].value_counts().reset_index()

city_counts.columns = ['city', 'count']

top_5_cities = city_counts.head(5)

fig = px.bar(top_5_cities, x='city', y='count', color='city', template='plotly')

fig.update_layout(title='Top 5 Cities by Number of Houses')

fig.update_traces(texttemplate='%{y}', textposition='outside')

fig.show()


# In[62]:



top10 = df.groupby('city')['bed'].sum().nlargest(10).reset_index()

fig = px.bar(top10, x='city', y='bed', color='bed', template='plotly', 
             title='Top 10 Cities by Total Bed Count')
fig.update_traces(texttemplate='%{y}', textposition='outside')
fig.show()


# In[63]:


fig = px.scatter(df_cleaned, x='bed', y='bath', size='house_size', color='bed', opacity=1,
                 marginal_x='histogram', marginal_y='histogram', 
                 labels={'bed': 'Bedrooms', 'bath': 'Bathrooms', 'house_size': 'House Size'},
                 title='House Size Distribution by Number of Bedrooms and Bathrooms')
fig.show()


# In[64]:


df_mean = df_cleaned.groupby('price')['house_size'].mean().reset_index()

fig = px.scatter(df_mean, x='price', y='house_size', trendline='ols', 
                 labels={'price':'Price', 'house_size':'Average House Size'})
fig.update_layout(title='Average House Size vs. Price Relationship')
fig.show()


# In[80]:


X = df_cleaned.drop('price', axis = 1)
y = df_cleaned['price']
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2)


# In[84]:


features = df_cleaned.drop('price', axis=1)
target = df_cleaned['price']

train_features, test_features, train_target, test_target = train_test_split(features, target, test_size=0.2, random_state=42)

linear_regression_model = LinearRegression()
linear_regression_model.fit(train_features, train_target)

predictions = linear_regression_model.predict(test_features)

mae_linear = mean_absolute_error(test_target, predictions)
mse_linear = mean_squared_error(test_target, predictions)
r2_linear = r2_score(test_target, predictions)

linear_regression_metrics = {'R2': r2_linear, 'MAE': mae_linear, 'MSE': mse_linear}

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R2 Score: {r2}")


# In[85]:


features = df_cleaned.drop('price', axis=1)
target = df_cleaned['price']

train_features, test_features, train_target, test_target = train_test_split(features, target, test_size=0.2, random_state=42)

random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_model.fit(train_features, train_target)

predictions = random_forest_model.predict(test_features)

mae_rf = mean_absolute_error(test_target, predictions)
mse_rf = mean_squared_error(test_target, predictions)
r2_rf = r2_score(test_target, predictions)

random_forest_metrics = {'R2': r2_rf, 'MAE': mae_rf, 'MSE': mse_rf}

# Printing the performance metrics
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R2 Score: {r2}")


# In[88]:


# Creating a DataFrame
data = {
    'Metric': ['R2', 'MAE', 'MSE'],
    'Random Forest': [r2_rf, mae_rf, mse_rf],
    'Linear Regression': [r2_linear, mae_linear, mse_linear]
}

df_predict = pd.DataFrame(data)

# Plotting
df_predict.plot(x='Metric', y=['Random Forest', 'Linear Regression'], kind='bar', figsize=(10, 6))
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.xlabel('Metrics')
plt.xticks(rotation=0)
plt.yscale('log')
plt.show()


# In[ ]:




