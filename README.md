# Housing-Data-Regression

#Import Libraries and sklearn

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge 


#Specify the full path to your CSV file
file_name = "//Users/reedlabar/Documents/Python /Python Project (IBM Course)/kc_house_data_NaN.csv"

#Load the CSV into a Pandas DataFrame
df = pd.read_csv(file_name)

#Display the first few rows of the DataFrame
print(df.head())

#Statistical Summary of the DF
df.describe()


#Drop the columns "id" and "Unnamed: 0"
df.drop(columns=["id", "Unnamed: 0"], axis=1, inplace=True)

#Display the statistical summary of the data
print(df.describe())

### HANDLING MISSING VALUES 

print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())

#Replace the missing bedroom values with mean

mean=df['bedrooms'].mean()
df['bedrooms'] = df['bedrooms'].fillna(df['bedrooms'].mean())

#Replace the missing bathroom values with mean
mean=df['bathrooms'].mean()
df['bathrooms'].replace(np.nan,mean, inplace=True)

print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())

### EXPLORATORY ANALYSIS

#Count the number of unique floor values
floor_counts = df['floors'].value_counts()

#Convert the result to df
floor_counts_df = floor_counts.to_frame()

#Display the df
print(floor_counts_df)


# Create a regression plot for sqft_above vs. price
plt.figure(figsize=(10, 6))
sns.regplot(x="sqft_above", y="price", data=df, scatter_kws={"alpha": 0.5}, line_kws={"color": "red"})
plt.title("Regression Plot: sqft_above vs. price")
plt.xlabel("Square Footage Above Ground (sqft_above)")
plt.ylabel("House Price (price)")
plt.show()

#Drop the 'date' column from the DataFrame
df = df.drop(columns=['date'], errors='ignore')

#Find feature most highly correlated to price using a correlation matrix

df.corr()['price'].sort_values()

### VISUALIZATION USING HEATMAP

#Create the correlation matrix
correlation_matrix = df.corr()

#Create a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)

#Add a title to the heatmap
plt.title("Correlation Heatmap", fontsize=16)

#Show the plot
plt.show()

### MODEL BUILDING
#Fit model to long and price and test R^2
X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm.fit(X,Y)
lm.score(X, Y)

# Fit model using sqft_living
X = df[['sqft_living']]
lm.fit(X,Y)
lm.score(X,Y)

# Final Model 
x = df[["floors", "waterfront", "lat", "bedrooms", "sqft_basement", 
        "view", "bathrooms", "sqft_living15", "sqft_above", "grade", "sqft_living"]]
y = df["price"]

#Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=1)

#Create the pipeline
pipeline = Pipeline([
    ('scale', StandardScaler()),
    ('polynomial', PolynomialFeatures(degree=2, include_bias=False)),
    ('model', LinearRegression())
])

#Fit the pipeline on the training data
pipeline.fit(x_train, y_train)

# Calculate the R^2 score on the test data
r2 = pipeline.score(x_test, y_test)

# Print the number of samples and R^2 value
print("Number of test samples:", x_test.shape[0])
print("Number of training samples:", x_train.shape[0])
print(f"R^2 Score (Test Data): {r2}")



