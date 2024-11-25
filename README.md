# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
<H3>NAME: SAI DARSHINI R S </H3>
<H3>REGISTER NO.: 212223230178</H3>
<H3>EX. NO.1</H3>
<H3>DATE: 12.08.24</H3>

## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Import Libraries**: Bring in essential libraries such as pandas, numpy, matplotlib, and sklearn.
2. **Load Dataset**: Import the dataset containing car prices along with relevant features.
3. **Data Preprocessing**: Manage missing data and select key features for the model, if required.
4. **Split Data**: Divide the dataset into training and testing subsets.
5. **Train Model**: Build a linear regression model and train it using the training data.
6. **Make Predictions**: Apply the model to predict outcomes for the test set.
7. **Evaluate Model**: Measure the model's performance using metrics like R² score, Mean Absolute Error (MAE), etc.
8. **Check Assumptions**: Plot residuals to verify assumptions like homoscedasticity, normality, and linearity.
9. **Output Results**: Present the predictions and evaluation metrics.

## Program:

```py
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/data/CarPrice_Assignment.csv")

# Display the first few rows of the dataset
print(data.head())

# Select features and target variable
X = data[['horsepower', 'curbweight', 'enginesize', 'highwaympg']]
y = data['price']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Check the assumption of linearity by plotting
plt.figure(figsize=(10, 6))
for i, col in enumerate(['horsepower', 'curbweight', 'enginesize', 'highwaympg']):
    plt.subplot(2, 2, i+1)
    plt.scatter(data[col], data['price'])
    plt.xlabel(col)
    plt.ylabel('Price')
    plt.title(f'Price vs {col}')
plt.tight_layout()
plt.show()

# Check homoscedasticity (residuals vs predictions)
plt.scatter(y_pred, y_test - y_pred)
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Prices')
plt.axhline(0, color='red', linestyle='--')
plt.show()

# Check normality of residuals (histogram of residuals)
plt.hist(y_test - y_pred, bins=30)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals')
plt.show()
```

## Output:
![image](https://github.com/user-attachments/assets/0df379cd-2339-4a3b-b266-95f02733d133)
![image](https://github.com/user-attachments/assets/b777f81a-a8d3-40ef-85db-9e894c284320)
![image](https://github.com/user-attachments/assets/998f2bff-95d9-4953-aa66-a854d236793a)
![image](https://github.com/user-attachments/assets/ceb74ade-8fd1-4744-9306-c48bc4839a37)

## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.

