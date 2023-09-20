import pandas as pd
from datas import *

data = pd.DataFrame({'Input_data': STUDENTS_DURATION, 'Engagement': ALL_SUCCESS_DURATION})

from sklearn.model_selection import train_test_split

X = data[['Input_data']]  # Features (input)
y = data['Engagement']              # Target variable (output)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

new_Input_data = [1]
new_predictions = model.predict(pd.DataFrame({'Input_data': new_Input_data}))

import matplotlib.pyplot as plt

plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('Interaction Duration')
plt.ylabel('Engagement')
plt.legend()
plt.show()
