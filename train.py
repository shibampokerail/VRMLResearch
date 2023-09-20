import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datas import *
import joblib

# Step 1: Load and prepare your data (if not already done)
data = pd.DataFrame({'Input_data': STUDENTS_DURATION, 'Engagement': ALL_SUCCESS_DURATION})

# Step 2: Split the data into training and testing sets
X = data[['Input_data']]  # Features (input)
y = data['Engagement']              # Target variable (output)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Choose a regression model (Linear Regression in this case)
model = LinearRegression()

# Step 4: Train the model
model.fit(X_train, y_train)

# Step 5: Make predictions on the test data
y_pred = model.predict(X_test)

# Step 6: Evaluate the model's performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# print(f"Mean Absolute Error: {mae}")
# print(f"Mean Squared Error: {mse}")
# print(f"R-squared: {r2}")

joblib.dump(model, 'engagement_model.pkl')

loaded_model = joblib.load('engagement_model.pkl')

for i in range(100,300):
    new_student_Input_data = i

    new_student_engagement = loaded_model.predict([[new_student_Input_data]])

    print(new_student_engagement)