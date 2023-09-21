import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor




def get_accuracy():
    data = pd.read_excel("student_data.xlsx")


    data = data[
        ["Professors-Duration", "Students-Count", "Students-Duration", "Staff-Count", "Staff-Duration", "Webpage-Count",
         "Webpage-Duration", "Docs-Count", "Docs-Duration", "Video-Count", "Video-Duration", "Informed-Duration.1",
         "All-Success-Duration"]]

    # Print rows with data not available
    # rows_with_nan = data[data.isnull().any(axis=1)]
    # print("Rows with NaN data:")
    # print(rows_with_nan)

    data = data.dropna()

    # print(data)

    predict = "All-Success-Duration"

    X = np.array(data.drop([predict], 1))
    y = np.array(data[predict])

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3)

    linear = linear_model.BayesianRidge()

    linear.fit(x_train, y_train)

    accuracy = linear.score(x_test, y_test)

    print("accuracy:", accuracy)
    return accuracy

    predictions = linear.predict(x_test)

   ## for i in range(len(predictions)):
        # print("Prediction:",predictions[i], " Input:",x_test[i], " Actual_value:",y_test[i])
        #print("Prediction:", predictions[i], " Actual_value:", y_test[i], " loss:", (predictions[i] - y_test[i]))
        ##print("-----------------------")

    # plt.figure(figsize=(8, 6))
    # plt.scatter(predictions, y_test, alpha=0.5)
    # plt.plot(y_test, y_test, label='Fitted Line', color='red')
    # plt.title(f"Prediction  vs. Actual_value")
    # plt.grid(True)
    # plt.show()
sum = 0
for i in range(20):
    sum =+ get_accuracy()


sum = sum * 100
print(sum / 20)