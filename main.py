import numpy as np
from sklearn.model_selection import train_test_split

# Assuming X1, X2, and y are your feature and label arrays
X = np.column_stack((X1, X2))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
