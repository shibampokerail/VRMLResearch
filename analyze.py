import pandas as pd
import matplotlib.pyplot as plt

# Load your data from the Excel file
data = pd.read_excel("student_data.xlsx")

# Omit the last row (totals row)
data = data.iloc[:-1]

# Select the columns of interest
selected_columns = data.columns

# Create scatter plots for each pair of columns
read = True
for col in selected_columns:
    if col == "All-Success-Duration":
        read = True
    if read:
        try:
            plt.figure(figsize=(8, 6))
            plt.scatter(data["All-Success-Duration"], data[col], alpha=0.5)
            plt.title(f"Scatter Plot: {col} vs. All-Success-Duration")
            plt.xlabel("All-Success-Duration")
            plt.ylabel(col)
            plt.grid(True)
            plt.show()
        except:
            pass
