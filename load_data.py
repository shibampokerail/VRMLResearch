import pandas as pd
import openpyxl
# Load the Excel file
file_path = 'student_data.xlsx'
df = pd.read_excel(file_path)

# Extract data from specific columns
interaction_duration = df['Success-Duration']
print(interaction_duration)

