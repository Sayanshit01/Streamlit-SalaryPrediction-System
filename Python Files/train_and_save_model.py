import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ✅ Step 1: Load Data
data = pd.read_csv('Updated Dataset.csv')  # Adjust to your CSV file name

# Example using Experience vs Salary:
data = data[['YearsCodePro', 'ConvertedCompYearly']].dropna()
X = data[['YearsCodePro']]
y = data['ConvertedCompYearly']

# ✅ Step 2: Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# ✅ Step 3: Create Model Folder If Not Exists
import os
if not os.path.exists('../Model'):
    os.makedirs('../Model')

# ✅ Step 4: Save Model
with open('../Model/model_3.pkl', 'wb') as file:
    pickle.dump(model, file)

print("✅ Model trained and saved as model_3.pkl in Model folder.")
