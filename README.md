# CodeAlpha_CreditScoring

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample credit scoring dataset
data = {
    "income": [25000, 40000, 50000, 60000, 30000,
               70000, 80000, 35000, 45000, 90000],
    "age": [22, 30, 35, 40, 25, 45, 50, 28, 32, 55],
    "loan_amount": [10000, 15000, 20000, 25000, 12000,
                    30000, 35000, 14000, 18000, 40000],
    "credit_score": [600, 650, 700, 720, 610,
                     750, 780, 620, 680, 800],
    "approved": [0, 1, 1, 1, 0, 1, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

X = df.drop("approved", axis=1)
y = df["approved"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Credit Scoring Model")
print("Accuracy:", accuracy)

new_customer = [[55000, 33, 18000, 690]]
result = model.predict(new_customer)

print("Loan Approval Prediction:",
      "Approved" if result[0] == 1 else "Not Approved")
