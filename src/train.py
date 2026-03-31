import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

data = pd.read_csv("../data/student_data.csv")

X = data[['study_hours', 'attendance', 'sleep_hours', 'previous_score']]
y = data['final_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

with open("../model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved!")
