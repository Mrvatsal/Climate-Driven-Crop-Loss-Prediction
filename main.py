
import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("crop_climate_data.csv")
print("Dataset Loaded Successfully\n")
print(df.head()) #loading datasheet(csv file)

le = LabelEncoder()
df['loss_risk'] = le.fit_transform(df['loss_risk'])  #for target variable

X = df.drop('loss_risk', axis=1)
y = df['loss_risk']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)#split target and feature

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)#for model training
print("\nModel Training Completed")

y_pred = model.predict(X_test) # Evaluating model
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


def predict_crop_loss(input_data):
    prediction = model.predict([input_data])
    return le.inverse_transform(prediction)[0]
#made function for prediction

with open("crop_loss_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

#for saving model and encoder

print("\nModel and Encoder Saved Successfully")


#this file is linked by streamlite file and csv file
#we saved two pkl file(encoder) so that we do not need to train the ml everytime we run the code