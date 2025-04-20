# Load the dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os
import joblib

# Get the current working directory
current_dir = os.getcwd()
# Get the parent directory
parent_dir = os.path.dirname(current_dir)


# Load the dataset

data = pd.read_csv('/input/diabetes.csv')
X = (data.drop(['Pregnancies','BloodPressure','DiabetesPedigreeFunction', 'Age','SkinThickness','Insulin'], axis=1)).iloc[:, :-1]
y = (data.drop(['Pregnancies','BloodPressure','DiabetesPedigreeFunction', 'Age','SkinThickness','Insulin'], axis=1)).iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
imputer = SimpleImputer(missing_values=0,strategy='median')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

clf3 = XGBClassifier()
clf3.fit(X_train, y_train)
print('\n\nAccuracy of XGBClassifier in Full Feature Space: {:.2f}'.format(clf3.score(X_test, y_test)))
columns = X.columns
coefficients = clf3.feature_importances_.reshape(X.columns.shape[0], 1)
absCoefficients = abs(coefficients)
fullList = pd.concat((pd.DataFrame(columns, columns = ['Variable']), pd.DataFrame(absCoefficients, columns = ['absCoefficient'])), axis = 1).sort_values(by='absCoefficient', ascending = False)
print('XGBClassifier - Feature Importance:')
print('\n',fullList,'\n')

# Guardar el modelo en formato .pkl dentro del directorio /app/models
model_dir = os.path.join(current_dir, 'models')  
os.makedirs(model_dir, exist_ok=True)  

model_file = os.path.join(model_dir, 'xgb_model.pkl')  # Ruta completa al archivo del modelo

joblib.dump(clf3, model_file)  # Guarda el modelo

print(f"Modelo guardado en: {model_file}")