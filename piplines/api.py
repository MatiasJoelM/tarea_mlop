from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Crear la aplicación FastAPI
app = FastAPI()

# Cargar el modelo entrenado
model_path = "models/xgb_model.pkl"
model = joblib.load(model_path)

# Definir el esquema de entrada usando Pydantic
class InputData(BaseModel):
    Glucose: float
    BMI: float

# Ruta raíz
@app.get("/")
def read_root():
    return {"message": "API para predicciones con el modelo XGBClassifier"}

# Ruta para realizar predicciones
@app.post("/predict")
def predict(data: InputData):
    # Convertir los datos de entrada en un array numpy
    input_array = np.array([[data.Glucose, data.BMI]])
    
    # Realizar la predicción
    prediction = model.predict(input_array)
    probability = model.predict_proba(input_array).max()

    # Retornar el resultado
    return {
        "prediction": int(prediction[0]),
        "probability": float(probability)
    }