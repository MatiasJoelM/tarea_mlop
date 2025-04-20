import joblib
import numpy as np

# Ruta al modelo entrenado
model_path = "models/xgb_model.pkl"

# Cargar el modelo
model = joblib.load(model_path)

def test_model_prediction():
    """
    Prueba que el modelo cargado pueda realizar una predicción con éxito.
    """
    # Ejemplo de entrada de prueba
    test_input = np.array([[120, 32.5]])  # Glucose=120, BMI=32.5

    # Realizar la predicción
    prediction = model.predict(test_input)
    probability = model.predict_proba(test_input).max()

    # Verificar que la predicción sea válida
    assert prediction is not None, "La predicción no debe ser None"
    assert probability > 0, "La probabilidad debe ser mayor que 0"

    # Imprimir los resultados
    print(f"Predicción: {int(prediction[0])}")
    print(f"Probabilidad: {float(probability)}")

# Ejecutar la prueba
if __name__ == "__main__":
    test_model_prediction()