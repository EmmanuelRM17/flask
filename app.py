import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

def load_model():
    # Cargar el modelo entrenado desde el archivo (buscar model.pkl primero)
    try:
        model = joblib.load('model.pkl')
        print("Modelo cargado desde model.pkl")
        return model
    except FileNotFoundError:
        try:
            model = joblib.load('modelo_insectos.pkl')
            print("Modelo cargado desde modelo_insectos.pkl")
            return model
        except FileNotFoundError:
            print("Error: No se encontró ningún archivo de modelo")
            return None

# Cargar modelo al iniciar la aplicación
model = load_model()
if model is not None:
    print("✅ Aplicación lista con modelo cargado")
else:
    print("❌ Aplicación iniciada sin modelo")

@app.route('/')
def home():
    # Renderizar el template HTML
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Verificar que el modelo esté cargado
    if model is None:
        return jsonify({
            'error': 'Modelo no encontrado. Verifica que el archivo model.pkl o modelo_insectos.pkl esté en la carpeta.'
        })
    
    try:
        # Obtener datos del formulario (no JSON)
        abdomen = request.form.get('abdomen')
        antena = request.form.get('antena')
        
        if not abdomen or not antena:
            return jsonify({'error': 'Faltan datos del formulario'})
        
        # Convertir a float
        abdomen_val = float(abdomen)
        antena_val = float(antena)
        
        if abdomen_val <= 0 or antena_val <= 0:
            return jsonify({'error': 'Los valores deben ser mayores a 0'})
        
        # Preparar datos para predicción
        features = np.array([[abdomen_val, antena_val]])
        
        # Realizar predicción
        prediction = model.predict(features)[0]
        
        # Devolver respuesta con 'categoria' (no 'prediction')
        return jsonify({
            'categoria': prediction,
            'abdomen': abdomen_val,
            'antena': antena_val
        })
        
    except ValueError as e:
        return jsonify({'error': 'Valores inválidos. Asegúrate de ingresar números válidos.'})
    except Exception as e:
        return jsonify({'error': f'Error interno: {str(e)}'})


@app.route('/health')
def health():
    # Endpoint para verificar el estado de la aplicación
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)