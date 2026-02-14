# ============================================
# API FLASK - PRÉDICTION D'ÉNERGIE SOLAIRE
# ============================================

from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Chargement du modèle au démarrage
print("Chargement du modèle...")
with open('Models/best_model_xgboost.pkl', 'rb') as f:
    model = pickle.load(f)

with open('Models/model_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

print("Modèle chargé avec succès !")

# ============================================
# ROUTE 1 : PAGE D'ACCUEIL
# ============================================
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": " API de Prédiction d'Énergie Solaire",
        "version": "1.0",
        "model": metadata['model_name'],
        "endpoints": {
            "/": "Page d'accueil (cette page)",
            "/info": "Informations sur le modèle",
            "/predict": "Faire une prédiction (POST)",
            "/health": "Vérifier l'état de l'API"
        }
    })

# ============================================
# ROUTE 2 : INFORMATIONS SUR LE MODÈLE
# ============================================
@app.route('/info', methods=['GET'])
def info():
    return jsonify({
        "model_info": {
            "name": metadata['model_name'],
            "type": metadata['model_type'],
            "n_features": metadata['n_features'],
            "training_date": metadata['date']
        },
        "performance": {
            "r2_test": round(metadata['performance']['r2_test'], 4),
            "mae_test": round(metadata['performance']['mae_test'], 2),
            "rmse_test": round(metadata['performance']['rmse_test'], 2)
        },
        "features_required": metadata['features']
    })

# ============================================
# ROUTE 3 : PRÉDICTION
# ============================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupération des données JSON
        data = request.get_json()
        
        # Vérification des features
        missing_features = []
        for feature in metadata['features']:
            if feature not in data:
                missing_features.append(feature)
        
        if missing_features:
            return jsonify({
                "error": "Missing features",
                "missing": missing_features
            }), 400
        
        # Création du DataFrame
        input_data = pd.DataFrame([data])[metadata['features']]
        
        # Prédiction
        prediction = model.predict(input_data)[0]
        
        # Retour de la réponse
        return jsonify({
            "prediction": {
                "value": round(float(prediction), 2),
                "unit": "kW"
            },
            "input_data": data,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

# ============================================
# ROUTE 4 : HEALTH CHECK
# ============================================
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

# ============================================
# LANCEMENT DE L'API
# ============================================
if __name__ == '__main__':
    print("="*60)
    print("Démarrage de l'API Flask...")
    print("="*60)
    print(f"Modèle : {metadata['model_name']}")
    print(f"R² Test : {metadata['performance']['r2_test']:.4f}")
    print(f"Port : 5000")
    print("="*60)
    print("\n Endpoints disponibles :")
    print("   - GET  http://localhost:5000/")
    print("   - GET  http://localhost:5000/info")
    print("   - POST http://localhost:5000/predict")
    print("   - GET  http://localhost:5000/health")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)

    