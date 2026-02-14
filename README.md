# 🌞 Prédiction de Production d'Énergie Solaire

Projet de Machine Learning pour prédire la production d'énergie solaire à partir de données météorologiques.

## 📊 Résumé du Projet

- **Dataset** : 4,213 observations avec 21 variables météorologiques
- **Modèle final** : XGBoost Optimisé
- **Performance** : R² = 0.8172 (81.7% de variance expliquée)
- **MAE** : 263.74 kW
- **RMSE** : 408.66 kW



## 📓 Notebooks Jupyter

### 1️⃣ Data Preparation (`01_data_preparation.ipynb`)
- Chargement et exploration des données
- Nettoyage et traitement des valeurs manquantes
- Feature Engineering (7 nouvelles features)
- Analyse exploratoire (EDA)
- Export : `cleaned_data.csv`

### 2️⃣ Modeling (`02_modeling.ipynb`)
- Comparaison de 4 modèles : Linear Regression, Random Forest, XGBoost, KNN
- Optimisation des hyperparamètres (GridSearchCV)
- Sélection du meilleur modèle (XGBoost)
- Export : `best_model_xgboost.pkl`

### 3️⃣ Evaluation (`03_evaluation.ipynb`)
- Analyse détaillée des prédictions
- Feature importance
- Tests sur différents scénarios
- Génération du rapport final

## 🌐 API Flask

### Lancer l'API
```bash
python app.py
```

L'API sera accessible sur `http://localhost:5000`

### Endpoints

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Page d'accueil |
| `/info` | GET | Informations sur le modèle |
| `/health` | GET | Vérifier l'état de l'API |
| `/predict` | POST | Faire une prédiction |

### Exemple de requête
```python
import requests

data = {
    "zenith": 30.0,
    "angle_of_incidence": 20.0,
    "total_cloud_cover_sfc": 0.0,
    # ... (20 features au total)
}

response = requests.post("http://localhost:5000/predict", json=data)
print(response.json())
```

### Tester l'API
```bash
python test_api_v2.py
```

## 📈 Features Importantes

| Rang | Feature | Importance |
|------|---------|-----------|
| 1 | angle_of_incidence | 15.24% |
| 2 | total_cloud_cover_sfc | 13.99% |
| 3 | total_cloud_all_layers | 11.79% |
| 4 | total_precipitation_sfc | 7.34% |
| 5 | zenith | 5.80% |

## 📊 Résultats

### Comparaison des Modèles (après optimisation)

| Modèle | MAE (kW) | RMSE (kW) | R² |
|--------|----------|-----------|-----|
| **XGBoost** | **263.74** | **408.66** | **0.8172** |
| Random Forest | 296.18 | 423.73 | 0.8034 |
| KNN | 349.55 | 498.84 | 0.7276 |
| Linear Regression | 451.59 | 564.50 | 0.6512 |

### Scénarios de Production

- ☀️ **Conditions idéales** : ~2,016 kW
- ⛅ **Conditions moyennes** : ~1,048 kW
- ☁️ **Conditions défavorables** : ~247 kW


## 👨‍💻 Auteur

SIVASARANAM Sajin - Projet Machine Learning

