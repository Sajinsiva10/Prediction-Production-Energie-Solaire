# ============================================
# TEST DE L'API - VERSION CORRIGÉE
# ============================================

import requests
import json
import pandas as pd

BASE_URL = "http://127.0.0.1:5000"

print("="*60)
print("🧪 TESTS DE L'API FLASK - VERSION CORRIGÉE")
print("="*60)

# ============================================
# ÉTAPE 1 : Récupérer les features exactes
# ============================================
print("\n📍 Récupération des features requises...")
print("-"*60)
response = requests.get(f"{BASE_URL}/info")
features_required = response.json()['features_required']

print(f"✅ {len(features_required)} features requises :")
for i, feat in enumerate(features_required, 1):
    print(f"   {i:2d}. {feat}")

# ============================================
# ÉTAPE 2 : Charger une vraie ligne de données
# ============================================
print("\n📍 Chargement d'une vraie observation...")
print("-"*60)

# Charger le dataset
df = pd.read_csv('Data/cleaned_data.csv')
X = df.drop('generated_power_kw', axis=1)
y = df['generated_power_kw']

# Prendre la première ligne comme exemple
sample_data = X.iloc[0].to_dict()
real_value = y.iloc[0]

print(f"✅ Observation chargée (valeur réelle : {real_value:.2f} kW)")

# ============================================
# TEST 1 : Prédiction avec données réelles
# ============================================
print("\n📍 TEST 1 : POST /predict - Données réelles")
print("-"*60)

response = requests.post(f"{BASE_URL}/predict", json=sample_data)
print(f"Status Code : {response.status_code}")

if response.status_code == 200:
    result = response.json()
    predicted = result['prediction']['value']
    print(f"\n✅ Prédiction réussie !")
    print(f"   Valeur réelle    : {real_value:.2f} kW")
    print(f"   Valeur prédite   : {predicted:.2f} kW")
    print(f"   Erreur absolue   : {abs(real_value - predicted):.2f} kW")
    print(f"   Erreur relative  : {abs(real_value - predicted) / real_value * 100:.2f}%")
else:
    error = response.json()
    print(f"❌ Erreur : {error}")

# ============================================
# TEST 2 : Scénario IDÉAL (conditions optimales)
# ============================================
print("\n📍 TEST 2 : POST /predict - ☀️ Scénario IDÉAL")
print("-"*60)

# Partir d'une vraie ligne et modifier les valeurs clés
ideal_data = X.iloc[0].to_dict()

# Modifier les features importantes pour un scénario idéal
ideal_data['zenith'] = 30.0  # Soleil haut
ideal_data['angle_of_incidence'] = 20.0  # Bon angle
ideal_data['total_cloud_cover_sfc'] = 0.0  # Ciel dégagé
ideal_data['total_cloud_all_layers'] = 0  # Pas de nuages
ideal_data['shortwave_radiation_backwards_sfc'] = 800.0  # Forte radiation
ideal_data['low_zenith'] = 1  # Indicateur positif
ideal_data['total_precipitation_sfc'] = 0.0  # Pas de pluie
ideal_data['total_precip'] = 0.0  # Pas de précipitations
ideal_data['relative_humidity_2_m_above_gnd'] = 30  # Faible humidité
ideal_data['temperature_2_m_above_gnd'] = 25.0  # Température agréable

response = requests.post(f"{BASE_URL}/predict", json=ideal_data)
print(f"Status Code : {response.status_code}")

if response.status_code == 200:
    result = response.json()
    print(f"\n☀️ Production prévue (conditions idéales) : {result['prediction']['value']:.2f} kW")
else:
    error = response.json()
    print(f"❌ Erreur : {error}")

# ============================================
# TEST 3 : Scénario DÉFAVORABLE
# ============================================
print("\n📍 TEST 3 : POST /predict - ☁️ Scénario DÉFAVORABLE")
print("-"*60)

bad_data = X.iloc[0].to_dict()

# Conditions défavorables
bad_data['zenith'] = 85.0  # Soleil très bas
bad_data['angle_of_incidence'] = 80.0  # Mauvais angle
bad_data['total_cloud_cover_sfc'] = 100.0  # Très nuageux
bad_data['total_cloud_all_layers'] = 300  # Nuages partout
bad_data['shortwave_radiation_backwards_sfc'] = 0.0  # Pas de radiation
bad_data['low_zenith'] = 0  # Indicateur négatif
bad_data['total_precipitation_sfc'] = 10.0  # Pluie
bad_data['total_precip'] = 10.0  # Précipitations
bad_data['relative_humidity_2_m_above_gnd'] = 95  # Forte humidité
bad_data['temperature_2_m_above_gnd'] = 5.0  # Froid

response = requests.post(f"{BASE_URL}/predict", json=bad_data)
print(f"Status Code : {response.status_code}")

if response.status_code == 200:
    result = response.json()
    print(f"\n☁️ Production prévue (conditions défavorables) : {result['prediction']['value']:.2f} kW")
else:
    error = response.json()
    print(f"❌ Erreur : {error}")

# ============================================
# TEST 4 : Plusieurs prédictions
# ============================================
print("\n📍 TEST 4 : POST /predict - 5 prédictions aléatoires")
print("-"*60)

sample_indices = [0, 100, 500, 1000, 2000]
errors = []

for idx in sample_indices:
    data = X.iloc[idx].to_dict()
    real = y.iloc[idx]
    
    response = requests.post(f"{BASE_URL}/predict", json=data)
    
    if response.status_code == 200:
        predicted = response.json()['prediction']['value']
        error = abs(real - predicted)
        errors.append(error)
        print(f"   Obs {idx:4d} : Réel={real:7.2f} kW | Prédit={predicted:7.2f} kW | Erreur={error:6.2f} kW")

if errors:
    print(f"\n📊 Statistiques :")
    print(f"   MAE moyenne : {sum(errors) / len(errors):.2f} kW")
    print(f"   Erreur min  : {min(errors):.2f} kW")
    print(f"   Erreur max  : {max(errors):.2f} kW")

print("\n" + "="*60)
print("✅ TOUS LES TESTS TERMINÉS !")
print("="*60)