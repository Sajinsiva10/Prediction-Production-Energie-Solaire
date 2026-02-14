# ============================================
# INTERFACE UTILISATEUR - PRÉDICTION SOLAIRE
# ============================================

import requests
import pandas as pd

print("="*70)
print("PRÉDICTEUR DE PRODUCTION D'ÉNERGIE SOLAIRE")
print("="*70)
print("\nInterface simplifiée : Entrez seulement les 5 features principales")
print("   Les autres seront automatiquement remplies avec des valeurs par défaut\n")

# ============================================
# CHARGEMENT DES VALEURS PAR DÉFAUT
# ============================================
print("Chargement des valeurs par défaut")

# Charger le dataset pour obtenir les valeurs médianes
try:
    df = pd.read_csv('Data/cleaned_data.csv')
    X = df.drop('generated_power_kw', axis=1)
    default_values = X.median().to_dict()
    print("✅ Valeurs par défaut chargées\n")
except:
    print("⚠️  Impossible de charger le dataset. Utilisation de valeurs fixes.\n")
    default_values = {
        "zenith": 62.79,
        "angle_of_incidence": 45.0,
        "shortwave_radiation_backwards_sfc": 371.13,
        "low_zenith": 0,
        "radiation_temp_interaction": 3000.0,
        "humidity_temp_ratio": 0.15,
        "relative_humidity_2_m_above_gnd": 60,
        "total_cloud_cover_sfc": 8.70,
        "total_cloud_all_layers": 50,
        "low_cloud_cover_low_cld_lay": 20,
        "medium_cloud_cover_mid_cld_lay": 15,
        "temperature_2_m_above_gnd": 15.0,
        "wind_speed_80_m_above_gnd": 8.0,
        "mean_sea_level_pressure_MSL": 1013.0,
        "high_cloud_cover_high_cld_lay": 15,
        "wind_gust_10_m_above_gnd": 10.0,
        "total_precip": 0.5,
        "avg_wind_speed": 7.0,
        "total_precipitation_sfc": 0.5,
        "wind_speed_900_mb": 6.0
    }

# ============================================
# SAISIE DES 5 FEATURES PRINCIPALES
# ============================================
print("="*70)
print("ENTREZ LES VALEURS DES 5 FEATURES PRINCIPALES")
print("="*70)
print("Conseil : Appuyez sur Entrée pour utiliser la valeur par défaut\n")

# Feature 1 : Zenith
print("ZENITH (angle du soleil)")
print("    → Valeur basse (20-40°) = soleil haut = bonne production")
print("    → Valeur haute (70-90°) = soleil bas = faible production")
zenith_input = input(f"    Entrez le zenith (défaut: {default_values['zenith']:.2f}°) : ").strip()
zenith = float(zenith_input) if zenith_input else default_values['zenith']

# Feature 2 : Angle of incidence
print("\nANGLE D'INCIDENCE")
print("    → Valeur basse (10-30°) = bon angle = bonne production")
print("    → Valeur haute (60-80°) = mauvais angle = faible production")
angle_input = input(f"    Entrez l'angle d'incidence (défaut: {default_values['angle_of_incidence']:.2f}°) : ").strip()
angle_of_incidence = float(angle_input) if angle_input else default_values['angle_of_incidence']

# Feature 3 : Total cloud cover
print("\nCOUVERTURE NUAGEUSE TOTALE")
print("    → 0% = ciel dégagé = excellente production")
print("    → 100% = très nuageux = faible production")
cloud_input = input(f"    Entrez la couverture nuageuse (défaut: {default_values['total_cloud_cover_sfc']:.2f}%) : ").strip()
total_cloud_cover_sfc = float(cloud_input) if cloud_input else default_values['total_cloud_cover_sfc']

# Feature 4 : Shortwave radiation
print("\nRADIATION SOLAIRE (shortwave radiation)")
print("    → 0-200 = très faible")
print("    → 200-600 = moyenne")
print("    → 600-1000 = forte = excellente production")
radiation_input = input(f"    Entrez la radiation (défaut: {default_values['shortwave_radiation_backwards_sfc']:.2f}) : ").strip()
shortwave_radiation = float(radiation_input) if radiation_input else default_values['shortwave_radiation_backwards_sfc']

# Feature 5 : Temperature
print("\nTEMPÉRATURE (2m au-dessus du sol)")
print("    → Température optimale pour les panneaux : 15-25°C")
temp_input = input(f"    Entrez la température (défaut: {default_values['temperature_2_m_above_gnd']:.2f}°C) : ").strip()
temperature = float(temp_input) if temp_input else default_values['temperature_2_m_above_gnd']

# ============================================
# CRÉATION DES DONNÉES COMPLÈTES
# ============================================
print("\n" + "="*70)
print("PRÉPARATION DES DONNÉES...")
print("="*70)

# Partir des valeurs par défaut
data = default_values.copy()

# Remplacer avec les valeurs saisies
data['zenith'] = zenith
data['angle_of_incidence'] = angle_of_incidence
data['total_cloud_cover_sfc'] = total_cloud_cover_sfc
data['shortwave_radiation_backwards_sfc'] = shortwave_radiation
data['temperature_2_m_above_gnd'] = temperature

# Calculer les features dérivées
data['low_zenith'] = 1 if zenith < 50 else 0
data['radiation_temp_interaction'] = shortwave_radiation * temperature
data['total_cloud_all_layers'] = int(total_cloud_cover_sfc * 3)  # Estimation

print("\n Données préparées :")
print(f"   • Zenith                : {data['zenith']:.2f}°")
print(f"   • Angle d'incidence     : {data['angle_of_incidence']:.2f}°")
print(f"   • Couverture nuageuse   : {data['total_cloud_cover_sfc']:.2f}%")
print(f"   • Radiation solaire     : {data['shortwave_radiation_backwards_sfc']:.2f}")
print(f"   • Température           : {data['temperature_2_m_above_gnd']:.2f}°C")
print(f"   • Low zenith (auto)     : {data['low_zenith']}")
print(f"   • + 14 autres features (valeurs par défaut)")

# ============================================
# ENVOI DE LA REQUÊTE À L'API
# ============================================
print("\n" + "="*70)
print("ENVOI DE LA REQUÊTE À L'API...")
print("="*70)

try:
    response = requests.post("http://localhost:5000/predict", json=data)
    
    if response.status_code == 200:
        result = response.json()
        prediction = result['prediction']['value']
        
        print("\n" + "="*70)
        print("RÉSULTAT DE LA PRÉDICTION")
        print("="*70)
        print(f"\nPRODUCTION SOLAIRE PRÉVUE : {prediction:.2f} kW\n")
        
        # Interprétation
        if prediction > 2000:
            print("Interprétation : EXCELLENTE production (conditions idéales)")
        elif prediction > 1000:
            print("Interprétation : BONNE production (conditions favorables)")
        elif prediction > 500:
            print("Interprétation : MOYENNE production (conditions correctes)")
        elif prediction > 100:
            print("Interprétation : FAIBLE production (conditions défavorables)")
        else:
            print("Interprétation : TRÈS FAIBLE production (conditions très mauvaises)")
        
        print("\n" + "="*70)
        
    else:
        error = response.json()
        print(f"\nERREUR {response.status_code}")
        print(f"   Message : {error}")
        
except requests.exceptions.ConnectionError:
    print("\nERREUR : Impossible de se connecter à l'API")
    print("   → Vérifiez que l'API est bien lancée (python app.py)")
    print("   → L'API doit tourner sur http://localhost:5000")
    
except Exception as e:
    print(f"\nERREUR INATTENDUE : {e}")

# ============================================
# PROPOSER UNE NOUVELLE PRÉDICTION
# ============================================
print("\n" + "="*70)
rejouer = input("Voulez-vous faire une nouvelle prédiction ? (o/n) : ").strip().lower()

if rejouer == 'o':
    print("\n" * 2)
    import os
    os.system('python interface_utilisateur.py')
else:
    print("\nAu revoir ! Merci d'avoir utilisé le prédicteur solaire.")
    print("="*70)