import streamlit as st
import pandas as pd
import sys
import os
import json
import numpy as np
from pathlib import Path
import base64 

# --- Configuration de la page et CSS personnalisé pour un look pro ---

# La mise en page "wide" est conservée pour les grands écrans, et Streamlit gère le responsive sur mobile
st.set_page_config(
    page_title=" Saudi Hotel Smart Search",
    
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ajouter le chemin src
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root / 'src'))


# 🎯 Fonction pour convertir l'image locale en URI Base64 (pour le carrousel)
def img_to_base64_uri(path):
    """Charge une image locale et la convertit en Data URI Base64 pour l'intégration HTML."""
    try:
        relative_path_from_root = path.replace('../', '')
        full_path = project_root / relative_path_from_root
        
        with open(full_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
            mime_type = "image/png"  # Assumant que vos images sont des PNG
            return f"data:{mime_type};base64,{encoded_string}"
    except FileNotFoundError:
        return "" 

# Chemins de fichiers locaux pour les 5 images du carrousel
LOCAL_IMAGE_PATHS = [
    "../images/1.png",
    "../images/2.png",
    "../images/3.png",
    "../images/4.png", 
    "../images/5.png",
]

# Conversion des chemins en URI Base64
BASE64_CAROUSEL_IMAGE_PATHS = [img_to_base64_uri(p) for p in LOCAL_IMAGE_PATHS if img_to_base64_uri(p) != ""]

# HTML du carrousel adapté pour 5 images Base64
CAROUSEL_HTML = f"""
<div class="slider-container">
    <div class="slide-track">
        {''.join([f'<div class="slide"><img src="{path_uri}" alt="Image Hôtel {i+1}" /></div>' for i, path_uri in enumerate(BASE64_CAROUSEL_IMAGE_PATHS)])}
    </div>
</div>
"""

# CSS pour l'arrière-plan, les couleurs, le style général ET le carrousel
CUSTOM_CSS = """
<style>
/* 1. Arrière-plan (Image de Riyad en placeholder) */
.stApp {
    background: url("https://images.unsplash.com/photo-1596701831802-0c9f136d8d9c?q=80&w=2670&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D") no-repeat center center fixed;
    background-size: cover;
    color: #333333; 
}
/* 2. Conteneur principal (lisibilité sur l'arrière-plan) */
section.main { 
    background-color: rgba(255, 255, 255, 0.95);
    padding: 30px;
    border-radius: 12px;
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    margin-top: 20px;
}
/* 3. Style du Header */
h1 {
    color: #1A73E8; 
    text-align: center;
    font-weight: 800;
    padding-bottom: 15px;
    border-bottom: 4px solid #EA4335; 
}
h3 {
    color: #34A853; 
    font-weight: 700;
}
.stApp section.main h2 {
    color: #F9AB00; 
    font-weight: 700;
    border-bottom: 2px solid #F9AB00;
    padding-bottom: 5px;
    margin-top: 20px;
}
/* 4. Sidebar */
.css-1d391kg { 
    background-color: rgba(240, 242, 246, 0.98); 
    border-right: 5px solid #1A73E8;
    padding: 20px;
}
/* 5. Metrics */
[data-testid="stMetricValue"] {
    font-size: 2.5rem;
    color: #EA4335;
}
/* 6. Bouton */
.stButton>button {
    background-color: #1A73E8;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    border: none;
    padding: 10px 10px;
    transition: all 0.3s;
    font-size: 1.1em;
}
.stButton>button:hover {
    background-color: #34A853;
    transform: scale(1.05); 
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}
/* 7. Styles du Carrousel (Slider) */
.slider-container {
    width: 100%;
    height: 300px; 
    overflow: hidden;
    position: relative;
    margin: 20px 0;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
}
.slide-track {
    display: flex;
    width: 500%; 
    animation: slide 20s linear infinite; 
}
.slide {
    width: 20%; 
    height: 300px;
}
.slide img {
    width: 100%;
    height: 100%;
    object-fit: cover; 
}
@keyframes slide {
    0% { transform: translateX(0); }
    15% { transform: translateX(0); } 
    20% { transform: translateX(-20%); } 
    35% { transform: translateX(-20%); }
    40% { transform: translateX(-40%); } 
    55% { transform: translateX(-40%); }
    60% { transform: translateX(-60%); } 
    75% { transform: translateX(-60%); }
    80% { transform: translateX(-80%); } 
    95% { transform: translateX(-80%); }
    100% { transform: translateX(0); } 
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- Fonctions de simulation de données (Inchangées) ---

try:
    from nlp_processor import NaturalLanguageProcessor
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    
def load_data():
    """Charge les données et la matrice de similarité"""
    try:
        metadata_path = project_root / 'data' / 'processed' / 'hotels_with_additional_info.json'
        
        if not metadata_path.exists():
            data = {
                'Name': ['Shaza Riyadh', 'Radisson Blu Hotel, Riyadh', 'Four Points by Sheraton Makkah Al Naseem', 
                         'Makarem Umm Al Qura Hotel', 'Copper Crown Furnished Apartments', 'Al Gosaibi Hotel'],
                'City': ['Riyadh', 'Riyadh', 'Makkah', 'Ajyad, Makkah', 'Khamis Mushayt', 'Al Yarmouk, Al Khobar'],
                'Star_Rating': [5, 5, 5, 5, 5, 5],
                'Customers_Rating': [8.8, 7.6, 8.7, 7.7, 9.0, 8.1],
                'Price_Cleaned': [400, 383, 225, 350, 195, 340],
                'Type_of_room': ['Deluxe Room', 'Standard Room', 'Superior Twin Room', 'Deluxe Twin Room', 'Deluxe Room (2 Adults + 1 Child)', 'Superior Twin Room']
            }
            hotels_df = pd.DataFrame(data)
            similarity_matrix = np.identity(len(hotels_df)).tolist()
            return hotels_df, np.array(similarity_matrix)
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            hotels_df = pd.read_json(f)
        
        similarity_path = project_root / 'data' / 'embeddings' / 'similarity_matrix.json'
        
        if not similarity_path.exists():
            similarity_matrix = np.identity(len(hotels_df))
        else:
            with open(similarity_path, 'r', encoding='utf-8') as f:
                similarity_matrix = np.array(json.load(f))
        
        return hotels_df, similarity_matrix
        
    except Exception as e:
        return None, None

def search_hotels(df, filters):
    """Logique de recherche (inchangée)."""
    initial_results = df.copy()
    applied_filters = []
    results_exact = initial_results.copy()
    
    if filters.get('city') is not None:
        city = filters['city']
        results_exact = results_exact[results_exact['City'].str.contains(city, case=False, na=False)]
        
    if filters.get('stars') is not None:
        stars = filters['stars']
        results_exact = results_exact[results_exact['Star_Rating'] == stars]
        
    if filters.get('price_range', {}).get('max') is not None:
        max_price = filters['price_range']['max']
        results_exact = results_exact[results_exact['Price_Cleaned'] <= max_price]
        
    if filters.get('min_rating') is not None:
        min_rating = filters['min_rating']
        results_exact = results_exact[results_exact['Customers_Rating'] >= min_rating]

    if not any([filters.get('stars'), filters.get('city'), 
                filters.get('price_range', {}).get('max'), filters.get('min_rating')]):
        results_exact = initial_results[initial_results['Customers_Rating'] >= 7.0]
        applied_filters = ["👥 Meilleurs ratings (≥7.0) par défaut"] 
        
    if len(results_exact) > 0:
        results = results_exact
        applied_filters = []
        if filters.get('city') is not None: applied_filters.append(f"📍 {filters['city']}")
        if filters.get('stars') is not None: applied_filters.append(f"⭐ {filters['stars']} étoiles")
        if filters.get('price_range', {}).get('max') is not None: applied_filters.append(f"💰 ≤ {filters['price_range']['max']}SAR")
        if filters.get('min_rating') is not None: applied_filters.append(f"👥 ≥ {filters['min_rating']}/10")
        
    else:
        results = initial_results.copy()
        applied_filters = [] 
        
        if filters.get('city') is not None:
            city = filters['city']
            results = results[results['City'].str.contains(city, case=False, na=False)]
            applied_filters.append(f"📍 {city} (Zones liées incluses)")
            
        if filters.get('price_range', {}).get('max') is not None:
            max_price = filters['price_range']['max']
            new_max = int(max_price * 1.2)
            results = results[results['Price_Cleaned'] <= new_max]
            applied_filters.append(f"💰 Budget ajusté à {new_max}SAR")
            
        if filters.get('stars') is not None:
            stars = filters['stars']
            min_stars = max(0, stars - 1)
            max_stars = min(5, stars + 1)
            results = results[(results['Star_Rating'] >= min_stars) & (results['Star_Rating'] <= max_stars)]
            applied_filters.append(f"⭐ Élargi à {min_stars}-{max_stars} étoiles")

        if filters.get('min_rating') is not None:
            min_rating = filters['min_rating']
            results = results[results['Customers_Rating'] >= min_rating]

        if len(results) == 0:
             results = pd.DataFrame()
             applied_filters = []
        elif not applied_filters and not results.empty:
            applied_filters = ["Recherche élargie sans critères spécifiés."]
            
    if not results.empty and len(results) > 50:
        before = len(results)
        results = results[results['Customers_Rating'] >= 7.5]
        after = len(results)
        if after < before:
            applied_filters.append("👥 Filtrage automatique (≥7.5)")
    
    if not results.empty:
        results = results.sort_values('Customers_Rating', ascending=False)
    
    return results, applied_filters

def get_hotel_details(df, hotel_name):
    """Logique de récupération inchangée."""
    try:
        matching_hotels = df[df['Name'].str.contains(hotel_name, case=False, na=False)]
        if len(matching_hotels) == 0:
            return None
            
        hotel_data = matching_hotels.iloc[0]
        return {
            'Name': hotel_data['Name'],
            'City': hotel_data['City'],
            'Star_Rating': hotel_data['Star_Rating'],
            'Customers_Rating': hotel_data['Customers_Rating'],
            'Price': hotel_data['Price_Cleaned'],
            'Type_of_room': hotel_data.get('Type_of_room', 'N/A'),
            'Price_Status': hotel_data.get('Price_Status', 'N/A')
        }
    except Exception as e:
        return None

def get_recommendations(df, similarity_matrix, hotel_name, top_n=5):
    """Obtient les recommandations pour un hôtel spécifique (inchangée)."""
    try:
        if similarity_matrix.ndim == 1 or similarity_matrix.shape[0] != len(df):
            return df.sort_values('Customers_Rating', ascending=False).head(top_n).to_dict('records')

        if hotel_name not in df['Name'].values:
            matching_hotels = df[df['Name'].str.contains(hotel_name, case=False, na=False)]
            if len(matching_hotels) > 0:
                hotel_name = matching_hotels['Name'].iloc[0]
            else:
                return None
        
        hotel_idx = df[df['Name'] == hotel_name].index[0]
        similarity_scores = list(enumerate(similarity_matrix[hotel_idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        
        recommendations = []
        for idx, score in similarity_scores:
            hotel_data = df.iloc[idx]
            recommendations.append({
                'Name': hotel_data['Name'],
                'City': hotel_data['City'],
                'Star_Rating': hotel_data['Star_Rating'],
                'Customers_Rating': hotel_data['Customers_Rating'],
                'Price': hotel_data['Price_Cleaned'],
                'Similarity_Score': round(score, 3),
                'Type_of_room': hotel_data.get('Type_of_room', 'N/A')
            })
        
        return recommendations
    except Exception as e:
        return None

# --- Données d'images factices pour le look professionnel (inchangées) ---

CITY_IMAGES = {
    'Riyadh': "https://images.unsplash.com/photo-1590492804576-8800539f15a1?q=80&w=2670&auto=format&fit=crop",
    'Makkah': "https://images.unsplash.com/photo-1549488344-93b17a1d471b?q=80&w=2670&auto=format&fit=crop",
    'Ajyad, Makkah': "https://images.unsplash.com/photo-1549488344-93b17a1d471b?q=80&w=2670&auto=format&fit=crop",
    'Al Khobar': "https://images.unsplash.com/photo-1627885732101-76677f51152d?q=80&w=2670&auto=format&fit=crop",
    'Khamis Mushayt': "https://images.unsplash.com/photo-1579208078873-1f19c08d5c4b?q=80&w=2670&auto=format&fit=crop",
    'Default': "https://images.unsplash.com/photo-1566073771259-d278ef04e4c2?q=80&w=2670&auto=format&fit=crop"
}

def get_city_image_url(city):
    """Retourne une URL d'image factice basée sur la ville pour les cartes d'hôtels."""
    for key, url in CITY_IMAGES.items():
        if key.lower() in city.lower():
            return url
    return CITY_IMAGES['Default']


def display_hotel_card(i, row, mode="search"):
    """
    Fonction utilitaire pour afficher une carte d'hôtel au format professionnel.
    MODIFIÉ: Ratio de colonne ajusté pour le prix dans les modes 'search' et 'popular'.
    """
    
    with st.container(border=True): 
        
        # Déterminer la mise en page des colonnes
        if mode in ["search", "popular"]:
            # Mode Recherche/Populaire: PAS d'image. Mise en page: Détails (2) | Prix/Bouton (1)
            # NOUVEAU RATIO [2, 1] pour donner plus de place au prix (vs [3, 1] précédent)
            col_details, col_price = st.columns([2, 1])
        else: 
            # Mode Recommandation/Détails (l'image est conservée ici)
            col_img, col_details, col_price = st.columns([1, 2, 1])

        # Colonne Image (uniquement pour le mode recommandation)
        if mode not in ["search", "popular"]:
            with col_img:
                image_url = get_city_image_url(row['City'])
                st.image(image_url, caption=f"Hôtel à {row['City'].split(',')[0]}", use_container_width=True) 
        
        # Colonne Détails (commune à tous les modes)
        with col_details:
            st.markdown(f"**<h3 style='color:#34A853;'>{i}. {row['Name']}</h3>**", unsafe_allow_html=True)
            st.markdown(f"""
            **📍 {row['City']}** | 
            ⭐ **{row['Star_Rating']}** | 
            👥 **{row['Customers_Rating']}/10**
            """)
            st.markdown(f"🛏️ *{row.get('Type_of_room', 'N/A')}*")
            
            if mode == "recommendation" and 'Similarity_Score' in row:
                st.markdown(f"🔗 **Score de Similarité:** `{row['Similarity_Score']:.3f}`")
                st.progress(float(row['Similarity_Score']))

        # Colonne Prix/Bouton (commune à tous les modes - affiche clairement le prix)
        with col_price:
            price_key = 'Price_Cleaned' if 'Price_Cleaned' in row else 'Price'
            
            # Affichage clair du prix. Le ratio [2, 1] résout le problème de troncature pour > 999
            # Utilisation de f-string avec :.0f pour un nombre entier, et un remplacement pour éviter 
            # l'utilisation de la virgule comme séparateur des milliers si cela pose problème d'affichage (Ex: 1,000)
            st.metric("Prix par Nuit", f"SAR {row[price_key]:,.0f}".replace(',', ' '))
            
            st.button("Réserver Maintenant 🚀", key=f"book_{i}_{row['Name']}_{mode}", use_container_width=True)
    
    st.markdown("---")


def main():
    st.title("🏨 Saudi Hotel Smart Search")
    st.markdown("""
    <p style='text-align: center; color: #555;'>
    <b>Recherche intelligente d'hôtels</b> – Exploitez la puissance de l'IA pour trouver votre séjour idéal en Arabie Saoudite.
    </p>
    """, unsafe_allow_html=True)
    
    # Intégration du carrousel d'images (Base64)
    st.markdown(CAROUSEL_HTML, unsafe_allow_html=True)
    
    # Chargement des données
    with st.spinner("Chargement des données hôtelières... ⌛"):
        hotels_df, similarity_matrix = load_data()
    
    if hotels_df is None or len(hotels_df) == 0:
        st.error("❌ DONNÉES NON CHARGÉES - Fichiers manquants ou base vide. Affichage des solutions dans la barre latérale.")
        st.sidebar.error("❌ FICHIERS MANQUANTS")
        st.sidebar.info("""
        **Solutions:**
        1. Exécutez: `python main.py` 
        2. Vérifiez les chemins de fichiers.
        """)
        return
    
    # AFFICHER LES STATS DES DONNÉES
    st.sidebar.subheader("📊 Stats Données (DEBUG)")
    st.sidebar.write(f"**Hôtels totaux:** {len(hotels_df)}")
    st.sidebar.write(f"**Villes uniques:** {hotels_df['City'].nunique()}")
    
    # Gestion des états de session
    if 'search_results' not in st.session_state: st.session_state.search_results = None
    if 'search_mode' not in st.session_state: st.session_state.search_mode = "🎯 Recherche Naturelle (NLP)"
    if 'current_filters' not in st.session_state: st.session_state.current_filters = {}
    if 'selected_hotel_details' not in st.session_state: st.session_state.selected_hotel_details = None
    if 'applied_filters' not in st.session_state: st.session_state.applied_filters = []
    
    if (st.session_state.get('search_mode_radio') is not None and 
        st.session_state.get('search_mode_radio') != st.session_state.search_mode):
        st.session_state.search_results = None
        st.session_state.selected_hotel_details = None
        st.session_state.applied_filters = []
    
    # Sidebar avec les modes de recherche
    st.sidebar.header("🔍 Mode de Recherche")
    search_mode = st.sidebar.radio(
        "Choisissez votre mode de recherche:",
        ["🎯 Recherche Naturelle (NLP)", "⚙️ Filtres Avancés", "🏨 Par Nom d'Hôtel"],
        key="search_mode_radio"
    )
    st.session_state.search_mode = search_mode
    filters = {}

    if search_mode == "🎯 Recherche Naturelle (NLP)":
        st.sidebar.subheader("💬 Recherche Vocale/Textuelle")
        if not NLP_AVAILABLE:
            st.sidebar.error("⚠️ Module NLP non disponible - utilisation des filtres basiques")
        else:
            query = st.sidebar.text_area(
                "Décrivez votre recherche:",
                placeholder="Ex: Je veux un hôtel 4 étoiles à Makkah pour 1000 SAR max...",
                height=100,
                key="nlp_query"
            )
            if st.sidebar.button("🔍 Analyser et Rechercher", type="primary", key="nlp_search"):
                with st.spinner("Analyse de votre requête... 🤖"):
                    try:
                        nlp = NaturalLanguageProcessor()
                        filters = nlp.parse_query(query)
                    except NameError:
                        filters = {}
                    st.session_state.current_filters = filters
                    st.subheader("📋 Analyse de Votre Requête")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1: st.metric("🌐 Langue", filters.get('detected_language', 'Inconnue'))
                    with col2: st.metric("⭐ Étoiles", filters.get('stars', 'Non spécifié'))
                    with col3: st.metric("📍 Ville", filters.get('city', 'Non spécifiée'))
                    with col4: st.metric("💰 Prix Max", f"SAR {filters.get('price_range', {}).get('max', 'Illimité')}")
                    
                    results, applied_filters = search_hotels(hotels_df, filters)
                    st.session_state.search_results = results
                    st.session_state.applied_filters = applied_filters
    
    elif search_mode == "⚙️ Filtres Avancés":
        st.sidebar.subheader("🎛️ Filtres de Recherche")
        stars_options = ["Toutes", "1", "2", "3", "4", "5"]
        stars = st.sidebar.selectbox("Nombre d'étoiles:", stars_options, key="stars_filter")
        if stars != "Toutes": filters['stars'] = int(stars)
        cities = ["Toutes"] + sorted(hotels_df['City'].unique().tolist())
        city = st.sidebar.selectbox("Ville:", cities, key="city_filter")
        if city != "Toutes": filters['city'] = city
        st.sidebar.subheader("💰 Plage de Prix")
        min_price = st.sidebar.number_input("Prix min (SAR):", value=0, key="min_price")
        max_price = st.sidebar.number_input("Prix max (SAR):", value=5000, key="max_price")
        if max_price > 0: filters['price_range'] = {'min': min_price, 'max': max_price}
        min_rating = st.sidebar.slider("Rating client minimum:", 0.0, 10.0, 6.0, 0.5, key="rating_filter")
        filters['min_rating'] = min_rating
        
        if st.sidebar.button("🔍 Rechercher avec Filtres", type="primary", key="filter_search"):
            st.session_state.current_filters = filters
            results, applied_filters = search_hotels(hotels_df, filters)
            st.session_state.search_results = results
            st.session_state.applied_filters = applied_filters
            
    else:
        st.sidebar.subheader("🏨 Recherche par Hôtel")
        hotel_names = sorted(hotels_df['Name'].unique())
        selected_hotel = st.sidebar.selectbox("Sélectionnez un hôtel:", hotel_names, key="hotel_select")
        filters['hotel_name'] = selected_hotel
        top_n = st.sidebar.slider("Nombre de recommandations:", 1, 10, 5, key="top_n_slider")
        filters['top_n'] = top_n
        
        if st.sidebar.button("🔍 Trouver des Hôtels Similaires", type="primary", key="similarity_search"):
            st.session_state.current_filters = filters
            hotel_details = get_hotel_details(hotels_df, selected_hotel)
            st.session_state.selected_hotel_details = hotel_details
            recommendations = get_recommendations(hotels_df, similarity_matrix, selected_hotel, top_n)
            st.session_state.search_results = recommendations

    # ==================== AFFICHAGE DES RÉSULTATS ====================
    if st.session_state.search_results is not None:
        results = st.session_state.search_results
        filters = st.session_state.current_filters
        
        if search_mode == "🏨 Par Nom d'Hôtel":
            st.subheader("🏨 Hôtel Sélectionné & Recommandations")
            
            if st.session_state.selected_hotel_details:
                hotel_details = st.session_state.selected_hotel_details
                with st.container(border=True):
                    col_det, col_img = st.columns([3, 1])
                    with col_det:
                        st.markdown(f"**<h2 style='color:#1A73E8;'>{hotel_details['Name']}</h2>**", unsafe_allow_html=True)
                        st.write(f"**📍 {hotel_details['City']}** | ⭐ {hotel_details['Star_Rating']} | 👥 {hotel_details['Customers_Rating']}/10")
                        st.write(f"**🛏️** {hotel_details['Type_of_room']}")
                    with col_img:
                        st.metric("Prix", f"SAR {hotel_details['Price']}")
                st.divider()
            
            if results and len(results) > 0:
                st.subheader("🎯 Hôtels Similaires Recommandés")
                st.success(f"✅ {len(results)} recommandations trouvées, basées sur le contenu et les caractéristiques.")
                
                for i, rec in enumerate(results, 1):
                    rec_series = pd.Series(rec)
                    # Utilise l'affichage avec image (car mode="recommendation")
                    display_hotel_card(i, rec_series, mode="recommendation") 
            else:
                st.error("Aucune recommandation trouvée pour cet hôtel.")
        
        else:
            if isinstance(results, pd.DataFrame) and len(results) > 0:
                st.subheader("🔍 Résultats de Recherche")

                requested_filters = []
                if filters.get('stars'): requested_filters.append(f"⭐ {filters['stars']} étoiles")
                if filters.get('city'): requested_filters.append(f"📍 {filters['city']}")
                if filters.get('price_range', {}).get('max'): requested_filters.append(f"💰 ≤ SAR {filters['price_range']['max']}")
                if filters.get('min_rating'): requested_filters.append(f"👥 ≥ {filters['min_rating']}/10")

                if requested_filters:
                    st.info(f"**Votre recherche:** {', '.join(requested_filters)}")

                if hasattr(st.session_state, 'applied_filters') and st.session_state.applied_filters:
                    if any("élargi" in f.lower() or "ajusté" in f.lower() for f in st.session_state.applied_filters):
                         st.success(f"**Ajustements appliqués (Recherche élargie):** {', '.join(st.session_state.applied_filters)}")
                
                st.success(f"✅ **{len(results)} hôtels trouvés** correspondant à vos critères.")
                
                for i, (_, row) in enumerate(results.iterrows(), 1):
                    # Utilise l'affichage SANS image (car mode="search")
                    display_hotel_card(i, row, mode="search")
            
            else:
                st.warning("🎯 **Aucun hôtel ne correspond exactement à vos critères.** Nous élargissons la recherche.")
                
                st.info("🏨 **Voici des hôtels populaires bien notés qui pourraient vous intéresser:**")
                popular_hotels = hotels_df[hotels_df['Customers_Rating'] >= 7.0].sort_values('Customers_Rating', ascending=False).head(10)
                
                if len(popular_hotels) > 0:
                    for i, (_, hotel) in enumerate(popular_hotels.iterrows(), 1):
                        # Utilise l'affichage SANS image (car mode="popular")
                        display_hotel_card(i, hotel, mode="popular")
                else:
                    st.error("❌ Aucun hôtel trouvé dans la base de données.")
    
    # Statistiques globales
    st.sidebar.divider()
    st.sidebar.subheader("📊 Statistiques Générales")
    st.sidebar.metric("Total Hôtels", len(hotels_df))
    st.sidebar.metric("Villes couvertes", hotels_df['City'].nunique())
    st.sidebar.metric("Prix Moyen", f"SAR {hotels_df['Price_Cleaned'].mean():.0f}")

if __name__ == "__main__":
    main()