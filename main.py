def main():
    # Header principal
    st.title("🏨 Saudi Hotel Smart Search")
    st.markdown("""
    **Recherche intelligente d'hôtels - Comprend le Français, Anglais et Arabe**  
    *Parlez naturellement ou utilisez les filtres avancés*
    """)
    
    # Chargement des données
    with st.spinner("Chargement des données hôtelières..."):
        hotels_df, similarity_matrix = load_data()
    
    if hotels_df is None:
        st.error("Données non disponibles. Exécutez 'python main.py' d'abord.")
        return
    
    # Initialisation de la session state pour garder les résultats
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    if 'search_mode' not in st.session_state:
        st.session_state.search_mode = "🎯 Recherche Naturelle (NLP)"
    
    # Sidebar avec les deux modes de recherche
    st.sidebar.header("🔍 Mode de Recherche")
    
    search_mode = st.sidebar.radio(
        "Choisissez votre mode de recherche:",
        ["🎯 Recherche Naturelle (NLP)", "⚙️ Filtres Avancés", "🏨 Par Nom d'Hôtel"],
        key="search_mode_radio"
    )
    
    st.session_state.search_mode = search_mode
    filters = {}
    search_performed = False
    
    if search_mode == "🎯 Recherche Naturelle (NLP)":
        st.sidebar.subheader("💬 Recherche Vocale/Textuelle")
        
        if not NLP_AVAILABLE:
            st.sidebar.error("Module NLP non disponible")
            st.info("""
            **Pour activer la recherche naturelle:**
            1. Créez le fichier `src/nlp_processor.py`
            2. Redémarrez l'application
            """)
        else:
            query = st.sidebar.text_area(
                "Décrivez votre recherche:",
                placeholder="Ex: Je veux un hôtel 4 étoiles à Makkah pour 1000 SAR max...",
                height=100,
                key="nlp_query"
            )
            
            if st.sidebar.button("🔍 Analyser et Rechercher", type="primary", key="nlp_search"):
                with st.spinner("Analyse de votre requête..."):
                    nlp = NaturalLanguageProcessor()
                    filters = nlp.parse_query(query)
                    
                    # Afficher l'analyse
                    st.subheader("📋 Analyse de Votre Requête")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🌐 Langue", filters.get('detected_language', 'Inconnue'))
                    with col2:
                        stars = filters.get('stars', 'Non spécifié')
                        st.metric("⭐ Étoiles", stars)
                    with col3:
                        city = filters.get('city', 'Non spécifiée')
                        st.metric("📍 Ville", city)
                    with col4:
                        max_price = filters.get('price_range', {}).get('max', 'Illimité')
                        st.metric("💰 Prix Max", f"SAR {max_price}" if max_price != 'Illimité' else max_price)
                    
                    # Recherche avec les filtres
                    results = search_hotels(hotels_df, filters)
                    st.session_state.search_results = results
                    search_performed = True
    
    elif search_mode == "⚙️ Filtres Avancés":
        st.sidebar.subheader("🎛️ Filtres de Recherche")
        
        # Filtre par étoiles
        stars_options = ["Toutes", "1", "2", "3", "4", "5"]
        stars = st.sidebar.selectbox("Nombre d'étoiles:", stars_options, key="stars_filter")
        if stars != "Toutes":
            filters['stars'] = int(stars)
        
        # Filtre par ville
        cities = ["Toutes"] + sorted(hotels_df['City'].unique().tolist())
        city = st.sidebar.selectbox("Ville:", cities, key="city_filter")
        if city != "Toutes":
            filters['city'] = city
        
        # Filtre par prix
        st.sidebar.subheader("💰 Plage de Prix")
        min_price = st.sidebar.number_input("Prix min (SAR):", value=0, key="min_price")
        max_price = st.sidebar.number_input("Prix max (SAR):", value=1000, key="max_price")
        
        if max_price > 0:
            filters['price_range'] = {'min': min_price, 'max': max_price}
        
        # Filtre par rating
        min_rating = st.sidebar.slider("Rating client minimum:", 0.0, 10.0, 6.0, 0.5, key="rating_filter")
        filters['min_rating'] = min_rating
        
        if st.sidebar.button("🔍 Rechercher avec Filtres", type="primary", key="filter_search"):
            results = search_hotels(hotels_df, filters)
            st.session_state.search_results = results
            search_performed = True
            
    else:  # Recherche par nom d'hôtel
        st.sidebar.subheader("🏨 Recherche par Hôtel")
        hotel_names = sorted(hotels_df['Name'].unique())
        selected_hotel = st.sidebar.selectbox("Sélectionnez un hôtel:", hotel_names, key="hotel_select")
        filters['hotel_name'] = selected_hotel
        
        top_n = st.sidebar.slider("Nombre de recommandations:", 1, 10, 5, key="top_n_slider")
        filters['top_n'] = top_n
        
        if st.sidebar.button("🔍 Trouver des Hôtels Similaires", type="primary", key="similarity_search"):
            recommendations = get_recommendations(
                hotels_df, similarity_matrix, 
                filters['hotel_name'], filters['top_n']
            )
            st.session_state.search_results = recommendations
            search_performed = True
    
    # AFFICHAGE DES RÉSULTATS
    if st.session_state.search_results is not None:
        results = st.session_state.search_results
        
        if search_mode == "🏨 Par Nom d'Hôtel":
            # Mode recommandations par similarité
            if results and len(results) > 0:
                st.success(f"✅ {len(results)} recommandations trouvées pour '{filters['hotel_name']}'")
                
                for i, rec in enumerate(results, 1):
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.subheader(f"{i}. {rec['Name']}")
                        with col2:
                            st.metric("💰 Prix", f"SAR {rec['Price']}")
                        
                        col_a, col_b, col_c, col_d = st.columns(4)
                        with col_a:
                            st.write(f"**📍** {rec['City']}")
                        with col_b:
                            st.write(f"**⭐** {rec['Star_Rating']}")
                        with col_c:
                            st.write(f"**👥** {rec['Customers_Rating']}/10")
                        with col_d:
                            st.write(f"**🔗** {rec['Similarity_Score']:.3f}")
                        
                        st.progress(float(rec['Similarity_Score']))
                        st.divider()
            else:
                st.error("Aucune recommandation trouvée")
        
        else:
            # Mode recherche par filtres ou NLP
            if len(results) > 0:
                st.success(f"✅ {len(results)} hôtels trouvés")
                
                for i, (_, row) in enumerate(results.iterrows(), 1):
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.subheader(f"{i}. {row['Name']}")
                            st.write(f"**📍 {row['City']}** | ⭐ {row['Star_Rating']} | 👥 {row['Customers_Rating']}/10")
                            st.write(f"**🛏️** {row.get('Type_of_room', 'N/A')}")
                        with col2:
                            st.metric("💰 Prix", f"SAR {row['Price_Cleaned']}")
                        
                        st.divider()
            else:
                st.warning("❌ Aucun hôtel ne correspond aux critères de recherche")
                st.info("""
                **Suggestions :**
                - Élargissez la plage de prix
                - Baissez le nombre d'étoiles
                - Changez de ville
                - Baissez le rating minimum
                """)
    
    # Statistiques globales
    st.sidebar.divider()
    st.sidebar.subheader("📊 Statistiques")
    st.sidebar.metric("Total Hôtels", len(hotels_df))
    st.sidebar.metric("Villes", hotels_df['City'].nunique())
    st.sidebar.metric("Prix Moyen", f"SAR {hotels_df['Price_Cleaned'].mean():.0f}")

if __name__ == "__main__":
    main()