import pandas as pd
import numpy as np
import re
import json
import logging
import os
from typing import Tuple, Dict, Any
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HotelDataProcessor:
    """
    Pipeline complet de traitement des données hôtelières
    Version professionnelle avec auto-création de la structure
    """
    
    def __init__(self, data_path: str = './hotels_saudi.csv'):
        self.data_path = data_path
        self.df = None
        self.similarity_matrix = None
        self._setup_environment()
        self._setup_nltk()
    
    def _setup_environment(self):
        """Crée la structure de dossiers si elle n'existe pas"""
        directories = [
            './data/raw',
            './data/processed', 
            './data/embeddings',
            './src',
            './logs'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"📁 Dossier créé/vérifié: {directory}")
    
    def _setup_nltk(self):
        """Configure les ressources NLTK"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
    
    def load_and_inspect_data(self) -> pd.DataFrame:
        """Charge et inspecte les données avec gestion des chemins"""
        logger.info("🔍 CHARGEMENT ET INSPECTION DES DONNÉES")
        
        try:
            # Essayer plusieurs chemins possibles
            possible_paths = [
                self.data_path,
                './hotels_saudi.csv',
                './data/raw/hotels_saudi.csv',
                'hotels_saudi.csv'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    self.df = pd.read_csv(path)
                    logger.info(f"✅ Fichier trouvé: {path}")
                    break
            else:
                # Si aucun fichier n'est trouvé, lister les fichiers disponibles
                available_files = [f for f in os.listdir('.') if os.path.isfile(f)]
                logger.error(f"❌ Aucun fichier CSV trouvé. Fichiers disponibles: {available_files}")
                raise FileNotFoundError(f"Le fichier {self.data_path} n'existe pas")
            
            # Nettoyage des noms de colonnes
            self.df.columns = self.df.columns.str.strip()
            
            logger.info(f"Dimensions du dataset: {self.df.shape}")
            logger.info(f"Colonnes disponibles: {list(self.df.columns)}")
            logger.info(f"Aperçu des données:\n{self.df.head(2)}")
            
            return self.df
            
        except Exception as e:
            logger.error(f"Erreur au chargement des données: {e}")
            raise
    
    def comprehensive_price_clean(self, price) -> Tuple[float, str]:
        """
        Nettoyage robuste des prix avec détection d'anomalies
        """
        if pd.isna(price) or price is None:
            return 0.0, 'MANQUANT'
        
        price_str = str(price).strip()
        
        # Détection des formats spéciaux
        if price_str in ['0', '0.0', '0.00', 'NaN', 'nan', 'None', '']:
            return 0.0, 'ZERO_DETECTED'
        
        # Recherche de valeurs numériques avec patterns améliorés
        patterns = [
            r'SAR\s*(\d+[.,]?\d*)',  # SAR 450
            r'USD\s*(\d+[.,]?\d*)',   # USD 120
            r'(\d+[.,]?\d*)\s*SAR',   # 450 SAR
            r'(\d+[.,]?\d*)',         # 450.00
        ]
        
        for pattern in patterns:
            match = re.search(pattern, price_str, re.IGNORECASE)
            if match:
                try:
                    price_value = float(match.group(1).replace(',', ''))
                    
                    # Détection d'anomalies avec seuils ajustés
                    if price_value <= 50:  # Prix trop bas pour un hôtel
                        return price_value, 'ANOMALY_LOW_PRICE'
                    elif price_value > 10000:  # Prix trop élevé
                        return price_value, 'ANOMALY_HIGH_PRICE'
                    else:
                        return price_value, 'VALID'
                        
                except ValueError:
                    continue
        
        return 0.0, 'NO_NUMBER'
    
    def clean_prices(self) -> pd.DataFrame:
        """Nettoie et analyse les prix"""
        logger.info("💰 VÉRIFICATION ET NETTOYAGE DES PRIX")
        
        price_analysis = self.df['Price'].apply(self.comprehensive_price_clean)
        self.df['Price_Cleaned'] = [price for price, status in price_analysis]
        self.df['Price_Status'] = [status for price, status in price_analysis]
        
        # Statistiques détaillées
        price_status_counts = self.df['Price_Status'].value_counts()
        logger.info("📊 RAPPORT DE VÉRIFICATION DES PRIX:")
        
        for status, count in price_status_counts.items():
            percentage = (count / len(self.df)) * 100
            logger.info(f"  {status}: {count} hôtels ({percentage:.1f}%)")
        
        # Gestion des anomalies
        anomalies = self.df[self.df['Price_Status'].isin(['ANOMALY_LOW_PRICE', 'ANOMALY_HIGH_PRICE', 'MANQUANT'])]
        if not anomalies.empty:
            logger.warning(f"⚠️  ANOMALIES DÉTECTÉES ({len(anomalies)} hôtels)")
            # Imputation simple pour les anomalies
            median_price = self.df[self.df['Price_Status'] == 'VALID']['Price_Cleaned'].median()
            self.df.loc[anomalies.index, 'Price_Cleaned'] = median_price
            self.df.loc[anomalies.index, 'Price_Status'] = 'IMPUTED'
            logger.info(f"💰 Prix anomalies imputés avec la médiane: {median_price:.2f}")
        
        return self.df
    
    def clean_numeric_features(self) -> pd.DataFrame:
        """Nettoie les features numériques"""
        logger.info("🔧 NETTOYAGE DES FEATURES NUMÉRIQUES")
        
        # Gestion des ratings avec coercition
        self.df['Customers_Rating'] = pd.to_numeric(self.df['Customers_Rating'], errors='coerce')
        self.df['Star_Rating'] = pd.to_numeric(self.df['Star_Rating'], errors='coerce')
        
        # Imputation des valeurs manquantes
        self.df['Customers_Rating'] = self.df['Customers_Rating'].fillna(0)
        self.df['Star_Rating'] = self.df['Star_Rating'].fillna(0)
        
        # Validation des plages
        self.df['Customers_Rating'] = self.df['Customers_Rating'].clip(0, 10)
        self.df['Star_Rating'] = self.df['Star_Rating'].clip(0, 5)
        
        logger.info(f"⭐ Rating clients - Min: {self.df['Customers_Rating'].min()}, Max: {self.df['Customers_Rating'].max()}")
        logger.info(f"🏨 Étoiles - Min: {self.df['Star_Rating'].min()}, Max: {self.df['Star_Rating'].max()}")
        logger.info(f"💰 Prix nettoyés - Min: {self.df['Price_Cleaned'].min()}, Max: {self.df['Price_Cleaned'].max()}")
        
        return self.df
    
    def advanced_text_processing(self, text: str) -> str:
        """Prétraitement textuel avancé avec gestion d'erreurs"""
        if not isinstance(text, str) or not text.strip():
            return ''
        
        try:
            stop_words = set(stopwords.words('english'))
            stemmer = PorterStemmer()
            
            # Tokenization
            words = word_tokenize(text.lower())
            
            # Filtrage et stemming
            processed_words = []
            for word in words:
                if (word not in stop_words and 
                    word not in string.punctuation and 
                    len(word) > 2 and
                    word.isalnum()):
                    processed_words.append(stemmer.stem(word))
            
            return ' '.join(processed_words)
        
        except Exception as e:
            logger.warning(f"Erreur lors du traitement texte: {e}")
            return text.lower()  # Fallback simple
    
    def prepare_text_features(self) -> pd.DataFrame:
        """Prépare les features textuelles pour NLP"""
        logger.info("📝 PRÉPARATION DES FEATURES TEXTUELLES")
        
        TEXT_COLUMNS = ['Name', 'City', 'Type_of_room', 'Review_title', 'Customers_Review']
        
        # Nettoyage des textes
        for col in TEXT_COLUMNS:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna('').astype(str)
            else:
                logger.warning(f"Colonne manquante: {col}")
                self.df[col] = ''
        
        # Combinaison des textes
        self.df['combined_text_for_nlp'] = self.df[TEXT_COLUMNS].apply(
            lambda x: ' '.join(x), axis=1
        )
        
        # Prétraitement avancé
        self.df['processed_text_for_tfidf'] = self.df['combined_text_for_nlp'].apply(
            self.advanced_text_processing
        )
        
        logger.info(f"✅ Texte combiné créé (exemple: {self.df['combined_text_for_nlp'].iloc[0][:100]}...)")
        
        return self.df
    
    def build_hybrid_similarity_matrix(self) -> np.ndarray:
        """Construit la matrice de similarité hybride"""
        logger.info("🎯 CRÉATION DE LA MATRICE DE SIMILARITÉ HYBRIDE")
        
        # A. Normalisation des features numériques
        NUMERIC_FEATURES = ['Star_Rating', 'Customers_Rating', 'Price_Cleaned']
        
        # Vérification des colonnes
        missing_columns = [col for col in NUMERIC_FEATURES if col not in self.df.columns]
        if missing_columns:
            logger.error(f"Colonnes manquantes: {missing_columns}")
            raise ValueError(f"Colonnes requises manquantes: {missing_columns}")
        
        # Sécurisation contre les NaN
        self.df[NUMERIC_FEATURES] = self.df[NUMERIC_FEATURES].fillna(0)
        
        # Normalisation
        scaler = MinMaxScaler()
        numeric_normalized = scaler.fit_transform(self.df[NUMERIC_FEATURES])
        
        # Ajout des colonnes normalisées
        normalized_columns = ['Star_Normalized', 'Customers_Normalized', 'Price_Normalized']
        for i, col in enumerate(normalized_columns):
            self.df[col] = numeric_normalized[:, i]
        
        logger.info("📈 Features numériques normalisées:")
        for col in normalized_columns:
            logger.info(f"  {col}: {self.df[col].min():.2f} - {self.df[col].max():.2f}")
        
        # B. Vectorisation TF-IDF avec paramètres optimisés
        tfidf = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.8,
            stop_words='english',
            ngram_range=(1, 2)  # Inclure les bigrammes
        )
        
        tfidf_matrix = tfidf.fit_transform(self.df['processed_text_for_tfidf'].fillna(''))
        logger.info(f"✅ Matrice TF-IDF créée: {tfidf_matrix.shape}")
        
        # C. Combinaison hybride améliorée
        NUMERIC_WEIGHT = 2.0  # Poids pour les features numériques
        
        # Conversion en matrice sparse pour la combinaison
        weighted_numeric_matrix = csr_matrix(numeric_normalized * NUMERIC_WEIGHT)
        
        # Combinaison
        combined_matrix = hstack([tfidf_matrix, weighted_numeric_matrix])
        logger.info(f"✅ Matrice combinée: {combined_matrix.shape}")
        
        # D. Calcul de similarité
        self.similarity_matrix = cosine_similarity(combined_matrix)
        logger.info(f"🎯 MATRICE DE SIMILARITÉ FINALE: {self.similarity_matrix.shape}")
        
        return self.similarity_matrix
    
    def save_artifacts(self, 
                      similarity_path: str = './data/embeddings/similarity_matrix.json',
                      metadata_path: str = './data/processed/hotels_with_additional_info.json'):
        """Sauvegarde les artefacts du modèle"""
        logger.info("💾 SAUVEGARDE DES ARTEFACTS")
        
        try:
            # Créer les dossiers si nécessaire
            os.makedirs(os.path.dirname(similarity_path), exist_ok=True)
            os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
            
            # Sauvegarde de la matrice de similarité
            with open(similarity_path, 'w', encoding='utf-8') as f:
                json.dump(self.similarity_matrix.tolist(), f, ensure_ascii=False)
            
            # Sauvegarde des métadonnées enrichies
            metadata_columns = [
                'Name', 'City', 'Star_Rating', 'Customers_Rating', 
                'Price_Cleaned', 'Price_Status', 'Type_of_room', 'Review_title'
            ]
            
            # Sélectionner seulement les colonnes existantes
            available_columns = [col for col in metadata_columns if col in self.df.columns]
            df_deploy = self.df[available_columns].copy()
            
            df_deploy.to_json(metadata_path, orient='records', indent=2, force_ascii=False)
            
            logger.info("✅ Fichiers sauvegardés:")
            logger.info(f"   - {similarity_path} (matrice de similarité)")
            logger.info(f"   - {metadata_path} (métadonnées)")
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde: {e}")
            raise
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Génère un rapport final du système"""
        logger.info("📊 RAPPORT FINAL DU SYSTÈME")
        
        report = {
            'total_hotels': len(self.df),
            'average_price': float(self.df['Price_Cleaned'].mean()),
            'average_customer_rating': float(self.df['Customers_Rating'].mean()),
            'average_star_rating': float(self.df['Star_Rating'].mean()),
            'similarity_matrix_shape': self.similarity_matrix.shape,
            'price_status_distribution': self.df['Price_Status'].value_counts().to_dict(),
            'cities_count': self.df['City'].nunique() if 'City' in self.df.columns else 0
        }
        
        logger.info(f"🏨 Nombre total d'hôtels: {report['total_hotels']}")
        logger.info(f"💰 Prix moyen: SAR {report['average_price']:.2f}")
        logger.info(f"⭐ Rating moyen clients: {report['average_customer_rating']:.1f}/10")
        logger.info(f"🏨 Étoiles moyennes: {report['average_star_rating']:.1f}/5")
        logger.info(f"🎯 Matrice de similarité: {report['similarity_matrix_shape']}")
        logger.info(f"🏙️  Nombre de villes: {report['cities_count']}")
        
        logger.info("🔍 Distribution des prix:")
        for status, count in report['price_status_distribution'].items():
            percentage = (count / report['total_hotels']) * 100
            logger.info(f"   {status}: {count} ({percentage:.1f}%)")
        
        return report
    
    def execute_full_pipeline(self) -> Dict[str, Any]:
        """
        Exécute le pipeline complet de traitement
        Returns: Rapport final du système
        """
        logger.info("🚀 DÉMARRAGE DU PIPELINE COMPLET")
        
        try:
            # Étapes du pipeline
            self.load_and_inspect_data()
            self.clean_prices()
            self.clean_numeric_features()
            self.prepare_text_features()
            self.build_hybrid_similarity_matrix()
            self.save_artifacts()
            report = self.generate_final_report()
            
            logger.info("🎉 SYSTÈME DE RECOMMANDATION PRÊT À L'UTILISATION!")
            return report
            
        except Exception as e:
            logger.error(f"❌ ERREUR CRITIQUE DANS LE PIPELINE: {e}")
            raise

# Fonctions utilitaires pour les tests et recommandations
def test_recommendation_system(df: pd.DataFrame, similarity_matrix: np.ndarray, hotel_name: str, top_n: int = 5):
    """Teste le système de recommandation"""
    logger.info(f"🧪 TEST DU SYSTÈME DE RECOMMANDATION pour: {hotel_name}")
    
    if hotel_name not in df['Name'].values:
        # Recherche partielle
        matching_hotels = df[df['Name'].str.contains(hotel_name, case=False, na=False)]
        if len(matching_hotels) > 0:
            logger.info(f"🔍 Hôtels similaires trouvés: {matching_hotels['Name'].tolist()[:3]}")
            hotel_name = matching_hotels['Name'].iloc[0]
            logger.info(f"🔄 Utilisation de: {hotel_name}")
        else:
            logger.error(f"❌ Hôtel '{hotel_name}' non trouvé")
            return None
    
    # Trouver l'index de l'hôtel
    hotel_idx = df[df['Name'] == hotel_name].index[0]
    
    # Obtenir les similarités
    similarity_scores = list(enumerate(similarity_matrix[hotel_idx]))
    
    # Trier par similarité (exclure l'hôtel lui-même)
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    
    # Récupérer les recommendations
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
            'Price_Status': hotel_data.get('Price_Status', 'UNKNOWN')
        })
    
    # Affichage des résultats
    logger.info("🎯 RECOMMANDATIONS:")
    for i, rec in enumerate(recommendations, 1):
        logger.info(f"{i}. {rec['Name']}")
        logger.info(f"   📍 {rec['City']} | ⭐ {rec['Star_Rating']} | 👥 {rec['Customers_Rating']}/10 | 💰 SAR {rec['Price']}")
        logger.info(f"   🔗 Similarité: {rec['Similarity_Score']}")
    
    return recommendations

def quick_start():
    """Démarrage rapide du système"""
    print("🚀 DÉMARRAGE RAPIDE DU SYSTÈME DE RECOMMANDATION")
    
    # Essayer plusieurs chemins
    possible_files = ['./hotels_saudi.csv', 'hotels_saudi.csv']
    data_file = None
    
    for file in possible_files:
        if os.path.exists(file):
            data_file = file
            break
    
    if not data_file:
        print("❌ Aucun fichier hotels_saudi.csv trouvé dans le dossier courant")
        print("📁 Fichiers disponibles:", [f for f in os.listdir('.') if os.path.isfile(f)])
        return None
    
    print(f"✅ Fichier trouvé: {data_file}")
    
    # Création et exécution du pipeline
    processor = HotelDataProcessor(data_file)
    report = processor.execute_full_pipeline()
    
    # Test automatique
    if processor.df is not None and len(processor.df) > 0:
        test_hotel = processor.df['Name'].iloc[0]
        print(f"\n🔍 Test de recommandation pour: {test_hotel}")
        test_recommendation_system(processor.df, processor.similarity_matrix, test_hotel, 3)
    
    return processor

# Point d'entrée principal
if __name__ == "__main__":
    processor = quick_start()