import re
import pandas as pd
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class NaturalLanguageProcessor:
    """
    Processeur de requêtes naturelles multilingues pour les hôtels
    Comprend le Français, Anglais, Arabe
    """
    
    def __init__(self):
        self.setup_patterns()
    
    def setup_patterns(self):
        """Configure les patterns regex pour chaque langue"""
        
        # Patterns pour les étoiles (multilingue)
        self.star_patterns = {
            'fr': [r'(\d+)\s*étoiles?', r'(\d+)\s*stars?', r'(\d+)\s*etoi'],
            'en': [r'(\d+)\s*stars?', r'(\d+)\s*star'],
            'ar': [r'(\d+)\s*نجوم?', r'(\d+)\s*نجم']  # nujoum/najm
        }
        
        # Patterns pour les prix (multilingue)
        self.price_patterns = {
            'fr': [r'(\d+)\s*sar', r'(\d+)\s*riyal', r'(\d+)\s*€', r'prix\s*max\s*(\d+)', r'max\s*(\d+)'],
            'en': [r'(\d+)\s*sar', r'(\d+)\s*riyal', r'(\d+)\s*usd', r'max\s*price\s*(\d+)', r'max\s*(\d+)'],
            'ar': [r'(\d+)\s*ريال', r'(\d+)\s*سار', r'(\d+)\s*دولار']  # riyal/sar/dollar
        }
        
        # Patterns pour les villes (multilingue) - VERSION AMÉLIORÉE
        self.city_mapping = {
            'fr': {
                'makkah': 'Makkah', 'mecca': 'Makkah', 'mecque': 'Makkah', 'la mecque': 'Makkah',
                'maka': 'Makkah', 'mcque': 'Makkah',
                'riad': 'Riyadh', 'riyad': 'Riyadh', 'riyadh': 'Riyadh',
                'jeddah': 'Jeddah', 'djeddah': 'Jeddah',
                'médine': 'Medina', 'medine': 'Medina', 'medina': 'Medina',
                'dammam': 'Dammam', 'al khobar': 'Khobar', 'khobar': 'Khobar'
            },
            'en': {
                'makkah': 'Makkah', 'mecca': 'Makkah', 
                'riyadh': 'Riyadh', 
                'jeddah': 'Jeddah',
                'medina': 'Medina',
                'dammam': 'Dammam', 'al khobar': 'Khobar', 'khobar': 'Khobar'
            },
            'ar': {
                'مكة': 'Makkah', 'مكه': 'Makkah', 'مكّة': 'Makkah',
                'الرياض': 'Riyadh', 'رياض': 'Riyadh',
                'جدة': 'Jeddah', 'جده': 'Jeddah',
                'المدينة': 'Medina', 'المدينه': 'Medina', 'مدينة': 'Medina',
                'الدمام': 'Dammam', 'دمام': 'Dammam',
                'الخبر': 'Khobar', 'خبر': 'Khobar'
            }
        }
        
        # Patterns pour les ratings
        self.rating_patterns = {
            'fr': [r'rating\s*(\d+)', r'note\s*(\d+)', r'avis\s*(\d+)', r'score\s*(\d+)'],
            'en': [r'rating\s*(\d+)', r'score\s*(\d+)', r'review\s*(\d+)'],
            'ar': [r'تقييم\s*(\d+)', r'نقاط\s*(\d+)']  # taqyeem/nuqat
        }
    
    def detect_language(self, query: str) -> str:
        """Détecte la langue de la requête"""
        query_lower = query.lower()
        
        # Détection par caractères arabes
        if any('\u0600' <= char <= '\u06FF' for char in query):
            return 'ar'
        
        # Détection par mots français
        french_indicators = ['étoiles', 'prix', 'hôtel', 'ville', 'max', 'min', 'mecque']
        if any(word in query_lower for word in french_indicators):
            return 'fr'
        
        # Par défaut anglais
        return 'en'
    
    def extract_stars(self, query: str, lang: str) -> int:
        """Extrait le nombre d'étoiles de la requête"""
        query_lower = query.lower()
        
        for pattern in self.star_patterns[lang]:
            match = re.search(pattern, query_lower)
            if match:
                stars = int(match.group(1))
                logger.info(f"⭐ Étoiles détectées: {stars} (lang: {lang})")
                return stars
        
        # Recherche de patterns simples
        if '4 star' in query_lower or '4 etoi' in query_lower or '4 نجوم' in query_lower:
            return 4
        if '5 star' in query_lower or '5 etoi' in query_lower or '5 نجوم' in query_lower:
            return 5
            
        return None
    
    def extract_price_range(self, query: str, lang: str) -> Dict[str, float]:
        """Extrait la plage de prix de la requête"""
        query_lower = query.lower()
        price_range = {'min': 0, 'max': float('inf')}
        
        # Extraction du prix maximum
        for pattern in self.price_patterns[lang]:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                if match.isdigit():
                    price_range['max'] = min(price_range['max'], float(match))
                    logger.info(f"💰 Prix max détecté: {match} SAR (lang: {lang})")
        
        # Patterns spécifiques pour "max"
        max_patterns = {
            'fr': r'max\s*(\d+)',
            'en': r'max\s*(\d+)', 
            'ar': r'حد\s*أقصى\s*(\d+)'  # had aqsa
        }
        
        match = re.search(max_patterns[lang], query_lower)
        if match:
            price_range['max'] = min(price_range['max'], float(match.group(1)))
        
        return price_range if price_range['max'] != float('inf') else {'min': 0, 'max': None}
    
    def extract_city(self, query: str, lang: str) -> str:
        """Extrait la ville de la requête - VERSION CORRIGÉE"""
        query_lower = query.lower().strip()
        
        print(f"🔍 Analyse de la ville dans: '{query_lower}' (lang: {lang})")
        
        # Nettoyage et normalisation des variations
        query_clean = query_lower
        variations = {
            'mecque': 'makkah',
            'mecca': 'makkah', 
            'maka': 'makkah',
            'la mecque': 'makkah',
            'mcque': 'makkah',
            'mekka': 'makkah'
        }
        
        for variation, normalized in variations.items():
            if variation in query_clean:
                query_clean = query_clean.replace(variation, normalized)
                print(f"🔄 Variation détectée: '{variation}' -> '{normalized}'")
        
        # Recherche dans le mapping
        for city_key, city_value in self.city_mapping[lang].items():
            if city_key in query_clean:
                print(f"✅ Ville trouvée dans mapping: {city_value}")
                return city_value
        
        # Recherche directe des noms de villes (backup)
        direct_cities = {
            'fr': ['makkah', 'mecca', 'mecque', 'riyad', 'riyadh', 'jeddah', 'djeddah', 'medine', 'medina', 'dammam'],
            'en': ['makkah', 'mecca', 'riyadh', 'jeddah', 'medina', 'dammam', 'khobar'],
            'ar': ['مكة', 'مكه', 'الرياض', 'جدة', 'المدينة', 'الدمام']
        }
        
        for city in direct_cities[lang]:
            if city in query_clean:
                print(f"✅ Ville trouvée en direct: {city}")
                # Mapping des noms vers le format standard
                city_standard_map = {
                    'makkah': 'Makkah', 'mecca': 'Makkah', 'mecque': 'Makkah',
                    'riyad': 'Riyadh', 'riyadh': 'Riyadh',
                    'jeddah': 'Jeddah', 'djeddah': 'Jeddah', 
                    'medine': 'Medina', 'medina': 'Medina',
                    'dammam': 'Dammam', 'khobar': 'Khobar',
                    'مكة': 'Makkah', 'مكه': 'Makkah',
                    'الرياض': 'Riyadh', 
                    'جدة': 'Jeddah',
                    'المدينة': 'Medina',
                    'الدمام': 'Dammam'
                }
                return city_standard_map.get(city, city.title())
        
        print(f"❌ Aucune ville détectée dans: '{query_lower}'")
        return None
    
    def extract_min_rating(self, query: str, lang: str) -> float:
        """Extrait le rating minimum"""
        query_lower = query.lower()
        
        for pattern in self.rating_patterns[lang]:
            match = re.search(pattern, query_lower)
            if match:
                rating = float(match.group(1))
                logger.info(f"👥 Rating min détecté: {rating} (lang: {lang})")
                return rating
        
        return None
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        Parse une requête naturelle et retourne les filtres
        """
        logger.info(f"🔍 Analyse de la requête: '{query}'")
        
        lang = self.detect_language(query)
        logger.info(f"🌐 Langue détectée: {lang}")
        
        filters = {
            'stars': self.extract_stars(query, lang),
            'price_range': self.extract_price_range(query, lang),
            'city': self.extract_city(query, lang),
            'min_rating': self.extract_min_rating(query, lang),
            'original_query': query,
            'detected_language': lang
        }
        
        # Log des résultats de l'analyse
        logger.info("📋 Résultats de l'analyse NLP:")
        logger.info(f"  - Étoiles: {filters['stars']}")
        logger.info(f"  - Prix: {filters['price_range']}")
        logger.info(f"  - Ville: {filters['city']}")
        logger.info(f"  - Rating min: {filters['min_rating']}")
        
        return filters

def test_city_detection():
    """Teste la détection de ville avec différentes requêtes"""
    nlp = NaturalLanguageProcessor()
    
    test_queries = [
        "Je veux un hôtel à makkah 4 étoiles max 1000 sar",
        "Hotel in mecca with 5 stars",
        "أريد فندق في مكة",
        "Hôtel à la mecque pas cher",
        "Recherche hotel mecca arabie saoudite"
    ]
    
    print("🧪 TEST DÉTECTION DE VILLE")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\n🔍 Requête: '{query}'")
        lang = nlp.detect_language(query)
        city = nlp.extract_city(query, lang)
        print(f"🌐 Langue: {lang}")
        print(f"📍 Ville détectée: {city}")
        print(f"✅ Succès: {city is not None}")

def test_nlp_processor():
    """Teste le processeur NLP avec différentes requêtes"""
    nlp = NaturalLanguageProcessor()
    
    test_queries = [
        "Je veux un hôtel 4 étoiles à Makkah pour 3000 SAR max",
        "I need a 5 star hotel in Riyadh with max price 2000",
        "أريد فندق ٥ نجوم في مكة بسعر ٢٥٠٠ ريال",
        "Hôtel pas cher à Jeddah avec bon rating",
        "Show me luxury hotels in Medina"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Requête: '{query}'")
        result = nlp.parse_query(query)
        print(f"📊 Résultat: {result}")

if __name__ == "__main__":
    # Testez d'abord la détection de ville
    test_city_detection()
    
    # Puis le test complet
    print("\n" + "=" * 50)
    print("🧪 TEST COMPLET NLP")
    print("=" * 50)
    test_nlp_processor()