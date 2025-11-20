#!/usr/bin/env python3
"""
Script de test du système
"""

import sys
import os

# Ajouter le src au path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processor import HotelDataProcessor, test_recommendation_system

def test_complete_pipeline():
    """Test complet du pipeline"""
    print("🧪 TEST DU PIPELINE COMPLET")
    
    try:
        # Initialisation
        processor = HotelDataProcessor()
        
        # Pipeline
        processor.load_and_inspect_data()
        processor.clean_prices()
        processor.clean_numeric_features()
        processor.prepare_text_features()
        processor.build_hybrid_similarity_matrix()
        
        # Test de recommandation
        if len(processor.df) > 0:
            test_hotel = processor.df['Name'].iloc[0]
            print(f"\n🔍 Test avec l'hôtel: {test_hotel}")
            test_recommendation_system(processor.df, processor.similarity_matrix, test_hotel, 3)
            
        return processor
        
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        return None

if __name__ == "__main__":
    test_complete_pipeline()