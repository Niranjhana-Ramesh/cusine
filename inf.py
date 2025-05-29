import pandas as pd
import numpy as np
import joblib
import re
import os
from scipy.sparse import hstack

# Define explicit paths for model and vectorizer
MODEL_PATH = r"models\stacking_model_simplified.pkl"
VECTORIZER_PATH = r"models\tfidf_vectorizer_simplified.pkl"

# Cuisine mapping (same as training)
cuisine_mapping = {
    'italian': 1, 'mexican': 2, 'southern_us': 3, 'indian': 4, 'chinese': 5,
    'french': 6, 'cajun_creole': 7, 'thai': 8, 'japanese': 9, 'greek': 10,
    'spanish': 11, 'korean': 12, 'vietnamese': 13, 'moroccan': 14, 'british': 15,
    'filipino': 16, 'irish': 17, 'jamaican': 18, 'russian': 19, 'brazilian': 20
}
# Reverse mapping for prediction output
reverse_cuisine_mapping = {v: k for k, v in cuisine_mapping.items()}

# Function to standardize ingredients (same as training)
def standardize_ingredients(ingredients):
    if not isinstance(ingredients, list):
        return []
    standardized = [re.sub(r'[^a-zA-Z\s]', '', ingr.lower().strip()) for ingr in ingredients]
    standardized = [re.sub(r'\s+', ' ', ingr) for ingr in standardized]
    synonyms = {
        'ground black pepper': 'black pepper',
        'extra virgin olive oil': 'olive oil',
        'fresh cilantro': 'cilantro',
        'chopped cilantro fresh': 'cilantro',
        'yellow corn meal': 'cornmeal'
    }
    return [synonyms.get(ingr, ingr) for ingr in standardized if ingr]

# Inference function
def predict_cuisine(ingredient_lists):
    """
    Predict cuisine for a list of ingredient lists.
    
    Parameters:
    ingredient_lists (list of lists): List where each element is a list of ingredients for a recipe.
    
    Returns:
    list: Predicted cuisine names for each input recipe.
    """
    # Check if model and vectorizer files exist
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return None
    if not os.path.exists(VECTORIZER_PATH):
        print(f"Error: Vectorizer file not found at {VECTORIZER_PATH}")
        return None

    # Load model and vectorizer
    try:
        model = joblib.load(MODEL_PATH)
        tfidf = joblib.load(VECTORIZER_PATH)
    except Exception as e:
        print(f"Error loading model or vectorizer: {e}")
        return None

    # Preprocess ingredients
    processed_data = []
    ingredient_counts = []
    
    for ingredients in ingredient_lists:
        # Standardize ingredients
        standardized = standardize_ingredients(ingredients)
        if not standardized:
            print("Warning: Empty ingredient list after standardization.")
            processed_data.append('missing')
            ingredient_counts.append(0)
        else:
            # Convert to string for TF-IDF
            processed_data.append(' '.join(standardized))
            ingredient_counts.append(len(standardized))

    # Create DataFrame
    df = pd.DataFrame({
        'ingredients': processed_data,
        'ingredient_count': ingredient_counts
    })

    # Filter out invalid entries
    valid_df = df[df['ingredients'] != 'missing']
    if valid_df.empty:
        print("Error: No valid ingredient lists provided.")
        return None

    # Apply TF-IDF transformation
    X_tfidf = tfidf.transform(valid_df['ingredients'])
    X_counts = valid_df[['ingredient_count']].values
    X_combined = hstack([X_tfidf, X_counts])

    # Make predictions
    predictions = model.predict(X_combined)
    
    # Map predictions to cuisine names
    predicted_cuisines = [reverse_cuisine_mapping.get(pred, 'Unknown') for pred in predictions]
    
    # Handle invalid entries in output
    result = []
    valid_idx = 0
    for i, ingredients in enumerate(processed_data):
        if ingredients == 'missing':
            result.append('Invalid input (empty ingredients)')
        else:
            result.append(predicted_cuisines[valid_idx])
            valid_idx += 1

    return result

# Example usage
if __name__ == "__main__":
    # Sample input: list of ingredient lists (from provided dataset)
    sample_inputs = [
        ['olive oil', 'garlic', 'tomato', 'basil', 'parmesan cheese'],  # Italian
    ['tortilla', 'avocado', 'chili powder', 'cilantro', 'lime'],    # Mexican
    ['cornmeal', 'butter', 'collard greens', 'black pepper'],       # Southern US
    ['turmeric', 'cumin', 'coriander', 'ginger', 'ghee'],           # Indian
    ['soy sauce', 'ginger', 'green onions', 'sesame oil'],          # Chinese
    ['butter', 'cream', 'wine', 'shallo', 'parsley'],             # French
    ['cayenne pepper', 'bell pepper', 'celery', 'onion'],            # Cajun Creole
    ['coconut milk', 'lemongrass', 'fish sauce', 'chili'],          # Thai
    ['soy sauce', 'mirin', 'sushi rice', 'nori', 'sesame seeds'],   # Japanese
    ['feta cheese', 'olive oil', 'oregano', 'cucumber', 'tomato'],  # Greek
    ['chorizo', 'saffron', 'paprika', 'bell pepper'],               # Spanish
    ['gochujang', 'sesame oil', 'kimchi', 'green onions'],          # Korean
    ['fish sauce', 'rice noodles', 'cilantro', 'lime'],             # Vietnamese
    ['couscous', 'cumin', 'cinnamon', 'dried apricots'],            # Moroccan
    ['potatoes', 'beef', 'peas', 'carrots', 'gravy'],               # British
    ['coconut milk', 'pork', 'garlic', 'soy sauce'],                # Filipino
    ['potatoes', 'cabbage', 'butter', 'bacon'],                     # Irish
    ['jerk seasoning', 'allspice', 'scotch bonnet pepper'],         # Jamaican
    ['beets', 'dill', 'sour cream', 'potatoes'],                    # Russian
    ['black beans', 'lime', 'cilantro', 'cassava'],
    []  # Test case for empty input
    ]

    # Predict cuisines
    predictions = predict_cuisine(sample_inputs)
    
    # Print results
    if predictions:
        print("\nPredictions:")
        for i, (ingredients, cuisine) in enumerate(zip(sample_inputs, predictions)):
            print(f"Recipe {i+1}: {ingredients}")
            print(f"Predicted Cuisine: {cuisine}\n")