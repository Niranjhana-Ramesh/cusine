import joblib
import pandas as pd

# Load the saved model and vectorizer
loaded_model = joblib.load('models/ensemble_model.pkl')
loaded_cv = joblib.load('models/count_vectorizer.pkl')

# New data (example)
new_data = [
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
    ['black beans', 'lime', 'cilantro', 'cassava'],                 # Brazilian
]

# Preprocess new data
new_data_processed = [' '.join(ingredients) for ingredients in new_data]
new_data_bow = loaded_cv.transform(new_data_processed).toarray()

# Make predictions
new_predictions = loaded_model.predict(new_data_bow)

# Map numeric predictions to cuisine names
cuisine_mapping = {
    'Italian': 1, 'Mexican': 2, 'Southern US': 3, 'Indian': 4, 'Chinese': 5,
    'French': 6, 'Cajun Creole': 7, 'Thai': 8, 'Japanese': 9, 'Greek': 10,
    'Spanish': 11, 'Korean': 12, 'Vietnamese': 13, 'Moroccan': 14, 'British': 15,
    'Filipino': 16, 'Irish': 17, 'Jamaican': 18, 'Russian': 19, 'Brazilian': 20
}
reverse_cuisine_mapping = {v: k for k, v in cuisine_mapping.items()}
new_predictions_labels = [reverse_cuisine_mapping[pred] for pred in new_predictions]

# Print results
for ingredients, cuisine in zip(new_data, new_predictions_labels):
    print(f"Ingredients: {ingredients} -> Predicted Cuisine: {cuisine}")