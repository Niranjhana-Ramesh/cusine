import numpy as np
import pandas as pd
import re
from scipy.sparse import hstack
from gensim.models import Word2Vec
import torch
import torch.nn as nn
import joblib
import os

# Define paths for saved models
MODEL_PATH = r"models\stacking_model_pytorch_gpu.pkl"
TFIDF_PATH = r"models\tfidf_vectorizer_pytorch_gpu.pkl"
W2V_PATH = r"models\word2vec_model_pytorch_gpu.pkl"
META_PATH = r"models\meta_learner_pytorch_gpu.pkl"
LSTM_PATH = r"models\lstm_model_pytorch_gpu.pkl"

# Verify GPU
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cuisine mapping
cuisine_mapping = {
    'italian': 1, 'mexican': 2, 'southern_us': 3, 'indian': 4, 'chinese': 5,
    'french': 6, 'cajun_creole': 7, 'thai': 8, 'japanese': 9, 'greek': 10,
    'spanish': 11, 'korean': 12, 'vietnamese': 13, 'moroccan': 14, 'british': 15,
    'filipino': 16, 'irish': 17, 'jamaican': 18, 'russian': 19, 'brazilian': 20
}
reverse_cuisine_mapping = {v: k for k, v in cuisine_mapping.items()}

# PyTorch LSTM model definition
class CuisineLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(CuisineLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

# PyTorch LSTM classifier wrapper
class PyTorchLSTMClassifier:
    def __init__(self, input_size=100, hidden_size=64, num_layers=1, num_classes=20):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, model_path):
        self.model = CuisineLSTM(self.input_size, self.hidden_size, self.num_layers, self.num_classes).to(self.device)
        checkpoint = joblib.load(model_path)
        self.model.load_state_dict(checkpoint.model.state_dict())
        self.model.eval()

    def predict_proba(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
        return probs.cpu().numpy()

# Standardize ingredients
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

# Word2Vec conversion
def ingredients_to_w2v(ingredients, w2v_model, max_len=20):
    embeddings = []
    for ingr in ingredients.split():
        if ingr in w2v_model.wv:
            embeddings.append(w2v_model.wv[ingr])
    if not embeddings:
        embeddings = [np.zeros(100)]
    embeddings = embeddings[:max_len]
    while len(embeddings) < max_len:
        embeddings.append(np.zeros(100))
    return np.array(embeddings)

# Inference function
def predict_cuisine(ingredient_lists):
    # Check file existence
    for path in [MODEL_PATH, TFIDF_PATH, W2V_PATH, META_PATH, LSTM_PATH]:
        if not os.path.exists(path):
            print(f"Error: File not found at {path}")
            return None

    # Load models
    try:
        stacking_model = joblib.load(MODEL_PATH)
        tfidf = joblib.load(TFIDF_PATH)
        w2v_model = joblib.load(W2V_PATH)
        meta_learner = joblib.load(META_PATH)
        lstm_classifier = PyTorchLSTMClassifier()
        lstm_classifier.load_model(LSTM_PATH)
    except Exception as e:
        print(f"Error loading models: {e}")
        return None

    # Preprocess ingredients
    processed_data = []
    ingredient_counts = []
    unique_ingredients = []
    
    for ingredients in ingredient_lists:
        standardized = standardize_ingredients(ingredients)
        if not standardized:
            processed_data.append('missing')
            ingredient_counts.append(0)
            unique_ingredients.append(0)
        else:
            processed_data.append(' '.join(standardized))
            ingredient_counts.append(len(standardized))
            unique_ingredients.append(len(set(standardized)))

    df = pd.DataFrame({
        'ingredients': processed_data,
        'ingredient_count': ingredient_counts,
        'unique_ingredients': unique_ingredients
    })

    valid_df = df[df['ingredients'] != 'missing']
    if valid_df.empty:
        print("Error: No valid ingredient lists provided.")
        return None

    # TF-IDF for ML models
    X_tfidf = tfidf.transform(valid_df['ingredients'])
    X_counts = valid_df[['ingredient_count', 'unique_ingredients']].values
    X_ml = hstack([X_tfidf, X_counts])

    # Word2Vec for LSTM
    X_w2v = np.array([ingredients_to_w2v(ingr, w2v_model) for ingr in valid_df['ingredients']])

    # ML and LSTM predictions
    ml_preds = stacking_model.predict_proba(X_ml)
    lstm_preds = lstm_classifier.predict_proba(X_w2v)

    # Combine predictions
    X_stack = np.hstack([ml_preds, lstm_preds])

    # Final prediction
    predictions = meta_learner.predict(X_stack)
    predicted_cuisines = [reverse_cuisine_mapping.get(pred, 'Unknown') for pred in predictions]

    # Handle invalid entries
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
    sample_inputs = [
    ["romaine lettuce", "black olives", "grape tomatoes", "garlic", "pepper", "purple onion", "seasoning", "garbanzo beans", "feta cheese crumbles"],
    ["plain flour", "ground pepper", "salt", "tomatoes", "ground black pepper", "thyme", "eggs", "green tomatoes", "yellow corn meal", "milk", "vegetable oil"],
    ["water", "vegetable oil", "wheat", "salt"],
    ["olive oil", "purple onion", "fresh pineapple", "pork", "poblano peppers", "corn tortillas", "cheddar cheese", "ground black pepper", "salt", "iceberg lettuce", "lime", "jalapeno chilies", "chopped cilantro fresh"],
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
    []
    ]

    predictions = predict_cuisine(sample_inputs)
    if predictions:
        print("\nPredictions:")
        for i, (ingredients, cuisine) in enumerate(zip(sample_inputs, predictions)):
            print(f"Recipe {i+1}: {ingredients}")
            print(f"Predicted Cuisine: {cuisine}\n")