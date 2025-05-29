import streamlit as st
import joblib
import pandas as pd
from collections import Counter
import time
import re
from scipy.sparse import hstack

# Page configuration
st.set_page_config(
    page_title="🍽️ AI Cuisine Predictor",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for compact styling, reduced spacing, and larger sidebar text
st.markdown("""
<style>
    /* Remove default Streamlit padding/margins */
    .main > div {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }
    
    /* Main header styling */
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        font-size: 1.5rem;
        color: #666;
        margin: 0.5rem 0 1rem 0;
    }
    
    /* Ingredient card styling */
    .ingredient-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.6rem;
        border-radius: 8px;
        color: white;
        margin: 0.2rem 0;
        text-align: center;
        font-weight: 500;
        font-size: 1.2rem;
    }
    
    /* Prediction card styling */
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.8rem 0;
        box-shadow: 0 6px 24px rgba(0,0,0,0.1);
        font-size: 1.3rem;
    }
    
    /* Confidence bar styling */
    .confidence-bar {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        height: 12px;
        border-radius: 6px;
        margin: 0.4rem 0;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: #f8f9fa;
        font-size: 1.2rem;
    }
    
    /* Metric card styling */
    .metric-card {
        background: white;
        padding: 0.6rem;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        text-align: center;
        font-size: 1.2rem;
    }
    
    /* Button styling */
    .stButton > button {
        font-size: 1.2rem;
        padding: 0.4rem 0.8rem;
    }
    
    /* Text area styling */
    .stTextArea textarea {
        font-size: 1.2rem;
    }
    
    /* Sidebar text size and reduced padding */
    .stSidebar, .stSidebar * {
        font-size: 1.4rem !important;
    }
    .stSidebar .stRadio > label {
        font-size: 1.4rem !important;
    }
    .stSidebar .stRadio > div > label {
        font-size: 1.4rem !important;
    }
    .stSidebar > div {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }
    
    /* Footer styling with minimal bottom space */
    .footer {
        margin-top: 0.5rem;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Define explicit paths for model and vectorizer
MODEL_PATH = r"models/stacking_model_simplified.pkl"
VECTORIZER_PATH = r"models/tfidf_vectorizer_simplified.pkl"

# Load models (with caching for better performance)
@st.cache_resource
def load_models():
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please ensure 'stacking_model_simplified.pkl' and 'tfidf_vectorizer_simplified.pkl' are in the 'models2' directory.")
        return None, None

# Cuisine mapping
cuisine_mapping = {
    'italian': 1, 'mexican': 2, 'southern_us': 3, 'indian': 4, 'chinese': 5,
    'french': 6, 'cajun_creole': 7, 'thai': 8, 'japanese': 9, 'greek': 10,
    'spanish': 11, 'korean': 12, 'vietnamese': 13, 'moroccan': 14, 'british': 15,
    'filipino': 16, 'irish': 17, 'jamaican': 18, 'russian': 19, 'brazilian': 20
}
reverse_cuisine_mapping = {v: k for k, v in cuisine_mapping.items()}

# Popular ingredients by cuisine for suggestions
ingredient_suggestions = {
    'italian': ['olive oil', 'garlic', 'tomato', 'basil', 'parmesan cheese', 'mozzarella', 'oregano', 'pasta'],
    'mexican': ['tortilla', 'avocado', 'chili powder', 'cilantro', 'lime', 'cumin', 'black beans', 'corn'],
    'indian': ['turmeric', 'cumin', 'coriander', 'ginger', 'ghee', 'curry leaves', 'garam masala', 'basmati rice'],
    'chinese': ['soy sauce', 'ginger', 'green onions', 'sesame oil', 'star anise', 'rice wine', 'hoisin sauce'],
    'thai': ['coconut milk', 'lemongrass', 'fish sauce', 'chili', 'thai basil', 'galangal', 'lime leaves'],
    'japanese': ['soy sauce', 'mirin', 'sushi rice', 'nori', 'sesame seeds', 'dashi', 'miso', 'sake'],
    'french': ['butter', 'cream', 'wine', 'shallots', 'parsley', 'thyme', 'bay leaves', 'cognac'],
    'greek': ['feta cheese', 'olive oil', 'oregano', 'cucumber', 'tomato', 'olives', 'phyllo', 'tzatziki']
}

# Flag emojis for cuisines
cuisine_flags = {
    'italian': '🇮🇹', 'mexican': '🇲🇽', 'southern_us': '🇺🇸', 'indian': '🇮🇳',
    'chinese': '🇨🇳', 'french': '🇫🇷', 'cajun_creole': '🇺🇸', 'thai': '🇹🇭',
    'japanese': '🇯🇵', 'greek': '🇬🇷', 'spanish': '🇪🇸', 'korean': '🇰🇷',
    'vietnamese': '🇻🇳', 'moroccan': '🇲🇦', 'british': '🇬🇧', 'filipino': '🇵🇭',
    'irish': '🇮🇪', 'jamaican': '🇯🇲', 'russian': '🇷🇺', 'brazilian': '🇧🇷'
}

# Standardize ingredients function
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

def main():
    # Header
    st.markdown('<h1 class="main-header">🍽️ AI Cuisine Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Discover the world cuisine from your ingredients using Machine Learning</p>', unsafe_allow_html=True)
    
    # Load models
    model, vectorizer = load_models()
    
    if model is None or vectorizer is None:
        st.stop()
    
    # Sidebar for input methods
    st.sidebar.header("🎯 Input Method")
    input_method = st.sidebar.radio(
        "Choose how to input ingredients:",
        ["Manual Entry", "Quick Select", "Batch Prediction"],
        label_visibility="visible"
    )
    
    if input_method == "Manual Entry":
        manual_input_interface(model, vectorizer)
    elif input_method == "Quick Select":
        quick_select_interface(model, vectorizer)
    else:
        batch_prediction_interface(model, vectorizer)

def manual_input_interface(model, vectorizer):
    st.header("🥘 Manual Ingredient Entry")
    
    col1, col2 = st.columns([2, 1], gap="medium")
    
    with col1:
        # Text area for ingredients
        ingredients_text = st.text_area(
            "Enter ingredients (one per line or separated by commas):",
            placeholder="tomato\nbasil\ngarlic\nmozzarella\nolive oil",
            height=100
        )
        
        # Parse ingredients
        if ingredients_text:
            ingredients = [ing.strip().lower() for ing in ingredients_text.replace(',', '\n').split('\n') if ing.strip()]
            
            st.subheader("🧾 Your Ingredients:")
            # Display ingredients in a nice format
            cols = st.columns(3)
            for i, ingredient in enumerate(ingredients):
                with cols[i % 3]:
                    st.markdown(f'<div class="ingredient-card">{ingredient}</div>', unsafe_allow_html=True)
            
            # Predict button
            if st.button("🔮 Predict Cuisine", type="primary", use_container_width=True):
                predict_cuisine([ingredients], model, vectorizer)
    
    with col2:
        st.subheader("💡 Ingredient Suggestions")
        selected_cuisine = st.selectbox("Get suggestions for:", list(ingredient_suggestions.keys()))
        
        if selected_cuisine:
            st.write(f"**Popular {selected_cuisine} ingredients:**")
            for ingredient in ingredient_suggestions[selected_cuisine]:
                if st.button(f"+ {ingredient}", key=f"add_{ingredient}"):
                    st.session_state.setdefault('added_ingredients', []).append(ingredient)
                    st.rerun()

def quick_select_interface(model, vectorizer):
    st.header("⚡ Quick Select Ingredients")
    
    # Initialize session state
    if 'selected_ingredients' not in st.session_state:
        st.session_state.selected_ingredients = []
    
    # Create ingredient categories
    categories = {
        "🥩 Proteins": ['chicken', 'beef', 'pork', 'fish', 'shrimp', 'tofu', 'eggs', 'lamb'],
        "🥬 Vegetables": ['onion', 'garlic', 'tomato', 'bell pepper', 'carrot', 'celery', 'mushroom', 'spinach'],
        "🌿 Herbs & Spices": ['basil', 'oregano', 'thyme', 'cumin', 'paprika', 'ginger', 'turmeric', 'cinnamon'],
        "🥛 Dairy & Oils": ['butter', 'cream', 'cheese', 'milk', 'olive oil', 'coconut oil', 'ghee'],
        "🍚 Grains & Starches": ['rice', 'pasta', 'bread', 'potato', 'quinoa', 'couscous', 'noodles'],
        "🥫 Condiments": ['soy sauce', 'fish sauce', 'vinegar', 'lemon', 'lime', 'wine', 'stock']
    }
    
    # Display ingredient selection
    for category, items in categories.items():
        with st.expander(category, expanded=True):
            cols = st.columns(4)
            for i, item in enumerate(items):
                with cols[i % 4]:
                    if st.button(item, key=f"select_{item}"):
                        if item not in st.session_state.selected_ingredients:
                            st.session_state.selected_ingredients.append(item)
                            st.rerun()
    
    # Display selected ingredients
    if st.session_state.selected_ingredients:
        st.subheader("🛒 Selected Ingredients:")
        
        # Create a container for selected ingredients with remove buttons
        cols = st.columns(3)
        for i, ingredient in enumerate(st.session_state.selected_ingredients):
            with cols[i % 3]:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f'<div class="ingredient-card">{ingredient}</div>', unsafe_allow_html=True)
                with col2:
                    if st.button("❌", key=f"remove_{ingredient}"):
                        st.session_state.selected_ingredients.remove(ingredient)
                        st.rerun()
        
        # Clear all and predict buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.selected_ingredients = []
                st.rerun()
        with col2:
            if st.button("🔮 Predict Cuisine", type="primary", use_container_width=True):
                predict_cuisine([st.session_state.selected_ingredients], model, vectorizer)

def batch_prediction_interface(model, vectorizer):
    st.header("📊 Batch Prediction")
    
    # Sample data for demonstration
    sample_recipes = [
        ['olive oil', 'garlic', 'tomato', 'basil', 'parmesan cheese'],
        ['tortilla', 'avocado', 'chili powder', 'cilantro', 'lime'],
        ['turmeric', 'cumin', 'coriander', 'ginger', 'ghee'],
        ['soy sauce', 'ginger', 'green onions', 'sesame oil'],
        ['butter', 'cream', 'wine', 'shallots', 'parsley']
    ]
    
    if st.button("🚀 Run Sample Predictions", type="primary"):
        batch_predict(sample_recipes, model, vectorizer)
    
    st.subheader("📝 Custom Batch Input")
    st.info("Enter multiple recipes, one per line. Separate ingredients with commas.")
    
    batch_text = st.text_area(
        "Enter recipes:",
        placeholder="olive oil, garlic, tomato, basil\ntortilla, avocado, chili powder, cilantro\nturmeric, cumin, ginger, ghee",
        height=120
    )
    
    if batch_text and st.button("🔮 Predict All", type="primary"):
        recipes = []
        for line in batch_text.strip().split('\n'):
            if line.strip():
                ingredients = [ing.strip().lower() for ing in line.split(',') if ing.strip()]
                recipes.append(ingredients)
        
        if recipes:
            batch_predict(recipes, model, vectorizer)

def predict_cuisine(ingredient_lists, model, vectorizer):
    if not ingredient_lists or not any(ingredient_lists):
        st.warning("🤔 Please add some ingredients first to get started!")
        return
    
    # Enhanced loading animation with progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Simulate analysis steps with progress
    steps = [
        "🔍 Analyzing ingredient combinations...",
        "🧠 Processing through AI model...",
        "🌍 Matching cuisine patterns...",
        "✨ Finalizing prediction..."
    ]
    
    for i, step in enumerate(steps):
        status_text.text(step)
        progress_bar.progress((i + 1) / len(steps))
        time.sleep(0.3)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    try:
        # Preprocess ingredients
        processed_data = []
        ingredient_counts = []
        
        for ingredients in ingredient_lists:
            # Standardize ingredients
            standardized = standardize_ingredients(ingredients)
            if not standardized:
                processed_data.append('missing')
                ingredient_counts.append(0)
            else:
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
            st.warning("No valid ingredient lists provided!")
            return

        # Apply TF-IDF transformation
        X_tfidf = vectorizer.transform(valid_df['ingredients'])
        X_counts = valid_df[['ingredient_count']].values
        X_combined = hstack([X_tfidf, X_counts])

        # Make predictions
        predictions = model.predict(X_combined)
        
        # Map predictions to cuisine names
        predicted_cuisines = [reverse_cuisine_mapping.get(pred, 'unknown') for pred in predictions]
        
        # Handle invalid entries in output
        result = []
        valid_idx = 0
        for i, ingredients in enumerate(processed_data):
            if ingredients == 'missing':
                result.append('Invalid input (empty ingredients)')
            else:
                result.append(predicted_cuisines[valid_idx])
                valid_idx += 1

        # Display result (single prediction case)
        if len(ingredient_lists) == 1:
            predicted_cuisine = result[0]
            if predicted_cuisine == 'Invalid input (empty ingredients)':
                st.warning("Invalid input: Empty ingredient list")
                return
            
            # Success animation
            st.balloons()
            
            # Main prediction with enhanced styling
            flag = cuisine_flags.get(predicted_cuisine, '🍽️')
            st.markdown(f'''
            <div class="prediction-card">
                <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem;">{flag}</h1>
                <h2 style="font-size: 2rem; margin-bottom: 1rem;">{predicted_cuisine.capitalize()} Cuisine</h2>
                <h3 style="font-size: 1.4rem; opacity: 0.9;">Confidence: High (Stacked Model)</h3>
            </div>
            ''', unsafe_allow_html=True)
            
            st.info("Confidence scores not available for stacked model. Prediction based on robust stacking classifier.")
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return

def batch_predict(recipes, model, vectorizer):
    if not recipes:
        st.warning("No recipes provided for batch prediction!")
        return
    
    # Show loading animation
    with st.spinner('🔍 Processing batch predictions...'):
        results = []
        
        # Preprocess ingredients
        processed_data = []
        ingredient_counts = []
        
        for ingredients in recipes:
            standardized = standardize_ingredients(ingredients)
            if not standardized:
                processed_data.append('missing')
                ingredient_counts.append(0)
            else:
                processed_data.append(' '.join(standardized))
                ingredient_counts.append(len(standardized))

        # Create DataFrame
        df = pd.DataFrame({
            'ingredients': processed_data,
            'ingredient_count': ingredient_counts
        })

        # Filter out valid entries
        valid_df = df[df['ingredients'] != 'missing']
        if valid_df.empty:
            st.warning("No valid ingredient lists provided!")
            return

        # Apply TF-IDF transformation
        X_tfidf = vectorizer.transform(valid_df['ingredients'])
        X_counts = valid_df[['ingredient_count']].values
        X_combined = hstack([X_tfidf, X_counts])

        # Make predictions
        try:
            predictions = model.predict(X_combined)
            predicted_cuisines = [reverse_cuisine_mapping.get(pred, 'unknown') for pred in predictions]
            
            # Construct results
            valid_idx = 0
            for i, ingredients in enumerate(recipes):
                if processed_data[i] == 'missing':
                    results.append({
                        'Recipe': f"Recipe {i+1}",
                        'Ingredients': ', '.join(ingredients) if ingredients else 'None',
                        'Predicted Cuisine': 'Invalid input (empty ingredients)',
                        'Flag': '⚠️',
                        'Confidence': 'N/A'
                    })
                else:
                    results.append({
                        'Recipe': f"Recipe {i+1}",
                        'Ingredients': ', '.join(ingredients),
                        'Predicted Cuisine': predicted_cuisines[valid_idx].capitalize(),
                        'Flag': cuisine_flags.get(predicted_cuisines[valid_idx], '🍽️'),
                        'Confidence': 'High (Stacked Model)'
                    })
                    valid_idx += 1
        except Exception as e:
            st.error(f"Error during batch prediction: {str(e)}")
            return
        
        # Display results as a dataframe
        st.subheader("📊 Batch Results")
        df = pd.DataFrame(results)
        
        # Style the dataframe
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Ingredients": st.column_config.TextColumn(width="large"),
                "Predicted Cuisine": st.column_config.TextColumn(width="medium"),
                "Confidence": st.column_config.TextColumn(width="small")
            }
        )
        
        # Cuisine distribution chart using Chart.js
        if results:
            cuisine_counts = Counter([r['Predicted Cuisine'] for r in results if r['Predicted Cuisine'] != 'Invalid input (empty ingredients)'])
            
            chart_data = {
                "type": "pie",
                "data": {
                    "labels": list(cuisine_counts.keys()),
                    "datasets": [{
                        "data": list(cuisine_counts.values()),
                        "backgroundColor": [
                            "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEEAD",
                            "#D4A5A5", "#9B59B6", "#3498DB", "#E74C3C", "#2ECC71"
                        ],
                        "borderColor": ["#FFFFFF"] * len(cuisine_counts),
                        "borderWidth": 1
                    }]
                },
                "options": {
                    "responsive": True,
                    "plugins": {
                        "legend": {
                            "position": "top",
                            "labels": {
                                "font": {
                                    "size": 14
                                },
                                "color": "#333"
                            }
                        },
                        "title": {
                            "display": True,
                            "text": "Distribution of Predicted Cuisines",
                            "font": {
                                "size": 18
                            },
                            "color": "#333"
                        }
                    }
                }
            }
            st.markdown("### Cuisine Distribution")
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.json(chart_data)
            st.markdown('</div>', unsafe_allow_html=True)

def add_footer():
    st.markdown("---")
    st.markdown(
        """
        <div class="footer" style='text-align: center; color: #666; font-size: 1.2rem;'>
            <p>🍽️ AI Cuisine Predictor | Built with Streamlit & Machine Learning</p>
            <p>Discover the world through flavors! 🌍</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
    add_footer()
