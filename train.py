import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import joblib
import re
from scipy.sparse import hstack

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

# Load dataset
recipe = pd.read_json(r'data\train.json')

# Convert cuisine to numeric
cuisine_mapping = {
    'italian': 1, 'mexican': 2, 'southern_us': 3, 'indian': 4, 'chinese': 5,
    'french': 6, 'cajun_creole': 7, 'thai': 8, 'japanese': 9, 'greek': 10,
    'spanish': 11, 'korean': 12, 'vietnamese': 13, 'moroccan': 14, 'british': 15,
    'filipino': 16, 'irish': 17, 'jamaican': 18, 'russian': 19, 'brazilian': 20
}
recipe['cuisine'] = recipe['cuisine'].map(cuisine_mapping)

# Preprocess ingredients
recipe['ingredients'] = recipe['ingredients'].apply(standardize_ingredients)
recipe['ingredient_count'] = recipe['ingredients'].apply(len)

# Split data
X = recipe[['ingredients', 'ingredient_count']]
y = recipe['cuisine']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Convert ingredients to strings
X_train['ingredients'] = X_train['ingredients'].apply(lambda x: ' '.join(x) if x else 'missing')
X_test['ingredients'] = X_test['ingredients'].apply(lambda x: ' '.join(x) if x else 'missing')

# Remove missing
X_train = X_train[X_train['ingredients'] != 'missing']
y_train = y_train[X_train.index]
X_test = X_test[X_test['ingredients'] != 'missing']
y_test = y_test[X_test.index]

# Extract counts
train_counts = X_train[['ingredient_count']].values
test_counts = X_test[['ingredient_count']].values

# TF-IDF
tfidf = TfidfVectorizer(stop_words=None, ngram_range=(1, 2), max_features=3000)
X_train_tfidf = tfidf.fit_transform(X_train['ingredients'])
X_test_tfidf = tfidf.transform(X_test['ingredients'])
print("TF-IDF Vocabulary size:", len(tfidf.get_feature_names_out()))

# Combine features
X_train_combined = hstack([X_train_tfidf, train_counts])
X_test_combined = hstack([X_test_tfidf, test_counts])

# Define models
xgb_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=20,
    max_depth=5,
    learning_rate=0.1,
    n_estimators=100,
    random_state=1
)
lgb_model = lgb.LGBMClassifier(
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=100,
    random_state=1,
    verbose=1
)
estimators = [('xgb', xgb_model), ('lgb', lgb_model)]
stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(multi_class='multinomial', max_iter=1000),
    cv=3,
    n_jobs=2,
    verbose=2
)

# Train
print("Starting training...")
stacking_model.fit(X_train_combined, y_train)

# Save
joblib.dump(stacking_model, 'stacking_model_simplified.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer_simplified.pkl')
print("Models saved.")

# Predict and evaluate
y_pred = stacking_model.predict(X_test_combined)
accuracy = accuracy_score(y_test, y_pred)
print("Stacking Model Accuracy:", accuracy)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
cuisine_labels = [k for k, v in sorted(cuisine_mapping.items(), key=lambda x: x[1])]
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=cuisine_labels, yticklabels=cuisine_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Cuisine')
plt.ylabel('True Cuisine')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('confusion_matrix_simplified.png')
plt.show()