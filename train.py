import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib  # Added missing import
import uuid

# Load the dataset
recipe = pd.read_json(r'data\train.json')

# Convert cuisine column to numeric
cuisine_mapping = {
    'italian': 1, 'mexican': 2, 'southern_us': 3, 'indian': 4, 'chinese': 5,
    'french': 6, 'cajun_creole': 7, 'thai': 8, 'japanese': 9, 'greek': 10,
    'spanish': 11, 'korean': 12, 'vietnamese': 13, 'moroccan': 14, 'british': 15,
    'filipino': 16, 'irish': 17, 'jamaican': 18, 'russian': 19, 'brazilian': 20
}
recipe['cuisine'] = recipe['cuisine'].map(cuisine_mapping)

# Split input (ingredients) and output (cuisine)
X = recipe['ingredients']
y = recipe['cuisine']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Convert ingredients lists to strings and handle potential issues
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

# Ensure ingredients are lists and convert to strings
X_train['ingredients'] = X_train['ingredients'].apply(lambda x: ' '.join(x) if isinstance(x, list) and x else 'missing')
X_test['ingredients'] = X_test['ingredients'].apply(lambda x: ' '.join(x) if isinstance(x, list) and x else 'missing')

# Check for empty or invalid entries
print("Checking X_train['ingredients'] for issues:")
print("Number of empty strings:", (X_train['ingredients'] == '').sum())
print("Number of 'missing' entries:", (X_train['ingredients'] == 'missing').sum())
print("Sample of X_train['ingredients']:\n", X_train['ingredients'].head())

# Remove rows with 'missing' or empty ingredients
X_train = X_train[X_train['ingredients'] != 'missing']
y_train = y_train[X_train.index]
X_test = X_test[X_test['ingredients'] != 'missing']
y_test = y_test[X_test.index]

# Apply Bag of Words transformation with stop_words=None
cv = CountVectorizer(stop_words=None)
try:
    X_train_bow = cv.fit_transform(X_train['ingredients']).toarray()
    X_test_bow = cv.transform(X_test['ingredients']).toarray()
    print("Vocabulary size:", len(cv.get_feature_names_out()))
    print("Sample vocabulary:", cv.get_feature_names_out()[:10])
except ValueError as e:
    print("Error in CountVectorizer:", e)
    raise

# Define individual models
gnb = GaussianNB()
rf = RandomForestClassifier(n_estimators=100, random_state=1)
svm = SVC(kernel='linear', random_state=1)

# Define the voting classifier (hard voting)
ensemble_model = VotingClassifier(
    estimators=[
        ('gnb', gnb),
        ('rf', rf),
        ('svm', svm)
    ],
    voting='hard',
    verbose=True
)

# Train the ensemble model
ensemble_model.fit(X_train_bow, y_train)

# Save the trained model and vectorizer
joblib.dump(ensemble_model, 'ensemble_model.pkl')
joblib.dump(cv, 'count_vectorizer.pkl')
print("Model and vectorizer saved as 'ensemble_model.pkl' and 'count_vectorizer.pkl'")

# Make predictions
y_pred = ensemble_model.predict(X_test_bow)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Ensemble Model Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)

# 1. Heatmap of Confusion Matrix
plt.figure(figsize=(10, 8))
cuisine_labels = [k for k, v in sorted(cuisine_mapping.items(), key=lambda x: x[1])]
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=cuisine_labels, yticklabels=cuisine_labels)
plt.title('Confusion Matrix for Ensemble Model')
plt.xlabel('Predicted Cuisine')
plt.ylabel('True Cuisine')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# 2. Bar Plot of Feature Importance (Random Forest)
rf.fit(X_train_bow, y_train)
feature_importances = rf.feature_importances_
feature_names = cv.get_feature_names_out()
top_n = 20
top_indices = np.argsort(feature_importances)[::-1][:top_n]
top_features = [feature_names[i] for i in top_indices]
top_importances = feature_importances[top_indices]

plt.figure(figsize=(10, 6))
sns.barplot(x=top_importances, y=top_features, palette='viridis')
plt.title(f'Top {top_n} Important Ingredients (Random Forest)')
plt.xlabel('Feature Importance')
plt.ylabel('Ingredient')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

# 3. Histogram of Ingredient Counts per Recipe
recipe['ingredient_count'] = recipe['ingredients'].apply(lambda x: len(x) if isinstance(x, list) else 0)
plt.figure(figsize=(10, 6))
sns.histplot(recipe['ingredient_count'], bins=30, kde=True, color='purple')
plt.title('Distribution of Ingredient Counts per Recipe')
plt.xlabel('Number of Ingredients')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('ingredient_count_histogram.png')
plt.show()

# 4. Pie Chart of Cuisine Distribution
cuisine_counts = recipe['cuisine'].value_counts()
cuisine_names = [list(cuisine_mapping.keys())[list(cuisine_mapping.values()).index(i)] for i in cuisine_counts.index]
df_cuisine = pd.DataFrame({
    'Cuisine': cuisine_names,
    'Count': cuisine_counts.values
})
fig = px.pie(df_cuisine, names='Cuisine', values='Count', title='Cuisine Distribution')
fig.show()