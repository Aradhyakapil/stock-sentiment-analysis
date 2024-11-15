# %%
import numpy as np
import pandas as pd
import nltk

import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score, precision_score, recall_score
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')


# %%
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# %%

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')


# %%
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# %%
nltk.data.path.append(r'C:\Users\hp\AppData\Local\Programs\Python\Python312\Lib\site-packages\nltk\nltk_data')


# %%
print(nltk.data.path)


# %%
print(word_tokenize("Testing NLTK punkt tokenizer."))

# %%

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def preprocess_text(headlines):
    corpus = []
    
    for headline in headlines:
        # Remove numbers and special characters but keep important punctuation
        headline = re.sub(r'[^a-zA-Z\.\!\?]', ' ', headline)
        
        # Convert to lowercase
        headline = headline.lower()
        
        # Tokenize
        words = word_tokenize(headline)
        
        # Enhanced stopwords list with custom financial terms to keep
        stop_words = set(stopwords.words('english'))
        financial_terms = {'increase', 'decrease', 'up', 'down', 'rise', 'fall', 'grow', 'drop'}
        stop_words = stop_words - financial_terms
        
        # Remove stopwords and lemmatize
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        
        # Join words back together
        headline = ' '.join(words)
        corpus.append(headline)
    
    return corpus


# %%
def create_word_clouds(train_corpus, y_train):
    """Generate and display word clouds for different market movements"""
    # Separate words for different market movements
    down_words = []
    up_words = []
    
    for text, label in zip(train_corpus, y_train):
        if label == 0:
            down_words.append(text)
        else:
            up_words.append(text)
    
    # Combine all words for each category
    down_text = ' '.join(down_words)
    up_text = ' '.join(up_words)
    
    # Create and plot word clouds
    plt.figure(figsize=(20, 8))
    
    # Down/Same trend word cloud
    plt.subplot(1, 2, 1)
    wordcloud_down = WordCloud(
        width=800, 
        height=400,
        background_color='white',
        colormap='Reds',
        max_words=100
    ).generate(down_text)
    
    plt.imshow(wordcloud_down)
    plt.axis('off')
    plt.title('Words Associated with Down/Same Stock Movement', fontsize=16, pad=20)
    
    # Up trend word cloud
    plt.subplot(1, 2, 2)
    wordcloud_up = WordCloud(
        width=800, 
        height=400,
        background_color='white',
        colormap='Greens',
        max_words=100
    ).generate(up_text)
    
    plt.imshow(wordcloud_up)
    plt.axis('off')
    plt.title('Words Associated with Up Stock Movement', fontsize=16, pad=20)
    
    plt.tight_layout(pad=3.0)
    plt.show()

# %%

def create_sentiment_features(headlines):
    """Create sentiment-based features using NLTK's VADER"""
    from nltk.sentiment import SentimentIntensityAnalyzer
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()
    
    sentiments = []
    for headline in headlines:
        scores = sia.polarity_scores(headline)
        sentiments.append([scores['neg'], scores['neu'], scores['pos'], scores['compound']])
    
    return np.array(sentiments)


# %%

def plot_learning_curves(model, X_train, y_train, X_test, y_test, title):
    """Plot learning curves to diagnose bias/variance"""
    from sklearn.model_selection import learning_curve
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10))
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.plot(train_sizes, test_mean, label='Cross-validation score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.title(f'Learning Curves - {title}')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


# %%

def plot_roc_curves(models_dict, X_test, y_test):
    """Plot ROC curves for all models"""
    plt.figure(figsize=(10, 6))
    
    for name, model in models_dict.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


# %%

def create_feature_importance_plot(model, feature_names, title):
    """Plot feature importance for models that support it"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return
    
    feature_imp = pd.Series(importances, index=feature_names)
    plt.figure(figsize=(12, 6))
    feature_imp.nlargest(20).plot(kind='bar')
    plt.title(f'Top 20 Most Important Features - {title}')
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


# %%

# Load and prepare the data
# Loading the dataset
df = pd.read_csv(r'C:\Users\hp\Desktop\stock\data\Stock_Headlines.csv', encoding = 'ISO-8859-1')
df_copy = df.copy()
df_copy.reset_index(inplace=True)


# %%
df.head(3)

# %%
# Importing essential libraries for 
!pip install seaborn

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Visualizing the count of 'Label' column with distinct colors for each label
plt.figure(figsize=(8, 8))
sns.countplot(x='Label', data=df, palette={"0": "skyblue", "1": "salmon"})
plt.xlabel('Stock Sentiments (0-Down/Same, 1-Up)')
plt.ylabel('Count')
plt.show()


# %%
df.isna().any()

# %%

# Dropping NaN values
df.dropna(inplace=True)
print(df.shape)

# %%

# Split into train and test
train = df_copy[df_copy['Date'] < '20150101']
test = df_copy[df_copy['Date'] > '20141231']
print('Train size: {}, Test size: {}'.format(train.shape, test.shape))

# %%

# Splitting the dataset
y_train = train['Label']
train = train.iloc[:, 3:28]
y_test = test['Label']
test = test.iloc[:, 3:28]


# %%

# Prepare headlines
train_headlines = []
test_headlines = []


# %%

for row in range(0, train.shape[0]):
    train_headlines.append(' '.join(str(x) for x in train.iloc[row, 3:28]))

for row in range(0, test.shape[0]):
    test_headlines.append(' '.join(str(x) for x in test.iloc[row, 3:28]))


# %%

# Preprocess the headlines
train_corpus = preprocess_text(train_headlines)
test_corpus = preprocess_text(test_headlines)


# %%
# Generate word clouds
print("Generating word clouds for market movements...")
create_word_clouds(train_corpus, y_train)

# %%

# Create TF-IDF vectors with enhanced parameters
tfidf = TfidfVectorizer(
    max_features=15000,  # Increased from 10000
    ngram_range=(1, 3),  # Include uni, bi, and trigrams
    min_df=2,            # Minimum document frequency
    max_df=0.95         # Maximum document frequency
)


# %%

X_train_tfidf = tfidf.fit_transform(train_corpus).toarray()
X_test_tfidf = tfidf.transform(test_corpus).toarray()


# %%

# Add sentiment features
X_train_sentiment = create_sentiment_features(train_headlines)
X_test_sentiment = create_sentiment_features(test_headlines)


# %%

# Combine TF-IDF and sentiment features
X_train = np.hstack((X_train_tfidf, X_train_sentiment))
X_test = np.hstack((X_test_tfidf, X_test_sentiment))


# %%

# Apply SMOTE with better parameters
!pip install imbalanced-learn

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


# %%

# Enhanced models with better parameters
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB


# %%

models = {
    'Logistic Regression': LogisticRegression(
        C=1.0,
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    )
   
}


# %%

# Train and evaluate models
trained_models = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_resampled, y_train_resampled)
    trained_models[name] = model
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Print classification report
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Down/Same', 'Up'],
                yticklabels=['Down/Same', 'Up'])
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # Plot learning curves
    plot_learning_curves(model, X_train_resampled, y_train_resampled, X_test, y_test, name)
    
    # Plot feature importance
    if name != 'Naive Bayes':  # Skip for Naive Bayes as it doesn't have feature importance
        feature_names = list(tfidf.get_feature_names_out()) + ['neg', 'neu', 'pos', 'compound']
        create_feature_importance_plot(model, feature_names, name)


# %%

# Plot ROC curves for all models
plot_roc_curves(trained_models, X_test, y_test)


# %%

# Compare model performances with additional metrics
def compare_models_detailed():
    from sklearn.metrics import f1_score, roc_auc_score
    
    metrics = {model_name: {} for model_name in models.keys()}
    
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics[name]['Accuracy'] = accuracy_score(y_test, y_pred)
        metrics[name]['Precision'] = precision_score(y_test, y_pred)
        metrics[name]['Recall'] = recall_score(y_test, y_pred)
        metrics[name]['F1 Score'] = f1_score(y_test, y_pred)
        metrics[name]['ROC AUC'] = roc_auc_score(y_test, y_pred_proba)
    
    metrics_df = pd.DataFrame(metrics).transpose()
    
    # Plot comparison
    plt.figure(figsize=(15, 8))
    metrics_df.plot(kind='bar', width=0.8)
    plt.title('Detailed Model Performance Comparison')
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return metrics_df


# %%

# Display detailed comparison
metrics_df = compare_models_detailed()
print("\nDetailed Model Performance Metrics:")
print(metrics_df.round(4))


# %%

# Save the best model and vectorizer
import pickle

# Find the best model based on F1 score
best_model_name = metrics_df['F1 Score'].idxmax()
best_model = trained_models[best_model_name]


# %%

print(f"\nBest performing model: {best_model_name}")

with open(r"C:\Users\hp\Desktop\stock\models\best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open(r"C:\Users\hp\Desktop\stock\models\tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)
    
   


