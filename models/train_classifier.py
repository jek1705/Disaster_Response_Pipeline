import sys
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import pickle


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    # load data from database
    # return X : features variables
    # return Y : target variables
    # category_names : list of category names
    df = pd.read_sql_table('DisasterTable','sqlite:///' + database_filepath)
    
    X = df['message'].values
    Y = df.iloc[:,4:].values
    category_names = df.iloc[:,4:].columns
    
    return X, Y, category_names



def tokenize(text):
    # to transform messages into inputs for ML model
    # case, normalize, lemmatize, and tokenize text
    # return the tokeninzed words
    
    # first removing punctuation and lower casing
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Split text into words using NLTK
    words = word_tokenize(text)
    # Stop words removal
    words = [w for w in words if w not in stopwords.words("english")]
    # Lemmatization - Reduce words to their root form
    words = [WordNetLemmatizer().lemmatize(w).strip() for w in words]
    
    return words



def build_model():
    # Building a model based on sklearn MultiOutputClassifier to forecast multiple categories at once.
    # return the full pipeline model using GridSearch on some parameters to find best model
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
        ('mlpc', MultiOutputClassifier(MultinomialNB()))
    ])
    
    # text processing and model pipeline
    # define parameters for GridSearchCV
    # create gridsearch object and return as final model pipeline
    
    parameters = {
        'tfidf__ngram_range': ((1, 1), (1, 2)),
        'tfidf__max_features': (None, 50, 100)
        }
        
    model = GridSearchCV(pipeline, param_grid=parameters, cv=2,verbose=3)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    # evaluate model by printing precision, recall, f1-score for each category
        
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=category_names))
    
    return None


def save_model(model, model_filepath):
    # Save model as a pickle file in model_filepath to be re-used in the future
    pickle.dump(model, open(model_filepath, 'wb'))
    
    return None


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()