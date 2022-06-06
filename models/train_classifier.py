# import libraries
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
import pickle
import sys



nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    """
    input:
    database_filepath: file path of the database
    
    output: 
    X: feature variable
    Y: target variable
    category_names: names of tagrt variable
    """
    #database_filepath = database_filepath.replace("data/","")
    #print("sqlite:///"+ database_filepath)
    engine = create_engine("sqlite:///" + database_filepath)
    df = pd.read_sql_table('MessageCategory', "sqlite:///" + database_filepath) 
    X = df['message']
    Y = df.iloc[: , -35:]
    category_names = Y.columns
    
    return X, Y , category_names

def tokenize(text):
    """
    input: 
    text: the piece of text we want to tokenize
    
    output:
    a list of tokens whic are 
    1. lowercased, 
    2. containing only alphabets or numbers 
    3. removed stop words
    3. and lemmatized
    """
    # remove non-numeric or non-alphabet text
    # and lower case all the text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # tokenize text
    tokens = word_tokenize(text)
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    """
    This function building a model by pipepline function.
    
    It uses FeatureUnion to implement different models all together
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))

        ])),

        ('clf', RandomForestClassifier())
        ])
    """
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'clf__n_estimators': [50, 100, 200],
        'clf__min_samples_split': [2, 3, 4]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    """
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    input:
    model: the model trained to predict y
    X_test: test data set in the X variable
    Y_test: test data set in the Y variable
    category_names: columns in the Y dataset
    output:
    print the performance of the predicted outcome against the true Y test data
    """
    # predicted value
    y_pred = model.predict(X_test)
    target_names = []
    for col in range(len(category_names)):
        target_names.append('v_' + str(col))
    print(classification_report(Y_test, y_pred, target_names=target_names))


def save_model(model, model_filepath):
    """
    input:
    model: the model trained to predict y
    model_filepath: the place we store the model
    output:
    store the model into a pickle file
    """
    with open(model_filepath, 'wb') as files:
        pickle.dump(model, files)


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