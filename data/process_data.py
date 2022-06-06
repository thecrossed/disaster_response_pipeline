# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sys


def load_data(messages_filepath, categories_filepath):
    """
    Input:
    messages_filepath: file path of messages data set
    categories_filepath: file path of catogires data set
    
    Output:
    
    df: merged dataset from messages and categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on="id")

    return df

def clean_data(df):
    """
    input: dataframe
    
    output: cleaned dataframe
    """
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";",expand=True)
    columns = []
    for category in categories.iloc[0]:
        columns.append(category.split("-")[0])
    categories.columns = columns
    for column in categories:
        # set each value to be the last character of the string
        # transform categories data into int
        categories[column] = categories[column].apply(lambda x: x[-1]).astype(int)
    df = df.drop(columns = ['categories'])
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df = df.drop_duplicates()
    # replace 2 with 1 in the column related
    df['related'] = df['related'].replace(2,1)
    return df


def save_data(df, database_filename):
    """
    input: 
    df: dataframe that to be loaded to database
    
    database_filename: file name in the database
    """
    engine = create_engine("sqlite:///" + database_filename)
    df.to_sql('MessageCategory', engine, index=False, if_exists='replace')
  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()