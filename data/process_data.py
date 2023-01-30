import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    # loading the data and return the merged dataframe of the two .csv
    # messages_filepath: string with path of messages data
    # categories_filepath: string with path of categories data
    
    df_messages = pd.read_csv(messages_filepath)
    df_categories = pd.read_csv(categories_filepath)
    
    # Merger: we simply concatenate
    df = pd.merge(df_messages, df_categories, on=['id'])
    
    return df



def clean_data(df):
    # clean df: creates categories columns, check NaN, check for duplicates
    # return clean df
    
    # First lets create a column for each category
    df = pd.concat([df, df.categories.str.split(';',expand=True)],axis=1)
    # Give proper name for new columns (starting at 5th), by removing last 2 characters (-0 or -1)
    column_names = df.iloc[1,5:].astype("string").str[:-2]
    df.rename(columns=dict(zip(df.columns[5:], column_names)), inplace=True)
    
    # convert all new columns values to 0 or 1
    for col_num in range(5,df.shape[1]):
        df.iloc[:,col_num] =  df.iloc[:,col_num].str[-1:].astype(int)
    
    # drop useless column
    df.drop(['categories'], axis=1,inplace=True)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    # drop duplicates IDs too.
    row_to_drop = df.id[df.id.duplicated()].index
    df.drop(row_to_drop, axis=0, inplace=True)
    
    # we noticed "related" values can be 0,1 or 2. We force all the 2-> 1 to have only binaries
    df.related = df.related.replace(2,1)
    
    return df



def save_data(df, database_filename):
    # Save data from df into the SQL database
    engine = create_engine('sqlite:///' + database_filename)
    # we replace previous one if exist. If it is additional data, need to switch to 'append'
    df.to_sql('DisasterTable', engine, index=False, if_exists='replace')
    
    return None




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