import sys
import pandas as pd
from sqlalchemy import create_engine
import os


def load_data(messages_filepath, categories_filepath) -> pd.DataFrame:
    ''' Handles CSV files reading to dataframes and returns a merged one'''
    messages_df = load_file(messages_filepath)
    categories_df = load_file(categories_filepath)
    df = merge_dataframes(messages_df, categories_df)
    return df


def clean_data(df, power_bi_format) -> pd.DataFrame:
    ''' Cleans data and create columns based on categories'''
    categories = split_categories(df)
    categories = drop_text(categories, power_bi_format)
    remove_columns(df, ["categories"])
    df = concat_dataframes(df, categories)
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename) -> None:
    ''' Handles the output of the data to a sqlite database'''
    export_to_sql(df, database_filename)


def load_file(file_name) -> pd.DataFrame:
    ''' Read a CSV file and loads it to a pandas dataframe '''
    df = pd.read_csv(file_name)
    return df


def merge_dataframes(first_df, second_df, merge_on="id") -> pd.DataFrame:
    ''' Joins two dataframes on a field, default on id '''
    df = first_df.merge(right=second_df, on=merge_on)
    return df


def split_categories(df) -> pd.DataFrame:
    ''' split a semicolon separeated string into 
        different columns.
        '''
    categories_labels = {}
    [categories_labels.update({i: x[0]}) for i, x in enumerate(
        df["categories"].str.split(pat=";", expand=True).loc[0].str.split(pat="-"))]
    categories = df["categories"].str.split(
        pat=";", expand=True).rename(columns=categories_labels)
    return categories


def drop_text(categories, power_bi_format) -> pd.DataFrame:
    ''' Splits a string separated by '-' and saves the 
        numbers on a pd.DataFrame. Ex. related-1 -> 1
        '''
    print(f"Using PowerBI format: {power_bi_format}")
    if power_bi_format:
        def filter_number(x): return True if int(
            x.split("-")[1]) == 1 else False
    else:
        def filter_number(x): return int(x.split("-")[1])
    for column in categories:
        categories[column] = categories[column].apply(filter_number)
    return categories


def remove_columns(df, col_names: list) -> None:
    ''' Remove columns from dataframe'''
    df.drop(columns=col_names, inplace=True)


def concat_dataframes(first_df, second_df) -> pd.DataFrame:
    ''' Concat columns of two dataframes into one DataFrame'''
    return pd.concat([first_df, second_df], axis=1)


def export_to_sql(df: pd.DataFrame, file_name, table_name="disaster_Fact") -> None:
    ''' Verify if database exists, if true deletes old DB and saves a new file with the same name'''
    if os.path.exists(file_name):
        os.remove(file_name)
    engine = create_engine(f'sqlite:///{file_name}')
    df.to_sql(f'{table_name}', engine, index=False)
    print(f"Saved table {table_name} in {file_name} ")


def main():
    if len(sys.argv) == 5:

        messages_filepath, categories_filepath, database_filepath, power_bi_format = sys.argv[
            1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df, power_bi_format)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
