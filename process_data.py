import pandas as pd
from sqlalchemy import create_engine

class etl_process:
    '''
    Reads two csv files, cleans int and save the result as a sqlite DB
    '''
    def __init__(self, messages_path, categories_path, db_name="disaster_data", table_name="disaster_Fact") -> None:
        self.messages_path = messages_path
        self.categories_path = categories_path
        self.db_name = db_name
        self.table_name = table_name

        print(f"Processing files: {self.messages_path}, {self.categories_path}.")
        messages_df = self.load_file(self.messages_path)
        categories_df = self.load_file(self.categories_path)
        self.df = self.merge_dataframes(messages_df, categories_df)
        categories = self.split_categories()
        categories = self.drop_text(categories)
        self.remove_columns(["categories"])
        self.df = self.concat_dataframes(self.df, categories)
        self.df.drop_duplicates(inplace=True)
        self.export_to_sql(self.db_name, self.table_name)
        print(f"{self.db_name} database created, records saved on table {self.table_name}")


    def load_file(self, file_name) -> pd.DataFrame:
        ''' Read a CSV file and loads it to a pandas dataframe '''
        df = pd.read_csv(file_name)
        return df

    def merge_dataframes(first_df, second_df, on="id") -> pd.DataFrame:
        ''' Joins two dataframes on a field, default on id '''
        df = first_df.merge(right=second_df, on="id")
        return df

    def split_categories(self) -> pd.DataFrame:
        ''' split a semicolon separeated string into 
        different columns.
        '''
        categories_labels = {}
        [categories_labels.update({i:x[0]}) for i, x in enumerate(self.df["categories"].str.split(pat=";", expand=True).loc[0].str.split(pat="-"))]
        categories = self.df["categories"].str.split(pat=";", expand=True).rename(columns=categories_labels)
        return categories
    
    def drop_text(categories) -> pd.DataFrame:
        ''' Splits a string separated by '-' and saves the 
        numbers on a pd.DataFrame. Ex. related-1 -> 1
        '''
        filter_number = lambda x: int(x.split("-")[1])
        for column in categories:
            categories[column] = categories[column].apply(filter_number)
        return categories

    def remove_columns(self, col_names: list) -> None:
        ''' Remove columns from dataframe'''
        self.df.drop(columns=col_names, inplace=True)

    def concat_dataframes(first_df, second_df) -> pd.DataFrame:
        ''' Concat columns of two dataframes into one Data Frame'''
        return pd.concat([first_df, second_df], axis=1)

    def export_to_sql(df: pd.DataFrame, file_name, table_name) -> None:
        engine = create_engine(f'sqlite:///{file_name}.db')
        df.to_sql(f'{table_name}', engine, index=False)
