# Tweets multi classification

## Main conde in web_app

## Generate requierements.txt

- First: pip install pipreqs
- Run pipreqs, outputs requirements.txt of only used packages.

## Data exploratory analysis

- Data exploration.

## Data cleaning pipeline

- Load two csv files.
- create columns from categories column.
- Merge the data.
- Drop duplicates.
- Export data to sqlite db file.

## Data dashboard creation

### Sqlite connector installation

Go to http://www.ch-werner.de/sqliteodbc/ and install sqliteodbc_w64.exe

Get data
-> More
-> Load data using the ODBC connector using default or custom authentication field empty.
-> data path: database=C:\Users\luigi\projects\tweets_classification\web_app\data\DisasterResponse.db
