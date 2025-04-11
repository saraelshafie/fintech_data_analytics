import os
import pandas as pd
from sqlalchemy import create_engine

DATABASE_URL = "postgresql://user:password@pgdatabase:5432/fintech_db"
engine = create_engine(DATABASE_URL)

def save_to_db(df, table_name):
    if(engine.connect()):
        print('Connected to Database')
        try:
            print('Writing cleaned dataset to database')
            df.to_sql(table_name, con=engine, if_exists='replace')
            print('Done writing to database')
        except ValueError as vx:
            print('Cleaned Table already exists.')
        except Exception as ex:
            print(ex)
    else:
        print('Failed to connect to Database')

def append_to_db(cleaned, name):
    if(engine.connect()):
        print('Connected to Database')
        try:
            print('Appending dataset to database')
            cleaned.to_sql(name, con=engine, if_exists='append')
            print('Done appending to database')
        except Exception as ex:
            print(ex)
    else:
        print('Failed to connect to Database')

def fetch_column_statistics(column_name, table_name):

    if engine.connect():
        print('Connected to Database')
        try:
            # Determine column data type
            data_type_query = f"""
                SELECT data_type 
                FROM information_schema.columns 
                WHERE table_name = '{table_name}' AND column_name = '{column_name}'
            """
            data_type = pd.read_sql_query(data_type_query, con=engine).iloc[0]['data_type']
            
            # Initialize result dictionary
            result = {"mean": None, "mode": None}
            
            mode_query = f"""
                SELECT {column_name} 
                FROM {table_name}
                GROUP BY {column_name}
                ORDER BY COUNT(*) DESC, {column_name} ASC
                LIMIT 1
            """
            mode_result = pd.read_sql_query(mode_query, con=engine).iloc[0][column_name]
            result["mode"] = mode_result
            
            # Calculate mean only for numeric columns
            if data_type in ["integer", "numeric", "real", "double precision", "bigint"]:
                mean_query = f"SELECT AVG({column_name}) AS mean FROM {table_name}"
                mean_result = pd.read_sql_query(mean_query, con=engine).iloc[0]['mean']
                result["mean"] = mean_result
            
            print(f"Column: {column_name}, Data Type: {data_type}, Mean: {result['mean']}, Mode: {result['mode']}")
            return result
        except Exception as ex:
            print(f"Error fetching statistics for column '{column_name}' in table '{table_name}': {ex}")
            return {"mean": None, "mode": None}
    else:
        print('Failed to connect to Database')
        return {"mean": None, "mode": None}
