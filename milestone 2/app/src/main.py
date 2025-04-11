import os
from cleaning import clean
from cleaning import fill_lookup_df
from db import append_to_db, save_to_db
import pandas as pd
from run_producer import start_producer, stop_container
from consumer import start_consumer

def create_lookup_df():
    lookup_df = pd.DataFrame(columns=['column', 'original', 'imputed'])
    return lookup_df


def main():
    data_dir = os.path.join(os.getcwd(), 'data/')
    cleaned_file = data_dir + 'fintech_data_MET_P2_52_0812_clean.csv'

    if os.path.exists(cleaned_file):
        print("Cleaned dataset already exists, loading from file.")
        cleaned_df = pd.read_csv(cleaned_file)
        lookup_df = pd.read_csv('data/lookup_table.csv')
        cleaned_df.set_index('customer_id', inplace=True)
        
        #save to database
        save_to_db(cleaned_df, 'fintech_data_MET_P02_52_0812_clean')
        save_to_db(lookup_df, 'lookup_fintech_data_MET_P02_52_0812')
    else:
        fintech_df = pd.read_csv(data_dir + 'fintech_data_43_52_0812.csv')
        lookup_df = create_lookup_df()
        cleaned_df = clean(fintech_df)
        lookup_df = fill_lookup_df(cleaned_df, lookup_df)

        cleaned_df.to_csv('data/fintech_data_MET_P2_52_0812_clean.csv')
        lookup_df.to_csv('data/lookup_table.csv', index=False)

        save_to_db(cleaned_df, 'fintech_data_MET_P02_52_0812_clean')
        save_to_db(lookup_df, 'lookup_fintech_data_MET_P02_52_0812')

    kafka_url = "localhost:9092"
    topic_name = "ms2-topic"
    id = "52_0812"
    producer_id = start_producer(id, kafka_url, topic_name)
    print(f"Producer started with ID {producer_id}")

    start_consumer()

    # Stop the Kafka producer container
    stop_container(producer_id)
    print("Producer stopped.")

if __name__ == '__main__':
    main()

