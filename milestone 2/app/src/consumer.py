import json
from kafka import KafkaConsumer
import pandas as pd
from db import append_to_db
from cleaning import clean

def start_consumer():
    # Initialize Kafka consumer
    consumer = KafkaConsumer(
        'ms2-topic',
        bootstrap_servers='kafka:29092',
        auto_offset_reset='earliest',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )
    while True:
        message = consumer.poll(timeout_ms=2000)  
        
        if message:
            for tp, messages in message.items():
                for msg in messages:
                    if msg.value == 'EOF':
                        print("Received EOF, stopping consumer...")
                        consumer.close()
                        return
                    print(f"Received: {msg.value}")
                    new_row = pd.DataFrame([msg.value], columns=['Customer Id','Emp Title','Emp Length','Home Ownership', 'Annual Inc', 'Annual Inc Joint', 'Verification Status', 'Zip Code', 'Addr State','Avg Cur Bal','Tot Cur Bal', 'Loan Id','Loan Status','Loan Amount','State', 'Funded Amount', 'Term', 'Int Rate','Grade', 'Issue Date', 'Pymnt Plan', 'Type', 'Purpose', 'Description'])
                    cleaned = clean(new_row)
                    append_to_db(cleaned, 'fintech_data_MET_P02_52_0812_clean')

        else:
            print("No messages received, polling again...")

    

