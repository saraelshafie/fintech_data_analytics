from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

from functions import extract_clean, transform, load_to_db
from fintech_dashboard import create_dashboard

default_args = {
    "owner": "data_engineering_team",
    "depends_on_past": False,
    'start_date': days_ago(2),
    "retries": 1,
}

dag = DAG(
    'fintech_etl_pipeline',
    default_args=default_args,
    description='fintech etl pipeline',
)

with DAG(
    dag_id='fintech_etl_pipeline',
    schedule_interval='@once',  # could be @daily, @hourly, etc or a cron expression '* * * * *'
    default_args=default_args,
    tags=['fintech-pipeline'],
) as dag:
    # Define the tasks
    extract_clean_task = PythonOperator(
        task_id='extract_clean',
        python_callable=extract_clean,
        op_kwargs={
            'file_path': '/opt/airflow/data/fintech_data_43_52_0812.csv'
        }
    )

    transform_task = PythonOperator(
        task_id='transform',
        python_callable=transform,
        op_kwargs={
            'file_path': '/opt/airflow/data/fintech_clean.csv'
        }
    )

    load_to_db_task = PythonOperator(
        task_id='load_to_db',
        python_callable=load_to_db,
        op_kwargs={
            'file_path': '/opt/airflow/data/fintech_transformed.csv'
        }
    )

    run_dashboard_task = PythonOperator(
        task_id='run_dashboard',
        python_callable=create_dashboard,
    )

    # Define the task dependencies
    extract_clean_task >> transform_task >> load_to_db_task >> run_dashboard_task
