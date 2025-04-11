FROM apache/airflow:2.10.3
EXPOSE 8050
USER airflow 
COPY dags /opt/airflow/dags
COPY requirements.txt /
RUN pip install -r /requirements.txt
