# Fintech Data Engineering and ETL Pipeline

This repository contains the source code, configurations, and documentation for a comprehensive fintech data engineering project focused on preprocessing, analyzing, and visualizing fintech data through robust ETL (Extract-Transform-Load) pipelines and interactive dashboards.

## Project Overview
The goal of this project was to:

- Perform extensive **Exploratory Data Analysis (EDA)** and **data preprocessing** on fintech datasets.
- Construct an automated **ETL pipeline** utilizing modern data engineering tools and technologies (Docker, Kafka, PostgreSQL, PySpark, Airflow).
- Develop an **interactive dashboard** to visualize critical financial data insights for analytical and business decision-making purposes.

## Features and Functionalities
- **Data Cleaning & Preprocessing**
  - Renaming columns, handling duplicates, outliers, and imputing missing values.
  - Generating insightful engineered features, including loan affordability checks and installment calculations.

- **Distributed Data Processing (PySpark)**
  - Efficiently managed large-scale datasets.
  - Encoded categorical features and performed complex feature engineering (historical loan tracking).

- **Real-Time Data Processing (Kafka)**
  - Set up Kafka producers and consumers for real-time fintech data streaming.
  - Integrated streaming data ingestion within the ETL pipeline.

- **Dockerized Pipeline**
  - Packaged application components into a Dockerized environment, ensuring reproducibility and scalability.
  - Managed databases using PostgreSQL with containers orchestrated through Docker Compose.

- **Workflow Automation (Apache Airflow)**
  - Created automated workflows (DAGs) that reliably executed data processing tasks.
  - Structured pipeline into modular tasks for extracting, transforming, loading, and visualizing data.

- **Interactive Visualization Dashboard (Dash)**
  - Interactive dashboard built to visualize loan amount distributions, correlations, issuance trends, and geographic financial insights.
  - Provided interactivity through dynamic filters and real-time visual updates.

## Technologies Used
- Python (pandas, PySpark, kafka-python, Dash, Airflow)
- Apache Kafka (Real-time data streaming)
- PostgreSQL (Data storage and management)
- Docker & Docker Compose (Containerization)
- Apache Airflow (Workflow management and scheduling)
- Dash (Interactive dashboards)

## Dashboard Visualization
The dashboard visualizes insightful metrics such as:
- Loan amount distributions across various grades.
- Relationship between loan amounts and annual incomes per state.
- Monthly loan issuance trends.
- Average loan amounts per state.
- Grade percentage distributions in the loan dataset.

### Prerequisites
- Docker & Docker Compose installed.

