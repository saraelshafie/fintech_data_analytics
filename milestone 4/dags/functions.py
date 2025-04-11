import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import requests
from bs4 import BeautifulSoup
import os
import pandas as pd
from sqlalchemy import create_engine

def clean_column_names(df):
    df.columns = df.columns.str.lower().str.replace(' ', '_')

def set_df_index(df, index_col):
    df = df.set_index(index_col, inplace=True)
    return df

def print_unique_values(fintech_df):
    # Loop through non-numeric columns and print the unique values for each
    for column in fintech_df.select_dtypes(exclude=['float64']).columns:
        unique_values = fintech_df[column].unique()
        print(f"Unique values in '{column}':")
        print(unique_values)
        print("\n")

def normalize_type_field(df):
    df['type'] = df['type'].replace({
        'Individual': 'Individual',
        'INDIVIDUAL': 'Individual',
        'Joint App': 'Joint',
        'JOINT': 'Joint',
        'DIRECT_PAY': 'Direct Pay'
    })

def clean_emp_length(df):
    emp_length_mapping = {
        '10+ years': 10,
        '< 1 year': 0.5,
        '1 year': 1,
        '2 years': 2,
        '3 years': 3,
        '4 years': 4,
        '5 years': 5,
        '6 years': 6,
        '7 years': 7,
        '8 years': 8,
        '9 years': 9,
        'n/a': None
    }
    df['emp_length'] = df['emp_length'].replace(emp_length_mapping)

def add_lookup_values(lookup_df, column_name, original_column, encoded_column):
    unique_values = original_column.unique()
    unique_encoded_values = encoded_column.unique()
    new_rows = pd.DataFrame({
        'column': column_name,
        'original': unique_values,
        'imputed': unique_encoded_values,
    })
    lookup_df = pd.concat([lookup_df, new_rows], ignore_index=True)
    return lookup_df

# Generic univariate imputation function
def univariate_imputation(df, column, fill_value):
    df[column].fillna(fill_value, inplace=True)

# Generic multivariate imputation function
def multivariate_imputation(df, column_to_impute, group_by_column, method='mode'):
    if method == 'mode':
        df[column_to_impute] = df.groupby(group_by_column)[column_to_impute].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Unknown')
        )
    elif method == 'median':
        global_median = df[column_to_impute].median()  # Calculate global median
        df[column_to_impute] = df.groupby(group_by_column)[column_to_impute].transform(
            lambda x: x.fillna(x.median() if not x.dropna().empty else global_median)
        )

def null_values_sum(df,column):
    return df[column].isnull().sum()

def update_lookup_df(lookup_df, column_name, original_value, imputed_value):
    lookup_df = pd.concat([lookup_df, pd.DataFrame([{'column': column_name, 'original': original_value, 'imputed': imputed_value}])], ignore_index=True)
    return lookup_df

def impute_emp_fields(df):
    # Define salary ranges without creating a new column
    salary_bins = [0, 25000, 50000, 75000, 100000, 150000, 200000, 300000, np.float64(9550000.0)]
    salary_labels = ['<25k', '25k-50k', '50k-75k', '75k-100k', '100k-150k', '150k-200k', '200k-300k', '>300k']
    salary_groups = pd.cut(df['annual_inc'], bins=salary_bins, labels=salary_labels)

    # Impute 'emp_title' using mode within salary ranges
    multivariate_imputation(df, 'emp_title', salary_groups, method='mode')

    # Impute 'emp_length' using median within salary ranges
    multivariate_imputation(df, 'emp_length', salary_groups, method='median')

def detect_outliers(df, col, method='Z-Score', threshold=3):
    
    if method == 'Z-Score':
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        df['z_score'] = z_scores
        z_outliers_mask = df['z_score'] > threshold
        df.drop(columns='z_score', inplace=True)
        outliers = df[z_outliers_mask] 
    elif method == 'IQR':
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        print(f'Outliers below: {Q1 - 1.5 * IQR:.3f}')
        print(f'Outliers above: {Q3 + 1.5 * IQR:.3f}')
        
        iqr_outliers_mask = (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
        outliers = df[iqr_outliers_mask]

    print(f'Percentage of Outliers: {len(outliers)/len(df)*100:.3f}%')

        
    return outliers

def get_log_transformation(df, column):
    return np.log(df[column])

def apply_transformation(df, col, transformed_col, ignore_zero=False):
    if ignore_zero:
        df.loc[df[col] != 0, col] = transformed_col
    else:
        df[col] = transformed_col
    return df

def cap_outliers(df, column):
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    
    # Calculate IQR (Interquartile Range)
    IQR = Q3 - Q1
    
    # Calculate the lower and upper bounds for capping
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print(f'Lower Bound: {lower_bound:.3f}')
    print(f'Upper Bound: {upper_bound:.3f}')
    
    cap_column = np.where(df[column] < lower_bound, lower_bound, 
                          np.where(df[column] > upper_bound, upper_bound, df[column]))
    
    return cap_column

def add_month_number(df, date_column):
    df[date_column] = pd.to_datetime(df[date_column])
    df['month_number'] = df[date_column].dt.month
    return df

def add_salary_can_cover(df, log_annual_income_column, loan_amount_column):
    # Reverse the log transformation of annual income
    df['salary_can_cover'] = (np.exp(df[log_annual_income_column]) >= df[loan_amount_column]).astype(int)
    return df

def map_grade(grade):
        if 1 <= grade <= 5:
            return 'A'
        elif 6 <= grade <= 10:
            return 'B'
        elif 11 <= grade <= 15:
            return 'C'
        elif 16 <= grade <= 20:
            return 'D'
        elif 21 <= grade <= 25:
            return 'E'
        elif 26 <= grade <= 30:
            return 'F'
        elif 31 <= grade <= 35:
            return 'G'
        else:
            return 'Unknown'  # In case there are grades outside the expected range

def update_lookup_with_grades(lookup_df):    
    
    for i in range(1, 36):
        letter = map_grade(i)
        lookup_df = pd.concat([lookup_df, pd.DataFrame([{'column': 'grade', 'original': str(i), 'imputed': letter}])], ignore_index=True)
    return lookup_df

def add_letter_grade(df, grade_column):
    df['letter_grade'] = df[grade_column].apply(map_grade)
    return df

def calculate_monthly_installment(df, loan_amount_column, log_int_rate_column, term_column):
    df_copy = df.copy()
    # Convert term to months (e.g., '36 months' -> 36)
    df_copy[term_column] = df_copy[term_column].str.extract('(\d+)').astype(int)
    
    # Calculate monthly installment directly in the apply function without adding intermediary columns
    df['installment_per_month'] = df_copy.apply(
        lambda row: (row[loan_amount_column] * (np.exp(row[log_int_rate_column]) / 12) * (1 + (np.exp(row[log_int_rate_column]) / 12)) ** row[term_column]) / 
                    ((1 + (np.exp(row[log_int_rate_column]) / 12)) ** row[term_column] - 1)
        if np.exp(row[log_int_rate_column]) > 0 else row[loan_amount_column] / row[term_column], axis=1
    )
    
    return df


def label_encode_column(df, column, new_column):
    le = LabelEncoder()
    df[new_column] = le.fit_transform(df[column])
    return df

def one_hot_encode_columns(df, columns):
    for column in columns:
        one_hot_encoded = pd.get_dummies(df[column], prefix=column)
        # Convert boolean values to integers (0 and 1)
        one_hot_encoded = one_hot_encoded.astype(int)
        # Concatenate the one-hot encoded columns to the original dataframe
        df = pd.concat([df, one_hot_encoded], axis=1)
        
        # df.drop(columns=column, inplace=True)
    return df

def label_encode_loan_status(df):
    mapping = {'Fully Paid':1,'Current': 2, 'In Grace Period': 3, 'Late (16-30 days)': 4, 'Late (31-120 days)': 5, 'Default': 6, 'Charged Off': 7}
    df['loan_status_encoded'] = df['loan_status'].map(mapping)

    return df

def label_encode_verification_status(df):
    mapping = {'Not Verified': 1, 'Verified': 2, 'Source Verified': 3}
    df['verification_status_encoded'] = df['verification_status'].map(mapping)    
    return df

def fetch_and_map_state_names(df, state_column):
  url = "https://www23.statcan.gc.ca/imdb/p3VD.pl?Function=getVD&TVD=53971"
  response = requests.get(url)
  soup = BeautifulSoup(response.text, 'html.parser')
  table = soup.find('table')

  # <AlphaCode, StateName>
  state_dict = {}

  for row in table.find_all('tr')[1:]:
    columns = row.find_all('td')
    
    if len(columns) >= 2:  
      alpha_code = columns[2].text.strip()
      state_name = columns[0].text.strip()  
      
      state_dict[alpha_code] = state_name

  df['state_name'] = df[state_column].map(state_dict)
  return df


def clean(fintech_df):
    clean_column_names(fintech_df)
    set_df_index(fintech_df, 'customer_id')
    normalize_type_field(fintech_df)
    clean_emp_length(fintech_df)

    impute_emp_fields(fintech_df)
    mean = fintech_df['int_rate'].mean()
    univariate_imputation(fintech_df, 'int_rate', mean)

    univariate_imputation(fintech_df, 'description', 'No Description')
    univariate_imputation(fintech_df, 'annual_inc_joint', 0)

    return fintech_df

def transformations(fintech_df):
    log_annual_inc = get_log_transformation(fintech_df, 'annual_inc')
    fintech_df = apply_transformation(fintech_df, 'annual_inc', log_annual_inc)

    filtered_df = fintech_df[fintech_df['annual_inc_joint'] != 0]
    log_annual_inc_joint = get_log_transformation(filtered_df, 'annual_inc_joint')
    fintech_df = apply_transformation(fintech_df, 'annual_inc_joint', log_annual_inc_joint, ignore_zero=True)

    log_avg_cur_bal = get_log_transformation(fintech_df, 'avg_cur_bal')
    fintech_df = apply_transformation(fintech_df, 'avg_cur_bal', log_avg_cur_bal)

    log_tot_cur_bal = get_log_transformation(fintech_df, 'tot_cur_bal')
    fintech_df = apply_transformation(fintech_df, 'tot_cur_bal', log_tot_cur_bal)

    cap_loan_amount = cap_outliers(fintech_df, 'loan_amount')
    fintech_df = apply_transformation(fintech_df, 'loan_amount', cap_loan_amount)

    cap_funded_amount = cap_outliers(fintech_df, 'funded_amount')
    fintech_df = apply_transformation(fintech_df, 'funded_amount', cap_funded_amount)

    log_int_rate = get_log_transformation(fintech_df, 'int_rate')
    fintech_df = apply_transformation(fintech_df, 'int_rate', log_int_rate)

    fintech_df = add_month_number(fintech_df, 'issue_date')
    fintech_df = add_salary_can_cover(fintech_df, 'annual_inc', 'loan_amount')
    fintech_df = add_letter_grade(fintech_df, 'grade')
    fintech_df = calculate_monthly_installment(fintech_df, 'loan_amount', 'int_rate', 'term')

    fintech_df_encoded = fintech_df.copy()
    fintech_df_encoded = label_encode_column(fintech_df_encoded, 'letter_grade','letter_grade_encoded')
    fintech_df_encoded = label_encode_column(fintech_df_encoded, 'addr_state', 'addr_state_encoded')
    fintech_df_encoded = label_encode_column(fintech_df_encoded, 'state','state_encoded')
    fintech_df_encoded = label_encode_column(fintech_df_encoded, 'purpose','purpose_encoded')

    fintech_df_encoded = label_encode_loan_status(fintech_df_encoded)
    fintech_df_encoded = label_encode_verification_status(fintech_df_encoded)
    fintech_df_encoded = one_hot_encode_columns(fintech_df_encoded, ['home_ownership', 'term', 'type'])

    fintech_df_encoded = fetch_and_map_state_names(fintech_df_encoded, 'state')

    return fintech_df_encoded


def extract_clean(file_path):
    fintech_df = pd.read_csv(file_path)
    
    fintech_df_cleaned = clean(fintech_df)
    
    # Save the cleaned dataframe as a CSV file
    fintech_df_cleaned.to_csv('/opt/airflow/data/fintech_clean.csv',index=False)
    print('loaded after cleaning succesfully')
    
def transform(file_path):
    fintech_df_cleaned = pd.read_csv(file_path)
    
    fintech_df_transformed = transformations(fintech_df_cleaned)
    
    try:
        # Save the transformed dataframe as a CSV file
        fintech_df_transformed.to_csv('/opt/airflow/data/fintech_transformed.csv', index=False, mode= 'x')
        print('loaded after transformation successfully')
    except FileExistsError:
        print('file already exists')

def load_to_db(file_path):
    DATABASE_URL = "postgresql://root:root@pgdatabase:5432/fintech_db"
    engine = create_engine(DATABASE_URL)

    if engine.connect():
        print('Connected to Database')
        try:
            df = pd.read_csv(file_path)
            print('Writing transformed dataset to database')
            df.to_sql('fintech', con=engine, if_exists='replace', index=False)
            print('Done writing to database')
        except ValueError as vx:
            print('Transformed Table already exists.')
        except Exception as ex:
            print(ex)
    else:
        print('Failed to connect to Database')