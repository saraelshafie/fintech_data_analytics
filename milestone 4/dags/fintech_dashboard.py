import dash
from dash import dcc, html, Input, Output
import numpy as np
import pandas as pd
import plotly.express as px

def create_dashboard():
    # Load your cleaned dataset
    # df = pd.read_parquet('/opt/airflow/data/fintech_data_MET_P2_52_0812_clean.parquet')
    df = pd.read_csv('/opt/airflow/data/fintech_transformed.csv')
    df['issue_date'] = pd.to_datetime(df['issue_date']) 
    df['year'] = df['issue_date'].dt.year  # Extract year

    # Normalize grade counts for percentage distribution
    grade_distribution = df['letter_grade'].value_counts(normalize=True).reset_index()
    grade_distribution.columns = ['letter_grade', 'percentage']
    grade_distribution['percentage'] *= 100  # Convert to percentage

    # Initialize the app
    app = dash.Dash(__name__)

    # Layout
    app.layout = html.Div([
        html.H1("FinTech Dashboard"),
        html.H4("By Sara Elshafie - ID: 52-0812"),
        dcc.Tabs([
            dcc.Tab(label="Loan Distribution by Grade", children=[
                html.H3(" What is the distribution of loan amounts across different grades?"),
                dcc.Graph(
                    id="loan-grade-plot",
                    figure=px.box(df, x="letter_grade", y="loan_amount", title="Loan Distribution by Grade")
                )
            ]),
            dcc.Tab(label="Loan vs Income by State", children=[
                html.H3("How does the loan amount relate to annual income across states?"),
                dcc.Dropdown(
                    id="state-filter",
                    options=[{"label": state, "value": state} for state in df['state'].unique()] + [{"label": "All", "value": "all"}],
                    value="all"
                ),
                dcc.Graph(id="loan-income-scatter")
            ]),
    
            dcc.Tab(label="Trend of Loan Issuance", children=[
                html.H3("What is the trend of loan issuance over the months?"),
                dcc.Dropdown(
                    id="year-filter",
                    options=[
                        {"label": "All", "value": "all"}  # Default option
                    ] + [{"label": str(year), "value": year} for year in sorted(df['year'].unique())],
                    value="all",  # Default selection
                    placeholder="Select a year"
                ),
                dcc.Graph(id="loan-trend-line")
            ]),
            dcc.Tab(label="State Loan Averages", children=[
                html.H3("Which states have the highest average loan amount?"),
                dcc.RadioItems(
                    id="map-or-bar",
                    options=[
                        {'label': 'Bar Chart', 'value': 'bar'},
                        {'label': 'Choropleth Map', 'value': 'map'}
                    ],
                    value='bar',
                    inline=True
                ),
                dcc.Graph(id="state-loan-chart")
            ]),
            dcc.Tab(label="Loan Grade Distribution", children=[
                html.H3("What is the percentage distribution of loan grades in the dataset?"),
                dcc.Graph(
                    id="loan-grade-distribution",
                    figure=px.bar(
                        grade_distribution,
                        x="letter_grade",
                        y="percentage",
                        title="Percentage Distribution of Loan Grades",
                        labels={"letter_grade": "Loan Grade", "percentage": "Percentage (%)"},
                        text="percentage"
                    ).update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                    .update_layout(yaxis=dict(title="Percentage (%)"))
                )
            ]),
        ])
    ])

    @app.callback(
        Output("loan-income-scatter", "figure"),
        Input("state-filter", "value")
    )
    def update_scatter(selected_state):
        filtered_df = df if selected_state == "all" else df[df["state"] == selected_state] 
        filtered_df["annual_inc_original"] = np.exp(filtered_df["annual_inc"])

        return px.scatter(
            filtered_df, x="annual_inc_original", y="loan_amount", color="loan_status",
            title=f"Loan Amount vs Annual Income ({selected_state})"
        )
    
    @app.callback(
        Output("loan-trend-line", "figure"),
        Input("year-filter", "value")
    )
    def update_loan_trend(selected_year):
        # Filter data based on selected year
        if selected_year != "all":
            filtered_df = df[df['year'] == selected_year]
        else:
            filtered_df = df

        # Group by month and aggregate counts of loans
        trend_data = filtered_df.groupby('month_number').agg(
            loan_count=('loan_amount', 'count'),
            total_loan_amount=('loan_amount', 'sum')
        ).reset_index()

        # Create a line graph
        fig = px.line(
            trend_data, x="month_number", y="loan_count",
            title="Trend of Loan Issuance Over Months",
            labels={"month_number": "Month", "loan_count": "Number of Loans"}
        )
        fig.update_layout(xaxis=dict(tickmode='linear')) 
        return fig
    
    @app.callback(
        Output("state-loan-chart", "figure"),
        Input("map-or-bar", "value")
    )
    def update_state_chart(chart_type):
        # Group by state and calculate average loan amount
        state_data = df.groupby('state', as_index=False).agg(
            avg_loan_amount=('loan_amount', 'mean'),
            state_code=('state', 'first') 
        )

        if chart_type == 'bar':
            # Create a bar chart
            fig = px.bar(
                state_data,
                x="state",
                y="avg_loan_amount",
                title="Average Loan Amount by State",
                labels={"state_name": "State", "avg_loan_amount": "Average Loan Amount"},
                text="avg_loan_amount"
            )
            fig.update_layout(xaxis={'categoryorder': 'total descending'})  # Sort by loan amount
        else:
            # Create a choropleth map
            fig = px.choropleth(
                state_data,
                locations="state_code",
                locationmode="USA-states",
                color="avg_loan_amount",
                scope="usa",
                title="Average Loan Amount by State",
                labels={"avg_loan_amount": "Average Loan Amount"}
            )

        return fig

    
    # Run the app
    app.run_server(host="0.0.0.0", port=8050)

if __name__ == "__main__":
    create_dashboard()

