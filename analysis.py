import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calendar

def process_pollutant(df, pollutant_column, plot_title):
    # Convert 'NaN' values to the mean of the column
    df[pollutant_column] = pd.to_numeric(df[pollutant_column], errors='coerce')
    df[pollutant_column].fillna(df[pollutant_column].mean(), inplace=True)

    # Convert 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Extract year and pollutant levels
    df['year'] = df['date'].dt.year

    # Calculate the mean pollutant levels for each year
    yearly_means = df.groupby('year')[pollutant_column].mean()

    # Plotting the bar chart
    plt.figure(figsize=(12, 6))
    yearly_means.plot(kind='bar', color=sns.color_palette('husl'))

    plt.title(f'Year-wise Mean {plot_title} Levels')
    plt.xlabel('Year')
    plt.ylabel(f'Mean {plot_title} Levels')

    return plt


def monthly_analysis(df, pollutant_column):
    # Convert 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Convert the pollutant column to numeric
    df[pollutant_column] = pd.to_numeric(df[pollutant_column], errors='coerce')

    # Check for non-numeric values
    non_numeric_values = df[pollutant_column][df[pollutant_column].isnull()]
    if not non_numeric_values.empty:
        print(f"Non-numeric values found in {pollutant_column}: {non_numeric_values}")

    # Calculate the mean pollutant levels for each month
    monthly_means = df.groupby([df['date'].dt.year, df['date'].dt.month])[pollutant_column].mean()

    # Plotting the bar chart for monthly analysis
    plt.figure(figsize=(12, 6))
    monthly_means.unstack().plot(kind='bar', title=f'Monthly {pollutant_column.upper()}')
    plt.xlabel('Year-Month')
    plt.ylabel(pollutant_column.upper())
    plt.tight_layout()
    plt.show()
