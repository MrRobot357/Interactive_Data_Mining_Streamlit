from datetime import datetime
import streamlit as st
import pandas as pd
import datetime as dt
import geopandas as gpd

pd.set_option('future.no_silent_downcasting', True)


# Diese Funktion lädt und verarbeitet die COVID-Daten und speichert sie zwischen
@st.cache_resource
def covid_data_processing():
    # CSV-Datei als Datenquelle einlesen
    csv_file = 'https://catalog.ourworldindata.org/garden/covid/latest/compact/compact.csv'
    data_df = pd.read_csv(csv_file)  # Laden der CSV-Daten


    # Datumsformat konvertieren und min/max Daten definieren
    data_df['date'] = pd.to_datetime(data_df['date'])
    min_date = datetime(2020, 1, 1)
    max_date = dt.datetime.today()
    all_dates = pd.date_range(start=min_date, end=max_date)

    # Vollständige Daten für jedes Land generieren
    full_date_list = []
    for country, group in data_df.groupby('country'):
        full_date = pd.DataFrame({
            'country': country,
            'date': all_dates
        })
        merged_dates = pd.merge(full_date, group, on=['country', 'date'], how='left')
        full_date_list.append(merged_dates)

    # Alle Merged_dates-DataFrames zu einem Dataframe kombinieren
    data_df = pd.concat(full_date_list, ignore_index=True).sort_values(by=['country', 'date'])


    # Änderungen in der Spalte 'country' vornehmen
    replacements = {
        "United States": "United States of America",
        "Democratic Republic of Congo": "Democratic Republic of the Congo",
        "Congo": "Republic of the Congo",
        "Tanzania": "United Republic of Tanzania",
        "Cote d'Ivoire": "Ivory Coast",
        "Bahamas": "The Bahamas",
        "Serbia": "Republic of Serbia",
    }
    data_df['country'] = data_df['country'].replace(replacements)

    # Liste mit Spaltennamen, die 'total' oder 'mortality' enthalten
    columns_ffill = [column for column in data_df.columns if 'total' in column or 'mortality' in column]

    # Vorwärts Auffüllen der Spalten aus der Liste
    data_df[columns_ffill] = data_df.groupby('country')[columns_ffill].transform(lambda x: x.ffill())

    # Liste mit Spalten die nur einen Wert pro Land enthalten
    single_val_col = ['code', 'continent', 'population','population_density',
                      'median_age', 'life_expectancy', 'gdp_per_capita', 'extreme_poverty',
                      'diabetes_prevalence', 'handwashing_facilities', 'hospital_beds_per_thousand',
                      'human_development_index'
                      ]
    # Vorwärts- und rückwärts Auffüllen aller fehlenden Werte pro Land
    data_df[single_val_col] = data_df.groupby('country')[single_val_col].transform(lambda x: x.ffill().bfill())
    data_df = data_df.fillna(0)

    return data_df, min_date, max_date


# Diese Funktionen laden Geodaten und speichern sie zwischen
@st.cache_data
def load_geodata_50():
    geo_file = 'Data/ne_50m_admin_0_countries.shp'
    return gpd.read_file(geo_file)



@st.cache_data
def load_geodata_110():
    geo_file = 'Data/ne_110m_admin_0_countries.shp'
    return gpd.read_file(geo_file)
