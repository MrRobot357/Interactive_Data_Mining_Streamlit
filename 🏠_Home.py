import streamlit as st
import pandas as pd
import pydeck as pdk
import random
from itertools import cycle
import time
from DataProcessing import covid_data_processing, load_geodata_110

# Konfiguration der Streamlit-Seite mit Layout
st.set_page_config(layout='wide')

# Titel der Anwendung setzen
st.title('Covid19 Data Visualization')

# Sidebar Dokumentation
st.sidebar.info('[Datasource Documentation](https://github.com/owid/covid-19-data/tree/master/public/data)')
st.sidebar.info('[Codebook](https://github.com/owid/covid-19-data/blob/master/public/data/owid-covid-codebook.csv)')

# Lade die Daten
data_df, min_date, max_date = covid_data_processing()
gdf = load_geodata_110()

# Layout für die Benutzeroberfläche erstellen
row1_col1 = st.columns([1])[0]
row2_col1 = st.columns([1])[0]
row3_col1, row3_col2 = st.columns([1, 1])

# Datum in Jahr und Monat konvertieren
data_df['date'] = pd.to_datetime(data_df['date'])
data_df['month'] = data_df['date'].dt.month
data_df['year'] = data_df['date'].dt.year
years_months_values = [(y, m) for y in data_df['year'].unique() for m in range(1, 13)]

# Platzhalter für die Datumsausgabe und Slider
with row1_col1:
    date_display = st.empty()
with row2_col1:
    month_slider = st.empty()

# Animationseinstellungen
animations = {'None': None, 'Slow': 0.4, 'Medium': 0.2, 'Fast': 0.05}
with row3_col1:
    animate = st.radio('Animation speed:', options=list(animations.keys()), index=2)
animation_speed = animations[animate]  # Geschwindigkeit für Animation
deck_map = st.empty()

# Auswahlmenü für die Datenspalte
data_columns = [col for col in data_df.columns if
                'smoothed' in col and 'per_thousand' not in col and 'per_hundred' not in col]

with row3_col2:
    selected_data_column = st.selectbox('Feature:', options=data_columns)
    elevation_scale = st.slider('Elevation scale:', min_value=0.1, max_value=5000.0, value=100.0, step=1.0)

# Reduzieren des DataFrames auf benötigte Spalten
data_df = data_df[['date', 'country', 'month', 'year', selected_data_column]]

# Berechnung des durchschnittlichen Wertes pro Monat
data_df = data_df.groupby(['year', 'month', 'country'])[[selected_data_column]].mean().reset_index()


# Slider für Jahr und Monat der die Werte zurückgibt
def render_slider(year, month):
    key = random.random() if animation_speed else None
    month_index = years_months_values.index((year, month))

    month_value = month_slider.slider('Progress:', min_value=0, max_value=len(years_months_values) - 1,
                                      value=month_index, format='', key=key)
    updated_year, updated_month = years_months_values[month_value]
    date_display.subheader(f'Month: {updated_year}-{updated_month:02d}')
    return updated_year, updated_month


# Visualisierung der Karte für das gegebene Jahr und den Monat
def render_map(year, month):
    mask = (data_df['year'] == year) & (data_df['month'] == month)
    month_data = data_df[mask]

    # Zusammenführen mit GeoDataFrame und Verwenden von label_x und label_y für Koordinaten
    merged = gdf.merge(month_data, left_on='ADMIN', right_on='country', how='left')
    merged = merged.rename(columns={'LABEL_X': 'lon', 'LABEL_Y': 'lat'})
    merged['lat'] = merged['lat'] + 2

    # Daten für Anzeige vorbereiten
    display_data = merged[['lon', 'lat', selected_data_column]].dropna()

    if display_data.empty:
        return  # Bei fehlenden Daten abbrechen

    # Mapping der Farben für die Darstellung
    max_val = display_data[selected_data_column].max()
    display_data['color'] = display_data[selected_data_column].apply(
        lambda x: [int(255 * (x / max_val)), int(255 * (1 - (x / max_val))), 0] if max_val > 0 else [120, 120, 120]
    )

    # Visualisierung mit pydeck
    deck_map.pydeck_chart(
        pdk.Deck(
            map_style='dark',
            initial_view_state=pdk.ViewState(
                latitude=display_data['lat'].mean(),
                longitude=display_data['lon'].mean(),
                zoom=0.8,
                pitch=50,
            ),
            layers=[
                pdk.Layer(
                    'ColumnLayer',
                    data=display_data,
                    disk_resolution=12,
                    radius=130000,
                    elevation_scale=elevation_scale,
                    get_position='[lon, lat]',
                    get_color='color',
                    get_elevation=f'[{selected_data_column}]',
                ),
            ],
        )
    )


# Hauptlogik für Animation oder statische Ansicht
if animation_speed:
    for year, month in cycle(years_months_values):
        time.sleep(animation_speed)
        render_slider(year, month)
        render_map(year, month)
else:
    year, month = render_slider(years_months_values[0][0], years_months_values[0][1])
    render_map(year, month)
