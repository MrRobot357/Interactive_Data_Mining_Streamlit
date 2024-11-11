import numpy as np
import pydeck as pdk
import streamlit as st
import leafmap.colormaps as cm
import plotly.graph_objects as go
from leafmap.common import hex_to_rgb
from DataProcessing import covid_data_processing
from DataProcessing import load_geodata_50

# Konfiguration der Streamlit-Seite mit Layout
st.set_page_config(layout='wide')

# Titel der Anwendung setzen
st.title('Interactive Map')

# Sidebar-Dokumentation hinzufügen
st.sidebar.info('[Datasource Documentation](https://github.com/owid/covid-19-data/tree/master/public/data)')
st.sidebar.info('[Codebook](https://github.com/owid/covid-19-data/blob/master/public/data/owid-covid-codebook.csv)')

# Daten laden und verarbeiten
data_df, min_date, max_date = covid_data_processing()

# Initialisiere den Sitzungsstatus mit Standardwerten
if 'selected_date' not in st.session_state:
    st.session_state.selected_date = (max_date)

if 'selected_col' not in st.session_state:
    st.session_state.selected_col = 'new_cases_smoothed'

if 'palette' not in st.session_state:
    st.session_state.palette = 'hot'

if 'n_colors' not in st.session_state:
    st.session_state.n_colors = 8

if 'elev_scale' not in st.session_state:
    st.session_state.elev_scale = 100.0

# Spalten für Layout erstellen
row1_col1, row1_col2, row1_col3, row1_col4 = st.columns([1.1, 1.5, 1.3, 2])
row2_col1, row2_col2, row2_col3, row2_col4 = st.columns([0.2, 0.2, 0.2, 1])
row3_col1, row3_col2 = st.columns([1, 0.1])

# Callback-Funktionen für die Aktualisierung des Sitzungsstatus
def update_selected_date():
    st.session_state.selected_date = st.session_state.date_selection

def update_selected_col():
    st.session_state.selected_col = st.session_state.col_selection

def update_palette():
    st.session_state.palette = st.session_state.palette_selection

def update_n_colors():
    st.session_state.n_colors = st.session_state.n_colors_selection

def update_elev_scale():
    st.session_state.elev_scale = st.session_state.elev_scale_selection

# Widget zur Auswahl des Datums
with row1_col1:
    st.date_input(
        'Date:',
        min_value=min_date,
        max_value=max_date,
        value=st.session_state.selected_date,
        key='date_selection',
        on_change=update_selected_date
    )

# Filtere die Daten basierend auf dem ausgewählten Datum
filtered_df = data_df[data_df['date'] == str(st.session_state.selected_date)]

# Funktion um die Auswahl an Merkmalen zu definieren
@st.cache_data
def get_features(filtered_df1):
    column_names = filtered_df1.columns.tolist()
    columns_excluded = ['code', 'continent', 'country', 'date', 'year', 'month']
    return [col for col in column_names if col not in columns_excluded and ('new' not in col or 'smoothed' in col)]

# Spaltennamen abrufen
features_menu = get_features(filtered_df)

# Auswahl-Widgets für Attribute, Farbschemata und Farbanzahl
with row1_col2:
    st.selectbox(
        'Feature:',
        features_menu,
        index=features_menu.index(st.session_state.selected_col),
        key='col_selection',
        on_change=update_selected_col
    )

with row1_col3:
    palettes = cm.list_colormaps()
    st.selectbox(
        'Color-Palette:',
        palettes,
        index=palettes.index(st.session_state.palette),
        key='palette_selection',
        on_change=update_palette
    )

with row1_col4:
    st.slider(
        'Number of Colors:',
        min_value=2,
        max_value=10,
        value=st.session_state.n_colors,
        key='n_colors_selection',
        on_change=update_n_colors
    )



# Auswahl für die Art der Farbzuweisung
with row2_col1:
    color_mapping = st.radio('Color-Mapping:', options=('Quantile', 'Equal Interval'))

# Checkbox für 3D-Ansicht
with row2_col2:
    show_3d = st.checkbox('Show 3D', value=True)

# Höhenfaktor-Slider nur anzeigen, wenn 3D-Ansicht aktiviert ist
if show_3d:
    with row2_col3:
        st.number_input(
            'Elevation scale:',
            value=st.session_state.elev_scale,
            step=0.001,
            format='%.3f',
            key='elev_scale_selection',
            on_change=update_elev_scale
        )

# Relevante Spalten für die Visualisierung filtern
columns_to_keep = ['country', st.session_state.selected_col]
filtered_df = filtered_df[columns_to_keep]

# Geodaten laden und zusammenführen
countries_gdf = load_geodata_50()
merged_gdf = countries_gdf.merge(filtered_df, left_on='ADMIN', right_on='country', how='left').fillna(0)

# Erstellen einer Farbpalette
@st.cache_data
def create_palette(palette, n_colors_f):
    colors_palette = cm.get_palette(palette, n_colors_f)
    return colors_palette

# # Funktion Aufrufen um die Farbpalette zu erstellen
colors_hex = create_palette(st.session_state.palette, st.session_state.n_colors)

# Umwandlung in RGB
colors = [hex_to_rgb(c) for c in colors_hex]

# Werte des gewählten Merkmals speichern
cases_column = merged_gdf[st.session_state.selected_col].values

# Maximalwert für das gewählte Merkmal
max_cases_f = merged_gdf[st.session_state.selected_col].max()


# Bestimmung der Intervallgrenzen
def get_borders(selected_col, n_colors_f):
    if color_mapping == 'Quantile':
        borders_q = np.round(np.quantile(merged_gdf[selected_col], np.linspace(0, 1, n_colors_f + 1)), decimals=3)
        return borders_q

    elif color_mapping == 'Equal Interval':
        group_range = max_cases_f / n_colors_f
        borders_e = [i * group_range for i in range(n_colors_f + 1)]
        return borders_e


# Intervallgrenzen aufrufen
borders = get_borders(st.session_state.selected_col, st.session_state.n_colors)


# Farbzuweisung
def assign_colors(borders_f, colors_f, n_colors_f):
    for i, cases_f in enumerate(cases_column):
        for j in range(n_colors_f):
            start_value = borders_f[j]
            end_value = borders_f[j + 1]
            if start_value <= cases_f < end_value:
                merged_gdf.loc[i, ['R', 'G', 'B']] = colors_f[j]
                break
        else:
            if cases_f == max_cases_f:
                merged_gdf.loc[i, ['R', 'G', 'B']] = colors_f[-1]


# Farbzuweisung aufrufen
assign_colors(borders, colors, st.session_state.n_colors)

# Visualisierung mit pydeck
with row3_col1:
    st.pydeck_chart(
        pdk.Deck(
            map_style='dark',
            initial_view_state=pdk.ViewState(
                latitude=51,
                longitude=10,
                zoom=0.4,
                pitch=60,
            ),
            layers=[
                pdk.Layer(
                    'GeoJsonLayer',
                    merged_gdf,
                    pickable=True,
                    opacity=0.75,
                    filled=True,
                    extruded=show_3d,
                    get_elevation=st.session_state.selected_col,
                    elevation_scale=st.session_state.elev_scale,
                    get_fill_color='[R, G, B]',
                )
            ],
            tooltip={
                'html': f'<b>{{ADMIN}}</b><br/><b> </b> {{{st.session_state.selected_col}}}',
                'style': {'backgroundColor': 'steelblue', 'color': 'white'}
            }
        ),
        height=800
    )

# Erstellen der Farblegende
fig_legend = go.Figure()

# Jeder Farbe als Linie
for i, color in enumerate(colors_hex):
    fig_legend.add_trace(go.Scatter(
        x=[0, 0],
        y=[i, i + 1],
        mode='lines',
        line=dict(color=f'#{color}', width=15),
        showlegend=False,
        hoverinfo='skip'
    ))


# Konfiguration der y-Achse mit den exakten Tick-Positionen
fig_legend.update_layout(
    yaxis=dict(
        tickvals=[i for i in range(len(borders))],
        ticktext=[f'{val:.2f}' for val in borders],
        title=st.session_state.selected_col.replace('_', ' ').title(),
        title_standoff=40,
        side='right',
        showgrid=False,
        zeroline=False,
        ticks='outside',
        ticklen=10,
        position=0.55
    ),
    xaxis=dict(visible=False),
    height=800,
)

# Balken in Streamlit anzeigen
with row3_col2:
    st.plotly_chart(fig_legend)
