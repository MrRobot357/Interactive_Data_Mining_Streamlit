import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from DataProcessing import covid_data_processing

# Konfiguration der Streamlit-Seite mit Layout
st.set_page_config(layout='wide')

# Titel der Anwendung
st.title('K-Means-Clustering')

# Sidebar-Dokumentation
st.sidebar.info('[Datasource Documentation](https://github.com/owid/covid-19-data/tree/master/public/data)')
st.sidebar.info('[Codebook](https://github.com/owid/covid-19-data/blob/master/public/data/owid-covid-codebook.csv)')

# Spalten für das Layout definieren
row1_col1, row1_col2 = st.columns([0.5, 1])
row2_col1, row2_col2, row2_col3 = st.columns([1, 0.5, 1])
row3_col1, row3_col2 = st.columns([1, 1])

# Daten laden und vorbereiten
data_df, min_date, max_date = covid_data_processing()

# Session-State initialisieren
if 'selected_data_c' not in st.session_state:
    st.session_state.selected_data_c = ['new_cases_smoothed',
                                        'new_vaccinations_smoothed',
                                        'new_deaths_smoothed']
if 'num_clusters' not in st.session_state:
    st.session_state.num_clusters = 2
if 'selected_dates' not in st.session_state:
    st.session_state.selected_dates = (min_date, max_date)
if 'selected_location_c' not in st.session_state:
    st.session_state.selected_location_c = ['Asia', 'Europe', 'Africa', 'Oceania',
                                            'North America', 'South America']
if 'scaling_method' not in st.session_state:
    st.session_state.scaling_method = 'No Scaling'
if 'aggregation' not in st.session_state:
    st.session_state.aggregation = 'Mean Value'


@st.cache_data
def get_features(filtered_df1):
    column_names = filtered_df1.columns.tolist()
    columns_excluded = ['code', 'continent', 'country', 'date', 'year', 'month']
    return [col for col in column_names if col not in columns_excluded and ('new' not in col or 'smoothed' in col)]


# Spaltennamen abrufen
features_menu = get_features(data_df)


# Callback-Funktionen für die Aktualisierung des Sitzungsstatus
def update_selected_data_c():
    st.session_state.selected_data_c = st.session_state.select_data_c


def update_num_clusters():
    st.session_state.num_clusters = st.session_state.cluster_selection


def update_selected_dates():
    st.session_state.selected_dates = st.session_state.date_selection


def update_scaling_method():
    st.session_state.scaling_method = st.session_state.scaler_selection


def update_selected_location_c():
    st.session_state.selected_location_c = st.session_state.location_selection_c


def update_aggregation():
    st.session_state.aggregation = st.session_state.agg_selection


# Auswahl aller Länder
def all_countries():
    all_countries = data_df[
        (data_df['code'].str.contains('OWID') == False) &
        (data_df['country'].str.contains('income') == False) &
        (data_df['country'] != 'World')
        ]['country'].unique()
    st.session_state.selected_location_c = list(all_countries)


# Auswahl aller Länder eines bestimmten Kontinents
def all_countries_continent(conti):
    countries_options = data_df[
        (data_df['continent'] == conti) &
        (data_df['code'].str.contains('OWID') == False) &
        (data_df['country'].str.contains('income') == False)
        ]['country'].unique()
    st.session_state.selected_location_c = list(countries_options)


with row1_col1:
    st.multiselect(
        'Feature(s):',
        features_menu,
        default=st.session_state.selected_data_c,
        key='select_data_c',
        on_change=update_selected_data_c,
        max_selections=3
    )

with row1_col2:
    selected_min_date, selected_max_date = st.slider(
        'Time period:',
        value=st.session_state.selected_dates,
        min_value=min_date,
        max_value=max_date,
        format='YYYY-MM-DD',
        key='date_selection',
        on_change=update_selected_dates
    )

with row2_col1:
    st.slider(
        'Number of Cluster:',
        min_value=2,
        max_value=10,
        value=st.session_state.num_clusters,
        key='cluster_selection',
        on_change=update_num_clusters
    )
    st.multiselect(
        'Regions:',
        options=data_df['country'].unique(),
        default=st.session_state.selected_location_c,
        key='location_selection_c',
        on_change=update_selected_location_c
    )

with row2_col2:
    aggregation = st.selectbox(
        'Aggregation method:',
        options=['Mean Value', 'Max Value'],
        index=0,
        key='agg_selection',
        on_change=update_aggregation
    )

    scaling = st.selectbox(
        'Scaling Method:',
        options=['No Scaling', 'StandardScaler', 'MinMaxScaler', 'RobustScaler'],
        index=0,
        key='scaler_selection',
        on_change=update_scaling_method
    )
    # Erklärung des Plots zum Ausklappen
    with st.expander('Plot Explanation:'):
        st.markdown('The point where the WCSS curve starts to flatten, indicates the optimal number of clusters.')

    st.markdown('Click the same button a second time to update the multiselect-widget.')

    # Button für alle Länder auswählen
    if st.button('All Countries'):
        all_countries()

    # Buttons für Kontinente
    continents = data_df['continent'].unique()
    for continent in continents:
        if continent != 0:
            if st.button(f'{continent}'):
                all_countries_continent(continent)

with row3_col1:
    labels = st.checkbox('Show Labels', value=True)

# Konvertiere die ausgewählten Min- und Max-Daten zu pandas datetime
selected_min_date = pd.to_datetime(selected_min_date)
selected_max_date = pd.to_datetime(selected_max_date)

# Daten filtern
filtered_data = data_df[(data_df['date'] >= selected_min_date) & (data_df['date'] <= selected_max_date)]
if st.session_state.selected_location_c:
    filtered_data = filtered_data[filtered_data['country'].isin(st.session_state.selected_location_c)]
if st.session_state.selected_data_c:
    filtered_data = filtered_data[['date', 'country'] + st.session_state.selected_data_c]


# Funktion zur Berechnung der WCSS
def calculate_wcss():
    warning_f = False
    wcss_f = []
    clustering_data_f = None
    scaled_values_f = None

    # Prüfung ob ein Merkmal gewählt ist
    if not st.session_state.selected_data_c:
        st.warning(
            'Select at least one Feature'
        )
        warning_f = True

    else:
        if st.session_state.aggregation == 'Mean Value':
            clustering_data_f = filtered_data.groupby('country')[st.session_state.selected_data_c].mean().reset_index()
        elif st.session_state.aggregation == 'Max Value':
            clustering_data_f = filtered_data.groupby('country')[st.session_state.selected_data_c].max().reset_index()

        if st.session_state.scaling_method == 'No Scaling':
            scaled_values_f = clustering_data_f[st.session_state.selected_data_c]
        elif st.session_state.scaling_method == 'StandardScaler':
            scaled_values_f = StandardScaler().fit_transform(clustering_data_f[st.session_state.selected_data_c])
        elif st.session_state.scaling_method == 'MinMaxScaler':
            scaled_values_f = MinMaxScaler().fit_transform(clustering_data_f[st.session_state.selected_data_c])
        elif st.session_state.scaling_method == 'RobustScaler':
            scaled_values_f = RobustScaler().fit_transform(clustering_data_f[st.session_state.selected_data_c])

        clustering_data_f[st.session_state.selected_data_c] = scaled_values_f

        # Prüfung, ob die Anzahl der gewählten Regionen kleiner als die Cluster-Anzahl ist
        if scaled_values_f.shape[0] < st.session_state.num_clusters:
            st.warning(
                'Number of Regions is smaller than the Number of Clusters. '
                'Select more Regions or reduce the number of Clusters.'
            )
            warning_f = True

        # WCSS berechnen
        if not warning_f:
            cluster_range = range(1, min(scaled_values_f.shape[0] + 1, 11))
            for k in cluster_range:
                kmns = KMeans(n_clusters=k, random_state=42)
                kmns.fit(scaled_values_f)
                wcss_f.append(kmns.inertia_)

    return wcss_f, clustering_data_f, scaled_values_f, warning_f


# Variable für das Fehlerhandling
warning = False

# Aufrufen der Funktion
wcss, clustering_data, scaled_values, warning = calculate_wcss()

if not warning:

    # K-Means Clustering auf den skalierten Daten
    kmeans = KMeans(n_clusters=st.session_state.num_clusters, random_state=42)
    clustering_labels = kmeans.fit_predict(scaled_values)
    clustering_data['Cluster'] = clustering_labels

    # Farbpalette für gut unterscheidbare Farben
    custom_colors = [
        '#FF0000',  # Rot
        '#008000',  # Grün
        '#0000FF',  # Blau
        '#FFD700',  # Gelb
        '#FF1493',  # Pink
        '#00FFFF',  # Cyan
        '#FFA500',  # Orange
        '#800080',  # Violett
        '#808080',  # Grau
        '#000000'  # Schwarz
    ]

    # Visualisierung der Cluster
    fig = go.Figure()
    if len(st.session_state.selected_data_c) == 1:
        for cluster_num in range(st.session_state.num_clusters):
            cluster_data = clustering_data[clustering_data['Cluster'] == cluster_num]

            fig.add_trace(go.Scatter(
                x=cluster_data[st.session_state.selected_data_c[0]],
                y=[0] * len(cluster_data),
                mode='markers+text' if labels else 'markers',
                marker=dict(color=custom_colors[cluster_num], size=10),
                text=cluster_data['country'],
                name=f'Cluster {cluster_num + 1}',
                hovertemplate='<b>%{text}</b><br>' +
                              f'{st.session_state.selected_data_c[0]} ' + '%{x}<br>' +
                              f'Cluster: {cluster_num + 1}<extra></extra>'
            ))

        fig.update_layout(
            title='Clusters of regions based on the selected features',
            xaxis_title=f'{st.session_state.selected_data_c[0]}',
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            height=1000
        )
    elif len(st.session_state.selected_data_c) == 2:
        for cluster_num in range(st.session_state.num_clusters):
            cluster_data = clustering_data[clustering_data['Cluster'] == cluster_num]

            fig.add_trace(go.Scatter(
                x=cluster_data[st.session_state.selected_data_c[0]],
                y=cluster_data[st.session_state.selected_data_c[1]],
                mode='markers+text' if labels else 'markers',
                marker=dict(color=custom_colors[cluster_num], size=10),
                text=cluster_data['country'],
                name=f'Cluster {cluster_num + 1}',
                hovertemplate='<b>%{text}</b><br>' +
                              f'{st.session_state.selected_data_c[0]} ' + '%{x}<br>' +
                              f'{st.session_state.selected_data_c[1]} ' + '%{y}<br>' +
                              f'Cluster: {cluster_num + 1}<extra></extra>'
            ))

        fig.update_layout(
            title='Clusters of regions based on the selected features',
            xaxis_title=f'{st.session_state.selected_data_c[0]}',
            yaxis_title=f'{st.session_state.selected_data_c[1]}',
            height=1000
        )

    elif len(st.session_state.selected_data_c) == 3:
        for cluster_num in range(st.session_state.num_clusters):
            cluster_data = clustering_data[clustering_data['Cluster'] == cluster_num]

            fig.add_trace(go.Scatter3d(
                x=cluster_data[st.session_state.selected_data_c[0]],
                y=cluster_data[st.session_state.selected_data_c[1]],
                z=cluster_data[st.session_state.selected_data_c[2]],
                mode='markers+text' if labels else 'markers',
                marker=dict(color=custom_colors[cluster_num], size=5),
                text=cluster_data['country'],
                name=f'Cluster {cluster_num + 1}',
                hovertemplate='<b>%{text}</b><br>' +
                              f'{st.session_state.selected_data_c[0]} ' + '%{x}<br>' +
                              f'{st.session_state.selected_data_c[1]} ' + '%{y}<br>' +
                              f'{st.session_state.selected_data_c[2]} ' + '%{z}<br>' +
                              f'Cluster: {cluster_num + 1}<extra></extra>'
            ))

        fig.update_layout(
            title='Clusters of regions based on the selected features',
            height=1000,
            scene_dragmode='turntable',
            scene=dict(
                xaxis_title=f'{st.session_state.selected_data_c[0]}',
                yaxis_title=f'{st.session_state.selected_data_c[1]}',
                zaxis_title=f'{st.session_state.selected_data_c[2]}'
            ),
        )

    fig.update_traces(textposition='top center')
    fig.update_layout(hovermode='closest', dragmode='zoom')
    st.plotly_chart(fig)

    # Elbow-Plot anzeigen
    with row2_col3:
        fig_num_cluster = go.Figure()

        # WCSS-Daten hinzufügen
        fig_num_cluster.add_trace(go.Scatter(
            x=list(range(1, len(wcss) + 1)),
            y=wcss,
            mode='lines+markers',
            marker=dict(size=8),
            line=dict(width=2),
        ))

        # Layout und Beschriftungen des Elbow-Plots
        fig_num_cluster.update_layout(
            title='Elbow-Plot to determine the optimal number of clusters',
            xaxis_title='Number of Clusters',
            yaxis_title='Within-Cluster-Sums-of-Squares (WCSS)',
        )

        # Plotly-Chart in Streamlit anzeigen
        st.plotly_chart(fig_num_cluster)
