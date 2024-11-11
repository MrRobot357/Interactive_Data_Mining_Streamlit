import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from DataProcessing import covid_data_processing

st.set_page_config(layout='wide')
st.title('Interactive Diagram')
st.sidebar.info('[Datasource Documentation](https://github.com/owid/covid-19-data/tree/master/public/data)')


data_df, min_date, max_date = covid_data_processing()

# Spalten für die Widgets erstellen
row1_col1, row1_col2 = st.columns([1, 1])

# Session-State prüfen und Standardwerte festlegen
if 'selected_dates' not in st.session_state:
    st.session_state.selected_dates = (min_date, max_date)

if 'selected_data_g' not in st.session_state:
    st.session_state.selected_data_g = ['new_cases_smoothed', 'new_vaccinations_smoothed']

if 'selected_location_g' not in st.session_state:
    st.session_state.selected_location_g = ['World']


# Auswahl-Callback-Funktionen
def update_selected_location_g():
    st.session_state.selected_location_g = st.session_state.location_selection_g


def update_selected_data_g():
    st.session_state.selected_data_g = st.session_state.data_selection_g


def update_selected_dates():
    st.session_state.selected_dates = st.session_state.date_selection


# Liste für die Auswahl der Regionen
unique_locations = data_df['country'].unique()

# Region-Auswahl
with row1_col1:
    st.multiselect(
        'Region(s):',
        unique_locations,
        default=st.session_state.selected_location_g,
        key='location_selection_g',
        on_change=update_selected_location_g
    )

# Checkboxen
add_data = False
if len(st.session_state.selected_location_g) >= 2:
    add_data = st.checkbox('Add up Regions')

multi_y_axis = False
if len(st.session_state.selected_data_g) >= 2:
    multi_y_axis = st.checkbox('Multi y-Axis')

# Merkmalauswahl
columns_to_exclude = ['continent', 'country', 'date', 'code', 'year', 'month']
desired_columns = [col for col in data_df.columns if col not in columns_to_exclude]

with row1_col2:
    st.multiselect(
        'Feature(s):',
        desired_columns,
        default=st.session_state.selected_data_g,
        key='data_selection_g',
        on_change=update_selected_data_g
    )

# Datumsbereich auswählen
selected_min_date, selected_max_date = st.slider(
    'Time period:',
    value=st.session_state.selected_dates,
    min_value=min_date,
    max_value=max_date,
    format='YYYY-MM-DD',
    key='date_selection',
    on_change=update_selected_dates
)

selected_min_date = pd.to_datetime(selected_min_date)
selected_max_date = pd.to_datetime(selected_max_date)

# Daten filtern
filtered_data = data_df[(data_df['date'] >= selected_min_date) & (data_df['date'] <= selected_max_date)]
if st.session_state.selected_location_g:
    filtered_data = filtered_data[filtered_data['country'].isin(st.session_state.selected_location_g)]
if st.session_state.selected_data_g:
    filtered_data = filtered_data[['date', 'country'] + st.session_state.selected_data_g]

# Diagramm mit Plotly erstellen
fig = go.Figure()

if add_data:
    # Werte summieren
    filtered_data = filtered_data.drop(columns=['country'])
    sum_by_date = filtered_data.groupby('date').sum()

    for j, col in enumerate(sum_by_date.columns):
        yaxis_name = f'y{j + 1}' if multi_y_axis else 'y'
        fig.add_trace(go.Scatter(
            x=sum_by_date.index,
            y=sum_by_date[col],
            mode='lines',
            name=f'{col} (Y-Axis {j + 1})' if multi_y_axis else f'{col}',
            hovertemplate='Datum: %{x}<br>Wert: %{y}',
            yaxis=yaxis_name
        ))

    # Y-Achsen für mehrere Merkmale anpassen
    if multi_y_axis:
        for j, col in enumerate(sum_by_date.columns):
            side = 'left' if j % 2 == 0 else 'right'
            position_offset = 0.03 + (0.05 * (j // 2)) if side == 'left' else 0.97 - (0.05 * (j // 2))
            fig.update_layout(**{
                f'yaxis{j + 1}': dict(
                    title=f'{col} (Y-Axis {j + 1})',
                    overlaying='y' if j > 0 else None,
                    side=side,
                    position=position_offset
                )
            })
else:

    for i, loc in enumerate(st.session_state.selected_location_g):
        for j, col in enumerate(st.session_state.selected_data_g):
            yaxis_name = f'y{j + 1}' if multi_y_axis else 'y'
            fig.add_trace(go.Scatter(
                x=filtered_data['date'],
                y=filtered_data.loc[filtered_data['country'] == loc, col],
                mode='lines',
                name=f'{loc} {col} (Y-Axis {j + 1})' if multi_y_axis else f'{loc} {col}',
                hovertemplate='Datum: %{x}<br>Wert: %{y}',
                yaxis=yaxis_name
            ))

            if multi_y_axis:
                for j, col in enumerate(st.session_state.selected_data_g):
                    side = 'left' if j % 2 == 0 else 'right'
                    position_offset = 0.03 + (0.05 * (j // 2)) if side == 'left' else 0.97 - (0.05 * (j // 2))
                    fig.update_layout(**{
                        f'yaxis{j + 1}': dict(
                            title=f'{col} (Y-Axis {j + 1})',
                            overlaying='y' if j > 0 else None,
                            side=side,
                            position=position_offset
                        )
                    })

fig.update_layout(
    xaxis_title='Datum',
    height=800,
    legend=dict(orientation='h', yanchor='bottom', y=1.1, xanchor='center', x=0.5),
    xaxis=dict(
        rangeselector=dict(
            buttons=[
                dict(count=1, label='1M', step='month'),
                dict(count=6, label='6M', step='month'),
                dict(count=1, label='1Y', step='year'),
                dict(label='All', step='all')
            ],
            xanchor='left', yanchor='top', y=1.05
        ),
        rangeslider=dict(visible=True),
    )
)

st.plotly_chart(fig)

codebook_url = 'https://raw.githubusercontent.com/owid/covid-19-data/c6b482425695ed67d3fff85ce614fc4189cf2c17/public' \
               '/data/owid-covid-codebook.csv'
cb_df = pd.read_csv(codebook_url)

with st.expander('Show Codebook'):
    st.markdown(cb_df.to_markdown(index=False))
