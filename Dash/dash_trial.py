#Dash setup
# pip install plotly==2.5.1
# pip install dash==0.21.0
# pip install dash-core-components==0.22.1
# pip install dash-html-components==0.10.0
# pip install dash-renderer==0.12.1
from datetime import datetime as dt
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import pandas.io.sql as sqlio
import psycopg2
from sqlalchemy import create_engine



app = dash.Dash()

colors = {
    'background': '#111111',
    'text': '#000008b'
}

app.layout = html.Div(children=[
    html.H1(children='Flight on-time performance check',
            style={
            'textAlign': 'center',
            'color': colors['text']
        }),
    html.Div(children='Please provide the information below'),

    html.Label('Carrier info'),
    dcc.Input(id='carrier_input',value = 'UA', type='text'),

    html.Label('Flight number'),
    dcc.Input(id='flightnum_input', value = '738', type='text'),

    html.Label('Origin'),
    dcc.Input(id='origins_input',value = 'SEA', type='text'),

    html.Label('Destination'),
    dcc.Input(id='destination_input', value = 'DEN',type='text'),

    html.Label('Date Range'),
    dcc.DatePickerRange(id='date_range',
            min_date_allowed=dt(2020, 1, 1),
            max_date_allowed=dt(2020, 12, 31),
            initial_visible_month=dt(2020, 2, 5),
            start_date=dt(2020, 2, 15)),
    
    html.Label('Select the performances'),
    dcc.Dropdown(
    id='response_df',
    options=[
        {'label': 'Delayed', 'value': 'flight_2020_delay'},
        {'label': 'Weather delay', 'value': 'flight_2020_cancel'},
        {'label': 'Cancellation', 'value': 'flight_2020_cancelled'}
    ],
    value='delayed'
), 
    dcc.Graph(id='graph')
])

@app.callback(
    Output('graph', 'figure'),
    [Input('carrier_input', 'value'), 
    Input('flightnum_input', 'value'),
    Input('origins_input', 'value'),
    Input('destination_input', 'value'),
    Input('date_range','start_date'),
    Input('date_range','end_date'),
    Input('response_df','value')
    ]
)
def udpate_figure(carrier_name, flight_num, origin, destination,start_date,end_date,response_df):
    conn = psycopg2.connect(
            host = "jdbc:postgresql://host:portnum/databasename",
            database = "databasename",
            user = "username",
            password = "yourpassword")
    c = conn.cursor()
    if start_date is not None:
        start_date = dt.strptime(start_date[:10], '%Y-%m-%d').date()
    if end_date is not None:
        end_date = dt.strptime(end_date[:10], '%Y-%m-%d').date()
    df = pd.read_sql("SELECT * FROM {} LIMIT 5000".format(response_df),conn)
    df.columns = ['year','month','day_of_month','day_of_week','date','carrier','flight_num','origin_airport','destination_airport','weather_delay','arrival_delay','cancelled','prediction','probability']
    df.query('carrier== "{}" and flight_num == "{}" and origin_airport == "{}" and destination_airport == "{}"'.format(carrier_name,flight_num,origin,destination))
    #dfc = df[df['carrier']==carrier_name]
    # dfcf = dfc[dfc['flight_num']==flight_num]
    # dfcfo = dfcf[dfcf['origin_airport']==origin]
    # dfcfod = dfcfo[dfcfo['destination_airport']==destination]
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] > start_date]
    df = df[df['date'] <= end_date]
    #df = df[df['date'] < end_date]
    return {'data':[go.Heatmap(
                   z=df['probability'],
                   x=df['date'],
                   y=df['day_of_week'],
                   colorscale='RdBu')],
                   #'Viridis')],
    # Scatter(x=df_filtered['date'], 
    #                             y=df_filtered['day_of_week'],
    #                             mode='markers',
    #                             marker=dict(color=df_filtered['probability'],size = 40, showscale=True))],
            'layout':go.Layout(title='Planner',
                                xaxis_nticks=15)}

if __name__ == '__main__':
    app.run_server(debug=True)

