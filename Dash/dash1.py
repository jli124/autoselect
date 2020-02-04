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
from sqlalchemy import create_engine
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd

engine = create_engine(url)

app = dash.Dash()

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

app.layout = html.Div(children=[
    html.H1(children='Flight on-time performance check',
            style={
            'textAlign': 'center',
            'color': colors['text']
        }),
    html.Div(children='Please provide the information below'),

    html.Label('Carrier info'),
    dcc.Input(id='carrier_input',value = 'Initial text', type='text'),

    html.Label('Flight number'),
    dcc.Input(id='flightnum_input', type='text'),

    html.Label('Origins'),
    dcc.Input(id='origins_input', type='text'),

    html.Label('Destinations'),
    dcc.Input(id='destination_input', type='text'),

    dcc.DatePickerRange(id='date_range',
            min_date_allowed=dt(2020, 1, 1),
            max_date_allowed=dt(2020, 12, 31),
            initial_visible_month=dt(2020, 8, 5),
            end_date=dt(2020, 8, 15))
    dcc.Graph(id='graph')
#         fig = go.Figure(data=[go.Scatter(
#         x=[min_date_allowed:max_date_allowed]
#         y=[1, 3.2, 5.4, 7.6, 9.8, 12.5],
#         mode='markers',
#         marker=dict(
#         color=[120, 125, 130, 135, 140, 145],
#         size=[15, 30, 55, 70, 90, 110],
#         showscale=True
#         )
# )])

])

@app.callback(
    Output('graph', 'figure'),
    [Input('carrier_input', 'value'), 
    Input('flightnum_input', 'value'),
    Input('origins_input', 'value'),
    Input('destination_input', 'value'),
    Input('date_range','end_date'),
    Input('my_date_picker','end_date')
    ]
)
def udpate_figure(start_date, end_date, carrier_name, flight_num, origins, destination):
    df = pd.read_sql("SELECT * FROM flight_2020_result \
                      WHERE flight_2020_result.date BETWEEN '{}' AND '{}' \
                      AND OP_UNIQUE_CARRIER = '{}' \
                      AND ORIGINS = '{}'\
                      AND FLIGHTS = '{}' \
                      AND DEST ='{}'".format(start_date, end_date, carrier_name, flight_num, origins, destination),engine)
    go.(
        x = date)
    return {
    'data': [go.Scatter(
        x = df['date']
        y = df['DAY'],
        mode = 'markers',
        marker = dict(size = df['probability'],
            color=df['probability'],
            showscale=True))]
    'layout': go.Layout(title = 'Bubble Chart',hovermode='closet')


    }



##plotyly bubble chart with given data
# fig = go.Figure(data=[go.Scatter(
#     x=[1, 3.2, 5.4, 7.6, 9.8, 12.5],
#     y=[1, 3.2, 5.4, 7.6, 9.8, 12.5],
#     mode='markers',
#     marker=dict(
#         color=[120, 125, 130, 135, 140, 145],
#         size=[15, 30, 55, 70, 90, 110],
#         showscale=True
#         )
# )])
# dcc.Dropdown(
#     options=[
#         {'label': 'New York City', 'value': 'NYC'},
#         {'label': 'Montreal', 'value': 'MTL'},
#         {'label': 'San Francisco', 'value': 'SF'}
#     ],
#     value='MTL'
# )  

# dcc.Input(
#     placeholder='Enter the origins',
#     type='text',
#     value=''
# )

# dcc.Input(
#     placeholder='Enter the destinations',
#     type='text',
#     value=''
# )
# @app.callback(
#     dash.dependencies.Output('output-container-button', 'children'),
#     [dash.dependencies.Input('button', 'n_clicks')],
#     [dash.dependencies.State('input-box', 'value')])

if __name__ == '__main__':
    app.run_server(debug=True)