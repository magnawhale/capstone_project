# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go


##### loading in the information we'll be displaying #####



#############################################
#### BELOW IS THE CODE FOR THE DASHBOARD ####
#############################################

app = dash.Dash(__name__)

tw_df = pd.read_csv('tw_sent.csv')
tw_df.date = pd.to_datetime(tw_df.date, infer_datetime_format=True)

moon_df = pd.read_csv('phases.csv')
moon_df.date = pd.to_datetime(moon_df.date, infer_datetime_format=True)

queries = ['love OR peace OR hate OR war',
           'happy OR sad OR life OR death',
           'music OR tunes OR dance',
           'stocks OR money OR taxes',
           '(no keywords entered)',
           'politics OR government OR Trump',
           'data science OR coding OR programming']
moons = ['Full Moon','Last Quarter','New Moon','First Quarter']


####Application layout
app.layout = html.Div(children=[
   #Giving the page a title/header 
    html.H1(
        children='Lunar Cycles & Human Behavior',
        style={'textAlign': 'center'}
    ),
    
    # The area with the dropdown menus
    html.Div([
        
        # Adding a dropdown menu
        html.Div([
            html.Label('Twitter Search Phrases:'),
            dcc.Dropdown(
                id='query-dropdown',
                options=[{'label': i, 'value': i} for i in queries],
                #value='MTL'       # default initial value...remove to default as blank
            ),
        ]),
        
        # Adding a second dropdown menu
        html.Div([
            html.Label('Moon Phrase:'),
            dcc.Dropdown(
                id='moon-dropdown',
                options=[{'label': i, 'value': i} for i in moons],
                #value='MTL'       # default initial value...remove to default as blank
            ),
        ]),
        
        # setting the layout of the dropdown DIV area
        #style={'width': '30%', 'float': 'right', 'display': 'inline-block'}  ###need to experiment with these
    ]),
    
    #The area with the display
    dcc.Graph(id='tw-sent-graph')

    
    
#     # TABS
#     dcc.Tabs(id='tabs', value='tab-1', children=[
#         dcc.Tab(id='tab1', label='Daily Sentiment', value='tab-1'),
#         dcc.Tab(id='tab2', label='Word Frequencies', value='tab-2'),
#     ]),
#     html.Div(id='tabs-content')
    
    
    
])



##### Callbacks section for linking everything together #####

# @app.callback(Output('tabs-content', 'children'),
#               [Input('tabs', 'value')])
# def render_content(tab):
#     if tab == 'tab-1':
#         return html.Div([
#             html.H3('Tab content 1')    ### Whatever you want to display here
#         ])
#     elif tab == 'tab-2':
#         return html.Div([
#             html.H3('Tab content 2')    ### Whatever you want to display here
#         ])

@app.callback(
    Output('tw-sent-graph', 'figure'),
    [Input('query-dropdown', 'value'),
     Input('moon-dropdown', 'value')])
def update_graph(selected_query, selected_moon):
    tweets_df = tw_df[tw_df['query'] == selected_query]
    grouped = pd.DataFrame(tweets_df.groupby(['date', 'sentiment'])['tally'].sum()).reset_index()
    traces = []
    for sentiment in grouped.sentiment.unique():
        temp_df = grouped[grouped.sentiment == sentiment]
        traces.append(go.Scatter(
                            x=temp_df.date,
                            y=temp_df['tally'],
                            name=sentiment,
                            text=temp_df['sentiment'],
                            mode='lines',
                            opacity=0.8))

    figure = {'data': traces,
        'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
#                            height=600,
                            title=f"Daily Sentiment at Midnight (UTC) for : '{selected_query}'",
                            xaxis={"title":"Date",
                                   'rangeslider': {'visible': True},
                                   'type': 'date'},
                            yaxis={"title":"Sentiment Quantity (~1,000/day total)"})}
    return figure
        

    
    
    

# automatically update HTML display if a change is made to code
if __name__ == '__main__':
    app.run_server(debug=True)