# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

# Facebook Prophet libraries
from fbprophet import Prophet
from fbprophet import plot
from fbprophet.plot import plot_plotly, plot_cross_validation_metric
from fbprophet.diagnostics import cross_validation, performance_metrics
import plotly.offline as py

###Imports for FBProphet plot.ly functionality
import numpy as np
from fbprophet.diagnostics import performance_metrics
import matplotlib.pyplot as plt
from matplotlib.dates import (
    MonthLocator,
    num2date,
    AutoDateLocator,
    AutoDateFormatter)
from matplotlib.ticker import FuncFormatter
from pandas.plotting import deregister_matplotlib_converters
deregister_matplotlib_converters()
from plotly import tools as plotly_tools




###############################
##### Necessary functions #####
###############################

# Code adapted from: https://github.com/facebook/prophet/blob/master/python/fbprophet/plot.py
def plot_holidays_component_plotly(m, fcst):
    """Plot 'holidays' component of the forecast using Plotly.
    ----- Parameters -----
    m: Prophet model.
    fcst: pd.DataFrame output of m.predict.
    figsize: The plot's size (in px).
    ------- Returns a Plotly Figure. -------
    """
    range_margin = (fcst['ds'].max() - fcst['ds'].min()) * 0.05
    range_x = [fcst['ds'].min() - range_margin, fcst['ds'].max() + range_margin]
    text = None
    fcst = fcst[fcst['holidays'] != 0].copy()
    # Combine holidays into one hover text
    holiday_features, _, _ = m.make_holiday_features(fcst['ds'], m.holidays)
    holiday_features.columns = holiday_features.columns.str.replace('_delim_', '', regex=False)
    holiday_features.columns = holiday_features.columns.str.replace('+0', '', regex=False)
    text = pd.Series(data='', index=holiday_features.index)
    for holiday_feature, idxs in holiday_features.iteritems():
        text[idxs.astype(bool) & (text != '')] += '<br>'  # Add newline if additional holiday
        text[idxs.astype(bool)] += holiday_feature
    traces = []
    traces.append(go.Scatter(
        name='holidays',
        x=fcst['ds'],
        y=fcst['holidays'],
        mode='lines',
        line=go.scatter.Line(color='#0072B2', width=2),
        text=text,
    ))
    xaxis = go.layout.XAxis(
        title='Date',
        type='date',
        rangeslider={'visible': True},
        range=range_x)
    yaxis = go.layout.YAxis(rangemode='tozero',
                            title='Lunar Phase Correlation',
                            zerolinecolor='#AAA')
    layout = go.Layout(
#        width=figsize[0],
#        height=figsize[1],
        showlegend=False,
        xaxis=xaxis,
        yaxis=yaxis
    )
    figure = go.Figure(data=traces, layout=layout)
    return figure







#############################################
#### BELOW IS THE CODE FOR THE DASHBOARD ####
#############################################

app = dash.Dash(__name__)

markdown_paragraph = '''
### About this Project

#### By Matthew E. Parker

Data Science Bootcamp Capstone Project for Flatiron School. 
For more information about this project, please read my 
[Medium article]('https://medium.com/@matthewparker_1059/modeling-lunar-cycles-in-tweets-and-financial-markets-using-facebook-prophet-d6ec0e9e20f'). 
If you wish to look at the code for yourself, please refer to the project's
[GitHub repository]('https://github.com/magnawhale/capstone_project').

For a fascinating (but slow-loading) sample exploration of Tweets, 
download and then open view my 
[Scattertext html file]('https://github.com/magnawhale/capstone_project/blob/master/Scattertext_nowords_example.html').
Just continue loading if you get any error messages about responsiveness when opening this file.
'''


stopwords = set(ENGLISH_STOP_WORDS)
stopwords.update(['twitter','com','pic','ve','ll','just','like','don','really','00'])

tw_df = pd.read_csv('tw_sent.csv')
tw_df.date = pd.to_datetime(tw_df.date, infer_datetime_format=True)

moon_df = pd.read_csv('phases.csv')
moon_df.date = pd.to_datetime(moon_df.date, infer_datetime_format=True)

# prepping lunar phases for Facebook Prophet
# NOTE: this is where I would change lower/upper windows 
phase_names = ['Full Moon','Last Quarter','New Moon','First Quarter']
ph_list = []
for phase in phase_names:
	moons = pd.DataFrame(moon_df.loc[moon_df['phase'] == phase]['date']).reset_index(drop=True)
	moons.columns = ['ds']
	moons['holiday'] = str(phase).lower().replace(" ", "")
	moons['lower_window'] = 0
	moons['upper_window'] = 1
	ph_list.append(moons)
phases_FBP = pd.concat((ph_list[0], ph_list[1], ph_list[2], ph_list[3]))

tw_word_freqs_df = pd.read_csv('tw_word_freqs.csv')
tw_word_freqs_df['count'] = tw_word_freqs_df['count'].astype(int)

queries = ['(no keywords entered)',
           'love OR peace OR hate OR war',
           'happy OR sad OR life OR death',
           'music OR tunes OR dance',
           'stocks OR money OR taxes',
           'politics OR government OR Trump',
           'data science OR coding OR programming']
moons = ['No Phase', 'Full Moon','Last Quarter','New Moon','First Quarter']


############################
#### Application layout ####
app.layout = html.Div(children=[    ### whole page
    html.H1(                        ### page header
        children='Lunar Cycles & Human Behavior',
        style={'textAlign': 'center'}
    ),

    ### two column area
    html.Div([    

        ####################################
        # The area with the dropdown menus #
        html.Div([

            # Adding a dropdown menu
            html.Div([
#                html.P('Twitter Search Phrases:'),
                dcc.Dropdown(
                    id='query-dropdown',
                    options=[{'label': i, 'value': i} for i in queries],
					placeholder="Select a Twitter Search Phrase",
                    value='(no keywords entered)'  # default initial value
                ),
                html.Br(),
                html.P('Mood for which to explore seasonality:'),
                dcc.RadioItems(
				    id='mood-radio',
                    options=[
                        {'label': 'Positivity', 'value': 'positive'},
                        {'label': 'Negativity', 'value': 'negative'},
                    ],
                    value='positive',
                    labelStyle={'display': 'inline-block'}
                ),
				html.Br(),
#                html.P('Moon Phase:'),
                dcc.Dropdown(
                    id='moon-dropdown',
                    options=[{'label': i, 'value': i} for i in moons],
					placeholder='Select a Moon Phase',
                    value='No Phase'   # default initial value
                ),
				html.Br(),
                dcc.Markdown(markdown_paragraph)
            ]),


            # setting the layout of the dropdown DIV area
            ], style = {'width': '20%',
                'height': '49%',
                'display': 'inline-block'
            }
        ),

        html.Div([], style={'width': '5%', 'display': 'inline-block'}),  ## just a spacer


        #############################
        # The area with the display #

        ### TABS ###
        html.Div([
            dcc.Tabs(id='tabs', value='tab-1', children=[
                dcc.Tab(id='tab1', label='Daily Sentiment', value='tab-1'),
                dcc.Tab(id='tab2', label='Word Frequencies', value='tab-2'),
                dcc.Tab(id='tab3', label='Facebook Prophet Seasonality', value='tab-3'),
            ]),

            ## displayed below tabs ##
            html.Div(id='tabs-content', children=[
                dcc.Graph(id='tw-graph'),

                dcc.Slider(
                    id='freq-slider',
                    min=1,
                    max=1000,
                    step=1,
                    value=50
                ),
                html.Div(id='slider-output-container')
            ]),
            ],
            style={'width': '75%', 'display': 'inline-block'}  ###setting the graph/tabs area to right of options
        )
    ])
])


####################################################
##### Callbacks section for linking everything #####
####################################################


@app.callback(
    Output('slider-output-container', 'children'),
    [Input('freq-slider', 'value'),
     Input('query-dropdown', 'value')])
def update_output(selected_value, selected_query):
    return f'Displaying the {selected_value} most frequent words from the query: "{selected_query}"'


@app.callback(
    Output('tw-graph', 'figure'),
    [Input('query-dropdown', 'value'),
	 Input('mood-radio', 'value'),
     Input('moon-dropdown', 'value'),
     Input('tabs', 'value'),
     Input('freq-slider', 'value')])
def update_graph(selected_query, selected_mood, selected_moon, selected_tab, selected_n):
    grouped = tw_df[tw_df['query'] == selected_query]
#    tweets_df.dropna(inplace=True)
#    grouped = pd.DataFrame(tweets_df.groupby(['date', 'sentiment'])['tally'].sum()).reset_index()

    if selected_tab == 'tab-1':
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
                                #height=600,
                                title=f"Daily Sentiment at Midnight (UTC) for : '{selected_query}'",
                                xaxis={"title":"Date",
                                       'rangeslider': {'visible': True},
                                       'type': 'date'},
                                yaxis={"title":"Sentiment Quantity (~1,000/day total)"})}
        return figure

    elif selected_tab == 'tab-2':
        df2 = tw_word_freqs_df[(tw_word_freqs_df['query'] == selected_query) &
                               (tw_word_freqs_df['phase'] == selected_moon)][:selected_n]
        ### plotting
        trace = [go.Bar(x=df2['text'], y=df2['count'], name='', )]
        figure = {'data': trace,
            'layout': go.Layout(title=f"Top {selected_n} Words for {selected_moon} at Midnight UTC",
                hovermode="closest",
                xaxis={
                    'title': f"Search Keywords: {selected_query}", 
                    'titlefont': {'color': 'black', 'size': 14},
                    'tickfont': {'size': 11, 'color': 'black'}},
                yaxis={'tickfont': {'color': 'black'}}
            )
        }
        return figure

    elif selected_tab == 'tab-3':
        #prepare data for FBProphet
        grp_mood = grouped[grouped.sentiment == selected_mood].drop(['sentiment', 'query'], axis=1).reset_index(drop=True)
        grp_mood.columns = ['ds','y']

        #make model and forecast
        m = Prophet(holidays=phases_FBP)
        m.fit(grp_mood)
        future = m.make_future_dataframe(periods=1)
        forecast = m.predict(future)

        #plot holidays (lunar seasonalities)
        figure = plot_holidays_component_plotly(m, forecast)
        return figure



# automatically update HTML display if a change is made to code
if __name__ == '__main__':
    app.run_server(debug=True)