# -*- coding: utf-8 -*-
import dash
import dash_bootstrap_components as dbc   #for getting columns/tabs to work
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
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
        showlegend=False,
        xaxis=xaxis,
        yaxis=yaxis
    )
    figure = go.Figure(data=traces, layout=layout)
    return figure







#############################################
#### BELOW IS THE CODE FOR THE DASHBOARD ####
#############################################

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.GRID])
server = app.server


markdown_paragraph = '''
### About this Project

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
    moons['lower_window'] = -0
    moons['upper_window'] = 1
    ph_list.append(moons)
phases_FBP = pd.concat((ph_list[0], ph_list[1], ph_list[2], ph_list[3]))

weekends_df = pd.read_csv('weekends_2018_19.csv')
weekends_df.date = pd.to_datetime(weekends_df.date, infer_datetime_format=True)

tw_word_freqs_df = pd.read_csv('tw_word_freqs.csv')
tw_word_freqs_df['count'] = tw_word_freqs_df['count'].astype(int)

fins_df = pd.read_csv('financials.csv')
fins_df.date = pd.to_datetime(fins_df.date, infer_datetime_format=True)

queries = ['(no keywords entered)',
           'love OR peace OR hate OR war',
           'happy OR sad OR life OR death',
           'music OR tunes OR dance',
           'stocks OR money OR taxes',
           'politics OR government OR Trump',
           'data science OR coding OR programming']
moons = ['No Phase', 'Full Moon','Last Quarter','New Moon','First Quarter']
stocks = {'Apple': 'AAPL',
          'Amazon':'AMZN',
          'Dow Jones Industrial':'DJI',
          'Facebook':'FB',
          'Microsoft':'MSFT',
          'Google':'GOOG',
          'S&P 500 Index':'INX',
          'SBA Communications':'SBAC',
          'Twitter':'TWTR'}
currencies = {'Euro':'EUR',
              'Great British Pound':'GBP',
              'Japanese Yen':'JPY',
              'Swiss Franc':'CHF'}
cryptocurrencies = {'Bitcoin':'BTC',
                    'Ethereum':'ETH'}
fin_metrics = ['open','close','close_24','change_24','range',
               'range_24','high','low','high_24','low_24']


tabs_styles = {
    'height': '36px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '11px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '11px',
    'fontWeight': 'bold'
}




############################
#### Application layout ####
app.layout = html.Div(children=[               ### whole page
    html.H1('Lunar Cycles & Human Behavior',   ### page header
        style={'textAlign': 'center', 'margin-bottom': 0.3}
    ),
    html.H3('by Matthew E. Parker',
        style={'textAlign': 'center', 'margin-top': 0.1, 'margin-bottom': 0.2}
    ),
    html.H5(['(best viewed in Google Chrome)'],
        style={'textAlign': 'center', 'margin-top': 0.2, 'margin-bottom': 0}
        ),
    html.Br(),

    ### three column area
    dbc.Row([    

        ####################################
        # The area with the dropdown menus #
        dbc.Col(id='left-column', width=3, children=[

            # Choosing a category
            html.Div( 
                children=[
                    dcc.Tabs(id='categories', 
                        value='cat-1', 
                        children=[
                            dcc.Tab(id='cat1', label='Twitter Sentiment', value='cat-1', 
                                style=tab_style, selected_style=tab_selected_style),
                            dcc.Tab(id='cat2', label='Financial Markets', value='cat-2', 
                                style=tab_style, selected_style=tab_selected_style),
                        ]
                    ),
                    
                    
                    ## displayed below category tabs ##
                    html.Div(id='categories-content', 
                        children=[
                            html.Table(style={'border-spacing': 7}, children=[
                                html.Tr(id='radio-label', children='Radio:'),
                                dcc.RadioItems(id='radio'),
                                html.Tr([
                                    html.Div(id='dropdown1_label', children="Dropdown1:"),
                                    dcc.Dropdown(id='dropdown1'),
                                ]),
                                html.Tr([
                                    html.Div(id='dropdown2_label', children="Dropdown2:"),
                                    dcc.Dropdown(id='dropdown2'),
                                ]),
                            ]),                    
                            dcc.Markdown(markdown_paragraph),
                        ]
                    ),
                ],
            ),
        ]),


        #############################
        # The area with the display #
        dbc.Col(id='right-column', width=9,
            children=[
                html.Div(id='cat-graphs', children=[
                    dcc.Tabs(id='tabs', 
                        value='tab-1', 
                        style={
                            'height': '45px'
                        },
                        children=[
                            dcc.Tab(id='tab1', value='tab-1', 
                                style=tab_style, selected_style=tab_selected_style),
                            dcc.Tab(id='tab2', value='tab-2', 
                                style=tab_style, selected_style=tab_selected_style),
                            dcc.Tab(id='tab3', value='tab-3', 
                                style=tab_style, selected_style=tab_selected_style),
                        ]
                    ),
                    ## displayed below tabs ##
                    html.Div(id='tabs-content', children=[
                        dcc.Graph(id='graph'),               ########### all graphs ##############
                        dcc.Slider(                  ### slider just for word freq. histograms ###
                            id='freq-slider',
                            min=1,
                            max=500,
                            step=1,
                            value=50
                        ),
                        html.Div(id='slider-output-container'),
                        html.Br()
                    ])
                ]),
            ]
        )
    ])
])



####################################################
##### Callbacks section for linking everything #####
####################################################


############## Callbacks for options section ###########
@app.callback(
    Output('slider-output-container', 'children'),
    [Input('freq-slider', 'value'),
     Input('dropdown1', 'value')])
def update_output(selected_value, selected_query):
    return f'Displaying the {selected_value} most frequent words from the query: "{selected_query}"'


@app.callback(
    Output('radio-label', 'children'),
    [Input('categories', 'value')])
def update_radio_label(selected_category):
    if selected_category == 'cat-1':    ### Twitter Sentiment
        return 'Mood for which to explore seasonality:'
    elif selected_category == 'cat-2':  ### Financial Markets
        return 'Type of financial object:'


@app.callback(
    [Output('radio', 'options'),
     Output('radio', 'value'),
     Output('radio', 'labelStyle')],
    [Input('categories', 'value')])
def update_radio(selected_category):
    if selected_category == 'cat-1':     ### Twitter Sentiment
        options=[{'label': 'Positivity', 'value': 'positive'},
                {'label': 'Negativity', 'value': 'negative'}]
        value='positive'
        labelStyle={'display': 'inline-block'}
        return options, value, labelStyle
    elif selected_category == 'cat-2':    ### Financial Markets
        options=[{'label': 'Stock/Index', 'value': 'stock'},
            {'label': 'Currency', 'value': 'currency'},
            {'label': 'Cryptocurrency', 'value': 'cryptocurrency'}]
        value='stock'
        labelStyle={'display': 'block'}
        return options, value, labelStyle


@app.callback(
    [Output('dropdown1_label', 'children'),
     Output('dropdown1', 'options'),
     Output('dropdown1', 'value'),
     Output('dropdown2_label', 'children'),
     Output('dropdown2', 'options'),
     Output('dropdown2', 'value')],
    [Input('categories', 'value'),
     Input('radio', 'value')],
    [State('dropdown1', 'value'),
     State('dropdown2', 'value')])
def update_dropdown_options(selected_category, selected_fin_type, drop1, drop2):
    if selected_category == 'cat-1':             ### Twitter Sentiment
        if drop1 in queries[1:] or drop2 in moons[1:]:
            raise PreventUpdate
        else:
            dropdown1_label = 'Select a Twitter Search Phrase:'
            options1 = [{'label': i, 'value': i} for i in queries]
            value1 = '(no keywords entered)'
            dropdown2_label = 'Select a Moon Phase:'
            options2 = [{'label': i, 'value': i} for i in moons]
            value2 = 'No Phase'
            return dropdown1_label, options1, value1, dropdown2_label, options2, value2
    elif selected_category == 'cat-2':            ### Financial Markets
        dropdown1_label = "Select a Ticker Symbol:"
        dropdown2_label = "Select a metric to graph:"
        options2 = [{'label': i, 'value': i} for i in fin_metrics]
        value2 = 'change_24'
        if selected_fin_type == 'stock':
            options1 = [{'label': i, 'value': j} for i,j in stocks.items()]
            value1 = 'MSFT'
            return dropdown1_label, options1, value1, dropdown2_label, options2, value2
        elif selected_fin_type == 'currency':
            options1 = [{'label': i, 'value': j} for i,j in currencies.items()]
            value1 = 'EUR'
            return dropdown1_label, options1, value1, dropdown2_label, options2, value2
        elif selected_fin_type == 'cryptocurrency':
            options1 = [{'label': i, 'value': j} for i,j in cryptocurrencies.items()]
            value1 = 'BTC'
            return dropdown1_label, options1, value1, dropdown2_label, options2, value2




##########   Callbacks for Graph section (including tabs)############

@app.callback(
    [Output('tabs', 'value'),
     Output('tab1', 'label'),
     Output('tab2', 'label'),
     Output('tab3', 'label')],
    [Input('categories', 'value')])
def update_tab_labels(selected_category):
    if selected_category == 'cat-1':
        return 'tab-1', 'Daily Sentiment', 'Word Frequencies', 'Facebook Prophet Seasonality'
    elif selected_category == 'cat-2':
        return 'tab-1', 'Overall Performance', 'Chosen Metric', 'Facebook Prophet Seasonality'


@app.callback(
    Output('graph', 'figure'),
    [Input('categories', 'value'),
     Input('radio', 'value'),
     Input('dropdown1', 'value'),
     Input('dropdown2', 'value'),
     Input('tabs', 'value'),
     Input('freq-slider', 'value')])
def update_graph(selected_category, 
                 selected_radio, 
                 selected_dropdown1, 
                 selected_dropdown2, 
                 selected_tab,
                 selected_n):
    if selected_category == 'cat-1':     ### Twitter Sentiment ###
        grouped = tw_df[tw_df['query'] == selected_dropdown1]
        if selected_tab == 'tab-1':      # 'Daily Sentiment'
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
                                    title=f"Daily Sentiment at Midnight (UTC) for : '{selected_dropdown1}'",
                                    xaxis={"title":"Date",
                                           'rangeslider': {'visible': True},
                                           'type': 'date'},
                                    yaxis={"title":"Sentiment Quantity (~1,000/day total)"})}
            return figure
        elif selected_tab == 'tab-2':      # 'Word Frequencies'
            df2 = tw_word_freqs_df[(tw_word_freqs_df['query'] == selected_dropdown1) &
                                   (tw_word_freqs_df['phase'] == selected_dropdown2)][:selected_n]
            ### plotting
            trace = [go.Bar(x=df2['text'], y=df2['count'], name='', )]
            figure = {'data': trace,
                'layout': go.Layout(title=f"Top {selected_n} Words for {selected_dropdown2} at Midnight UTC",
                    hovermode="closest",
                    xaxis={
                        'title': f"Search Keywords: {selected_dropdown1}", 
                        'titlefont': {'color': 'black', 'size': 14},
                        'tickfont': {'size': 11, 'color': 'black'}},
                    yaxis={'tickfont': {'color': 'black'}}
                )
            }
            return figure
        elif selected_tab == 'tab-3':      # 'Facebook Prophet Seasonality'
            #prepare data for FBProphet
            grp_mood = grouped[grouped.sentiment == selected_radio].drop(['sentiment', 'query'], axis=1).reset_index(drop=True)
            grp_mood.columns = ['ds','y']
            #make model and forecast
            m = Prophet(holidays=phases_FBP)
            m.fit(grp_mood)
            future = m.make_future_dataframe(periods=1)
            forecast = m.predict(future)
            #plot holidays (lunar seasonalities)
            figure = plot_holidays_component_plotly(m, forecast)
            return figure
    elif selected_category == 'cat-2':       ### Financial Markets ###
        temp_df = fins_df[fins_df['ticker'] == selected_dropdown1].reset_index(drop=True)
        if selected_tab == 'tab-1':        # 'Overall Performance'
            traces = []
            traces.append(go.Scatter(
                x=temp_df['date'],
                y=temp_df['close'],
                name=selected_dropdown1,
                text=temp_df['close'],
                mode='lines',
                opacity=0.8))
            figure = {'data': traces,
                'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                    title=f"{selected_dropdown1} Daily Closing Price for 2018",
                    xaxis={"title":"Date",
                           'rangeslider': {'visible': True},
                           'type': 'date'},
                    yaxis={"title": "Closing Price (in USD)"})}
            return figure
        elif selected_tab == 'tab-2':      # 'Chosen Metric'
            traces = []
            traces.append(go.Scatter(
                x=temp_df['date'],
                y=temp_df[selected_dropdown2],
                name=selected_dropdown1,
                text=temp_df[selected_dropdown2],
                mode='lines',
                opacity=0.8))
            figure = {'data': traces,
                'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                    title=f"{selected_dropdown1} Daily Performance of '{selected_dropdown2}' metric for 2018",
                    xaxis={"title":"Date",
                           'rangeslider': {'visible': True},
                           'type': 'date'},
                    yaxis={"title":f"{selected_dropdown2} (in USD)"})}
            return figure
        elif selected_tab == 'tab-3':      # 'Facebook Prophet Seasonality'
            #prepare data for FBProphet
            fbp_temp_df = temp_df[['date', selected_dropdown2]]
            fbp_temp_df.columns = ['ds','y']
            #make model and forecast
            m = Prophet(holidays=phases_FBP)
            m.fit(fbp_temp_df)
            future = m.make_future_dataframe(periods=1)
            if selected_radio != 'cryptocurrency':
                future = future[~future['ds'].isin(weekends_df.date)]            
            forecast = m.predict(future)
            #plot holidays (lunar seasonalities)
            figure = plot_holidays_component_plotly(m, forecast)
            return figure



# automatically update HTML display if a change is made to code
if __name__ == '__main__':
    app.run_server(debug=False)   #set to False for production so users don't get errors