import numpy as np
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import folium
import plotly.graph_objs as go
import plotly.io as pio


from dash.dependencies import Output, Input
from joblib import load


###################################################################
# clean csv

df = pd.read_csv('final.csv', index_col=0)

# change int column to Int64 to handle missing value
for i in ['CIVIC_NUMBER', 'bathroom', 'bedroom']:
    df[i] = df[i].astype('Int64')

# drop rows with null price, area and built_year, then clean data
df = df.dropna(subset=['total_value', 'prev_value', 'area'])
df['total_value'] = df['total_value'].apply(lambda x: int(x.replace('$', '').replace(',', '')))
df['prev_value'] = df['prev_value'].apply(lambda x: int(x.replace('$', '').replace(',', '')))
df['garage'] = df['garage'].apply(lambda x: 1 if not pd.isnull(x) else 0)
df['carport'] = df['carport'].apply(lambda x: 1 if not pd.isnull(x) else 0)
df['built_year'] = df['built_year'].apply(lambda x:  int(x) if x != ' ' else np.nan)
df['bedroom'] = df['bedroom'].fillna(0).astype(int)
df['bathroom'] = df['bathroom'].fillna(0).astype(int)


def get_area(str_area):
    str_lst = str_area.split()
    if 'x' in str_lst:
        x_index = str_lst.index('x')
        return float(str_lst[x_index - 1]) * float(str_lst[x_index + 1])
    if 'Sq' in str_lst:
        sq_i = str_lst.index('Sq')
        return float(str_lst[sq_i - 1])


df['area'] = df['area'].apply(lambda x: get_area(x))
df['unit_price'] = df['total_value'] / df['area']
df['price_change'] = df['total_value'] - df['prev_value']
df['change_rate'] = df['price_change'] / df['prev_value']

df.to_csv('cleaned.csv', index=False)
# correlation between all parameters
corr_df = df.corr()



###############################################################################################
# Visualization

df = pd.read_csv('cleaned.csv')
df = df.dropna(subset=['area', 'CIVIC_NUMBER', 'actual_address'])
region_df = df.groupby(['Geo Local Area'])['unit_price', 'change_rate', 'total_value'].mean()
region_df['district'] = region_df.index
bedroom_df = df.groupby(['bedroom']).agg({'total_value':['mean', 'count'], 'unit_price':'mean', 'change_rate':'mean'})
bathroom_df = df.groupby(['bathroom']).agg({'total_value':['mean', 'count'], 'unit_price':'mean', 'change_rate':'mean'})
garage_df = df[df['bedroom']!=0].groupby(['garage']).agg({'total_value':['mean', 'count'], 'unit_price':'mean', 'change_rate':'mean'})

# binning year value
bins = [1880, 1890, 1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020]
labels = [1880, 1890, 1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010]
df['binned_year'] = pd.cut(df['built_year'], bins=bins, labels=labels)
year_df = df.groupby(['binned_year'])[['total_value', 'unit_price', 'change_rate']].mean()

# binning area value
bins = [0, 500, 1000, 2000, 3000, 4000, 5000, 10000, 4000000]
labels = [0, 500, 1000, 2000, 3000, 4000, 5000, 10000]
df['binned_area'] = pd.cut(df['area'], bins=bins, labels=labels)
area_df = df.groupby(['binned_year'])[['total_value', 'unit_price', 'change_rate']].mean()


pio.renderers.default = 'browser'

# create graph for price change rate based on region
change_rate, change_rate_region = (list(t) for t in zip(*sorted(zip(region_df['change_rate'], region_df.index))))
price_change_fig = go.Figure(data=[
    go.Bar(name='avg price change rate', x=change_rate, y=change_rate_region, orientation='h')
])
price_change_fig.update_layout(title_text='Price change rate')

# create graph for unit price based on region
unit_price, unit_price_region = (list(t) for t in zip(*sorted(zip(region_df['unit_price'], region_df.index))))
unit_price_fig = go.Figure(data=[
    go.Bar(name='avg unit price', x=unit_price, y=unit_price_region, orientation='h')
])
unit_price_fig.update_layout(title_text='Value per Square Feet')

# use folium to create map graph
region_df['log_unit_price'] = np.log(region_df['unit_price'])
m = folium.Map(location=[49.258, -123.15], zoom_start=13)
folium.Choropleth(
    geo_data='./vancouver.json',
    name='choropleth',
    data=region_df,
    columns=['district', 'log_unit_price'],
    key_on='feature.id',
    fill_color='Oranges',
    fill_opacity=0.9,
    line_opacity=0.2,
    legend_name='unit price: dollar per Sq Ft (value is log'
).add_to(m)
folium.LayerControl().add_to(m)
m.save('region_unit_price.html')

#################################################################################################
#Build dashboard
external_stylesheets = ['https://cdn.rawgit.com/plotly/dash-app-stylesheets/2d266c578d2a6e8850ebce48fdb52759b2aef506/stylesheet-oil-and-gas.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

colors = {
    'background': '#F0FFFF',
    'text': '#000000'
    # 'text': '#7FDBFF'
}

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(children='Vancouver house price',
            style={
                'textAlign': 'center',
                'color': colors['text'],
                'padding': '50px, 50px'
            }
    ),
    # region based graph
    html.Div(
        html.Div([
            html.Div([
                dcc.Graph(style={'height': '700px'}, id='region-price-change', figure=price_change_fig)
            ], className='six columns'),
            html.Div([
                dcc.Graph(style={'height': '700px'}, id='region-unit-price', figure=unit_price_fig)
            ], className='six columns')
        ], className='row'),
        style={'padding': '50px, 50px', 'height': '1000', 'width': '80%', 'margin': 'auto'}
    ),
    html.Iframe(id='map', srcDoc=open('region_unit_price.html', 'r').read(), width='80%', height='800',
                style={'display': 'block', 'margin-left': 'auto','margin-right': 'auto',
                       'padding':'25px 50px', 'border-style':'none'}),
    # drop down menu for region selection
    html.Div([
        html.Label('Select Regions', style={'color': colors['text'], 'margin': 'auto'}),
        dcc.Dropdown(
            id= 'graph-dropdown',
            options=[
                {'label': 'Kensington-Cedar Cottage', 'value': 'Kensington-Cedar Cottage'},
                {'label': 'Renfrew-Collingwood', 'value': 'Renfrew-Collingwood'},
                {'label': 'Sunset', 'value': 'Sunset'},
                {'label': 'Hastings-Sunrise', 'value': 'Hastings-Sunrise'},
                {'label': 'Dunbar-Southlands', 'value': 'Dunbar-Southlands'},
                {'label': 'Victoria-Fraserview', 'value': 'Victoria-Fraserview'},
                {'label': 'Riley Park', 'value': 'Riley Park'},
                {'label': 'Marpole', 'value': 'Marpole'},
                {'label': 'Killarney', 'value': 'Killarney'},
                {'label': 'Kerrisdale', 'value': 'Kerrisdale'},
                {'label': 'Kitsilano', 'value': 'Kitsilano'},
                {'label': 'Grandview-Woodland', 'value': 'Grandview-Woodland'},
                {'label': 'Arbutus-Ridge', 'value': 'Arbutus-Ridge'},
                {'label': 'Mount Pleasant', 'value': 'Mount Pleasant'},
                {'label': 'Shaughnessy', 'value': 'Shaughnessy'},
                {'label': 'Oakridge', 'value': 'Oakridge'},
                {'label': 'West Point Grey', 'value': 'West Point Grey'},
                {'label': 'Fairview', 'value': 'Fairview'},
                {'label': 'Strathcona', 'value': 'Strathcona'},
                {'label': 'South Cambie', 'value': 'South Cambie'},
                {'label': 'Downtown', 'value': 'Downtown'},
                {'label': 'West End', 'value': 'West End'}

            ],
            value=['Kensington-Cedar Cottage', 'Renfrew-Collingwood', 'Sunset',
               'Hastings-Sunrise', 'Dunbar-Southlands', 'Victoria-Fraserview',
               'Riley Park', 'Marpole', 'Killarney', 'Kerrisdale', 'Kitsilano',
               'Grandview-Woodland', 'Arbutus-Ridge', 'Mount Pleasant', 'Shaughnessy',
               'Oakridge', 'West Point Grey', 'Fairview', 'Strathcona', 'South Cambie',
               'Downtown', 'West End'],
            multi=True,
            style={'backgroundColor': colors['background'], 'margin': 'auto' }
        )
        ], style={'backgroundColor': colors['background'], 'margin': 'auto', 'width': '80%'}
    ),
    # line chart based on rooms
    html.Div(
        html.Div([
            html.Div([
                dcc.Graph(id='total value based on room', figure={})
            ], className='four columns'),
            html.Div([
                dcc.Graph(id='unit price based on room', figure={})
            ], className='four columns'),
            html.Div([
                dcc.Graph(id='change rate based on room', figure={})
            ], className='four columns')
        ], className='row'),
        style={'padding': '50px, 50px', 'height': '30%', 'width': '80%', 'margin': 'auto'}
    ),
    # line chart based on year
    html.Div(
        html.Div([
            html.Div([
                dcc.Graph(id='total value based on year', figure={})
            ], className='four columns'),
            html.Div([
                dcc.Graph(id='unit price based on year', figure={})
            ], className='four columns'),
            html.Div([
                dcc.Graph(id='change rate based on year', figure={})
            ], className='four columns')
        ], className='row'),
        style={'padding': '50px, 50px', 'height': '30%', 'width': '80%', 'margin': 'auto'}
    ),
    # line chart based on house area
    html.Div(
        html.Div([
            html.Div([
                dcc.Graph(id='total value based on area', figure={})
            ], className='four columns'),
            html.Div([
                dcc.Graph(id='unit price based on area', figure={})
            ], className='four columns'),
            html.Div([
                dcc.Graph(id='change rate based on area', figure={})
            ], className='four columns')
        ], className='row'),
        style={'padding': '50px, 50px', 'height': '30%', 'width': '80%', 'margin': 'auto'}
    ),
    # apply machine learning model
    html.Div([
    html.Header('House price prediction', style={'color': colors['text'], 'margin': 'auto'}),
    html.Div([

        html.Div([
            dcc.Dropdown(
                id='prediction region',
                options=[
                    {'label': 'Kensington-Cedar Cottage', 'value': 'Kensington-Cedar Cottage'},
                    {'label': 'Renfrew-Collingwood', 'value': 'Renfrew-Collingwood'},
                    {'label': 'Sunset', 'value': 'Sunset'},
                    {'label': 'Hastings-Sunrise', 'value': 'Hastings-Sunrise'},
                    {'label': 'Dunbar-Southlands', 'value': 'Dunbar-Southlands'},
                    {'label': 'Victoria-Fraserview', 'value': 'Victoria-Fraserview'},
                    {'label': 'Riley Park', 'value': 'Riley Park'},
                    {'label': 'Marpole', 'value': 'Marpole'},
                    {'label': 'Killarney', 'value': 'Killarney'},
                    {'label': 'Kerrisdale', 'value': 'Kerrisdale'},
                    {'label': 'Kitsilano', 'value': 'Kitsilano'},
                    {'label': 'Grandview-Woodland', 'value': 'Grandview-Woodland'},
                    {'label': 'Arbutus-Ridge', 'value': 'Arbutus-Ridge'},
                    {'label': 'Mount Pleasant', 'value': 'Mount Pleasant'},
                    {'label': 'Shaughnessy', 'value': 'Shaughnessy'},
                    {'label': 'Oakridge', 'value': 'Oakridge'},
                    {'label': 'West Point Grey', 'value': 'West Point Grey'},
                    {'label': 'Fairview', 'value': 'Fairview'},
                    {'label': 'Strathcona', 'value': 'Strathcona'},
                    {'label': 'South Cambie', 'value': 'South Cambie'},
                    {'label': 'Downtown', 'value': 'Downtown'},
                    {'label': 'West End', 'value': 'West End'}

                ],
                multi=False,
                style={'backgroundColor': colors['background'], 'margin': 'auto'}
            )
        ], style={'backgroundColor': colors['background'], 'margin': 'auto'}, className='two columns'),
        html.Div(dcc.Input(id='area-text', value='area (1000 for 1000 Sq Ft', type='text'), className='two columns'),
        html.Div(dcc.Input(id='bedroom-text', value='# of bedroom', type='text'), className='two columns'),
        html.Div(dcc.Input(id='bathroom-text', value='# of bathroom', type='text'), className='two columns'),
        html.Div(dcc.Input(id='garage-text', value='garage (1 for yes)', type='text'), className='two columns'),
        html.Div(dcc.Input(id='year-text', value='built year', type='text'), className='two columns'),

    ], className='row'),
    html.Div(id='price-prediction', style={'margin':'auto'}),


    ], style={'width':'80%', 'margin': 'auto'})

])

# line chart based on rooms and regions
@app.callback(
    Output('total value based on room', 'figure'),
    [Input('graph-dropdown', 'value')])
def update_room_total_value_graph(value):
    fig = create_line_graph(value, 'total_value')
    return fig

@app.callback(
    Output('unit price based on room', 'figure'),
    [Input('graph-dropdown', 'value')])
def update_room_unit_price_graph(value):
    fig = create_line_graph(value, 'unit_price')
    return fig

@app.callback(
    Output('change rate based on room', 'figure'),
    [Input('graph-dropdown', 'value')])
def update_room_change_rate_graph(value):
    fig = create_line_graph(value, 'change_rate')
    return fig


def create_line_graph(dist, value):
    """
    create plotly figure based on district
    :param dist: list of str, regions collected from dropdown
    :param value: str, column we want to explore in dataframe ('total_value', 'unit_price', 'change_rate')
    :return: plotly figure
    """
    if len(dist) != 0:
        dff = pd.concat([df[df['Geo Local Area'] == i] for i in dist])
    else:
        dff = df.copy()
    bedroom_df = dff.groupby(['bedroom']).agg(
        {'total_value': ['mean', 'count'], 'unit_price': 'mean', 'change_rate': 'mean'})
    bathroom_df = dff.groupby(['bathroom']).agg(
        {'total_value': ['mean', 'count'], 'unit_price': 'mean', 'change_rate': 'mean'})
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=bedroom_df.index, y=bedroom_df[value]['mean'], mode='lines+markers',
                   name='# of bedrooms'))
    fig.add_trace(
        go.Scatter(x=bathroom_df.index, y=bathroom_df[value]['mean'], mode='lines+markers',
                   name='# of bathrooms'))
    if value == 'total_value':
        fig.update_layout(title_text='Building price based on rooms')
    elif value == 'unit_price':
        fig.update_layout(title_text='Unit price based on rooms')
    elif value == 'change_rate':
        fig.update_layout(title_text='Price Change rate based on rooms')
    fig.update_layout(legend_orientation="h")
    return fig

# year based graph
@app.callback(
Output('total value based on year', 'figure'),
    [Input('graph-dropdown', 'value')])
def update_year_total_value_graph(value):
    """
    create plotly figure based on regions and year
    :param value: list of str, regions collected from dropdown
    :return: plotly figure
    """
    if len(value) != 0:
        dff = pd.concat([df[df['Geo Local Area'] == i] for i in value])
    else:
        dff = df.copy()
    year_df = dff.groupby(['binned_year'])[['total_value', 'unit_price', 'change_rate']].mean()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=year_df.index, y=year_df['total_value'], mode='lines+markers',))
    fig.update_layout(title_text='Building price based on built year', xaxis_title="built year")
    return fig

@app.callback(
Output('unit price based on year', 'figure'),
    [Input('graph-dropdown', 'value')])
def update_year_unit_price_graph(value):
    if len(value) != 0:
        dff = pd.concat([df[df['Geo Local Area'] == i] for i in value])
    else:
        dff = df.copy()
    year_df = dff.groupby(['binned_year'])[['total_value', 'unit_price', 'change_rate']].mean()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=year_df.index, y=year_df['unit_price'], mode='lines+markers',))
    fig.update_layout(title_text='Unit price based on built year', xaxis_title="built year")
    return fig

@app.callback(
Output('change rate based on year', 'figure'),
    [Input('graph-dropdown', 'value')])
def update_year_change_rate_graph(value):
    if len(value) != 0:
        dff = pd.concat([df[df['Geo Local Area'] == i] for i in value])
    else:
        dff = df.copy()
    year_df = dff.groupby(['binned_year'])[['total_value', 'unit_price', 'change_rate']].mean()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=year_df.index, y=year_df['change_rate'], mode='lines+markers',))
    fig.update_layout(title_text='Price change rate based on built year', xaxis_title="built year")
    return fig

# area based graph
@app.callback(
Output('total value based on area', 'figure'),
    [Input('graph-dropdown', 'value')])
def update_area_total_value_graph(value):
    """
    create plotly figure based on regions and area
    :param value: list of str, regions collected from dropdown
    :return: plotly figure
    """
    if len(value) != 0:
        dff = pd.concat([df[df['Geo Local Area'] == i] for i in value])
    else:
        dff = df.copy()
    area_df = dff.groupby(['binned_area'])[['total_value', 'unit_price', 'change_rate']].mean()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=area_df.index, y=area_df['total_value'], mode='lines+markers',))
    fig.update_layout(title_text='Building price based on built area')
    return fig

@app.callback(
Output('unit price based on area', 'figure'),
    [Input('graph-dropdown', 'value')])
def update_area_unit_price_graph(value):
    if len(value) != 0:
        dff = pd.concat([df[df['Geo Local Area'] == i] for i in value])
    else:
        dff = df.copy()
    area_df = dff.groupby(['binned_area'])[['total_value', 'unit_price', 'change_rate']].mean()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=area_df.index, y=area_df['unit_price'], mode='lines+markers',))
    fig.update_layout(title_text='Unit price based on built area')
    return fig

@app.callback(
Output('change rate based on area', 'figure'),
    [Input('graph-dropdown', 'value')])
def update_area_change_rate_graph(value):
    if len(value) != 0:
        dff = pd.concat([df[df['Geo Local Area'] == i] for i in value])
    else:
        dff = df.copy()
    area_df = dff.groupby(['binned_area'])[['total_value', 'unit_price', 'change_rate']].mean()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=area_df.index, y=area_df['change_rate'], mode='lines+markers',))
    fig.update_layout(title_text='Price change rate based on built area')
    return fig


# using linear regression model to predict house price
@app.callback(
    Output('price-prediction', 'children'),
    [Input('prediction region', 'value'), Input('area-text', 'value'), Input('bedroom-text', 'value'),
     Input('bathroom-text', 'value'), Input('garage-text', 'value'), Input('year-text', 'value'),])
def lr(region, area, bedr, bathr, gar, year):
    if region is not None:
        model = load(f'{region}.joblib')
        if area.isdigit() and bathr.isdigit() and bedr.isdigit() and year.isdigit() and gar.isdigit():
            input = pd.DataFrame([[int(area), int(bathr), int(bedr), int(year), int(gar)]])
            return f'Predicted price: $ {int(model.predict(input)[0])}'

if __name__ == '__main__':
    app.run_server(debug=True)




