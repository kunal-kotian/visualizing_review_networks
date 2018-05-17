# Create an interactive app to visualize a network of restaurants
# Kunal Kotian, Sooraj Subrahmannian
# May 15, 2018

from dash import Dash
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc
from os import environ
import pickle

# load required pre-processed data
with open('./data/df_final_urls_updated.pkl', 'rb') as f:
    df = pickle.load(f)
with open('./data/thresh_to_graph.pkl','rb') as f:
    thresh_to_graph = pickle.load(f)
with open('./data/thresh_to_XnYn.pkl','rb') as f:
    thresh_to_XnYn = pickle.load(f)
with open('./data/thresh_to_pos.pkl','rb') as f:
    thresh_to_pos = pickle.load(f)


app = Dash('restaurant_network')
server = app.server

if 'DYNO' in environ:
    app.scripts.append_script({
        'external_url': 'https://cdn.rawgit.com/chriddyp/ca0d8f02a1659981a0ea7f013a378bbd/raw/e79f3f789517deec58f41251f7dbb6bee72c44ab/plotly_ga.js'
    })
    
BACKGROUND = 'rgb(230, 230, 230)'

COLORSCALE = [ [0, "rgb(244,236,21)"], [0.3, "rgb(249,210,41)"], [0.4, "rgb(134,191,118)"],
                [0.5, "rgb(37,180,167)"], [0.65, "rgb(17,123,215)"], [1, "rgb(54,50,153)"] ]

# arbitrary threshold for deciding whether 2 observations are 'similar' or not
threshold_all = [0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62]
def th_mark(x):
    if x==min(threshold_all):
        return 'Low'
    elif x==max(threshold_all):
        return 'High'
    else:
        return ''
    
threshold_mark = {str(th):th_mark(th) for th in threshold_all}

def update_slider_mark(slider_mark, font_size):
    # update display style of position markers for the slider
    slider_mark_updated = {}
    for position in slider_mark:
        slider_mark_updated[position] = {
            'label': slider_mark[position],
            'style': {'fontSize':font_size, 'font-family': 'Arial'}
        }
    return slider_mark_updated

threshold_mark_updated = update_slider_mark(threshold_mark, 15)


# CREATE THE DASH APP:

# figure data is the data object we pass into figure function 
# molecules will be the selected business
# change this function for our needs
def add_markers(selected_threshold, df, molecules, plot_type = 'scatter' ):
    indices = []
    rest_data = df
    for m in molecules:
        # this is the text attribute of data object 
        hover_text = rest_data.NAME.tolist()
        for i in range(len(hover_text)):
            if m == hover_text[i]:
                indices.append(i)

    trace_markers = []
    for point_number in indices:
        trace = dict(
            x = [rest_data.loc[point_number,'Xn']],
            y = [rest_data.loc[point_number,'Yn']],
            marker = dict(
                color = 'rgb(102, 255, 51)',
                size = 20,
                opacity = 0.6,
                symbol = 'cross'),
                hoverinfo=None,
                showlegend=False,
            type = plot_type
        )
        trace_markers.append(trace)  
        print(trace_markers)
    Xn_strip, Yn_strip, Xn_notstrip, Yn_notstrip = thresh_to_XnYn[selected_threshold]
    trace_nodes1 =[]
    trace_nodes2 = []
    # define a trace for plotly
    trace_nodes1 = dict(type='scatter', 
                        x=Xn_strip, 
                        y=Yn_strip,
                        mode='markers',
                        marker=dict(symbol='dot', 
                                    size=10, color='rgb(255,0,0)'),
                        name='On The Strip',
                        showlegend=True, 
                        text = [],
                        hoverinfo='text',
                        visible=True)
    trace_nodes2 = dict(type='scatter', 
                        x=Xn_notstrip, 
                        y=Yn_notstrip,
                        mode='markers',
                        marker=dict(symbol='dot', 
                                    size=10, color='rgb(0, 0, 255)'),
                        name='Not on The Strip',
                        showlegend=True, 
                        text = [],
                        hoverinfo='text',
                        visible=True)        
    
    # record the coordinates of the ends of edges
    Xe = []
    Ye = []
    G = thresh_to_graph[selected_threshold]
    for e in G.edges():
        pos = thresh_to_pos[selected_threshold]
        Xe.extend([pos[e[0]][0], pos[e[1]][0], None])
        Ye.extend([pos[e[0]][1], pos[e[1]][1], None])

    # trace_edges defines the graph edges as a trace of type scatter (line)
    trace_edges = []
    trace_edges=dict(type='scatter',
                     mode='lines',
                     x=Xe,
                     y=Ye,
                     line=dict(width=0.1, color='rgb(51, 51, 51)'),
                     hoverinfo='none', showlegend=False)
    traces = [trace_nodes1] + [trace_nodes2] + [trace_edges]+trace_markers
    return traces

# change this function 
def scatter_plot_3d(selected_threshold, xlabel='', ylabel='', plot_type='scatter', markers=[]):
    df_for_plot = df.copy()
    df_for_plot = df_for_plot.loc[(df_for_plot.Xn.isnull() == False) & (df_for_plot.Yn.isnull() == False) & (df_for_plot.threshold == selected_threshold), :]
    df_for_plot = df_for_plot.reset_index().drop('index',axis=1)
    x= df_for_plot['Xn'] 
    y= df_for_plot['Yn']

    def axis_template_2d(title):
        return dict(
            xgap = 10, ygap = 10,
            backgroundcolor = BACKGROUND,
            gridcolor = 'rgb(255, 255, 255)',
            title = title,
            zerolinecolor = 'rgb(255, 255, 255)',
            showspikes=True,
            spikethickness=1,
            spikedash='solid',
            spikemode='across',
            showticklabels = False
        )
    # change data
    data = [dict(
        x = x,
        y = y,
        mode = 'markers',
        text = df_for_plot['NAME'],
        type = plot_type,
    ) ]
    # change layour
    layout = dict(
        font = dict( family = 'Raleway' ),
        hovermode = 'closest',
        hoverdistance = 15,
        margin = dict( r=0, t=0, l=0, b=0 ),
        showlegend = True,
        legend=dict(x=0, y=1)
    )
    # change scatter
    if plot_type in ['scatter']:
        layout['xaxis'] = axis_template_2d(xlabel)
        layout['yaxis'] = axis_template_2d(ylabel)
        layout['plot_bgcolor'] = 'rgba(0,0,0,0)'
        layout['paper_bgcolor'] = 'rgba(0,0,0,0)'
        
    data = add_markers(selected_threshold, df_for_plot, markers, plot_type = plot_type )

    return dict(data=data, layout=layout)

selected_threshold = threshold_all[0]
FIGURE = scatter_plot_3d(selected_threshold)
df_for_plot = df.copy()
df_for_plot = df_for_plot.loc[(df_for_plot.Xn.isnull() == False) & (df_for_plot.Yn.isnull() == False) & (df_for_plot.threshold == selected_threshold), :]
df_for_plot = df_for_plot.reset_index().drop('index',axis=1)
STARTING_DRUG = df_for_plot.loc[0,'NAME']
# DRUG_DESCRIPTION = df_for_plot.loc[df_for_plot['NAME'] == STARTING_DRUG]['DESC'].iloc[0]
STAR_RATING =df_for_plot.loc[df_for_plot['NAME'] == STARTING_DRUG]['stars'].iloc[0]
DRUG_IMG = df_for_plot.loc[df_for_plot['NAME'] == STARTING_DRUG]['IMG_URL'].iloc[0]
topic1 = df_for_plot.loc[df_for_plot['NAME'] == STARTING_DRUG]['topic1_for_disp'].iloc[0]
topic2 = df_for_plot.loc[df_for_plot['NAME'] == STARTING_DRUG]['topic2_for_disp'].iloc[0]
topic3 = df_for_plot.loc[df_for_plot['NAME'] == STARTING_DRUG]['topic3_for_disp'].iloc[0]

app.layout = html.Div([
    html.H2('Network of Restaurants based on User Reviews',style ={'textAlign':'center'}),
    # Row 1: Header and Intro text
    html.Div([
        html.Div([
            html.Div([
            ], style={'margin-left': '10px'}),
            dcc.Dropdown(id='chem_dropdown',
                        multi=True,
                        value=[ STARTING_DRUG ],
                        options=[{'label': i, 'value': i} for i in df_for_plot['NAME'].tolist()]),
            ], className='twelve columns' )

    ], className='row' ),

    # Row 2: Hover Panel and Graph
    html.Div([
        html.Div([
        html.Div([
            html.Br(),
            
            html.Img(id='chem_img', src=DRUG_IMG,style=dict(width='150px',height='150px')),

            html.Br(),
            
            html.A(STARTING_DRUG,
                  id='chem_name',
                  href="https://www.drugbank.ca/drugs/DB01002",
                  target="_blank"),

            html.Br(),
            ### Star rating
            html.Div([
                    html.Div(html.B("Average Rating")),
                    html.Div(STAR_RATING, id='star_rating',style={'marginTop':'0.005em'})]),

            html.Br(),
            ### Words in a topictopic1_for_disp
            html.Div([
                    html.Div(html.B("Major Topics in Reviews")),
                    html.Div(topic1, id='topic1',style={'marginTop':'0.005em'}),
                    html.Div(topic2, id='topic2',style={'marginTop':'0.005em'}),
                    html.Div(topic3, id='topic3',style={'marginTop':'0.005em'})]), 

        ],className ='row',style=dict(height = '450px')),
        html.Br(),
        html.Div([
        html.Div([
            html.Div(html.B('Similarity Cutoff'),style=dict( maxHeight='200px', fontSize='20px',  marginLeft=-12)),
            dcc.Slider(
                id='threshold-slider',
                min=min(threshold_all),
                max=max(threshold_all),
                value=selected_threshold,
                step=None,
                marks=threshold_mark_updated),
                ], style={'width': '100%','marginBottom': 0, 'marginTop': 0, 'marginLeft':17, 'marginRight':'auto',
                  'fontSize':12},className='three columns')
    ],className='row')
        
        ], className='three columns', style=dict(height='300px')),

        html.Div([
            dcc.Graph(id='clickable-graph',
                      style=dict(width='700px',height='550px'),
                      hoverData=dict(points=[dict(pointNumber=0)] ),
                      figure=FIGURE ),

        ], className='nine columns', style=dict(textAlign='center')),
    ], className='row' ),

], className='container', style={'width':'85%', 'marginBottom': 0, 'marginTop': 0, 'marginLeft':'auto', 'marginRight':'auto'})


@app.callback(
    Output('clickable-graph', 'figure'),
    [Input('chem_dropdown', 'value'), Input('threshold-slider', 'value')])
def highlight_molecule(chem_dropdown_values, selected_threshold):
    return scatter_plot_3d(selected_threshold=selected_threshold, 
                           markers = chem_dropdown_values, plot_type = 'scatter')



def dfRowFromHover(hoverData,selected_threshold,figure):
    ''' Returns row for hover point as a Pandas Series '''
    if hoverData is not None:
        print(hoverData)
        if 'points' in hoverData:
            firstPoint = hoverData['points'][0]
            if 'x' in firstPoint:
                point_number = firstPoint['pointNumber']
                xdata = firstPoint['x']
                if xdata in figure['data'][0]['x']:
                    strip_flag = True
                else:
                    strip_flag = False
#                 molecule_name = str(FIGURE['data'][0]['text'][point_number]).strip()
                # belong to strip 
                df_for_plot = df.copy()
                df_for_plot = df_for_plot.loc[(df_for_plot.Xn.isnull() == False) & (df_for_plot.Yn.isnull() == False) & (df_for_plot.threshold == selected_threshold)&(df_for_plot.is_strip==strip_flag), :]
                df_for_plot = df_for_plot.reset_index().drop('index',axis=1)
                try:
                    molecule_name = df_for_plot.loc[point_number,'NAME']
                    return df_for_plot.loc[df_for_plot['NAME'] == molecule_name]
                except:
                    return pd.Series()
    return pd.Series()


@app.callback(
    Output('chem_dropdown', 'options'),
    [Input('threshold-slider', 'value')])
def set_dropdown_options(selected_threshold):
    df_for_plot = df.copy()
    df_for_plot = df_for_plot.loc[(df_for_plot.Xn.isnull() == False) & (df_for_plot.Yn.isnull() == False) & (df_for_plot.threshold == selected_threshold), :]
    df_for_plot = df_for_plot.reset_index().drop('index',axis=1)
    return [{'label': i, 'value': i} for i in df_for_plot['NAME'].tolist()]

@app.callback(
    Output('chem_dropdown', 'value'),
    [Input('chem_dropdown', 'options')])
def set_dropdown_value(available_options):
    return available_options[0]['value']

@app.callback(
    Output('chem_name', 'children'),
    [Input('clickable-graph', 'hoverData'),Input('threshold-slider', 'value'),Input('clickable-graph', 'figure')])
def return_molecule_name(hoverData,selected_threshold,figure):
    if hoverData is not None:
        if 'points' in hoverData:
            firstPoint = hoverData['points'][0]
            
            if 'x' in firstPoint:
                point_number = firstPoint['pointNumber']
                xdata = firstPoint['x']
                if xdata in figure['data'][0]['x']:
                    strip_flag = True
                else:
                    strip_flag = False
                df_for_plot = df.copy()
                df_for_plot = df_for_plot.loc[(df_for_plot.Xn.isnull() == False) & (df_for_plot.Yn.isnull() == False) & (df_for_plot.threshold == selected_threshold) & (df_for_plot.is_strip==strip_flag), :]
                df_for_plot = df_for_plot.reset_index().drop('index',axis=1)
                try:
                    molecule_name = df_for_plot.loc[point_number,'NAME']
                except KeyError:
                    return None
    
                return molecule_name


@app.callback(
    Output('chem_name', 'href'),
    [Input('clickable-graph', 'hoverData'),Input('threshold-slider', 'value'),Input('clickable-graph', 'figure')])
def return_href(hoverData,selected_threshold,figure):
    row = dfRowFromHover(hoverData,selected_threshold,figure)
    if row.empty:
        return
    datasheet_link = row['PAGE'].iloc[0]
    return datasheet_link


@app.callback(
    Output('chem_img', 'src'),
    [Input('clickable-graph', 'hoverData'),Input('threshold-slider', 'value'),Input('clickable-graph', 'figure')])
def display_image(hoverData,selected_threshold,figure):
    row = dfRowFromHover(hoverData,selected_threshold,figure)
    if row.empty:
        return
    img_src = row['IMG_URL'].iloc[0]
    return img_src

@app.callback(
    Output('star_rating', 'children'),
    [Input('clickable-graph', 'hoverData'),Input('threshold-slider', 'value'),Input('clickable-graph', 'figure')])
def display_star(hoverData,selected_threshold,figure):
    row = dfRowFromHover(hoverData,selected_threshold,figure)
    if row.empty:
        return
    star = row['stars'].iloc[0]
    return star

@app.callback(
    Output('topic1', 'children'),
    [Input('clickable-graph', 'hoverData'),Input('threshold-slider', 'value'),Input('clickable-graph', 'figure')])
def display_topic1(hoverData,selected_threshold,figure):
    row = dfRowFromHover(hoverData,selected_threshold,figure)
    if row.empty:
        return
    topic1 = row['topic1_for_disp'].iloc[0]
    return topic1

@app.callback(
    Output('topic2', 'children'),
    [Input('clickable-graph', 'hoverData'),Input('threshold-slider', 'value'),Input('clickable-graph', 'figure')])
def display_topic2(hoverData,selected_threshold,figure):
    row = dfRowFromHover(hoverData,selected_threshold,figure)
    if row.empty:
        return
    topic2 = row['topic2_for_disp'].iloc[0]
    return topic2

@app.callback(
    Output('topic3', 'children'),
    [Input('clickable-graph', 'hoverData'),Input('threshold-slider', 'value'),Input('clickable-graph', 'figure')])
def display_topic3(hoverData,selected_threshold,figure):
    row = dfRowFromHover(hoverData,selected_threshold,figure)
    if row.empty:
        return
    topic3 = row['topic3_for_disp'].iloc[0]
    return topic3

external_css = ["https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.min.css",
                "//fonts.googleapis.com/css?family=Raleway:400,300,600",
                "//fonts.googleapis.com/css?family=Dosis:Medium",
                "https://rawgit.com/smsubrahmannian/smsubrahmannian.github.io/master/review_networks/custom_container.css"]


for css in external_css:
    app.css.append_css({"external_url": css})


if __name__ == '__main__':
    app.run_server(port=8052,host='0.0.0.0',debug=True)


