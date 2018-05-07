import pandas as pd
import numpy as np
from scipy.stats import entropy
from sklearn.metrics.pairwise import pairwise_distances
import networkx as nx

import plotly.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode,  iplot, plot
import plotly.graph_objs as go
import plotly.offline as offline

import dash
import dash_core_components as dcc
import dash_html_components as html


# Define helper functions
def jensen_shannon(_P, _Q):
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

def th_mark(x):
    if x==np.min(threshold_all):
        return 'Low'
    elif x==np.max(threshold_all):
        return 'High'
    else:
        return ''

def create_graph(adj):
    # input: adjaccency matrix
    # returns a graph with the isolates removed
    G = nx.from_numpy_matrix(adj)
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)
    return G

def update_slider_mark(slider_mark, font_size):
    # update display style of position markers for the slider
    slider_mark_updated = {}
    for position in slider_mark:
        slider_mark_updated[position] = {
            'label': slider_mark[position],
            'style': {'fontSize':font_size, 'font-family': 'Arial'}
        }
    return slider_mark_updated


# Load Data
df = pd.read_feather('df_final_doc2topics.feather')
df.drop(['business_id'], axis=1, inplace=True)
data = df.drop(['name', 'is_strip', 'stars'], axis=1).as_matrix()

# find the topic most closely related to each restaurant
topic_closest_ind = np.argmax(data, axis=1)
topic_names_ord = ['Cost & Quality', 'Bars', 'Casino Hotel', 'Fine Dining', 'Asian', 'Pizza', 'Steakhouse', 
                   'Italian', 'Coffee Shop', 'High Customer Satisfaction', 'Night Club', 'Wait Time', 'Mexican', 
                   'Lunch', 'Sushi', 'Fast Food', 'Breakfast', 'Low Customer Satisfaction']
# names of topics most closely related to each restaurant (ordered by the order of restaurants in df)
topic_closest = [topic_names_ord[ind] for ind in topic_closest_ind]

# Pairwise Jensen-Shannon distance between each pair of observations based on the 18 topic-probabilities
pairwise_dist = pairwise_distances(X=data, metric=jensen_shannon)

# predetermined 'k' values for the Fruchterman-Reingold layout
threshold2k ={
    0.55: 0.7,
    0.56: 0.9,
    0.57: 0.3,
    0.58: 5,
    0.59: 2,
    0.6: 5,
    0.61: 5,
    0.62: 5
}

# arbitrary threshold for deciding whether 2 observations are 'similar' or not
threshold_all = [0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62]
threshold_mark = {str(th):th_mark(th) for th in threshold_all}
adjacency = [np.where(pairwise_dist > threshold, 1, 0) for threshold in threshold_all]

# map threshold value to adjacency matrix
thresh_to_adj = {thresh: adj for thresh, adj in zip(threshold_all, adjacency)}

# map threshold value to graph
thresh_to_graph = {thresh: create_graph(adj) for thresh, adj in zip(threshold_all, adjacency)}

# extract node positions
fruchterman_iter = 1000
# map threshold values to positions of nodes
thresh_to_pos = {}
for thresh in thresh_to_graph:
    graph = nx.fruchterman_reingold_layout(thresh_to_graph[thresh], k = threshold2k[thresh], iterations=fruchterman_iter)
    thresh_to_pos[thresh] = graph

thresh_to_XnYn = {}
for thresh in thresh_to_pos:
    pos = thresh_to_pos[thresh]
    # define lists of node coordinates
    Xn_strip = [pos[k][0] for k in sorted(pos.keys()) if k in df.index[df.is_strip == True]]
    Yn_strip = [pos[k][1] for k in sorted(pos.keys()) if k in df.index[df.is_strip == True]]
    Xn_notstrip = [pos[k][0] for k in sorted(pos.keys()) if k in df.index[df.is_strip == False]]
    Yn_notstrip = [pos[k][1] for k in sorted(pos.keys()) if k in df.index[df.is_strip == False]]
    thresh_to_XnYn[thresh] = (Xn_strip, Yn_strip, Xn_notstrip, Yn_notstrip)

threshold_mark_updated = update_slider_mark(threshold_mark, 15)


# Create and run the Dash app
app = dash.Dash()

app.layout = html.Div([
    html.Div([
    dcc.Graph(id='graph-with-slider')
    ],style={'marginLeft':140, 'marginRight':'auto'}),
    html.Div([
    html.H2('Similarity Cutoff'),
    dcc.Slider(
        id='threshold-slider',
        min=min(threshold_all),
        max=max(threshold_all),
        value=threshold_all[int(np.floor(len(threshold_all)/2))],
        step=None,
        marks=threshold_mark_updated
    )
    ], style={'width': '47%','marginBottom': 0, 'marginTop': 0, 'marginLeft':'auto', 'marginRight':'auto',
              'fontSize':12, 'font-family': 'Arial'})
])


@app.callback(dash.dependencies.Output('graph-with-slider', 'figure'),
              [dash.dependencies.Input('threshold-slider', 'value')])
def update_figure(selected_threshold):
    # Work to be done: subset the Xn and Yn for given threshold
    Xn_strip, Yn_strip, Xn_notstrip, Yn_notstrip = thresh_to_XnYn[selected_threshold]

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
    
    # Add labels for nodes
    for index, row in df.iterrows():
        # for strip restaurants
        if index in df.index[df.is_strip == True]:
            node_info = df.name.iloc[index] + ', ' + str(df.stars.iloc[index]) + '/5, Related to: ' + topic_closest[index]
            trace_nodes1['text'].append(node_info)
        # for non strip restaurants
        if index in df.index[df.is_strip == False]:
            node_info = df.name.iloc[index] + ', ' + str(df.stars.iloc[index]) + '/5, Related to: ' + topic_closest[index]
            trace_nodes2['text'].append(node_info)
        
    
    # record the coordinates of the ends of edges
    Xe = []
    Ye = []
    G = thresh_to_graph[selected_threshold]
    for e in G.edges():
        pos = thresh_to_pos[selected_threshold]
        Xe.extend([pos[e[0]][0], pos[e[1]][0], None])
        Ye.extend([pos[e[0]][1], pos[e[1]][1], None])

    # trace_edges defines the graph edges as a trace of type scatter (line)
    trace_edges=dict(type='scatter',
                     mode='lines',
                     x=Xe,
                     y=Ye,
                     line=dict(width=0.1, color='rgb(51, 51, 51)'),
                     hoverinfo='none', showlegend=False)

    axis=dict(showline=False, # hide axis line, grid, ticklabels and  title
              zeroline=False,
              showgrid=False,
              showticklabels=False,
              title='' 
              )
    layout=dict(title= 'Network of Restaurants based on User Reviews',  
                font= dict(family='Arial', size=17),
                width=1000,
                height=800,
                autosize=False,
                showlegend=True,
                xaxis=axis,
                yaxis=axis,
                margin=dict(
                l=40,
                r=40,
                b=10,
                t=50,
                pad=0,
       
        ),
                hovermode='closest',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
        )


    return {
        'data': [trace_edges, trace_nodes1, trace_nodes2],
        'layout': layout}


if __name__ == '__main__':
    app.run_server()

