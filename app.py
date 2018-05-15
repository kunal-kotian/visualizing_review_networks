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
def get_relevant_words(vis,lam=0.3,topn=10):
    a = vis.topic_info
    a['finalscore'] = a['logprob']*lam+(1-lam)*a['loglift']
    a = a.loc[:,['Category','Term','finalscore']].groupby(['Category'])\
    .apply(lambda x: x.sort_values(by='finalscore',ascending=False).head(topn))
    a = a.loc[:,'Term'].reset_index().loc[:,['Category','Term']]
    a = a[a['Category']!='Default']
    a = a.to_dict('split')['data']
    d ={}
    for k,v in a: 
        if k not in d.keys():
            d[k] =set()
            d[k].add(v)
        else:
            d[k].add(v)
    finalData = pd.DataFrame([],columns=['Topic','words with Relevance'])
    finalData['Topic']=d.keys()
    finalData['words with Relevance']=d.values()
    return finalData

def jensen_shannon(_P, _Q):
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

def get_top_n_words_list(num_topics, vis, lam=0.6, topn=5):
    """returns a sorted list of top n words, where the list follows the order Topic 1, ..., Topic n.
    Each element of the list is a string composed of a list of the top n words
    num_topics: number of topics
    vis: pyLDAvis object
    lam: relevance value
    topn: number of topics
    """
    topic_ids_ordered = ['Topic' + str(num) for num in range(1, num_topics + 1)]
    top_topic_words_df = get_relevant_words(vis, lam, topn)
    top_topic_words_df.set_index('Topic', drop=True, inplace=True)
    top_topic_words = [top_topic_words_df.loc[topic_id]['words with Relevance'] for topic_id in topic_ids_ordered]
    top_topic_words_display = [', '.join(words) for words in top_topic_words]
    return top_topic_words_display
    
def rel_mark(x, relevance_all):
    if x==np.min(relevance_all):
        return 'Rare'
    elif x==np.max(relevance_all):
        return 'Frequent'
    else:
        return ''

def th_mark(x, threshold_all):
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

def get_topic_size_ord(num_topics, topic2tokenpercent):
    """returns a list of token percentages, following the order Topic 1, ..., Topic n.
    topic2tokenpercent: dict linking topic names to token percentages
    """
    topic_ids_ordered = ['Topic' + str(num) for num in range(1, num_topics + 1)]
    topic_size_ord = [topic2tokenpercent[topic_id] for topic_id in topic_ids_ordered]
    return topic_size_ord

# Load Data
df = pd.read_csv('topic2word.csv')
df.drop(['Unnamed: 0'], axis=1, inplace=True)
vis = pd.read_pickle('vis.pkl')

topic2tokenpercent = {'Topic1': 9.1,
                     'Topic2': 4.7,
                     'Topic3': 2.7,
                     'Topic4': 7.3,
                     'Topic5': 4.8,
                     'Topic6': 2.1,
                     'Topic7': 1.4,
                     'Topic8': 3.1,
                     'Topic9': 1.5,
                     'Topic10': 9.8,
                     'Topic11': 6.5,
                     'Topic12': 9.5,
                     'Topic13': 2.6,
                     'Topic14': 10.4,
                     'Topic15': 3,
                     'Topic16': 2.2,
                     'Topic17': 8.4,
                     'Topic18':10.7}

topic_name_mapper = {
    'Topic1': 'Cost & Quality',
    'Topic2': 'Bars',
    'Topic3': 'Casino Hotel',
    'Topic4': 'Fine Dining',
    'Topic5': 'Asian',
    'Topic6': 'Pizza',
    'Topic7': 'Steakhouse',
    'Topic8': 'Italian',
    'Topic9': 'Coffee Shop',
    'Topic10': 'High Customer Satisfaction',
    'Topic11': 'Night Club',
    'Topic12': 'Wait Time',
    'Topic13': 'Mexican',
    'Topic14': 'Lunch',
    'Topic15': 'Sushi',
    'Topic16': 'Fast Food',
    'Topic17': 'Breakfast',
    'Topic18': 'Low Customer Satisfaction'
}


# Create the Network
data = df.as_matrix()
# Pairwise Jensen-Shannon distance between each pair of observations based on the 18 topic-probabilities
pairwise_dist = pairwise_distances(X=data, metric=jensen_shannon)
# arbitrary threshold for deciding whether 2 observations are 'similar' or not
threshold_all = [0.1, 0.11, 0.14, 0.18, 0.19, 0.2, 0.23, 0.25]
threshold_mark = {str(th):th_mark(th, threshold_all) for th in threshold_all}
threshold_mark_updated = update_slider_mark(threshold_mark, 15)

relevance_all = [0, 0.25, 0.5, 0.75, 1]
relevance_mark = {str(th):rel_mark(th, relevance_all) for th in relevance_all}
relevance_mark_updated = update_slider_mark(relevance_mark, 15)

adjacency = [np.where(pairwise_dist > threshold, 1, 0) for threshold in threshold_all]
# map threshold value to adjacency matrix
thresh_to_adj = {thresh: adj for thresh, adj in zip(threshold_all, adjacency)}
# map threshold value to graph
thresh_to_graph = {thresh: create_graph(adj) for thresh, adj in zip(threshold_all, adjacency)}

# extract node positions
fruchterman_k = 5
fruchterman_iter = 1000

# predetermined 'k' values for the Fruchterman-Reingold layout
threshold2k ={
   0.10: 0.7,
   0.11: 10,
   0.14: 0.9,
   0.18: 20,
   0.19: 10,
   0.20: 4,
   0.23: 10,
   0.25: 10
}

# map threshold values to positions of nodes
thresh_to_pos = {}
for thresh in thresh_to_graph:
    graph = nx.fruchterman_reingold_layout(thresh_to_graph[thresh], k = threshold2k[thresh], iterations=fruchterman_iter)
    thresh_to_pos[thresh] = graph

thresh_to_XnYn = {}
for thresh in thresh_to_pos:
    pos = thresh_to_pos[thresh]
    # define lists of node coordinates
    Xn = [pos[k][0] for k in sorted(pos.keys())]
    Yn = [pos[k][1] for k in sorted(pos.keys())]
    thresh_to_XnYn[thresh] = (Xn, Yn)


# Create and run the Dash app
get_topic_size_ord(18, topic2tokenpercent)

app = dash.Dash()

app.layout = html.Div([
    html.Div([
    dcc.Graph(id='graph-with-slider')
        ], style={'marginLeft':140, 'marginRight':'auto'}),
    html.Div([
    html.H2('Similarity Cutoff'),
    dcc.Slider(
        id='threshold-slider',
        min=min(threshold_all),
        max=max(threshold_all),
        value=threshold_all[int(np.floor(len(threshold_all)/2))],
        step=None,
        marks=threshold_mark_updated
    ),
    ], style={'width': '47%','marginBottom': 0, 'marginTop': 0, 'marginLeft':'auto', 'marginRight':'auto',
              'fontSize':12, 'font-family': 'Arial'}
    ),
    html.Div([
    html.H2('Characteristic Words'), 
    dcc.Slider(
        id='relevance-slider',
        min=min(relevance_all),
        max=max(relevance_all),
        value=relevance_all[int(np.floor(len(relevance_all)/2))],
        step=None,
        marks=relevance_mark_updated
    )], style={'width': '47%','marginBottom': 0, 'marginTop': 50, 'marginLeft':'auto', 'marginRight':'auto',
              'fontSize':12, 'font-family': 'Arial'})  
    ])


@app.callback(
    dash.dependencies.Output('graph-with-slider', 'figure'),
    [dash.dependencies.Input('threshold-slider', 'value'), 
     dash.dependencies.Input('relevance-slider', 'value')])
def update_figure(selected_threshold, selected_relevance):

    Xn, Yn = thresh_to_XnYn[selected_threshold]
    
    node_sizes = np.array(get_topic_size_ord(18, topic2tokenpercent)) * 7
    
    # define a trace for plotly
    trace_nodes0 = dict(type='scatter', 
                        x=Xn, 
                        y=Yn,
                        mode='markers+text',
                        marker=dict(symbol='dot', size=node_sizes, color='rgb(255, 128, 0)'),
                        showlegend=False,
                        text = [],
                        textposition='bottom',
                        textfont=dict(
                            family='sans serif',
                            size=14),
                        hoverinfo='skip',
                        visible=True)
    
    # Add labels for nodes
    for index, row in df.iterrows():
        node_info = get_top_n_words_list(num_topics=18, vis=vis, lam=selected_relevance, topn=3)[index]
        trace_nodes0['text'].append(node_info)


    # Create dummy nodes for displaying topic names
    # define a trace for plotly
    trace_nodes1 = dict(type='scatter', 
                        x=Xn, 
                        y=Yn,
                        mode='markers',
                        marker=dict(symbol='dot', size=node_sizes, color='rgb(255, 128, 0)'),
                        showlegend=False,
                        text = [],
                        textposition='bottom',
                        textfont=dict(
                            family='sans serif',
                            size=14),
                        hoverinfo='text',
                        visible=True)
    
    # Add labels for nodes
    for index, row in df.iterrows():
        node_topic = topic_name_mapper['Topic' + str(index + 1)]
        trace_nodes1['text'].append(node_topic)

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
    layout=dict(title= 'Network of Topics based on User Reviews',  
                font= dict(family='Arial', size=17, textposition='center'),
                            width=750,
                            height=750,
                            autosize=False,
                            showlegend=True,
                            xaxis=axis,
                            yaxis=axis,
                            margin=dict(
                                l=40,
                                r=40,
                                b=10,
                                t=50,
                                pad=0),
                hovermode='closest',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                
        )

    return {
        'data': [trace_edges, trace_nodes0, trace_nodes1],
        'layout': layout}


if __name__ == '__main__':
    app.run_server(host='0.0.0.0',port=8051)
