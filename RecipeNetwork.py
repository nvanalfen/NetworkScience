# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 14:57:19 2020

@author: nvana
"""

from RecipeScraper import read_master, unpack_list
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import numpy as np

def write_edgelist(f_name, edges, header=None, spacer="\t"):
    file = open(f_name, "w", encoding="utf-8")
    if not header is None:
        for i in range(len(header)):
            file.write(str(header[i]))
            if i < len(header)-1:
                file.write(spacer)
    file.write("\n")
                
    for edge in edges:
        for i in range(len(edge)):
            file.write( str(edge[i]) )
            if i < len(edge)-1:
                file.write(spacer)
        if edge != edges[-1]:
            file.write("\n")
    file.close()

def write_edgelist_from_graph(f_name, G, weight_keyword=None, header=None, spacer="\t"):
    edges = []
    for e in G.edges(data=True):
        row = [e[0], e[1]]
        if not weight_keyword is None:
            row.append(e[2][weight_keyword])
        edges.append(row)
    
    write_edgelist(f_name, edges, header=header, spacer=spacer)

# Reading ingredients from the master Excel file gives list. This turns those lists into sets
def convert_ingredients_to_sets(foods):
    new_foods = {}
    for key in foods:
        new_foods[key] = set( foods[key] )
    return new_foods

# removes the ingredients specified from each food
# Useful for eliminating trivial links from common ingredients like water, salt, sugar, etc.
def remove_specified_ingredients(foods, ingredients):
    for key in foods:
        for element in ingredients:
            if element in foods[key]:
                foods[key].remove(element)

# Replace ingredients with a broader classification (e.g. chicken, pork, beef all become meat)
def replace_parent_ingredients(foods, parent_ingredients):
    for food in foods:
        for parent in parent_ingredients:
            for child in parent_ingredients[parent]:
                if child in foods[food]:
                    foods[food].remove(child)
                    foods[food].add(parent)
    return foods

def get_parent_ingredients_to_exclude(f_name):
    df = pd.read_csv(f_name)
    exclude = {}
    
    for index in df.index:
        if df["Use"][index]:
            exclude[ df["Parent"][index] ] = unpack_list( df["Ingredients"][index] )
    
    return exclude

# Read recipes and return dict from food name to ingredients
# f_name                -> Name of the file to read from
# remove_ingredients    -> list of ingredients to remove from consideration (for common things like salt and sugar)
# parent_ingredients    -> dict from parent ingredient to list of child ingredients to consider all children as the parent
def read_ingredients(f_name, remove_ingredients=None, parent_ingredients=None):
    foods = read_master(f_name)
    foods = convert_ingredients_to_sets(foods)
    
    # Remove ingredients before checking to replace parent ingredients
    if not remove_ingredients is None:
        remove_specified_ingredients(foods, remove_ingredients)
    
    # Replace some ingredients with a broader classification
    if not parent_ingredients is None:
        foods = replace_parent_ingredients(foods, parent_ingredients)
        
    # Do one more sweep of removing ingredients in case any ingredients specified were the parent ingredient
    if not remove_ingredients is None:
        remove_specified_ingredients(foods, remove_ingredients)
        
    return foods

# NETWORK CREATION METHODS ####################################################
    
# UNDIRECTED WEIGHTED NETWORKS #####

def add_undirected_weight(source, target, foods, edges):
    weight = len( foods[source] & foods[target] )/len( foods[source] | foods[target] )
    if weight != 0:
        edges.append( (source, target, weight) )

def create_undirected_weighted_network(f_name, remove_ingredients=None, parent_ingredients_f_name=None):
    parent_ingredients = None
    if not parent_ingredients_f_name is None:
        parent_ingredients = get_parent_ingredients_to_exclude(parent_ingredients_f_name)
    
    foods = read_ingredients(f_name, remove_ingredients, parent_ingredients)
    edges = []
    keys = list( foods.keys() )
    
    # Run through the list of foods
    # Second loop starts at i+1 because this is an unweighted network and we only need each edge once
    for i in range(len(keys)):
        for j in range(len(keys))[i+1:]:
            add_undirected_weight(keys[i], keys[j], foods, edges)
    
    return foods, edges

# DIRECTED WEIGHTED NETWORKS #####

# Very much like add undirected weight, but creates an directed link from source to target
# with the weight of the proportion of ingredients from source are also in target
# rather than jsut what proportion of the sum total ingredients are shared
def add_directed_weight(source, target, foods, edges):
    weight = len( foods[source] & foods[target] )/len( foods[source] )
    if weight != 0:
        edges.append( (source, target, weight) )

def create_directed_weighted_network(f_name, remove_ingredients=None, parent_ingredients=None):
    foods = read_ingredients(f_name, remove_ingredients, parent_ingredients)
    edges = []
    
    # Run through the list of foods
    # Second loop starts at i+1 because this is an unweighted network and we only need each edge once
    for source in foods:
        for target in foods:
            if source != target:
                add_directed_weight(source, target, foods, edges)
    
    return foods, edges

# OTEHR NETWORK METHODS #####

# Given a list of edges with weights, only keep the edges with weights higher than a certain value
# weighted          -> list of weighted edges
# threshold         -> all adges with a weight above this value are kept
def weighted_above(weighted, threshold=0):
    new_weighted = []
    for edge in weighted:
        if edge[2] >= threshold:
            new_weighted.append( edge )
    return new_weighted
    
# Given a list of weighted edges, return an edge list of unweighted edges
def weighted_to_unweighted(weighted, threshold=0):
    unweighted = []
    for edge in weighted:
        if edge[2] >= threshold:
            unweighted.append( (edge[0], edge[1]) )
    return unweighted

# If the network is undirected, in_degree is used for all the degrees
def weight_distribution(nodes, edges, directed=False):
    in_degree = []
    out_degree = []
    for node in nodes:
        in_deg = 0
        out_deg = 0
        
        for edge in edges:
            if not directed and node in edge:
                in_deg += edge[-1]
            elif directed and node == edge[0]:
                out_deg += edge[-1]
            elif directed and node == edge[1]:
                in_deg += edge[-1]
        
        in_degree.append( in_deg )
        if directed:
            out_degree.append( out_deg )
    
    if directed:
        return in_degree, out_degree
    return in_degree

# Returns a sorted list of the weights and a list of the probability of having at least that weight
def cumulative_weight_distribution(degrees):
    new_deg = list(degrees)
    new_deg.sort()
    prob = []
    total = sum(new_deg)
    for i in range(len(new_deg)):
        prob.append( sum( new_deg[i:] )/total )
    
    return new_deg, prob

def edge_weight_distribution(edges):
    return [ edge[-1] for edge in edges ]

def cumulative_edge_weight_distribution(edges):
    weights = [ edge[-1] for edge in edges ]
    weights.sort()
    total = sum(weights)
    prob = []
    for i in range(len(weights)):
        prob.append( sum(weights[i:])/total )
    
    return weights, prob

# Cast our edges and nodes into a networkx graph
def to_nx_graph(nodes, edges, weighted=True):
    G = nx.Graph()
    for node in nodes:
        G.add_node(node)
    for edge in edges:
        if weighted:
            G.add_edge(edge[0], edge[1], weight=edge[2])
        else:
            G.add_edge(edge[0], edge[1])
    return G

# Returns a list or dict of ingredients and in what proportion of recipes they appear
def ingredient_frequency(foods, return_dict=False, sort=True, reverse=True):
    freq = {}
    for key in foods:
        for ingredient in foods[key]:
            if not ingredient in freq:
                freq[ingredient] = 0
            freq[ingredient] += 1
    
    for key in freq:
        freq[key] /= len(foods)
        
    if return_dict:
        return freq
    
    new_freq = []
    for key in freq:
        new_freq.append( [key, freq[key]] )
    
    if sort:
        new_freq.sort( key=lambda x:x[1], reverse=reverse )
            
    return new_freq

# Takes a table in the format returned by the functions in backboning.py and converts it to a networkx graph
# Also pass in the full list of nodes to account for isolated nodes
def graph_from_table(nodes,table):
    G = nx.Graph()
    for node in nodes:
        G.add_node(node)
    
    sources = np.unique( table["src"] )

    for src in sources:
        for trg in table[ table["src"] == src ]["trg"]:
            edge_weight = float( table[ (table["src"] == src) & (table["trg"] == trg) ]["nij"] )
            G.add_edge(src, trg, weight=edge_weight)
    
    return G

# Reads in csv from Gephi where communities have already been detected by modularity class
# Finds out how often different countries appear in different communities
def community_country_distribution(f_name, giant_only=True):
    data = pd.read_csv(f_name)
    if giant_only:
        data = data[ data["componentnumber"] == 0 ]
    
    mod_class = {}              # dict from modularity class number to another dict from country to frequency of occurances in that class
    for c in np.unique( data["modularity_class"] ):
        freq = {}
        for country in data[ data["modularity_class"] == c ]["country"]:
            if not country in freq:
                freq[country] = 0
            freq[country] += 1
        mod_class[c] = freq
    
    return mod_class

def plot(x, y, label, scatter=True):
    figure = plt.figure()
    ax = figure.add_subplot(111)
    
    for i in range(len(x)):
        if scatter:
            ax.scatter(x[i],y[i],label=label[i])
        else:
            ax.plot(x[i],y[i],label=label[i])
        
    plt.legend()

###############################################################################
##### ANALYSIS ################################################################
###############################################################################

def graph_from_lists_country(country):
    # The other function creates a network of all countries.
    # Here we just want a network of foods from a specific country.
    # Then we can draw conclusions from it (distributions, etc).
    
    G = nx.Graph()
    
    nodes = pd.read_excel('final_node_list.xlsx')
    edges = pd.read_csv('full_edgelist.csv')
    
    for i in nodes.index:
        if nodes["Country"][i] == country:
            G.add_node( nodes["Id"][i], country=nodes["Country"][i], ingredients=nodes["Ingredients"][i] )
        
    for i in edges.index:
        graph_nodes = G.nodes
        if edges['src'][i] in graph_nodes and edges['trg'][i] in graph_nodes:
            G.add_edge( edges["src"][i], edges["trg"][i], weight=edges["weight"][i], distance=1/edges["weight"][i] )
    
    return G

def graph_from_lists(nodelist, edgelist):
    G = nx.Graph()
    
    nodes = pd.read_excel(nodelist)
    edges = pd.read_csv(edgelist)
    
    for i in nodes.index:
        G.add_node( nodes["Id"][i], country=nodes["Country"][i], ingredients=nodes["Ingredients"][i] )
        
    for i in edges.index:
        G.add_edge( edges["src"][i], edges["trg"][i], weight=edges["weight"][i], distance=1/edges["weight"][i] )
    
    return G

def average_path_length(G, country=None, weight="distance"):
    
    # Only use the giant component
    giant = max( nx.connected_components(G) )
    used = giant
    
    # Only look at a specific country
    if not country is None:
        new_used = set()
        for food in used:
            if G.nodes[food]["country"] == country:
                new_used.add(food)
        used = new_used
    
    total_distance = 0
    number_paths = 0
    for path_list in nx.algorithms.all_pairs_dijkstra_path_length(G, weight=weight):
        # Iterate through all the shortest paths
        # Only worry about those nodes in our used list
        node_A = path_list[0]
        if node_A in used:
            paths = path_list[1]
            for node_B in paths:
                if node_A != node_B and node_B in giant:
                    total_distance += paths[node_B]
                    number_paths += 1
    
    return total_distance/number_paths

def cumulative_probability(data):
    data.sort()
    prob = []
    tot = len(data)
    prob = [ ( len(data[i:]) )/tot for i in range(len(data)) ]
    
    return data, prob

def plot_distributions(data, attribute, country=None):
    xs = []
    ys = []
    labels = []
    
    for key in data:
        if country is None or key == country:
            xs.append( data[key][attribute]["value"] )
            ys.append( data[key][attribute]["probability"] )
            if key is None:
                labels.append( "Total" )
            else:
                labels.append( key )

    plot(xs, ys, labels)
    matplotlib.pyplot.show()

def cluster_of_k(clustering, k):
    counts = {}
    value = {}
    for i in range(len(clustering)):
        if not k[i] in counts and not k[i] in value:
            counts[k[i]] = 0
            value[k[i]] = 0
        counts[k[i]] += 1
        value[k[i]] += clustering[i]
    
    val = []
    degs = []
    for key in value:
        val.append( value[key]/counts[key] )
        degs.append(key)
    
    return degs, val
   
def distributions(G, country=None):
    cluster_coeff = []
    weighted_degree = []
    degree = []
    
    clusters = nx.algorithms.clustering(G)

    for node in G.nodes(data=True):
        name = node[0]
        
        if country is None or node[1]["country"] == country:
            cluster_coeff.append( clusters[name] )
            weighted_degree.append( G.degree(name, "weight") )
            degree.append( G.degree(name) )
    
    cluster_k, cluster_val = cluster_of_k(cluster_coeff, degree)
    cluster_coeff, cluster_prob = cumulative_probability(cluster_coeff)
    weighted_degree, weighted_prob = cumulative_probability(weighted_degree)
    degree, prob = cumulative_probability(degree)
    
    c_summary = { "value":cluster_coeff, "probability":cluster_prob, "average":np.mean(cluster_coeff), "c":cluster_val, "k":cluster_k }
    w_summary = { "value":weighted_degree, "probability":weighted_prob, "average":np.mean(weighted_degree) }
    d_summary = { "value":degree, "probability":prob, "average":np.mean(degree) }
    summary = {"clustering":c_summary, "weighted degree":w_summary, "degree":d_summary}
    return summary

def get_averages(country):
    g = graph_from_lists_country(country)
    d = distributions(g)
    p = average_path_length(g)
    d['pathlength'] = p
    
    return d

def plot_ck_of_each_country_in_one_graph():

    import matplotlib.pyplot
    
    countries = ['Argentina', 'Belgium', 'Iran', 'Israel', 'Italy', 'Japan', 'Spain', 'Sweden', 'Turkey', 'USA']

    for c in countries:
        d = get_averages(c)
        x_axis = d['clustering']['value']
        y_axis = d['clustering']['probability']

        # while (x_axis[0] == 0 or x_axis[0] == 0.0)
        # print(x_axis)
        # print(y_axis)
        matplotlib.pyplot.scatter(x_axis, y_axis, label=c)

    axis = matplotlib.pyplot.gca()
    axis.set_title('Individual Countries: Probability of Clustering')
    axis.set_xlabel('Clustering')
    axis.set_ylabel('Probability')
    axis.legend()

    matplotlib.pyplot.show()
        

def main():
    plot_ck_of_each_country_in_one_graph()
    
if __name__ == '__main__':
    main()
