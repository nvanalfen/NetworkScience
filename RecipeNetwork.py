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

def null_model():
    pass

def plot(x, y, label, scatter=True):
    figure = plt.figure()
    ax = figure.add_subplot(111)
    
    for i in range(len(x)):
        if scatter:
            ax.scatter(x[i],y[i],label=label[i])
        else:
            ax.plot(x[i],y[i],label=label[i])
        
    plt.legend()

def test(directed=False, remove_ingredients=None, parent_f_name="parent_ingredients.csv"):
    x = []
    y = []
    labels = []
    parent_ingredients = get_parent_ingredients_to_exclude(parent_f_name)
    if directed:
        foods, edges = create_directed_weighted_network("Final_Food_List.xlsx", remove_ingredients=remove_ingredients, parent_ingredients=parent_ingredients)
        in_deg, out_deg = weight_distribution(foods, edges, directed=directed)
        in_w, in_prob = cumulative_weight_distribution(in_deg)
        out_w, out_prob = cumulative_weight_distribution(out_deg)
        x = [in_w, out_w]
        y = [in_prob, out_prob]
        labels = ["In", "Out"]
    else:
        foods, edges = create_undirected_weighted_network("Final_Food_List.xlsx", remove_ingredients=remove_ingredients, parent_ingredients=parent_ingredients)
        deg = weight_distribution(foods, edges)
        w, prob = cumulative_weight_distribution(deg)
        x = [w]
        y = [prob]
        labels = ["Weight Distribution"]
        
    print(len(edges))
    plot(x, y, labels)
    
    edge_weights, edge_prob = cumulative_edge_weight_distribution(edges)
    plot([edge_weights],[edge_prob],label=["Edge Weights"])
    