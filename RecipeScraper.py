# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 10:19:47 2020

@author: nvana
"""

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests
import os

base = "https://www.epicurious.com/recipes/food/views/{}"
test_recipe = "spicy tomato-tuna noodle skillet casserole with aioli"
recipe_header = "Recipe"

uncleaned_directory = "Uncleaned Data"
cleaned_directory = "Cleaned Data"

#tag = "li"
#category = "class"
#label = "ingredient"

def save_uncleaned_data(f_name, data):
    file = open( os.path.join( uncleaned_directory, f_name ), "w", encoding="utf8" )
    
    for recipe in data:
        file.write("{")
        file.write(recipe)
        file.write("\n")
        file.write( "\n".join( data[recipe] ) )
        file.write("}\n\n")
        
    file.close()
    
def transfer_cleaned_data(uncleaned_f_name="uncleaned_data.txt", cleaned_f_name="Cleaned_data.txt"):
    pass

# Read the cleaned txt file and store it in a dict to convert to an Excel file
def read_cleaned_data(cleaned_f_name):
    file = open( os.path.join( cleaned_directory, cleaned_f_name ), encoding="UTF8" )
    data = {}
    
    active_add = False              # Set to true when adding ingredients
    current_name = None
    for line in file:
        if "{" in line:
            active_add = True
            #food_name = line.replace("{","").replace(":","").strip()
            info = line.replace("{","").split(":")
            food_name = info[0].strip()
            country = ""
            if len(info) > 1:
                country = info[1].strip()
            #data[ food_name ] = []
            data[ food_name ] = [country]
            current_name = food_name
        elif "}" in line:
            data[ current_name ].append( line.replace("}","").strip().lower() )
            active_add = False
        elif active_add:
            data[ current_name ].append( line.strip() )
    
    file.close()
    return data

# Reads the data from the master file containing a column of food names and a column of
# the list of ingredients.
# Stores the data in a dict of {name : [ingredients]}
def read_master(master_recipe_f_name):
    info = pd.read_excel(master_recipe_f_name, encoding="utf-8")
    data = {}
    for i in info.index:
        data[ info["Name"][i] ] = unpack_list( info["Ingredients"][i] )
    return data

# The lists of ingredients in the master recipe file are stored as a string
# the string looks like '[ 'ingredient_A', 'ingredient_B', ...]' but we want a list
# This will unpack it and turn it into an actual list
def unpack_list(line):
    hold = line.replace("[","").replace("]","")         # Remove the brackets from the string
    hold = hold.split(",")                              # No ingredients should have a comma
    hold = [ el.strip()[1:-1].strip().lower() for el in hold ]  # the first and last elements of each string are '. There may be ' in the food name, so we can't just replace
    return hold

# Take a cleaned data txt file and add it to the elements already in an master Excel file
# This master file holds a column of the food name and a column of the list of ingredients
# THERE IS NO CHECK FOR DUPLICATE RECIPES
def add_cleaned_to_master(cleaned_f_name, master_recipe_f_name):
    cleaned_recipes = read_cleaned_data(cleaned_f_name)
    #cleaned_recipes = { name:[name, cleaned_recipes[name] ] for name in cleaned_recipes.keys() }
    #to_add = pd.DataFrame().from_dict( cleaned_recipes, orient="index", columns=["Name","Ingredients"] )
    cleaned_recipes = { name:[name, cleaned_recipes[name][0], cleaned_recipes[name][1:] ] for name in cleaned_recipes.keys() }
    to_add = pd.DataFrame().from_dict( cleaned_recipes, orient="index", columns=["Name","Country","Ingredients"] )
    df = pd.DataFrame()
    if os.path.exists( master_recipe_f_name ):
        df = pd.read_excel(master_recipe_f_name)
    df = df.append( to_add, ignore_index=True )
    #df.index = df.index.str.encode('utf-8')
    df.to_excel( master_recipe_f_name, encoding="utf-8", index=False )

# get the recipie from a url request using BeautifulSoup
# url               - url of the webpage
# tag_type          - type of html tag holding the info (div, ul, ol, etc.)
# category_type     - category used to specify within the tag (class, etc.)
# category label    - label of the category (e.g. class="test", "test" is category_label)
def get_html(url, tag_type, category_type, category_label):
    try:
        soup = BeautifulSoup(requests.get(url).text, 'html.parser')
        elements = soup.findAll(tag_type, attrs={category_type : category_label})
        return elements
    except:
        print("Error scraping ",url)
        return []

# Pull elements (ingredients) of of a list of list item tags
def extract_elements(elements):
    items = []
    for el in elements:
        items.append( el.get_text().strip().replace("\n", " ") )
    return items

def recipie_to_url(recipe_name):
    text = "-".join( recipe_name.split() )
    return base.format(text)

def gather_recipies(master_file, output_file_name="Uncleaned_data.txt"):
    contents = pd.read_csv( master_file )
    recipe_names = contents[ recipe_header ]
    recipes = {}
    for name in recipe_names:
        url = recipie_to_url(name)
        elements = get_html(url, tag, category, label)
        ingredients = extract_elements(elements)
        recipes[ name ] = ingredients
        
    save_uncleaned_data(output_file_name, recipes)
    
def flexible_gather(master_file, web_structure_file, output_file_name="Uncleaned_data.txt", country=None):
    contents = pd.read_csv( master_file, index_col=0 ).fillna("")
    web = pd.read_csv( web_structure_file, index_col=0 ).fillna("")
    sites = web.index
    recipe_names = contents.index
    recipes = {}
    for name in recipe_names:
        # Only bother scraping the links of the country we specify (if applicable) that have not been scraped
        if (country is None or contents["Country Of Origin"][name] == country) and not contents["Data Cleaned"][name]:
            link = contents["Link"][name]
            site = None
            for s in sites:
                if s in link:
                    site = s
            
            tag = web["Tag Type"][site]
            category = web["Category Type"][site]
            label = web["Category Label"][site]
            
            elements = get_html(link, tag, category, label)
            ingredients = extract_elements(elements)
            origin = contents["Country Of Origin"][name]
            label = name+":"+origin
            recipes[ label ] = ingredients
            #recipes[ name ] = ingredients

    save_uncleaned_data(output_file_name, recipes)
    