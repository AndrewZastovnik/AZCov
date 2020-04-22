# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:43:23 2020

@author: haldf
"""

import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import pickle

print('0')

URL = 'https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States_by_population'
page = requests.get(URL)
soup = BeautifulSoup(page.content, 'html.parser')
table = soup.find_all('table')[0] 
states = pd.read_html(str(table))[0]
states = states.iloc[0:,[2,3]]
states.columns = ["State","Population"]
states = states.iloc[0:np.where(states.State =="Contiguous United States")[0][0]]

print('1')

covid_states = {}
infection_time = 10

for i in states['State']:
    URL = 'https://covidtracking.com/data/state/'+ i.replace(" ","-").replace(".","") +'#historical' 
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')
    table = soup.find_all('table')[1] 
    df = pd.read_html(str(table))[0]
    df = df.iloc[np.arange(df.shape[0]-1,0,-1),0:]
    df["Date"] = df["Date"].str[4:]
    active = df['Positive']
    n = df['Positive'].shape[0]
    recovered = np.zeros(n)
    recovered[infection_time:] =  df['Positive'][0:(n-infection_time)]
    df["Active"] = active - recovered
    df['Recovered'] = recovered
    print(i)
    covid_states[i] = df
    
pickle.dump(states,open('states','wb'))
pickle.dump(covid_states,open('covid_states','wb'))
