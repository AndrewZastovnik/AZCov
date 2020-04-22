# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 02:14:46 2020

@author: haldf
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 12:31:21 2020

@author: haldf
"""

import numpy as np
import pandas as pd
import pickle 

states = pickle.load(open('states','rb'))
covid_states = pickle.load(open('covid_states','rb'))
    
    
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Select, Slider
from bokeh.plotting import figure
from pygam import LinearGAM, s

def gam_results(x,y,df,param,infection_time):
    gam = LinearGAM(s(0),lam=.5).fit(x, y)
    y_new = gam.predict(x)
    confi1 = gam.prediction_intervals(x, width=.95)
    pred = np.zeros(x.shape[0])
    for i in np.arange(x.shape[0]):
        if i ==0:
            pred[i] = np.mean(df[param].iloc[0:3])
        else:
            if i < infection_time:
                pred[i] = pred[i-1]*y_new[i] + pred[i-1]
            else:
                pred[i] = pred[i-1]*y_new[i] + pred[i-1]
                
            
    if param == 'Positive':
        pred = pred + np.concatenate((np.zeros(infection_time),pred[0:(pred.shape[0]-infection_time)]),axis = 0)
        
    x_forcast = np.arange(np.max(x),np.max(x)+10)
    y_forcast = gam.predict(x_forcast)
    confi = gam.prediction_intervals(x_forcast, width=.95)

    forcast = np.zeros(x_forcast.shape[0])
    forcast_L = np.zeros(x_forcast.shape[0])
    forcast_U = np.zeros(x_forcast.shape[0])
    for i in np.arange(x_forcast.shape[0]):
        if i ==0:
            forcast[i] = df[param].iloc[-1]
            forcast_L[i] = forcast[i]
            forcast_U[i] = forcast[i]
        else:
            forcast[i] = forcast[i-1]*y_forcast[i-1] + forcast[i-1]
            forcast_L[i] = forcast_L[i-1]*confi[i-1,0] + forcast_L[i-1]
            forcast_U[i] = forcast_U[i-1]*confi[i-1,1] + forcast_U[i-1]
    return([pred,forcast,forcast_L,forcast_U,y_new,confi1])
    
def transform(state,param,infection_time = 12):   
    n = covid_states[state].shape[0]
    active = covid_states[state]['Positive']
    recovered = np.zeros(n)
    recovered[infection_time:] =  covid_states[state]['Positive'][0:(n-infection_time)]
    covid_states[state]["Active"] = active - recovered
    covid_states[state]['Recovered'] = recovered
    rate = np.diff(covid_states[state][param])/(covid_states[state]["Active"].iloc[0:(covid_states[state].shape[0]-1)])
    rate = rate.reset_index(drop=True)
    time = pd.to_datetime(covid_states[state]['Date'])
    index = np.where(np.logical_or(np.isnan(covid_states[state]['Deaths']),covid_states[state]['Deaths']==0))[0][-1] + 1
    days_from_death = ((time[0:(time.shape[0]-1)] - time.iloc[0:(time.shape[0]-1)].iloc[index]).dt.days).reset_index(drop=True)
    index = ~(np.logical_or(np.isnan(rate),np.abs(rate) == np.inf))
    days_from_death = days_from_death[index]
    rate = rate[index]
    return([days_from_death,rate,time])
    
text1 = Select(title = 'Sate1',options = list(covid_states.keys()))
text2 = Select(title = 'Param',options = ['Positive','Active','Deaths'])
inf_time = Slider(title="Days Spent Infecting Others", value=10.0, start=2, end=20, step=1)
    
text1.value = 'California'
text2.value = 'Active'
inf_time.value = 12

state = text1.value
param = text2.value
infection_time = inf_time.value

days_from_death, y, time = transform(state,param,infection_time)
pred,forcast,forcast_L,forcast_U, y_new, confi = gam_results(days_from_death,y,covid_states[state],'Active',infection_time)
for_dates = time.iloc[-1] + pd.to_timedelta(np.arange(10), unit='d')

plot = figure(plot_height=400, plot_width=900, title="Active Infections",
              tools="crosshair,pan,reset,save,wheel_zoom", x_axis_type="datetime")
o = plot.line(time,
              covid_states[state][param],color='blue', legend_label='Observed Infections')
p = plot.line(for_dates,forcast,color='red', legend_label='Forcasted Infections Next 10 Days')
#l = plot.line(for_dates,forcast_L,color='red',line_dash='dashed')
#u = plot.line(for_dates,forcast_U,color='red',line_dash='dashed')
k = plot.line(time.iloc[np.arange(time.shape[0]-pred.shape[0],time.shape[0])],
                        pred,color='purple', legend_label='Model Predicted',line_dash='dotted')
plot.legend.location = "top_left"

plot2 = figure(plot_height=400, plot_width=900, title="Infection Rate",
              tools="crosshair,pan,reset,save,wheel_zoom", x_axis_type="datetime")
a = plot2.line(days_from_death,y_new,color='red', legend_label='Average Rate')
b = plot2.line(days_from_death,confi[0:,0],color='red',line_dash='dashed',
               legend_label='95% prediction interval')
c = plot2.line(days_from_death,confi[0:,1],color='red',line_dash='dashed', 
               legend_label='95% prediction interval')
d = plot2.scatter(days_from_death,y,color='gray', legend_label='Observed Rates')

def update(attr,old,new):
    state = text1.value
    param = text2.value
    infection_time = inf_time.value
    days_from_death, y, time = transform(state,param,infection_time)
    pred,forcast,forcast_L,forcast_U, y_new, confi = gam_results(days_from_death,y,covid_states[state],'Active',infection_time)
    for_dates = time.iloc[-1] + pd.to_timedelta(np.arange(10), unit='d')            
    o.data_source.data = {'x': time,'y':covid_states[state][param]}
    p.data_source.data = {'x': for_dates,'y':forcast}
    #l.data_source.data = {'x': for_dates,'y':forcast_L}
    #u.data_source.data = {'x': for_dates,'y':forcast_U}
    k.data_source.data = {'x': time.iloc[np.arange(time.shape[0]-pred.shape[0],time.shape[0])],'y':pred}
    a.data_source.data = {'x': days_from_death,'y':y_new}
    b.data_source.data = {'x': days_from_death,'y':confi[0:,0]}
    c.data_source.data = {'x': days_from_death,'y':confi[0:,1]}
    d.data_source.data = {'x': days_from_death,'y':y}


for w in [text1, text2, inf_time]:
    w.on_change('value', update)


inputs = column(text1, text2, inf_time)
    
curdoc().add_root(column(row(inputs, plot, width=800,name='covid'),plot2))
curdoc().title = "Covid"