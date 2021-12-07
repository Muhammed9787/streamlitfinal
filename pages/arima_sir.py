#!/usr/bin/env python
# coding: utf-8

# # COVID-19 Global Forecasting 
# 
# #### Burak Emre Özer - 21527266
# #### Didem Yanıktepe - 21527563 
# 
# <img src="images/cover.png" />
# 
# ### Abstract
# 
# Along with the rapidly increasing human population, the intervention of this population in natural life causes pandemic cases worldwide. These pandemics have important effects on the economy, politics and, most importantly, on human health. Considering that these pandemics can be repeated more frequently in the future, it is necessary to be prepared for this situation. In this project, we will conduct an investigation of estimation methods using time series data. As the data set, we will use a data set containing statistics day by day from the beginning of the pandemic. We will try to make sense of data, map visualizations with data analysis techniques. Then we will make future predictions with statistical approaches (SIR, ARIMA, SARIMAX). These estimates will enable us to make inferences about the routes and treatments followed during the epidemic on a country basis. In the future, we plan to make our pandemic forecasting models more complex (SEIR). So we can come up with different scenarios. This allows models to provide more accurate predictions and inferences.

# ## Contents
# 
# * [1-Problem](#Problem)
# * [2-Data Understanding](#Data-Understanding)
# * [3-Data Preparation](#Data-Preparation)
# * [4-Modeling & Evaluation](#Modeling-&-Evaluation)
#     * [4.a-SIR Model](#SIR-Model)
#     * [4.b-ARIMA & SARIMAX](#ARIMA-&-SARIMAX)
# * [5-Conclusion](#Conclusion)
# * [6-References](#References)

# # Problem

# Pandemics affect the world significantly. Having assumptions about the progress of this pandemic can make the management more effective and solution-oriented. Our aim in this project is to develop different models using statistical methods that can predict the course of the pandemic, that is, the number of cases and deaths. We want to make inferences about the pandemic with data analysis techniques, examine the transmission speed of COVID-19 with the isolation level of people and produce accurate predictions using statistical methods.
# * "Can we identify the main centers of the pandemic?"
# * "How does the pandemic spread from which regions to which regions?",
# * "What is the effect of different social isolation values on the pandemic?",
# * "Can we create a model that makes successful predictions for the future in the pandemic process?"
# 
# Answers to these questions can provide benefits such as how governments should take action against an epidemic or monitor the results of the actions.

# # Data Understanding

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib.dates import DateFormatter
import datetime
# !pip install folium
# import folium as folium
# !pip install plotly
import plotly.express as px
import itertools
from math import sqrt
import matplotlib.dates as mdates
from statsmodels.tsa.arima_model import ARIMA
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
sns.set()
import streamlit as st
def app():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # !pip list
    plt.style.use('bmh')
    # get_ipython().run_line_magic('matplotlib', 'inline')
    plt.rcParams['figure.figsize'] = [9.5, 6]


    # ## Gathering data
    # 
    # We obtained the data provided by Kaggle and some university sources in csv format. These included case, death and recovered statistics in the form of time series. 
    # 
    # [**COVID-19 Dataset** (Confirmed, Death and Recovered cases every day across the globe)](https://www.kaggle.com/imdevskp/corona-virus-report)
    # 
    # We also found a data set about the countries, lockdown type, average age and total population.
    # 
    # [**COVID-19 Useful features by country**](https://www.kaggle.com/c/covid19-global-forecasting-week-5/discussion/148263)
    # 
    # 
    # Oxford COVID-19 Tracker: 
    # 
    # Its aim is to track and compare government responses to the coronavirus outbreak worldwide rigorously and consistently. **Stringency Index (SI)**. The SI is a simple score, based on seven different lockdown indicators, to produce a value from 0 to 100: 0 means no restrictions, **100 means a full Wuhan-style** lockdown.
    # 
    # [**Coronavirus Government Response Tracker**](https://www.bsg.ox.ac.uk/research/research-projects/coronavirus-government-response-tracker)

    # In[3]:


    clean_complete_df = pd.read_csv('covid_19_clean_complete.csv', parse_dates=['Date'])
    useful_features = pd.read_csv("countries_useful_features.csv")
    stringency_index = pd.read_csv("covid-stringency-index.csv", parse_dates=['Date'])


    # ## Describing data
    # 
    # Basically, we have determined the mandatory parts that should be included in our data as Date (datetime), Number of confirmed cases (int64), Number of deaths (int64) and Number of recovered (int64), Country (object).

    # In[4]:


    # display(clean_complete_df.info())
    # display(clean_complete_df.isna().sum())


    # In[5]:


    # display(stringency_index.info())
    # display(stringency_index.isna().sum())


    # ## Exploring data
    # 
    # The dataset contains cumulative statistics for all countries of every day. In order to examine the **daily case statistics**, we may need to take the difference of the lines and create a new dataframe in the future. We can create a function to do this for every country.

    # In[6]:


    clean_complete_df.sort_values(by='Date').describe()


    # In[7]:


    # display(clean_complete_df.head())
    # display(useful_features.head())
    # display(stringency_index.head())


    # ## Map Visualization Timeline

    # In[7]:


    # Attention! It may cause the internet browser to crash.

    # map_df = clean_complete_df.copy()
    # map_df['Date'] = map_df['Date'].astype(str)
    # map_df['size'] = map_df['Confirmed'].pow(0.3)


    # fig = px.scatter_geo(map_df, lat="Lat",lon="Long", color="Confirmed",
    #                      hover_name="Country/Region", size="size",
    #                      animation_frame="Date",
    #                      projection="natural earth")
    # fig.show()


    # In[8]:


    import math
    # from folium.plugins import MarkerCluster
    # from folium import Marker

    # map = folium.Map(location=[48, -102], zoom_start=4)

    # world_map = folium.Map(tiles="cartodbpositron")
    # marker_cluster = MarkerCluster().add_to(world_map)
    # total_confirmed =  clean_complete_df.groupby(['Country/Region']).agg({'Confirmed':'max','Lat':'max','Long':'max'})
    # world_map = folium.Map(location=[33.000000, 65.000000], zoom_start=2)

    # countries = total_confirmed.index
    # i = 0
    # df = total_confirmed
    # for lat, lng, num in zip(df.Lat, df.Long, df.Confirmed): 
            
    #         popup_text = """Country : {} % of Confirmed Cases : {}"""
    #         popup_text = popup_text.format(countries[i],num)
    #         i = i + 1
    #         color='#BDCCFF'
    #         radius = 0.001
    #         if num < 20000:
    #             color = '#FE2F57'
    #             radius = 5
    #         elif num < 50000:
    #             color = '#CE9FFC'
    #             radius = 10
    #         elif num < 100000:
    #             color = '#FF9415'
    #             radius = 15
    #         elif num < 500000:
    #             color = '#4A00E0'
    #             radius = 20
    #         elif num < 1000000:
    #             color = '#EC008C'
    #             radius = 25
    #         else:
    #             color = '#EC008C'
    #             radius = 35
    #         folium.CircleMarker(
    #                     [lat,lng],
    #                     radius=radius,
    #                     fill=True,
    #                     color=color,
    #                     fill_opacity=0.7,
    #                     popup=popup_text
    #             ).add_to(world_map)
                
    # world_map


    # The dataset contains data collected from January 22, 2020, the first date when the data began to be collected, until May 16, 2020.

    # In[9]:


    # print("The lowest date in the data set is", clean_complete_df['Date'].min() ,"and the highest", clean_complete_df['Date'].max())


    # ## Verifying data quality
    # 
    # The data set we have has all the content about the outbreak forecast and contains accurate statistics. No other data source is required.

    # # Data Preparation

    # To determine the number of **active** patients, we subtracted the number of patients who died and recovered from the total number of cases. Also, since we will not use the Province/State column, we removed this column from the data.

    # In[8]:


    clean_complete_df['Active'] = clean_complete_df['Confirmed'] - clean_complete_df['Deaths'] - clean_complete_df['Recovered']
    df = clean_complete_df.groupby(['Date', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
    # display(df.head())


    # In[9]:


    # df.dtypes


    # We filled in the missing data in the table with "None".

    # In[10]:


    useful_features.fillna("None", inplace=True)


    # To shorten the name of the index column in the table, we changed it to **SI**. Likewise, we changed the Entity column to Country.

    # In[11]:


    stringency_index.rename(columns={'Government Response Stringency Index ((0 to 100, 100 = strictest))':'SI',
                                    'Entity':'Country_Region'}, inplace=True )
    # stringency_index


    # ## Constructing data

    # ### Stringency Index Heatmap
    # 
    # We can analyze the government response using the heatmap function from the seaborn python package. While the dates are included on the x axis, we see the responses of the countries for the pandemic between 0-100. We see that the United States taking strict measures in the pandemic was delayed compared to other countries. We understand that Turkey is the most stringent measures taken as soon as possible.

    # In[12]:


    import matplotlib.dates as mdate

    # fig, ax = plt.subplots(figsize=(12, 8))

    # countries = ['Turkey', 'Spain', 'Italy', 'Iran', 'France', 'Germany', 'United States']
    # countries_si = stringency_index.loc[stringency_index['Country_Region'].isin(countries)]
    # # display(countries_si.head())
    # ax = sns.heatmap(countries_si.pivot('Country_Region', 'Date', 'SI'), linewidths=0.5, xticklabels=7)
    # plt.xticks(rotation=30)
    # plt.title('The Stringency Index Heatmap')
    # st.pyplot()


    # We have grouped the Confirmed cases and death columns by Date with the groupby function to reach the worldwide **cumulative number of cases**. We created the table where we obtained **daily statistics** according to time by taking the difference with the diff function to the cumulative table.

    # In[13]:


    cumulative_df = df.groupby('Date', as_index=False)['Confirmed','Deaths', 'Recovered'].sum().sort_values(by='Date')

    daily_df = cumulative_df.copy(deep=True)
    daily_df[daily_df.columns.difference(['Date'])] = daily_df[daily_df.columns.difference(['Date'])].diff(axis=0).fillna(0)
    daily_df[daily_df.select_dtypes(['float64']).columns] = daily_df.select_dtypes(['float64']).apply(lambda x: x.astype('int64'))

    # display(cumulative_df.tail())
    # display(daily_df.tail())


    # ### Mortality and Recovery Rate analysis around the world

    # In order to observe the severity of the epidemic, we will examine the level of mortality compared to other corona types. The rate of recovery increases with new treatment methods.

    # In[14]:


    rate_df = cumulative_df.groupby(["Date"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

    rate_df["Mortality Rate"] = (rate_df["Deaths"]/rate_df["Confirmed"])*100
    rate_df["Recovery Rate"]  = (rate_df["Recovered"]/rate_df["Confirmed"])*100

    print("Average Mortality Rate", rate_df["Mortality Rate"].mean())
    print("Average Recovery Rate", rate_df["Recovery Rate"].mean())


    # **Mortality rate**
    # 
    # SARS 9.63%, MERS 34.45%, COVID-19 4.63% (increasing)

    # In[15]:


    ax = sns.lineplot(x=rate_df.index, y="Mortality Rate", data=rate_df)


    # In[16]:


    ax = sns.lineplot(x=rate_df.index, y="Recovery Rate", data=rate_df)


    # The number of cases reported per day does not make a steep climb. But we can say that the epidemic still continues to spread rapidly. The total number of cases exceeded 4 million and the total number of deaths exceeded 300 thousand.

    # In[17]:


    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # ax1.plot(cumulative_df.loc[1:, 'Date'], daily_df.loc[1:, 'Confirmed'], label="World", color="r")
    # ax1.set(xlabel="Date",
    #        ylabel="Confirmed",
    #        title="Daily Number of Confirmed Cases")

    # ax2.plot(cumulative_df['Date'].values, cumulative_df['Confirmed'], label="World", color="b")
    # ax2.set(xlabel="Date",
    #        ylabel="ConfirmedCases",
    #        title="Total Number of Confirmed Cases")

    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    # plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=15))
    # plt.legend()
    # st.pyplot()
    # plt.gcf().autofmt_xdate()


    # When we look at the daily data, death cases started to decrease. This indicates that the fight against the epidemic is getting better. The new treatment methods applied seem successful.

    # In[18]:


    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    # ax1.plot(cumulative_df.loc[1:, 'Date'], daily_df.loc[1:, 'Deaths'], label="World", color="r")
    # ax1.set(xlabel="Date",
    #        ylabel="Deaths",
    #        title="Daily Number of Confirmed Deaths")

    # ax2.plot(cumulative_df['Date'].values, cumulative_df['Deaths'], label="World", color="b")
    # ax2.set(xlabel="Date",
    #        ylabel="Deaths",
    #        title="Total Number of Confirmed Deaths")

    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    # plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=15))
    # plt.legend()
    # st.pyplot()
    # plt.gcf().autofmt_xdate()


    # We wrote a function like the one below for future use. This function returns daily or cumulative statistics according to the desired country name. We decide whether it is daily or not with the "daily" parameter.

    # In[19]:


    def get_df_by_country(df_country, country, daily=True):
        tmp = df_country.loc[:, ['Date', 'Country/Region', 'Confirmed', 'Deaths', 'Recovered']].loc[(df_country['Country/Region'] == country)]
        tmp.index = np.arange(0, len(tmp))
        if daily:
            tmp[tmp.columns.difference(['Date', 'Country/Region'])] = tmp[tmp.columns.difference(['Date', 'Country/Region'])].diff(axis=0).fillna(0)
            tmp[tmp.select_dtypes(['float64']).columns] = tmp.select_dtypes(['float64']).apply(lambda x: x.astype('int64'))
            return tmp 
        else:
            return tmp


    # Total confirmed cases: how rapidly have they increased compared to other countries?
    # 
    # We have grouped the countries with the highest number of cases currently by Country / Region column. We observe that the number of positive cases is rapidly increasing in **US and Russia** compared to other countries. We can say that Russia is successful in treatment compared to the number of cases. On the other hand, we see that **Italy reported a high number of fatalities** in treatment failures.

    # In[20]:


    top10_countries = df[df['Date'] == df['Date'].max()]
    top10_countries = top10_countries.groupby('Country/Region', as_index=False)['Confirmed','Deaths'].sum()
    top10_countries = top10_countries.nlargest(10, 'Confirmed')
    # top10_countries


    # In the graph below, we see how high the number of confirmed cases is due to the loose policy of the US at the beginning of the pandemic.

    # In[21]:


    # fig, ax = plt.subplots(figsize=(12, 8))

    # colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    # for i, country_name in enumerate(top10_countries['Country/Region'].values):
        
    #     tmp_df = get_df_by_country(df, country_name, daily=False)
    #     ylabels = [format(label, ',.0f') for label in ax.get_yticks()]
    #     ax.set_yticklabels(ylabels)
    #     ax.plot(tmp_df['Date'].iloc[40:].values, tmp_df.iloc[40:]['Confirmed'], label=country_name, color=colors[i])

    # ax.set(xlabel="Date",
    #        ylabel="Cases",
    #        title="Total Confirmed Cases for Top 10 Country")

    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    # plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=15))
    # plt.legend()
    # st.pyplot()
    # plt.gcf().autofmt_xdate()


    # # Modeling & Evaluation
    # We applied 3 different models in our project. We will explain the SIR, ARIMA and SARIMAX models, respectively.
    # 

    # # SIR Model
    # 
    # SIR model is the basis of simple and many other derivative models used for mathematical modeling of infected diseases. The order of the letters in its name actually shows the flow pattern of each case. 
    # 
    # <img src="https://www.lewuathe.com/assets/img/posts/2020-03-11-covid-19-dynamics-with-sir-model/sir.png" width="600" height="180" />
    # 
    # The model try to predict things such as how a disease spreads, or the total number infected, or the duration of an epidemic. ***S*** the Susceptible (healthy) population, ***I*** the Infected population, ***R*** the population which has Recovered from the disease. These variables (S, I, and R) represent the number of people in each compartment at a particular time. 
    # 
    # <img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/29728a7d4bebe8197dca7d873d81b9dce954522e" width="150" height="150" />
    # 
    # To represent that the number of susceptible, infected and recovered  individuals may vary over time (even if the total population size remains constant), we make the precise numbers a function of t (time): **S(t)**, **I(t)** and **R(t)**. R0, reproductive number:
    # 
    # <img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/4aae42f8253a395c52a798a9ad5a7e4adb6fceea" width="60" height="50" /> 
    # 
    # where ***ß***  is the contamination probability, ***gamma*** is the recovery probability,  
    # 
    # 
    # 

    # In[22]:


    from scipy.integrate import odeint
    from scipy import integrate, optimize
    from sklearn import preprocessing
    from sklearn.metrics import mean_squared_error


    # We have implemented the model below and given the initial parameters. The odeint function will help us solve differential equations, so we sent it to odeint. Initially, we identified 99 percent of the population healthy and 1 percent of the infected. You see how these three values change on the chart by time. Since the number of healthy people will decrease over time, there is a negative sign in its derivative. On the contrary, as the number of infected people will increase, its derivative is positive.

    # In[23]:


    def SIR_model_test(y, t, N, beta, gamma):
        S, I, R = y
        dSt = (-beta * S * I / N)
        dIt = (beta * S * I / N) - (gamma * I)
        dRt = gamma * I
        return dSt, dIt, dRt

    N = 1
    beta = 0.5
    gamma = 0.1
    S0, I0, R0 = (0.99, 0.01, 0.0)

    t = np.linspace(0, 100, 1000)
    y0 = S0, I0, R0 # Initial conditions vector

    result = odeint(SIR_model_test,[S0, I0, R0], t, args=(N, beta,gamma))
    S, I, R = result.T

    f, ax = plt.subplots(1,1,figsize=(12,6))
    ax.plot(t, S, linewidth=4, label='Susceptible')
    ax.plot(t, I, linewidth=2, label='Infected')
    ax.plot(t, R, linewidth=1, label='Recovered', color='green')
    ax.set_xlabel('Time (days)')
    ax.legend()
    st.pyplot()

    # Let's look at how the number of infected people changes by time by increasing the mobility rate of communities. The epidemic is **spreading faster** as the **increase in mobility** rate will also increase the chances of infection. We observe that the pandemic has reached its peak in almost 1 month in an environment where everyone is free to move around (mobility = 1).

    # In[24]:


    m = [0.3, 0.5, 0.7, 1]
    results = list()
    for i in m:
        result = odeint(SIR_model_test,[S0, I0, R0], t, args=(N, beta*i, gamma))
        S, I, R = result.T
        results.append(I)

    for i in range(len(m)):
        plt.plot(t, results[i], label = "people's mobility: " + str(m[i]))

    plt.title("Contamination probability increases with mobility")
    plt.xlabel("Days")
    plt.ylabel("Infected cases")
    plt.legend()
    st.pyplot()


    # Let's create our model for Turkey. We've set the value of the N parameter in the model which is Turkey's population. We have completed the table operations for the S, I and R values then integrated them into the original table. For example, "I" column shows the numbers of active infected people in Turkey.

    # In[25]:


    country_name = 'Sudan'
    turkey_df = get_df_by_country(df, country_name, daily=False)

    # using LabelEncoder to transform Date column
    turkey_df['DayNo'] = preprocessing.LabelEncoder().fit_transform(turkey_df.Date)

    turkey_population = useful_features[useful_features['Country_Region'] == country_name].Population_Size.values[0]
    turkey_df['R'] = turkey_df['Deaths'] + turkey_df['Recovered']
    turkey_df['I'] = turkey_df['Confirmed'] - turkey_df['R']
    turkey_df['S'] = turkey_population-turkey_df['Confirmed'] - turkey_df['I'] - turkey_df['R']


    # In[26]:


    # turkey_df.tail()


    # In[27]:


    def SIR_model_turkey(y, t, beta, gamma):
        S, I, R = y
        N = turkey_population
        dSt = (-beta * S * I / N)
        dIt = (beta * S * I / N) - (gamma * I)
        dRt = gamma * I
        return dSt, dIt, dRt

    def fit_ode(x, beta, gamma):
        return integrate.odeint(SIR_model_turkey, (S0, I0, R0), x, args=(beta, gamma))[:,1]


    # First of all, the necessary parameters were given to make the model estimate until the **25th of June**. We fit the model with our initial values and got the most suitable Beta and Gamma parameters. According to the output of the model, it can be estimated that the outbreak will end in mid-June. Due to the slowdown of the trend in the decrease of the pandemic in recent days, the difference between the estimated values and the real values has increased.

    # In[28]:


    N, I0 = turkey_population, 1
    S0, R0 = turkey_population-I0, 0

    x_values = turkey_df.DayNo
    x_values = np.array(x_values, dtype=float)

    y_values = np.array(turkey_df['I'], dtype=float)
    y_values = np.array(y_values, dtype=float)

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
    # Use non-linear least squares to fit a function, f, to data.
    popt, pcov = optimize.curve_fit(fit_ode, x_values, y_values)
    beta = popt[0]
    gamma = popt[1]
    print("Beta=", beta, ", Gamma=", gamma)

    time = np.arange(0, 156, 1)
    fitted = integrate.odeint(SIR_model_turkey, (S0, I0, R0), time, args=(beta, gamma))[:,1]

    errors = list()
    error = sqrt(mean_squared_error(y_values, fitted[:len(y_values)]))
    errors.append(('SIR', error))
    print("RMSE:", error)

    x_dates = pd.date_range(start='2020-01-22', end='2020-05-16')
    dates = pd.date_range(start='2020-01-22', end='2020-06-25')
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(x_dates, y_values, label='Original Confirmed Cases', color='b', marker='o')
    ax.plot(dates, fitted, label='SIR Predictions', color='r')

    ax.set(xlabel="Date",
        ylabel="Cases",
        title="SIR Model: COVID-19 Forecasting for Turkey")

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=15))
    plt.xticks(rotation=20)
    plt.legend()
    st.pyplot()
    plt.gcf().autofmt_xdate()


    # Let's make it a function to do all this for any country.

    # In[29]:


    def SIR_model_by_country(_df, country_name):

        def SIR_model(y, t, beta, gamma):
            S, I, R = y
            N = country_population
            dSt = (-beta * S * I / N)
            dIt = (beta * S * I / N) - (gamma * I)
            dRt = gamma * I
            return dSt, dIt, dRt

        def fit_ode(x, beta, gamma):
            return integrate.odeint(SIR_model, (S0, I0, R0), x, args=(beta, gamma))[:,1]

        sir_df = get_df_by_country(_df, country_name, daily=False)

        # using LabelEncoder to transform Date column
        sir_df['DayNo'] = preprocessing.LabelEncoder().fit_transform(sir_df.Date)

        country_population = useful_features[useful_features['Country_Region'] == country_name].Population_Size.values[0]
        sir_df['R'] = sir_df['Deaths'] + sir_df['Recovered']
        sir_df['I'] = sir_df['Confirmed'] - sir_df['R']
        sir_df['S'] = country_population-sir_df['Confirmed'] - sir_df['I'] - sir_df['R']

        N, I0 = country_population, 1
        S0, R0 = country_population-I0, 0

        x_values = sir_df.DayNo
        x_values = np.array(x_values, dtype=float)

        y_values = np.array(sir_df['I'], dtype=float)
        y_values = np.array(y_values, dtype=float)

        popt, pcov = optimize.curve_fit(fit_ode, x_values, y_values)
        beta = popt[0]
        gamma = popt[1]
        print("Beta=", beta, ", Gamma=", gamma)
        time = np.arange(0, 156, 1)
        x_dates = pd.date_range(start='2020-01-22', end='2020-05-16')
        dates = pd.date_range(start='2020-01-22', end='2020-06-25')

        fitted = integrate.odeint(SIR_model, (S0, I0, R0), time, args=(beta, gamma))[:,1]
        print("RMSE:", sqrt(mean_squared_error(y_values, fitted[:len(y_values)])))

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(x_dates, y_values, label='Original Confirmed Cases', color='b', marker='o')
        ax.plot(dates, fitted, label='SIR Predictions', color='r')

        ax.set(xlabel="Date",
            ylabel="Cases",
            title="SIR Model: COVID-19 Forecasting for " + country_name)

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=15))
        plt.xticks(rotation=20)
        plt.legend()
        st.pyplot()
        plt.gcf().autofmt_xdate()


    # We observe that Italy has entered the slowdown phase in the epidemic.

    # In[32]:


    # SIR_model_by_country(df, 'Italy')


    # Spain seems to be recovering the situation in the same way.

    # In[60]:


    # SIR_model_by_country(df, 'Turkey')


    # It seems possible for Iran to face a second wave. The model could not predict this wave and could not make a successful fit.

    # In[34]:


    # SIR_model_by_country(df, 'Iran')


    # # ARIMA & SARIMAX
    # 
    # ### ARIMA
    # 
    # <img src="images/arima.png">
    # 
    # We will model our time series data using **ARIMA** (autoregressive integration moving average) and make predictions about the future. Its general use is ARIMA (p, d, q). p autoregressive ar(p) is used for the part, and d indicates how many times the data series will be reduced. In other words, it means the new series obtained by removing each data from the previous data. d is generally used to eliminate the trend in the data and to reach stationarity. You can interpret q as how many prior noises (errors) affect the current value
    # 
    # ### SARIMAX
    # 
    # SARIMAX contains a few differences from the ARIMA method. The new thing we see here is this "S" so that s stands for seasonality,  it takes into account seasonal information. Seasonality is a repeating pattern in regular intervals with in the data. 

    # Let's create tables that we're doing examples with Turkey.

    # In[35]:


    df_turkey_daily = get_df_by_country(df, 'Sudan', daily=True)
    df_turkey_cumulative = get_df_by_country(df, 'Sudan', daily=False)

    # display(df_turkey_daily.tail())
    # display(df_turkey_cumulative.tail())


    # ## Decomposing the time series data for Turkey
    # 
    # Any time series may be split into the following components: 
    # Base Level + Trend + Seasonality + Error
    # 
    # A trend is observed when there is an increasing or decreasing slope in the time series. Seasonality is observed in the form of repeated patterns that are distinct of each other and at regular time intervals.
    # 
    # The graph refers to the time series of dailys cases of Turkey. We used the decompose method of the statsmodels library to observe the key components from this time series data which are trend and seasonality.

    # 
    # It seems to have displayed an increase and then a decreasing trend. As seen in the bottom area, Residual section displays the trend and seasonal effect eliminated on the original data. The spread of the pandemic peaked between 75 and 100 days.

    # In[36]:


    from statsmodels.tsa.seasonal import seasonal_decompose
    # display(df_turkey_daily.shape)
    fig=sm.tsa.seasonal_decompose(df_turkey_daily.Confirmed.values, model='additive', freq=7).plot()


    # ## Building ARIMA and SARIMAX Model
    # 
    # Before we building a model, we must ensure that the time series is stationary. First of all, let's explain what it means to be stationary of a time series data with the example below. 
    # 
    # ![stationary](http://www.seanabu.com/img/Mean_nonstationary.png)
    # 
    # The value shown in the plot is the mean of the time series data. As on the right, if this value **increases over time**, our data is not stationary. In other words, our data can be expressed **as a function of time**. However, in the graph on the left, this value **does not increase with time**, so it seems stationary.
    # 

    # 
    # When using a statistical model, we assume that the data are independent from each other. However, in this case, we know that each data point is dependent on a certain time. To use such a model, statistical properties must be constant over time. So the data must be stationary.
    # 
    # There are two methods to decide whether the data is stationary. One of them is to plot the mean and std. of the data depending on time. If these values remain constant over time, it is stationary. However, making visual inferences is not always accurate and feasible.
    # 
    # The other method is the Augmented Dickey-Fuller test is a type of statistical test called a unit root test. The idea behind a unit root test is that it determines how strongly a time series is defined by a trend.
    # 
    # *   **Rolling Statistics:** Plot statistical properties then make visual inferences. The time series is stationary if they remain constant with time.
    # *   **Augmented Dickey-Fuller Test:** In order to call a data stationary, the test statistic value must be smaller than or closer the critical values and the p-value must be less than 0.05.

    # In[37]:

    from statsmodels.tsa.stattools import adfuller

    def is_stationary(df_ts):
        # starting from the first day, we calculate the mean 
        # and standard deviation in a 7-day window width called rolling mean/std.
        rolling_mean = df_ts.rolling(window=7).mean()
        rolling_std = df_ts.rolling(window=7).std()
        
        # Rolling Statistics:
        plt.plot(df_ts, color='blue', label='Original')
        plt.plot(rolling_mean, color='red', label='Rolling Mean')
        plt.plot(rolling_std, color='black', label='Rolling Std')
        plt.title('Rolling Mean & Standard Deviation')
        plt.legend()
        # st.pyplot()
        
        # Augmented Dickey-Fuller Test:
        result = adfuller(df_ts)
        print('ADF Statistic: {}'.format(result[0]))
        print('p-value: {}'.format(result[1]))
        print('Critical Values:')
        for key, value in result[4].items():
            print('\t{}: {}'.format(key, value))


    # For the original data, we run the tests in the above function. We see that statistical properties increase with respect to time. In addition, in the ADF test, we see that the ADF Statistic value is greater than the critical values and the p-value is greater than 0.05. This indicates that the data is **not stationary**.

    # In[38]:


    # is_stationary(df_turkey_daily.Confirmed)


    # We need to **transform the data to stationary** before using our model. There are several methods for this. We will use **logarithmic** transformation from these. After taking the logarithm of the data, we also take the logarithms of rolling mean. Then we subtract this from the rolling log. mean. These stages enable us to discard the trend in the data.

    # In[39]:


    turkey_daily_log = np.log(df_turkey_daily.loc[:, ['Confirmed']])
    turkey_daily_log = turkey_daily_log.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

    log_rolling_mean = turkey_daily_log.rolling(window=7).mean()
    df_log_minus_mean = turkey_daily_log - log_rolling_mean
    df_log_minus_mean = df_log_minus_mean.replace([np.inf, -np.inf], np.nan).dropna(axis=0)


    # As seen below, according to ADF test, our statistical values are in the form of stationary time series data. Test statistics value is less than critical value and p-value is less than 0.05. We **obtained more stationary** data than the original data.

    # In[40]:


    # is_stationary(df_log_minus_mean.Confirmed)


    # We shift the data once and get the difference again. As a result, it becomes more stationary.

    # In[41]:


    turkey_daily_log_shift = turkey_daily_log - turkey_daily_log.shift()
    turkey_daily_log_shift = turkey_daily_log_shift.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    # is_stationary(turkey_daily_log_shift.Confirmed)


    # # ARIMA
    # 
    # In order to make the data stationary, it seems enough to take the d parameter as 1. Since we will estimate each value with the previous 2 values and calculate the error accordingly, we select p and q values 2.

    # In[63]:


    date_idx = [116 + x for x in range(0,40)]
    dates = pd.date_range(start='2020-01-22', end='2020-06-25')

    def calculate_arima_by_country(_df, country_name, split_perc=0.85):
        param = (1,1,2)

        if country_name == 'Global':
            df_ts = daily_df
            param = (2,1,2)
        else:
            df_ts = get_df_by_country(_df, country_name, daily=True)

        country_train = df_ts[:int(len(df_ts)*split_perc)].loc[:, 'Confirmed'].values
        country_train = pd.DataFrame(data=country_train[:], index=dates[:int(len(df_ts)*split_perc)], columns=['Confirmed'])
        country_test = df_ts[int(len(df_ts)*split_perc):].loc[:, 'Confirmed'].values

        model = ARIMA(country_train, order=param)
        result = model.fit()
        fig, ax = plt.subplots(figsize=(14, 6))
        ax = country_train.loc['2020-03-01':].plot(ax=ax)
        fig = result.plot_predict(start=int(len(country_train)/1.01), end=int(len(country_train)* 1.5), 
                                ax=ax, plot_insample=False)
        predictions = result.forecast(steps=len(country_test))[0]
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=21))
        plt.legend()
        plt.xlabel("Dates")
        plt.ylabel("Cases")
        plt.title("ARIMA: COVID-19 Forecasting for " + country_name)
        plt.gcf().autofmt_xdate()
        st.pyplot()
        # root mean square error(RMSE), is a measure of
        # prediction accuracy of a forecasting method in statistics
        error = sqrt(mean_squared_error(country_test, predictions[:len(country_test)]))
        errors.append(('ARIMA', error))
        print("RMSE:", error)

        return predictions


    # According to the estimates, the number of daily world-wide cases will remain stable for a long time in high numbers.

    # In[64]:


    # forecast_values = calculate_arima_by_country(df, 'Global', 0.85)


    # In Turkey, according to estimates made for the lower limit of the epidemic it is expected to end in **June**.

    # In[46]:


    forecast_values = calculate_arima_by_country(df, 'Sudan', 0.85)


    # In[45]:


    # forecast_values = calculate_arima_by_country(df, 'US', 0.85)


    # # Seasonal ARIMA

    # We apply the same stabilization processes that we do for the ARIMA model.

    # In[47]:


    df_turkey_daily['first_diff'] = df_turkey_daily.Confirmed - df_turkey_daily.Confirmed.shift(1)
    # is_stationary(df_turkey_daily.first_diff.dropna(inplace=False))


    # In[48]:


    df_turkey_daily['seasonal_diff'] = df_turkey_daily.Confirmed - df_turkey_daily.Confirmed.shift(7)
    # is_stationary(df_turkey_daily.seasonal_diff.dropna(inplace=False))


    # In[49]:


    df_turkey_daily['seasonal_first_diff'] = df_turkey_daily.first_diff - df_turkey_daily.first_diff.shift(14)
    # is_stationary(df_turkey_daily.seasonal_first_diff.dropna(inplace=False))


    # In[50]:


    fig = sm.graphics.tsa.plot_acf(df_turkey_daily.Confirmed.diff().dropna())


    # In[51]:


    fig = sm.graphics.tsa.plot_pacf(df_turkey_daily.Confirmed.diff().dropna(), lags=40)


    # We chose to make an estimate compared to the previous 2 weeks since we received the data rounding average of 7 days.

    # In[52]:


    cases_model = sm.tsa.statespace.SARIMAX(df_turkey_daily.Confirmed, trend='n', order=(14,0,7))
    deaths_model = sm.tsa.statespace.SARIMAX(df_turkey_daily.Deaths, trend='n', order=(14,0,7))

    cases_results = cases_model.fit()
    deaths_results = deaths_model.fit()
    print(cases_results.summary())


    # In[53]:


    # cases_results.plot_diagnostics(figsize=(16, 8))
    # st.pyplot()

    # We wanted to make a 40-day forecast. For this, we have added the datetime lines of the next 40 days to our data.

    # In[54]:


    future = pd.DataFrame(index=date_idx, columns=df_turkey_daily.columns)

    df_turkey_concat = pd.concat([df_turkey_daily, future])
    df_turkey_concat['Date'] = dates
    df_turkey_concat = df_turkey_concat.loc[:, ['Date', 'Country/Region', 'Confirmed', 'Deaths']]


    # In[55]:


    df_turkey_concat['forecast'] = cases_results.predict(start = 90, end = 155)


    # According to the parameters we have chosen for SARIMAX, the outbreak is expected to end on **June 29**. This data matches the June value we obtained with ARIMA. The error rate was close to the error rate of the ARIMA model.

    # In[56]:


    fig, ax = plt.subplots(figsize=(14, 6))
    df_turkey_concat = df_turkey_concat.iloc[-110:]

    ax.plot(df_turkey_concat['Date'].values, df_turkey_concat['Confirmed'], label='Original Confirmed Cases', color='b', marker='o')
    ax.plot(df_turkey_concat['Date'].values, df_turkey_concat['forecast'], label='SARIMAX Predictions', color='r')

    ax.set(xlabel="Date",
        ylabel="Confirmed Cases",
        title="SARIMAX: COVID-19 Confirmed Cases Forecasting for Turkey")

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=14))
    plt.legend()
    st.pyplot()
    plt.gcf().autofmt_xdate()

    error = sqrt(mean_squared_error(df_turkey_concat.iloc[50:70].Confirmed, df_turkey_concat.iloc[50:70].forecast))
    errors.append(('SARIMAX', error))
    print("RMSE:", error)


    # In[57]:


    df_turkey_concat['forecast'] = deaths_results.predict(start = 90, end = 155)

    fig, ax = plt.subplots(figsize=(14, 6))
    df_turkey_concat = df_turkey_concat.iloc[-110:]

    ax.plot(df_turkey_concat['Date'].values, df_turkey_concat['Deaths'], label='Original Deaths', color='b', marker='o')
    ax.plot(df_turkey_concat['Date'].values, df_turkey_concat['forecast'], label='SARIMAX Predictions', color='r')

    ax.set(xlabel="Date",
        ylabel="Deaths",
        title="SARIMAX: COVID-19 Deaths Forecasting for Turkey")

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=14))
    plt.legend()
    st.pyplot()
    plt.gcf().autofmt_xdate()

    error = sqrt(mean_squared_error(df_turkey_concat.iloc[50:70].Deaths, df_turkey_concat.iloc[50:70].forecast))
    errors.append(('SARIMAX', error))
    print("RMSE:", error)


    # ### RMSE (Root-mean Square Error)
    # We used RMSE for each model to measure performance in time series data. 
    # 
    # RMSE Formula             |  RMSE Representation
    # :-------------------------:|:-------------------------:
    # <img src="images/rmse_f.png">  |  <img src="images/rmse_plot.png">
    # 
    # 
    # RMSE is a measure of accuracy, to compare forecasting errors of different models for a particular dataset.

    # In[59]:


    errors = [('SIR', 4405.029835823134),
    ('ARIMA', 945.8815028659368),
    ('SARIMAX', 262.9830764867462)]

    fig, ax = plt.subplots()

    models = ('SIR', 'ARIMA', 'SARIMAX')
    y_pos = np.arange(len(models))

    ax.barh(y_pos, [e[1] for e in errors], align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.invert_yaxis()
    ax.set_xlabel('RMSE')
    ax.set_ylabel('Models')
    ax.set_title('Forecasting Evaluation')

    st.pyplot()

    # # Conclusion

    # We conducted research on how to process the data collected in the form of a time series related to the COVID-19 outbreak. Using the Folium and plotly libraries, we visualized the location information in the data on the map. We explained concepts such as trend and seasonality. We applied the SIR model, which is an epidemiological model, in its simplest form. With this model, we presented graphs on how the parameters of the outbreak affect the outbreak process. We simulated how the level of isolation of people affects the peak of the outbreak or a flat orientation. We have taken steps on what stationarity is in time series data, why it should be and how to make our data stationary. Then, we used ARIMA and SARIMAX, which are statistical models, by fitting the most appropriate parameters.
    # 
    # Autocorrelation             |  Partial Autocorrelation
    # :-------------------------:|:-------------------------:
    # <img src="images/auto.png">  |  <img src="images/pauto.png">
    # 
    # We observed that ARIMA models can make very different estimates according to the selected p, d and q parameters. We found that we can use some graphical methods on how to select these parameters. For example, in the chart above, we can decide the values ​​of p and q with the value of the point that comes out of the blue area for the first time. In the SARIMAX model, we tried the weekly or 2-week parameters to highlight the seasonal concept. We chose these parameters because of the validity of about 1 week in showing the effect of the disease. With the effect of the concept of seasonality, the most successful prediction was realized by the SARIMAX model. According to the estimate of this model, the outbreak is expected to end in late June. In order to develop this project, we want to implement more complex models in the SIR models in the future. Also, using poly. regression, we want to make forecasting and comparisons based on multiple parameters (country's mean age, stringency index, population, hospital capacity).
    # 
    # We wish you healthy days!

    # # References
    # 
    # Forecasting:
    # 
    # https://www.kaggle.com/eswarchandt/timeseries-forecasting-of-covid-19-arima
    # 
    # https://www.kaggle.com/neelkudu28/covid-19-visualizations-predictions-forecasting
    # 
    # https://www.kaggle.com/chrischow/demographic-factors-for-explaining-covid19
    # 
    # SIR:
    # 
    # http://erdos2.matkafasi.com/?p=1065
    # 
    # https://www.kaggle.com/khanalkiran/covid19-ca-sir-model
    # 
    # https://www.kaggle.com/carloslira/covid19-mexico-sir-map-with-geopandas
    # 
    # https://github.com/RemiTheWarrior/epidemic-simulator/blob/master/epidemic.py
    # 
    # https://www.kaggle.com/dgrechka/sir-model-fit-for-italy
    # 
    # https://www.kaggle.com/abhijithchandradas/sir-model-don-t-understand-calculus-don-t-worry
    # 
    # Building ARIMA and SARIMAX Model:
    # 
    # https://dataplatform.cloud.ibm.com/exchange/public/entry/view/815137c868b916821dec777bdc23013c
    # 
    # https://machinelearningmastery.com/time-series-data-stationary-python/
