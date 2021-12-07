# big data
# python libraries
from datetime import datetime, timedelta
import json
import itertools
import time
import os
# data tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as p
import plotly.express as px
import plotly.graph_objs as go
import pylab
import seaborn as sns
import pycountry
from PIL import Image
from IPython.display import HTML as html_print
# machine learning libraries
import scipy
from scipy.integrate import odeint
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
# app
import streamlit as st



# APP
def app():
  # Add a title
  st.title('COVID19 predictions')
  # countries, codes, cases, world_cases_now, evolution_of_cases_worldwide = get_codes()
  st.header('Predict the spread of COVID-19')
  # pick your country
  try:
      # query selected country
      with st.spinner('Data is being prepared...'):
        time.sleep(3)
      st.sidebar.title("Sub-notification")
      st.sidebar.info('You can test predictions assuming which percentage of the official number of cases are not being reported. For example, if only 50% of cases are being reported, move the slider to the middle. Notification depends on the capacity health services have for testing the population, and may vary greatly from country to country, and even from region to region within a country.')
      notification_percentual = st.sidebar.slider(
        "Notification in %", 
        min_value=0,
        max_value=100,
        step=5,
        value=100)
    
      # notification factor where official data == 100% accurate
      #notification_percentual = 100
      # create sidebar for sub-notification scenarios
      
      #show timeline
      first_day, df = timeline_of_cases_and_deaths(notification_percentual)
      # show also the data using a logarithic scale
      plot_logarithmic(notification_percentual)
      # Data & projections for cases, for today and the former 2 days
      pred = prediction_of_maximum_cases(df, notification_percentual)
      # Data & projections for deaths, for today and the former 2 days
      prediction_of_deaths(df, notification_percentual, pred)
      # Final considerations
      

      
  except Exception as e:
    st.write(e)

def plot_logarithmic(notification_percentual):
  plt.rcParams["font.family"] = "Times New Roman"
  plt.rcParams["font.size"] = "8"
  plt.rcParams['axes.grid'] = True
  # filter target data
  cases_df = pd.read_excel('cases.xlsx')

  # apply subnotification percentage
  # if none was entered, it is == 1
  cases_df.cases = cases_df.cases*100/notification_percentual
  #a = [pow(10, i) for i in range(10)]
  fig = plt.figure()
  ax = fig.add_subplot(2, 1, 1)

  line, = ax.plot(cases_df.cases, color='blue', lw=1)
  ax.set_yscale('log')
  st.write(fig)
  st.write('**Logarithmic scale**')
  # st.pyplot()

# here we get the main dataframe
def timeline_of_cases_and_deaths(notification_percentual):

  cases_df = pd.read_excel('cases.xlsx')
  # apply subnotification percentage
  # if none was entered, it is == 1
  cases_df.cases = cases_df.cases*100/notification_percentual
  # create dataframes for deaths
  deaths_df = pd.read_excel('deaths.xlsx')
 
  # merge into one single dataframe
  df = cases_df.merge(deaths_df, on='day')
  # add culumn for 'day'
  df = df.loc[:, ['day','deaths','cases']]
  # set first day of pendemic
  first_day = datetime(2020, 1, 2) - timedelta(days=1)
  # time format
  FMT = "%Y-%m-%dT%H:%M:%SZ"
  # strip and correct timelines
  df['day'] = df['day'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')

  df['day'] = df['day'].map(lambda x: (datetime.strptime(x, FMT) - first_day).days)
  # bring steramlit to the stage
  st.header('Timeline of cases and deaths')
  st.write('Day 01 of pandemic outbreak is January 1st, 2020.')
  st.write('(*For scenarios with sub-notification, click on side bar*)')
  
  st.write('The data plots the following line chart for cases and deaths.')
  # show data on a line chart
  st.write(type(df['day']))
  st.line_chart(df)
  st.write(first_day)
  st.write(df)
  return first_day, df

# formula for the model
def logistic_model(x,a,b,c):
    return c/(1+np.exp(-(x-b)/a))

# relevant functions
def predict_logistic_maximum(df, column = 'cases'):
      samples = df.shape[0]
      x_days = df['day'].tolist()
      y_cases = df[column].tolist()
      speed_guess = 2.5
      peak_guess = 120
      amplitude_guess = 250000
      if (column == 'deaths'):
        amplitude_guess = (amplitude_guess * speed_guess/100)   
      initial_guess =speed_guess, peak_guess, amplitude_guess

      fit = curve_fit(logistic_model, x_days, y_cases,p0=initial_guess,  maxfev=9999)

      # parse the result of the fit
      speed, x_peak, y_max = fit[0]
      speed_error, x_peak_error, y_max_error = [np.sqrt(fit[1][i][i]) for i in [0, 1, 2]]

      # find the "end date", as the x (day of year) where the function reaches 99.99%
      end = int(fsolve(lambda x: logistic_model(x, speed, x_peak, y_max) - y_max * 0.9999, x_peak))

      return x_days, y_cases, speed, x_peak, y_max, x_peak_error, y_max_error, end, samples


def print_prediction(df, label, column = 'cases'):
    x, y, speed, x_peak, y_max, x_peak_error, y_max_error, end, samples = predict_logistic_maximum(df, column)
    print(label + "'s prediction: " +
          "maximum " + column + " : " + str(np.int64(round(y_max))) +
          " (± " + str(np.int64(round(y_max_error))) + ")" +
          ", peak at calendar day: " + str(datetime(2020, 1, 2) + timedelta(days=int(round(x_peak)))) +
          " (± " + str(round(x_peak_error, 2)) + ")" +
          ", ending on day: " + str(datetime(2020, 1, 2) + timedelta(days=end)))

    st.markdown(label + "'s prediction: " + "maximum " + column + " : **" + str(np.int64(round(y_max))) + "** (± " + str(np.int64(round(y_max_error))) + ")" + ", peak at calendar day: " + str(datetime(2020, 1, 2) + timedelta(days=int(round(x_peak)))) + " (± " + str(round(x_peak_error, 2)) + ")" + ", ending on day: " + str(datetime(2020, 1, 2) + timedelta(days=end)))

    return y_max


def add_real_data(df, label,column = 'cases', color=None):
    x = df['day'].tolist()
    y = df[column].tolist()
    plt.scatter(x, y, label="Data (" + label + ")", c=color)
    return x , y


def add_logistic_curve(df, label,column = 'cases', **kwargs):
    x, _, speed, x_peak, y_max, _, _, end, _ = predict_logistic_maximum(df, column)
    x_range = list(range(min(x), end))

    # plt.plot(x_range,
    #          [logistic_model(i, speed, x_peak, y_max) for i in x_range],
    #          label="Logistic model (" + label + "): " + str(int(round(y_max))),
    #          **kwargs)
    return y_max


def label_and_show_plot(plt, title, y_max=None):
    plt.title(title)
    plt.xlabel("Days since 1 January 2020")
    plt.ylabel("Total number of people")
    if (y_max):
        plt.ylim(0, y_max * 1.1)
    plt.legend()
    # plt.show()


def prediction_of_maximum_cases(df, notification_percentual):
  d=plt.figure(figsize=(12, 8))
  add_real_data(df[:-2], "2 days ago")
  
  add_real_data(df[-2:-1], "yesterday")
  
  add_real_data(df[-1:], "today")


  # add_logistic_curve(df[:-2], "2 days ago", dashes=[8, 8])
  x, _, speed, x_peak, y_max, _, _, end, _ = predict_logistic_maximum(df[:-2], column='cases')
  x_range = list(range(min(x), end))
  plt.plot(x_range,
             [logistic_model(i, speed, x_peak, y_max) for i in x_range],
             label="Logistic model " + str(int(round(y_max))))
  # add_logistic_curve(df[:-1], "yesterday", dashes=[4, 4])
  x, _, speed, x_peak, y_max, _, _, end, _ = predict_logistic_maximum(df[:-1], column='cases')
  x_range = list(range(min(x), end))
  plt.plot(x_range,
             [logistic_model(i, speed, x_peak, y_max) for i in x_range],
             label="Logistic model " + str(int(round(y_max))))
  # y_max = add_logistic_curve(df, "today")
  x, _, speed, x_peak, y_max, _, _, end, _ = predict_logistic_maximum(df, column='cases')
  x_range = list(range(min(x), end))
  plt.plot(x_range,
             [logistic_model(i, speed, x_peak, y_max) for i in x_range],
             label="Logistic model" + str(int(round(y_max))))
  plt.title("Best logistic fit with the freshest data")
  plt.xlabel("Days since 1 January 2020")
  plt.ylabel("Total number of people")
  if (y_max):
   plt.ylim(0, y_max * 1.1)
  plt.legend()       
  # label_and_show_plot(plt, "Best logistic fit with the freshest data", y_max)
  # st.write(y_max)
  # A bit more theory 
  st.header('Prediction of maximum cases')
  # considering the user entered notification values
  if notification_percentual == 1:
    st.markdown("With sub-notification of 0%.")
  else:
    st.markdown("With sub-notification of " + str(int(round(100 - notification_percentual))) + " %.")

  st.write('At high time values, the number of infected people gets closer and closer to *c* and that’s the point at which we can say that the infection has ended. This function has also an inflection point at *b*, that is the point at which the first derivative starts to decrease (i.e. the peak after which the infection starts to become less aggressive and decreases).')
  # plot
  st.pyplot(d)
  # fit the data to the model (find the model variables that best approximate)
  st.subheader('Predictions as of *today*, *yesterday* and *2 days ago*')

  print_prediction(df[:-2], "2 days ago")
  print_prediction(df[:-1], "yesterday")
  pred = print_prediction(df, "today")
  # PREDICTION 1
  st.header('Infection stabilization')
  st.markdown("Predictions as of today, the total infection should stabilize at **" + str(int(round(pred))) + "** cases.")
  return int(round(pred))


def prediction_of_deaths(df, notification_percentual, pred):


  # With subotification, deaths prediction must be within the range of 0.5% and 3.0% of total cases
  if notification_percentual < 100:
    st.write('With the present notification value of ' + str(notification_percentual) + "%, we apply the global mortality rate of 3.5% of total cases.")
    st.markdown('[COVID-19 Global Mortality Rate](https://www.worldometers.info/coronavirus/coronavirus-death-rate/)')
    prediction_of_deaths_3_100 = pred*3.5/100
    st.markdown("- Considering maximum death rate being around 3.5% of the total number of cases, we should expect ** " + str(int(round(prediction_of_deaths_3_100))) + "** deaths")
  else:
    e = plt.figure(figsize=(12, 8))
    add_real_data(df[:-2], "2 days ago", column = 'deaths')
    add_real_data(df[-2:-1], "yesterday", column = 'deaths')
    add_real_data(df[-1:], "today", column = 'deaths')
    add_logistic_curve(df[:-2], "2 days ago",column='deaths', dashes=[8, 8])
    add_logistic_curve(df[:-1], "yesterday",column='deaths', dashes=[4, 4])
    y_max = add_logistic_curve(df, "today", column='deaths')
    label_and_show_plot(plt, "Best logistic fit with the freshest data", y_max)

    st.header('Prediction of deaths')
    st.pyplot(e)

    st.subheader('Predictions as of *today*, *yesterday* and *2 days ago*')

    print_prediction(df[:-2], "2 days ago", 'deaths')
    print_prediction(df[:-1], "yesterday", 'deaths')
    pred = print_prediction(df, "today", 'deaths')
    print()
    html_print("As of today, the total deaths should stabilize at <b>" + str(int(round(pred))) + "</b>")
    # PREDICTION 2
    st.header('Deaths stabilization')
    st.markdown("As of today, the total number of deaths should stabilize at **" + str(int(round(pred))) + "** cases.")
