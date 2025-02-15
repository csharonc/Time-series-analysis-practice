import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


import keras
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam


def create_dataset(noise):
    daterange = pd.date_range(start="2015-01-01", end="2024-12-31", freq='D')
    days_since_start = (daterange - daterange[0]).days
    days_since_start = (daterange - daterange[0]).days
    temp_altitude = 14
    temp_offset = 15
    phase_shift = 90
    DEGREES_PER_YEAR = 1
    sinus_wave = temp_altitude * np.sin(2 * np.pi * (days_since_start - phase_shift) /365) + temp_offset

    trend = days_since_start * DEGREES_PER_YEAR/365
    #voeg noise en jaartrend toe aan data
    if noise == 0:       
        temperatures = np.round((sinus_wave + trend), 1)
    else:
        noise_2 = np.random.uniform(-noise, noise, size = len(sinus_wave))
        temperatures = np.round((sinus_wave + noise_2 + trend), 1)

    
    demo_df = pd.DataFrame({"Data":daterange, "Temperature": temperatures})
    demo_df["Year"] = demo_df["Data"].dt.year
    demo_df["Month"] = demo_df["Data"].dt.month
    demo_df["DayOfWeek"] = demo_df["Data"].dt.weekday +1 #1 = monday
    demo_df["Season"] = demo_df["Data"].dt.quarter
    df = demo_df
    df = df.set_index("Data")
    return df, daterange, temperatures

def plot_one_year(dateranges, temperatures_list, titles):
    fig, axes = plt.subplots(1, 3, figsize = (15, 5))
    for ax, daterange, temperatures, title in zip(axes, dateranges, temperatures_list, titles):
        ax.plot(daterange[:365], temperatures[:365])
        ax.set_title(title)
    fig.supxlabel("Date")
    axes[0].set_ylabel("Temperature (°C)")
    fig.suptitle("Temperature throughout one year for each dataset")

    plt.tight_layout()
    plt.show()


def plot_by_month(dfs, titles):
    fig, axes = plt.subplots(3, 1, figsize = (15,8), sharex = True)
    for ax, df, title in zip(axes, dfs, titles):
        sns.boxplot(ax = ax,
            data=df.replace({
                "Month": {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
                    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
                }}), 
            x="Month", y="Temperature", palette = "Blues")
        ax.set_title(title)
        ax.set_ylabel("")

    fig.legend()
    fig.supylabel("Temperature")
    fig.suptitle("Temperature by Month")
    plt.tight_layout()
    plt.show()

def plot_by_season(dfs, titles):
    fig, axes = plt.subplots(1, 3, figsize = (15, 5))
    for ax, df, title in zip(axes, dfs, titles):
        sns.boxplot(ax = ax, data = df.replace({"Season": {1 : "Winter", 2 : "Spring", 3 : "Summer", 4 : "Autumn"}}), x = "Season", y = "Temperature")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(title)

    fig.supxlabel("Season")
    fig.supylabel("Temperature")
    fig.suptitle("Temperature by season")
    plt.tight_layout()
    plt.show()

def split_train_test(df):
    train = df.loc[df.index < "2024-01-01"]
    test = df.loc[df.index >= "2024-01-01"]
    return train, test


def create_sequences(dfs, seq_length, overlap):
    X_set = []
    y_set = []
    for df in dfs:
        data = df["Temperature"]
        X, y = [], []
        for i in range(0, len(data) - seq_length, overlap):
            X.append(data.iloc[i : i + seq_length]) #in pandas df series gebruik je iloc om de positie van de index aan te duiden, dit is anders dan by numpy
            y.append(data.iloc[i + seq_length])
        X_set.append(X)
        y_set.append(y)
    return np.array(X_set), np.array(y_set)

def scaling(x_data, y_data):
    X, y = [], []
    for i in range(len(x_data)):
        mean = float(np.mean(x_data[i]))
        sd = float(np.std(x_data[i]))
        # newx = [float(((x-mean) / sd)) for x in x_data[i]] #not needed, can subtract mean and divide sd in one go
        newx = (x_data[i] - mean) / sd
        X.append(newx)

        newy = float((y_data[i] - mean) / sd)
        y.append(newy)
    return np.array(X), np.array(y)

