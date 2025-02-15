import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


import keras
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam

def plot_train_and_test_set(train_sets, test_sets, titles):
    fig, axes = plt.subplots(3, 1, figsize=(15, 5), sharex = True)
    for ax, train, test, title in zip(axes, train_sets, test_sets, titles):
        ax.scatter(train.index, train["Temperature"], label="Training set", color="blue", alpha=0.5, s=10)

        ax.scatter(test.index, test["Temperature"], label="Test set", color="red", alpha=0.5, s=10)
        ax.set_title(title)
        ax.legend()
        ax.set_ylim(bottom=-20, top=50)

    fig.supxlabel("Year")
    fig.supylabel("Temperature")
    fig.suptitle("Temperature over time (Train/Test split)")

    plt.tight_layout()
    plt.show()

def train_model(X_train_scaled, y_train_scaled, epochs):
    model = Sequential([LSTM(50, activation = "relu", input_shape = (seq_length, 1)), Dense(1)])
    optimizer = Adam(learning_rate = 1e-3)
    model.compile(optimizer = optimizer, loss = "mse")
    history = model.fit(X_train_scaled, y_train_scaled, epochs = epochs, batch_size = 64, verbose = 2)
    return model, history

def plot_loss(history, xlim, ylim):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.ylim(0, ylim)
    plt.xlim(-1, xlim)
    plt.legend()
    plt.show()

def compare_performance(model, X_train_scaled_set, y_train_scaled_set):
    dfs = []
    for X_train_scaled, y_train_scaled in zip(X_train_scaled_set, y_train_scaled_set):
        prediction = model.predict(X_train_scaled)
        comparison_df = pd.DataFrame(columns = ["Prediction", "Correct value"])
        comparison_df["Prediction"] = prediction.flatten()
        comparison_df["Correct value"] = y_train_scaled.flatten()
        comparison_df["Difference"] = np.sqrt((comparison_df["Prediction"] - comparison_df["Correct value"])**2)
        dfs.append(comparison_df)
    return dfs

