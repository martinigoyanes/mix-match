#!/usr/bin/env python
# coding: utf-8

import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

import os
import pandas as pd
import matplotlib.pyplot as plt

filenames = []
for filename in os.listdir("plot_csv"):
   with open(os.path.join("plot_csv", filename), 'r') as f:
    filenames.append(filename)
    
train_acc = []
train_loss = []
val_acc = []
val_loss = []

for i in filenames:
    if "train_acc" in i:
        train_acc.append(i)
    if "train_loss" in i:
        train_loss.append(i)
    if "val_acc" in i:
        val_acc.append(i)
    if "val_loss" in i:
        val_loss.append(i)
        
labels = [250, 500, 1000, 2000, 4000]

y_train_acc = []
y_train_loss = []
y_val_acc = []
y_val_loss = []

train_acc.sort(key=natural_keys)
train_loss.sort(key=natural_keys)
val_acc.sort(key=natural_keys)
val_loss.sort(key=natural_keys)

def fetchdata(files, vec):
    for i, j in enumerate(files):
        df = pd.read_csv("plot_csv/" + files[i])
        vec.append(df['Value'])
    return vec

def plotting(lists, version, name, mode):
    plt.figure(figsize = [10,8])
    for i in range(len(lists)):
        string = "no. labels: " + str(labels[i])
        plt.plot(lists[i], label = string)
        plt.legend()
    
    string = name + "_" + "v" + str(version)
    plt.xlim([0,50])
    plt.xlabel("Epochs")
    plt.ylabel(mode)
    plt.grid()
    plt.savefig("plots/" + string + ".png")
    plt.show()
    
fetchdata(train_acc, y_train_acc)
fetchdata(train_loss, y_train_loss)
fetchdata(val_acc, y_val_acc)
fetchdata(val_loss, y_val_loss)

plotting(y_train_acc, 2, "train_acc", "Accuracy (%)")
plotting(y_train_loss, 2, "train_loss", "Loss")
plotting(y_val_acc, 2, "val_acc", "Accuracy (%)")
plotting(y_val_loss, 2, "val_loss", "Loss")
