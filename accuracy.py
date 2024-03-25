import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from estrazione import grafico_loss
import os

filepath = 'C:\\Users\\corra\\OneDrive\\Desktop\\PaperCNN\\results\\'

os.chdir(filepath)
d = {}
for folder in os.listdir():
    os.chdir(folder)
    for elem in os.listdir():
        if elem[0:4] not in  ["Adad","Adag","FTRL"]:
            os.chdir(elem)
            try:
                f = pd.read_csv('history_aug.csv')
            except:
                f = pd.read_csv('_history.csv')
            acc = f.val_accuracy.values[-1]
            print(f'P: {folder}   A: {elem}   Acc: {acc*100:.1f}')
            d[(folder,elem)] = acc*100

            os.chdir(filepath+folder)
    os.chdir(filepath)




