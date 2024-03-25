import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from estrazione import grafico_loss
import os


dss = ['CIFAR10','CIFAR100']
archs = ['DEEP','WIDE','DEEPWIDE','Baseline','Resnet50','Mobilenetv2']
flags = ['opt','def']
Dataseet = 'CIFAR10'
network = 'Baseline'
flag_def = 'opt'
tme = 23
filepath = 'C:\\Users\\corra\\OneDrive\\Desktop\\PaperCNN\\results\\'+network+'_'+Dataseet+'\\'
algoritmi_8 = {0:'Adadelta',1:'Adagrad',2:'Adam',3:'Adamax',4:'FTRL',5:'Nadam',6:'RMSProp',7:'SGD'}
algoritmi_5 = {0: 'Adam', 1:'Adamax', 2:'Nadam', 3:'RMSProp', 4:'SGD'}
colors_8 = ['brown','cyan','blue','green','pink','orange','grey','red']
colors_5 = ['blue','green','orange','grey','red']
plt.figure()
plt.figure(figsize=(6,4.5))
plt.xlabel('Epochs',fontsize=14)
plt.ylabel('Training Loss',fontsize=14)
plt.ylim(0.6,2.1)
#plt.title('Opt_Default_Aug0')
labels = []
k=0
fig, ax1 = plt.subplots()
ax1.set_xlabel('Number of Epochs')
ax1.set_ylabel('Training Loss')
ax1.set_ylim(-0.15,3.2)
ax2 = ax1.twiny()
ax2.set_xlabel('CPU Time (seconds)')
ax1.grid(linewidth=0.5)

miao = True
for nf in list(algoritmi_5.values()):
    if miao:
        if nf!='SGD':
            folder_name = nf+'_'+flag_def
        else:
            folder_name = nf+'_opt_def'
    else:
        folder_name = nf + '_' + flag_def
    f = filepath+folder_name+'\\history_aug.csv'
    loss = grafico_loss(f,'linear')
    algo = algoritmi_5[k]
    epochs = np.array([h for h in range(1,len(loss)+1)])
    cpu_time = epochs*tme
    ax1.plot(epochs,loss,colors_5[k],linewidth=1.75)
    ax2.plot(cpu_time, loss, colors_5[k], linewidth=1.75)
    #plt.plot(loss,colors_5[k],linewidth=1.75)
    k+=1
    labels.append(algo)

#plt.legend(labels)
plt.savefig(Dataseet+'_'+network+'_'+flag_def+'.pdf')
plt.show()

