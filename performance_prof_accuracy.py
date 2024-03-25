import matplotlib.pyplot as plt
import pandas as pd
import xlsxwriter, xlrd
import os
from estrazione import get_loss, algoritmi_5, algoritmi_8, get_loss_accuracy, get_loss_initial
import matplotlib
import matplotlib.pyplot as plt



path = 'C:\\Users\\corra\\OneDrive\\Desktop\\PaperCNN\\results\\'
file_excel = 'C:\\Users\\corra\\OneDrive\\Desktop\\PaperCNN\\final_loss.xlsx'
workbook = xlsxwriter.Workbook(file_excel)
worksheet = workbook.add_worksheet('Loss')
for i in range(5):
    worksheet.write('B'+str(2+i),algoritmi_5[i])

d = {}
initial_losses = []
conta = 0
flag = 'opt'
algos = ['Adam','Adamax','Nadam','RMSprop','SGD']
new_path = path
for folder in os.listdir(path):
    problem = folder
    algo = 'Adam'
    if "ucmerced" in folder:
        file = path+ folder + '\\'+ algo + '_' + flag + '\\_history.csv'
    else:
        file = path +folder+ '\\'+algo + '_' + flag + '\\history_aug.csv'
    initial_losses.append(get_loss_initial(file))
    worksheet.write(chr(ord('C')+conta)+'1',problem)
    h = 2
    new_path = path + folder + '\\'
    for algo in algos:
        if "ucmerced" in folder:
            file = new_path + algo + '_' + flag + '\\_history.csv'
        else:
            file = new_path+algo+'_'+flag+'\\history_aug.csv'
        try:
            loss = get_loss(file)#get_loss_accuracy(file)
        except:
            if "ucmerced" in folder:
                file = new_path + algo + '_' + 'opt_def' + '\\_history.csv'
            else:
                file = new_path + algo + '_' + 'opt_def' + '\\history_aug.csv'

            loss = get_loss(file)#get_loss_accuracy(file)

        #d[problem].append(loss)
        worksheet.write(chr(ord('C')+conta)+str(h),loss)
        h+=1
    conta += 1

workbook.close()



def calcolo_pp(i,j):
    u = len(loss_values[0])
    loss_algo = loss_values[j]
    conta = 0
    for k in range(u):
        if loss_algo[k] <= list_f_L[k] + tau[i]*(initial_losses[k]-list_f_L[k]):
            conta += 1
    sr = conta/u
    return sr



file = file_excel
tau = [0.0001,0.0005,0.001,0.005,0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
wb = pd.read_excel(file)
index_list = [_ for _ in range(2,20)]
loss_values = wb.values[0:5, index_list]
u = len(index_list)
list_f_L = [min(loss_values[:,i]) for i in range(u)]
al_pp = []
for j in range(5):
    pp = []
    for i in range(len(tau)):
        sr = calcolo_pp(i,j)
        pp.append(sr)
    al_pp.append(pp)


plt.figure()
plt.grid()
plt.title('Performance Profile - Optimal')
plt.ylabel('rho')
plt.xlabel('tau')
plt.xscale('log')
plt.ylim(-0.05,1.05)
col = ['blue','green','orange','grey','red']
file = file_excel
tau = [0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.3,0.5,0.7,0.9,1.0]
wb = pd.read_excel(file)
loss_values = wb.values[0:5,index_list]
u = len(index_list)
list_f_L = [min(loss_values[:,i]) for i in range(u)]
al_pp = []
for j in range(5):
    pp = []
    for i in range(len(tau)):
        sr = calcolo_pp(i,j)
        pp.append(sr)
    al_pp.append(pp)

for i in range(len(al_pp)):
    plt.step(tau,al_pp[i],col[i],linestyle='solid',linewidth=1.75)

plt.legend(algos)
plt.savefig('PerformanceProfiles_opt.pdf')
plt.show()