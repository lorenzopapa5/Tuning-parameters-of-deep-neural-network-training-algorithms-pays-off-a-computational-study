import matplotlib.pyplot as plt
import pandas as pd
import xlsxwriter
import os
loss_0 = 6.98

# path = 'C:\\Users\\corra\\OneDrive\\Desktop\\PaperCNN\\'
# folder = 'Opt_Default_Aug1\\'
algoritmi_8 = {0:'Adadelta',1:'Adagrad',2:'Adam',3:'Adamax',4:'FTRL',5:'Nadam',6:'RMSProp',7:'SGD'}
algoritmi_5 = {0: 'Adam', 1:'Adamax', 2:'Nadam', 3:'RMSProp', 4:'SGD'}

def tau_opt_def(path,folder):
    algoritmi = {0:'Adadelta',1:'Adagrad',2:'Adam',3:'Adamax',4:'FTRL',5:'Nadam',6:'RMSProp',7:'SGD'}
    lista_file = os.listdir(path+folder)
    lista_loss = []
    for file in lista_file:
        f = path+folder+file
        wb = pd.read_csv(f)
        values = list(wb.values)
        final_loss = values[-1][1]
        lista_loss.append(final_loss)

    lista_tau = []
    f_L = min(lista_loss)
    for i in range(len(lista_loss)):
        tau = (lista_loss[i]-f_L)/(loss_0-f_L)
        lista_tau.append(tau)
    return lista_tau

def get_accuracy(file):
    wb = pd.read_csv(file)
    values = list(wb.values)
    accuracy = values[-1][2]
    return accuracy

def get_loss(file):
    wb = pd.read_csv(file)
    values = list(wb.values)
    loss = values[-1][1]
    #loss = min(wb.values[:,1])
    return loss

def get_loss_initial(file):
    wb = pd.read_csv(file)
    values = list(wb.values)
    loss = values[0][1]
    #loss = min(wb.values[:,1])
    return loss


def get_loss_accuracy(file):
    wb = pd.read_csv(file)
    values = list(wb.values)
    loss =  max(wb.values[:,-1]) #values[-1][-1]
    return 1-loss

def trova_hp_ottimi(aug):
    path = 'C:\\Users\\corra\\OneDrive\\Desktop\\PaperCNN\\hparams\\Grid_Search\\'
    algo = os.listdir(path)
    d = {}
    losses = {}
    for optimizer in algo:
        direct = os.listdir(path+optimizer+'\\')
        dir = os.listdir(path+optimizer+'\\'+direct[aug]+'\\')
        acc_max = 0
        f_max = -1
        d[optimizer] = f_max
        for filename in dir:
            if filename[-3:] == 'csv':
                file = path+optimizer+'\\'+direct[aug]+'\\'+filename
                accuracy = get_accuracy(file)
                if accuracy >= acc_max:
                    acc_max = accuracy
                    f_max = filename
                    d[optimizer] = filename

        loss = get_loss(file)
        losses[optimizer] = loss
        print(aug,optimizer,f_max)

    return losses

def tau_optimal(aug):
    lista_tau = []
    l = trova_hp_ottimi(aug)
    lv = list(l.values())
    f_L = min(lv)
    for i in range(len(l)):
        tau = (lv[i] - f_L) / (loss_0 - f_L)
        lista_tau.append(tau)
    return lista_tau
'''
tau_opt_default_aug0 = tau_opt_def(path,'Opt_Default_Aug0\\')
tau_opt_default_aug1 = tau_opt_def(path,'Opt_Default_Aug1\\')
tau_opt_optimal_aug0 = tau_optimal(0)
tau_opt_optimal_aug1 = tau_optimal(1)

tau_Deep_aug0_def = tau_opt_def(path,'Deep_Aug0_Def\\')
tau_Deep_aug1_def = tau_opt_def(path,'Deep_Aug1_Def\\')
tau_DW_aug0_def = tau_opt_def(path,'DW_Aug0_Def\\')
tau_DW_aug1_def = tau_opt_def(path,'DW_Aug1_Def\\')
tau_Wide_aug0_def = tau_opt_def(path,'Wide_Aug0_Def\\')
tau_Wide_aug1_def = tau_opt_def(path,'Wide_Aug1_Def\\')

tau_Deep_aug0_opt = tau_opt_def(path,'Deep_Aug0_Opt\\')
tau_Deep_aug1_opt = tau_opt_def(path,'Deep_Aug1_Opt\\')
tau_DW_aug0_opt = tau_opt_def(path,'DW_Aug0_Opt\\')
tau_DW_aug1_opt = tau_opt_def(path,'DW_Aug1_Opt\\')
tau_Wide_aug0_opt = tau_opt_def(path,'Wide_Aug0_Opt\\')
tau_Wide_aug1_opt = tau_opt_def(path,'Wide_Aug1_Opt\\')
'''



def scrivi_su_excel():
    file_excel = 'C:\\Users\\corra\\OneDrive\\Desktop\\PaperCNN\\risultati_tau.xlsx'
    workbook = xlsxwriter.Workbook(file_excel)
    worksheet = workbook.add_worksheet('tau_Default')
    worksheet.write('B2', 'Algo')
    worksheet.write('C2', 'tau_opt_default_aug0')
    worksheet.write('D2', 'tau_opt_default_aug1')
    for i in range(8):
        worksheet.write('B'+str(i+3), algoritmi_8[i])
        worksheet.write('C' + str(i + 3), tau_opt_default_aug0[i])
        worksheet.write('D' + str(i + 3), tau_opt_default_aug1[i])

    worksheet = workbook.add_worksheet('tau')
    worksheet.write('B2', 'Algo')
    worksheet.write('C2', 'tau_opt_optimal_aug0')
    worksheet.write('D2', 'tau_opt_optimal_aug1')
    worksheet.write('E2', 'tau_Deep_aug0_def')
    worksheet.write('F2', 'tau_Deep_aug1_def')
    worksheet.write('G2', 'tau_DW_aug0_def')
    worksheet.write('H2', 'tau_DW_aug1_def')
    worksheet.write('I2', 'tau_Wide_aug0_def')
    worksheet.write('J2', 'tau_Wide_aug1_def')

    worksheet.write('K2', 'tau_Deep_aug0_opt')
    worksheet.write('L2', 'tau_Deep_aug1_opt')
    worksheet.write('M2', 'tau_DW_aug0_opt')
    worksheet.write('N2', 'tau_DW_aug1_opt')
    worksheet.write('O2', 'tau_Wide_aug0_opt')
    worksheet.write('P2', 'tau_Wide_aug1_opt')
    for i in range(5):
        worksheet.write('B'+str(i+3), algoritmi_5[i])
        worksheet.write('C' + str(i + 3), tau_opt_optimal_aug0[i])
        worksheet.write('D' + str(i + 3), tau_opt_optimal_aug1[i])

        worksheet.write('E' + str(i + 3), tau_Deep_aug0_def[i])
        worksheet.write('F' + str(i + 3), tau_Deep_aug1_def[i])
        worksheet.write('G' + str(i + 3), tau_DW_aug0_def[i])
        worksheet.write('H' + str(i + 3), tau_DW_aug1_def[i])
        worksheet.write('I' + str(i + 3), tau_Wide_aug0_def[i])
        worksheet.write('J' + str(i + 3), tau_Wide_aug1_def[i])

        worksheet.write('K' + str(i + 3), tau_Deep_aug0_opt[i])
        worksheet.write('L' + str(i + 3), tau_Deep_aug1_opt[i])
        worksheet.write('M' + str(i + 3), tau_DW_aug0_opt[i])
        worksheet.write('N' + str(i + 3), tau_DW_aug1_opt[i])
        worksheet.write('O' + str(i + 3), tau_Wide_aug0_opt[i])
        worksheet.write('P' + str(i + 3), tau_Wide_aug1_opt[i])

    workbook.close()
    return None

def grafico_loss(file, scala,printflag=False):
    wb = pd.read_csv(file)
    values = wb.values
    loss = values[:,1]
    if printflag==True:
        plt.figure()
        plt.xscale(scala)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.plot(loss)
        plt.show()
    return loss


# path2 = path+folder
# lf = os.listdir(path2)
# f = lf[0]
# wb = pd.read_csv(path2+f)
# losses = wb.values[:,1]
