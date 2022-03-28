#獲得関数のライブラリ
import numpy as np
from scipy.stats import norm
from scipy.stats.mvn import mvnun
import sys
import matplotlib.pyplot as plt
from numba import jit
from scipy.stats import multivariate_normal
import pickle
import copy
import random

def multi_mes(X,y_star,pred_mean_task_list, pred_var_task_list,task_var,func_num,pred_matrix):
    grid_num = 100  #区分求積法のグリッド数
    pred_var_task = np.array(pred_var_task_list)  #リストをnumpyに変換 (3,200,1)
    pred_mean_task = np.array(pred_mean_task_list)  #リストをnumpyに変換 (3,200,1)
    y_star = np.array(y_star) #リストをnumpyに変換
    task_entropy = np.log(np.sqrt(2 * np.pi * np.e * pred_var_task)) #第一項
    ppf_low = 1e-12
    lower_ppf = norm.ppf(q=ppf_low, loc=pred_mean_task, scale=np.sqrt(pred_var_task)) #(3,200,1)
    upper_ppf = norm.ppf(q=1 - ppf_low, loc=pred_mean_task, scale=np.sqrt(pred_var_task))
    alpha_task = np.zeros(task_entropy.shape) #(3,200,1)

    for i in range(len(X)):
        #予測共分散行列の作成
        var = np.array([
            [pred_matrix[i, i], pred_matrix[i, len(X) + i], pred_matrix[i, 2 * len(X) + i], pred_matrix[i, 3 * len(X) + i]],
            [pred_matrix[len(X) + i, i], pred_matrix[len(X) + i, len(X) + i], pred_matrix[len(X) + i, 2 * len(X) + i], pred_matrix[len(X) + i, 3 * len(X) + i]],
            [pred_matrix[2 * len(X) + i, i], pred_matrix[2 * len(X) + i, len(X) + i], pred_matrix[2 * len(X) + i, 2 * len(X) + i], pred_matrix[2 * len(X) + i, 3 * len(X) + i]],
            [pred_matrix[3 * len(X) + i, i], pred_matrix[3 * len(X) + i, len(X) + i], pred_matrix[3 * len(X) + i, 2 * len(X) + i], pred_matrix[3 * len(X) + i, 3 * len(X) + i]]
        ])

        for y_index in range(func_num):  # モンテカルロ ここをいずれ改良
            upper = y_star[:,y_index]
            #all_cdf, _ = mvnun([-np.inf, -np.inf, -np.inf], [upper_ppf[0,i],upper_ppf[1,i],upper_ppf[2,i]], [pred_mean_task[0,i], pred_mean_task[1,i], pred_mean_task[2,i]], var)  #第二項の分母の部分 ,第二返却値tはintの情報らしいが使わない。floatのvalueのみ使う
            all_cdf, _ = mvnun([-np.inf, -np.inf, -np.inf, -np.inf], upper, pred_mean_task[:, i], var)  #第二項の分母の部分 ,第二返却値tはintの情報らしいが使わない。floatのvalueのみ使う
            
            upper_t = np.minimum(upper, upper_ppf[:, i])
            y_grid = np.linspace(lower_ppf[:, i], upper_t, grid_num, endpoint=False) +((upper_t - lower_ppf[:, i])/grid_num)/2  # y_tのグリッド数 (400,3,1)
            for task_index in range(len(pred_mean_task_list)):  # タスク数のループ
                tmp = np.c_[np.delete(var, task_index, 0)[:, task_index]]  #Sigma_{ab}
                conditional_var = np.delete(np.delete(var, task_index, 0), task_index, 1) - np.dot(tmp / var[task_index, task_index], tmp.T)
                
                # ここからチェック
                y_t_pdf = norm.pdf((y_grid[:, task_index] - pred_mean_task[task_index, i]) / np.sqrt(pred_var_task[task_index, i]), loc=0, scale=1) / np.sqrt(pred_var_task[task_index, i])
                integral_val = 0
                for y_t_grid in range(grid_num):
                    conditional_mean = np.delete(pred_mean_task[:, i], task_index, 0) + tmp / var[task_index, task_index] * (y_grid[y_t_grid, task_index] - pred_mean_task[task_index, i])
                    #print(conditional_mean)
                    #sys.exit()
                    conditional_cdf, _ = mvnun([-np.inf, -np.inf,-np.inf], np.delete(upper_t, task_index, 0), conditional_mean, conditional_var)
                    #if conditional_cdf != 0:
                    integral_val += y_t_pdf[y_t_grid]*conditional_cdf*np.log(y_t_pdf[y_t_grid]*conditional_cdf) *((upper_t[task_index] - lower_ppf[task_index,i])/grid_num)
                alpha_task[task_index,i] += integral_val/all_cdf - np.log(all_cdf)
        
    alpha_task /= func_num
    alpha_task += task_entropy
    return alpha_task

def multi_ei(y_train_list, pred_mean_list, pred_var_list):
    alpha_list = []
    for i in range(len(y_train_list)):
        tau=y_train_list[i].max()
        tau=np.full(pred_mean_list[i].shape,tau)
        t=(pred_mean_list[i]-tau-0.01)/np.sqrt(pred_var_list[i])
        #norm.cdf、norm.pdfはscipy.statsのライブラリ。それぞれ標準正規分布の累積密度関数と、密度関数を示す
        acq = (pred_mean_list[i] - tau - 0.01) * norm.cdf(x=t, loc=0, scale=1) + np.sqrt(pred_var_list[i]) * norm.pdf(x=t, loc=0, scale=1)
        alpha_list.append(acq)
    return alpha_list
    
def expected_improvement(y_train, pred_mean, pred_var):
    tau=y_train.max()
    tau=np.full(pred_mean.shape,tau)
    t=(pred_mean-tau-0.01)/np.sqrt(pred_var)
    #norm.cdf、norm.pdfはscipy.statsのライブラリ。それぞれ標準正規分布の累積密度関数と、密度関数を示す
    acq=(pred_mean-tau-0.01)*norm.cdf(x=t, loc=0, scale=1)+np.sqrt(pred_var)*norm.pdf(x=t, loc=0, scale=1)
    return acq

def upper_confidence_bound(X_train,pred_mean, pred_var):
    t = X_train.shape[0]
    acq = pred_mean + np.sqrt(2 * np.log(t ** 2 + 1)) * np.sqrt(pred_var)
    return acq

def mes(y_star, pred_mean_list, pred_var_list):
    pred_mu = np.array(pred_mean_list)
    pred_var = np.array(pred_var_list)
    task_num = len(pred_mean_list)
    func_num = len(y_star[0])

    alpha_list = []
    for task in range(task_num):
        y_sample = np.tile(y_star[task], len(pred_mu[task])).T
        gamma_y = ((y_sample - pred_mu[task]) / np.sqrt(pred_var[task])).T
        psi_gamma = norm.pdf(gamma_y, loc=0, scale=1)
        large_psi_gamma = norm.cdf(gamma_y, loc=0, scale=1)
        #log_large_psi_gamma = norm.logcdf(gamma_y, loc=0, scale=1)
        log_large_psi_gamma = np.log(large_psi_gamma) 

        A = gamma_y*psi_gamma
        B = 2*large_psi_gamma
        temp = A / B - log_large_psi_gamma
        alpha = np.sum(temp, axis=0) / func_num
        alpha_list.append(alpha)
        #alpha[train_index]=0 #観測済みの点の獲得関数値は0にする
    alpha_list = np.array(alpha_list)
    return alpha_list

def single_mes(y_star, pred_mu, pred_var, func_num):
    y_sample = np.tile(y_star, (1,pred_mu.shape[0])).T
    gamma_y = (y_sample - pred_mu) / np.sqrt(pred_var)
    psi_gamma = norm.pdf(gamma_y, loc=0, scale=1)
    large_psi_gamma = norm.cdf(gamma_y, loc=0, scale=1)
    #log_large_psi_gamma = norm.logcdf(gamma_y, loc=0, scale=1)
    log_large_psi_gamma = np.log(large_psi_gamma)
    A = gamma_y*psi_gamma
    B = 2*large_psi_gamma
    temp = A / B - log_large_psi_gamma
    alpha = np.sum(temp, axis=1) / func_num
    #alpha[train_index]=0 #観測済みの点の獲得関数値は0にする

    return alpha

# 昔のsum_mes
# def sum_mt_mes(X, y_star, pred_mean_task_list, pred_var_task_list, func_num, pred_matrix):
#     task_num = len(pred_mean_task_list)
#     grid_num = 500 #区分求積法のグリッド数
#     pred_var_task = np.array(pred_var_task_list)  #リストをnumpyに変換 (4,200,1)
#     pred_mean_task = np.array(pred_mean_task_list)  #リストをnumpyに変換 (4,200,1)
#     y_star = np.array(y_star)  #リストをnumpyに変換 (4,5,1)
#     task_entropy = (task_num-1) * np.log(np.sqrt(2 * np.pi * np.e * pred_var_task))  # 第一項 (4,200,1) あえてT-1にしておく。MESをそのまま足したいので
#     ppf_low = 1e-12 #e-7以降はcdfが0なってlogがバグる
#     lower_ppf = norm.ppf(q=ppf_low, loc=pred_mean_task, scale=np.sqrt(pred_var_task))  #(4,200,1)
#     upper_ppf = norm.ppf(q=1 - ppf_low, loc=pred_mean_task, scale=np.sqrt(pred_var_task)) #(4,200,1)
#     alpha_task = np.zeros(task_entropy.shape)  #(4,200,1)

#     y_star_i = np.tile(y_star, len(X))  #(4,5,200)
#     #print(y_star_i.shape)
#     pred_mean_i = np.tile(pred_mean_task, func_num) #(4,200,1) -> #(4,200,5)
#     pred_mean_i = pred_mean_i.transpose((0, 2, 1))  #(4,5,200)
    
#     pred_std_i = np.tile(np.sqrt(pred_var_task), func_num)
#     pred_std_i = pred_std_i.transpose((0, 2, 1)) #(4,5,200)
    
#     gamma_f_i_star = (y_star_i - pred_mean_i) / pred_std_i
#     f_star_i_cdf = norm.cdf(gamma_f_i_star, loc=0, scale=1)
    
#     # MES用
#     psi_gamma = norm.pdf(gamma_f_i_star,loc = 0,scale=1)
#     large_psi_gamma = norm.cdf(gamma_f_i_star, loc = 0,scale =1)
#     log_large_psi_gamma = np.log(large_psi_gamma)
#     A = gamma_f_i_star*psi_gamma
#     B = 2*large_psi_gamma
#     temp = A / B - log_large_psi_gamma
#     mes = np.sum(temp, axis=1) / func_num
    
#     grid = np.linspace(lower_ppf, upper_ppf, grid_num, endpoint=False) +((upper_ppf - lower_ppf)/grid_num)/2 
#     grid = grid.transpose((1, 2, 0, 3)).squeeze() #(4,200,800)
    
#     gamma_f_t = (grid - pred_mean_task)/np.sqrt(pred_var_task)
#     f_t_pdf = norm.pdf(gamma_f_t, loc=0, scale=1) / np.sqrt(pred_var_task) #(4,200,800)
#     #print(grid.shape)
#     #print(np.diag(pred_matrix[len(X) * 0 : len(X) * (0 + 1), len(X) * 3 : len(X) * (3 + 1)]).shape)
#     for i in range(task_num):
#         alpha_task[i] += task_entropy[i]
#         y_star_dash = y_star_i[i][:, :, np.newaxis]
#         y_star_dash = np.repeat(y_star_dash, grid_num, axis=2)  #(5,200,800)
        
#         for j in range(task_num): #jのループ一回分があるtの獲得関数値
#             if(i != j):
#                 # conditional_mean = pred_mean_task[i] + np.c_[np.diag(pred_matrix[len(X) * i : len(X) * (i + 1), len(X) * j : len(X) * (j + 1)]) / np.diag(pred_matrix[len(X) * j : len(X) * (j + 1), len(X) * j : len(X) * (j + 1)])] * (grid[j] - pred_mean_task[j])
#                 # conditional_var = pred_var_task[i] - np.c_[np.diag(pred_matrix[len(X) * i : len(X) * (i + 1), len(X) * j : len(X) * (j + 1)]) / np.diag(pred_matrix[len(X) * j : len(X) * (j + 1), len(X) * j : len(X) * (j + 1)]) * np.diag(pred_matrix[len(X) * j : len(X) * (j + 1), len(X) * i : len(X) * (i + 1)])]
#                 # pred_matrixをなんとかしてdiagをはずした 9/6 修正：9/15
#                 conditional_mean = pred_mean_task[i] + np.c_[pred_matrix[i * task_num + j] / pred_matrix[j * task_num+j]] * (grid[j] - pred_mean_task[j])
#                 conditional_var = pred_var_task[i] - np.c_[pred_matrix[i * task_num + j] / pred_matrix[j * task_num + j] * pred_matrix[i + task_num * j]]
#                 conditional_var = np.tile(conditional_var, len(grid[j, 1]))  #(200,1) -> (200,800)
#                 conditional_var = conditional_var[np.newaxis,:,:]
#                 conditional_var = np.repeat(conditional_var, func_num, axis=0)  #(200,800) -> (5, 200,800)
                
#                 #sys.exit()
#                 conditional_mean = np.tile(conditional_mean, (func_num,1,1)) #(200,800) -> (5,200,800)
#                 conditional_gamma_f_i_star = (y_star_dash - conditional_mean)/np.sqrt(conditional_var)
#                 conditional_cdf = norm.cdf(conditional_gamma_f_i_star, loc=0, scale=1)
                
#                 f_t_pdf_dash = f_t_pdf[i][np.newaxis, :, :]
                
#                 f_t_pdf_dash = np.repeat(f_t_pdf_dash,func_num,axis=0) #(5,200,800)
#                 #print(f_t_pdf_dash.shape) #(4,200,800)
                
#                 integral_value = f_t_pdf_dash * conditional_cdf * (np.log(f_t_pdf_dash) + np.log(conditional_cdf))  #(5,200,800)

#                 np.nan_to_num(integral_value, copy=False)
                
#                 # 区分求積の処理
#                 #print(upper_ppf[i].shape)
#                 kubun = np.repeat((upper_ppf[i] - lower_ppf[i]) / grid_num, grid_num, axis=1)
                
#                 kubun = kubun[np.newaxis, :, :]
#                 kubun = np.repeat(kubun, func_num, axis=0)
#                 #print(kubun.shape)
#                 integral_value *= kubun
#                 integral_value = np.sum(integral_value, axis=2)
                
#                 integral_value = np.sum(integral_value / f_star_i_cdf[i] - np.log(f_star_i_cdf[i]), axis=0) / func_num
#                 integral_value = np.c_[integral_value]
#                 alpha_task[i] += integral_value
                
                
#             else:
#                 alpha_task[i] += np.c_[mes[j]]
        
#     '''
#     plt.plot(X, mes[j], "r")
#     plt.plot(X, alpha_task[0], "b")
#     plt.savefig("temp.png")
#     plt.close()
#     sys.exit()
#     '''
#     return alpha_task

def sum_mt_mes(X,y_star, pred_mean_task_list, pred_var_task_list, func_num, pred_matrix,task_feature, near_task_num):
    # 近傍のタスク数 (task_entrop！ をこの数に変えることを忘れない！)
    task_num = len(pred_mean_task_list)
    grid_num = 100 #区分求積法のグリッド数
    pred_var_task = np.array(pred_var_task_list)  #リストをnumpyに変換 (4,200,1)
    pred_mean_task = np.array(pred_mean_task_list)  #リストをnumpyに変換 (4,200,1)
    
    y_star = np.array(y_star)  #リストをnumpyに変換 (4,5,1)
    task_entropy = (near_task_num - 1) * np.log(np.sqrt(2 * np.pi * np.e * pred_var_task))  # 第一項 (4,200,1) あえてT-1にしておく。MESをそのまま足したいので
    # plt.plot(X, task_entropy[0], "b",label="Task1")
    # plt.plot(X, task_entropy[1], "r", label="Task2")
    # plt.legend(loc="lower left")
    # plt.savefig("task_entropy.pdf")
    # plt.close()
    ppf_low = 1e-12 #e-7以降はcdfが0なってlogがバグる
    lower_ppf = norm.ppf(q=ppf_low, loc=pred_mean_task, scale=np.sqrt(pred_var_task))  #(4,200,1)
    upper_ppf = norm.ppf(q=1 - ppf_low, loc=pred_mean_task, scale=np.sqrt(pred_var_task)) #(4,200,1)
    alpha_task = np.zeros(task_entropy.shape)  #(4,200,1)
    y_star_i = np.tile(y_star,len(X))  #(4,5,200)

    #print(y_star_i.shape)
    pred_mean_i = np.tile(pred_mean_task, func_num) #(4,200,1) -> #(4,200,5)
    pred_mean_i = pred_mean_i.transpose((0, 2, 1))  #(4,5,200)
    
    pred_std_i = np.tile(np.sqrt(pred_var_task), func_num)
    pred_std_i = pred_std_i.transpose((0, 2, 1)) #(4,5,200)
    
    gamma_f_i_star = (y_star_i - pred_mean_i) / pred_std_i
    f_star_i_cdf = norm.cdf(gamma_f_i_star, loc=0, scale=1)
    
    # MES用
    psi_gamma = norm.pdf(gamma_f_i_star,loc = 0,scale=1)
    large_psi_gamma = norm.cdf(gamma_f_i_star, loc = 0,scale =1)
    log_large_psi_gamma = np.log(large_psi_gamma)
    A = gamma_f_i_star*psi_gamma
    B = 2*large_psi_gamma
    temp = A / B - log_large_psi_gamma

    #print((integral_value / f_star_i_cdf[i] - np.log(f_star_i_cdf[i])).shape)
    mes = np.sum(temp, axis=1) / func_num

    # save_data_lists = [X, mes]
    # data_name_lists = ["X", "alpha_list"]
    # for i in range(len(data_name_lists)):
    #     with open(save_data_path+data_name_lists[i]+".pickle", 'wb') as f:
    #         pickle.dump(save_data_lists[i], f) 
    # sys.exit()

    # print(mes)
    # plt.plot(X, task_entropy[0], 'g')
    # plt.plot(X, task_entropy[1], 'r')
    # plt.savefig("tmp.png")
    # # plt.show()
    # sys.exit()
    
    grid = np.linspace(lower_ppf, upper_ppf, grid_num, endpoint=False) +((upper_ppf - lower_ppf)/grid_num)/2 
    grid = grid.transpose((1, 2, 0, 3)).squeeze() #(4,200,800)
    
    gamma_f_t = (grid - pred_mean_task)/np.sqrt(pred_var_task)
    f_t_pdf = norm.pdf(gamma_f_t, loc=0, scale=1) / np.sqrt(pred_var_task) #(4,200,800)
    #True_index = [0,98,99]
    #print(np.diag(pred_matrix[len(X) * 0 : len(X) * (0 + 1), len(X) * 3 : len(X) * (3 + 1)]).shape)
    for i in range(task_num):
        alpha_task[i] += task_entropy[i]
        y_star_dash = y_star_i[i][:, :, np.newaxis]
        y_star_dash = np.repeat(y_star_dash, grid_num, axis=2)  #(5,200,800)

        #sort
        abs = np.abs(task_feature - task_feature[i])
        sorted_index = np.argsort(abs[:,0])[:near_task_num]
        #print(sorted_index)
        for j in sorted_index: #jのループ一回分があるtの獲得関数値
            #print(j)
            if (i!=j):
                # conditional_mean = pred_mean_task[i] + np.c_[np.diag(pred_matrix[len(X) * i : len(X) * (i + 1), len(X) * j : len(X) * (j + 1)]) / np.diag(pred_matrix[len(X) * j : len(X) * (j + 1), len(X) * j : len(X) * (j + 1)])] * (grid[j] - pred_mean_task[j])
                # conditional_var = pred_var_task[i] - np.c_[np.diag(pred_matrix[len(X) * i : len(X) * (i + 1), len(X) * j : len(X) * (j + 1)]) / np.diag(pred_matrix[len(X) * j : len(X) * (j + 1), len(X) * j : len(X) * (j + 1)]) * np.diag(pred_matrix[len(X) * j : len(X) * (j + 1), len(X) * i : len(X) * (i + 1)])]
                # pred_matrixをなんとかしてdiagをはずした 9/6 修正：9/15

                conditional_mean = pred_mean_task[i] + np.c_[pred_matrix[i * task_num + j] / pred_matrix[j * task_num+j]] * (grid[j] - pred_mean_task[j])
                conditional_var = pred_var_task[i] - np.c_[pred_matrix[i * task_num + j] / pred_matrix[j * task_num + j] * pred_matrix[i + task_num * j]]
                conditional_var = np.tile(conditional_var, len(grid[j, 1]))  #(200,1) -> (200,800)
                conditional_var = conditional_var[np.newaxis,:,:]
                conditional_var = np.repeat(conditional_var, func_num, axis=0)  #(200,800) -> (5, 200,800)
                
                #sys.exit()
                conditional_mean = np.tile(conditional_mean, (func_num,1,1)) #(200,800) -> (5,200,800)
                conditional_gamma_f_i_star = (y_star_dash - conditional_mean)/np.sqrt(conditional_var)
                conditional_cdf = norm.cdf(conditional_gamma_f_i_star, loc=0, scale=1)
                
                f_t_pdf_dash = f_t_pdf[i][np.newaxis, :, :]
                
                f_t_pdf_dash = np.repeat(f_t_pdf_dash,func_num,axis=0) #(5,200,800)
                #print(f_t_pdf_dash.shape) #(4,200,800)
                integral_value = f_t_pdf_dash * conditional_cdf * (np.log(f_t_pdf_dash) + np.log(conditional_cdf))  #(5,200,800)
                np.nan_to_num(integral_value, copy=False)
                
                # 区分求積の処理
                #print(upper_ppf[i].shape)
                kubun = np.repeat((upper_ppf[i] - lower_ppf[i]) / grid_num, grid_num, axis=1)
                kubun = kubun[np.newaxis, :, :]
                kubun = np.repeat(kubun, func_num, axis=0)
                #print(kubun.shape)
                integral_value *= kubun
                integral_value = np.sum(integral_value, axis=2)
                
                # #print((integral_value / f_star_i_cdf[i] - np.log(f_star_i_cdf[i])).shape)

                # if task_feasible[j] == False:
                #     rate = np.c_[multivariate_normal.pdf(out_list[0], mean=Theta_mean_list[j].ravel(), cov=Theta_var_list[j]) / multivariate_normal.pdf(out_list[0], mean=Theta_mean_list[0].ravel(), cov=Theta_var_list[0])]

                #print(rate.shape)
                integral_value = np.sum(integral_value / f_star_i_cdf[i] - np.log(f_star_i_cdf[i]), axis=0) / func_num
                #tmp = np.sum(- rate * np.log(f_star_i_cdf[i]), axis=0) / func_num
                #tmp = np.sum(rate*np.log(f_star_i_cdf[i]), axis=0) / func_num
                #print(tmp.shape)
                # if i == 0:
                #     #print("================")
                #     #print(rate)
                #     plt.plot(X, tmp, 'r')
                #     plt.savefig("tmp1.png")
                #     sys.exit()
                integral_value = np.c_[integral_value]
                # plt.plot(X, integral_value, "r")
                # plt.plot(X, alpha_task[i], "g")
                alpha_task[i] += integral_value
                # print(alpha_task[i])
                # plt.plot(X, alpha_task[i], "b")
                # plt.savefig("temp.png")
            else:
                alpha_task[i] += np.c_[mes[j]]
        #sys.exit()
    # print("ははは")
    # plt.plot(X, alpha_task[0], "b",label="Task1")
    # plt.plot(X, alpha_task[1], "r",label="Task2")
    # plt.legend(loc="lower left")
    # plt.savefig("all.png")
    # sys.exit()
        
    '''
    plt.plot(X, mes[j], "r")
    plt.plot(X, alpha_task[0], "b")
    plt.savefig("temp.png")
    plt.close()
    sys.exit()
    '''
    
    return alpha_task

#y_starを前もって変えとく
def imp_sum_mt_mes(X, y_star, pred_mean_task_list, pred_var_task_list, func_num, pred_matrix,Theta_mean_list,Theta_var_list,out_list,task_feasible,task_feature,near_task_num):
    task_num = len(pred_mean_task_list)
    grid_num = 500 #区分求積法のグリッド数
    pred_var_task = np.array(pred_var_task_list)  #リストをnumpyに変換 (4,200,1)
    pred_mean_task = np.array(pred_mean_task_list)  #リストをnumpyに変換 (4,200,1)
    y_star = np.array(y_star)  #リストをnumpyに変換 (4,5,1)
    task_entropy = (near_task_num - 1) * np.log(np.sqrt(2 * np.pi * np.e * pred_var_task))  # 第一項 (4,200,1) あえてT-1にしておく。MESをそのまま足したいので
    # plt.plot(X, task_entropy[0], "b",label="Task1")
    # plt.plot(X, task_entropy[1], "r", label="Task2")
    # plt.legend(loc="lower left")
    # plt.savefig("task_entropy.pdf")
    # plt.close()
    ppf_low = 1e-12 #e-7以降はcdfが0なってlogがバグる
    lower_ppf = norm.ppf(q=ppf_low, loc=pred_mean_task, scale=np.sqrt(pred_var_task))  #(4,200,1)
    upper_ppf = norm.ppf(q=1 - ppf_low, loc=pred_mean_task, scale=np.sqrt(pred_var_task)) #(4,200,1)
    alpha_task = np.zeros(task_entropy.shape)  #(4,200,1)

    y_star_i = np.tile(y_star, len(X))  #(4,5,200)
    #print(y_star_i.shape)
    pred_mean_i = np.tile(pred_mean_task, func_num) #(4,200,1) -> #(4,200,5)
    pred_mean_i = pred_mean_i.transpose((0, 2, 1))  #(4,5,200)
    
    pred_std_i = np.tile(np.sqrt(pred_var_task), func_num)
    pred_std_i = pred_std_i.transpose((0, 2, 1)) #(4,5,200)
    # print(y_star_i)
    # sys.exit()
    
    
    gamma_f_i_star = (y_star_i - pred_mean_i) / pred_std_i
    f_star_i_cdf = norm.cdf(gamma_f_i_star, loc=0, scale=1)
    
    # MES用
    psi_gamma = norm.pdf(gamma_f_i_star,loc = 0,scale=1)
    large_psi_gamma = norm.cdf(gamma_f_i_star, loc = 0,scale =1)
    log_large_psi_gamma = np.log(large_psi_gamma)
    A = gamma_f_i_star*psi_gamma
    B = 2*large_psi_gamma
    temp = A / B - log_large_psi_gamma
    
    #print((integral_value / f_star_i_cdf[i] - np.log(f_star_i_cdf[i])).shape)
    for j in range(task_num):
        if task_feasible[j] == False:
            rate = np.c_[multivariate_normal.pdf(out_list[0], mean=Theta_mean_list[j].ravel(), cov=Theta_var_list[j]) / multivariate_normal.pdf(out_list[0], mean=Theta_mean_list[0].ravel(), cov=Theta_var_list[0])]
            temp[j] = rate*temp[j]
    mes = np.sum(temp, axis=1) / func_num
    # print(mes.shape)
    # save_data_lists = [X, mes]
    # data_name_lists = ["X", "alpha_list"]
    # for i in range(len(data_name_lists)):
    #     with open(save_data_path+data_name_lists[i]+".pickle", 'wb') as f:
    #         pickle.dump(save_data_lists[i], f) 
    # sys.exit()

    # plt.plot(X, task_entropy[0], 'g')
    # plt.plot(X, task_entropy[1], 'r')
    # plt.savefig("tmp.png")
    # # plt.show()
    # sys.exit()
    
    grid = np.linspace(lower_ppf, upper_ppf, grid_num, endpoint=False) +((upper_ppf - lower_ppf)/grid_num)/2 
    grid = grid.transpose((1, 2, 0, 3)).squeeze() #(4,200,800)
    
    gamma_f_t = (grid - pred_mean_task)/np.sqrt(pred_var_task)
    f_t_pdf = norm.pdf(gamma_f_t, loc=0, scale=1) / np.sqrt(pred_var_task) #(4,200,800)
    
    #print(np.diag(pred_matrix[len(X) * 0 : len(X) * (0 + 1), len(X) * 3 : len(X) * (3 + 1)]).shape)
    
    True_index = [0,98,99]
    for i in True_index: #かんそくできるやつだけ
        alpha_task[i] += task_entropy[i]
        y_star_dash = y_star_i[i][:, :, np.newaxis]
        y_star_dash = np.repeat(y_star_dash, grid_num, axis=2)  #(5,200,800)

        #sort
        abs = np.abs(task_feature - task_feature[i])
        sorted_index = np.argsort(abs[:,0])[:near_task_num]
        for j in sorted_index: #jのループ一回分があるtの獲得関数値
            if (i!=j):
                # conditional_mean = pred_mean_task[i] + np.c_[np.diag(pred_matrix[len(X) * i : len(X) * (i + 1), len(X) * j : len(X) * (j + 1)]) / np.diag(pred_matrix[len(X) * j : len(X) * (j + 1), len(X) * j : len(X) * (j + 1)])] * (grid[j] - pred_mean_task[j])
                # conditional_var = pred_var_task[i] - np.c_[np.diag(pred_matrix[len(X) * i : len(X) * (i + 1), len(X) * j : len(X) * (j + 1)]) / np.diag(pred_matrix[len(X) * j : len(X) * (j + 1), len(X) * j : len(X) * (j + 1)]) * np.diag(pred_matrix[len(X) * j : len(X) * (j + 1), len(X) * i : len(X) * (i + 1)])]
                # pred_matrixをなんとかしてdiagをはずした 9/6 修正：9/15

                conditional_mean = pred_mean_task[i] + np.c_[pred_matrix[i * task_num + j] / pred_matrix[j * task_num+j]] * (grid[j] - pred_mean_task[j])
                conditional_var = pred_var_task[i] - np.c_[pred_matrix[i * task_num + j] / pred_matrix[j * task_num + j] * pred_matrix[i + task_num * j]]
                conditional_var = np.tile(conditional_var, len(grid[j, 1]))  #(200,1) -> (200,800)
                conditional_var = conditional_var[np.newaxis,:,:]
                conditional_var = np.repeat(conditional_var, func_num, axis=0)  #(200,800) -> (5, 200,800)
                
                #sys.exit()
                conditional_mean = np.tile(conditional_mean, (func_num,1,1)) #(200,800) -> (5,200,800)
                conditional_gamma_f_i_star = (y_star_dash - conditional_mean)/np.sqrt(conditional_var)
                conditional_cdf = norm.cdf(conditional_gamma_f_i_star, loc=0, scale=1)
                
                f_t_pdf_dash = f_t_pdf[i][np.newaxis, :, :]
                
                f_t_pdf_dash = np.repeat(f_t_pdf_dash,func_num,axis=0) #(5,200,800)
                #print(f_t_pdf_dash.shape) #(4,200,800)
                integral_value = f_t_pdf_dash * conditional_cdf * (np.log(f_t_pdf_dash) + np.log(conditional_cdf))  #(5,200,800)
                np.nan_to_num(integral_value, copy=False)
                
                # 区分求積の処理
                #print(upper_ppf[i].shape)
                kubun = np.repeat((upper_ppf[i] - lower_ppf[i]) / grid_num, grid_num, axis=1)
                kubun = kubun[np.newaxis, :, :]
                kubun = np.repeat(kubun, func_num, axis=0)
                #print(kubun.shape)
                integral_value *= kubun
                integral_value = np.sum(integral_value, axis=2)
                rate = 1
                # #print((integral_value / f_star_i_cdf[i] - np.log(f_star_i_cdf[i])).shape)

                if task_feasible[j] == False:
                    rate = np.c_[multivariate_normal.pdf(out_list[2], mean=Theta_mean_list[j].ravel(), cov=Theta_var_list[j]) / multivariate_normal.pdf(out_list[2], mean=Theta_mean_list[2].ravel(), cov=Theta_var_list[2])]

                #print(rate.shape)
                integral_value = np.sum(rate * integral_value / f_star_i_cdf[i] - rate * np.log(f_star_i_cdf[i]), axis=0) / func_num
                #tmp = np.sum(- rate * np.log(f_star_i_cdf[i]), axis=0) / func_num
                #tmp = np.sum(rate*np.log(f_star_i_cdf[i]), axis=0) / func_num
                #print(tmp.shape)
                # if i == 0:
                #     #print("================")
                #     #print(rate)
                #     plt.plot(X, tmp, 'r')
                #     plt.savefig("tmp1.png")
                #     sys.exit()
                integral_value = np.c_[integral_value]
                # plt.plot(X, integral_value, "r")
                # plt.plot(X, alpha_task[i], "g")
                alpha_task[i] += integral_value
                # print(alpha_task[i])
                # plt.plot(X, alpha_task[i], "b")
                # plt.savefig("temp.png")
                # sys.exit()
                
                
            else:
                alpha_task[i] += np.c_[mes[j]]
    # print("ははは")
    # plt.plot(X, alpha_task[0], "b",label="Task1")
    # plt.plot(X, alpha_task[1], "r",label="Task2")
    # plt.legend(loc="lower left")
    # plt.savefig("all.png")
        
    '''
    plt.plot(X, mes[j], "r")
    plt.plot(X, alpha_task[0], "b")
    plt.savefig("temp.png")
    plt.close()
    sys.exit()
    '''
    
    return alpha_task

def hybrid_sum_mes(X, y_star, pred_mean_task_list, pred_var_task_list, func_num, pred_matrix,Theta_mean_list,Theta_var_list,out_list,task_feasible,save_data_path):
    task_num = len(pred_mean_task_list)
    grid_num = 500 #区分求積法のグリッド数
    pred_var_task = np.array(pred_var_task_list)  #リストをnumpyに変換 (4,200,1)
    pred_mean_task = np.array(pred_mean_task_list)  #リストをnumpyに変換 (4,200,1)
    y_star = np.array(y_star)  #リストをnumpyに変換 (4,5,1)
    task_entropy = (task_num - 1) * np.log(np.sqrt(2 * np.pi * np.e * pred_var_task))  # 第一項 (4,200,1) あえてT-1にしておく。MESをそのまま足したいので
    # plt.plot(X, task_entropy[0], "b",label="Task1")
    # plt.plot(X, task_entropy[1], "r", label="Task2")
    # plt.legend(loc="lower left")
    # plt.savefig("task_entropy.pdf")
    # plt.close()
    ppf_low = 1e-12 #e-7以降はcdfが0なってlogがバグる
    lower_ppf = norm.ppf(q=ppf_low, loc=pred_mean_task, scale=np.sqrt(pred_var_task))  #(4,200,1)
    upper_ppf = norm.ppf(q=1 - ppf_low, loc=pred_mean_task, scale=np.sqrt(pred_var_task)) #(4,200,1)
    alpha_task = np.zeros(task_entropy.shape)  #(4,200,1)

    y_star_i = np.tile(y_star, len(X))  #(4,5,200)
    #print(y_star_i.shape)
    pred_mean_i = np.tile(pred_mean_task, func_num) #(4,200,1) -> #(4,200,5)
    pred_mean_i = pred_mean_i.transpose((0, 2, 1))  #(4,5,200)
    
    pred_std_i = np.tile(np.sqrt(pred_var_task), func_num)
    pred_std_i = pred_std_i.transpose((0, 2, 1)) #(4,5,200)
    # print(y_star_i)
    # sys.exit()
    
    
    gamma_f_i_star = (y_star_i - pred_mean_i) / pred_std_i
    f_star_i_cdf = norm.cdf(gamma_f_i_star, loc=0, scale=1)
    
    # MES用
    psi_gamma = norm.pdf(gamma_f_i_star,loc = 0,scale=1)
    large_psi_gamma = norm.cdf(gamma_f_i_star, loc = 0,scale =1)
    log_large_psi_gamma = np.log(large_psi_gamma)
    A = gamma_f_i_star*psi_gamma
    B = 2*large_psi_gamma
    temp = A / B - log_large_psi_gamma
    
    #print((integral_value / f_star_i_cdf[i] - np.log(f_star_i_cdf[i])).shape)
    for j in range(task_num):
        if task_feasible[j] == False:
            rate = np.c_[multivariate_normal.pdf(out_list[0], mean=Theta_mean_list[j].ravel(), cov=Theta_var_list[j]) / multivariate_normal.pdf(out_list[0], mean=Theta_mean_list[0].ravel(), cov=Theta_var_list[0])]
            temp[j] = rate*temp[j]
    mes = np.sum(temp, axis=1) / func_num
    # print(mes.shape)
    # save_data_lists = [X, mes]
    # data_name_lists = ["X", "alpha_list"]
    # for i in range(len(data_name_lists)):
    #     with open(save_data_path+data_name_lists[i]+".pickle", 'wb') as f:
    #         pickle.dump(save_data_lists[i], f) 
    # sys.exit()

    # plt.plot(X, task_entropy[0], 'g')
    # plt.plot(X, task_entropy[1], 'r')
    # plt.savefig("tmp.png")
    # # plt.show()
    # sys.exit()
    
    grid = np.linspace(lower_ppf, upper_ppf, grid_num, endpoint=False) +((upper_ppf - lower_ppf)/grid_num)/2 
    grid = grid.transpose((1, 2, 0, 3)).squeeze() #(4,200,800)
    
    gamma_f_t = (grid - pred_mean_task)/np.sqrt(pred_var_task)
    f_t_pdf = norm.pdf(gamma_f_t, loc=0, scale=1) / np.sqrt(pred_var_task) #(4,200,800)
    
    #print(np.diag(pred_matrix[len(X) * 0 : len(X) * (0 + 1), len(X) * 3 : len(X) * (3 + 1)]).shape)
    for i in range(task_num):
        alpha_task[i] += task_entropy[i]
        y_star_dash = y_star_i[i][:, :, np.newaxis]
        y_star_dash = np.repeat(y_star_dash, grid_num, axis=2)  #(5,200,800)
        for j in range(task_num): #jのループ一回分があるtの獲得関数値
            if (i!=j):
                # conditional_mean = pred_mean_task[i] + np.c_[np.diag(pred_matrix[len(X) * i : len(X) * (i + 1), len(X) * j : len(X) * (j + 1)]) / np.diag(pred_matrix[len(X) * j : len(X) * (j + 1), len(X) * j : len(X) * (j + 1)])] * (grid[j] - pred_mean_task[j])
                # conditional_var = pred_var_task[i] - np.c_[np.diag(pred_matrix[len(X) * i : len(X) * (i + 1), len(X) * j : len(X) * (j + 1)]) / np.diag(pred_matrix[len(X) * j : len(X) * (j + 1), len(X) * j : len(X) * (j + 1)]) * np.diag(pred_matrix[len(X) * j : len(X) * (j + 1), len(X) * i : len(X) * (i + 1)])]
                # pred_matrixをなんとかしてdiagをはずした 9/6 修正：9/15

                conditional_mean = pred_mean_task[i] + np.c_[pred_matrix[i * task_num + j] / pred_matrix[j * task_num+j]] * (grid[j] - pred_mean_task[j])
                conditional_var = pred_var_task[i] - np.c_[pred_matrix[i * task_num + j] / pred_matrix[j * task_num + j] * pred_matrix[i + task_num * j]]
                conditional_var = np.tile(conditional_var, len(grid[j, 1]))  #(200,1) -> (200,800)
                conditional_var = conditional_var[np.newaxis,:,:]
                conditional_var = np.repeat(conditional_var, func_num, axis=0)  #(200,800) -> (5, 200,800)
                
                #sys.exit()
                conditional_mean = np.tile(conditional_mean, (func_num,1,1)) #(200,800) -> (5,200,800)
                conditional_gamma_f_i_star = (y_star_dash - conditional_mean)/np.sqrt(conditional_var)
                conditional_cdf = norm.cdf(conditional_gamma_f_i_star, loc=0, scale=1)
                
                f_t_pdf_dash = f_t_pdf[i][np.newaxis, :, :]
                
                f_t_pdf_dash = np.repeat(f_t_pdf_dash,func_num,axis=0) #(5,200,800)
                #print(f_t_pdf_dash.shape) #(4,200,800)
                integral_value = f_t_pdf_dash * conditional_cdf * (np.log(f_t_pdf_dash) + np.log(conditional_cdf))  #(5,200,800)
                np.nan_to_num(integral_value, copy=False)
                
                # 区分求積の処理
                #print(upper_ppf[i].shape)
                kubun = np.repeat((upper_ppf[i] - lower_ppf[i]) / grid_num, grid_num, axis=1)
                kubun = kubun[np.newaxis, :, :]
                kubun = np.repeat(kubun, func_num, axis=0)
                #print(kubun.shape)
                integral_value *= kubun
                integral_value = np.sum(integral_value, axis=2)
                rate = 1
                # #print((integral_value / f_star_i_cdf[i] - np.log(f_star_i_cdf[i])).shape)

                if task_feasible[j] == False:
                    rate = np.c_[multivariate_normal.pdf(out_list[2], mean=Theta_mean_list[j].ravel(), cov=Theta_var_list[j]) / multivariate_normal.pdf(out_list[2], mean=Theta_mean_list[2].ravel(), cov=Theta_var_list[2])]

                #print(rate.shape)
                integral_value = np.sum(rate * integral_value / f_star_i_cdf[i] - rate * np.log(f_star_i_cdf[i]), axis=0) / func_num
                #tmp = np.sum(- rate * np.log(f_star_i_cdf[i]), axis=0) / func_num
                #tmp = np.sum(rate*np.log(f_star_i_cdf[i]), axis=0) / func_num
                #print(tmp.shape)
                # if i == 0:
                #     #print("================")
                #     #print(rate)
                #     plt.plot(X, tmp, 'r')
                #     plt.savefig("tmp1.png")
                #     sys.exit()
                integral_value = np.c_[integral_value]
                # plt.plot(X, integral_value, "r")
                # plt.plot(X, alpha_task[i], "g")
                alpha_task[i] += integral_value
                # print(alpha_task[i])
                # plt.plot(X, alpha_task[i], "b")
                # plt.savefig("temp.png")
                # sys.exit()
                
                
            else:
                alpha_task[i] += np.c_[mes[j]]
    # print("ははは")
    # plt.plot(X, alpha_task[0], "b",label="Task1")
    # plt.plot(X, alpha_task[1], "r",label="Task2")
    # plt.legend(loc="lower left")
    # plt.savefig("all.png")
    # sys.exit()
        
    '''
    plt.plot(X, mes[j], "r")
    plt.plot(X, alpha_task[0], "b")
    plt.savefig("temp.png")
    plt.close()
    sys.exit()
    '''
    
    return alpha_task

def imp_sum_mt_mes_perfect(X,y_star, pred_mean_task_list, pred_var_task_list, func_num, pred_matrix):

    task_num = len(pred_mean_task_list)
    grid_num = 100 #区分求積法のグリッド数
    pred_var_task = np.array(pred_var_task_list)  #リストをnumpyに変換 (4,200,1)
    pred_mean_task = np.array(pred_mean_task_list)  #リストをnumpyに変換 (4,200,1)
    
    y_star = np.array(y_star)  #リストをnumpyに変換 (4,5,1)
    task_entropy = (task_num - 1) * np.log(np.sqrt(2 * np.pi * np.e * pred_var_task))  # 第一項 (4,200,1) あえてT-1にしておく。MESをそのまま足したいので
    # plt.plot(X, task_entropy[0], "b",label="Task1")
    # plt.plot(X, task_entropy[1], "r", label="Task2")
    # plt.legend(loc="lower left")
    # plt.savefig("task_entropy.pdf")
    # plt.close()
    ppf_low = 1e-12 #e-7以降はcdfが0なってlogがバグる
    lower_ppf = norm.ppf(q=ppf_low, loc=pred_mean_task, scale=np.sqrt(pred_var_task))  #(4,200,1)
    upper_ppf = norm.ppf(q=1 - ppf_low, loc=pred_mean_task, scale=np.sqrt(pred_var_task)) #(4,200,1)
    alpha_task = np.zeros(task_entropy.shape)  #(4,200,1)
    y_star_i = np.tile(y_star,len(X))  #(4,5,200)

    #print(y_star_i.shape)
    pred_mean_i = np.tile(pred_mean_task, func_num) #(4,200,1) -> #(4,200,5)
    pred_mean_i = pred_mean_i.transpose((0, 2, 1))  #(4,5,200)
    
    pred_std_i = np.tile(np.sqrt(pred_var_task), func_num)
    pred_std_i = pred_std_i.transpose((0, 2, 1)) #(4,5,200)
    
    gamma_f_i_star = (y_star_i - pred_mean_i) / pred_std_i
    f_star_i_cdf = norm.cdf(gamma_f_i_star, loc=0, scale=1)
    
    # MES用
    psi_gamma = norm.pdf(gamma_f_i_star,loc = 0,scale=1)
    large_psi_gamma = norm.cdf(gamma_f_i_star, loc = 0,scale =1)
    log_large_psi_gamma = np.log(large_psi_gamma)
    A = gamma_f_i_star*psi_gamma
    B = 2*large_psi_gamma
    temp = A / B - log_large_psi_gamma

    #print((integral_value / f_star_i_cdf[i] - np.log(f_star_i_cdf[i])).shape)
    mes = np.sum(temp, axis=1) / func_num

    # save_data_lists = [X, mes]
    # data_name_lists = ["X", "alpha_list"]
    # for i in range(len(data_name_lists)):
    #     with open(save_data_path+data_name_lists[i]+".pickle", 'wb') as f:
    #         pickle.dump(save_data_lists[i], f) 
    # sys.exit()

    # print(mes)
    # plt.plot(X, task_entropy[0], 'g')
    # plt.plot(X, task_entropy[1], 'r')
    # plt.savefig("tmp.png")
    # # plt.show()
    # sys.exit()
    
    grid = np.linspace(lower_ppf, upper_ppf, grid_num, endpoint=False) +((upper_ppf - lower_ppf)/grid_num)/2 
    grid = grid.transpose((1, 2, 0, 3)).squeeze() #(4,200,800)
    
    gamma_f_t = (grid - pred_mean_task)/np.sqrt(pred_var_task)
    f_t_pdf = norm.pdf(gamma_f_t, loc=0, scale=1) / np.sqrt(pred_var_task) #(4,200,800)
    
    #print(np.diag(pred_matrix[len(X) * 0 : len(X) * (0 + 1), len(X) * 3 : len(X) * (3 + 1)]).shape)
    for i in range(task_num):
        alpha_task[i] += task_entropy[i]
        y_star_dash = y_star_i[i][:, :, np.newaxis]
        y_star_dash = np.repeat(y_star_dash, grid_num, axis=2)  #(5,200,800)
        
        #print(sorted_index)
        for j in range(task_num): #jのループ一回分があるtの獲得関数値
            #print(j)
            if (i!=j):
                # conditional_mean = pred_mean_task[i] + np.c_[np.diag(pred_matrix[len(X) * i : len(X) * (i + 1), len(X) * j : len(X) * (j + 1)]) / np.diag(pred_matrix[len(X) * j : len(X) * (j + 1), len(X) * j : len(X) * (j + 1)])] * (grid[j] - pred_mean_task[j])
                # conditional_var = pred_var_task[i] - np.c_[np.diag(pred_matrix[len(X) * i : len(X) * (i + 1), len(X) * j : len(X) * (j + 1)]) / np.diag(pred_matrix[len(X) * j : len(X) * (j + 1), len(X) * j : len(X) * (j + 1)]) * np.diag(pred_matrix[len(X) * j : len(X) * (j + 1), len(X) * i : len(X) * (i + 1)])]
                # pred_matrixをなんとかしてdiagをはずした 9/6 修正：9/15

                conditional_mean = pred_mean_task[i] + np.c_[pred_matrix[i * task_num + j] / pred_matrix[j * task_num+j]] * (grid[j] - pred_mean_task[j])
                conditional_var = pred_var_task[i] - np.c_[pred_matrix[i * task_num + j] / pred_matrix[j * task_num + j] * pred_matrix[i + task_num * j]]
                conditional_var = np.tile(conditional_var, len(grid[j, 1]))  #(200,1) -> (200,800)
                conditional_var = conditional_var[np.newaxis,:,:]
                conditional_var = np.repeat(conditional_var, func_num, axis=0)  #(200,800) -> (5, 200,800)
                
                #sys.exit()
                conditional_mean = np.tile(conditional_mean, (func_num,1,1)) #(200,800) -> (5,200,800)
                conditional_gamma_f_i_star = (y_star_dash - conditional_mean)/np.sqrt(conditional_var)
                conditional_cdf = norm.cdf(conditional_gamma_f_i_star, loc=0, scale=1)
                
                f_t_pdf_dash = f_t_pdf[i][np.newaxis, :, :]
                
                f_t_pdf_dash = np.repeat(f_t_pdf_dash,func_num,axis=0) #(5,200,800)
                #print(f_t_pdf_dash.shape) #(4,200,800)
                integral_value = f_t_pdf_dash * conditional_cdf * (np.log(f_t_pdf_dash) + np.log(conditional_cdf))  #(5,200,800)
                np.nan_to_num(integral_value, copy=False)
                
                # 区分求積の処理
                #print(upper_ppf[i].shape)
                kubun = np.repeat((upper_ppf[i] - lower_ppf[i]) / grid_num, grid_num, axis=1)
                kubun = kubun[np.newaxis, :, :]
                kubun = np.repeat(kubun, func_num, axis=0)
                #print(kubun.shape)
                integral_value *= kubun
                integral_value = np.sum(integral_value, axis=2)
                
                # #print((integral_value / f_star_i_cdf[i] - np.log(f_star_i_cdf[i])).shape)

                # if task_feasible[j] == False:
                #     rate = np.c_[multivariate_normal.pdf(out_list[0], mean=Theta_mean_list[j].ravel(), cov=Theta_var_list[j]) / multivariate_normal.pdf(out_list[0], mean=Theta_mean_list[0].ravel(), cov=Theta_var_list[0])]

                #print(rate.shape)
                integral_value = np.sum(integral_value / f_star_i_cdf[i] - np.log(f_star_i_cdf[i]), axis=0) / func_num
                #tmp = np.sum(- rate * np.log(f_star_i_cdf[i]), axis=0) / func_num
                #tmp = np.sum(rate*np.log(f_star_i_cdf[i]), axis=0) / func_num
                #print(tmp.shape)
                # if i == 0:
                #     #print("================")
                #     #print(rate)
                #     plt.plot(X, tmp, 'r')
                #     plt.savefig("tmp1.png")
                #     sys.exit()
                integral_value = np.c_[integral_value]
                # plt.plot(X, integral_value, "r")
                # plt.plot(X, alpha_task[i], "g")
                alpha_task[i] += integral_value
                # print(alpha_task[i])
                # plt.plot(X, alpha_task[i], "b")
                # plt.savefig("temp.png")
            else:
                alpha_task[i] += np.c_[mes[j]]
        #sys.exit()
    # print("ははは")
    # plt.plot(X, alpha_task[0], "b",label="Task1")
    # plt.plot(X, alpha_task[1], "r",label="Task2")
    # plt.legend(loc="lower left")
    # plt.savefig("all.png")
    # sys.exit()
        
    '''
    plt.plot(X, mes[j], "r")
    plt.plot(X, alpha_task[0], "b")
    plt.savefig("temp.png")
    plt.close()
    sys.exit()
    '''
    
    return alpha_task

def imp_sum_mt_mes_perfect_approximate(X, y_star, pred_mean_task_list, pred_var_task_list, func_num, pred_matrix,Theta_mean_list,Theta_var_list,out_list,task_feasible,save_data_path):
    task_num = len(pred_mean_task_list)
    grid_num = 200 #区分求積法のグリッド数
    pred_var_task = np.array(pred_var_task_list)  #リストをnumpyに変換 (4,200,1)
    pred_mean_task = np.array(pred_mean_task_list)  #リストをnumpyに変換 (4,200,1)
    y_star = np.array(y_star)  #リストをnumpyに変換 (4,5,1)
    task_entropy = (task_num - 1) * np.log(np.sqrt(2 * np.pi * np.e * pred_var_task))  # 第一項 (4,200,1) あえてT-1にしておく。MESをそのまま足したいので
    # plt.plot(X, task_entropy[0], "b",label="Task1")
    # plt.plot(X, task_entropy[1], "r", label="Task2")
    # plt.legend(loc="lower left")
    # plt.savefig("task_entropy.pdf")
    # plt.close()
    ppf_low = 1e-12 #e-7以降はcdfが0なってlogがバグる
    lower_ppf = norm.ppf(q=ppf_low, loc=pred_mean_task, scale=np.sqrt(pred_var_task))  #(4,200,1)
    upper_ppf = norm.ppf(q=1 - ppf_low, loc=pred_mean_task, scale=np.sqrt(pred_var_task)) #(4,200,1)
    alpha_task = np.zeros(task_entropy.shape)  #(4,200,1)

    y_star_i = np.tile(y_star, len(X))  #(4,5,200)
    #print(y_star_i.shape)
    pred_mean_i = np.tile(pred_mean_task, func_num) #(4,200,1) -> #(4,200,5)
    pred_mean_i = pred_mean_i.transpose((0, 2, 1))  #(4,5,200)
    
    pred_std_i = np.tile(np.sqrt(pred_var_task), func_num)
    pred_std_i = pred_std_i.transpose((0, 2, 1)) #(4,5,200)

    gamma_f_i_star = (y_star_i - pred_mean_i) / pred_std_i
    f_star_i_cdf = norm.cdf(gamma_f_i_star, loc=0, scale=1)
    
    # MES用
    psi_gamma = norm.pdf(gamma_f_i_star,loc = 0,scale=1)
    large_psi_gamma = norm.cdf(gamma_f_i_star, loc = 0,scale =1)
    log_large_psi_gamma = np.log(large_psi_gamma)
    A = gamma_f_i_star*psi_gamma
    B = 2*large_psi_gamma
    temp = A / B - log_large_psi_gamma

    #print((integral_value / f_star_i_cdf[i] - np.log(f_star_i_cdf[i])).shape)
    mes = np.sum(temp, axis=1) / func_num

    # save_data_lists = [X, mes]
    # data_name_lists = ["X", "alpha_list"]
    # for i in range(len(data_name_lists)):
    #     with open(save_data_path+data_name_lists[i]+".pickle", 'wb') as f:
    #         pickle.dump(save_data_lists[i], f) 
    # sys.exit()

    # print(mes)
    # plt.plot(X, task_entropy[0], 'g')
    # plt.plot(X, task_entropy[1], 'r')
    # plt.savefig("tmp.png")
    # # plt.show()
    # sys.exit()
    
    grid = np.linspace(lower_ppf, upper_ppf, grid_num, endpoint=False) +((upper_ppf - lower_ppf)/grid_num)/2 
    grid = grid.transpose((1, 2, 0, 3)).squeeze() #(4,200,800)
    
    gamma_f_t = (grid - pred_mean_task)/np.sqrt(pred_var_task)
    f_t_pdf = norm.pdf(gamma_f_t, loc=0, scale=1) / np.sqrt(pred_var_task) #(4,200,800)
    
    #print(np.diag(pred_matrix[len(X) * 0 : len(X) * (0 + 1), len(X) * 3 : len(X) * (3 + 1)]).shape)
    for i in range(task_num):
        alpha_task[i] += task_entropy[i]
        y_star_dash = y_star_i[i][:, :, np.newaxis]
        y_star_dash = np.repeat(y_star_dash, grid_num, axis=2)  #(5,200,800)
        for j in range(task_num): #jのループ一回分があるtの獲得関数値
            if (i!=j):
                # conditional_mean = pred_mean_task[i] + np.c_[np.diag(pred_matrix[len(X) * i : len(X) * (i + 1), len(X) * j : len(X) * (j + 1)]) / np.diag(pred_matrix[len(X) * j : len(X) * (j + 1), len(X) * j : len(X) * (j + 1)])] * (grid[j] - pred_mean_task[j])
                # conditional_var = pred_var_task[i] - np.c_[np.diag(pred_matrix[len(X) * i : len(X) * (i + 1), len(X) * j : len(X) * (j + 1)]) / np.diag(pred_matrix[len(X) * j : len(X) * (j + 1), len(X) * j : len(X) * (j + 1)]) * np.diag(pred_matrix[len(X) * j : len(X) * (j + 1), len(X) * i : len(X) * (i + 1)])]
                # pred_matrixをなんとかしてdiagをはずした 9/6 修正：9/15

                conditional_mean = pred_mean_task[i] + np.c_[pred_matrix[i * task_num + j] / pred_matrix[j * task_num+j]] * (grid[j] - pred_mean_task[j])
                conditional_var = pred_var_task[i] - np.c_[pred_matrix[i * task_num + j] / pred_matrix[j * task_num + j] * pred_matrix[i + task_num * j]]
                conditional_var = np.tile(conditional_var, len(grid[j, 1]))  #(200,1) -> (200,800)
                conditional_var = conditional_var[np.newaxis,:,:]
                conditional_var = np.repeat(conditional_var, func_num, axis=0)  #(200,800) -> (5, 200,800)
                
                #sys.exit()
                conditional_mean = np.tile(conditional_mean, (func_num,1,1)) #(200,800) -> (5,200,800)
                conditional_gamma_f_i_star = (y_star_dash - conditional_mean)/np.sqrt(conditional_var)
                conditional_cdf = norm.cdf(conditional_gamma_f_i_star, loc=0, scale=1)
                
                f_t_pdf_dash = f_t_pdf[i][np.newaxis, :, :]
                
                f_t_pdf_dash = np.repeat(f_t_pdf_dash,func_num,axis=0) #(5,200,800)
                #print(f_t_pdf_dash.shape) #(4,200,800)
                integral_value = f_t_pdf_dash * conditional_cdf * (np.log(f_t_pdf_dash) + np.log(conditional_cdf))  #(5,200,800)
                np.nan_to_num(integral_value, copy=False)
                
                # 区分求積の処理
                #print(upper_ppf[i].shape)
                kubun = np.repeat((upper_ppf[i] - lower_ppf[i]) / grid_num, grid_num, axis=1)
                kubun = kubun[np.newaxis, :, :]
                kubun = np.repeat(kubun, func_num, axis=0)
                #print(kubun.shape)
                integral_value *= kubun
                integral_value = np.sum(integral_value, axis=2)
                rate = 1
                # #print((integral_value / f_star_i_cdf[i] - np.log(f_star_i_cdf[i])).shape)

                # if task_feasible[j] == False:
                #     rate = np.c_[multivariate_normal.pdf(out_list[0], mean=Theta_mean_list[j].ravel(), cov=Theta_var_list[j]) / multivariate_normal.pdf(out_list[0], mean=Theta_mean_list[0].ravel(), cov=Theta_var_list[0])]

                #print(rate.shape)
                integral_value = np.sum(rate * integral_value / f_star_i_cdf[i] - rate * np.log(f_star_i_cdf[i]), axis=0) / func_num
                #tmp = np.sum(- rate * np.log(f_star_i_cdf[i]), axis=0) / func_num
                #tmp = np.sum(rate*np.log(f_star_i_cdf[i]), axis=0) / func_num
                #print(tmp.shape)
                # if i == 0:
                #     #print("================")
                #     #print(rate)
                #     plt.plot(X, tmp, 'r')
                #     plt.savefig("tmp1.png")
                #     sys.exit()
                integral_value = np.c_[integral_value]
                # plt.plot(X, integral_value, "r")
                # plt.plot(X, alpha_task[i], "g")
                alpha_task[i] += integral_value
                # print(alpha_task[i])
                # plt.plot(X, alpha_task[i], "b")
                # plt.savefig("temp.png")
                # sys.exit()
                
                
            else:
                alpha_task[i] += np.c_[mes[j]]
    # print("ははは")
    # plt.plot(X, alpha_task[0], "b",label="Task1")
    # plt.plot(X, alpha_task[1], "r",label="Task2")
    # plt.legend(loc="lower left")
    # plt.savefig("all.png")
    # sys.exit()
        
    '''
    plt.plot(X, mes[j], "r")
    plt.plot(X, alpha_task[0], "b")
    plt.savefig("temp.png")
    plt.close()
    sys.exit()
    '''
    
    return alpha_task, mes

def predict(kernel, XX, X_train, y_train):
    noise_var = 1.0e-4
    Sigma = kernel.K(X_train, X_train) + noise_var * np.eye(X_train.shape[0])
    Sigma_inv = np.linalg.inv(Sigma)
    k_star = kernel.K(XX, X_train)
    temp = np.dot(k_star, Sigma_inv)
    pred_mean = np.dot(temp, y_train)
    return pred_mean

def kg(kernel, X, XX, X_train_list, y_train_list, gpy_pred_mean_task_list,gpy_pred_var_task_list, max_pred_mu,task_feature,grid_num,near_task_num):
    sample_size = 10
    sample_task_num = 6
    task_num_total = len(gpy_pred_mean_task_list)
    alpha_task = np.zeros((task_num_total,X.shape[0]))
    sampled_task_index = random.sample(range(task_num_total),sample_task_num)
    for i in sampled_task_index: #タスク数のループ
        abs = np.abs(task_feature - task_feature[i])
        sorted_index = np.argsort(abs[:,0])[:near_task_num]
        XX_dummy = np.empty((0,XX.shape[1]))
        for j in sorted_index:
            XX_dummy = np.vstack((XX_dummy, XX[j*len(X):(j+1)*len(X)]))
        for column in range(len(X)):
            if X[column] in X_train_list[i]:
                alpha_task[i][column] = 0
            else:
                samples = np.random.normal(loc= gpy_pred_mean_task_list[i][column], scale = np.sqrt(gpy_pred_var_task_list[i][column]), size = sample_size)
                #x_pred_mean = np.zeros((XX.shape[0],1))
                X_dammy_list = copy.deepcopy(X_train_list)
                X_dammy_list[i] = np.vstack((X_train_list[i], X[column]))
                XX_train = np.empty((0, X.shape[1]+task_feature.shape[1]))
                for j in range(len(X_dammy_list)):
                    XX_train = np.vstack((XX_train, np.hstack((X_dammy_list[j],np.tile(task_feature[j],(X_dammy_list[j].shape[0],1))))))
                
                pred_mean_max_list = [0] * near_task_num
                for row in range(sample_size):
                    y_dammy_list = copy.deepcopy(y_train_list)
                    y_dammy_list[i] = np.vstack((y_dammy_list[i], samples[row]))
                    y_train = np.empty((0, 1))
                    for input in y_dammy_list:
                        y_train = np.vstack((y_train, input))
                    gpy_pred_mean = predict(kernel,XX_dummy,XX_train,y_train)
                    pred_mean_max_list = [k + l for (k,l) in zip(pred_mean_max_list , [gpy_pred_mean[grid_num * task: grid_num * (task + 1)].max() for task in range(near_task_num)])]
                
                expected_max_list = list(map(lambda x: x / sample_size, pred_mean_max_list))
                alpha_task[i][column] = sum(expected_max_list - max_pred_mu[sorted_index])
            
    return alpha_task