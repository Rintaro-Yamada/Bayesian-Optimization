# coding: utf-8
'''
##実装メモ
あとからタスクが増やせるようにする
'''

import GPy
import numpy as np
import random
import sys
import subprocess
import pickle
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler
from ThetaGenerator import ThetaGenerator
from scipy.stats import multivariate_normal
from function_generator import ReImportanceSampling, ReMultiFunctionGenerator
from f_plot import multi_func_plot,plot_alpha_multi, d1_plot, d2_plot,multi_imp_plot,plot_alpha_multi_ystar
import acquisition as acq
from pyDOE import lhs
import os
import glob
import pandas as pd
import f_plot
import scipy.stats as ss

def predict_domain_diag(model, XX, XX_train, y_train, task_num_total, grid_num):
    K_inv = np.linalg.inv(model.kern.K(XX_train)+ model['Gaussian_noise.variance'][0]*np.eye(XX_train.shape[0]))
    k_star = model.kern.K(XX_train, XX)
    k_star_trans_K_inv = np.dot(k_star.T, K_inv)
    pred_matrix = []
    for row in range(task_num_total):
        for col in range(task_num_total):
            pred_matrix.append(np.c_[np.einsum("ij,ji->i", k_star_trans_K_inv[row * grid_num : (row + 1) * grid_num, :], k_star[:, col * grid_num : (col + 1) * grid_num])])
    pred_matrix = np.array(pred_matrix).squeeze()
    star_index = np.arange(task_num_total) * grid_num
    k_star_star = model.kern.K(XX[star_index])
    k_star_star = k_star_star.reshape(pred_matrix.shape[0], 1)
    pred_matrix = k_star_star - pred_matrix
    # normalizer == False のときは以下は不必要
    y_var = np.var(y_train)
    pred_matrix = pred_matrix*y_var
    return pred_matrix

def artificial_data_import(init_num, input_dim,task_feature, task_num_total, func_seed, variance, length_scale, noise_var, task_lengthscale, task_variance):
    if real_data == True:
        sys.exit()
    else:
        grid_num = init_num ** input_dim

        ### 1d input ###
        if input_dim ==1:
            X = np.c_[np.linspace(0, 1, init_num)]

        ### 2d input ###
        elif input_dim ==2:
            X = np.c_[np.linspace(0, 1, init_num)]
            xx , yy = np.meshgrid(X,X)
            aa = xx.reshape(grid_num,1)
            bb = yy.reshape(grid_num,1)
            X = np.hstack((bb, aa))

        ### 3d input ###
        elif input_dim ==3:
            X = np.c_[np.linspace(0, 0.1, init_num)]
            xx , yy, zz= np.meshgrid(X,X,X)
            aa = xx.reshape(grid_num,1)
            bb = yy.reshape(grid_num, 1)
            cc = zz.reshape(grid_num, 1)
            X = np.hstack((bb, aa))
            X = np.hstack((X, cc))

        XX = np.zeros((0, X.shape[1]+task_feature.shape[1]))

        for i in range(task_num_total):
            t = np.tile(task_feature[i], (X.shape[0], 1))
            XX_tmp = np.hstack((X, t))
            XX = np.vstack((XX, XX_tmp))

        np.random.seed(func_seed)
        random.seed(func_seed)

        kernel_gpy = GPy.kern.RBF(input_dim=X.shape[1],lengthscale=length_scale,variance=variance, active_dims=list(range(XX.shape[1]-1)))
        task_kernel_gpy = GPy.kern.RBF(input_dim=task_feature.shape[1],lengthscale=task_lengthscale,variance=task_variance, active_dims=list(range(XX.shape[1]-1, XX.shape[1])), name="task_kern")
        kern = kernel_gpy * task_kernel_gpy

        ### Random Feature (ライブラリ任せ)
        # dim = 1000
        # rbf_feature = RBFSampler(gamma=np.c_[np.array([1/(2*length_scale**2),1/(2*task_lengthscale**2)])], n_components=dim, random_state=1)
        # features = rbf_feature.fit_transform(XX)
        # Theta = ThetaGenerator(func_seed,dim, noise_var)
        # Theta.calc_init(features)
        # func_num =1
        # theta=Theta.getTheta(func_num)
        # f = np.dot(theta, features.T)

        ### 同時分布から
        mu = np.zeros((XX.shape[0],XX.shape[1]-1))
        var = kern.K(XX)
        f = np.random.multivariate_normal(mu.ravel(), var, size = 1)
        
        ### テスト関数を生成(旧)
        # sampled_functions = ReMultiFunctionGenerator(
        #     seed=func_seed, lengthscale=length_scale, task_lengthscale=task_lengthscale, variance=variance*task_variance, noise_var=noise_var, XX=XX)
        
        # func_num_total = 1
        # f = sampled_functions.gen_prior(func_num=func_num_total)

        y_list = [f[0][grid_num * i: grid_num *
            (i + 1)][:, None] for i in range(task_num_total)]

        # 1d test-function plot
        # d1_plot(X,y_list)
        # sys.exit()

        # 2d test-function plot
        # d2_plot(X,y_list,xx,yy,init_num)
        # sys.exit()

        return grid_num, X, XX, y_list,kern

# def predict_cov(kernel, XX, X_train, y_train):
#     noise_var = 1.0e-4
#     Sigma = kernel.K(X_train, X_train) + noise_var * np.eye(X_train.shape[0])
#     Sigma_inv = np.linalg.inv(Sigma)
#     k_star = kernel.K(XX, X_train)
#     temp = np.dot(k_star, Sigma_inv)
#     k_star_star = kernel.K(XX)
#     return pred_mean

def real_data_import(task_num_total):
    sep = os.sep
    data_dir = "data" # 最大37
    GB_list = np.sort(glob.glob("." + sep + data_dir + sep + "*"))
    data_dir = len(data_dir)
    task_feature = np.empty(0)
    y_list = []
    cost_list = []
    start_pos = 6 + data_dir
    end_pos = start_pos + 3
    # partition = 240 #240はX[0] == 0のデータ数
    
    for i in range(task_num_total):
        with open(GB_list[i], 'rb') as f:
            task_feature = np.append(task_feature, int(GB_list[i][start_pos:end_pos]))
            gb = pickle.load(f, encoding='latin1')
            # X = gb["input_des"][:partition]
            # X = np.delete(X,0,1)
            X = gb["input_des"]
            cost = np.c_[gb["cost"]]
            des = gb["task_des"]
            print(des.shape)
            sys.exit()
            # y = np.c_[-gb["Ene"]][:partition]
            y = np.c_[-gb["Ene"]]
            y[np.isnan(y)] = np.nanmedian(y)
			#print(len(y))
            y_list.append(y)
            cost_list.append(cost)
    task = np.c_[task_feature]
    grid_num = len(X)
    XX = np.zeros((0, X.shape[1]+task.shape[1]))
    for i in range(task_num_total):
        t = np.tile(task[i], (X.shape[0], 1))
        XX_tmp = np.hstack((X, t))
        XX = np.vstack((XX, XX_tmp))

    return grid_num, X, XX, y_list, task, cost_list

def zscore(y_list):
    func_num = len(y_list)
    zscore = []
    for i in range(func_num):
        mean = np.mean(y_list[i])
        std  = np.std(y_list[i])
        zscore.append((y_list[i]-mean)/std)
    return zscore

def experiment(seed, initial_num_list, max_iter, acq_name, func_seed,task_feature,task_observable):
    task_num_total = len(initial_num_list)
    #人工データ
    if real_data == False:
        # データの生成
        init_num = 200
        input_dim = 1
        
        # testfunc param
        variance = 1.0
        task_variance = 1.0
        length_scale = 0.1
        noise_var = 1.0e-4
        task_lengthscale = 2.0
        
        grid_num, X, XX, y_list,kern = artificial_data_import(init_num, input_dim,task_feature, task_num_total, func_seed, variance, length_scale, noise_var, task_lengthscale, task_variance)

    #実データ
    else:
        grid_num , X, XX, y_list, task_feature, cost_list = real_data_import(task_num_total)
    
    #正規化 超重要！！！！
    # y_list = zscore(y_list)


    # 実データのデータセット生成
    '''
    tmp = 100
    index = 0
    wanted_task_num = 10
    for i in range(len(task_feature)-(wanted_task_num-1)):
        kensyou = task_feature[i+(wanted_task_num-1)]-task_feature[i]
        if tmp > (kensyou):
            tmp = kensyou
            index = i
    print(tmp)
    print(index)
    sys.exit()
    '''

    # 初期値設定
    np.random.seed(seed)
    random.seed(seed)

    # if initial_random == True:
    #     initial_num = 1
    #     for i in range(initial_num):
    #         next = np.random.randint(task_num_total)
    #         initial_num_list[next] = initial_num_list[next] + 1

    # if initial_random == True:
    #     initial_num = 5
    #     # next = 0
    #     for i in range(initial_num):
    #         next = random.randint(0,17)
    #         initial_num_list[next] = initial_num_list[next] + 1
        # next= 98
        # for i in range(1):
        #     initial_num_list[next] = initial_num_list[next] + 1
        # # next= 99
        # for i in range(1):
        #     initial_num_list[next] = initial_num_list[next] + 1
    index_list = range(grid_num)
    # randomサンプリング
    train_index = [random.sample(index_list, i) for i in initial_num_list]
    
    
    # lhsサンプリング
    # sum_init_num = sum(initial_num_list)
    # train_tmp = ((grid_num * lhs(sum_init_num, 1)).astype(int)).ravel().tolist()  
    # init = 0
    # train_index = []
    # for i in range(len(initial_num_list)):
    #     train_index.append(train_tmp[init:init+initial_num_list[i]])
    #     init += initial_num_list[i]

    X_train_list = [X[i] for i in train_index]
    y_train_list = [y_list[t][i]
        for (t, i) in zip(range(task_num_total), train_index)]

    XX_train = np.empty((0, X.shape[1]+task_feature.shape[1]))

    for i in range(len(X_train_list)):
        XX_train = np.vstack((XX_train, np.hstack((X_train_list[i],np.tile(task_feature[i],(X_train_list[i].shape[0],1))))))
    y_train = np.empty((0, 1))
    for input in y_train_list:
        y_train = np.vstack((y_train, input))

    if GP_type == 'single':
        model = []
        kernel_gpy = GPy.kern.RBF(input_dim = XX_train.shape[1]-task_feature.shape[1],lengthscale=length_scale,variance=variance)
    
    # 学習用モデルの設計
    if real_data == False:
        model = GPy.models.GPRegression(
            XX_train, y_train, kernel=kern,normalizer=True)
        model['.*Gaussian_noise.variance'].constrain_fixed(noise_var)
        model['mul.rbf.variance'].constrain_fixed(1.0)
        model['mul.task_kern.variance'].constrain_fixed(1.0)
        
    else:
        noise_var = 1.0e-4
        kernel = GPy.kern.RBF(
		input_dim=XX_train.shape[1]-task_feature.shape[1], active_dims=list(range(XX_train.shape[1]-1)),ARD=True)
        
        task_kernel = GPy.kern.RBF(
            input_dim=task_feature.shape[1], active_dims=list(range(XX_train.shape[1]-task_feature.shape[1], XX_train.shape[1])), name="task_kern",ARD=True)
        kern = kernel * task_kernel
        model = GPy.models.GPRegression(
            XX_train, y_train, kernel=kern, normalizer=True)
        model['.*Gaussian_noise.variance'].constrain_fixed(noise_var)
        model['mul.rbf.variance'].constrain_fixed(1.0)
        model['mul.task_kern.variance'].constrain_fixed(1.0)
        #model['mul.rbf.lengthscale'].constrain_fixed([30, 100, 40]) # median huristic
        #model['mul.rbf.lengthscale'].constrain_fixed([100, 40]) # median huristic
        #model['mul.task_kern.lengthscale'].constrain_fixed(3.43366673485045)  # Task:50, init:10
        #model['mul.task_kern.lengthscale'].constrain_fixed(4.28168980658717)  # Task:5, init:300

        #model['mul.rbf.lengthscale'].constrain_fixed([12.96530508451918, 10.0])
        #model['mul.task_kern.lengthscale'].constrain_fixed(10.000000003090314 )


        model['mul.rbf.lengthscale'].constrain_fixed([101.26752103288666, 160.728118826718, 207.1466738764376]) #Task:37, init:50
        model['mul.task_kern.lengthscale'].constrain_fixed(3.1354437286607952) #Task:37, init:50

        # model['mul.rbf.lengthscale'].constrain_fixed([35.22387348903825, 51.63612797555372, 14.249282742155254]) #Task:37, init:50
        # model['mul.task_kern.lengthscale'].constrain_fixed(8.000000014076866) #Task:37, init:50

        # model['mul.rbf.lengthscale'].constrain_fixed([7.925361951467911, 19.22431823375, 8.375320383937053]) #Task:37, init:50
        # model['mul.task_kern.lengthscale'].constrain_fixed(30.000000739310465) #Task:37, init:50

        # model['mul.rbf.lengthscale'].constrain_fixed([96.25747680135564, 110.27797221402115, 174.15331779289272]) #Task:37, init:50
        # model['mul.task_kern.lengthscale'].constrain_fixed(5.000000004970913) #Task:37, init:50  これよさそう　！！！６以上だとだめだった

        ## Task15 setting
        # model['mul.rbf.lengthscale'].constrain_fixed([111.22089881865439, 167.2215142895854]) #Task:15, init:50
        # model['mul.task_kern.lengthscale'].constrain_fixed(6.000000000158428) #Task:15, init:50

        ## Task15 2dim
        # model['mul.rbf.lengthscale'].constrain_fixed([83.79857756794266, 107.87751796589069]) #Task:15, init:50
        # model['mul.task_kern.lengthscale'].constrain_fixed(6.00000000041493) #Task:15, init:50

        ## Task15 2dim
        # model['mul.rbf.lengthscale'].constrain_fixed([109.79685149715283, 128.62878166542356]) #Task:15, init:100
        # model['mul.task_kern.lengthscale'].constrain_fixed(6.00000000041493) #Task:15, init:100

        ##Task10 2dim
        # model['mul.rbf.lengthscale'].constrain_fixed([121.57975243366553, 147.5767577680832]) #Task:10, init:50
        # model['mul.task_kern.lengthscale'].constrain_fixed(6.0000000033613095) #Task:10, init:50

        # Task10
        #model['mul.rbf.lengthscale'].constrain_fixed([96.25747680135564, 110.27797221402115, 174.15331779289272])
        #model['mul.rbf.lengthscale'].constrain_fixed([53.12195264759786, 96.12491312412173, 50.64857039809234]) #Task:10, init:50
        #model['mul.task_kern.lengthscale'].constrain_fixed(6.000000019391109) #Task:10, init:50
        #model['mul.task_kern.lengthscale'].constrain_fixed(5.000000004970913)

        # Task4 
        # model['mul.rbf.lengthscale'].constrain_fixed([24.675240377510228, 53.95790574287922, 43.49905580600205])
        # model['mul.task_kern.lengthscale'].constrain_fixed(6.0000000386154015)

        # Task 37, Fulldim, input50
        # model['mul.rbf.lengthscale'].constrain_fixed([22.166141951795343, 22.30123741139694,111.6344591238481]) #Task:15, init:50
        # model['mul.task_kern.lengthscale'].constrain_fixed(6.000000002379202) #Task:15, init:50
        #### setting
        # model['mul.task_kern.lengthscale'].constrain_bounded(lower=6, upper=10)
        # model['mul.rbf.lengthscale'].constrain_bounded(lower=0, upper=200)
        #### 

        # Task 37, Fulldim, input50
        # model['mul.rbf.lengthscale'].constrain_fixed([1.8608770849910516, 0.48707371878915756,0.7096919017495303]) #Task:15, init:50
        # model['mul.task_kern.lengthscale'].constrain_fixed(4.827010576532458) #Task:15, init:50

        # model['mul.task_kern.lengthscale'].constrain_bounded(lower=6, upper=10)
        # model['mul.rbf.lengthscale'].constrain_bounded(lower=0, upper=200)
        # model.optimize_restarts(num_restarts=5, parallel=True)
        # print(model)
        # print(model.mul.rbf.lengthscale[0])
        # print(model.mul.rbf.lengthscale[1])
        # print(model.mul.rbf.lengthscale[2])
        # sys.exit()

        variance = model.mul.rbf.variance[0]
        length_scale = np.array([model.mul.rbf.lengthscale[0],model.mul.rbf.lengthscale[1],model.mul.rbf.lengthscale[2]])
        task_variance = model.mul.task_kern.variance[0]
        task_lengthscale = model.mul.task_kern.lengthscale[0]

    # タスク間の相関
    star_index = np.arange(task_num_total) * grid_num
    task_var = model.kern.K(XX[star_index])
    np.set_printoptions(precision=1)
    print(task_var)
    sys.exit()
    
    #gpy_pred_matrix = predict_domain_diag(model, XX, XX_train, y_train, task_num_total, grid_num)
    gpy_pred_mean, gpy_pred_var = model.predict_noiseless(XX)
    
    gpy_pred_mean_task_list = [
        gpy_pred_mean[grid_num * i: grid_num * (i + 1)] for i in range(task_num_total)]
    gpy_pred_var_task_list = [
        gpy_pred_var[grid_num * i : grid_num * (i + 1)] for i in range(task_num_total)]
    f_plot.residualplot(y_list, gpy_pred_mean_task_list, gpy_pred_var_task_list, "residualplot.png")
    sys.exit()

    # 回帰の様子をplot
    #multi_func_plot(X, y_list, X_train_list, y_train_list, gpy_pred_mean_task_list, gpy_pred_var_task_list, name = "test.png")
    #sys.exit()
    
    #初期点のsimple regret
    regret_list =[]
    for i in range(task_num_total):
        if len(y_train_list[i]) != 0:
            regret_list.append(y_list[i].max(axis=0) - y_train_list[i].max(axis=0))
    regret_multi = sum(regret_list)
    
    inference_regret_observe_list = []
    inference_regret_nonobserve_list = []
    for i in range(task_num_total):
        if len(y_train_list[i]) != 0:
            inference_regret_observe_list.append(y_list[i].max(axis=0) - y_list[i][gpy_pred_mean_task_list[i].argmax(axis=0)])
        else:
            inference_regret_nonobserve_list.append(y_list[i].max(axis=0) - y_list[i][gpy_pred_mean_task_list[i].argmax(axis=0)])
    inference_regret_observe_multi = sum(inference_regret_observe_list)
    inference_regret_nonobserve_multi = sum(inference_regret_nonobserve_list)
    inference_regret_list = [y_list[i].max(
        axis=0) - y_list[i][gpy_pred_mean_task_list[i].argmax(axis=0)] for i in range(task_num_total)]
    inference_regret_multi = sum(inference_regret_list)
    # if cost_key == True:
    #     total_cost = np.array([0])

    # プロットや結果保存のためのディレクトリをつくる
    result_dir_path = "./"+experiment_name+"/"+acq_name+"/seed_"+str(seed)+"/"
    _ = subprocess.check_call(["mkdir", "-p", result_dir_path])
    print(inference_regret_multi)
    # print(regret_multi) 
    #sys.exit()
    # ベイズ最適化
    for iter in range(max_iter):
        print("iter: ", iter)
        save_data_path = result_dir_path+"iter_"+str(iter)+"/"
        _ = subprocess.check_call(["mkdir", "-p", save_data_path])

        if acq_name in {'multi_mes', 'mes','sum_mes','imp_sum_mes', 'imp_sum_mes_perfect'}:
            # RFMを使って関数のサンプリング

            func_num = 5
            # prod_variance = variance * task_variance
            # F_multi = ReMultiFunctionGenerator(seed, length_scale, task_lengthscale, prod_variance, noise_var, XX)
            # f = F_multi.gen(XX_train, y_train, func_num=func_num)
            # f_list = [f[:, grid_num * i : grid_num * (i + 1)] for i in range(task_num_total)]
            
            # Random Feature 使う場合
            
            # prod_variance = variance * task_variance
            # F_multi = ReImportanceSampling(seed, length_scale, task_lengthscale, prod_variance, noise_var, task_var, XX, X_train_list, star_index, model,y_list,X,y_train_list, gpy_pred_mean_task_list, gpy_pred_var_task_list)
            # f_list, Theta_mean_list, Theta_var_list,out_list = F_multi.gen(XX_train, y_train,func_num=func_num)

            # y_star = []
            # for i in range(task_num_total): 
            #     if acq_name == 'imp_sum_mes':
            #         if task_observable[i] == True: #ここは今、Task0が観測可能なら動くが、そうでない場合は動かない
            #             y_star.append(np.c_[f_list[i].max(axis=1)])
            #         else:
            #             y_star.append(np.c_[f_list[0].max(axis=1)])
            #     else:
            #         y_star.append(np.c_[f_list[i].max(axis=1)])
            # print(f_list[0].shape)
            # sys.exit()
            # multi_imp_plot(X, y_list, X_train_list, y_train_list, gpy_pred_mean_task_list, gpy_pred_var_task_list, f_list, 'imp.png')
            # sys.exit()
            # RFMから得られた連続な関数の表示
            #f_plot.multi_rfm_plot(X,y_list,XX_train,y_train_list,gpy_ed_mean_task_list,gpy_pred_var_task_list,f_list,"alpha.pdf")

            # Random Feature 使わない場合
            # y_star = []
            # sample_path = []
            # for i in range(task_num_total):
            #     samples = np.random.normal(loc = gpy_pred_mean_task_list[i], scale = np.sqrt(gpy_pred_var_task_list[i]),size = (grid_num,func_num))
            #     print(samples.shape)
            #     sample_path.append(samples)
            #     star = np.c_[samples.max(axis=0)]
            #     print(star.shape)
            #     sys.exit()
            #    y_star.append(star)
            
            # タスクも同時
            # me, gpy_pred_cov = model.predict_noiseless(XX, full_cov = True)
            # samples = np.random.multivariate_normal(me.ravel(), gpy_pred_cov,size = func_num).T
            # y_star = []
            # f_list = []
            # for i in range(task_num_total):
            #     f_list.append(samples[i*len(X):(i+1)*len(X)].T)
            #     y_star.append(f_list[i].max(axis=1))

            # タスクごとに独立にサンプリング
            # y_star = []
            # f_list = []
            # for i in range(task_num_total):
            #     me, gpy_pred_cov = model.predict(XX[i*len(X):(i+1)*len(X)], full_cov = True)
            #     z = np.random.randn(len(X),func_num)
            #     A = np.linalg.cholesky(gpy_pred_cov)
            #     samples = me + np.dot(A,z)
            #     f_list.append(samples.T)
            #     star = np.c_[samples.max(axis=0)]
            #     y_star.append(star)
            
            ### RBF Sampler
            dim = 1000
            # rbf_feature = RBFSampler(gamma=np.c_[np.array([1/(2*length_scale**2),1/(2*task_lengthscale**2)])], n_components=dim, random_state=0)
            rbf_feature = RBFSampler(gamma=np.c_[np.array([1/(2*length_scale[0]**2),1/(2*length_scale[1]**2),1/(2*length_scale[2]**2),1/(2*task_lengthscale**2)])], n_components=dim, random_state=0)
            features = rbf_feature.fit_transform(XX_train,y_train)
            Theta = ThetaGenerator(func_seed,dim, noise_var)
            Theta.calc(features,y_train)
            theta=Theta.getTheta(func_num)
            features = rbf_feature.fit_transform(XX)
            f = np.dot(theta, features.T)
            f_list = [f[:, grid_num * i : grid_num * (i + 1)] for i in range(task_num_total)]
            y_star = []
            for i in range(task_num_total):
                y_star.append(np.c_[f_list[i].max(axis=1)])

            # RFMから得られた連続な関数の表示
            # fig, axes = plt.subplots(2, 3, 
            #            gridspec_kw={
            #                'width_ratios': [1, 1,1],
            #                'height_ratios': [1,1]},sharex = "all",sharey='row', tight_layout=True,figsize=(15, 8))

            # task_num = len(y_list)
            # for i in range(task_num-3):
            #     axes[0][i].plot(X,y_list[i],"r",label = "true_fn")
            #     axes[0][i].plot(X, gpy_pred_mean_task_list[i], "b", label="pred_mean")
            #     axes[0][i].plot(X_train_list[i], y_train_list[i], "ro", label="observed")
            #     axes[0][i].fill_between(X.ravel(), (gpy_pred_mean_task_list[i] + 1.96 * np.sqrt(gpy_pred_var_task_list[i])).ravel(), (gpy_pred_mean_task_list[i] - 1.96 * np.sqrt(gpy_pred_var_task_list[i])).ravel(), alpha=0.3, color="blue", label="credible interval")
            #     for j in range(func_num):
            #         axes[0][i].axhline(y_star[i][j], ls = "-.", color = "magenta")
            #         axes[0][i].plot(X,f_list[i][j])
            # for i in range(task_num-3):
            #     axes[1][i].plot(X,y_list[i+3],"r",label = "true_fn")
            #     axes[1][i].plot(X, gpy_pred_mean_task_list[i+3], "b", label="pred_mean")
            #     axes[1][i].plot(X_train_list[i+3], y_train_list[i+3], "ro", label="observed")
            #     axes[1][i].fill_between(X.ravel(), (gpy_pred_mean_task_list[i+3] + 1.96 * np.sqrt(gpy_pred_var_task_list[i+3])).ravel(), (gpy_pred_mean_task_list[i+3] - 1.96 * np.sqrt(gpy_pred_var_task_list[i+3])).ravel(), alpha=0.3, color="blue", label="credible interval")
            #     for j in range(func_num):
            #         axes[1][i].axhline(y_star[i+3][j], ls = "-.", color = "magenta")
            #         axes[1][i].plot(X,f_list[i+3][j])
            # axes[0, 0].set_title("Task 1")
            # axes[0, 1].set_title("Task 2")
            # axes[0, 2].set_title("Task 3")
            # axes[1, 0].set_title("Task 4")
            # axes[1, 1].set_title("Task 5")
            # axes[1, 2].set_title("Task 6")
            # plt.legend(loc="best")
            # plt.savefig("alpha.png")
            # plt.close()
            # # f_plot.multi_rfm_plot(X,y_list,XX_train,y_train_list,gpy_pred_mean_task_list,gpy_pred_var_task_list,f_list,"alpha.png")
            # sys.exit()

            # multi_imp_plot(X, y_list, X_train_list, y_train_list, gpy_pred_mean_task_list, gpy_pred_var_task_list, sample_path, 'imp.png')
            # sys.exit()

            # Random Feature 使わない場合(試してないから後で試しとけよz)
            # y_star = []
            # sampler = np.random.randn(func_num,len(X))
            # sample_path = []
            # for i in range(task_num_total):
            #     samples_of_onetask = gpy_pred_mean_task_list[i].ravel() + np.sqrt(gpy_pred_var_task_list[i].ravel())*sampler
            #     #samples = np.random.normal(loc = gpy_pred_mean_task_list[i],scale =np.sqrt(gpy_pred_var_task_list[i]),size = (grid_num,func_num))
            #     sample_path.append(samples_of_onetask)
            #     star = np.c_[samples_of_onetask.max(axis=1)]
            #     y_star.append(star)

            # print(y_star[0])
            # print(max(y_list[0]))
            # print(y_star[1])
            # print(max(y_list[1])) 
            # print(y_star[2])
            # print(max(y_list[2]))
            #sys.exit()

            for i in range(task_num_total):
                if task_observable[i] == True:
                    if initial_num_list[i] > 0:
                        y_star[i][y_star[i] < y_train_list[i].max(
                        ) + np.sqrt(noise_var) * 5] = y_train_list[i].max() + np.sqrt(noise_var) * 5

            if acq_name == 'sum_mes': #近似あり(近傍タスクのみ計算)
                alpha_list = acq.sum_mt_mes(X,y_star, gpy_pred_mean_task_list, gpy_pred_var_task_list, func_num, gpy_pred_matrix,task_feature,near_task_num)
            if acq_name == 'mes':
                alpha_list = acq.mes(
                    y_star, gpy_pred_mean_task_list, gpy_pred_var_task_list)

            # if acq_name == 'imp_sum_mes':
            #     alpha_list = acq.imp_sum_mt_mes(
            #         X, y_star, gpy_pred_mean_task_list, gpy_pred_var_task_list, func_num, gpy_pred_matrix, Theta_mean_list, Theta_var_list, out_list, task_observable,task_feature,near_task_num)
            
            if acq_name == 'imp_sum_mes_perfect':
                alpha_list = acq.imp_sum_mt_mes_perfect(X,
                    y_star, gpy_pred_mean_task_list, gpy_pred_var_task_list, func_num, gpy_pred_matrix)

            if acq_name == 'multi_mes':
                alpha_list = acq.multi_mes(
                    X, y_star, gpy_pred_mean_task_list, gpy_pred_var_task_list, task_var, func_num, gpy_pred_matrix)
            

        elif acq_name == "multi_ei":
            alpha_list = acq.multi_ei(y_train_list, gpy_pred_mean_task_list, gpy_pred_var_task_list)

        elif acq_name == "kg":
            max_pred_mu = [gpy_pred_mean_task_list[i].max() for i in range(task_num_total)]
            max_pred_mu = np.array(max_pred_mu)
            alpha_list = acq.kg(kern,X,XX, X_train_list, y_train_list, gpy_pred_mean_task_list,gpy_pred_var_task_list, max_pred_mu,task_feature,grid_num,near_task_num)
        # 獲得関数値の確認
        # print(alpha_list)
        # sys.exit()
        #plot_alpha_multi(X, y_list, X_train_list, y_train_list, gpy_pred_mean_task_list, gpy_pred_var_task_list, alpha_list, save_data_path + "alpha.png")

        #データ保存
        # save_data_lists = [X, alpha_list]
        # data_name_lists = ["X", "alpha_list"]
        # for i in range(len(data_name_lists)):
        #     with open(save_data_path+data_name_lists[i]+".pickle", 'wb') as f:
        #         pickle.dump(save_data_lists[i], f)

        if acq_name != "random":
            for index in range(task_num_total):
                alpha_list[index][train_index[index]] = 0
            
            print(alpha_list[0].shape)

            if cost_key == True:
                for task in range(task_num_total):
                    alpha_list[task] = alpha_list[task]/cost_list[task]
            
            alpha_max = np.empty(0)
            for index in range(task_num_total):
                alpha_max = np.append(alpha_max, np.max(alpha_list[index]))
            
            for next_t in np.argsort(alpha_max)[::-1]:
                if task_observable[next_t] == True:
                    next_task = next_t
                    break
            # print(alpha_list[0].shape)
            # sys.exit()
            print("next: ", next_task)
            next_index = np.argmax(alpha_list[next_task])
            print("next_index: ", next_index)
            y_next = y_list[next_task][next_index]
            
            #データの保存
            if data_save == True:
                save_data_lists = [X, y_list, X_train_list,y_train_list, gpy_pred_mean_task_list, gpy_pred_var_task_list, alpha_list, next_task, next_index,regret_multi,inference_regret_multi,inference_regret_observe_multi,inference_regret_nonobserve_multi,initial_num_list,y_star,gpy_pred_matrix,task_feature]#,total_cost]
                data_name_lists = ["X", "y_list", "X_train_list", "y_train_list", "pred_mean_task_list", "pred_var_task_list", "alpha_list", "next_task", "next_index", "regret_multi","inference_regret_multi","inference_regret_observe_multi","inference_regret_nonobserve_multi","initial_num_list","y_star","gpy_pred_matrix","task_feature"]#,"total_cost"]

                for i in range(len(data_name_lists)):
                    with open(save_data_path+data_name_lists[i]+".pickle", 'wb') as f:
                        pickle.dump(save_data_lists[i], f)
            
        else:
            # True_index = [0,98,99]
            next_task = random.choice(range(task_num_total))
            print("next: ", next_task)
            next_index = np.random.randint(0,grid_num,1)[0]
            print("next_index: ", next_index)
            y_next = y_list[next_task][next_index]

            #データの保存
            if data_save == True:
                save_data_lists = [X, y_list, X_train_list,y_train_list, gpy_pred_mean_task_list, gpy_pred_var_task_list, next_task, next_index,regret_multi,inference_regret_multi,inference_regret_observe_multi,inference_regret_nonobserve_multi,initial_num_list,task_feature]#,total_cost]
                data_name_lists = ["X", "y_list", "X_train_list", "y_train_list", "pred_mean_task_list", "pred_var_task_list", "next_task", "next_index", "regret_multi","inference_regret_multi", "inference_regret_observe_multi","inference_regret_nonobserve_multi","initial_num_list","task_feature"]#,"total_cost"]
                for i in range(len(data_name_lists)):
                    with open(save_data_path+data_name_lists[i]+".pickle", 'wb') as f:
                        pickle.dump(save_data_lists[i], f)
        print(X[next_index])

        #plot_alpha_multi(X, y_list, X_train_list, y_train_list, gpy_pred_mean_task_list, gpy_pred_var_task_list, alpha_list, result_dir_path + str(iter) + "_fig.png")
        #plot_alpha_multi_ystar(X, y_list, X_train_list, y_train_list, gpy_pred_mean_task_list, gpy_pred_var_task_list, alpha_list,y_star,sample_path, result_dir_path + str(iter) + "_fig.png")
        #最適点を観測点として追加
        X_train_list[next_task] = np.vstack((X_train_list[next_task], X[next_index]))
        y_train_list[next_task] = np.vstack((y_train_list[next_task], y_next))
        train_index[next_task].append(next_index)

        # 後の行列計算のためにy_trainを結合してndarray化しとく
        y_train = np.empty((0, 1))
        for input in y_train_list:
            y_train = np.vstack((y_train, input))
        XX_train = np.empty((0, X.shape[1]+task_feature.shape[1]))
        
        for i in range(len(X_train_list)):
            XX_train = np.vstack((XX_train, np.hstack((X_train_list[i],np.tile(task_feature[i],(X_train_list[i].shape[0],1))))))
        initial_num_list[next_task]+=1

        # print(XX_train)
        # sys.exit()
        model.set_XY(XX_train, y_train)

        # gpy_pred_mean, gpy_pred_matrix = model.predict_noiseless(XX, full_cov=True)
        # gpy_pred_var = np.c_[np.diag(gpy_pred_matrix)]
        gpy_pred_matrix = predict_domain_diag(model, XX, XX_train, y_train, task_num_total, grid_num)
        gpy_pred_mean, gpy_pred_var = model.predict_noiseless(XX)

        gpy_pred_mean_task_list = [
            gpy_pred_mean[grid_num * i: grid_num * (i + 1)] for i in range(task_num_total)]
        gpy_pred_var_task_list = [
            gpy_pred_var[grid_num * i: grid_num * (i + 1)] for i in range(task_num_total)]

        regret_list = []
        for i in range(task_num_total):
            if len(y_train_list[i]) != 0:
                regret_list.append(y_list[i].max(axis=0) - y_train_list[i].max(axis=0))
        inference_regret_list = [y_list[i].max(axis=0) - y_list[i][gpy_pred_mean_task_list[i].argmax(axis=0)] for i in range(task_num_total)]
        # print(regret_list)
        # sys.exit()
        inference_regret_observe_list = []
        inference_regret_nonobserve_list = []
        for i in range(task_num_total):
            if len(y_train_list[i]) != 0:
                inference_regret_observe_list.append(y_list[i].max(axis=0) - y_list[i][gpy_pred_mean_task_list[i].argmax(axis=0)])
            else:
                inference_regret_nonobserve_list.append(y_list[i].max(axis=0) - y_list[i][gpy_pred_mean_task_list[i].argmax(axis=0)])
        inference_regret_observe_multi = np.append(inference_regret_observe_multi,sum(inference_regret_observe_list))
        inference_regret_nonobserve_multi = np.append(inference_regret_nonobserve_multi,sum(inference_regret_nonobserve_list))
        regret_multi = np.append(regret_multi, sum(regret_list))
        inference_regret_multi = np.append(inference_regret_multi, sum(inference_regret_list))
        #cost_sum = total_cost[-1] + cost_list[next_task][-1]
        #total_cost = np.append(total_cost, cost_sum)
        print("inf_regret: ", inference_regret_multi)
        # print("sin_regret: ", regret_multi)
        print("iter ",iter," :",initial_num_list)
        #print("total_cost: ",total_cost)
def main():
    os.environ["OMP_NUM_THREADS"] = "1"
    argv = sys.argv
    task_num_total = int(argv[1])
    #initial_num_list = []

    initial_num_list = [30]*task_num_total

    # initial_num_list[2] = 1
    # initial_num_list[7] = 1
    # for i in range(imitial_num):
    #     initial_num_list[np.random.randint(task_num_total)]+

    #for i in range(task_num_total):
        #initial_num_list.append(int(argv[2 + i]))
    #acq_name = str(argv[2 + task_num_total])
    acq_name = str(argv[2])
    max_iter = 200  # 6

    func_seed = 1  # test_functionのシード
    feature_seed = 0
    #task_observable = [False,False,True,False,False]
    #task_observable = [False,False,True,False,False,True,False]
    task_observable = [True] * task_num_total
    # task_observable[0] = True
    # task_observable[99] = True
    # task_observable[98] = True

    if real_data == False:
        if feature_random == "Normal":
            np.random.seed(feature_seed)
            random.seed(feature_seed)
            dist_1 = 3
            dist_2 = 6
            dist_3 = 9
            dist_1_feature = np.random.multivariate_normal(loc=[0.0,0.0], scale= [[1, 0], [0, 1]], size=dist_1)
            dist_2_feature = np.random.multivariate_normal(loc=[10.0,10.0], scale= [[1, 0], [0, 1]], size=dist_2)
            dist_3_feature = np.random.multivariate_normal(loc=[20.0,20.0], scale= [[1, 0], [0, 1]], size=dist_3)
            task_feature = np.c_[np.hstack([dist_1_feature, dist_2_feature,dist_3_feature])]
            # task_feature = np.c_[np.hstack([dist_1_feature, dist_3_feature])]

        if feature_random == "random":
            np.random.seed(feature_seed)
            random.seed(feature_seed)
            dist_1 = 5
            # dist_2 = 10
            # dist_3 = 10
            dist_1_feature = np.random.uniform(low=0.0, high=1.0, size=dist_1)
            # dist_2_feature = np.random.uniform(low=2.0, high=3.0, size=dist_2)
            # dist_3_feature = np.random.uniform(low=4.0, high=5.0, size=dist_3)
            # task_feature = np.c_[np.hstack([dist_1_feature, dist_2_feature,dist_3_feature])]
            task_feature = np.c_[dist_1_feature]
        # else:
        #     task_feature = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 10.0, 11.0, 12.0, 20.0, 22.0, 30.0, 31.0, 31.3]
        #     task_feature = np.c_[np.array(task_feature)]

        if feature_random == "mix":
            # Set-up.
            # n = 10000
            # # Parameters of the mixture components
            # norm_params = np.array([[1, 1.5],
            #                         [10, 0.5]
            #                         ])
            # n_components = norm_params.shape[0]
            # # Weight of each component, in this case all of them are 1/3
            # #weights = np.ones(n_components, dtype=np.float64) / float(n_components)
            # weights = [0.5,0.5]
            # # A stream of indices from which to choose the component
            # # mixture_idx = np.random.choice(len(weights), size=n, replace=True, p=weights)
            # mixture_idx = np.random.choice(n_components, size=n, replace=True, p=weights)
            # # y is the mixture sample
            # y = np.fromiter((ss.norm.rvs(*(norm_params[i])) for i in mixture_idx),
            #                 dtype=np.float64)
            #f = [-0.725, -0.02, 1.0, 2.02, 1.725, 9.425, 9.66,10,10.34, 10.575] #task10ko
            f = [-0.02, 1.0, 2.02, 9.66,10, 10.34] #task6ko
            # print(f)
            # # Theoretical PDF plotting -- generate the x and y plotting positions
            # xs = np.linspace(y.min(), y.max(), 300)
            # ys = np.zeros_like(xs)

            # for (l, s), w in zip(norm_params, weights):
            #     ys += ss.norm.pdf(xs, loc=l, scale=s) * w

            # plt.plot(xs, ys)
            # #plt.hist(y,density=True, bins="fd")
            # plt.vlines(f,ymin=0, ymax=0.1,color='k',linestyle='dotted')
            # plt.ylim(0, 0.42)
            # plt.xlabel("s")
            # plt.ylabel("probability")
            # plt.savefig("mix_task.pdf")
            #sys.exit()
            task_feature = np.c_[f]


    else:
        task_feature = 0        
    if not acq_name in {'mes', 'multi_mes', 'ei', 'ucb', 'multi_ei','sum_mes','imp_sum_mes','random','imp_sum_mes_perfect','kg'} :
        print("獲得関数名を正しく入力してください")
        sys.exit()
    # 単体テスト用
    if mode != 'mul':
        seed = 7 # 初期点のシード
        experiment(seed, initial_num_list, max_iter,
                acq_name, func_seed,task_feature,task_observable)
    else:
        # 初期点を変えた10通りの実験を並列に行う
        parallel_num = 10
        _ = Parallel(n_jobs=parallel_num)([
            delayed(experiment)(i, initial_num_list, max_iter, acq_name, func_seed,task_feature,task_observable) for i in range(parallel_num)
        ])
    
if __name__ == "__main__":
    near_task_num = 6
    GP_type = 'multi' # multiq
    feature_random = "mix"
    experiment_name = 'task3_test'
    data_save = False
    real_data = True
    cost_key = False
    initial_random = False
    if real_data == True:
        experiment_name = 'real_' + experiment_name
    if initial_random ==True:
        experiment_name += '_random'
    if cost_key ==True:
        experiment_name += '_cost'
    mode = 'sin'
    main()
