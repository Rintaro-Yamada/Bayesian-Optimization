import matplotlib.pyplot as plt
import numpy as np
import sys

#rcParams
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"]=15
#plt.rcParams["xtick.labelsize"]= 15
#plt.rcParams["ytick.labelsize"] = 15
plt.rcParams["figure.titlesize"] = 20
#plt.rcParams["grid.linewidth"] = 1.0
plt.rcParams["legend.loc"] = "best"
#plt.rcParams["axes.grid"] = True


def plot_alpha(X, y1, y2, X1_train, y1_train, X2_train, y2_train, pred_mean_1, pred_var_1, pred_mean_2, pred_var_2, param, alpha_1, alpha_2, name):
    fig = plt.figure(figsize=(12,15))
    #Output 1
    ax1 = fig.add_subplot(411)
    ax1.set_title('Output 1:    $f(x)= -(6x - 2)^2 sin(12x - 4)$')
    plt.plot(X.ravel(),y1,"r--",label="true")
    plt.plot(X.ravel(), pred_mean_1, "b", label="pred mean")
    plt.fill_between(X.ravel(), (pred_mean_1 + 2 * np.sqrt(pred_var_1)).ravel(), (pred_mean_1 - 2 * np.sqrt(pred_var_1)).ravel(), alpha=0.3, color="blue",label="credible interval")
    plt.plot(X1_train.ravel(), y1_train, "ro", label="observation")
    plt.legend(loc="lower left")
    ax2 = fig.add_subplot(412)
    plt.plot(X.ravel(),alpha_1,"g-",label="alpha_1")
    plt.legend(loc="upper left")
    #Output 2
    ax3 = fig.add_subplot(413)
    ax3.set_title('Output 2:    $f(x)= -('+str(param)+'x - 2)^2 sin(12x - 4)$')
    plt.plot(X.ravel(),y2,"r--",label="true")
    plt.plot(X.ravel(), pred_mean_2, "b", label="pred mean")
    plt.fill_between(X.ravel(), (pred_mean_2 + 1.96 * np.sqrt(pred_var_2)).ravel(), (pred_mean_2 - 1.96 * np.sqrt(pred_var_2)).ravel(), alpha=0.3, color="blue",label="credible interval")
    plt.plot(X2_train.ravel(), y2_train, "ro", label="observation")
    plt.legend(loc="lower left")

    ax4 = fig.add_subplot(414)
    plt.plot(X.ravel(),alpha_2,"g-",label="alpha_2")
    plt.legend(loc="upper left")

    plt.savefig(name)
    plt.close()

def d1_plot(X, y_list):
    print(X.shape)
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(1, 5, 1)
    ax1.plot(X, y_list[0], "r")
    ax1 = fig.add_subplot(1, 5, 2)
    ax1.plot(X, y_list[1], "r")
    ax1 = fig.add_subplot(1, 5, 3)
    ax1.plot(X, y_list[2], "r")
    ax1 = fig.add_subplot(1, 5, 4)
    ax1.plot(X, y_list[3], "r")
    ax1 = fig.add_subplot(1, 5, 5)
    ax1.plot(X,y_list[4],"r")
    plt.savefig('tes.png')
    plt.close()

def d2_plot(X, y_list,xx,yy,init_num):
    fig = plt.figure(figsize=(8, 6))  #プロット領域の作成
    ax1 = fig.add_subplot(2, 2, 1,projection='3d')
    ax1.plot_surface(xx, yy, y_list[0].reshape(init_num,init_num), rstride=1, cstride=1, cmap='jet', alpha=0.8)
    
    ax2 = fig.add_subplot(2, 2, 2,projection='3d')
    ax2.plot_surface(xx, yy, y_list[1].reshape(init_num,init_num), rstride=1, cstride=1, cmap='jet', alpha=0.8)
    ax3 = fig.add_subplot(2, 2, 3,projection='3d')
    ax3.plot_surface(xx, yy, y_list[3].reshape(init_num,init_num), rstride=1, cstride=1, cmap='jet', alpha=0.8)
    ax4 = fig.add_subplot(2, 2, 4,projection='3d')
    surface = ax4.plot_surface(xx, yy, y_list[4].reshape(init_num, init_num), rstride=1, cstride=1, cmap='jet', alpha=0.8)
    # カラーバーを設定
    fig.colorbar(surface, ax=ax4, shrink=0.5)
    #ax1.scatter(xx2, yy2, y_list_2[0].reshape(init_num2,init_num2), s = 10, c = "blue")
    plt.savefig("tes.png")
    #ax1.plot_surface(xx, yy, y_list[0].reshape(10, 10), color='blue', linewidth=0.3)  # 曲面のプロット。rstrideとcstrideはステップサイズ，cmapは彩色，linewidthは曲面のメッシュの線の太さ，をそれぞれ表す。
    #ax2 = fig.gca(projection='3d') #プロット中の軸の取得。gca は"Get Current Axes" の略。
    
    #ax2.plot_surface(xx, yy, y_list[0].reshape(10,10), color='blue',linewidth=0.3) # 曲面のプロット。rstrideとcstrideはステップサイズ，cmapは彩色，linewidthは曲面のメッシュの線の太さ，をそれぞれ表す。


def rfm_plot(X,y1,y2,X1_train,y1_train,X2_train,y2_train,pred_mean_1,pred_var_1,pred_mean_2,pred_var_2,f_1,f_2,func_num,param,name):
    fig = plt.figure(figsize=(12,8))
    #Output 1
    ax1 = fig.add_subplot(211)
    ax1.set_title('Output 1:    $f(x)= -(6x - 2)^2 sin(12x - 4)$')
    plt.plot(X.ravel(),y1,"r--",label="true")
    plt.plot(X.ravel(), pred_mean_1, "b", label="pred mean")
    plt.fill_between(X.ravel(), (pred_mean_1 + 1.96 * np.sqrt(pred_var_1)).ravel(), (pred_mean_1 - 1.96 * np.sqrt(pred_var_1)).ravel(), alpha=0.3, color="blue",label="credible interval")
    for j in range(func_num):
        if j<func_num-1:
            ax1.plot(X.ravel(), f_1[j].ravel(), "g")
        else:
            ax1.plot(X.ravel(), f_1[j].ravel(), "g", label="sampled_function")
    plt.legend(loc="lower left")
    #Output 2
    ax2 = fig.add_subplot(212)
    ax2.set_title('Output 2:    $f(x)= -('+str(param)+'x - 2)^2 sin(12x - 4)$')
    plt.plot(X.ravel(),y2,"r--",label="true")
    plt.plot(X.ravel(), pred_mean_2, "b", label="pred mean")
    plt.fill_between(X.ravel(), (pred_mean_2 + 1.96 * np.sqrt(pred_var_2)).ravel(), (pred_mean_2 - 1.96 * np.sqrt(pred_var_2)).ravel(), alpha=0.3, color="blue",label="credible interval")
    for j in range(func_num):
        if j<func_num-1:
            ax2.plot(X.ravel(), f_2[j].ravel(), "g")
        else:
            ax2.plot(X.ravel(), f_2[j].ravel(), "g", label="sampled_function")
    plt.legend(loc="lower left")
    plt.savefig(name)
    plt.close()

def multi_func_plot(X,y_list,X_train_list,y_train_list,pred_mean_list,pred_var_list,name = 'unknown.pdf'):
    fig, axes = plt.subplots(2, 3, 
                       gridspec_kw={
                           'width_ratios': [1, 1,1],
                           'height_ratios': [1,1]},sharex = "all",sharey='row', tight_layout=True,figsize=(15, 8))

    task_num = len(y_list)
    # print(alpha_list[3])
    # sys.exit()
    #fig, ax = plt.subplots(2,task_num,  figsize=(18, 8)) 
    for i in range(task_num-3):
        axes[0][i].plot(X,y_list[i],"r",label = "true_fn")
        axes[0][i].plot(X, pred_mean_list[i], "b", label="pred_mean")
        axes[0][i].plot(X_train_list[i], y_train_list[i], "ro", label="observed")
        axes[0][i].fill_between(X.ravel(), (pred_mean_list[i] + 1.96 * np.sqrt(pred_var_list[i])).ravel(), (pred_mean_list[i] - 1.96 * np.sqrt(pred_var_list[i])).ravel(), alpha=0.3, color="blue", label="credible interval")
    for i in range(task_num-3):
        axes[1][i].plot(X,y_list[i+3],"r",label = "true_fn")
        axes[1][i].plot(X, pred_mean_list[i+3], "b", label="pred_mean")
        axes[1][i].plot(X_train_list[i+3], y_train_list[i+3], "ro", label="observed")
        axes[1][i].fill_between(X.ravel(), (pred_mean_list[i+3] + 1.96 * np.sqrt(pred_var_list[i+3])).ravel(), (pred_mean_list[i+3] - 1.96 * np.sqrt(pred_var_list[i+3])).ravel(), alpha=0.3, color="blue", label="credible interval")
    axes[0, 0].set_title("Task 1")
    axes[0, 1].set_title("Task 2")
    axes[0, 2].set_title("Task 3")
    axes[1, 0].set_title("Task 4")
    axes[1, 1].set_title("Task 5")
    axes[1, 2].set_title("Task 6")
    plt.legend(loc="best")
    #plt.show()
    plt.savefig(name)
    #plt.savefig(name,bbox_inches='tight', pad_inches=0.01)
    plt.close()

def multi_rfm_plot(X, y_list, X_train_list, y_train_list, pred_mean_list, pred_var_list, f_list, name):
    func_num = f_list[0].shape[0]
    task_num = len(f_list)
    fig, axes = plt.subplots(int(task_num/2)+task_num%2,2,  figsize=(12, 8)) 
    
    for i in range(int(task_num / 2) + task_num % 2):
        axes[i][0].plot(X, y_list[2*i], "r", label="true")
        axes[i][0].plot(X, pred_mean_list[2 * i], "b", label="mean")
        axes[i][0].fill_between(X.ravel(), (pred_mean_list[2 * i] + 1.96 * np.sqrt(pred_var_list[2 * i])).ravel(), (pred_mean_list[2 * i] - 1.96 * np.sqrt(pred_var_list[2 * i])).ravel(), alpha=0.3, color="blue", label="credible interval")
        axes[i][0].plot(X_train_list[2 * i], y_train_list[2 * i], "ro", label="observed")
        for j in range(func_num):
            axes[i][0].plot(X, f_list[2*i][j], "g", label="rfm")
        
        if (task_num % 2 == 0 or i != int(task_num / 2)):
            axes[i][1].plot(X, y_list[2*i + 1], "r", label="true")
            axes[i][1].plot(X, pred_mean_list[2 * i + 1], "b", label="mean")
            axes[i][1].fill_between(X.ravel(), (pred_mean_list[2 * i + 1] + 1.96 * np.sqrt(pred_var_list[2 * i + 1])).ravel(), (pred_mean_list[2 * i + 1] - 1.96 * np.sqrt(pred_var_list[2 * i + 1])).ravel(), alpha=0.3, color="blue", label="credible interval")
            axes[i][1].plot(X_train_list[2 * i + 1], y_train_list[2 * i + 1], "ro", label="observed")
            for j in range(func_num):
                axes[i][1].plot(X, f_list[2 * i + 1][j], "g", label="rfm")
    
    plt.savefig(name,bbox_inches='tight', pad_inches=0.01)
    plt.close()

def plot_alpha_multi(X, y_list, X_train_list, y_train_list, pred_mean_list, pred_var_list, alpha_list, name):
    task_num = len(y_list)
    fig, axes = plt.subplots(2, len(pred_mean_list), figsize=(18, 8))
    row =0
    for i in range(task_num):
        axes[row][i].plot(X, y_list[i], "r", label="true")
        axes[row][i].plot(X, pred_mean_list[i], "b", label="mean")
        axes[row][i].fill_between(X.ravel(), (pred_mean_list[i] + 1.96 * np.sqrt(pred_var_list[i])).ravel(), (pred_mean_list[i] - 1.96 * np.sqrt(pred_var_list[i])).ravel(), alpha=0.2, color="blue", label="credible interval")
        axes[row][i].plot(X_train_list[i], y_train_list[i], "o", label="observed")

        axes[row+1][i].plot(X,alpha_list[i],"g",label = "alpha")
    
    plt.savefig(name,bbox_inches='tight', pad_inches=0.01)
    plt.close()

def plot_alpha_fig(X, y_list, X_train_list, y_train_list, pred_mean_list, pred_var_list, alpha_list,alpha_list_2, name):
    task_num = len(y_list)
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['legend.fontsize'] = 13
    mes_max = [alpha_list[i].max() for i in range(2)]
    mes_max_index = [alpha_list[i].argmax() for i in range(2)]
    if mes_max[0] > mes_max[1]:
        mes_task = 0
    else:
        mes_task = 1
    proposed_max = [alpha_list_2[i].max() for i in range(2)]
    proposed_max_index = [alpha_list_2[i].argmax() for i in range(2)]
    if proposed_max[0] > proposed_max[1]:
        proposed_task = 0
    else:
        proposed_task = 1
    
    print('mes: ',mes_max_index)
    print('sum_mes: ',proposed_max_index)
    fig, axes = plt.subplots(3, 2, 
                       gridspec_kw={
                           'width_ratios': [1, 1],
                           'height_ratios': [2,1,1]},sharex = "all",sharey='row', tight_layout=True,figsize=(10, 5))
    # ax = plt.gca()
    # ax.xaxis.set_visible([])
    axes[1][0].xaxis.set_ticklabels([])
    axes[1][1].xaxis.set_ticklabels([])
    axes[0][0].yaxis.set_ticklabels([])
    axes[1][0].yaxis.set_ticklabels([])
    alp_colors = ['b','r','g']
    colors = ['b','r']
    for i in range(2):
        axes[1][i].set_ylim(-0.01,1.0)
        axes[2][i].set_ylim(-0.01,1.0)
        if i == 0:
            #axes[1][i].fill_between(X.ravel(),alpha_list[i].ravel(),alpha_list_2[i].ravel(),color =alp_colors[1],alpha=0.1,label = "Task2 gain")
            axes[1][i].plot(X,alpha_list_2[i],color= "c",label = "proposed")
            axes[1][i].plot(X,alpha_list[i],alp_colors[2]+":")
        if i ==1:
            #axes[1][i].fill_between(X.ravel(),alpha_list[i].ravel(),alpha_list_2[i].ravel(),color =alp_colors[0],alpha=0.1,label = "Task1 gain")
            axes[1][i].plot(X,alpha_list_2[i],color= "c",label = "proposed")
            axes[1][i].plot(X,alpha_list[i],alp_colors[2]+":")
        axes[2][i].plot(X,alpha_list[i],alp_colors[2],label = "mes")
        if i == mes_task:
            axes[2][i].axvline(x=X[mes_max_index[i]],color='k',linestyle='dotted')
        if i == proposed_task:
            axes[1][i].axvline(x=X[proposed_max_index[i]],color='k',linestyle='dotted')
            # axes[0][i].scatter(X[proposed_max_index[i]],y_list[i][proposed_max_index[i]], s=300,marker = '.', color = 'r',alpha=1,label = 'next')
        axes[0][i].plot(X, pred_mean_list[i], colors[i], label="mean")
        axes[0][i].fill_between(X.ravel(), (pred_mean_list[i] + 1.96 * np.sqrt(pred_var_list[i])).ravel(), (pred_mean_list[i] - 1.96 * np.sqrt(pred_var_list[i])).ravel(), alpha=0.2, color=colors[i], label="credible interval")
        #axes[0][i].plot(X_train_list[i], y_train_list[i], "ro", label="observed")
        axes[0][i].plot(X, y_list[i], 'k'+'--', label="true")
        axes[0][i].scatter(X_train_list[i], y_train_list[i], s=50,color='black', marker='x',label="observed")
        axes[0][i].scatter(X[y_list[i].argmax()],y_list[i].max(), s=300,marker = '*', color = colors[i],alpha=0.5)
        if i == mes_task:
            axes[0][i].axvline(x=X[mes_max_index[i]],color=alp_colors[2],linestyle='dashed')
        if i == proposed_task:
            axes[0][i].axvline(x=X[proposed_max_index[i]],color="c",linestyle='dashed')
    for i in range(3):
        for j in range(2):
            axes[i][j].tick_params(bottom=False,left=False,right=False,top=False)
            axes[i][j].tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
    axes[2][0].set_xlabel("x")
    axes[2][1].set_xlabel("x")
    # axes[2][0].axis("off")
    # axes[2][1].axis("off")
    # axes[1][0].set_xlabel("x")
    # axes[1][1].set_xlabel("x")
    axes[0][0].set_ylabel("f(x)")
    axes[1][0].set_ylabel(r"$\alpha(x)$")
    axes[2][0].set_ylabel(r"$\alpha(x)$")
    axes[0][0].set_title("Task1")
    axes[0][1].set_title("Task2")
    #plt.legend()
    axes[1][0].legend(loc='upper left')
    #axes[1][1].legend(loc='upper right')
    axes[2][0].legend(loc='upper left')
    plt.savefig(name,bbox_inches='tight', pad_inches=0.02)
    plt.close()

def plot_alpha_multi_ystar(X, y_list, X_train_list, y_train_list, pred_mean_list, pred_var_list, alpha_list,y_star,sample_path, name):
    task_num = len(y_list)
    fig, axes = plt.subplots(2, len(pred_mean_list), figsize=(18, 8))
    row =0
    for i in range(task_num):
        axes[row][i].plot(X, y_list[i], "r", label="true")
        axes[row][i].plot(X, pred_mean_list[i], "b", label="mean")
        axes[row][i].fill_between(X.ravel(), (pred_mean_list[i] + 1.96 * np.sqrt(pred_var_list[i])).ravel(), (pred_mean_list[i] - 1.96 * np.sqrt(pred_var_list[i])).ravel(), alpha=0.3, color="blue", label="credible interval")
        axes[row][i].plot(X_train_list[i], y_train_list[i], "ro", label="observed")
        for j in range(10):
            axes[row][i].plot(X, np.tile(y_star[i][j],200), "g--", label= "y_star")
            #axes[row][i].scatter(X, sample_path[i][:,j], c='r', s=1)
            
            #axes[row][i].plot(X, sample_path[i][:,j], "g--", label= "y_star")
        axes[row+1][i].plot(X,alpha_list[i],"g",label = "alpha")
    plt.legend()
    plt.savefig(name,bbox_inches='tight', pad_inches=0.01)
    plt.close()

def residualplot(y_list, pred_mean_list, pred_var_list, name):
    task_num_total = len(y_list)
    #fig, axes = plt.subplots(2, task_num_total, figsize=(10, 20))
    #for i in range(task_num_total):
    #    axes[0][i].scatter(y_list[i], pred_mean_list[i], c=pred_var_list[i], cmap='Blues')
    #plt.colorbar() 
    plt.scatter(y_list[0], pred_mean_list[0], c=pred_var_list[0], cmap='jet')
    plt.plot([min(y_list[0]),max(y_list[0])],[min(y_list[0]),max(y_list[0])],'b',alpha=0.3)
    plt.colorbar()
    plt.savefig(name, bbox_inches='tight', pad_inches=0.01)
    
    plt.close()



def multi_imp_plot(X, y_list, X_train_list, y_train_list, pred_mean_list, pred_var_list, f_list, name):
    task_num = len(y_list)
    fig, ax = plt.subplots(task_num, 1, figsize=(18, 8))
    #print(f_list[0][j,:].shape)
    #sys.exit()
    for i in range(task_num):
        ax[i].plot(X,y_list[i],"r",label = "true_fn")
        ax[i].plot(X, pred_mean_list[i], "b", label="pred_mean")
        ax[i].plot(X_train_list[i], y_train_list[i], "ro", label="observed")
        ax[i].fill_between(X.ravel(), (pred_mean_list[i] + 1.96 * np.sqrt(pred_var_list[i])).ravel(), (pred_mean_list[i] - 1.96 * np.sqrt(pred_var_list[i])).ravel(), alpha=0.3, color="blue", label="credible interval")
        for j in range(f_list[0].shape[0]):
            ax[i].plot(X, f_list[i][j,:],'g')
    #plt.show()
    plt.legend(loc="lower left")
    plt.savefig(name,bbox_inches='tight', pad_inches=0.01)
    plt.close()

def multi_imp_plot2(X, y_list, X_train_list, y_train_list, pred_mean, pred_var, f_list, name):
    task_num = len(y_list)
    fig, ax = plt.subplots(task_num, 1, figsize=(18, 8))

    pred_mean = pred_mean.ravel()
    for i in range(task_num):
        ax[i].plot(X,y_list[i],"r",label = "true_fn")
    #     ax[i].plot(X, pred_mean[i*200:(i+1)*200], "green", label="random pred_mean")
    #     ax[i].plot(X_train_list[i], y_train_list[i], "ro", label="observed")
    #     ax[i].fill_between(X.ravel(), (pred_mean[i*200:(i+1)*200] + 1.96 * np.sqrt(pred_var[i*200:(i+1)*200])).ravel(), (pred_mean[i*200:(i+1)*200] - 1.96 * np.sqrt(pred_var[i*200:(i+1)*200])).ravel(), alpha=0.3, color="green", label="random credible interval")
    #     for j in range(f_list[0].shape[1]):
    #         ax[i].plot(X, f_list[i][:,j],'g')
    # plt.legend(loc="lower left")
    #plt.show()
    plt.savefig(name,bbox_inches='tight', pad_inches=0.01)
    plt.close()

def multi_imp_plot3(X, y_list, X_train_list, y_train_list, pred_mean, pred_var,pred_mean_list, pred_var_list,f_list,dim, name):
    task_num = len(y_list)
    fig, ax = plt.subplots(task_num, 1, figsize=(18, 8))

    pred_mean = pred_mean.ravel()
    for i in range(task_num):
        ax[i].plot(X,y_list[i],"r",label = "true_fn")
        ax[i].plot(X, pred_mean[i*200:(i+1)*200], "g", label="random_pred_mean")
        ax[i].fill_between(X.ravel(), (pred_mean[i*200:(i+1)*200] + 1.96 * np.sqrt(pred_var[i*200:(i+1)*200])).ravel(), (pred_mean[i*200:(i+1)*200] - 1.96 * np.sqrt(pred_var[i*200:(i+1)*200])).ravel(), alpha=0.3, color="green", label="random credible interval")
    
        ax[i].plot(X, pred_mean_list[i], "b", label="gpy_pred_mean")
        ax[i].fill_between(X.ravel(), (pred_mean_list[i] + 2 * np.sqrt(pred_var_list[i])).ravel(), (pred_mean_list[i] - 2 * np.sqrt(pred_var_list[i])).ravel(), alpha=0.3, color="blue", label="gpy credible interval")
        
        #for j in range(f_list[0].shape[1]):
        #    ax[i].plot(X, f_list[i][:, j], 'g')
        ax[i].plot(X_train_list[i], y_train_list[i], "ro", label="observed")
    plt.legend(loc="lower left")   
    #plt.show()
    #plt.title("次元数: "+ str(dim))
    plt.savefig(name,bbox_inches='tight', pad_inches=0.01)
    plt.close()

# def func_plot(X,y_list,X_train_list,y_train_list,pred_mean_list,pred_var_list,name = 'unknown.pdf'):