import numpy as np
import sys
from ThetaGenerator import ThetaGenerator
from sklearn.utils import check_array, check_random_state, as_float_array
from scipy.stats import multivariate_normal
from f_plot import multi_imp_plot2, multi_imp_plot3
def RFM(x, dim, omega, b, variance):
    phi = np.sqrt(variance * 2 / dim) * (np.cos(np.dot(omega, x.T).T + b.T))
    return phi

def multi_rfm(input_xt, dim, omega, b, variance):
    phi = np.sqrt(variance * 2 / dim) * (np.cos(np.dot(omega, input_xt).T + b.T))
    return phi

class FunctionGenerator():
    def __init__(self, seed, lengthscale, variance, noise_var, X):
        self.seed = seed
        self.lengthscale = lengthscale
        self.variance = variance
        self.noise_var = noise_var
        self.X = X

    def gen(self,X_train,y_train,func_num):   #func_numは生成する関数の個数
        
        
        omega = (np.sqrt(1/(self.lengthscale**2)) * random_state.normal(size=(dim, X_train.shape[1])))
        b = np.c_[np.random.rand(dim) * 2 * np.pi]  #[0,2π]の一様乱数
        #RFMから特徴量ベクトルΦ(x)を取得
        large_phi = RFM(X_train, dim,omega,b, self.variance)  #D=100とした  10*1000
        Theta = ThetaGenerator(self.seed,dim, self.noise_var)
        Theta.calc(large_phi, y_train)
        phi = RFM(self.X, dim, omega, b, self.variance)
        
        theta=Theta.getTheta(func_num)
        #目的関数fの近似を取得する。
        f = np.dot(theta,phi.T)
        return f

    def gen_prior(self,func_num):
        dim = 1000
        random_state = check_random_state(self.seed)
        omega = (np.sqrt(1 / (self.lengthscale ** 2)) * random_state.normal(size=(dim, self.X.shape[1])))
        b = np.c_[np.random.rand(dim) * 2 * np.pi]  #[0,2π]の一様乱数
        #RFMから特徴量ベクトルΦ(x)を取得
        large_phi = RFM(self.X, dim,omega,b, self.variance)  #D=100とした  10*1000
        Theta = ThetaGenerator(self.seed, dim, self.noise_var)
        Theta.calc_init(large_phi)
        theta=Theta.getTheta(func_num)
        #目的関数fの近似を取得する。
        f = np.dot(theta,large_phi.T)
        return f

class importanceSampling():
    def __init__(self, seed, lengthscale, variance, noise_var, X):
        self.seed = seed
        self.lengthscale = lengthscale
        self.variance = variance
        self.noise_var = noise_var
        self.X = X

    def gen(self, X_train, y_train, func_num, y_train_teian, X_train_teian):  #func_numは生成する関数の個数
        dim=1000
        random_state = check_random_state(self.seed)
        omega = (np.sqrt(1/(self.lengthscale**2)) * random_state.normal(size=(dim, X_train.shape[1])))
        b = np.c_[np.random.rand(dim) * 2 * np.pi]  #[0,2π]の一様乱数
        #RFMから特徴量ベクトルΦ(x)を取得
        large_phi = RFM(X_train_teian, dim,omega,b, self.variance)  #D=100とした  10*1000
        Theta = ThetaGenerator(self.seed ,dim, self.noise_var)
        Theta.calc(large_phi, y_train_teian)
        phi = RFM(self.X, dim, omega, b, self.variance)
        theta=Theta.getTheta(func_num)
        
        
        Theta.calc(large_phi, y_train)

        print(phi.shape)
        sys.exit()

        
        #目的関数fの近似を取得する。
        f = np.dot(theta,phi.T)
        return f

class MultiFunctionGenerator():
    def __init__(self, seed, lengthscale,task_lengthscale, variance, noise_var, X,task):
        self.seed = seed
        self.lengthscale = lengthscale
        self.task_lengthscale = task_lengthscale
        self.variance = variance
        self.noise_var = noise_var
        self.X = X
        self.task=task

    def gen(self,XX_train, y_train,func_num):   #func_numは生成する関数の個数
        dim = 1000
        random_state = check_random_state(self.seed)
        omega_x = (np.sqrt(1 / (self.lengthscale ** 2)) * random_state.normal(size=(dim, self.X.shape[1]))) #(1000,3)
        omega_t = (np.sqrt(1 / (self.task_lengthscale ** 2)) * random_state.normal(size=(dim, self.task.shape[1]))) #(1000,1)
        omega = np.hstack((omega_x, omega_t)) 
        b = np.c_[np.random.rand(dim) * 2 * np.pi]  #[0,2π]の一様乱数 #(1000,1)
        #input_xt = np.zeros((self.X.shape[1] + self.task.shape[1], 0))
        #print(input_xt.shape)
        
        for i in range(len(XX_train)):
            if i == 0:
                input_xt = XX_train[i]
            else:
                input_xt = np.vstack((input_xt, XX_train[i]))
        input_xt = input_xt.T
        '''
        for i in range(self.task.shape[0]):
            tmp_xt = np.c_[np.full(X_train[i].shape[0], self.task[i])]
            print(X_train[i].shape)
            tmp_xt = np.hstack((X_train[i], tmp_xt)).T
            sys.exit()
            input_xt = np.hstack((input_xt,tmp_xt))
        '''
        large_phi = multi_rfm(input_xt, dim, omega, b, self.variance)  #D=100とした  10*1000
        Theta = ThetaGenerator(self.seed, dim, self.noise_var)
        Theta.calc(large_phi, y_train)
        
        '''
        input_Xt = np.empty((self.X.shape[1] + self.task.shape[1], 0))
        tmp_Tm = np.full((len(self.task),len(self.X)), self.task).reshape(1,-1)
        tmp_Xt = np.full((len(self.task),len(self.X)), self.X.T).reshape(1,-1)
        input_Xt = np.vstack((tmp_Xt, tmp_Tm))
        '''
        
        input_Xt = np.zeros((self.X.shape[1]+1,0))
        for i in range(self.task.shape[0]):
            tmp_Xt = np.c_[np.full(self.X.shape[0], self.task[i])]
            tmp_Xt = np.hstack((self.X, tmp_Xt)).T
            input_Xt = np.hstack((input_Xt,tmp_Xt))
        phi = multi_rfm(input_Xt, dim, omega, b, self.variance)
    
        theta=Theta.getTheta(func_num)
        #目的関数fの近似を取得する。
        f = np.dot(theta, phi.T)
        return f
    
    def gen_prior(self,func_num):
        dim = 1000
        random_state = check_random_state(self.seed)
        omega_x = (np.sqrt(1 / (self.lengthscale ** 2)) * random_state.normal(size=(dim, self.X.shape[1])))
        omega_t = (np.sqrt(1 / (self.task_lengthscale ** 2)) * random_state.normal(size=(dim, self.task.shape[1])))
        omega = np.hstack((omega_x, omega_t))
        b = np.c_[np.random.rand(dim) * 2 * np.pi]  #[0,2π]の一様乱数
        #RFMから特徴量ベクトルΦ(x)を取得
        input_xt = np.empty((self.X.shape[1] + 1, 0))
        tmp_tm = np.full((len(self.task), len(self.X)), self.task)
        print(tmp_tm.shape)
        
        tmp_xt = np.full((len(self.task), len(self.X)), self.X.T)
        input_xt = np.vstack((tmp_xt, tmp_tm))
        large_phi = multi_rfm(input_xt, dim, omega, b, self.variance)  #D=100とした  10*1000
        Theta = ThetaGenerator(self.seed,dim, self.noise_var)
        Theta.calc_init(large_phi)
        theta=Theta.getTheta(func_num)
        f = np.dot(theta, large_phi.T)
        return f


'''
class InitMultiFunctionGenerator():
    def __init__(self, seed, lengthscale,task_lengthscale, variance, noise_var, X,task):
        self.seed = seed
        self.lengthscale = lengthscale
        self.task_lengthscale = task_lengthscale
        self.variance = variance
        self.noise_var = noise_var
        self.X = X
        self.task=task

    def gen(self,X1_train,X2_train, y_train,func_num):   #func_numは生成する関数の個数
        dim = 1000
        random_state = check_random_state(self.seed)
        omega_x = (np.sqrt(1 / (self.lengthscale ** 2)) * random_state.normal(size=(dim, X1_train.shape[1])))
        omega_t = (np.sqrt(1 / (self.task_lengthscale ** 2)) * random_state.normal(size=(dim, X1_train.shape[1])))
        omega = np.hstack((omega_x,omega_t))
        b = np.c_[np.random.rand(dim) * 2 * np.pi]  #[0,2π]の一様乱数
        #RFMから特徴量ベクトルΦ(x)を取得
        tmp_xt = np.c_[np.full(X1_train.shape, self.task[0])]
        tmp_xt = np.hstack((X1_train, tmp_xt)).T
        tmp2_xt = np.c_[np.full(X2_train.shape, self.task[1])]
        tmp2_xt = np.hstack((X2_train, tmp2_xt)).T
        input_xt = np.hstack((tmp_xt, tmp2_xt))
        large_phi = multi_rfm(input_xt, dim, omega, b, self.variance)  #D=100とした  10*1000
        Theta = ThetaGenerator(dim, self.noise_var)
        Theta.calc(large_phi, y_train)

        tmp_Xt = np.c_[np.full(self.X.shape, self.task[0])]
        tmp_Xt = np.hstack((self.X, tmp_Xt)).T
        tmp2_Xt = np.c_[np.full(self.X.shape, self.task[1])]
        tmp2_Xt = np.hstack((self.X, tmp2_Xt)).T
        input_Xt = np.hstack((tmp_Xt, tmp2_Xt))
        phi = multi_rfm(input_Xt, dim, omega, b, self.variance)
        sys.exit()
        theta=Theta.getTheta(func_num)
        #目的関数fの近似を取得する。
        f = np.dot(theta, phi.T)
        return f 
'''

class ReMultiFunctionGenerator():
    def __init__(self, seed, lengthscale,task_lengthscale, variance, noise_var, XX):
        self.seed = seed
        self.lengthscale = lengthscale
        self.task_lengthscale = task_lengthscale
        self.variance = variance
        self.noise_var = noise_var
        self.XX = XX

    def gen(self, XX_train, y_train, func_num):  #func_numは生成する関数の個数
        dim = 1000
        random_state = check_random_state(self.seed)
        omega_x = (np.sqrt(1 / (self.lengthscale ** 2)) * random_state.normal(size=(dim, self.XX.shape[1]-1))) #(1000,3)
        omega_t = (np.sqrt(1 / (self.task_lengthscale ** 2)) * random_state.normal(size=(dim, 1))) #(1000,1)
        omega = np.hstack((omega_x, omega_t)) 
        b = np.c_[np.random.rand(dim) * 2 * np.pi]  #[0,2π]の一様乱数 #(1000,1)
        #input_xt = np.zeros((self.X.shape[1] + self.task.shape[1], 0))
        #print(input_xt.shape)
        
        
        for i in range(len(XX_train)):
            if i == 0:
                input_xt = XX_train[i]
            else:
                input_xt = np.vstack((input_xt, XX_train[i]))
        input_xt = input_xt.T
        '''
        for i in range(self.task.shape[0]):
            tmp_xt = np.c_[np.full(X_train[i].shape[0], self.task[i])]
            print(X_train[i].shape)
            tmp_xt = np.hstack((X_train[i], tmp_xt)).T
            sys.exit()
            input_xt = np.hstack((input_xt,tmp_xt))
        '''
        large_phi = multi_rfm(input_xt, dim, omega, b, self.variance)  #D=100とした  10*1000
        
        Theta = ThetaGenerator(self.seed, dim, self.noise_var)
        Theta.calc(large_phi, y_train)
        '''
        input_Xt = np.empty((self.X.shape[1] + self.task.shape[1], 0))
        tmp_Tm = np.full((len(self.task),len(self.X)), self.task).reshape(1,-1)
        tmp_Xt = np.full((len(self.task),len(self.X)), self.X.T).reshape(1,-1)
        input_Xt = np.vstack((tmp_Xt, tmp_Tm))
        '''
        
        input_Xt = self.XX
        phi = multi_rfm(input_Xt.T, dim, omega, b, self.variance)
    
        theta=Theta.getTheta(func_num)
        #目的関数fの近似を取得する。
        f = np.dot(theta, phi.T)
        return f
    
    def gen_prior(self,func_num):
        dim = 1000
        random_state = check_random_state(self.seed)
        omega_x = (np.sqrt(1 / (self.lengthscale ** 2)) * random_state.normal(size=(dim, self.XX.shape[1]-1)))
        omega_t = (np.sqrt(1 / (self.task_lengthscale ** 2)) * random_state.normal(size=(dim, 1)))
        omega = np.hstack((omega_x, omega_t))
        b = np.c_[np.random.rand(dim) * 2 * np.pi]  #[0,2π]の一様乱数
        #RFMから特徴量ベクトルΦ(x)を取得
        input_xt = self.XX
        large_phi = multi_rfm(input_xt.T, dim, omega, b, self.variance)  #D=100とした  10*1000
        Theta = ThetaGenerator(self.seed,dim, self.noise_var)
        Theta.calc_init(large_phi)
        theta=Theta.getTheta(func_num)
        f = np.dot(theta, large_phi.T)
        return f

class ReImportanceSampling():
    def __init__(self, seed, lengthscale, task_lengthscale, variance, noise_var, task_var, XX, X_train_list, star_index, model, y_list, X, y_train_list, gpy_pred_mean_task_list, gpy_pred_var_task_list):  #y_list以降はデバッグのためつかってるので、後で消す
        self.seed = seed
        self.lengthscale = lengthscale
        self.task_lengthscale = task_lengthscale
        self.variance = variance
        self.noise_var = noise_var
        self.task_var = task_var
        self.XX = XX
        self.X_train_list = X_train_list
        self.star_index = star_index
        self.model = model
        self.y_list = y_list
        self.X = X
        self.y_train_list = y_train_list
        self.gpy_pred_mean_task_list = gpy_pred_mean_task_list
        self.gpy_pred_var_task_list = gpy_pred_var_task_list

    def _calc_mean_var_by_phi(self, phi_x, phi_X, y_train, task_var,X_train_list):
        k_star = np.dot(phi_x, phi_X.T)
        K = np.dot(phi_x, phi_x.T)
        k_star_star = np.dot(phi_X[:200], phi_X[:200].T)

        k_star_star = np.kron(task_var,k_star_star)  + np.eye(600) * self.noise_var #k_star_star * task_var
        task_num = len(X_train_list)
        l = 0
        m = 0
        for i in range(task_num): #K * task_var
            for j in range(task_num):
                K[l: l+len(X_train_list[i]), m: m+ len(X_train_list[j])] = K[l: l+len(X_train_list[i]), m: m+ len(X_train_list[j])] * task_var[i,j]
                m += len(X_train_list[j])
            l += len(X_train_list[i])
            m = 0

        l = 0
        m = 0
        for i in range(task_num): #k_star * task_var
            for j in range(task_num):
                k_star[l: l+len(X_train_list[i]),m: m+len(self.X)] = k_star[l: l+len(X_train_list[i]),m: m+len(self.X)] * task_var[i,j]
                m += len(self.X)
            l += len(X_train_list[i])
            m = 0
        
        k_star_trans_K_inv = np.dot(k_star.T,np.linalg.inv(K))
        mu = np.dot(k_star_trans_K_inv, y_train)
        var = np.diag(k_star_star - np.dot(k_star_trans_K_inv, k_star))
        return mu, var

    def gen(self, XX_train, y_train, func_num):  #func_numは生成する関数の個数
        dim = 100
        random_state = check_random_state(self.seed)
        theta_eye = np.eye(dim)
        
        theta_var = np.kron(self.task_var, theta_eye)  + np.eye(dim * len(self.task_var)) * self.noise_var
        #print(np.kron(self.task_var, np.eye(10)))
        L = np.linalg.cholesky(theta_var)
        z = np.random.randn(theta_var.shape[1],dim) # (taskの個数, 次元)
        tmp = np.dot(L, z)
        theta = tmp
        #print(theta.shape) #(30,10)
        omega = (np.sqrt(1 / (self.lengthscale ** 2)) * random_state.normal(size=(dim, self.XX.shape[1] - 1)))  #(1000,3)
        #print(omega.shape) #(10,1)
        b = np.c_[np.random.rand(dim) * 2 * np.pi]  #[0,2π]の一様乱数 #(1000,1)
        #print(b.shape) #(10,1)
        #input_xt = np.zeros((self.X.shape[1] + self.task.shape[1], 0))
        #print(input_xt.shape)
        train_num = XX_train.shape[0]
        task_num = len(self.X_train_list)
        input_x = XX_train[:, 0: XX_train.shape[1] - 1]
        large_phi_element = RFM(input_x, dim, omega, b, self.variance)  # タスク全部合わせた phi. ここからタスクごとにphi_1,phi_2をつくる
        #print(large_phi_element.shape) #(6,10)
        large_phi = np.zeros((train_num, dim * self.task_var.shape[1]))
        #print(large_phi.shape) #(6,30)
        j = 0
        for i in range(self.task_var.shape[1]):
            large_phi[j: j + len(self.X_train_list[i]), i * dim : (i + 1) * dim] = large_phi_element[j: j + len(self.X_train_list[i]), :]
            j += len(self.X_train_list[i])
        #print(large_phi.shape) #(6,30)
        
        cov_y_theta = np.dot(large_phi, theta_var)
        #print(cov_y_theta.shape) #(6,30)
        
        var_index = np.empty((1, 0))
        for i in range(task_num):
            var_index = np.append(var_index, np.tile(self.star_index[i], len(self.X_train_list[i])))
        var_index = var_index.astype(np.int64)
        var = self.model.kern.K(self.XX[var_index])

        cov_y_y = np.dot(np.dot(large_phi, theta_var),large_phi.T) + np.eye(large_phi.shape[0]) * self.noise_var
        
        sigma_simga_inv = np.dot(cov_y_theta.T, np.linalg.inv(cov_y_y))
        Theta_mean = np.dot(sigma_simga_inv, y_train)
        Theta_var = theta_var - np.dot(sigma_simga_inv, cov_y_theta)
        # print(Theta_mean.shape)  #(30,1)
        # print(Theta_var.shape)  #(30,30)

        #　同時分布からサンプリング
        L = np.linalg.cholesky(Theta_var)
        z = np.random.randn(dim*task_num, func_num) # 1の部分は最終的にサンプリングされる, 関数の数に対応
        out = Theta_mean.T + np.dot(L, z).T

        out_list = []
        Theta_mean_list = []
        Theta_var_list = []
        for i in range(task_num):
            Theta_mean_list.append(Theta_mean[i * dim : (i + 1) * dim, :])
            Theta_var_list.append(Theta_var[i * dim : (i + 1) * dim, i * dim : (i + 1) * dim])
            out_list.append(out[:, i * dim : (i + 1) * dim])

        # 以下は別々にサンプリング
        # for i in range(task_num):
        #     L = np.linalg.cholesky(Theta_var_list[i])
        #     z = np.random.randn(dim, func_num)   # 1の部分は最終的にサンプリングされる, 関数の数に対応
        #     out = Theta_mean_list[i] + np.dot(L, z)
        #     out_list.append(out.T)

        #print(np.linalg.det(Theta_var_list[0]))
        #print()
        #print(Theta_var_list[0])
        #print(np.diag(Theta_var_list[1]))
        #print(np.hstack((np.hstack((Theta_mean_list[0],Theta_mean_list[1])),))))
        #print(Theta_var_list[1])
        #d = Theta_mean_list[0] - Theta_mean_list[1]
        #inv = np.linalg.inv(Theta_var_list[1])
        #p0 = multivariate_normal.pdf(out_list[0], mean=Theta_mean_list[0].ravel(), cov=Theta_var_list[0])
        #p1 = multivariate_normal.pdf(out_list[0], mean=Theta_mean_list[1].ravel(), cov=Theta_var_list[1]) 
        
        #print(np.exp(-1/2*np.dot(d.T, np.dot(inv, d))))
        #print(np.linalg.det(Theta_var_list[1]))
        #print('p0: ',p0)
        #print('p1: ',p1)
        #print('p1/p0: ', p1 / p0)
        #sys.exit()
        grid_num = int(self.XX.shape[0] / task_num)
        input_XX = self.XX[:, 0: self.XX.shape[1] - 1]  #200の部分は, xの候補点に合わせる
        phi = RFM(input_XX, dim, omega, b, self.variance)  # input_XT は　タスク特徴量ぬいたべくとるにしなよ！
        #print(np.dot(phi[:grid_num, :], out_list[0]))
        f_list = []
        for i in range(task_num):
            f = np.dot(out_list[i],phi[:grid_num].T)
            f = f
            f_list.append(f)
        #phi_mu, phi_var = self._calc_mean_var_by_phi(large_phi_element, phi, y_train,self.task_var,self.X_train_list)  # phiのデバッグ
        #multi_imp_plot2(self.X, self.y_list, self.X_train_list, self.y_train_list, phi_mu, phi_var, f_list, 'imp_pred_with_rfm_'+str(dim)+'.pdf')
        #multi_imp_plot3(self.X,self.y_list,self.X_train_list,self.y_train_list,phi_mu,phi_var,self.gpy_pred_mean_task_list,self.gpy_pred_var_task_list,f_list,dim,'random_vs_gpy_pred_dim'+str(dim)+'.pdf')
        return f_list, Theta_mean_list, Theta_var_list, out_list
        
        #目的関数fの近似を取得する。
        #f = np.dot(theta, phi.T)
        #return f
    
    def gen_prior(self,func_num):
        dim = 1000
        random_state = check_random_state(self.seed)
        omega_x = (np.sqrt(1 / (self.lengthscale ** 2)) * random_state.normal(size=(dim, self.XX.shape[1]-1)))
        omega_t = (np.sqrt(1 / (self.task_lengthscale ** 2)) * random_state.normal(size=(dim, 1)))
        omega = np.hstack((omega_x, omega_t))
        b = np.c_[np.random.rand(dim) * 2 * np.pi]  #[0,2π]の一様乱数
        #RFMから特徴量ベクトルΦ(x)を取得
        
        large_phi = multi_rfm(input_xt.T, dim, omega, b, self.variance)  #D=100とした  10*1000
        Theta = ThetaGenerator(self.seed,dim, self.noise_var)
        Theta.calc_init(large_phi)
        theta=Theta.getTheta(func_num)
        f = np.dot(theta, large_phi.T)
        return f