import time
import random

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from algos.GRAPH.Ising import Ising
from algos.GRAPH.Precision import Precision
from algos.GRAPH.Covariance import Covariance

class ModelTest:
    def __init__(self,model_type,data_name='',normalization=True,showfig=True):
        self.model_type = model_type
        self.showfig = showfig
        self.normalization = normalization
        self.name = model_type + '_' + data_name

    def processing(self,data,group_data):
        self.data = data
        self.group_data = group_data
        self.feature_size = self.data.shape[1]
        self.sample_size = self.data.shape[0]
        self.group_size = len(self.group_data)
        self.group_sample_size_list = [len(self.group_data[i]) for i in range(self.group_size)]

        # Normalization
        if self.normalization:
            scaler = StandardScaler()
            self.data = scaler.fit_transform(self.data)
            for i in range(self.group_size):
                scaler = StandardScaler()
                self.group_data[i] = scaler.fit_transform(self.group_data[i])
            self.global_covariance = self.data.T@self.data/self.sample_size
        else:
            self.global_covariance = self.data
        return self.data, self.group_data
    
    def load_model(self,parameters):
        self.lam = parameters['lam']
        if self.model_type == 'Precision':
            model = Precision(T=parameters['max_iter'],N=self.feature_size,lam=self.lam)
        elif self.model_type == 'Covariance':
            self.tau = parameters['tau']
            model = Covariance(T=parameters['max_iter'],N=self.feature_size,tau=self.tau,lam=self.lam)
        elif self.model_type == 'Ising':
            model = Ising(T=parameters['max_iter'],N=self.feature_size,lam=self.lam,step_size=parameters['step_size'])
        return model
    
    def global_graph(self,data,group_data,parameters):
        data, _ = self.processing(data,group_data)
        model = self.load_model(parameters)

        def history_plot(name):
            plt.figure(figsize=(5,2.5))
            plt.plot(history)
            plt.ylabel('Loss', fontsize=10)
            plt.xlabel('Iteration', fontsize=10)
            plt.title(name, fontsize=10)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.show()
        
        if self.model_type == 'Ising':
            self.global_output, history = model.compute(data,True)
        else:
            self.global_output, history = model.compute(self.global_covariance,True)

        if self.showfig:
            history_plot(self.model_type)
        return

    def group_graph(self,data,group_data,parameters):
        _, group_data = self.processing(data,group_data)
        model = self.load_model(parameters)
        
        self.group_output = []
        if self.model_type == 'Ising':
            # self.global_output, history = model.compute(data,True)
            for i in range(self.group_size):
                self.group_output.append(model.compute(group_data[i]))
        else:
            # self.global_output, history = model.compute(self.global_covariance,True)
            for i in range(self.group_size):
                self.group_output.append(model.compute(self.group_data[i].T@self.group_data[i]/self.group_sample_size_list[i]))
        return
    
    def optuna_score(self):
        if self.model_type == 'Covariance':
            from utils.GRAPH.covariance import disparity
            fair_disparity = disparity(self.group_output,self.mf_output,self.group_data,self.tau,self.lam)/self.group_size

        if self.model_type == 'Precision':
            from utils.GRAPH.precision import disparity
            fair_disparity = disparity(self.group_output,self.mf_output,self.group_data,self.lam)/self.group_size

        if self.model_type == 'Ising':
            from utils.GRAPH.ising import disparity
            fair_disparity = disparity(self.group_output,self.mf_output,self.group_data,self.lam)/self.group_size

        return fair_disparity
            
    def plot(self):
        cmap = 'Blues'
        _, axs = plt.subplots(1, 2, figsize=(5, 2.5))
        arr1 = np.abs(self.global_output-np.diag(np.diag(self.global_output)))

        sns.heatmap(arr1, ax=axs[0], cmap=cmap, cbar=False)
        axs[0].set_title('Global Output',fontsize=7)
        axs[0].axis('off')

        axs[1].axis('off')
        arr2 = np.abs(self.mf_output-np.diag(np.diag(self.mf_output)))
        sns.heatmap(arr2, ax=axs[1], cmap=cmap, cbar=False)
        axs[1].set_title('Fair Output (multi)',fontsize=7)

        plt.tight_layout()
        plt.show()

        return

    def difference_plot(self):
        cmap = 'Blues'
        fair_output = self.mf_output

        _, axs = plt.subplots(1, 3, figsize=(7.5, 2.5))
        arr1 = np.abs(self.global_output-np.diag(np.diag(self.global_output)))
        arr2 = np.abs(fair_output-np.diag(np.diag(fair_output)))

        sns.heatmap(arr1, ax=axs[0], cmap=cmap, cbar=False)
        axs[0].set_title('Global Output',fontsize=7)
        axs[0].axis('off')

        sns.heatmap(arr2, ax=axs[1], cmap=cmap, cbar=False)
        axs[1].set_title('Fair Output',fontsize=7)
        axs[1].axis('off')

        sns.heatmap(np.abs(self.global_output-fair_output), ax=axs[2], cmap=cmap, cbar=False)
        axs[2].set_title('Difference',fontsize=7)
        axs[2].axis('off')

        plt.tight_layout()
        plt.show()
        return
    
    def summary(self,saveres=False):
        if self.model_type == 'Precision':
            from utils.GRAPH.precision import loss_l1, disparity
        elif self.model_type == 'Covariance':
            from utils.GRAPH.covariance import loss_l1, disparity
        elif self.model_type == 'Ising':
            from utils.GRAPH.ising import loss_l1, disparity

        if self.model_type == 'Covariance':
            print("\033[1m"+'Graph Learning'+"\033[0m")
            original_loss = loss_l1(self.global_output,self.global_covariance,self.tau,self.lam)
            print("\033[4m"+'overall global results:'+"\033[0m", f'{original_loss:.5f}')
            original_disparity = disparity(self.group_output,self.global_output,self.group_data,self.tau,self.lam)/self.group_size
            print("\033[4m"+'disparity error:'+"\033[0m", f'{original_disparity:.6f}', '\n')
        else:
            print("\033[1m"+'Graph Learning'+"\033[0m")
            original_loss = loss_l1(self.global_output,self.global_covariance,self.lam)
            print("\033[4m"+'overall global results:'+"\033[0m", f'{original_loss:.5f}')

            original_disparity = disparity(self.group_output,self.global_output,self.group_data,self.lam)/self.group_size
            print("\033[4m"+'disparity error:'+"\033[0m", f'{original_disparity:.6f}', '\n')
        
        if saveres:
            column_names = ['method', 'loss', 'disparity', 'loss improve', 'loss improve (%)', 'disparity improve', 'disparity improve (%)']
            res = pd.DataFrame(columns=column_names)
            res = res.set_index('method')
            res.loc['graph'] = [original_loss, original_disparity, 0, 0, 0, 0]

        def subsummary(fair_output):
            if self.model_type == 'Covariance':
                print("\033[1m"+'Objective Fair Graph Learning'+"\033[0m")
                fair_loss = loss_l1(fair_output,self.global_covariance,self.tau,self.lam)
                print("\033[4m"+'overall global results:'+"\033[0m", f'{fair_loss:.5f}')
                fair_disparity = disparity(self.group_output,fair_output,self.group_data,self.tau,self.lam)/self.group_size
                print("\033[4m"+'disparity error:'+"\033[0m", f'{fair_disparity:.6f}')

            else:
                print("\033[1m"+'Objective Fair Graph Learning'+"\033[0m")
                fair_loss = loss_l1(fair_output,self.global_covariance,self.lam)
                print("\033[4m"+'overall global results:'+"\033[0m", f'{fair_loss:.5f}')
                fair_disparity = disparity(self.group_output,fair_output,self.group_data,self.lam)/self.group_size
                print("\033[4m"+'disparity error:'+"\033[0m", f'{fair_disparity:.6f}')

            objective_improve = original_loss-fair_loss
            percent_change_objective = objective_improve/original_loss*100
            print("\033[3m"+'objective improvement:', f'{objective_improve:.5f}', f'({percent_change_objective:.5f}%)'"\033[0m")
            disparity_improve = original_disparity-fair_disparity
            percent_change_disparity = disparity_improve/original_disparity*100
            print("\033[3m"+'disparity improvement:', f'{disparity_improve:.6f}', f'({percent_change_disparity:.5f}%)'"\033[0m")

            return [fair_loss, fair_disparity, objective_improve, percent_change_objective, disparity_improve, percent_change_disparity]

        temp_res = subsummary(self.mf_output)
        if saveres:
            res.loc['multi'] = temp_res

        if saveres:
            res.to_csv('res/'+self.model_type.lower()+'/numerical_res/'+'numerical_'+self.name+'.csv')
            return
        return
        
    def runtime(self,num_times,data,group_data,parameters,output=False):
        data, group_data = self.processing(data,group_data)
        model = self.load_model(parameters)

        if self.model_type == 'Precision':
            from utils.GRAPH.precision import loss_l1, disparity
        elif self.model_type == 'Covariance':
            from utils.GRAPH.covariance import loss_l1, disparity
        elif self.model_type == 'Ising':
            from utils.GRAPH.ising import loss_l1, disparity

        global_loss, global_disp, global_runtime = [], [], []
        mf_loss, mf_disp, mf_runtime = [], [], []

        for _ in range(num_times):

            if self.model_type == 'Ising':
                start_time = time.time()
                self.global_output = model.compute(data)
                end_time = time.time()
                global_runtime.append(end_time - start_time)

            else:
                start_time = time.time()
                self.global_output = model.compute(self.global_covariance)
                end_time = time.time()
                global_runtime.append(end_time - start_time)

            start_time = time.time()
            self.mf_output = model.multi_fair_compute(Ys=self.group_data,Y=self.data,rhom=parameters['rhom'],lamm=parameters['lamm'],tol=parameters['tol'])
            end_time = time.time()
            mf_runtime.append(end_time - start_time)
            
            if self.model_type == 'Covariance':
                original_loss = loss_l1(self.global_output,self.global_covariance,self.tau,self.lam)
                original_disparity = disparity(self.group_output,self.global_output,self.group_data,self.tau,self.lam)/self.group_size
                fair_loss = loss_l1(self.mf_output,self.global_covariance,self.tau,self.lam)
                fair_disparity = disparity(self.group_output,self.mf_output,self.group_data,self.tau,self.lam)/self.group_size
            else:
                original_loss = loss_l1(self.global_output,self.global_covariance,self.lam)
                original_disparity = disparity(self.group_output,self.global_output,self.group_data,self.lam)/self.group_size
                fair_loss = loss_l1(self.mf_output,self.global_covariance,self.lam)
                fair_disparity = disparity(self.group_output,self.mf_output,self.group_data,self.lam)/self.group_size

            global_loss.append(original_loss)
            global_disp.append(original_disparity)

            mf_loss.append(fair_loss)
            mf_disp.append(fair_disparity)

        if output:
            return global_runtime, mf_runtime, global_loss, mf_loss, global_disp, mf_disp
        else:
            return