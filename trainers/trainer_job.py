import numpy as np
import sys
from utils.Visualizer import Visualizer
import time
import shutil
import descarteslabs as dl
import matplotlib.pyplot as plt
import os
from utils.dirs import create_dirs,emptyInsideFolder
import psutil

class Trainer():
    def __init__(self,  model, datagen, config , tester):
        self.model = model
        self.config = config
        self.datagen = datagen
        self.tester = tester
        self.visualizer = Visualizer(config,'train')
        self.storage_client=dl.Storage()

    def plotGraphs(self,folder,results):
        for result in results:
            epoch_list = list(range(1,len(results[result])+1))
            new_list = [float(x) for x in results[result]]
            plt.plot(epoch_list,new_list)
#             print('result :',  result)
#             print('results[result]:',results[result])
            plt.xlabel('epoch')
            plt.ylabel(result)
            plt.savefig(os.path.join(folder,result+'.png'))
            self.storage_client.set_file(self.config.exp_name+"_trainviz_"+result, 
                os.path.join(folder,result+'.png'), storage_type='data')
            plt.close()
            plt.clf()

    def train(self):
        accumulated_results = dict()
        for cur_epoch in range(self.config.num_epochs):
#             print()
#             print()
#             print()
#             print('Epoch ',cur_epoch)
#             print(' -------------------------------')
            results_epoch = self.train_epoch(cur_epoch)
            if(not (self.tester is None)):
                self.tester.test(cur_epoch)

            for current_result in results_epoch:
                if current_result in accumulated_results:
                    accumulated_results[current_result].append(results_epoch[current_result])
                else:
                    accumulated_results[current_result] = [results_epoch[current_result]]

            self.plotGraphs(self.config.summary_dir, accumulated_results)

    def train_one_epoch(self,accumulated_results,cur_epoch):
        results_epoch = self.train_epoch(cur_epoch)

        for current_result in results_epoch:
            if current_result in accumulated_results:
                accumulated_results[current_result].append(str(results_epoch[current_result]))
            else:
                accumulated_results[current_result] = [str(results_epoch[current_result])]

        self.plotGraphs(self.config.summary_dir, accumulated_results)

        return accumulated_results

    def train_epoch(self,cur_epoch):
        num_steps = self.datagen.get_num_batches()
        accumulated_results  = dict()
        start_time_epoch = time.time()
        fo = open('/tmp/step.txt','w')
        fo.close()
        for i in range(num_steps):            
            start_time = time.time()
            results_step,data_batch = self.train_step(i)

            if ( (((i+1)%self.config.step_result_print_frequency)==0) ):
#                 print(' Step : ',i+1 , " of total : ",num_steps)
                fo = open('/tmp/step.txt','w')
                fo.write('\nStep : '+str(i+1)+", epoch : "+str(cur_epoch))
                # fo.close()
                # self.storage_client.set_file(self.config.exp_name+'_steplog','/tmp/step.txt', storage_type='data')

            for current_result in results_step:
                if current_result in accumulated_results:
                    accumulated_results[current_result].append(results_step[current_result])
                else:
                    accumulated_results[current_result] = [results_step[current_result]]
#                 if ( (((i+1)%self.config.step_result_print_frequency)==0) ):
#                     print(current_result, " : ", results_step[current_result])
            

            if((i%self.config.visualization_frequency)==0):
                start_t = time.time()
                order = data_batch['order']
                type_list = data_batch['type_list']
                self.visualizer.Visualize(data_batch,order,type_list,cur_epoch,accumulate=False)
                shutil.make_archive(self.config.exp_name+"_viz",'zip',self.config.visualization_dir)
                self.storage_client.set_file(self.config.exp_name+"_viz", self.config.exp_name+"_viz.zip", storage_type='data')
                emptyInsideFolder(self.config.visualization_dir)
#                 print('viz time : ',time.time()-start_t)

            sys.stdout.flush()

            if ( (((i+1)%self.config.step_result_print_frequency)==0) ):
                # fo = open('/tmp/step.txt','a+')
                fo.write('\nstep took : '+str(time.time()-start_time))
                hdd = psutil.disk_usage('/home/')
                fo.write("\nTotal: "+str(hdd.total*1.0 / (2**30))+" GiB")  
                fo.write("\nUsed: "+str(hdd.used*1.0 / (2**30))+" GiB")    
                fo.write("\nFree: "+str(hdd.free*1.0 / (2**30))+" GiB")    
                hdd = psutil.disk_usage('/')
                fo.write("\nTotal: "+str(hdd.total*1.0 / (2**30))+" GiB")  
                fo.write("\nUsed: "+str(hdd.used*1.0 / (2**30))+" GiB")    
                fo.write("\nFree: "+str(hdd.free*1.0 / (2**30))+" GiB")    
                fo.close()
                self.storage_client.set_file(self.config.exp_name+'_steplog','/tmp/step.txt', storage_type='data')
                os.remove('/tmp/step.txt')
#                 print('step took : ',time.time()-start_time)

#         print('Overall train results of epoch ',cur_epoch,' : ')
        for current_result in accumulated_results:
            accumulated_results[current_result] = np.mean(accumulated_results[current_result])
#             print(current_result,' : ',accumulated_results[current_result])
        if( ((cur_epoch%self.config.model_save_frequency)==0)):
#             print('saving')
            self.model.save(cur_epoch)
        self.visualizer.reset()
#         print('epoch took : ',time.time()-start_time_epoch)
#         print('-------------------------------')

        return accumulated_results



    def train_step(self,cur_step):
        start_time = time.time()
        data_train_step = self.datagen.get_batch()
        data_time = time.time()
#         print('data_time ',data_time-start_time)
        results_step,data_batch = self.model.run_batch(data_train_step,"train",cur_step)
        run_time = time.time()
#         print('run_time',run_time-data_time)
        return results_step,data_batch







