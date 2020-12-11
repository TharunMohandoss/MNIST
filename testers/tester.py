import numpy as np
import sys
from utils.Visualizer import Visualizer
import time
import matplotlib.pyplot as plt
import os
import time

def plotGraphs(folder,results):
    for result in results:
        epoch_list = list(range(1,len(results[result])+1))
        plt.plot(epoch_list,results[result])
        # print('result :',  result)
        # print('results[result]:',results[result])
        plt.xlabel('epoch')
        plt.ylabel(result)
        plt.savefig(os.path.join(folder,result+'_test.png'))
        plt.close()
        plt.clf()



class Tester():
    def __init__(self,  model, datagen, config):
        self.model = model
        self.config = config
        self.datagen = datagen
        self.visualizer = Visualizer(config,'test')

    def test(self,cur_epoch):
        accumulated_results = dict()
        print()
        print()
        print()
        print('Testing now')
        print('Epoch ',cur_epoch)
        print(' -------------------------------')
        results_epoch = self.test_epoch(cur_epoch)

        for current_result in results_epoch:
            if current_result in accumulated_results:
                accumulated_results[current_result].append(results_epoch[current_result])
            else:
                accumulated_results[current_result] = [results_epoch[current_result]]

        plotGraphs(self.config.summary_dir, accumulated_results)



    def test_epoch(self,cur_epoch):
        num_steps = self.datagen.get_num_batches()
        accumulated_results  = dict()
        start_time_epoch = time.time()
        for i in range(num_steps):            
            start_time = time.time()
            results_step,data_batch = self.test_step(i)

            if ( (((i+1)%self.config.step_result_print_frequency)==0) ):
                print(' Step : ',i+1 , " of total : ",num_steps)

            for current_result in results_step:
                if current_result in accumulated_results:
                    accumulated_results[current_result].append(results_step[current_result])
                else:
                    accumulated_results[current_result] = [results_step[current_result]]
                if ( (((i+1)%self.config.step_result_print_frequency)==0) ):
                    print(current_result, " : ", results_step[current_result])
            

            if((i%self.config.visualization_frequency)==0):
                start_t = time.time()
                order = data_batch['order']
                type_list = data_batch['type_list']
                self.visualizer.Visualize(data_batch,order,type_list,cur_epoch,accumulate=True)

            sys.stdout.flush()

            if ( (((i+1)%self.config.step_result_print_frequency)==0) ):
                print('step took : ',time.time()-start_time)

        print('Overall test results of epoch ',cur_epoch,' : ')
        for current_result in accumulated_results:
            accumulated_results[current_result] = np.mean(accumulated_results[current_result])
            print(current_result,' : ',accumulated_results[current_result])
        self.visualizer.reset()
        print('test took : ',time.time()-start_time_epoch)
        print('-------------------------------')

        return accumulated_results


    def test_step(self,cur_step):
        start_time = time.time()
        data_train_step = self.datagen.get_batch()
        data_time = time.time()
#         print('data_time ',data_time-start_time)
        results_step,data_batch = self.model.run_batch(data_train_step,"test",cur_step)
        run_time = time.time()
#         print('run_time',run_time-data_time)
        return results_step,data_batch







