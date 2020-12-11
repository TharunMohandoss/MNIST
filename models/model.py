import torch
from utils.dirs import create_dirs
import os
from Generators.Generator import Generator
from Discriminators.Discriminator import Discriminator
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd

def calc_gradient_penalty(netD,real_data_list,fake_data_list,BATCH_SIZE):

    interpolates_list = []
    alpha = torch.rand(BATCH_SIZE, 1).cuda()
    for i in range(len(real_data_list)):
        real_data = real_data_list[i]
        fake_data = fake_data_list[i]
        alpha2 = alpha.expand(BATCH_SIZE, int(real_data.nelement()/BATCH_SIZE)).contiguous().view(BATCH_SIZE, 
        list(real_data.shape)[1], list(real_data.shape)[2],  list(real_data.shape)[3]).cuda()
        interpolates = alpha2 * real_data + ((1 - alpha2) * fake_data)
        interpolates_list.append(interpolates)

    disc_interpolates = netD(interpolates_list,True)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates_list[0],
        grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    for i in range(1,len(real_data_list)):
        grad = autograd.grad(outputs=disc_interpolates, inputs=interpolates_list[i],
            grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
            create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad = grad.view(grad.size(0),-1)
        gradients = torch.cat((gradients,grad),1)

    gradient_penalty_elementwise = ((grad.norm(2, dim = 1) - 1) ** 2)*10
    return gradient_penalty_elementwise




class Model():
    def __init__(self, config):
        self.config = config
        self.build_model()
        torch.autograd.set_detect_anomaly(True)

    def build_model(self):
        self.Generator = Generator()
        self.Generator = self.Generator.cuda()
        self.Discriminator = Discriminator()
        self.Discriminator = self.Discriminator.cuda()
        self.generator_optimizer = optim.RMSprop(self.Generator.parameters(), 
            lr=self.config.learning_rate_generator)
        self.discriminator_optimizer = optim.RMSprop(self.Discriminator.parameters(),
            lr=self.config.learning_rate_discriminator)




    def run_batch(self,data_batch,mode,cur_step):
        self.run_batch2(data_batch,mode,cur_step,True)
        return self.run_batch2(data_batch,mode,cur_step,False)


    def run_batch2(self,data_batch,mode,cur_step,train_gen):
        #use mode
        if mode=="train":
            self.Generator.train(True)
            self.Discriminator.train(True)
        elif mode=="test":
            self.Generator.train(False)
            self.Discriminator.train(False)

        #results dict
        step_log = dict()
        data_out_dict = data_batch

        true_images_tensor = torch.from_numpy(data_batch["true_images"]).cuda()
        labels = data_batch["labels"]

        true_images_tensor_list = [true_images_tensor]
        scale_factor = 0.5
        for i in range(3):
            resized = F.interpolate(true_images_tensor,scale_factor=scale_factor,mode='bilinear')
            true_images_tensor_list = [resized]+true_images_tensor_list
            scale_factor *= 0.5


        # batch_size = len(labels)
        # one_hot = np.zeros((batch_size,10)).astype(np.float32)
        # for i in range(batch_size):
        #     one_hot[i][labels[i]] = 1
        # one_hot = torch.from_numpy(one_hot).cuda()

        generated_images_list = self.Generator(self.config.batch_size)
        generated_images = generated_images_list[-1]
        # print('shape : ',generated_images.shape)

        discriminator_fake_output = self.Discriminator(generated_images_list,classifier_only=True)
        discriminator_real_output,classifier_output = self.Discriminator(true_images_tensor_list,classifier_only=False)
        # print(discriminator_real_output)
        # print(discriminator_fake_output)
        # print(classifier_output)


        grad_penalty = calc_gradient_penalty(self.Discriminator,true_images_tensor_list,generated_images_list,self.config.batch_size)
        
        discriminator_loss_actual_examplewise = discriminator_fake_output \
            - discriminator_real_output 
        discriminator_loss_examplewise = discriminator_fake_output \
            - discriminator_real_output + grad_penalty.unsqueeze(1) \
            + 0.0001*(discriminator_fake_output*discriminator_fake_output
                +discriminator_real_output*discriminator_real_output)
        discriminator_loss = torch.mean(discriminator_loss_examplewise)

        generator_loss_examplewise = - discriminator_fake_output
        generator_loss = torch.mean(generator_loss_examplewise)

        if(mode=="train") and train_gen:
            #Train generator
            self.generator_optimizer.zero_grad()
            generator_loss.backward()
            self.generator_optimizer.step()

        if(mode=="train") and not train_gen:
            #Train discriminator
            self.discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            self.discriminator_optimizer.step()

        if not train_gen:
            if((cur_step%self.config.visualization_frequency)==0):

                bgr_images_list = []
                for i in range(len(generated_images_list)):
                    gen_numpy = F.interpolate(generated_images_list[i],size=(256,256),mode='nearest').detach().cpu().numpy()
                    bgr_image = np.expand_dims(gen_numpy[:,0,:,:],axis=1)
                    bgr_images_list.append(bgr_image)

                data_out_dict['bgr_generated_images'] = bgr_images_list[-1]
                bgr_images_list = np.swapaxes(np.asarray(bgr_images_list),0,1)
                data_out_dict['bgr_gen_images_list'] = bgr_images_list


                bgr_true_images_list = []
                for i in range(len(generated_images_list)):
                    gen_numpy = F.interpolate(true_images_tensor_list[i],size=(256,256),mode='nearest').detach().cpu().numpy()
                    bgr_image = np.expand_dims(gen_numpy[:,0,:,:],axis=1)
                    bgr_true_images_list.append(bgr_image)
                bgr_true_images_list = np.swapaxes(np.asarray(bgr_true_images_list),0,1)
                data_out_dict['bgr_true_images_list'] = bgr_true_images_list


                data_out_dict['bgr_true_images'] = np.expand_dims(data_out_dict['true_images'][:,0,:,:],axis=1)


                data_out_dict['generated_images'] = generated_images.detach().cpu().numpy()
                data_out_dict['discriminator_real_output'] = discriminator_real_output.detach().cpu().numpy()
                data_out_dict['discriminator_fake_output'] = discriminator_fake_output.detach().cpu().numpy()
                data_out_dict['generator_loss'] = generator_loss_examplewise.detach().cpu().numpy()
                data_out_dict['discriminator_loss'] = discriminator_loss_examplewise.detach().cpu().numpy()
                data_out_dict['discriminator_loss_actual'] = discriminator_loss_actual_examplewise.detach().cpu().numpy()
                data_out_dict['grad_penalty'] = grad_penalty.detach().cpu().numpy()

                data_out_dict['order'] = ['bgr_true_images','bgr_generated_images','discriminator_real_output','discriminator_fake_output',
                        'generator_loss','grad_penalty','discriminator_loss','discriminator_loss_actual',
                        "bgr_gen_images_list","bgr_true_images_list"]
                data_out_dict['type_list'] = ['image','image',"numerical","numerical",
                    "numerical","numerical","numerical","numerical","imglist","imglist"]
                    



            step_log["generator_loss"] = generator_loss.detach().cpu().numpy()
            step_log["discriminator_loss"] = discriminator_loss.detach().cpu().numpy()
            step_log["discriminator_actual_loss"] = torch.mean(discriminator_loss_actual_examplewise).detach().cpu().numpy()
            step_log["grad_penalty"] = torch.mean(grad_penalty).detach().cpu().numpy()

            return step_log,data_out_dict





