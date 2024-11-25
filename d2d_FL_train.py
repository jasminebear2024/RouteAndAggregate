import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, ConnectionPatch
import math
# from lwc_d2dFL.topology_update import TopologyManager

import copy
import logging
import random
import time
from scipy.stats import bernoulli
import torch

import string

from FedAvgAPI.client import Client
DEFAULT_TRAIN_CLIENTS_NUM = 20
DEFAULT_TEST_CLIENTS_NUM = 20

class FedAvgAPI(object):
    def __init__(self, dataset, device, args, topology, model_trainer):
        self.device = device
        self.args = args
        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.topology = topology
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.it_list = []
        # self.client_ids = train_client_ids
        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict

        self.model_trainer = model_trainer
        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer)

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.d2d_user_num):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer)
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")



    def train(self):
        logging.info(self.model_trainer)
        w_global = self.model_trainer.get_model_params()
        models_of_last_round = []
        models_of_new_round = []
        for client_idx in range(self.args.d2d_user_num):
            models_of_last_round.append(w_global)
            models_of_new_round.append(w_global)


        for round_idx in range(self.args.comm_round):



            logging.info("################Communication round : {}".format(round_idx))

            # w_locals = []
            start = time.time()
            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """
            N = self.args.localN
            logging.info("Iteration " + str(N))

            if self.args.label_divided_num==10:
                client_indexes=range(10)
            else:
                client_indexes = self._client_sampling(6, DEFAULT_TRAIN_CLIENTS_NUM, self.args.d2d_user_num)

            # if round_idx == 0:
            #     self._local_test_on_all_clients(round_idx, client_indexes, models_of_last_round)

            for idx, client in enumerate(self.client_list):#user iteration change
                # update dataset
                #logging.info("client_indexes = " + str(idx))
                client_idx = client_indexes[idx]
               

                client.update_local_dataset(idx, self.train_data_local_dict[client_idx],self.test_data_local_dict[client_idx],self.train_data_local_num_dict[client_idx])
                Iterations = N
                for its in range(Iterations):
                   if its == 0:
                       w = client.train(copy.deepcopy(models_of_last_round[idx]))
                   else:
                       w = client.train(copy.deepcopy(w) )

                models_of_new_round[idx] = (client.get_sample_number(), copy.deepcopy(w))

            if self.args.topology_type == 0:
                models_of_last_round=self.routing_PER_replace_with_own_model(models_of_new_round)
            if self.args.topology_type == 1:
                models_of_last_round=self.routing_PER_replace_with_zero(models_of_new_round)#routing_aggregate_average
            if self.args.topology_type == 2:
                models_of_last_round=self.routing_PER_replace_with_averaged_model(models_of_new_round)
            if self.args.topology_type == 3:
                models_of_last_round=self.consensus_aggregate_PER_replace_with_own_model(models_of_new_round)
            if self.args.topology_type == 4:
                models_of_last_round=self.consensus_aggregate_PER_replace_with_zero(models_of_new_round)#consensus_aggregate_with_communication_errors_one_node_failure
            if self.args.topology_type == 5:
                models_of_last_round = self.consensus_aggregate_PER_replace_with_averaged_model(models_of_new_round)
            if self.args.topology_type == 6:
                models_of_last_round = self.centralized_PER_replace_with_own_model(models_of_new_round)
            if self.args.topology_type == 7:
                models_of_last_round = self.centralized_PER_replace_with_zero(models_of_new_round)
            if self.args.topology_type == 8:
                models_of_last_round = self.centralized_PER_replace_with_averaged_model(models_of_new_round)
            end = time.time()
            logging.info("training time = " + str(end-start))
            # test results at last round
            if round_idx == self.args.comm_round - 1:
                self._local_test_on_all_clients(round_idx,client_indexes,models_of_last_round)
                # per {frequency_of_the_test} round
            elif round_idx % self.args.frequency_of_the_test == 0:
                if self.args.dataset.startswith("stackoverflow"):
                    self._local_test_on_validation_set(round_idx)
                else:
                    self._local_test_on_all_clients(round_idx,client_indexes,models_of_last_round)

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if self.args.dataset == 'femnist':
            if client_num_in_total == client_num_per_round:
                client_indexes = [client_index for client_index in range(client_num_in_total)]
            else:
                num_clients = min(client_num_per_round, client_num_in_total)
                np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
                client_indexes = []
                step=client_num_in_total // self.args.label_divided_num
                if num_clients==13:
                    client_indexes=[355, 447, 938,1610,1726, 2295,2942,3172,3173,3119, 447, 2295,938]
                elif num_clients==16:
                    client_indexes=[355, 447, 938,1610,1726, 2295,2942,3172,3173,3119, 447, 2295,938,1610,1726,355]
                elif num_clients==10:
                    client_indexes=[355, 447, 938,1610,1726, 2295,2942,3172,3173,3119]
                elif num_clients == 6:
                    client_indexes = [355, 447, 938, 1610, 1726, 2295]
                elif num_clients == 8:
                    client_indexes = [355, 447, 938, 1610, 1726, 2295,2942,3172]
                elif num_clients == 12:
                    client_indexes = [355, 447, 938, 1610, 1726, 2295, 2942, 3172, 3173, 3119, 447, 2295]
                elif num_clients == 15:
                    client_indexes = [355, 447, 938, 1610, 1726,2295,2942,3172,3173,3119,229,467,1600,1740,2567]
                else:
                    for i in range(self.args.label_divided_num):

                        if num_clients % self.args.label_divided_num >i:
                            sample_num=(num_clients // self.args.label_divided_num)+1
                        else:
                            sample_num=(num_clients // self.args.label_divided_num)
                        client_indexes = np.append(client_indexes,np.random.choice(range(step*i,step*(i+1)), (sample_num), replace=False))
                    # client_indexes = np.append(client_indexes, np.random.choice(range(client_num_in_total-step, client_num_in_total),
                    #                                                             (num_clients - (self.args.label_divided_num - 1) * (num_clients//self.args.label_divided_num)),
                    #                                                             replace=False))
            logging.info("client_indexes = %s" % str(client_indexes))
        elif self.args.dataset=='fed_cifar100':
            client_num_in_total=100
            if client_num_in_total == client_num_per_round:
                client_indexes = [client_index for client_index in range(client_num_in_total)]
            else:
                num_clients = min(client_num_per_round, client_num_in_total)
                np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
                client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
            logging.info("client_indexes = %s" % str(client_indexes))
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
            logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        test_data_num  = len(self.test_global.dataset)
        sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
        subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
        sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
        self.val_global = sample_testset

    def _aggregate(self, w_locals):
        training_num = 0
        logging.info("len(w_locals) " + str(len(w_locals)))
        U = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            ww = sample_num/training_num
            U += ww*self.it_list[idx]
        logging.info("U " + str(U))
        # \
        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params
    def routing_aggregate_average(self, models_last_round):
        (sample_num, averaged_params) = models_last_round[0]
        # logging.info("(sample_num, averaged_params）: %s, %s"%(sample_num, averaged_params))
        new_models = []


        for useri in range(0, len(models_last_round)):

            for k in averaged_params.keys():
                # logging.info("averaged_params keys:" + str(k))
                # logging.info("averaged_params:"+str(averaged_params))
                params = None
                for userj in range(0, len(models_last_round)):

                    local_sample_number, local_model_params = models_last_round[userj]

                        # logging.info("str"+str(userj))
                        # logging.info("local_model_params[k]"+str(local_model_params[k]))
                    if params is None:
                        params =  self.topology.onehop_error[useri][userj]*local_model_params[k] / self.args.d2d_user_num
                    else:
                        params +=  self.topology.onehop_error[useri][userj]*local_model_params[k] / self.args.d2d_user_num
                    # params = self.topology.onehop_error[useri][userj]*local_model_params[k] / self.args.d2d_user_num if params is None \
                    #     else params + self.topology.onehop_error[useri][userj]*local_model_params[k] / self.args.d2d_user_num

                    # logging.info('2')
                    # logging.info(params2)
                averaged_params[k] =copy.deepcopy(params)
                # logging.info("local_model_params[k]" + str(params))
            new_models.append(copy.deepcopy(averaged_params))
        return new_models
    def routing_PER_replace_with_averaged_model(self,models_last_round):
        (sample_num, averaged_params) = models_last_round[0]
        # logging.info("(sample_num, averaged_params）: %s, %s"%(sample_num, averaged_params))
        new_averaged_params = copy.deepcopy(averaged_params)
        # new_models2=[]#用于过渡
        new_models = []
        one_hop_error = self.topology.onehop_error
        path = self.topology.path
        for useri in range(0, len(models_last_round)):
            for k in averaged_params.keys():
                # logging.info("averaged_params keys:" + str(k))
                # logging.info("averaged_params:"+str(averaged_params))
                params = None
                ones_sum=torch.zeros_like(averaged_params[k])


                for userj in range(0, len(models_last_round)):

                    local_sample_number, local_model_params = models_last_round[userj]
                    path_nodes = len(path[userj][useri])
                    zeros_ones = torch.ones_like(local_model_params[k])
                    for i in range(path_nodes - 1):
                        a = torch.ones_like(local_model_params[k])
                        a = a * one_hop_error[path[userj][useri][i], path[userj][useri][i + 1]]
                        zeros_ones1 = torch.bernoulli(a)
                        zeros_ones = copy.deepcopy(torch.mul(zeros_ones1, zeros_ones))
                    ones_sum = ones_sum + zeros_ones
                    if params is None:
                        params = torch.mul(zeros_ones, local_model_params[k])/self.args.d2d_user_num
                    else:
                        params += torch.mul(zeros_ones, local_model_params[k])/self.args.d2d_user_num

                new_averaged_params[k] = copy.deepcopy(self.args.d2d_user_num*torch.div(params,ones_sum))
                # logging.info("local_model_params[k]" + str(params))
            new_models.append(copy.deepcopy(new_averaged_params))
        return new_models
    def routing_PER_replace_with_own_model(self, models_last_round):
        (sample_num, averaged_params) = models_last_round[0]
        # logging.info("(sample_num, averaged_params）: %s, %s"%(sample_num, averaged_params))
        new_averaged_params = copy.deepcopy(averaged_params)
        # new_models2=[]#用于过渡
        new_models = []
        one_hop_error = self.topology.onehop_error
        path = self.topology.path
        for useri in range(0, len(models_last_round)):
            local_sample_numberi, local_model_paramsi = models_last_round[useri]
            for k in averaged_params.keys():
                # logging.info("averaged_params keys:" + str(k))
                # logging.info("averaged_params:"+str(averaged_params))
                params = None
                for userj in range(0, len(models_last_round)):

                    local_sample_number, local_model_params = models_last_round[userj]
                    path_nodes = len(path[userj][useri])
                    zeros_ones = torch.ones_like(local_model_params[k])
                    # logging.info('userj '+str(userj)+' useri '+str(useri))
                    for i in range(path_nodes - 1):
                        a = torch.ones_like(local_model_params[k])
                        a = a * one_hop_error[path[userj][useri][i], path[userj][useri][i + 1]]
                        zeros_ones1 = torch.bernoulli(a)
                        # logging.info(i)
                        # logging.info(zeros_ones1)
                        zeros_ones = copy.deepcopy(torch.mul(zeros_ones1, zeros_ones))
                    error_1 = zeros_ones * (-1) + torch.ones_like(zeros_ones)
                    local_error_1 = torch.mul(local_model_paramsi[k], error_1)
                    if params is None:
                        params = (torch.mul(zeros_ones, local_model_params[k]) + local_error_1) / self.args.d2d_user_num
                    else:
                        params += (torch.mul(zeros_ones, local_model_params[k])+local_error_1) / self.args.d2d_user_num

                new_averaged_params[k] = copy.deepcopy(params)
                # logging.info("local_model_params[k]" + str(params))
            new_models.append(copy.deepcopy(new_averaged_params))
        return new_models
    def routing_PER_replace_with_zero(self, models_last_round):
        (sample_num, averaged_params) = models_last_round[0]
        # logging.info("(sample_num, averaged_params）: %s, %s"%(sample_num, averaged_params))
        new_averaged_params = copy.deepcopy(averaged_params)
        # new_models2=[]#用于过渡
        new_models = []
        one_hop_error = self.topology.onehop_error
        path = self.topology.path
        for useri in range(0, len(models_last_round)):
            for k in averaged_params.keys():
                # logging.info("averaged_params keys:" + str(k))
                # logging.info("averaged_params:"+str(averaged_params))
                params = None
                for userj in range(0, len(models_last_round)):

                    local_sample_number, local_model_params = models_last_round[userj]
                    path_nodes = len(path[userj][useri])
                    zeros_ones = torch.ones_like(local_model_params[k])
                    for i in range(path_nodes - 1):
                        a = torch.ones_like(local_model_params[k])
                        a = a * one_hop_error[path[userj][useri][i], path[userj][useri][i + 1]]
                        zeros_ones1 = torch.bernoulli(a)
                        # logging.info(zeros_ones1)
                        zeros_ones = copy.deepcopy(torch.mul(zeros_ones1, zeros_ones))
                    if params is None:
                        params = torch.mul(zeros_ones, local_model_params[k]) / self.args.d2d_user_num
                    else:
                        params += torch.mul(zeros_ones, local_model_params[k]) / self.args.d2d_user_num

                new_averaged_params[k] = copy.deepcopy(params)
                # logging.info("local_model_params[k]" + str(params))
            new_models.append(copy.deepcopy(new_averaged_params))
        return new_models
    def centralized_PER_replace_with_zero(self, models_last_round):
        (sample_num, averaged_params) = models_last_round[0]
        # logging.info("(sample_num, averaged_params）: %s, %s"%(sample_num, averaged_params))
        new_averaged_params = copy.deepcopy(averaged_params)
        new_averaged_params2 = copy.deepcopy(averaged_params)
        # new_models2=[]#用于过渡
        new_models=[]
        one_hop_error=self.topology.onehop_error
        path=self.topology.path
        # source_node=self.topology.source_node
        for k in averaged_params.keys():
            # logging.info("averaged_params keys:" + str(k))
            # logging.info("averaged_params:"+str(averaged_params))
            params = None
            for userj in range(0, len(models_last_round)):

                local_sample_number, local_model_params = models_last_round[userj]

                path_nodes=len(path[userj])
                # logging.info(path_nodes)
                zeros_ones=torch.ones_like(local_model_params[k])
                for i in range(path_nodes - 1):
                    a=torch.ones_like(local_model_params[k])
                    a=a* one_hop_error[path[userj][i], path[userj][i + 1]]
                    zeros_ones1=torch.bernoulli(a)
                    # logging.info(zeros_ones1)
                    zeros_ones=copy.deepcopy(torch.mul(zeros_ones1,zeros_ones))
                if params is None:
                    params =torch.mul(zeros_ones, local_model_params[k])*(1/self.args.d2d_user_num)
                else:
                    params += torch.mul(zeros_ones, local_model_params[k]) *(1/self.args.d2d_user_num)


            new_averaged_params[k] = copy.deepcopy(params)

        for userj in range(0, len(models_last_round)):
            for k in averaged_params.keys():
                path_nodes = len(path[userj])
                zeros_ones = torch.ones_like(averaged_params[k])
                for i in range(path_nodes - 1):
                    a = torch.ones_like(averaged_params[k])
                    a = a * one_hop_error[path[userj][i], path[userj][i + 1]]
                    zeros_ones1 = torch.bernoulli(a)
                    zeros_ones = copy.deepcopy(torch.mul(zeros_ones1, zeros_ones))
                new_averaged_params2[k]=copy.deepcopy(torch.mul(zeros_ones, new_averaged_params[k]))
            new_models.append(copy.deepcopy(new_averaged_params2))
        return new_models

    def centralized_PER_replace_with_own_model(self, models_last_round):
        (sample_num, averaged_params) = models_last_round[self.topology.source_node]
        # logging.info("(sample_num, averaged_params）: %s, %s"%(sample_num, averaged_params))
        new_averaged_params = copy.deepcopy(averaged_params)
        new_averaged_params2 = copy.deepcopy(averaged_params)
        # new_models2=[]#用于过渡
        new_models = []
        one_hop_error = self.topology.onehop_error
        path = self.topology.path
        # source_node=self.topology.source_node
        for k in averaged_params.keys():
            # logging.info("averaged_params keys:" + str(k))
            # logging.info("averaged_params:"+str(averaged_params))
            params = None
            for userj in range(0, len(models_last_round)):

                local_sample_number, local_model_params = models_last_round[userj]

                path_nodes = len(path[userj])
                # logging.info(path_nodes)
                zeros_ones = torch.ones_like(local_model_params[k])
                for i in range(path_nodes - 1):
                    a = torch.ones_like(local_model_params[k])
                    a = a * one_hop_error[path[userj][i], path[userj][i + 1]]
                    zeros_ones1 = torch.bernoulli(a)
                    # logging.info(zeros_ones1)
                    zeros_ones = copy.deepcopy(torch.mul(zeros_ones1, zeros_ones))


                error_1 = zeros_ones * (-1) + torch.ones_like(zeros_ones)
                local_error_1 = torch.mul(averaged_params[k], error_1)
                if params is None:
                    params = (torch.mul(zeros_ones, local_model_params[k])+local_error_1)
                else:
                    params += (torch.mul(zeros_ones, local_model_params[k])+local_error_1)
            new_averaged_params[k] = copy.deepcopy(params/ self.args.d2d_user_num)
            # print(ones_sum)
        for userj in range(0, len(models_last_round)):
            local_sample_number, local_model_params = models_last_round[userj]
            for k in averaged_params.keys():
                path_nodes = len(path[userj])
                zeros_ones = torch.ones_like(local_model_params[k])
                for i in range(path_nodes - 1):
                    a = torch.ones_like(local_model_params[k])
                    a = a * one_hop_error[path[userj][i], path[userj][i + 1]]
                    zeros_ones1 = torch.bernoulli(a)
                    zeros_ones = copy.deepcopy(torch.mul(zeros_ones1, zeros_ones))

                error_1 = zeros_ones * (-1) + torch.ones_like(zeros_ones)
                local_error_1 = torch.mul(local_model_params[k], error_1)
                new_averaged_params2[k] = copy.deepcopy(local_error_1+torch.mul(zeros_ones, new_averaged_params[k]))
            new_models.append(copy.deepcopy(new_averaged_params2))
        return new_models

    def centralized_PER_replace_with_averaged_model(self, models_last_round):
        (sample_num, averaged_params) = models_last_round[0]
        # logging.info("(sample_num, averaged_params）: %s, %s"%(sample_num, averaged_params))
        new_averaged_params = copy.deepcopy(averaged_params)
        new_averaged_params2 = copy.deepcopy(averaged_params)
        # new_models2=[]#用于过渡
        new_models = []
        one_hop_error = self.topology.onehop_error
        path = self.topology.path
        for k in averaged_params.keys():
            params = None
            ones_sum = torch.zeros_like(averaged_params[k])
            for userj in range(0, len(models_last_round)):
                local_sample_number, local_model_params = models_last_round[userj]
                path_nodes = len(path[userj])
                zeros_ones = torch.ones_like(averaged_params[k])
                for i in range(path_nodes - 1):
                    a = torch.ones_like(local_model_params[k])
                    a = a * one_hop_error[path[userj][i], path[userj][i + 1]]
                    zeros_ones1 = torch.bernoulli(a)
                    zeros_ones = copy.deepcopy(torch.mul(zeros_ones1, zeros_ones))
                ones_sum = ones_sum + zeros_ones
                # error_1 = zeros_ones * (-1) + torch.ones_like(zeros_ones)
                if params is None:
                    params = torch.mul(zeros_ones, local_model_params[k])
                else:
                    params += torch.mul(zeros_ones, local_model_params[k])

            new_averaged_params[k] = copy.deepcopy(torch.div(params,ones_sum))
            # print(ones_sum)
        for userj in range(0, len(models_last_round)):
            local_sample_number, local_model_params = models_last_round[userj]
            for k in averaged_params.keys():
                path_nodes = len(path[userj])
                zeros_ones = torch.ones_like(local_model_params[k])
                for i in range(path_nodes - 1):
                    a = torch.ones_like(local_model_params[k])
                    a = a * one_hop_error[path[userj][i], path[userj][i + 1]]
                    zeros_ones1 = torch.bernoulli(a)
                    zeros_ones = copy.deepcopy(torch.mul(zeros_ones1, zeros_ones))

                error_1 = zeros_ones * (-1) + torch.ones_like(zeros_ones)
                local_error_1 = torch.mul(local_model_params[k], error_1)
                new_averaged_params2[k] = copy.deepcopy(local_error_1 + torch.mul(zeros_ones, new_averaged_params[k]))
            new_models.append(copy.deepcopy(new_averaged_params2))
        return new_models
    def consensus_aggregate(self, models_last_round):
        (sample_num, averaged_params) = models_last_round[0]
        # logging.info("(sample_num, averaged_params）: %s, %s"%(sample_num, averaged_params))
        new_models = []
        new_averaged_params=copy.deepcopy(averaged_params)

        for useri in range(0, len(models_last_round)):
            training_num = 0
            # logging.info("neighbors" + str(self.topology.neighbors[useri]))
            for k in averaged_params.keys():
                # logging.info("averaged_params keys:" + str(k))
                # logging.info("averaged_params:"+str(averaged_params))
                params =None
                for userj in range(0, len(models_last_round)):
                    # logging.info("topology.neighbors[%s][%s]: %s" % (useri,userj,self.topology.neighbors[useri, userj]))
                    if (self.topology.neighbors[useri, userj] != 0):
                        local_sample_number, local_model_params = models_last_round[userj]
                        if params is None:
                            params = local_model_params[k] * self.topology.neighbors[useri, userj]
                        else: params += local_model_params[k] * self.topology.neighbors[useri, userj]

                new_averaged_params[k] = copy.deepcopy(params)
                # logging.info("local_model_params[k]" + str(params))
            new_models.append(copy.deepcopy(new_averaged_params))

        return new_models

    def consensus_aggregate_with_communication_errors_one_node_failure(self, models_last_round):
        (sample_num, averaged_params) = models_last_round[0]
        # logging.info("(sample_num, averaged_params）: %s, %s"%(sample_num, averaged_params))
        new_averaged_params = copy.deepcopy(averaged_params)
        # new_models2=[]#用于过渡
        for update_time in range(self.topology.model_update_times):
            new_models = []
            for useri in range(0, len(models_last_round)):
                training_num = 0
                # logging.info("neighbors" + str(self.topology.neighbors[useri]))
                for k in averaged_params.keys():
                    # logging.info("averaged_params keys:" + str(k))
                    # logging.info("averaged_params:"+str(averaged_params))
                    params = None
                    userj_i_num=[]
                    neighbor_num = 0
                    weight = 0
                    test=0
                    for userj in range(0, len(models_last_round)):
                        if (self.topology.neighbors[useri, userj] != 0):
                            a=np.random.binomial(1,self.topology.onehop_error[useri, userj],1)
                            userj_i_num.append(a[0])
                            # userj_i_num.append(1)
                            neighbor_num = neighbor_num + 1
                            if userj_i_num[userj] == 0:
                                weight += self.topology.neighbors[useri, userj]

                            test+=self.topology.neighbors[useri, userj]*a[0]
                        else:
                            userj_i_num.append(0)

                    # print(userj_i_num)
                    # print('test')
                    # print((1-weight))
                    # print(test)
                    for userj in range(0, len(models_last_round)):
                        # logging.info("topology.neighbors[%s][%s]: %s" % (useri,userj,self.topology.neighbors[useri, userj]))

                        if (self.topology.neighbors[useri, userj] != 0):

                            # print()
                            # logging.info("True")
                            # logging.info(str(useri)+' '+str(userj))
                            if update_time == 0:
                                local_sample_number, local_model_params = models_last_round[userj]
                            else:
                                local_model_params = models_last_round[userj]

                            # logging.info(zeros_ones)
                            if params is None:
                                params = userj_i_num[userj]*local_model_params[k] * (self.topology.neighbors[
                                    useri, userj]+weight/sum(userj_i_num))
                            else:
                                params +=   userj_i_num[userj]*local_model_params[k]  * (self.topology.neighbors[
                                    useri, userj]+weight/sum(userj_i_num))
                    # print('node')
                    # print(weight)
                    # print((1/(1-weight)))
                    # print((weight/sum(userj_i_num)))
                    new_averaged_params[k] = copy.deepcopy(params)
                    # logging.info("local_model_params[k]" + str(params))
                new_models.append(copy.deepcopy(new_averaged_params))
            models_last_round = copy.deepcopy(new_models)

        return models_last_round
    def consensus_aggregate_PER_replace_with_zero(self, models_last_round):
        (sample_num, averaged_params) = models_last_round[0]
        # logging.info("(sample_num, averaged_params）: %s, %s"%(sample_num, averaged_params))
        new_averaged_params = copy.deepcopy(averaged_params)
        # new_models2=[]#用于过渡
        for update_time in range(self.topology.model_update_times):
            new_models = []
            for useri in range(0, len(models_last_round)):
                training_num = 0
                for k in averaged_params.keys():
                    params = None
                    for userj in range(0, len(models_last_round)):
                        # logging.info("topology.neighbors[%s][%s]: %s" % (useri,userj,self.topology.neighbors[useri, userj]))
                        if (self.topology.neighbors[useri, userj] != 0):
                            # logging.info("True")
                            # logging.info(str(useri)+' '+str(userj))
                            if update_time==0:
                                local_sample_number, local_model_params = models_last_round[userj]
                            else:
                                local_model_params = models_last_round[userj]

                            a = torch.ones_like(local_model_params[k])
                            # logging.info(self.topology.onehop_error[useri, userj])
                            a=a*self.topology.onehop_error[useri, userj]
                            zeros_ones=torch.bernoulli(a)
                            if params is None:
                                params = torch.mul(zeros_ones, local_model_params[k]) * self.topology.neighbors[useri, userj]
                            else:
                                params += torch.mul(zeros_ones, local_model_params[k]) * self.topology.neighbors[useri, userj]

                    new_averaged_params[k] = copy.deepcopy(params)
                    # logging.info("local_model_params[k]" + str(params))
                new_models.append(copy.deepcopy(new_averaged_params))
            models_last_round=copy.deepcopy(new_models)

        return models_last_round
    def consensus_aggregate_PER_replace_with_own_model(self, models_last_round):
        (sample_num, averaged_params) = models_last_round[0]
        # logging.info("(sample_num, averaged_params）: %s, %s"%(sample_num, averaged_params))
        new_averaged_params = copy.deepcopy(averaged_params)
        # new_models2=[]#用于过渡
        for update_time in range(self.topology.model_update_times):
            new_models = []
            for useri in range(0, len(models_last_round)):
                if update_time == 0:
                    local_sample_number_i, local_model_params_i = models_last_round[useri]
                else:
                    local_model_params_i = models_last_round[useri]
                training_num = 0
                # logging.info("neighbors" + str(self.topology.neighbors[useri]))
                for k in averaged_params.keys():
                    # logging.info("averaged_params keys:" + str(k))
                    # logging.info("averaged_params:"+str(averaged_params))
                    params = None
                    for userj in range(0, len(models_last_round)):
                        # logging.info("topology.neighbors[%s][%s]: %s" % (useri,userj,self.topology.neighbors[useri, userj]))
                        if (self.topology.neighbors[useri, userj] != 0):
                            # logging.info("True")
                            # logging.info(str(useri)+' '+str(userj))
                            if update_time==0:
                                local_sample_number, local_model_params = models_last_round[userj]
                            else:
                                local_model_params = models_last_round[userj]

                            a = torch.ones_like(local_model_params[k])
                            # logging.info(self.topology.onehop_error[useri, userj])
                            a=a*self.topology.onehop_error[useri, userj]
                            zeros_ones=torch.bernoulli(a)
                            # zeros_ones=torch.zeros_like(local_model_params[k])
                            error_1=zeros_ones*(-1)+torch.ones_like(zeros_ones)
                            local_error_1=torch.mul(local_model_params_i[k],error_1)

                            if params is None:
                                params = (torch.mul(zeros_ones, local_model_params[k])+local_error_1) * self.topology.neighbors[useri, userj]
                            else:
                                params += (torch.mul(zeros_ones, local_model_params[k])+local_error_1) * self.topology.neighbors[useri, userj]
                    new_averaged_params[k] = copy.deepcopy(params)
                    # logging.info("local_model_params[k]" + str(params))
                new_models.append(copy.deepcopy(new_averaged_params))
            models_last_round=copy.deepcopy(new_models)

        return models_last_round
    def consensus_aggregate_PER_replace_with_averaged_model(self, models_last_round):
        (sample_num, averaged_params) = models_last_round[0]
        # logging.info("(sample_num, averaged_params）: %s, %s"%(sample_num, averaged_params))
        new_averaged_params = copy.deepcopy(averaged_params)
        # new_models2=[]#用于过渡
        for update_time in range(self.topology.model_update_times):
            new_models = []
            for useri in range(0, len(models_last_round)):
                if update_time == 0:
                    local_sample_number_i, local_model_params_i = models_last_round[useri]
                else:
                    local_model_params_i = models_last_round[useri]
                training_num = 0
                # logging.info("neighbors" + str(self.topology.neighbors[useri]))
                for k in averaged_params.keys():
                    # logging.info("averaged_params keys:" + str(k))
                    # logging.info("averaged_params:"+str(averaged_params))
                    params = None
                    neighbor_zero_sum = torch.zeros_like(local_model_params_i[k])
                    neighbor_one_sum= torch.zeros_like(local_model_params_i[k])
                    ones_sum = torch.zeros_like(local_model_params_i[k])
                    neighbor_weight=[]
                    neighbor_error=[]
                    for userj in range(0, len(models_last_round)):
                        # logging.info("topology.neighbors[%s][%s]: %s" % (useri,userj,self.topology.neighbors[useri, userj]))
                        if (self.topology.neighbors[useri, userj] != 0):
                            # logging.info("True")
                            # logging.info(str(useri)+' '+str(userj))
                            if update_time == 0:
                                local_sample_number, local_model_params = models_last_round[userj]
                            else:
                                local_model_params = models_last_round[userj]

                            a = torch.ones_like(local_model_params[k])
                            # logging.info(self.topology.onehop_error[useri, userj])
                            a = a * self.topology.onehop_error[useri, userj]
                            zeros_ones = torch.bernoulli(a)
                            # zeros_ones=torch.zeros_like(local_model_params[k])
                            error_1 = zeros_ones * (-1) + torch.ones_like(zeros_ones)
                            ones_sum=ones_sum+zeros_ones
                            neighbor_zero_sum=neighbor_zero_sum+torch.mul(error_1,self.topology.neighbors[useri, userj])
                            # neighbor_one_sum=neighbor_one_sum+zeros_ones*self.topology.neighbors[useri, userj]
                            neighbor_weight.append(zeros_ones*self.topology.neighbors[useri, userj])
                            neighbor_error.append(zeros_ones)
                    count=0
                    # test_sum_is_1=torch.zeros_like(local_model_params_i[k])
                    for userj in range(0, len(models_last_round)):
                        # logging.info("topology.neighbors[%s][%s]: %s" % (useri,userj,self.topology.neighbors[useri, userj]))
                        if (self.topology.neighbors[useri, userj] != 0):
                            # logging.info("True")
                            # logging.info(str(useri)+' '+str(userj))
                            if update_time==0:
                                local_sample_number, local_model_params = models_last_round[userj]
                            else:
                                local_model_params = models_last_round[userj]

                            # print(neighbor_weight[count] + torch.div(neighbor_zero_sum, ones_sum))
                            if params is None:
                                params = (torch.mul(neighbor_weight[count]+torch.mul(neighbor_error[count],torch.div(neighbor_zero_sum,ones_sum)), local_model_params[k]))
                                # params = (torch.mul(zeros_ones, local_model_params[k]) ) * \
                                #          self.topology.neighbors[useri, userj]
                            else:
                                params += (torch.mul(neighbor_weight[count]+torch.mul(neighbor_error[count],torch.div(neighbor_zero_sum,ones_sum)), local_model_params[k]))
                            # test_sum_is_1 = test_sum_is_1 + torch.mul(neighbor_error[count],torch.div(neighbor_zero_sum,ones_sum))
                            count+=1
                    # print('neighbor_zero_sum + neighbor_one_sum')
                    # print(neighbor_zero_sum)
                    # print('test_sum_is_1')
                    # print(test_sum_is_1)
                    # print('ones_sum')
                    # print(ones_sum)

                                # logging.info("userj"+str(userj)+"str(k) "+str(k))
                            # params = local_model_params[k] * 0.125 if params is None \
                            #         else params + local_model_params[k] *0.125
                            # if k=='linear_2.bias':
                            #     logging.info(str(local_model_params[k][0:5]))

                    new_averaged_params[k] = copy.deepcopy(params)
                    # logging.info("local_model_params[k]" + str(params))
                new_models.append(copy.deepcopy(new_averaged_params))
            models_last_round=copy.deepcopy(new_models)

        return models_last_round

    def _local_test_on_one_client(self, round_idx, client_id):
        logging.info("################local_test_on_%s_client : %s th user"%(client_id, round_idx))

        train_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }
        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }
        client = self.client_list[client_id]
        # logging.info("client：%s"%client.client_idx)

        client.update_local_dataset(client_id, self.train_data_local_dict[client_id],
                                    self.test_data_local_dict[client_id],
                                    self.train_data_local_num_dict[client_id])
        # train data
        train_local_metrics = client.local_test(False)
        train_metrics['num_samples'].append(copy.deepcopy(train_local_metrics['test_total']))
        train_metrics['num_correct'].append(copy.deepcopy(train_local_metrics['test_correct']))
        train_metrics['losses'].append(copy.deepcopy(train_local_metrics['test_loss']))

        # test data
        test_local_metrics = client.local_test(True)
        test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
        test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
        test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))

        """
        Note: CI environment is CPU-based computing. 
        The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
        """


        # test on training dataset
        train_acc = sum(train_metrics['num_correct']) / sum(train_metrics['num_samples'])
        train_loss = sum(train_metrics['losses']) / sum(train_metrics['num_samples'])

        # test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])

        stats = {'training_acc': train_acc, 'training_loss': train_loss}
        # wandb.log({"Train/Acc": train_acc, "round": round_idx})
        # wandb.log({"Train/Loss": train_loss, "round": round_idx})
        logging.info(stats)

        stats = {'test_acc': test_acc, 'test_loss': test_loss}
        # wandb.log({"Test/Acc": test_acc, "round": round_idx})
        # wandb.log({"Test/Loss": test_loss, "round": round_idx})
        logging.info(stats)

    def _local_test_on_all_clients(self, round_idx,client_indexes,models_of_last_round):

        logging.info("################local_test_on_all_clients : {}".format(round_idx))


        # client = self.client_list[0]
        # logging.info("client：%s"%client.client_idx)
        for idx in range(self.args.d2d_user_num):
            logging.info("local_test_on_all_clients: client:{} th user".format(idx))
            train_metrics = {
                'num_samples': [],
                'num_correct': [],
                'losses': []
            }

            test_metrics = {
                'num_samples': [],
                'num_correct': [],
                'losses': []
            }

            self.model_trainer.set_model_params(models_of_last_round[idx])
            for client_idx in range(self.args.d2d_user_num):
                client = self.client_list[idx]

                idxtest = client_indexes[client_idx]
                # logging.info("local_test_on_all_clients: test data of:{}".format(idxtest))
                """
                Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
                the training client number is larger than the testing client number
                """
                if self.test_data_local_dict[idx] is None:
                    continue
                client.update_local_dataset(idx, self.train_data_local_dict[idxtest],
                                            self.test_data_local_dict[idxtest],
                                            self.train_data_local_num_dict[idxtest])
                # train data

                train_local_metrics = client.local_test(False)
                train_metrics['num_samples'].append(copy.deepcopy(train_local_metrics['test_total']))
                train_metrics['num_correct'].append(copy.deepcopy(train_local_metrics['test_correct']))
                train_metrics['losses'].append(copy.deepcopy(train_local_metrics['test_loss']))

                # test data
                test_local_metrics = client.local_test(True)
                test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
                test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
                test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))

                """
                Note: CI environment is CPU-based computing. 
                The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
                """
                if self.args.ci == 1:
                    break

            # test on training dataset
            train_acc = sum(train_metrics['num_correct']) / sum(train_metrics['num_samples'])
            train_loss = sum(train_metrics['losses']) / sum(train_metrics['num_samples'])

            # test on test dataset
            test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
            test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])

            stats = {'training_acc': train_acc, 'training_loss': train_loss}
            # wandb.log({"Train/Acc": train_acc, "round": round_idx})
            # wandb.log({"Train/Loss": train_loss, "round": round_idx})
            logging.info(stats)

            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            # wandb.log({"Test/Acc": test_acc, "round": round_idx})
            # wandb.log({"Test/Loss": test_loss, "round": round_idx})
            logging.info(stats)


    def _local_test_on_validation_set(self, round_idx):

        logging.info("################local_test_on_validation_set : {}".format(round_idx))

        if self.val_global is None:
            self._generate_validation_set()

        client = self.client_list[0]
        client.update_local_dataset(0, None, self.val_global, None)
        # test data
        test_metrics = client.local_test(True)

        if self.args.dataset == "stackoverflow_nwp":
            test_acc = test_metrics['test_correct'] / test_metrics['test_total']
            test_loss = test_metrics['test_loss'] / test_metrics['test_total']
            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            # wandb.log({"Test/Acc": test_acc, "round": round_idx})
            # wandb.log({"Test/Loss": test_loss, "round": round_idx})
        elif self.args.dataset == "stackoverflow_lr":
            test_acc = test_metrics['test_correct'] / test_metrics['test_total']
            test_pre = test_metrics['test_precision'] / test_metrics['test_total']
            test_rec = test_metrics['test_recall'] / test_metrics['test_total']
            test_loss = test_metrics['test_loss'] / test_metrics['test_total']
            stats = {'test_acc': test_acc, 'test_pre': test_pre, 'test_rec': test_rec, 'test_loss': test_loss}
            # wandb.log({"Test/Acc": test_acc, "round": round_idx})
            # wandb.log({"Test/Pre": test_pre, "round": round_idx})
            # wandb.log({"Test/Rec": test_rec, "round": round_idx})
            # wandb.log({"Test/Loss": test_loss, "round": round_idx})
        else:
            raise Exception("Unknown format to log metrics for dataset {}!"%self.args.dataset)

        logging.info(stats)
