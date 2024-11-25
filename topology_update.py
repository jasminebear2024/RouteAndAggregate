import numpy as np
import logging
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, ConnectionPatch
import math
import consensus_matric
import copy
class TopologyManager:
    location = None
    neighbors = None
    channel_gain = None
    transmitter_power = None
    dataset = None
    model = None
    neighbor_distance = None
    def __init__(self, user_num,x_range,y_range,communication_radius,subchannel_num,pmin,pmax,topology_type,model_update_times):
        self.user_num = user_num
        self.x_range = x_range
        self.y_range = y_range
        self.communication_radius = communication_radius
        self.subchannel_num = subchannel_num
        self.model_update_times=model_update_times
        self.pmin = pmin
        self.pmax = pmax
        if topology_type==0 or topology_type==1 or topology_type==2:
            self.routing_update_topology()
        elif topology_type==3 or topology_type==4 or topology_type==5:
            self.consensus_aggregate_update_topology()
        elif topology_type==6 or topology_type==7 or topology_type==8:
            self.centralized_update_topology()




#location and channel update
    def topology_update(self):
        location = np.zeros((self.user_num, 2))
        transmitter_power = np.zeros((self.user_num, self.subchannel_num))
        for user in range(self.user_num):
            location[user, 0] = np.random.uniform(0, self.x_range)
            location[user, 1] = np.random.uniform(0, self.y_range)
            transmitter_power[user] = np.random.uniform(self.pmin, self.pmax, size=(1, self.subchannel_num))
        self.location = location
        self.transmitter_power = transmitter_power
        self.neighbor_update()

    #neighbors
    @staticmethod
    def distance(x1, y1, x2, y2):
        sq = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
        return math.sqrt(sq)

    def neighbor_update(self):
        self.neighbors = np.zeros((self.user_num, self.user_num))
        self.neighbor_distance = np.zeros((self.user_num, self.user_num))
        for useri in range(self.user_num):
            for userj in range(self.user_num):
                if useri == userj:
                    self.neighbors[useri, userj] = 1
                elif self.distance(self.location[useri, 0], self.location[useri, 1], self.location[userj, 0],
                              self.location[userj, 1]) < self.communication_radius:
                    self.neighbors[useri, userj] = 1
                else:
                    self.neighbors[useri, userj] = 0
                self.neighbor_distance[useri, userj] = 0

    def channel_gain_update(self):
        self.channel_gain = np.zeros((self.user_num, self.user_num, self.subchannel_num))
        for useri in range(self.user_num):
            for userj in range(self.user_num):
                for subchannel in range(self.subchannel_num):
                    if useri == userj:
                        self.channel_gain[useri, userj, subchannel] = 0
                    else:
                        self.channel_gain[useri, userj, subchannel] = self.transmitter_power[useri, subchannel] / (
                                    self.distance(self.location[useri, 0], self.location[useri, 1], self.location[userj, 0],
                                             self.location[userj, 1]) * self.distance(self.location[useri, 0],
                                                                                self.location[useri, 1],
                                                                                self.location[userj, 0],
                                                                                self.location[userj, 1]))
    def delete_user(self):
        a = 0

    def add_user(self):
        n = 0

    def model_update_once(self):
        new_model = np.zeros((self.user_num, self.user_num))
        for useri in range(self.user_num):
            sum_neighbor = 0
            for index in range(self.user_num):
                if self.neighbors[useri, index] == 1:
                    new_model[useri] = new_model[useri] + self.model[index]
                    sum_neighbor = sum_neighbor + 1
            new_model[useri] = new_model[useri] / sum_neighbor
        self.model = new_model


    def consensus_aggregate_update_topology(self):

        New_weight, onehop_error = consensus_matric.consensus_error2(self.user_num)
        self.neighbors = New_weight
        self.onehop_error = onehop_error
    def routing_update_topology(self):
        onehop_error,path = consensus_matric.routing( self.user_num)
        self.onehop_error=onehop_error
        self.path = path
    def centralized_update_topology(self):
        onehop_error,path,source_node = consensus_matric.centralized(self.user_num)
        self.path=path
        self.onehop_error = onehop_error
        self.source_node=source_node