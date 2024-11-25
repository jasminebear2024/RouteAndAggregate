import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import copy
import logging
# import seaborn as sns
# import pandas as pd
import math

from scipy.stats import rice

scale = 5350*1.2
packetlength=24200
density_global = 0.9

def consensus_error2(nodes_num):
    logging.info("Consensus matrix")
    # consensus_time = 2000
    pi = 3.14
    # nodes_num = 13
    density = density_global
    seed1 = 89684
    logging.info("density =  {}".format(density))
    logging.info("packetlength =  {}".format(packetlength))
    # # Use seed when creating the graph for reproducibility
    G = nx.random_geometric_graph(nodes_num, density, seed=seed1)  # ,,seed=89680
    # # position is stored as node attribute data for random_geometric_graph
    while not nx.is_connected(G):
        density += 0.05
        G = nx.random_geometric_graph(nodes_num, density, seed=seed1)
    pos = nx.get_node_attributes(G, "pos")

    one_hop_error = np.ones((nodes_num, nodes_num))
    for node1 in range(nodes_num):
        for node2 in range(nodes_num):
            if G.has_edge(node1, node2):
                distance = [pos[node1][0] - pos[node2][0], pos[node1][1] - pos[node2][1]]
                d = math.hypot(distance[0], distance[1])

                d = d * scale
                pi = 3.14
                lamda = 3 / 25
                np.random.seed(4)
                a = (lamda / (4 * pi * d))
                snr = (a) ** 2 * 0.1 / 1e-13

                # print(snr)
                snr2 = math.sqrt(snr)
                error = 0.5 * math.erfc(snr2)
                if error < 0.0000001:
                    error = 0.000001
                if error>0.009:
                    error=0.005
                # error=0.0003
                one_hop_error[node1][node2] = (1 - error) ** (packetlength)
                Weight = -math.log2(one_hop_error[node1][node2])
                # print(error)
                G[node1][node2].update({"weight": 1})

    #计算最优consensus系数(边）
    L = nx.laplacian_matrix(G).todense()
    # print(L)
    c = np.linalg.eig(L)
    sortedc = np.sort(c[0])
    alpha = 2 / (sortedc[1] + sortedc[nodes_num - 1])

    # consensus
    edges = nx.edges(G)
    New_weight = np.zeros((nodes_num, nodes_num))
    for i in range(nodes_num):
        for j in range(nodes_num):
            if (i, j) in edges:
                # print(f"edge:{i},{j}")
                New_weight[i, j] = alpha
            if i == j:
                New_weight[i, j] = 1 - alpha * G.degree[i]
    logging.info(f"one_hop_error: {one_hop_error}")
    logging.info(f"Consensus matrix: {New_weight}")
    return New_weight,one_hop_error


def routing(nodes_num):
    # consensus_time = 2000
    pi = 3.14
    # nodes_num = 13
    density = density_global
    logging.info("density =  {}".format(density))
    logging.info("packetlength =  {}".format(packetlength))
    seed1 = 89684
    # # Use seed when creating the graph for reproducibility
    G = nx.random_geometric_graph(nodes_num, density, seed=seed1)  # ,,seed=89680
    # # position is stored as node attribute data for random_geometric_graph
    while not nx.is_connected(G):
        density += 0.05
        G = nx.random_geometric_graph(nodes_num, density, seed=seed1)
    pos = nx.get_node_attributes(G, "pos")

    one_hop_error = np.ones((nodes_num, nodes_num))
    for node1 in range(nodes_num):
        for node2 in range(nodes_num):
            if G.has_edge(node1, node2):
                distance = [pos[node1][0] - pos[node2][0], pos[node1][1] - pos[node2][1]]
                d = math.hypot(distance[0], distance[1])
                # scale = 5350
                d = d * scale
                pi = 3.14
                lamda = 3 / 25
                np.random.seed(4)
                a = (lamda / (4 * pi * d))
                snr = (a) ** 2 * 0.1 / 1e-13

                # print(snr)
                snr2 = math.sqrt(snr)
                error = 0.5 * math.erfc(snr2)
                if error < 0.0000001:
                    error = 0.000001
                if error>0.009:
                    error=0.005
                # error=0.0003
                one_hop_error[node1][node2] = (1 - error) ** (packetlength)
                Weight = -math.log2(one_hop_error[node1][node2])
                # print(error)
                G[node1][node2].update({"weight": Weight})

    pathlengths=dict(nx.all_pairs_dijkstra_path_length(G, cutoff=None, weight='weight'))
    path = dict(nx.all_pairs_dijkstra_path(G))
    logging.info(f"routing one_hop_error: {one_hop_error}")
    return one_hop_error,path


def centralized(nodes_num):
    # consensus_time = 2000
    pi = 3.14
    # nodes_num = 13
    density = density_global
    logging.info("density =  {}".format(density))

    seed1 = 89684
    # # Use seed when creating the graph for reproducibility
    G = nx.random_geometric_graph(nodes_num, density, seed=seed1)  # ,,seed=89680
    # # position is stored as node attribute data for random_geometric_graph
    while not nx.is_connected(G):
        density += 0.05
        G = nx.random_geometric_graph(nodes_num, density, seed=seed1)
    pos = nx.get_node_attributes(G, "pos")

    one_hop_error = np.ones((nodes_num, nodes_num))
    for node1 in range(nodes_num):
        for node2 in range(nodes_num):
            if G.has_edge(node1, node2):
                distance = [pos[node1][0] - pos[node2][0], pos[node1][1] - pos[node2][1]]
                d = math.hypot(distance[0], distance[1])
                # scale = 5350*2
                d = d * scale
                pi = 3.14
                lamda = 3 / 25
                np.random.seed(4)
                a = (lamda / (4 * pi * d))
                snr = (a) ** 2 * 0.1 / 1e-13

                # print(snr)
                snr2 = math.sqrt(snr)
                error = 0.5 * math.erfc(snr2)
                if error < 0.0000001:
                    error = 0.000001
                if error>0.009:
                    error=0.005
                # error=0.0003
                one_hop_error[node1][node2] = (1 - error) ** (packetlength)
                Weight = -math.log2(one_hop_error[node1][node2])
                # print(error)
                G[node1][node2].update({"weight": Weight})
    source_node=6
    length, path = nx.single_source_dijkstra(G, source_node, weight='weight')
    logging.info(one_hop_error)

    return one_hop_error, path,source_node
