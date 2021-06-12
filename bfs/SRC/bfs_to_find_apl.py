import time
import csv
from random import sample
import networkx as nx
import statistics
from itertools import chain
from collections import defaultdict
from collections import deque
PATHS_FROM_ECLIPSE_TO_PYTHON='[["client_src_fullpath", "remote_src_fullpath"]]'
import random
import numpy as np
import scipy as sp

# net=nx.read_edgelist("../DATA/ca-HepTh.txt",nodetype=int)
# net_direct=net.to_directed()
# nx.write_adjlist(net_direct, "../DATA/youtube.adjlist")

### for reading undirected graphs
print("reading data")
with open("../DATA/dblp.adjlist") as f:
    lines = f.read().strip().split("\n")

#data=[[a, b] for [a, b, c] in data]
print("reading", f)

d = defaultdict(list)

for line in lines:
    ls = line.split(" ")
    d[ls[0]] = ls[1:]


## for reading directed graphs
#print("reading data")
# with open('../DATA/ca-HepTh.txt', 'r') as inputfile:
#     csv_input= csv.reader(inputfile, delimiter="\t")
#     data=[row for row in csv_input]
# #
# #data=[[a, b] for [a, b, c] in data]
# #
# d={}
# for i in data:
# #    print (i)
#     if i[0] not in d:
#
#         d[i[0]]= [i[1]]
#     else:
#         d[i[0]].append(i[1])

### preprocessing livemocha data and others
# with open('../DATA/facebook-wosn-links/facebook-wosn-links.edges', 'r') as inputfile:
#     csv_input= csv.reader(inputfile, delimiter=",")
#     data=[row for row in csv_input]
#
# l_d=[[a, b] for [a, b, c, d] in data]
#
# filename = '../DATA/facebook-wosn.txt'
#
# with open(filename, 'w') as f:
#      for sublist in l_d:
#           line = "{} {}\n".format(sublist[0], sublist[1])
#           f.write(line)



def bfs_shortest_path(graph, start, goal):
    # keep track of explored nodes
    #print(graph)
    explored = []
    # keep track of all the paths to be checked
    #queue = [[start]]
    queue=deque([start])
    #depths={start:0}
    m=0
    maxdepth=500

    # return path if start is goal
    if start == goal:
        return "That was easy! Start = goal"

    # keeps looping until all possible paths have been checked
    while queue:
        # pop the first path from the queue
        #path = queue.pop(0)
        path = queue.popleft()
        #print(path)
        #print(path)
        #print(start)
        # get the last node from the path

        node = path[-1]
#        m=m+1
#         if m>= maxdepth:
# #            return None
# #            print("hello")
#             continue
        #print("starting with if condition")

        if node not in explored:
            neighbours = graph[node]

            # go through all neighbour nodes, construct a new path and
            # push it into the queue
            for neighbour in neighbours:
                # if neighbour in depths:
                #     continue
                new_path = list(path)
                #print(len(new_path))
                len_new_path=len(new_path)
                if len(new_path) >=11:
                    #print("hello")
                    break
                #else:
                    #print("hello")

                new_path.append(neighbour)
                queue.append(new_path)
                # depths[neighbour] =depths[node] +1
                # len_depth=len(depths)


                    # return path if neighbour is goal
                if neighbour == goal:
                    return new_path

            # if len(new_path) >=11:
            #     return None

            # mark node as explored
            explored.append(node)

    # in case there's no path between the 2 nodes
    return "So sorry, but a connecting path doesn't exist :("

t1 = time.time()

bfs_result=[]
nodes=list(d.keys())

threshold=100

new_nodes = nodes
k=0
for i in range(len(nodes)):
    #print(i)
    k=k+1
    print(k)
    if k<=threshold:
        the_sample = random.sample(nodes, 2)
        print(the_sample)
        g=str(the_sample[0])
        h=str(the_sample[1])
        # for b in the_sample:
        #     new_nodes.remove(b)

        try:
            bfs = len(bfs_shortest_path(d, g, h))
            #print(bfs)
            # print('bfs is',bfs)
            bfs_result.append(bfs)

        except Exception as e:
            print("exception received")
            continue
    else:
        break



t2 = time.time()
print('time taken: ',t2 - t1, 's')
#print(bfs_result)
print(len(bfs_result))
def Average(lst):
    return sum(lst) / len(lst)
#print(Average(bfs_result))

def CountFrequency(my_list):
    freq = {}
    for item in my_list:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1

    for key, value in freq.items():
        print("% d : % d" % (key, value))

bfs_result_minus=[x-1 for x in bfs_result]
print(Average(bfs_result_minus))
print(CountFrequency(bfs_result_minus))

bfs_result_processed=[x for x in bfs_result_minus if x!=47 and x!=26]

print(Average(bfs_result_processed))
print(CountFrequency(bfs_result_processed))
print('diameter is', max(bfs_result_processed))
print('median diameter is' , np.median(bfs_result_processed))


###############################
### eccentricity

# nodes=list(d.keys())
# m_nodes = sample(nodes, 50)
# n_nodes=sample(nodes, 10)
# bfs_ecc=[]
# ecc_list=[]
#
# for i in m_nodes:
#     for j in n_nodes:
#         i=str(i)
#         j=str(j)
#         print([i,j])
#
#         try:
#             bfs=len(bfs_shortest_path(d, i, j))
#             bfs_ecc.append(bfs)
#             bfs_ecc_edited=[x for x in bfs_ecc if x<26]
#         except Exception as e:
#             print("exception received")
#             continue
#     max_ecc=max(bfs_ecc_edited, default =99)
#     ecc_list.append(max_ecc)
#
# print('length of eccentricity list is', len(bfs_ecc_edited))
# ecc_list=[x for x in ecc_list if x!=99]
# def Average(lst):
#     return sum(lst) / len(lst)
#
# eccentr=Average(ecc_list)
# print('eccentricity is',eccentr)
#




#######
##### assortavity coefficient
#
# nodes=list(d.keys())
# #sample_nodes=sample(nodes, 1000)
# degree_list=[]
#
# threshold=50000
# k=0
# for i in nodes:
#     print(i)
#     i=str(i)
#     k=k+1
#     print(k)
#     if k<=threshold:
#         values=d[i]
#         sample_value=random.choice(values)
#         v=str(sample_value)
#         value_outdegree=len(d[v]) -1
#         node_outdegree=len(values) -1
#         degree_list.append(node_outdegree)
#         degree_list.append(value_outdegree)
#     else:
#         break
#
# assort_degree_list = [degree_list[i:i+2] for i in range(0, len(degree_list), 2)]
# print('length of list is',len(assort_degree_list))
# import pandas as pd
# import scipy.stats
# assort_data=pd.DataFrame(assort_degree_list)
# x=assort_data.iloc[:, 0]
# y=assort_data.iloc[:, 1]
# assortavity_coef=scipy.stats.pearsonr(x, y)[0]
# print('assortavity coefficient is', assortavity_coef)




# NewA={
#       '1':['2','3'],
#       '2': ['4','5','6'],
#       '3': ['7','8'],
#       '4': ['9'],
#       '5': [],
#       '6': [],
#       '7': [],
#       '8': [],
#       '9': ['10'],
#       '10':[]
#       }


#################
#################
### comparing the sample data file with networkx

# fb_nt = nx.read_edgelist("../DATA/email-Eu-core.txt", nodetype=int)
# fb_direct = fb_nt.to_directed()
# #apl=nx.average_shortest_path_length(fb_direct)
#
# print("starting on apl calculation")
# apl = dict(nx.shortest_path_length(fb_direct))  # source,target not specified
# print("apl calculation ended")
# values=list(apl.values())
# path_lengths = (x.values() for x in apl.values())
# dist=statistics.mean(chain.from_iterable(path_lengths))
# print(dist)
# #




# apl = 2.5238095238095237
#
# # startnodes = [x for x in fb_direct.nodes() if fb_direct.out_degree(x)==1 and fb_direct.in_degree(x)==0]
# # endnode = [x for x in fb_direct.nodes() if fb_direct.out_degree(x)==0 and fb_direct.in_degree(x)==1][0]
#
# startnode= [node for node in tw_nt.nodes if tw_nt.in_degree(node) == 0]
# endnode= [node for node in tw_nt.nodes if tw_nt.out_degree(node) == 0]
# #print(startnode, endnode)
#
# diameter_result=[]
# for i in startnode:
#     for j in endnode:
# #        if i < j:
#         i=str(i)
#         j=str(j)
#         try:
#             bfs=len(bfs_shortest_path(d, i, j))
#             diameter_result.append(bfs)
#         except Exception as e:
#             print("the connection doesnt exist and is empty")
#         else:
#             print("not in values")
#



    #    for j in values_flatten:
#    for k in range(0, len(nodes)):
#        for j in values[k]:

        #if i < j:
# if j in d.values():

# bfs_result=[]
# for i in nodes:
#     for j in values[i]:
#         if i< j:
#             print([i, j,])
#             i=str(i)
#             j=str(j)
#             bfs=len(bfs_shortest_path(d, i, j))
#             bfs_result.append(bfs)
#
#
# bfs_result=[]
# for i in nodes:
#     #i=int(i)
#     #print(type(i))
#     for j in d[i]:
#         print([i, j])
#         bfs = len(bfs_shortest_path(d, i, j))
#         bfs_result.append(bfs)
#

#######################################################################
##---------------------------------------------------------------------
##------------------- notes -------------------------------------------
######################################################################


#
# # bfs_shortest_path(d, v1, v2)  # returns ['G', 'C', 'A', 'B', 'D']
# A = [[1, 4, 5, 12], [-5, 8, 9, 0], [-6, 7, 11, 19]]
#
# A= [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0],[0, 1, 1 ,0 ,0 ,0, 0, 0, 0, 0],[0, 0, 0, 1, 1, 1, 0, 0, 0, 0],[0, 0, 0 ,0 ,0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],[0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0 ,0],[0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,1 ,0], [0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0]]
#
#
# NewA={
#       '1':['2','3'],
#       '2': ['4','5','6'],
#       '3': ['7','8'],
#       '4': ['9'],
#       '5': [],
#       '6': [],
#       '7': [],
#       '8': [],
#       '9': ['10'],
#       '10':[]
#       }
# bfs(NewA,'1','5')
#
#
# adj = {
#     'u': ['v', 'x'],
#     'x': ['u', 'v', 'y'],
#     'v': ['u', 'x', 'y'],
#     'y': ['w'],
#     'w': ['y', 'z'],
#     'z': ['w']
#     }
# #
# # def Breadth_first_search(adj_list):
# #     visited = {}
# #     level = {}
# #     parent = {}
# #     traversal_output = []
# #     queue = queue.Queue()
# #     for node in adj_list.keys():
# #         visited[node] = False
# #         parent[node] = None
# #         level[node] = -1
# #     s = "u"
# #     visited[s] = True
# #     level[s] = 0
# #     queue.put(s)
# #     while not queue.empty():
# #         u = queue.get()
# #         traversal_output.append(u)
# #         for v in adj_list[u]:
# #             if not visited[v]:
# #                 visited[v] = True
# #                 parent[v] = u
# #                 level[v] = level[u] + 1
# #                 queue.put(v)
# #     return traversal_output, visited, level, parent
#
#
# # def bfs(Adj, s):  # Adj: adjacency list, s: starting vertex
# #     parent = [None for v in Adj]  # O(V) (use hash if unlabeled)
# #     parent[s] = s  # O(1) root
# #     dist = [None for v in Adj]
# #     dist[s] = 0
# #     levels = [[s]]  # O(1) initialize levels
# #     while levels[-1]:  # O(?) last level contains vertices
# #         frontier = []  # O(1), make new level
# #         for u in levels[-1]:  # O(?) loop over last full level
# #             for v in Adj[u]:  # O(Adj[u]) loop over neighbors
# #                 if parent[v] is None:  # O(1) parent not yet assigned
# #                     parent[v] = u  # O(1) assign parent from levels[-1]
# #                     dist[v] = dist[u] + 1
# #                     frontier.append(v)  # O(1) amortized, add to border
# #         levels.append(frontier)  # add the new level to levels
# #     return parent, dist
# #
#
# from collections import deque
# # graph is represented by adjacency list: List[List[int]]
# # s: start vertex
# # d: destination vertex
# # based on BFS
# def find_shortest_path(graph, s, d):
#     # pred[i] stores predecessor of i in the path
#     pred = [-1] * len(graph)
#     # set is used to mark visited vertices
#     visited = set()
#     # create a queue for BFS
#     queue = deque()
#     # Mark the start vertex as visited and enqueue it
#     visited.add(s)
#     queue.appendleft(s)
#
#     while queue:
#         current_vertex = queue.pop()
#         print(current_vertex)
#         print(type(current_vertex))
#         # Get all adjacent vertices of current_vertex
#         # If a adjacent has not been visited, then mark it
#         # visited and enqueue it
#         for v in graph[current_vertex]:
#             if v not in visited:
#                 visited.add(v)
#                 queue.appendleft(v)
#                 # record the predecessor
#                 #v=int(v)
#                 pred[v] = current_vertex
#                 # reach to the destination
#                 if v == d:
#                     break
#
#     # no path to d
#     if pred[d] == -1:
#         return []
#     # retrieve the path
#     path = [d]
#     current = d
#     while pred[current] != -1:
#         current = pred[current]
#         path.append(current)
#
#     return path[::-1]
#
#
# node=0
# #there is a pair (i, j)


# d={0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
#  1: [15, 17, 18, 19, 20],
#  19: [23, 16, 30]}
# # res = [dict([str(key), value]
# #             for key, value in d.items())
# #        for dicts in d]
#
# #
# # #finds shortest path between 2 nodes of a graph using BFS
# # def bfs_shortest_path(graph, start, goal):
# #     # keep track of explored nodes
# #     #print(graph)
# #     explored = []
# #     # keep track of all the paths to be checked
# #     queue = [[start]]
# #
# #     # return path if start is goal
# #     if start == goal:
# #         return "That was easy! Start = goal"
# #
# #     # keeps looping until all possible paths have been checked
# #     while queue:
# #         # pop the first path from the queue
# #         path = queue.pop(0)
# #         #print(path)
# #         #print(start)
# #         # get the last node from the path
# #         node = path[-1]
# #         # if goal in graph[node]:
# #         #     return 1
# #
# #         #node=start
# #         if node not in explored:
# #             neighbours = graph[node]
# #
# #             # go through all neighbour nodes, construct a new path and
# #             # push it into the queue
# #             for neighbour in neighbours:
# #                 print(neighbour)
# #                 #new_path = list(path)
# #                 new_path=path
# #                 new_path.append(neighbour)
# #                 print(new_path)
# #                 queue.append(new_path)
# #                 print(queue)
# #                 # return path if neighbour is goal
# #                 if neighbour == goal:
# #                     return len(new_path) # return the length of the new path
# #
# #             # mark node as explored
# #             explored.append(node)
# #
# #     # in case there's no path between the 2 nodes
# #     return "So sorry, but a connecting path doesn't exist :(" # chnage the statement
# #
# # # v1 = np.random.choice(G.nodes())
# # # v2 = np.random.choice(G.nodes())
# # # print(v1, v2)
#
#
# values=list(d.values())
# values_flatten=sum(values, [])

#print(nodes)
# values=d.values()
# N=len(nodes)
# explored = [False] * N


# maxdepth=5
# m=0
# def bfs_shortest_path(graph, start, goal):
#     # keep track of explored nodes
#     #print(graph)
#     explored = []
#     # keep track of all the paths to be checked
#     queue = [[start]]
#
#     # return path if start is goal
#     if start == goal:
#         return "That was easy! Start = goal"
#
#     # keeps looping until all possible paths have been checked
#     maxdepth=1000
#     m=0
#     while queue:
#         #print(queue)
#         # pop the first path from the queue
# #        m=m+1
# #        if m>maxdepth:
# #            break
# #        print('m is',m)
#         path = queue.pop(0)
#
#         # get the last node from the path
#         node = path[-1]
#
# #        print('node is', node)
# #        m=m+1
# #        if m<=maxdepth:
# #        if node>=maxdepth:
# #            print('m is', m)
# #            break
#         if node not in explored:
#   #          m=m+1
#   #          if m<=maxdepth:
#             #print(graph)
#             #print(node)
#             neighbours = graph[node]
#
#             # go through all neighbour nodes, construct a new path and
#             # push it into the queue
#             for neighbour in neighbours:
#                 #print('neighbour is',neighbour)
#                 new_path = list(path)
#                 #new_path=path
#                 new_path.append(neighbour)
#                 queue.append(new_path)
#                 # return path if neighbour is goal
#                 if neighbour == goal:
#                     return new_path
#
#             # mark node as explored
#             explored.append(node)
#
#     # in case there's no path between the 2 nodes
#     return "So sorry, but a connecting path doesn't exist :("

# threshold=500
# k=0
# for i in nodes:
#     #print(i)
#     for j in nodes:
#
# #for i, j in nodes:
#
# #        if i < j:
#         k=k+1
#         if k <= threshold:
#             #print('k- threshold is',k)
#             i=str(i)
#             j=str(j)
#             print([i,j])
#
#             try:
#                 bfs=len(bfs_shortest_path(d, i, j))
#                 #print('bfs is',bfs)
#                 bfs_result.append(bfs)
#                 #print("appended")
#             #if bfs==48:
#             #        node_pair=(i,j)
#              #       odd_numbers.append(node_pair)
#             except Exception as e:
#                 print("exception received")
#                 continue
#         else:
#               break
#                 #print("the connection doesnt exist and is empty")
#             #else:
#             #    print("not in values")


#new_x = x[:]


# def bfs_maxdepth(graph, start, maxdepth=10):
#     queue = deque([start])
#     depths = {start: 0}
#     while queue:
#         vertex = queue.popleft()
#         if depths[vertex] == maxdepth:
#             break
#         for neighbour in graph[vertex]:
#             if neighbour in depths:
#                 continue
#             queue.append(neighbour)
#             depths[neighbour] = depths[vertex] + 1
#     return len(depths)
#
#print(len(bfs_shortest_path(d, '0', '19')))


