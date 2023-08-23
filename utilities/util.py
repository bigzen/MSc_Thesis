
import csv
import os
import random
import numpy as np

def TrTstSplit(dir_list):
    tr_splt = len(dir_list)*4/5
    random.shuffle(dir_list)
    train = dir_list[0:int(tr_splt)]
    test = dir_list[int(tr_splt):]
    print(train)
    return train, test
    
def GetInputOutputSplit(dir_list):
    head = []
    aud = []
    landM = []
    emo = []
    for dirs in dir_list:
        with open(dirs,'r') as read_obj:
            csv_reader = csv.reader(read_obj)
            readlist = np.array(list(csv_reader))
            head.extend(list(readlist[:,0]))
            aud.extend(list(readlist[:,1]))
            landM.extend(list(readlist[:,2]))
            emo.extend(list(readlist[:,3])) 
    return head, aud, landM, emo

def FaceGraph():
    dire = '../backchannel_gesture/utilities/land.csv'
    graph = []
    with open(dire,'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        readlist = list(csv_reader)[1:]
        for line in readlist:
            graph.append(line[1:])
        grp = np.nan_to_num(np.array(graph))
        edge = np.argwhere(grp=='1')
    edge = np.transpose(edge)
    return edge

