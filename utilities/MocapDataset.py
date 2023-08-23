import torch as T
import numpy as np
import torchaudio as Ta
#from itertools import pairwise

class MocapDataset(T.utils.data.Dataset):
    def __init__(self, head_dir: list, aud_dir: list, landM_dir: list, emo_vec: list):
        #self.df = df
        self.head = head_dir
        self.aud = aud_dir
        self.landM = landM_dir
        self.emo = emo_vec
        self.time = 0

    def __len__(self):
        return len(self.head)

    def __getitem__(self, idx):
        inp = self.aud[idx]
        out = self.landM[idx]
        X = self.readAudInp(inp)
        y, strt = self.readLandMark(out)
        return X, y, inp, strt

    def readAudInp(self, directory:list):
        waveform, sample_rate = Ta.load(directory)
        transform = Ta.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=28,
                melkwargs={"n_fft":1024, "hop_length":533, "n_mels":28, "center":False}
            )
        mfcc = transform(waveform)
        X = T.Tensor(mfcc)
        X = T.squeeze(X.permute([0,2,1]))
        X = X[:-1,:]
        self.time = X.shape[0]
        return X

    def readLandMark(self, directory:list):
        files = open(directory,'r')
        readlist = files.readlines()[2:]
        vals = []
        start = []
        for n, line in enumerate(readlist):
            if n==0:
                line = line.split(' ')
                line = list(map(float,line[2:]))  #handle nan values and \n in last entry
                del line[1::3]
                line = np.nan_to_num(np.array(line[:-4]))
                line0 = line
                start = line
                continue
            if n%4==0:
              line = line.split(' ')
              line = list(map(float,line[2:]))  #handle nan values and \n in last entry
              del line[1::3]
              line = np.nan_to_num(np.array(line[:-4]))
              res = list(line-line0)
              vals.append(res)
              line0=line
        #res = [y - x for x,y in pairwise(vals)]
        #print(len(vals),self.time)
        if len(vals)>self.time:
          diff = len(vals)-self.time                           #just commented
          vals = vals[int(diff/2):-(diff-int(diff/2))]     #just commented       
        y = T.Tensor(vals)
        return y, start
    
class MocapDatasetLand(T.utils.data.Dataset):
#This data set is for landmark prediction
    def __init__(self, head_dir: list, aud_dir: list, landM_dir: list, emo_vec: list):
        #self.df = df
        self.head = head_dir
        self.aud = aud_dir
        self.landM = landM_dir
        self.emo = emo_vec
        self.time = 0
        self._indices = None
        self.transform = None

    def __len__(self):
        return len(self.head)
    
    def len(self):
        return len(self.head)
    
    def __getitem__(self, idx):
        inp = self.aud[idx]
        out = self.landM[idx]
        X = self.readAudInp(inp)
        y = self.readLandMark(out)
        #data = Data(x=X, edge_index = edge_index, y=y)
        return X, y, inp

    def readAudInp(self, directory:list):
        waveform, sample_rate = Ta.load(directory)
        transform = Ta.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=28,
                melkwargs={"n_fft":1024, "hop_length":533, "n_mels":28, "center":False}
            )
        mfcc = transform(waveform)
        X = T.Tensor(mfcc)
        X = T.squeeze(X.permute([0,2,1]))
        X = X[:-1,:]
        self.time = X.shape[0]
        return X

    def readLandMark(self, directory:list):
        files = open(directory,'r')
        readlist = files.readlines()[2:]
        vals = []
        for n, line in enumerate(readlist):
            if n%4==0:
                line = line.split(' ')
                line = list(map(float,line[2:]))  #handle nan values and \n in last entry
                del line[1::3]
                line = np.nan_to_num(np.array(line[:-4]))
                vals.append(line)
        #res = [y - x for x,y in pairwise(vals)]
        #print(len(vals),self.time)
        if len(vals)>self.time:
            diff = len(vals)-self.time                           #just commented
            vals = vals[int(diff/2):-(diff-int(diff/2))]     #just commented       
        y = T.Tensor(vals)
        return y
    
class MocapDataset_head(T.utils.data.Dataset):
    def __init__(self, input_dir: list, output_dir: list):
        #self.df = df
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.time = 0

    def __len__(self):
        return len(self.input_dir)

    def __getitem__(self, idx):
        inp = self.input_dir[idx]
        out = self.output_dir[idx]
        X = self.readAudInp(inp)
        y = self.readHeadPose(out)
        return X, y, inp

    def readAudInp(self, directory:list):
        waveform, sample_rate = Ta.load(directory)
        transform = Ta.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=28,
                melkwargs={"n_fft":1024, "hop_length":533, "n_mels":28, "center":False}
            )
        mfcc = transform(waveform)
        X = T.Tensor(mfcc)
        X = T.squeeze(X.permute([0,2,1]))
        self.time = X.shape[0]
        #print(X.shape, 'X')
        return X

    def readHeadPose(self, directory:list):
        y = []
        files = open(directory,'r')
        readlist = files.readlines()[2:]
        vals = []
        for n, line in enumerate(readlist):
            if n%4==0:
              line = line.split(' ')
              line = list(map(float,line[2:5]))
              vals.append(line)
        #print(len(readlist))
        #readlist = readlist
        if len(vals)>self.time:
          diff = len(vals)-self.time                           #just commented
          #print(len(readlist),self.time, int(diff/2))
          vals = vals[int(diff/2):-(diff-int(diff/2))]     #just commented
        
        y = T.Tensor(vals)
        #print(y.shape, 'y')
        return y
