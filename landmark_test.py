import os
#import glob
import imageio
import torch as T
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
#from tqdm import tqdm
from utilities.MocapDataset import MocapDataset
from models.models import RNN
from utilities.util import TrTstSplit, GetInputOutputSplit


def main():
    base_dir = '../IEMOCAP_full_release/'
    os.chdir(base_dir)
    #dirs = glob.glob('*.csv')
    dir = ['Session3.csv']
    label = ['CH1', 'CH2', 'CH3', 'FH1', 'FH2', 'FH3', 'LC1', 'LC2', 'LC3', 'LC4', 'LC5',\
         'LC6', 'LC7', 'LC8', 'RC1', 'RC2', 'RC3', 'RC4', 'RC5', 'RC6', 'RC7', 'RC8',\
         'LLID', 'RLID', 'MH', 'MNOSE', 'LNSTRL', 'TNOSE', 'RNSTRL', 'LBM0',\
         'LBM1', 'LBM2', 'LBM3', 'RBM0', 'RBM1', 'RBM2', 'RBM3', 'LBRO1',\
         'LBRO2', 'LBRO3', 'LBRO4', 'RBRO1', 'RBRO2', 'RBRO3', 'RBRO4',\
         'Mou1', 'Mou2', 'Mou3', 'Mou4', 'Mou5', 'Mou6', 'Mou7', 'Mou8']

    criterion = T.nn.MSELoss()
    device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
    rnn = RNN(28,256,1,106).to(device)
    rnn.load_state_dict(T.load('landmark_checkpoint/rnn_bestmodel.pth')['model_state_dict'])
    rnn.eval()
    #test=['Session5.csv']
    head, aud, landM, emo = GetInputOutputSplit(dir)
    #print(inp)
    dataset = MocapDataset(head, aud, landM, emo)
    dataset_size = len(head)
    dataloader = T.utils.data.DataLoader(dataset, batch_size=1, num_workers=2)
    running_loss = 0.0
    rnn.to(device)
    count = 0
    for audio, pose, name, strt in dataloader:
        audio = audio.to(device)
        pose = pose.to(device)
        #pose = np.squeeze(pose)
        outputs = rnn(T.squeeze(audio))
        loss = criterion(outputs, T.squeeze(pose))
        running_loss += loss.item()
        outputs = outputs.to('cpu').detach().numpy()
        pose = np.squeeze(pose.to('cpu').detach().numpy())
        if count<5:
            name = name[0].split('/')
            name = name[-1].split('.')[0]
            #print(name)
            with imageio.get_writer(name+'.gif', mode='I') as writer:
                #strt_in = strt
                #strt_out = strt
                for i in range(pose.shape[0]):
                    #fig = Figure(figsize=(10, 8), dpi=100)
                    #canvas = FigureCanvasAgg(fig)
                    #strt_in = strt_in+pose[i,:]
                    #strt_out = strt_out+outputs[i,:]
                    plt.scatter(pose[i,0::2], pose[i,1::2], c='r',label='ground_truth')
                    #dat = zip(label, pose[i,0:-6:3], pose[i,2:-6:3])
                    plt.scatter(outputs[i,0::2], outputs[i,1::2],c='b', label='prediction')
                    ax=plt.gca()
                    #[ax.annotate(lab,(x,y)) for lab,x,y in dat]
                    ax.set_aspect('equal', adjustable='box')
                    #plt.axis('square')
                    plt.savefig(name+'.png')
                    plt.close()
                    image = imageio.imread(name+'.png')
                    writer.append_data(image)
            writer.close()
        count=count+1

    test_loss = running_loss / dataset_size
    print(test_loss)

if __name__ == "__main__":
    main()