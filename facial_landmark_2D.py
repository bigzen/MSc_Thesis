import os
import glob
import torch as T
from tqdm import tqdm
from utilities.MocapDataset import MocapDataset
from models.models import RNN
from utilities.util import TrTstSplit, GetInputOutputSplit

def main():
    base_dir = '../IEMOCAP_full_release/'
    os.chdir(base_dir)
    dirs = glob.glob('*.csv')
    train, test = TrTstSplit(dirs)
    #print(train)
    head, aud, landM, emo = GetInputOutputSplit(train)
    dataset = MocapDataset(head, aud, landM, emo)
    dataset_size = len(head)
    dataloader = T.utils.data.DataLoader(dataset, batch_size=1, num_workers=2)
    device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
    rnn = RNN(28,256,1,106).to(device)
    #checkpoint = T.load('landmark_checkpoint/rnn_bestmodel.pth')
    #rnn.load_state_dict(checkpoint['model_state_dict'])
    criterion = T.nn.MSELoss()
    optimizer = T.optim.Adam(rnn.parameters(), lr=0.0001)
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler = T.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
    min_loss = 100
    for epoch in range(500):
        loop = tqdm(dataloader)
        #print('epoch:',epoch)
        rnn.train()
        running_loss = 0.0
        running_corrects = 0.0
        for idx, (audio, pose, name) in enumerate(loop):
            audio = audio.to(device)
            pose = pose.to(device)
            optimizer.zero_grad()
            with T.set_grad_enabled(True):
                #print(name, audio.shape, pose.shape)
                outputs = rnn(T.squeeze(audio))
                loss = criterion(outputs, T.squeeze(pose))

                # backward + optimize only if in training phase
                #if phase == 'train':
                loss.backward()
                optimizer.step()
            

            # statistics
            running_loss += loss.item()
            loop.set_description(f"Epoch [{epoch}/{500}]")
            loop.set_postfix(loss=loss.item())
        scheduler.step()

        epoch_loss = running_loss / dataset_size
        #epoch_acc = float(running_corrects) / dataset_size
        print('Epoch: {:.4f} Loss: {:.4f}'.format(
                    epoch, epoch_loss ))
        if (epoch+1)%100==0:
            T.save({'epoch': epoch,
                    'model_state_dict': rnn.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss}, 'landmark_checkpoint/rnn_checkpoint_{0}.pth'.format(epoch))
        if min_loss>epoch_loss:
            min_loss=epoch_loss
            T.save({'epoch': epoch,
                    'model_state_dict': rnn.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss}, 'landmark_checkpoint/rnn_bestmodel.pth'.format(epoch))

if __name__ == "__main__":
    main()