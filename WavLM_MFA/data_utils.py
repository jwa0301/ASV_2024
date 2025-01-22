import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset
import pickle as pk
from glob import glob
from pathlib import Path
import os, random
from scipy import signal
___author__ = "Hemlata Tak, Jee-weon Jung"
__email__ = "tak@eurecom.fr, jeeweon.jung@navercorp.com"


def genSpoof_list(dir_meta, is_train=False, is_eval=False):

    

    if is_train:
        d_meta = {}
        file_list = []
        with open(dir_meta, "r") as f:
            l_meta = f.readlines()
        for line in l_meta:
            _, key, _, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list

    elif is_eval:
        eval_files = sorted(glob(str(dir_meta) + '/*'))
        return eval_files
    else:
        d_meta = {}
        file_list = []
        with open(dir_meta, "r") as f:
            l_meta = f.readlines()
        for line in l_meta:
            _, key, _, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, list_IDs, labels, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(str(self.base_dir / f"{key}.flac"))#f"flac/{key}.flac"))
        X_pad = pad_random(X, self.cut)
        x_inp = Tensor(X_pad)
        y = self.labels[key]
        return x_inp, y


class Dataset_ASVspoof2019_devNeval(Dataset):
    def __init__(self, list_IDs, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
        """
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(str(self.base_dir / f"{key}.flac"))#f"flac/{key}.flac"))
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, key
    

    
class Dataset_ASVspoof2024_train_aug(Dataset):
    def __init__(self, list_IDs, labels, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        if 'T' in key :
            X, _ = sf.read(str(self.base_dir / f"{key}.flac"))#f"flac/{key}.flac"))
        else :
            X, _ = sf.read(str(self.base_dir / f"{key}.flac").replace('flac_T','flac_A'))
        X_pad = pad_random(X, self.cut)
        x_inp = Tensor(X_pad)
        y = self.labels[key]
        return x_inp, y
    
class Dataset_ASVspoof2024_train_aug2(Dataset):
    def __init__(self, list_IDs, labels, base_dir,add):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)
        
        
        
        # Load and configure augmentation files
        musan_path = "/Data1/VoxCeleb/VoxCeleb2/Audio/musan_split"
        rir_path = "/Data1/VoxCeleb/VoxCeleb2/Audio/RIRS_NOISES/simulated_rirs"
        self.add_prob = add
        self.noisetypes = ['noise','speech','music']
        self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
        self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}
        self.noiselist = {}
        augment_files   = glob(os.path.join(musan_path,'*/*/*/*.wav'))
        for file in augment_files:
            if file.split('/')[-4] not in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)
        self.rir_files  = glob(os.path.join(rir_path,'*/*/*.wav'))
        

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        if 'T' in key :
            X, _ = sf.read(str(self.base_dir / f"{key}.flac"))#f"flac/{key}.flac"))
        else :
            X, _ = sf.read(str(self.base_dir / f"{key}.flac").replace('flac_T','flac_A'))
        audio = pad_random(X, self.cut)
        
        audio = np.stack([audio],axis=0)
        # Data Augmentation
        augtype = random.randint(0,5)
        if augtype == 0:   # Original
            audio = audio
        elif augtype == 1: # Reverberation
            audio = self.add_rev(audio)
        elif augtype == 2: # Babble
            audio = self.add_noise(audio, 'speech')
        elif augtype == 3: # Music
            audio = self.add_noise(audio, 'music')
        elif augtype == 4: # Noise
            audio = self.add_noise(audio, 'noise')
        elif augtype == 5: # Television noise
            audio = self.add_noise(audio, 'speech')
            audio = self.add_noise(audio, 'music')
        
        x_inp = Tensor(audio[0])
        y = self.labels[key]
        
        return x_inp, y
    
    def add_rev(self, audio):
        rir_file    = random.choice(self.rir_files)
        rir, sr     = sf.read(rir_file)
        rir         = np.expand_dims(rir.astype(np.float64),0)
        rir         = rir / np.sqrt(np.sum(rir**2))
        return signal.convolve(audio, rir, mode='full')[:,:self.cut]

    def add_noise(self, audio, noisecat):
        clean_db    = 10 * np.log10(np.mean(audio ** 2)+1e-4) 
        numnoise    = self.numnoise[noisecat]
        noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio, sr = sf.read(noise)
            length = self.cut
            noiseaudio = pad_random(noiseaudio, self.cut)
#            if noiseaudio.shape[0] <= length:
#                shortage = length - noiseaudio.shape[0]
#                noiseaudio = np.pad(noiseaudio, (0, shortage), 'wrap')
#            start_frame = np.int64(random.random()*(noiseaudio.shape[0]-length))
#            noiseaudio = noiseaudio[start_frame:start_frame + length]
            noiseaudio = np.stack([noiseaudio],axis=0)
            noise_db = 10 * np.log10(np.mean(noiseaudio ** 2)+1e-4) 
            noisesnr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
            noises.append(np.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
        noise = np.sum(np.concatenate(noises,axis=0),axis=0,keepdims=True)
        return noise + audio
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

class Dataset_ASVspoof2024_eval(Dataset):
    def __init__(self, list_IDs, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
        """
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index].split('/')[-1].split('.')[0]
        X, _ = sf.read(str(self.base_dir / f"{key}.flac"))#f"flac/{key}.flac"))
        X_pad = pad_random(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, key
    
    
    
class Dataset_embd_train(Dataset):
    def __init__(self, list_IDs, labels, base_dir, aasist, resmax, lcnn, ofd):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
#        self.base_dir = base_dir
#        self.cut = 64600  # take ~4 sec audio (64600 samples)
        self.aasist = aasist
        self.resmax = resmax
        self.lcnn = lcnn
        self.ofd = ofd
        if self.aasist :
     
            with open('/Data1/aasist/embedding/aasist_cm_embd_trn.pk', "rb") as f:
                self.aasist_embd = pk.load(f)

        if self.resmax :
    
            with open('/Data1/aasist/embedding/resmax_cm_embd_trn.pk', "rb") as f1:
                self.resmax_embd = pk.load(f1)
    
        if self.lcnn :
   
            with open('/Data1/aasist/embedding/lcnn_cm_embd_trn.pk', "rb") as f2:
                self.lcnn_embd = pk.load(f2)

        if self.ofd :
 
            with open('/Data1/aasist/embedding/ofd_cm_embd_trn.pk', "rb") as f3:
                self.ofd_embd = pk.load(f3)
 

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
#        X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
#        X_pad = pad_random(X, self.cut)
#        x_inp = Tensor(X_pad)
        if self.aasist :
            x_inp = self.aasist_embd[key]
        if self.resmax :
            x_inp = np.concatenate((x_inp,self.resmax_embd[key]),axis=0)
        if self.lcnn :
            x_inp = np.concatenate((x_inp,self.lcnn_embd[key]),axis=0)
        if self.ofd :
            x_inp = np.concatenate((x_inp,self.ofd_embd[key]),axis=0)
        
        x_inp = Tensor(x_inp)
            
        y = self.labels[key]
        return x_inp, y
    
    
    
    
class Dataset_embd_devNeval(Dataset):
    def __init__(self, list_IDs, base_dir,aasist,resmax,lcnn,ofd,mode):
        """self.list_IDs	: list of strings (each string: utt key),
        """
        self.list_IDs = list_IDs
        self.base_dir = base_dir
#        self.cut = 64600  # take ~4 sec audio (64600 samples)
        self.aasist = aasist
        self.resmax = resmax
        self.lcnn = lcnn
        self.ofd = ofd
        self.mode = mode

        if self.aasist :
     
            with open('/Data1/aasist/embedding/aasist_cm_embd_dev.pk', "rb") as f:
                self.aasist_embd_dev = pk.load(f)
            with open('/Data1/aasist/embedding/aasist_cm_embd_eval.pk', "rb") as f4:
                self.aasist_embd_eval = pk.load(f4)

        if self.resmax :
    
            with open('/Data1/aasist/embedding/resmax_cm_embd_dev.pk', "rb") as f1:
                self.resmax_embd_dev = pk.load(f1)
            with open('/Data1/aasist/embedding/resmax_cm_embd_eval.pk', "rb") as f5:
                self.resmax_embd_eval = pk.load(f5)
    
        if self.lcnn :
   
            with open('/Data1/aasist/embedding/lcnn_cm_embd_dev.pk', "rb") as f2:
                self.lcnn_embd_dev = pk.load(f2)
            with open('/Data1/aasist/embedding/lcnn_cm_embd_eval.pk', "rb") as f6:
                self.lcnn_embd_eval = pk.load(f6)

        if self.ofd :
 
            with open('/Data1/aasist/embedding/ofd_cm_embd_dev.pk', "rb") as f3:
                self.ofd_embd_dev = pk.load(f3)
            with open('/Data1/aasist/embedding/ofd_cm_embd_eval.pk', "rb") as f7:
                self.ofd_embd_eval = pk.load(f7)



    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
#        X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
#        X_pad = pad(X, self.cut)
        if self.aasist :
            if self.mode == 'dev' :
                dev_x_inp = self.aasist_embd_dev[key]
            if self.mode == 'eval' :
                eval_x_inp = self.aasist_embd_eval[key]
        if self.resmax :
            if self.mode == 'dev' :
                dev_x_inp = np.concatenate((dev_x_inp,self.resmax_embd_dev[key]),axis=0)
            if self.mode == 'eval' :
                eval_x_inp = np.concatenate((eval_x_inp,self.resmax_embd_eval[key]),axis=0)
        if self.lcnn :
            if self.mode == 'dev' :
                dev_x_inp = np.concatenate((dev_x_inp,self.lcnn_embd_dev[key]),axis=0)
            if self.mode == 'eval' :
                eval_x_inp = np.concatenate((eval_x_inp,self.lcnn_embd_eval[key]),axis=0)
        if self.ofd :
            if self.mode == 'dev' :
                dev_x_inp = np.concatenate((dev_x_inp,self.ofd_embd_dev[key]),axis=0)
            if self.mode == 'eval' :
                eval_x_inp = np.concatenate((eval_x_inp,self.ofd_embd_eval[key]),axis=0)
        if self.mode == 'dev':
            dev_x_inp = Tensor(dev_x_inp)
            return dev_x_inp, key
        if self.mode == 'eval':
            eval_x_inp = Tensor(eval_x_inp)
            return eval_x_inp, key
