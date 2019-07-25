#coding:utf-8
import numpy as np
import glob
import os
import re
import random
import editdistance
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import matplotlib.pyplot as plt
import option as opt

#the length of mfcc features max:
class MyDataset(Dataset):
    letters = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 
    'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 
    'Z', '-', "'", ',', ';', '"', ':']

    def __init__(self, audio_path, audio_pad, txt_pad, phase):
        self.audio_path = audio_path
        self.audio_pad = audio_pad
        self.txt_pad = txt_pad
        self.phase = phase

        self.audios = glob.glob(os.path.join(audio_path, self.phase, '*', '*', '*.wav'))
        #self.txts = glob.glob(os.path.join(audio_path, '*', '*', '*', '*.TXT'))
        self.data = []
        
        for audio in self.audios:
            #print(audio)
            #items = os.path.split(audio)
            fs, sound = wav.read(audio)
            mfcc_features = mfcc(sound, samplerate=16000)
            if mfcc_features.shape[0] <= 200:
                items = audio.strip('').split('.')               
                txt = items[0] + '.TXT'
                if self.phase == 'test':
                    txt = items[0] + '.TXT'
                self.data.append((audio, txt))
        #print(len(self.data))

    def __getitem__(self, idx):
        (audio, txt) = self.data[idx]
        #print(audio, txt)
        audio = self._load_audio(audio)
        #print(audio)
        audio = self._padding(audio, self.audio_pad)
        #print(audio.shape)
        #print(txt)
        anno = self._load_text(txt)
        #print(anno)
        anno_len = anno.shape[0]
        anno = self._padding(anno, self.txt_pad)
        #print(anno)
        return {'audio':torch.FloatTensor(audio),
                'txt':torch.LongTensor(anno),
                'txt_len':anno_len,
                'audio_len':self.audio_pad}

    def __len__(self):
        return len(self.data)

    def _load_audio(self, f):
        fs, sound = wav.read(f)
        mfcc_features = mfcc(sound, samplerate=16000)
    
        return mfcc_features
    
    def _load_text(self, text):
        with open(text, 'r') as f:
            lines = [line.strip().upper() for line in f.readlines()]
            #print(lines)
            #lines = lines.upper()
            lines_list = list(lines[0])
            #print(lines_list)
            lines_list = lines_list[8:-2]
            #print(lines_list)

        return MyDataset.txt2arr(' '.join(lines_list).upper(), 1)

    def _padding(self, array, length):
        array = [array[_] for _ in range(array.shape[0])]
        size = array[0].shape
        for i in range(length - len(array)):
            array.append(np.zeros(size))
        return np.stack(array, axis=0)

    @staticmethod
    def txt2arr(txt, start):
        arr = []
        for c in list(txt):
            arr.append(MyDataset.letters.index(c) + start)
        return np.array(arr)   

    @staticmethod
    def arr2txt(arr, start):
        txt = []
        for n in arr:
            if(n >= start):
                txt.append(MyDataset.letters[n - start])     
        return ''.join(txt)

    @staticmethod
    def ctc_arr2txt(arr, start):
        pre = -1
        txt = []
        for n in arr:
            if(pre != n and n >= start):
                txt.append(MyDataset.letters[n - start])
            pre = n
        return ''.join(txt)

    @staticmethod
    def wer(predict, truth):        
        word_pairs = [(p[0].split(' '), p[1].split(' ')) for p in zip(predict, truth)]
        wer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in word_pairs]
        return np.array(wer).mean()
        
    @staticmethod
    def cer(predict, truth):        
        cer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in zip(predict, truth)]
        return np.array(cer).mean() 
'''
train_dataset = MyDataset(opt.audio_path,
                        opt.audio_pad,
                        opt.txt_pad,
                        'train')
print(len(train_dataset.data))
loader = DataLoader(train_dataset, 
                    batch_size=opt.batch_size,
                    num_workers=opt.num_workers,
                    drop_last=False,
                    shuffle=True)
'''

