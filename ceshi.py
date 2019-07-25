#coding:utf-8
import glob
import os
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import matplotlib.pyplot as plt
'''
###查看训练数据中最长的文本长度是多少(这里是指字符，即全部置为一个个字母)
audio_path = '/home/luomingshuang/ASR-DATA/TIMIT'
audios = glob.glob(os.path.join(audio_path, 'train', '*','*', '*.wav'))
data = []
for audio in audios:
    #items = os.path.split(audio)
    items = audio.strip('').split('.')               
    txt = items[0] + '.TXT'
    data.append((audio, txt))

lens = []
for one in data:
    txt = one[1]
    with open(txt, 'r') as f:
        for line in f.readlines():
            lines = line.strip() 
            lines = lines.upper()
            lines_list = list(lines)
            lines_list = lines_list[8:-2]
            lens.append(len(lines_list))

print(max(lens))
index = lens.index(max(lens))
print(data[index])
###length_max = 78

audio_path = '/home/luomingshuang/ASR-DATA/TIMIT'
audios = glob.glob(os.path.join(audio_path, 'train', '*','*', '*.wav'))
length=[]
data=[]
n=0
m=0
i=0
for audio in audios:
    fs, sound = wav.read(audio)
    length.append(len(sound))
    data.append(sound)
for d in data:
    mfcc_features = mfcc(d, samplerate=16000)
    if mfcc_features.shape[0]<200:
        n+=1
    if mfcc_features.shape[0]>=200 & mfcc_features.shape[0]<=400:
        m+=1
    if mfcc_features.shape[0]>400:
        i+=1
print('n<200:', n, '200<=n<=400:', m, 'n>400:', i)
maxvalue = max(length)
minvalue = min(length)
#print(maxvalue, minvalue)
index1 = length.index(maxvalue)
index2 = length.index(minvalue)
#print(index1, index2)
input_mfcc_max = mfcc(data[index1], samplerate=16000)
input_mfcc_min = mfcc(data[index2], samplerate=16000)
mfcc_min = input_mfcc_min.T
plt.matshow(mfcc_min)
plt.title('MFCC')
min_mfcc = np.asarray(input_mfcc_min)
max_length_mfcc = input_mfcc_max.shape[0]
padding= np.zeros(13)
array = []
if len(input_mfcc_min) < max_length_mfcc:
    array = [input_mfcc_min[_] for _ in range(input_mfcc_min.shape[0])]
    size = array[0].shape
    for i in range(max_length_mfcc-len(array)):
        array.append(padding)
    array = np.stack(array, axis=0)
print('padding_array:', array, array.shape)
print(len(data[index2]))
print(maxvalue, index1, input_mfcc_max, input_mfcc_max.shape, data[index1], len(data[index1]), audios[index1])
print(minvalue, index2, input_mfcc_min, input_mfcc_min.shape, data[index2], len(data[index2]), audios[index2])
'''
import torch
import torch.nn.functional as F
log_probs = torch.randn(190, 36, 32).log_softmax(2).detach().requires_grad_()
targets = torch.randint(1, 32, (36, 90), dtype=torch.long)
input_lengths = torch.full((36,), 190, dtype=torch.long)
target_lengths = torch.randint(10,90,(36,), dtype=torch.long)
print(log_probs.size(), targets.size(), input_lengths.size(), target_lengths.size())
loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths)
loss.backward()
print(loss)
