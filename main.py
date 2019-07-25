#coding:utf-8
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import os
import sys
from data import MyDataset
import numpy as np
import time
from model import ASR_CTC
import torch.optim as optim
import re
import json
from tensorboardX import SummaryWriter 



if (__name__=='__main__'):
    torch.manual_seed(55)
    torch.cuda.manual_seed_all(55)
    opt = __import__('option') 

def dataset2dataloader(dataset):
    #dataset = filter(lambda audio: len(audio) < 200, dataset)
    return DataLoader(dataset,
        batch_size = opt.batch_size, 
        shuffle = True,
        num_workers = opt.num_workers,
        drop_last = False)

def show_lr(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return np.array(lr).mean()

if (__name__=='__main__'):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

    model = ASR_CTC(32, 256, 512).cuda()
    net = nn.DataParallel(model).cuda()

    writer = SummaryWriter()

    if(hasattr(opt, 'weights')):
        pretrained_dict = torch.load(opt.weights)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
        missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
        print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
        print('miss matched params:{}'.format(missed_params))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    train_dataset = MyDataset(opt.audio_path,
                        opt.audio_pad,
                        opt.txt_pad,
                        'train')
    val_dataset = MyDataset(opt.audio_path,
                            opt.audio_pad,
                            opt.txt_pad,
                            'test')

    print('num_data:{}'.format(len(train_dataset.data)))    
    print('num_data:{}'.format(len(val_dataset.data)))  
    #print(train_dataset)
    #print(train_dataset)
    train_loader = dataset2dataloader(train_dataset) 
    val_loader = dataset2dataloader(val_dataset)

    
    optimizer = optim.Adam(model.parameters(),
                lr = opt.base_lr,
                weight_decay = 0.,
                amsgrad = True)

    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                                step_size=1, 
                                                gamma=0.8)

    iteration = 0
    n = 0
    #tic = time.time()   
    
    for epoch in range(opt.max_epoch):
        #train_wer = []
        for (idx, input) in enumerate(train_loader):
            #model.train()
            audio = input.get('audio').cuda()
            txt = input.get('txt')
            audio_len = input.get('audio_len')
            txt_len = input.get('txt_len')
            
            optimizer.zero_grad()
            y = net(audio)
            #print('Inputs:',y.transpose(0,1).size(), 'targets:', txt.size(), audio_len.size(), txt_len.size())
            #txt1 = target = torch.randint(low=1, high=32, size=(36, 90), dtype=torch.long)
            log_probs = torch.randn(190, 36, 32).log_softmax(2).detach().requires_grad_()
            targets = torch.randint(1, 32, (36, 90), dtype=torch.long)
            
            input_lengths = torch.full((36,), 190, dtype=torch.long)
            target_lengths = torch.randint(10,90,(36,), dtype=torch.long)
            #print('Inputs:', log_probs.size(), 'targets:',targets.size())
            #print(targets)
            targets = txt
            #print(targets)
            #print(input_lengths)
            #print(audio_len.view(-1))
            #input_lengths = audio_len
            #loss = F.ctc_loss(log_probs, targets, audio_len.view(-1), txt_len.view(-1))
            loss = F.ctc_loss(y.transpose(0,1).log_softmax(2), txt, audio_len.view(-1), txt_len.view(-1))
            #writer.add_scalar('data/ctc_loss', loss.detach().cpu().numpy(), iteration)
            
            loss.backward()
            optimizer.step()
            
            train_loss = loss.item()

            print('iteration:%d, epoch:%d, train_loss:%.6f'%(iteration, epoch, train_loss))

            iteration += 1
            
            if(iteration  == 1):
                with torch.no_grad():
                    if iteration % 50001 == 0:
                        savename = os.path.join(opt.savedir, 'iteration_{}.pt'.format(iteration))
                        savepath = os.path.split(savename)[0]
                        if (not os.path.exists(savepath)): os.makedirs(savepath)
                        torch.save(model.state_dict(), savename)
                    
                    predict_txt_total = []
                    truth_txt_total = []
                    wer = []
                    for (idx, input) in enumerate(val_loader):            
                        vid = input.get('audio').cuda()
                        txt = input.get('txt').cuda()
                        vid_len = input.get('audio_len').cuda()
                        txt_len = input.get('txt_len').cuda()

                        y = net(vid)
                        pred_txt = y.argmax(-1)
                        pred_txt = [MyDataset.ctc_arr2txt(pred_txt[_], start=1) for _ in range(pred_txt.size(0))]
                        truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
                        wer.append(MyDataset.wer(pred_txt, truth_txt))
                        wer_mean = np.array(wer).mean()

                        #writer.add_scalar('data/wer_mean', wer_mean, n)

                        for (predict, truth) in list(zip(pred_txt, truth_txt))[:6]:
                            print('{:<50}|{:>50}'.format(predict.lower(), truth.lower()))
                        print(''.join(101*'-'))                
                        print('iteration={},epoch={},loss={},wer_mean={}'.format(iteration, epoch, 
                                                                train_loss, wer_mean))
                        print(''.join(101*'-'))
                        '''
                        savename = os.path.join(opt.savedir, 'iteration_{}_epoch_{}_wer_{:6f}.pt'.format(
                            iteration, epoch, wer_mean))  
                        savepath = os.path.split(savename)[0]
                        if (not os.path.exists(savepath)): os.makedirs(savepath)
                        #if wer <= 0.08: 
                        torch.save(model.state_dict(), savename)
                        '''
                        n += 1
                        break
    
    writer.close()
