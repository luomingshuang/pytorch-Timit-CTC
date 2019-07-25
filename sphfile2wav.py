#coding:utf-8
import params as hp
from sphfile import SPHFile
import glob
import os
 
if __name__ == "__main__":
    path = '/home/luomingshuang/ASR-DATA/TIMIT/train/*/*/*_.wav'
    sph_files = glob.glob(path)
    print(len(sph_files),"train utterences")
    for i in sph_files:
        sph = SPHFile(i)
        sph.write_wav(filename=i.replace("_.wav",".wav"))
        os.remove(i)
    path = '/home/luomingshuang/ASR-DATA/TIMIT/test/*/*/*.WAV'
    sph_files_test = glob.glob(path)
    print(len(sph_files_test),"test utterences")
    for i in sph_files_test:
        sph = SPHFile(i)
        sph.write_wav(filename=i.replace(".WAV",".wav"))
        os.remove(i)
    print("Completed")
