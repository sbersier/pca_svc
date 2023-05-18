import torch
from glob import glob
import numpy as np
import os, sys
from joblib import load
from tqdm import tqdm
import argparse
# usage: python test_pca.py -r=true -v=-2,0,0,... 
    
def getSpeakers(config):
    List_speakers=[]
    with open(config, 'r') as f:
        conf=f.readlines()
    for i in range(len(conf)):
        if 'spk' in conf[i]:
            ind=i
    conf=conf[ind+1:-2]
    for i in range(len(conf)):
        List_speakers.append(conf[i].strip('"').split(':')[0].strip().strip('"'))
        
    return List_speakers

argParser = argparse.ArgumentParser()
argParser.add_argument("-c","--components", type=str)
argParser.add_argument("-r","--randomize_other_components",type=bool, help='true / false', default=False)

args = argParser.parse_args()
w=args.components
random=args.randomize_other_components
w=[float(x) for x in w.split(',')]
N_pca=len(w)

model_0=torch.load('G_38_speakers_0_v74.pth',weights_only=True)
PCA_mean=np.load('PCA_MEAN.npy')
XPCA_std=np.load('XPCA_STD.npy')
principalVectors=np.load('PCA_VECTORS.npy')


config='config_pca_38.json'
speakers=getSpeakers(config)
N_speakers=len(speakers)

device=model_0['model']['emb_g.weight'][0][1].device


voice='SPEAKER_01'
ind=speakers.index(voice)



PCA_COMP=np.random.randn(N_speakers,256)
for i in range(N_pca):
    PCA_COMP[:,i]*=XPCA_std[i]
V=np.dot(PCA_COMP, principalVectors) + PCA_mean


W=np.dot(w, principalVectors[0:N_pca]) + PCA_mean
V[0:N_pca]=W

V=torch.tensor(V)
V=V.to(device)
model_0['model']['emb_g.weight'][ind,:]=V
torch.save(model_0,'G_result.pth')
print('Saved to G_result.pth in voice SPEAKER_01')
print('Done.')
