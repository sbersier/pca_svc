import torch
from glob import glob
import numpy as np
import os, sys
from joblib import load
from tqdm import tqdm
import argparse
# usage: python test_pca.py -v=-1,+2,0,0,3
    
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

argParser.add_argument("-v","--vector")

args = argParser.parse_args()
w=args.vector
print(w)
w=[float(x) for x in w.split(',')]
N_pca=len(w)

model_0=torch.load('G_38_speakers_0_v74.pth')
config='config_pca_38.json'
speakers=getSpeakers(config)
device=model_0['model']['emb_g.weight'][0][1].device
principalComponents=np.load('PRINCIPAL_COMPONENTS.npy')
principalVectors=np.load('PRINCIPAL_VECTORS.npy')
Mean=np.mean(principalComponents, axis=0)
Std=np.std(principalComponents, axis=0)


#w=[+1,0,0,0,0]
voice='SPEAKER_01'
ind=speakers.index(voice)
for j in range(N_pca):
    w[j]=(w[j]*Std[j]+Mean[j])
    
V=np.zeros_like(principalVectors[0,:])

for j in range(N_pca):
    V+=w[j]*principalVectors[j,:]


V=torch.tensor(V)
V=V.to(device)
model_0['model']['emb_g.weight'][ind,:]=V
torch.save(model_0,'G_result.pth')
print('Saved to G_result.pth')
print('Done.')
