#############################################################################################
# RANDOM GENERATION:
#############################################################################################
import torch
from glob import glob
import numpy as np
import os, sys
from joblib import load
from tqdm import tqdm
import argparse
import json

# Example of usage: python randomVoices.py --n_pca 38 --amplification 1.5 --seed 42 --input_file G_38_speakers_0_v74.pth --output_file G_random_seed_25.pth

argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--n_pca", help="nb. of principal components (int)", type=int, default=None)
argParser.add_argument("-s", "--seed", help="set fixed random seed", type=int, default=None)
argParser.add_argument("-a", "--amplification", help="amplification factor", type=float, default=1)
argParser.add_argument("-i", "--input_file", help="neutral multispeaker file (.pth)", type=str)
argParser.add_argument("-c", "--config", help="original config file (.json)", type=str)
argParser.add_argument("-o", "--output_file", help="name of output .pth file (should start with \"G_\")",type=str)
argParser.add_argument("-g", "--output_config", help="name of output config file (.json)",type=str,default='random_config.json')
argParser.add_argument("-d", "--dump",help="dump generated components to dump.npy (true/false)",type=bool, default=True)

args = argParser.parse_args()

N_pca=args.n_pca
Amplification=args.amplification
# Set random seed
Seed=args.seed

model_0=torch.load(args.input_file,weights_only=True)
device=model_0['model']['emb_g.weight'].device

if N_pca==None:
    N_pca=model_0['model']['emb_g.weight'].shape[0]

if Seed:
    np.random.seed(Seed)

if args.output_file==None:
    output_file='G_random_seed_'+str(Seed)+'_PCA_'+str(args.n_pca)+'_scale_'+str(Amplification)+'.pth'
else:
    output_file=args.output_file


# Load PCA components:
PCA_mean=np.load('PCA_MEAN.npy')
principalVectors=np.load('PCA_VECTORS.npy')
XPCA_std=np.load('XPCA_STD.npy')

if N_pca>len(PCA_mean):
    print('*'*32)
    print('ERROR:')
    print('n_pca should be lower than the nb. of components in PRINCIPAL_COMPONENTS.py : ',len(principalComponents))
    print('*'*32)
    sys.exit()
if N_pca<1:
    print('*'*32)
    print('ERROR:')
    print('n_pca should >= 1')
    print('*'*32)   
    sys.exit()       


N_speakers=len(model_0['model']['emb_g.weight'][:,0])

PCA_COMP=np.random.randn(N_speakers,N_pca)

for i in range(N_pca):
    PCA_COMP[:,i]*=XPCA_std[i]

PCA_COMP*=Amplification

V=np.dot(PCA_COMP, principalVectors[0:N_pca]) + PCA_mean
V=torch.tensor(V)
V=V.to(device)
model_0['model']['emb_g.weight']=V
    
torch.save(model_0,output_file)
print('*'*32)
print('Result saved to : ',output_file)
print('Done')
print('*'*32)

if args.dump:
    np.array(V.cpu()).dump('dump.npy')
    print('*'*32)
    print('Components dumped to : dump.npy')
    print('Done')
print('*'*32)

# Generate new config file:

with open(args.config,'r') as f:
    conf=json.loads(f.read())
ns={'SPEAKER_'+"{:03d}".format(i+1):i for i in range(N_speakers)}

# dum
conf['spk']=ns
if args.dump:
    with open(args.output_config,'w') as f:
        json.dump(conf,f,indent=2)   

print('*'*32)
print('Config saved to : ',args.output_config)
print('Done')
print('*'*32)

#############################################################################################
