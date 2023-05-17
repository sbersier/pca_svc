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


argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--n_pca", help="nb. of principal components (int)", type=int, default=10)
argParser.add_argument("-s", "--seed", help="set fixed random seed", type=int, default=None)
argParser.add_argument("-a", "--amplification", help="amplification factor", type=float, default=1)
argParser.add_argument("-i", "--input_file", help="starting multispeaker generator (.pth)", type=str)
argParser.add_argument("-o", "--output_file", help="name of output .pth file (should start with \"G_\")",type=str,default='G_random_seed_<seed>_PCA_<n_pca>_scale_<scale>.pth')

args = argParser.parse_args()

N_pca=args.n_pca
Amplification=args.amplification
# Set random seed
Seed=args.seed

if Seed:
    np.random.seed(Seed)

if args.output_file=='G_random_seed_<seed>_PCA_<n_pca>_scale_<scale>.pth':
    output_file='G_random_seed_'+str(Seed)+'_PCA_'+str(args.n_pca)+'_scale_'+str(Amplification)+'.pth'
else:
    output_file=args.output_file


# Load model 
model_0=torch.load(args.input_file,weights_only=True)
device=model_0['model']['emb_g.weight'][0][1].device

# Load PCA components:
principalComponents=np.load('PRINCIPAL_COMPONENTS.npy')
principalVectors=np.load('PRINCIPAL_VECTORS.npy')
if N_pca>len(principalComponents):
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
# Compute stats on principal components for later
Mean=np.mean(principalComponents, axis=0)
Std=np.std(principalComponents, axis=0)

# A 38 multispeaker model will generate 38 random voices
N_random_samples=len(model_0['model']['emb_g.weight'][:,0])


for i in tqdm(range(N_random_samples)):
    
    w=np.random.randn( N_pca)
    for j in range(N_pca):
        w[j]=(w[j]*Std[j]*Amplification+Mean[j])
            
    # Constructing the 'emb_g.weight' voice embedding tensor 
    V=np.zeros_like(principalVectors[0,:])
    
    for j in range(N_pca):
        V+=w[j]*principalVectors[j,:]

    V=torch.tensor(V)
    V=V.to(device)
    model_0['model']['emb_g.weight'][i]=V
    
torch.save(model_0,output_file)
print('*'*32)
print('Result saved to : ',output_file)
print('Done')
print('*'*32)
#############################################################################################
