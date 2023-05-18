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
argParser.add_argument("-n", "--n_pca", help="nb. of principal components (int)", type=int, default=38)
argParser.add_argument("-s", "--seed", help="set fixed random seed", type=int, default=None)
argParser.add_argument("-a", "--amplification", help="amplification factor", type=float, default=1)
argParser.add_argument("-i", "--input_file", help="starting multispeaker generator (.pth)", type=str)
argParser.add_argument("-o", "--output_file", help="name of output .pth file (should start with \"G_\")",type=str,default='G_random_seed_<seed>_PCA_<n_pca>_scale_<scale>.pth')

args = argParser.parse_args()

N_pca=args.n_pca
Amplification=args.amplification
# Set random seed
Seed=args.seed

model_0=torch.load('G_38_speakers_0_v74.pth',weights_only=True)

Mean_C0_males=+1.5
Mean_C0_females=-1.5
Std_C0_males=0.8
Std_C0_females=0.8
def bimodal(MF, Mean_C0_females,Mean_C0_males, Std_C0_females, Std_C0_males):
    if MF==+1:
        return (np.random.randn()+Mean_C0_females)*Std_C0_females
    else:
        return (np.random.randn()+Mean_C0_males)*Std_C0_males
    
    
if Seed:
    np.random.seed(Seed)

if args.output_file=='G_random_seed_<seed>_PCA_<n_pca>_scale_<scale>.pth':
    output_file='G_random_seed_'+str(Seed)+'_PCA_'+str(args.n_pca)+'_scale_'+str(Amplification)+'.pth'
else:
    output_file=args.output_file


# Load model 
#model_0=torch.load(args.input_file,weights_only=True)
device=model_0['model']['emb_g.weight'].device

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
for i in range(N_speakers):
    PCA_COMP[i,0]=bimodal(np.mod(i,2), Mean_C0_females,Mean_C0_males, Std_C0_females, Std_C0_males)

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
#############################################################################################
