#
# Fit the PCA model:
#
import numpy as np
from sklearn.decomposition import PCA

import os
import torch
import json
import argparse

# Example of usage: python fit_pca.py --conf config.json --model G_74.pth --output G_neutral_38_speakers.pth

argParser = argparse.ArgumentParser()
argParser.add_argument("-c", "--conf", help="config file", type=str, default="config.json")
argParser.add_argument("-m", "--model", help="multipeaker model (.pth)", type=str)
argParser.add_argument("-o", "--output", help="name of output .pth file (should start with \"G_\")",type=str,default='G_neutral_model.pth')

args = argParser.parse_args()
print(args)

if args.model==None:
    raise ValueError('You have to give the path to an existing multispeaker model.')

# Load the trained multispeakers model
m=torch.load(args.model,weights_only=True)

# We only care about emb_g.weight
X=m['model']['emb_g.weight']
X=X.cpu()

with open(args.conf,'r') as f:
    conf=json.loads(f.read())
N_speakers=len(list(conf['spk']))

X=X[0:N_speakers,:]

# Fitting
pca = PCA(svd_solver='full')
pca.fit(X)
X_pca = pca.transform(X)
principalVectors=pca.components_
principalComponents=X_pca

# Saving the results
#np.save('PRINCIPAL_COMPONENTS.npy', pca.transform(X))
np.save('PCA_VECTORS.npy', principalVectors)
np.save('PCA_MEAN.npy',pca.mean_)
np.save('XPCA_STD.npy',np.std(X_pca,axis=0)) 

# Set voice embeddings to mean value
mean=torch.tensor(pca.mean_).to(X.device)
for i in range(N_speakers):
    m['model']['emb_g.weight'][i]=mean

# Save neutral model    
torch.save(m, args.output)
print('Neutral model saved to '+args.output)

exit()
# Below, the plot is specific to my dataset

# plot
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize = (8,8))
p=[0,1]
ax = fig.add_subplot(1,1,1) 
N=[37,2,19,34,38,11,15,30,12,21,26,24,3,32,18,33,29,9,27,6,28,10,36,14,4,8,22,1,17,7,13,25,35,16,31,23,20,5]
h=-1
f=+1
G=[f,h,f,f,h,\
   h,h,f,f,h,\
   h,f,f,f,h,\
   f,f,h,h,f,\
   h,h,h,f,h,\
   h,h,f,h,f,\
   f,h,f,f,h,\
   f,f,f]
ax.set_xlabel('Principal Component '+str(p[0]), fontsize = 15)
ax.set_ylabel('Principal Component '+str(p[1]), fontsize = 15)
ax.set_title('PCA components', fontsize = 20)
s=ax.scatter(principalComponents[:,p[0]],principalComponents[:,p[1]],c=G)
for i in range(len(principalComponents)):
    plt.text(principalComponents[i,p[0]]-.100, principalComponents[i,p[1]]+.200, str(N[i]).zfill(2), fontsize=12)
