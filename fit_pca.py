#
# Fit the PCA model:
#
import numpy as np
from sklearn.decomposition import PCA

import os
import torch

# Load the trained multispeakers model
m=torch.load('../logs/44k/G_74.pth')

# We only care about emb_g.weight
X=m['model']['emb_g.weight']

X=X.cpu()
X=np.array(X)

# Nb of principal components to fit: 38 (we have 38 speakers)
N_pca=38

# Fitting
pca = PCA(n_components=N_pca)
pca = pca.fit(X)
X_pca = pca.transform(X)
#X_reconstruct = pca.inverse_transform(X_pca[0:N_pca])
principalVectors=pca.components_
principalComponents=X_pca
print(principalComponents.shape)

# Saving the result
np.save('PRINCIPAL_COMPONENTS.npy', principalComponents)
np.save('PRINCIPAL_VECTORS.npy', principalVectors)

# Set embeddings to zero
m['model']['emb_g.weight']=0*m['model']['emb_g.weight']
torch.save(m, 'G_38_speakers_0_v74.pth')
print('Neutral model saved to G_38_speakers_0_v74.pth')

