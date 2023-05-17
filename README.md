# pca_svc

A multispeaker model for so-vits-svc-fork was trained on 38 speakers (18 males/20 females) adult, english speaking readers from librivox.
The voice embeddings ( `model['model']['emb_g.weight']` ) were processed to extract the principal components (`PRINCIPAL_COMPONENTS.npy`) and the principal directions (`PRINCIPAL_VECTORS.npy`) that are later used to generate new voices.

First, you need to download `G_38_speakers_0_v74.pth` here:

git clone 
This is the "neutral" model where all speakers embeddings have been set to zero.
Which means that all 38 voices are the same. This is normal.


This repo contains: 

## randomVoice.py: 
Generates random voices
Example:

`python randomVoices.py --n_pca 10 --seed 2023 --input_file G_38_speakers_0_v74.pth --output_file G_random.pth`

Will generate G_random.pth that will containe 38 randomly generated voices based on 10 principal components. (max n_pca=38)

## test_pca.py
Example:
`python test_pca.py -v=1,0,-5`

will generate G_result.pth using the specified first 3 components. You can add components if you wish (max nb. of components is 38). For `test_pca.py`, only the first speaker (SPEAKER_01) contains the generated voice.

