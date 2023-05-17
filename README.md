# pca_svc

A multispeaker model for so-vits-svc-fork was trained on 38 speakers (18 males/20 females) adult, english speaking readers from librivox.
The voice embeddings ( `model['model']['emb_g.weight']` ) were processed to extract the principal components (`PRINCIPAL_COMPONENTS.npy`) and the principal directions (`PRINCIPAL_VECTORS.npy`) that are later used to generate new voices.

Thanks to @hataori-p for useful discussions.

NOTE: All this is a bit shaky and not user-friendly (no nice GUI for the moment). Also, the principal directions would probably be more meaningful with a lot more speakers during the training...

Below, components 0 and 1 of the original voices used for the training (women=yellow, men=violet)
The other components don't seem to exhibit clustering. Is it normal? Is it due to the small number of speakers? I don't know.

![Figure_1](https://github.com/sbersier/pca_svc/assets/34165937/f9ba27e4-1c3c-483f-a51d-bbfeb1684068)



The meaning of these components is also still unclear.
Nevertheless,
- Clearly, the first component is related to male/female (or pitch?)
- The second component seems to be related to the high-mids content...
- Third component... ?
- Other components... ???


To give it a try:

`git clone https://github.com/sbersier/pca_svc.git`

`cd pca_svc`

Then download `G_38_speakers_0_v74.pth` from [here](https://drive.google.com/file/d/14ikSGDTG9GgabmlToEMlIy5szVJ3dtol/view?usp=sharing)

and put it into your `pca_svc` folder

`G_38_speakers_0_v74.pth` is the "neutral" model where all speakers embeddings have been set to zero.
Which means that all 38 voices are the same. This is normal.



## randomVoice.py: 
Generates random voices
Example:

`python randomVoices.py --n_pca 10 --seed 2023 --input_file G_38_speakers_0_v74.pth --output_file G_random.pth`

Will generate G_random.pth that will containe 38 randomly generated voices based on 10 principal components. (max n_pca=38)

To listen to it:
- launch svcg
- set the model file to G_random.pth
- set the config file to config_pca_38.json
- select a voice
- infer (from file or from mic)

## test_pca.py
Example:
`python test_pca.py -v=1.2,0.0,-2.0`

will generate G_result.pth using the specified first 3 components. You can add components if you wish (max nb. of components is 38). For `test_pca.py`, only the first speaker (SPEAKER_01) contains the generated voice.

You can experiment by starting from 0,0,0,... and increase or decrease each component separately and try to figure out what they correspond to.

To listen to it:
- launch svcg
- set the model file to G_result.pth
- set the config file to config_pca_38.json
- select voice "SPEAKER_01"
- infer (from file or from mic)

## fit_pca.py
The script used to compute the components (not very useful for you). I put it here for reference.
