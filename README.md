# pca_svc

A multispeaker model for [so-vits-svc-fork](https://github.com/voicepaw/so-vits-svc-fork) was trained on 38 speakers (18 males/20 females) adult, english speaking readers from librivox. (note: Because of this, it might not be suited for "singing" voices. I haven't tried.)

The resulting voice embeddings ( `model['model']['emb_g.weight']` ) were processed to extract the principal components and the principal directions in order to generate new voices. 
The good point with voices that do not exist is that you can use them without fear. You choose (or randomly generate) a few numbers and you get a voice. 

Thanks to [@hataori-p](https://github.com/hataori-p) and [@Z3Coder](https://github.com/Z3Coder) for useful discussions.

NOTE: All this is a bit shaky and not user-friendly (no nice GUI). Also, the principal directions would probably be more meaningful with a lot more speakers during the training... For the moment, I would say that only the first, the second and, maybe the third components (components 0,1,2) are "useful".

Below, components 0 and 1 of the original voices used for the training (women=yellow, men=violet)
The other components don't seem to exhibit clustering (except, maybe component 2). Is it normal? Is it due to the small number of speakers? I don't know.

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

Then download `G_38_speakers_0_v74.pth` from [here](https://drive.google.com/file/d/1wVuJqNt52KvLAOAYP7_Ib4nXaCzMkl-j/view?usp=sharing)

and put it into the `pca_svc` folder

`G_38_speakers_0_v74.pth` is the "neutral" model where all speakers embeddings have been set to their average.
Which means that all 38 voices are the same. This is absolutely normal.


## test_pca.py
Example:
`python test_pca.py -c=1.2,0.0,-2.0`   (NOTE: Don't forget the "=" sign and no space)

will generate `G_result.pth` using the specified first 3 components. You can add components if you wish (max nb. of components is 38). Note that only the first speaker (SPEAKER_01) contains the generated voice. (see `randomVoices.py` below for more speakers)
You can generate the test example with:

`svc infer -a -fm crepe -m G_result.pth -c config_pca_38.json -o test.out.mp3 -s SPEAKER_01 test.mp3` 

There is also an option:`--randomize_other_components=true`    (by default set to `false`).
This option will fill the remaining components with random values compatible with the original dataset. 

EXPLORATION GUIDE:

NOTE: The whole thing is a bit tedious, I must admit.

To explore, I would recommend:

1) Start with 1 component (for example: +4.0 or -4.0) to see the effect:
```
python test_pca.py -c=+4.0
svc infer -a -fm crepe -m G_result.pth -c config_pca_38.json -o test.out.mp3 -s SPEAKER_01 test.mp3 
```
Listen to test.out.mp3

Then do the same with:
```
python test_pca.py -c=-4.0
svc infer -a -fm crepe -m G_result.pth -c config_pca_38.json -o test.out.mp3 -s SPEAKER_01 test.mp3 
```

2) Once you have chosen your preferred value, pass to the second component:
```
python test_pca.py -v=+4.0,-4.0
...
python test_pca.py -v=+4.0,+4.0
```
and so on...

3) Once you have set all the parameters you want (max 38 but I wouldn't recommend going beyond 3...):
You can fill the rest of the components with the --randomize_other_components=true 
You can run it a few times, it should stay relatively close to your choice but add a bit of variations.


## randomVoice.py: 
Generates 38 random voices, given a specified number of components (n_pca)
Example:

`python randomVoices.py --n_pca 25 --amplification 1.5 --seed 42 --input_file G_38_speakers_0_v74.pth --output_file G_random.pth`

Will generate G_random.pth that will contain 38 randomly generated voices based on 10 principal components. (max n_pca=38)

To listen to it:
- launch svcg
- set the model file to G_random.pth
- set the config file to config_pca_38.json
- select a voice
- select input test file test.mp3
- Check Auto predict F0, method to crepe
- infer
- You can select another voice in the speaker dropdown menu
- If you think the generated voices lack variations you can set the option `--amplification` to 1.5 or 2.0 (default is 1.0). It just scales the components by a constant factor. Note that if you increase `--amplification` too much the voices will sound less natural.


## fit_pca.py
The script used to compute the components (not very useful for you). I put it here for reference.
