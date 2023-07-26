# pca_svc

A multispeaker model for [so-vits-svc-fork](https://github.com/voicepaw/so-vits-svc-fork) was trained on 38 speakers (18 males/20 females) adult, english speaking readers from librivox. (note: Because of this, it might not be suited for "singing" voices. I haven't tried.)

The resulting voice embeddings ( `model['model']['emb_g.weight']` ) were processed to extract the principal components and the principal directions in order to generate new voices. 
The good point with voices that do not exist is that you can use them without fear. You choose (or randomly generate) a few numbers and you get a voice. 

Thanks to [@hataori-p](https://github.com/hataori-p) and [@Z3Coder](https://github.com/Z3Coder) and Void Stryker for useful discussions.

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

(help: `python test_pca.py --help`)

Example:

`python test_pca.py -c -8-6 0  -n G_38_speakers_0_v74.pth -f config_pca_38.json -o G_Alice_young.pth -s Alice -g Alice_young.json`

NOTES: 

1) Don't forget the "=" sign and no space)
2) models names ALWAYS must start with "G_" otherwise svcg doesn't show them.
   
will generate `G_Alice_young.pth` (the model) and `Alice_young.json` (the config file) containing 1 voice named "Alice" using the specified first 3 components. You can add components if you wish (max nb. of components is 38 in our case). 
You can generate a test example with:

`svc infer -a -fm crepe -m G_Alice_young.pth -c Alice_young.json -o test.out.mp3 test.mp3` 

There is also an option:`--randomize_other_components=true`    (by default set to `false`).
This option will fill the remaining components with random values compatible with the original dataset. 

EXPLORATION GUIDE:

NOTE: The whole thing is a bit tedious, I must admit.

To explore, I would recommend:

1) Start with 1 component (for example: +4.0 or -4.0) to see the effect:
```
python test_pca.py -c=-6  -n G_38_speakers_0_v74.pth -f config_pca_38.json -o G_test.pth -s test -g test.json
svc infer -a -fm crepe -m G_test.pth -c config_pca_38.json -o test.out.mp3 test.mp3 
```
Listen to test.out.mp3

Then do the same with:
```
python test_pca.py -c=+6  -n G_38_speakers_0_v74.pth -f config_pca_38.json -o G_test.pth -s test -g test.json
svc infer -a -fm crepe -m G_test.pth -c config_pca_38.json -o test.out.mp3 test.mp3 
```

2) Once you have chosen your preferred value, pass to the second component:
```
python test_pca.py -c=+6,-6  -n G_38_speakers_0_v74.pth -f config_pca_38.json -o G_test.pth -s test -g test.json
.
.
.
python test_pca.py -c=+4,+4  -n G_38_speakers_0_v74.pth -f config_pca_38.json -o G_test.pth -s test -g test.json
```
and so on...

3) Once you have set all the parameters you want (max 38 but I wouldn't recommend going beyond 3...):
You can fill the rest of the components with the --randomize_other_components=true 
You can run it a few times, it should stay relatively close to your choice but add a bit of variations.

For example, for a low booming male voice:
`python test_pca.py -c=10,-10,-10 -n G_38_speakers_0_v74.pth -f config_pca_38.json -o G_test.pth -s test -g test.json`

## randomVoice.py: 

(help: `python randomVoices.py --help`)

Generates 38 random voices, given a specified number of components (n_pca)
Example:

`python randomVoices.py --amplification 1.5 --seed 123456 --input_file G_38_speakers_0_v74.pth --config config_pca_38.json`

Will generate `G_random_seed_123456_PCA_38_scale_1.5.pth` that will contain 38 randomly generated voices and the config file `random_config.json`
The `--amplitication` (float point number) scales scales the generated random vectors. It will make the voices more diverse. Conversely, if you set it to 0.0 then all voices will be equal to the neutral voice. If you go too high, then voices start to sound weird (but it can be funny).

To listen to it:
- launch svcg
- set the model file to G_random_seed_123456_PCA_38_scale_1.5.pth
- set the config file to random_config.json
- select a voice
- select input test file test.mp3
- Check Auto predict F0, method to crepe
- infer
- You can select another voice in the speaker dropdown menu
- If you think the generated voices lack variations you can set the option `--amplification` to 1.5 or 2.0 (default is 1.0). It just scales the components by a constant factor. Note that if you increase `--amplification` too much the voices will sound less natural.

## extractVoices.py

(help: `python extractVoices.py --help`)

Assuming you found interesting voices in `G_random_seed_123456_PCA_38_scale_1.5.pth` (see example above with randomVoices.py):
voices 6, 30 and 18 and decide to name them Alice, Bob and Charlie
You extract thes voices from randomly generated model using:

python extractVoices.py --model G_random_seed_123456_PCA_38_scale_1.5.pth --conf random_config.json --list 6,30,18 --name Alice,Bob,Charlie --output G_Alice_Bob_Charlie.pth

The terminal output:

```
********************************
Model saved to:  G_Alice_Bob_Charlie.pth
config file saved to :  Alice_Bob_Charlie.json
********************************
```

Then you can use svcg as usual.


## fit_pca.py

(help: `python fit_pca.py --help`)

The script used to compute the principal components and the statistical their properties.

Assuming you have trained a multispeaker model (note: the number of speakers must be as large as you can afford. I used 38 speakers)

And assuming your model is named: G_multispeaker.pth

With configuration file: multispeaker.json

Then,

`python fit_pca.py --conf multispeaker.json --model G_multispeaker.pth --output G_neutral.pth`

will produce:
1) a neutral model G_neutral.pth (by neutral, I mean all voices are the same)
2) PCA_VECTORS.npy : containing the principal components
3) PCA_MEAN.npy : the average value of the projection of the trained voices on the principal components 
4) XPCA_STD.npy : the standard deviation of the projection of the trained voices on the the principal components 


