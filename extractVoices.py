#!/usr/bin/env python
# coding: utf-8

# Extract selected voices from random model
###########################################

import torch
import argparse
import json
import numpy as np

argParser = argparse.ArgumentParser()
argParser.add_argument("-c", "--conf", help="the random config file", type=str)
argParser.add_argument("-m", "--model", help="the random model (.pth)", type=str)
argParser.add_argument("-l", "--list",help="comma separated list of numbers corresponding the selected voices (ex. 13,17,3)",type=str)
argParser.add_argument("-n", "--name",help="comma separated list of names the selected voices (ex. Alice,Bob,Charlie)",type=str)
argParser.add_argument("-o", "--output", help="name of output .pth file (should start with \"G_\") The config file will have the same name with .json extention.",type=str)

args = argParser.parse_args()
conf=args.conf
model=args.model
l=[int(x) for x in args.list.split(',')]
name=args.name.split(',')
output=args.output

if output[:2]!='G_':
    output='G_'+output
if output[-4:]!='.pth':
    output=output+'.pth'
out_conf=output[2:-4]+'.json'

m=torch.load(model)
emb=m['model']['emb_g.weight'].clone()

for i in range(len(l)):
    m['model']['emb_g.weight'][i]=emb[l[i]-1]

torch.save(m,output)

with open(conf,'r') as f:
    conf=json.loads(f.read())

spk={}
for i in range(len(l)):
    spk[name[i]]=i

conf['spk']=spk

with open(out_conf,'w') as f:
    json.dump(conf,f,indent=2)   

print('*'*32)
print('Model saved to: ',output)
print('config file saved to : ',out_conf)
print('*'*32)

