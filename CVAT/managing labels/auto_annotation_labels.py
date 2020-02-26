# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 13:45:38 2020

@author: smhas
"""
import json

with open('label_map_mrcnn.json', 'r') as jf:
    jdict = json.load(jf)


nlst = list()
for key, value in jdict['label_map'].items():
    ndict = dict()
    ndict['name'] = value
    # ndict['id'] = key
    ndict['attributes'] = []

    nlst.append(ndict)

with open('cvat_mrcnn_label.json', 'w') as jf2:
    json.dump(nlst, jf2, indent=4)
