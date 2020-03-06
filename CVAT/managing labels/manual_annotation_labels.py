import csv
import json


natural_list = list()
artificial_list = list()
with open('natural.csv', encoding='utf8', errors='ignore') as csvfile:
    csv_list = csv.DictReader(csvfile)
    for row in csv_list:
        natural_list.append(row['natural'])
        artificial_list.append(row['artificial'])

natural_list = list(filter(None, natural_list))
artificial_list = list(filter(None, artificial_list))

# print(natural_list, artificial_list)
nnatural_list = list()
nartificial_list = list()
for label in natural_list:
    ndict = dict()
    ndict['name'] = label

    ndict['attributes'] = []

    nnatural_list.append(ndict)

with open('natural_lables.json', 'w') as jf:
    json.dump(nnatural_list, jf, indent=4)
