# Chose Images and correspond json_file is located in the same directory
# throgh code we get access to the directory and extract the name of each image file (id)
import os
import json


creativecommons = [1, 2, 3, 4, 5, 6, 7, 8]
natural_dir = os.scandir()
natural_list = list()

for dir1 in natural_dir:
    if not dir1.name.endswith('.py'):
        natural_list.append(dir1.name)


lst = list()

for label in natural_list:
    for num in creativecommons:
        path1 = os.path.join(os.getcwd(), '{label}'.format(label=label))
        path2 = os.path.join(path1, str(num))
        with os.scandir(path2) as it:
            for entry in it:
                if entry.name.endswith('.jpg'):
                    lst.append(entry.name.split('.')[0])
        with open(os.path.join(path2, 'json_file.json'), ) as jf:
            raw_json = json.load(jf)
        
        my_dict = {'sub_dict': []}
        for img_id in lst:
            
            dic_to_json = {}
            for img in raw_json['photos']['photo']:
                if img_id == img['id']:
                    try:
                        dic_to_json['id'] = img['id']
                        dic_to_json["secret"] = img["secret"]
                        dic_to_json["server"] = img["server"]
                        dic_to_json["farm"] = img["farm"]
                        dic_to_json["license"] = img["license"]
                        dic_to_json["flickr_url"] = img["flickr_url"]
                        dic_to_json["file_name"] = img["file_name"]
                        dic_to_json["date_captured"] = img["date_captured"]
                        dic_to_json["width"] = img["width"]
                        dic_to_json["height"] = img["height"]
                    except(KeyError):
                        print('KeyError')
                    my_dict['sub_dict'].append(dic_to_json)
                    break
                    
        with open (os.path.join(path2,'new_json_file.json'), 'a') as jf2:
            json.dump(my_dict, jf2, indent=4)
        
        my_dict.clear()
                    

# it = os.scandir()
# new_lst = list()

# for entry in it:
#     if entry.name.endswith('.jpg'):
#         new_lst.append(entry.name.split('.')[0])

# print(new_lst)

# now in this step we are going to go through the coresspond jsonfile to extract and parse what we want


#with open('json_file.json', ) as jf:
#    raw_json = json.load(jf)
#
#dic_to_json = {}
#for img_id in lst:
#    for img in raw_json['photos']['photo']:
#        if img_id == img['id']:
#            dic_to_json['id'] = img['id']
#            dic_to_json["secret"] = img["secret"]
#            dic_to_json["server"] = img["server"]
#            dic_to_json["farm"] = img["farm"]
#            dic_to_json["license"] = img["license"]
#            dic_to_json["flickr_url"] = img["flickr_url"]
#            dic_to_json["file_name"] = img["file_name"]
#            dic_to_json["date_captured"] = img["date_captured"]
#            dic_to_json["width"] = img["width"]
#            dic_to_json["height"] = img["height"]
#    
#    with open ('new_json_file.json', 'a') as jf2:
#        json.dump(dic_to_json, jf2, indent=2)