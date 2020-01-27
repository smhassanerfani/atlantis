# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 17:03:41 2020

@author: SERFANI
"""
import os
import csv
import json
import requests

# Key:
api_key = 'da03d4d2c9d70753c5b918009711b9ea'

# Secret:
secret = 'f35c6f99121e3e83' 

natural_list = ['flood', 'river']
artificial_list = ['dam', 'canal']

#natural_list = list()
#artificial_list = list()

#with open('list.csv', encoding="utf8", errors='ignore') as csvfile:
#    list_csv_dict = csv.DictReader(csvfile)
#    for row in list_csv_dict:
#        natural_list.append(row.get('natural'))
#        artificial_list.append(row.get('artificial'))
#natural_list = list(filter(None, natural_list))

licenses = [
    {"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/","id": 1,"name": "Attribution-NonCommercial-ShareAlike License"},
    {"url": "http://creativecommons.org/licenses/by-nc/2.0/","id": 2,"name": "Attribution-NonCommercial License"},
    {"url": "http://creativecommons.org/licenses/by-nc-nd/2.0/","id": 3,"name": "Attribution-NonCommercial-NoDerivs License"},
    {"url": "http://creativecommons.org/licenses/by/2.0/","id": 4,"name": "Attribution License"},
    {"url": "http://creativecommons.org/licenses/by-sa/2.0/","id": 5,"name": "Attribution-ShareAlike License"},
    {"url": "http://creativecommons.org/licenses/by-nd/2.0/","id": 6,"name": "Attribution-NoDerivs License"},
    {"url": "http://flickr.com/commons/usage/","id": 7,"name": "No known copyright restrictions"},
    {"url": "http://www.usa.gov/copyright.shtml","id": 8,"name": "United States Government Work"}
    ]

os.mkdir('images')   
main_dir = os.path.join(os.getcwd(), 'images')

for dir1 in natural_list:
    for num in range(len(licenses)):
        file_path = os.path.join(dir1, licenses[num].get('name'))
        file_path2 = os.path.join(main_dir, file_path)
        os.makedirs(file_path2)
        
        tag = dir1
        license_type = licenses[num].get('id')
        per_page = 3
        page_number = 1
        
        url = 'https://www.flickr.com/services/rest/?method=flickr.photos.search&api_key={api_key}&tags={tag}&license={license_type}&is_commons=&per_page={per_page}&page={page_number}&format=json&nojsoncallback=1'.format(api_key=api_key, tag=tag, license_type=license_type, per_page=per_page, page_number=page_number)
        r = requests.get(url)
        data = r.json()
        
        for element in data['photos']['photo']:
            del element['title'], element['owner'], element['ispublic'], element['isfriend'], element['isfamily']

        with open(os.path.join(file_path2, 'json_file.json'), 'w') as tf:
            json.dump(data, tf, indent=2)
        
        
        for element in data['photos']['photo']:
            try:
                img_lnk = requests.get('https://farm{farm_id}.staticflickr.com/{server_id}/{id}_{secret}_z.jpg'.format(farm_id=element['farm'], server_id=element['server'], id=element['id'], secret=element['secret']))
                with open(os.path.join(file_path2, str(element) + '.jpg'), 'wb') as tf2:
                    tf2.write(img_lnk.content)
            except OSError:
                print('File name too long')
            except (requests.ConnectTimeout,requests.HTTPError, requests.ReadTimeout, requests.Timeout, requests.ConnectionError):
                print(' Max retries exceeded with url')


''' we need to pars the information that we got from flickr in such a format:       
{
    "id": 229753
    "license": 4,
    "flickr_url": "http://farm4.staticflickr.com/3271/2787713866_34ab4ca3d3_z.jpg",
    "width": 640,
    "height": 427,
    "file_name": "000000229753.jpg",
    "date_captured": "2013-11-17 00:12:59",
}
{
  "photos": {
    "page": 1,
    "pages": 123934,
    "perpage": 1,
    "total": "123934",
    "photo": [
      {
        "id": "49379823277",
        "owner": "82256086@N00",
        "secret": "d2afbf2523",
        "server": "65535",
        "farm": 66,
        "title": "Hudson River Sunset",
        "ispublic": 1,
        "isfriend": 0,
        "isfamily": 0
      }
    ]
  },
  "stat": "ok"
}
'''