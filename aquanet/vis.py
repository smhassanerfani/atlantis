from shutil import copyfile
import os
from skimage import io
import numpy as np

path = '/home/zhenyao/Project/atlantis_file/results/val/'
method = ['aquanet4', 'pspnet', 'deeplabv3', 'ocnet', 'ocrnet', 'ocnet', 'danet', 'gcnet', 'dnlnet']
# save_path = path+'all/'
# for root, dirs, files in os.walk(path+method[0], topdown=True):
#     for name in files:
#         if name.endswith("_color.png"):
#             imglist = []
#             for i in range(len(method)):
#                 img_path = path + method[i]+'/'+name
#                 im = io.imread(img_path)
#                 imglist.append(im)
#             img_path = path + method[0]+'/'+name.replace('color', 'gt')
#             im = io.imread(img_path)
#             imglist.append(im)
#             save_img = np.hstack(imglist)
#             io.imsave(save_path+name, save_img)

img_name = '48874819496_color.png'
namebase = '6'
for i in range(len(method)):
    img_path = path + method[i] + '/' + img_name
    new_path = path+'vis/'
    new_name = namebase+'_'+str(method[i])+'.png'
    copyfile(img_path, new_path+new_name)

img_path = path + method[0] + '/' + img_name.replace('color', 'gt')
new_path = path+'vis/'
new_name = namebase+'gt_'+'.png'
copyfile(img_path, new_path+new_name)

