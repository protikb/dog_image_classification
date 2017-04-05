import xml.etree.cElementTree as ET
from PIL import Image
#tree=ET.ElementTree(file="/media/protik/PROTIK/dogs_data/Annotation/n02085620-Chihuahua/n02085620_7")
import os
#os.chdir("/media/protik/PROTIK/dogs_data/Annotation/n02085620-Chihuahua")
path_to_data="/media/protik/PROTIK/dogs_data/Annotation"
j=0
for foldername in os.listdir(path_to_data):
    path_each_folder=path_to_data+'/'+foldername
    i=0
    newpath ='/home/protik/Pictures/images/'
    newpath=newpath+`j`
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    for filename in os.listdir(path_each_folder):
        path=path_each_folder+"/"+filename
        print path
        tree=ET.ElementTree(file=path)
        root=tree.getroot()
        xmin=ymin=xmax=ymax=0
        for child in root:
            if child.tag=='object':
                for attr in child:
                    if attr.tag=='bndbox':
                        for data in attr:
                            if data.tag=='xmin':
                                xmin=data.text
                            if data.tag=='ymin':
                                ymin=data.text
                            if data.tag=='xmax':
                                xmax=data.text
                            if data.tag=='ymax':
                                ymax=data.text
        image_path='/media/protik/PROTIK/dogs_data/Images/'
        image_path=image_path+foldername+'/'
        if filename=='n02085620_7~':
            filename='n02085620_7'
        image_path+=filename
        image_path=image_path+'.jpg'
        img = Image.open(image_path) 
        img5 = img.crop((float(xmin),float(ymin),float(xmax),float(ymax)))
        saved_name=newpath
        saved_name+=`i`
        saved_name=saved_name+'.jpg'
        img5.save(saved_name)
        i=i+1
    j=j+1
