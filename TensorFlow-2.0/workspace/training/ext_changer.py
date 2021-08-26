
import os
images_path = "./images/train"
image_list = os.listdir(images_path)
for i,  image in enumerate(image_list):
    title = os.path.splitext(image)[0]
    ext = os.path.splitext(image)[1]
    if ext == '.jpeg':
        src = images_path + '/' + image
        dst = images_path + '/' + title + '.jpg'
        os.rename(src, dst)