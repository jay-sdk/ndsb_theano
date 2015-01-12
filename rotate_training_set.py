#!/usr/bin/python
import glob
import os
try:
  import PIL.Image as Image
except ImportError:
  import Image

### Grab images and rotate to increase training set
print "Utiiity to rotate original training images by 90 degrees"
print "Should help against rescale issues and increase training set"
print "Loading training images...."

directory_names = list(set(glob.glob(os.path.join("orig_data","train", "*"))).difference(set(glob.glob(os.path.join("orig_data","train","*.*")))))
for folder in directory_names:
    current_class = folder.split(os.path.sep)[-1]
    print "Processing class " + current_class
    for filename_dir in os.walk(folder):
        for filename in filename_dir[2]:
            #read image files (jpg)
            if filename[-4:] != ".jpg":
              continue
            namefile_img = "{0}{1}{2}".format(filename_dir[0], os.path.sep, filename)
            #print namefile_img
            img = Image.open(namefile_img)
            rotimg = img.transpose(Image.ROTATE_90)
            #outfile = os.path.splitext(os.path.basename(namefile_img))[0]+'_r90.jpg'
            outfile = os.path.splitext(namefile_img)[0]+'_r90.jpg'
            #print outfile
            rotimg.save(outfile)
            
print "Done."
