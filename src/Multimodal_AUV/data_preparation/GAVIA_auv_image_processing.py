"""
Created on Tue Jan 11 22:22:56 2022

@author: SA03JH
Processes jpg images from GAVIA AUV Grasshopper camera

Corrects illumination by divinding by average image and normalisation
Extracts metadata using exiftool.exe (must be in path) and converts to decimal coords

"""
import os  
import glob 
import skimage
from skimage import io
from skimage import filters
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#must have exiftool downloaded and saved in working dictionary and PyExifTool python module downloaded
import exiftool 
import numpy, PIL
from PIL import Image
import re

#Path to process with /, using \ alone will fail
path=r"D:/Strangford Lough_June2021_TOM/AUV Data/images/20210615075602/"
#As above, use /
save_folder='answers/'
#Image Enhancement method, 'CLAHE' or 'AverageSubtraction'
ImageEnhancement='AverageSubtraction'
#outpath, this is where it will save stuff
outpath=os.path.join(path,save_folder)
print(outpath)
print(path)
#path to ExifTool:
exiftool_path = r'D:/Strangford Lough_June2021_TOM/AUV Data/images/20210615075602/'
#Makes save folder, if there is already a folder it will just ignore this and print an error
try: 
   
    os.mkdir(os.path.join(path,save_folder))
except OSError as error: 
    print(error)
    
    
 #lists files of JPGs  
files=glob.glob(path+'*.jpg')



# Assuming all images are the same size, get dimensions of first image
h,w,d=(io.imread(files[0])).shape

N=len(files)
# Create a numpy array of floats to store the average (assume RGB images), numpy.float has been dropped
arr=numpy.zeros((h,w,3),float)

# Build up average pixel intensities, casting each image as an array of floats
for im in files:
    imarr=numpy.array(io.imread(im),dtype=float)
    arr=arr+imarr/N

# Round values in array and cast as 8-bit integer
arr1=numpy.array(numpy.round(arr),dtype=numpy.uint8)

# Generate, save and preview final image (this is the average which is removed from all photos)
out=Image.fromarray(arr1,mode="RGB")
out.save(os.path.join(outpath,"Average.png"))
out.show()

#creates a dataframe to save meta data
df = pd.DataFrame(columns=['Image_Name','path', 'altitude', 'depth','heading','lat','lon','pitch','roll','surge','sway'])



loopy=range(len(files))

###I added this to work out how to get the exiftool to work
os.chdir(exiftool_path)
# Get the current working directory
current_directory = os.getcwd()

# Print the current working directory
#EXIFTOOL.EXE  MUST BE DOWNLOADED AND SAVED IN THIS FILE
#Download from: https://exiftool.org/
print("Current Working Directory:", current_directory)

#extracts metadata
with exiftool.ExifToolHelper() as et:
        metadata = et.get_metadata(files)

 #loops over all photos
for i in loopy:
    #this block moves to file i then prints its processing this
    file=files[i]
    files1 = [file]
    print('processing image  '+file)

   #searches the comment to look for something in between '<altitude>(.*)</altitude>' etc. ad saves it in df
    comment=metadata[i]['File:Comment']
    altitude = re.search('<altitude>(.*)</altitude>', comment).group(1)
    depth = re.search('<depth>(.*)</depth>', comment).group(1)
    heading = re.search('<heading>(.*)</heading>', comment).group(1)
    lat = re.search('<lat>(.*)</lat>', comment).group(1)
    lon = re.search('<lon>(.*)</lon>', comment).group(1)
    pitch = re.search('<pitch>(.*)</pitch>', comment).group(1)
    roll = re.search('<roll>(.*)</roll>', comment).group(1)
    surge = re.search('<surge>(.*)</surge>', comment).group(1)
    sway = re.search('<sway>(.*)</sway>', comment).group(1)
    
    #These convert into latitude and longitude
    signlat = 1
    if lat[-1] == "S":
        signlat = -1    
    lenlat = len(lat)
    latCor = signlat * (float(lat[:2]) + float(lat[2:lenlat-2])/60.0)
    
    signlon=1
    if lon[-1] == "W":
        signlon = -1
    lenlon = len(lon)
    lonCor = signlon * (float(lon[:3]) + float(lon[3:lenlon-2])/60.0)
    
    
  
    #these undertake the corrections defined above on the images
    if ImageEnhancement=="AverageSubtraction":
        im1=numpy.array(io.imread(file),dtype=float)
        imcor=im1-arr
        out2=skimage.exposure.rescale_intensity(imcor, out_range='uint8')
    if ImageEnhancement=="CLAHE":
        im1=numpy.array(io.imread(file))
        #imcor=out2 = skimage.exposure.equalize_adapthist(im1, kernel_size=64)
        imcor=out2 = skimage.exposure.equalize_adapthist(im1)
        out2=skimage.exposure.rescale_intensity(imcor, out_range='uint8')
    

    #
    #it finally saves these photos
    io.imsave(os.path.join(path,save_folder, os.path.basename(file)),out2)

    #saves the lat etc into the dataframe
    df.loc[i] = [os.path.basename(file),file,altitude,str(-float(depth)),heading,latCor,lonCor,pitch,roll,surge,sway]
    
    
df.to_csv(os.path.join(path,save_folder,'coords.csv'))