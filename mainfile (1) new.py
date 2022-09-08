import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib as mpl
import traceback
import pandas as pd
import os
import imutils
import numpy as np
#import dlib
from keras.models import model_from_json
from keras.models import Model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import ZeroPadding2D,Convolution2D,MaxPooling2D
from tensorflow.keras.layers import Dense,Dropout,Softmax,Flatten,Activation,BatchNormalization
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from sklearn.neural_network import MLPClassifier
import tensorflow.keras.backend as K
import cv2


from tkinter import *
from PIL import ImageTk,Image
from tkinter import filedialog
import tkinter as tk
from tkinter import filedialog as fd
from tkinter import messagebox


# create a deep neural model
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1,1)))	
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))

# Load model weights
model.load_weights('vgg_face_weights.h5')

vgg_face_descriptor = Model(inputs=model.layers[0].input
, outputs=model.layers[-2].output)

def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

    

train=pd.read_csv("personality.csv");

trainv=train.values

my_dict = {}

trainfname=[]
trainlabel=[]

result_dict = {}
result_count=0
h=0
w=0
letterGray=0
colcnt=0
for i in range(len(trainv)):
    my_dict[trainv[i][0]]=trainv[i][1]
    filename= './train/' + trainv[i][0] + ".png"
    try:
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        thresh=255-thresh;
        thinned = cv2.ximgproc.thinning(thresh);
        cv2.imwrite('temp.png', thinned)
    
        img2_representation = vgg_face_descriptor.predict(preprocess_image('temp.png'))[0,:]
        trainfname.append(img2_representation)
        trainlabel.append(trainv[i][0])
    except Exception as e:
        print('faild for image' ,filename)
    
    
print('Loaded training images')
neuralclf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(35), random_state=1)
neuralclf.fit(trainfname,trainlabel)


def detectPersonality(imgname):
    global trainfname,trainlabel,my_dict,result_count
    img = cv2.imread(imgname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    thinned = cv2.ximgproc.thinning(thresh);
    cv2.imwrite('temp2.png', thinned)
    
    img1_representation = vgg_face_descriptor.predict(preprocess_image('temp2.png'))[0,:]
    #mindis = neuralclf.predict(img1_representation)
    mindis=999999
    minwho='notknown'
    print('Finding match for character ', imgname)
    for i in range(len(trainfname)):
            
            cval=findCosineDistance(img1_representation,trainfname[i])
            if cval<mindis:
               mindis=cval
               minwho=trainlabel[i]

    pers=my_dict[minwho]
    result_count=result_count+1
    if minwho in result_dict:
        result_dict[minwho]=result_dict[minwho]+1
    else:
        result_dict[minwho]=1
    
    print('Personality found ', pers, ' with scpre ',mindis)               
    

def processPerResults():
    global result_dict,result_count,my_dict,label1,label2,label3,lable4,label5
    i=0;
    marklist = sorted(result_dict.items(), key=lambda x:x[1],reverse=True)
    result_dict = dict(marklist)
    sth=""  
    #result_dict=sorted(result_dict.items(), key=lambda x: x[1])

    print("+++++++++++++++++++++++++++++++++++++++++");
    print("Personality " + "\t\t\t" + "Probability"); 
    for plabel in result_dict:
        per=result_dict[plabel]*100.0/result_count;
        persona=my_dict[plabel]
        print(persona,'\t\t\t',per)
        sth = sth + "\n--------------------------------------------------------------------------\n\n" + persona + "\nPercentage = " + "{:.2f}".format(per);
        
        #if i==0:
        #    label1.config(text=sth)
        #if i==1:
        #    label2.config(text=sth)
        #if i==2:
        #    label3.config(text=sth)
        #if i==3:
        #    label4.config(text=sth)
        #if i==4:
        #    label5.config(text=sth)
        i=i+1
        if i==5:
            break

    print("+++++++++++++++++++++++++++++++++++++++++");     
    messagebox.showinfo("Personality results", sth)  

mpl.rcParams['legend.fontsize'] = 10

pd.set_option('display.expand_frame_repr', False)
fn=0
path='./result/'



# In[findFeaturPoints]
def findCapPoints(img):
    cpoints=[]
    dpoints=[]
    for i in range(img.shape[1]):
        col = img[:,i:i+1]
        k = col.shape[0]
        while k > 0:
            if col[k-1]==255:
                dpoints.append((i,k))
                break
            k-=1
        
        for j in range(col.shape[0]):
            if col[j]==255:
                cpoints.append((i,j))
                break
    return cpoints,dpoints


# In[wordSegment]
#*****************************************************************************#
def wordSegment(textLines):
    wordImgList=[]
    counter=0
    cl=0
    for txtLine in textLines:
        gray = cv.cvtColor(txtLine, cv.COLOR_BGR2GRAY)
        th, threshed = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV|cv.THRESH_OTSU)
        final_thr = cv.dilate(threshed,None,iterations = 20)

        plt.imshow(final_thr)
        plt.title("Word segment")
        plt.show()
        
        contours, hierarchy = cv.findContours(final_thr,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        boundingBoxes = [cv.boundingRect(c) for c in contours]
        (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes), key=lambda b: b[1][0], reverse=False))
       
        for cnt in contours:
            area = cv.contourArea(cnt)
 
#            print area
            if area > 10000:
                print ('Area= ',area)
                x,y,w,h = cv.boundingRect(cnt)
                print (x,y,w,h)
                letterBgr = txtLine[0:txtLine.shape[1],x:x+w]
                wordImgList.append(letterBgr)
 
                cv.imwrite("./result/words/" + str(counter) +".jpg",letterBgr)
                counter=counter+1
        cl=cl+1
       
    return wordImgList
#*****************************************************************************#
    
# In[fitToSize]
#*****************************************************************************#
def fitToSize(thresh1):
    
    mask = thresh1 > 0
    coords = np.argwhere(mask)

    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1   # slices are exclusive at the top
    cropped = thresh1[x0:x1,y0:y1]
    return cropped
   
#*****************************************************************************#
    
# In[lineSegment]
#*****************************************************************************#
def lineSegment(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    th, threshed = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV|cv.THRESH_OTSU)
   
    upper=[]
    lower=[]
    flag=True
    for i in range(threshed.shape[0]):

        col = threshed[i:i+1,:]
        cnt=0
        if flag:
            cnt=np.count_nonzero(col == 255)
            if cnt >0:
                upper.append(i)
                flag=False
        else:
            cnt=np.count_nonzero(col == 255)
            if cnt <2:
                lower.append(i)
                flag=True
    textLines=[]
    if len(upper)!= len(lower):lower.append(threshed.shape[0])
#    print upper
#    print lower
    for i in range(len(upper)):
        timg=img[upper[i]:lower[i],0:]
        
        if timg.shape[0]>5:
            fig1 = plt.figure()
            plt.imshow(timg)
            plt.title("Line segment")
            plt.show()
            timg=cv.resize(timg,((timg.shape[1]*5,timg.shape[0]*8)))
            textLines.append(timg)

    return textLines
#*****************************************************************************#

# In[baselines]:
##******************************************************************************#
def baselines(letter2, upoints, dpoints):
##-------------------------Creating upper baseline-------------------------------##
    global h,w,letterGray
    colu = []
    for i in range(len(upoints)):
        colu.append(upoints[i][1])
    
    maxyu = max(colu)
    minyu = min(colu)
    avgu = (maxyu + minyu) // 2
    meanu = np.around(np.mean(colu)).astype(int)
    print('Upper:: Max, min, avg, mean:: ',maxyu, minyu, avgu, meanu)
    
##-------------------------------------------------------------------------------##
##-------------------------Creating lower baseline process 1--------------------------##
    cold = []
    for i in range(len(dpoints)):
        cold.append(dpoints[i][1])
    
    maxyd = max(cold)
    minyd = min(cold)
    avgd = (maxyd + minyd) // 2
    meand = np.around(np.mean(cold)).astype(int)
    print('Lower:: Max, min, avg, mean:: ',maxyd, minyd, avgd, meand)
    
##-------------------------------------------------------------------------------##
##-------------------------Creating lower baseline process 2---------------------------##
    cn = []
    count = 0

    for i in range(h):
        for j in range(w):
            if(letterGray[i,j] == 255):
                count+=1
        if(count != 0):
            cn.append(count)
            count = 0    
    maxindex = cn.index(max(cn))
    print('Max pixels at: ',maxindex)
    
##------------------Printing upper and lower baselines-----------------------------##
    
    cv.line(letter2,(0,meanu),(w,meanu),(255,0,0),2)
    lb = 0
    if(maxindex > meand):
        lb = maxindex
        cv.line(letter2,(0,maxindex),(w,maxindex),(255,0,0),2)
    else:
        lb = meand
        cv.line(letter2,(0,meand),(w,meand),(255,0,0),2)
        
    #plt.imshow(letter2)
    #plt.show()
    return meanu, lb
##******************************************************************************###

# In[histogram]:
##*******************************************************************************###
def histogram(letter2, upper_baseline, lower_baseline):
    ##------------Making Histograms (Default)------------------------######
    cropped = letter2[upper_baseline:lower_baseline,0:w]
    #plt.imshow(cropped)
    #plt.show()
    colcnt = np.sum(cropped==255, axis=0)
    x = list(range(len(colcnt)))
    plt.plot(colcnt)
    plt.xlabel('pixel');
    plt.ylabel('Count');
    plt.title("Histogram")
    plt.fill_between(x, colcnt, 1, facecolor='blue', alpha=0.5)
    plt.show()  
    return colcnt     
####---------------------------------------------------------------------------#####

# In[Visualize]:
##*******************************************************************************###
def visualize(letter2, upper_baseline, lower_baseline, min_pixel_threshold, min_separation_threshold, min_round_letter_threshold):
    global colcnt
    seg = []
    seg1 = []
    seg2 = []
   ## Check if pixel count is less than min_pixel_threshold, add segmentation point
    for i in range(len(colcnt)):
      if(colcnt[i] < min_pixel_threshold):
          seg1.append(i)
          
    ## Check if 2 consequtive seg points are greater than min_separation_threshold in distance
    for i in range(len(seg1)-1):
        if(seg1[i+1]-seg1[i] > min_separation_threshold):
            seg2.append(seg1[i])

##------------Modified segmentation for removing circles----------------------------###            
    arr=[]
    for i in (seg2):
        arr1 = []
        j = upper_baseline
        while(j <= lower_baseline):
            if(letterGray[j,i] == 255):
                arr1.append(1)
            else:
                arr1.append(0)
            j+=1
        arr.append(arr1)
    print('At arr Seg here: ', seg2)
    
    ones = []
    for i in (arr):
        ones1 = []
        for j in range(len(i)):
            if (i[j] == 1):
                ones1.append([j])
        ones.append(ones1)
    
    diffarr = []
    for i in (ones):
        diff = i[len(i)-1][0] - i[0][0]
        diffarr.append(diff)
    print('Difference array: ',diffarr)
    
    for i in range(len(seg2)):
        if(diffarr[i] < min_round_letter_threshold):
            seg.append(seg2[i])
##---------------------------------------------------------------------------##
    ## Make the Cut 
    for i in range(len(seg)):
        letter3 = cv.line(letter2,(seg[i],0),(seg[i],h),(255,0,0),2)
    
    print("Does it work::::")
    plt.imshow(letter3)
    plt.xlabel("width");
    plt.ylabel("height");
    plt.title("Word Segment")
    plt.show()
    return seg 
###---------------------------------------------------------------------------#####  

# In[segmentCharacters]
def segmentCharacters(seg,lettergray):
    s=0
    wordImgList = []
    global fn
    for i in range(len(seg)):
        if i==0:
            s=seg[i]
            if s > 15:
                wordImg = lettergray[0:,0:s]
                cntx=np.count_nonzero(wordImg == 255) 
                print ('count',cntx)
                #plt.imshow(wordImg)
                #plt.show()
                fn=fn+1
            else:
                continue
        elif (i != (len(seg)-1)):
            if seg[i]-s > 15:
                wordImg = lettergray[0:,s:seg[i]]
                cntx=np.count_nonzero(wordImg == 255) 
                print ('count',cntx)
                #plt.imshow(wordImg)
                #plt.show()
                fn=fn+1
                s=seg[i]
            else:
                continue
        else:
            wordImg = lettergray[0:,seg[len(seg)-1]:]
            cntx=np.count_nonzero(wordImg == 255) 
            print ('count',cntx)
            #plt.imshow(wordImg)
            #plt.show()
            fn=fn+1
        wordImgList.append(wordImg)

    return wordImgList

#*****************************************************************************#
def findPersonality(img):
    # In[Main]:
    global h,w,letterGray,colcnt
    try:
        textLines=lineSegment(img)
        print ('No. of Lines',len(textLines))
        imgList=wordSegment(textLines)
        print ('No. of Words',len(imgList))
        counter = 0
        for letterGray in imgList:
            print ('LetterGray shape: ',letterGray.shape)
            gray = cv.cvtColor(letterGray, cv.COLOR_BGR2GRAY)
            th, letterGray = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV|cv.THRESH_OTSU)
            letterGray = fitToSize(letterGray)
            letter2 = letterGray.copy()
            letterGray = cv.dilate(letterGray,None,iterations = 4)

            h = letterGray.shape[0]
            w = letterGray.shape[1]
            
            upoints, dpoints=findCapPoints(letterGray)        
            meanu, lb = baselines(letter2, upoints, dpoints)
            
    ##-----------Final Baseline row numbers-----------------------####
    #       Ignore all points avove and below these rows 
            upper_baseline = meanu
            lower_baseline = lb
            
    ##--------------------Make histogram-------------------------------------###   
            
            colcnt = histogram(letter2, upper_baseline, lower_baseline)
            
    ###------------------------Visualize segmentation------------------------------#####        
            ## Tuning Parameters
            min_pixel_threshold = 25
            min_separation_threshold = 35
            min_round_letter_threshold = 190
            
            seg = visualize(letter2, upper_baseline, lower_baseline, min_pixel_threshold, min_separation_threshold, min_round_letter_threshold)
            wordImgList = segmentCharacters(seg,letterGray)
            for i in wordImgList:
                charfilename="./result/characters/" + str(counter) +".jpg";
                cv.imwrite("./result/characters/" + str(counter) +".jpg",i)

                detectPersonality(charfilename);
                
                counter=counter+1
            
    ###---------------------------------------------------------------------------#####        
            
        print('Original Image')         
        plt.imshow(img)
        plt.title("Original image")
        plt.show()

    except Exception as e:
        print ('Error Message ',e)
        cv.destroyAllWindows()
        traceback.print_exc()
        pass

    traceback.print_exc()
    processPerResults()



#Taking any image from the sample images
#In case of slanted image, straighten it using image-straighten.py, then use it
def doDetect():
    global result_dict,result_count,h,w,letterGray,colcnt
    filenametoclassify = fd.askopenfilename()
    img = cv.imread(filenametoclassify)
    image = cv2.imread('C:/Desktop/img.png', flags=cv2.IMREAD_COLOR)
    np.set_printoptions(threshold=np.inf)
    kernel = np.array([[0, -1, 0],
                   	[-1, 5, -1],
                  	 [0, -1, 0]])
    image_sharp = cv2.filter2D(src=gray, ddepth=-1, kernel=kernel)
    cv2.imwrite('C:/Desktop/1.png',image_sharp)
    medBlur = cv2.medianBlur(img,5)
    cv2.imwrite('C:/Desktop/2.png',medBlur)
    result_dict = {}
    result_count=0
    h=0
    w=0
    letterGray=0
    colcnt=0
    findPersonality(img)


if __name__ == "__main__":
    global e1,e2,label1,label2,label3,lable4,label5
    parent = tk.Tk()
    imga=Image.open('C:\\Users\\DELL\\Desktop\\codee\\code\\images\\b.png')
    parent.title("Graphology")
    bga = ImageTk.PhotoImage(imga)
    parent.geometry('800x600')
    label1=Label(parent,image=bga)
    label1.place(x=450,y=20)
    label2=Label(parent,image=bga)
    label2.place(x=450,y=550)
    frame = tk.Frame(parent)
    frame.pack(padx="6", pady="200")

    w = tk.Label(frame, text="PREDICT YOUR PERSONALITY FROM HANDWRITING!",
                     fg = "black",
                     bg = "#856ff8",
                     font = ("Algerian",26))
    w.pack()

    text_disp= tk.Button(frame,
                       text="BROWSE HANDWRITING",
                       fg = "red",
                       bg = "green",
                       font = ("Algerian",16),
                       command=doDetect
                       )
    text_disp.pack(padx="6", pady="50")

    label1 = tk.Label(frame, fg="red")
    label1.pack()    

    label2 = tk.Label(frame, fg="red")
    label2.pack()

    label3 = tk.Label(frame, fg="red")
    label3.pack()

    label4 = tk.Label(frame, fg="red")
    label4.pack()

    label5 = tk.Label(frame, fg="red")
    label5.pack()

    parent.mainloop()
    


    

    




