from tkinter import *
from tkinter import filedialog as fd
from PIL import Image, ImageTk
import cv2 as cv
import glob




def match(img):
    #create the descriptor and matcher
    sift = cv.SIFT_create()
    matcher=cv.BFMatcher()
    #make our image gray
    query = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #detect blobs in our image
    query_keypoints,query_descriptors = sift.detectAndCompute(query, None)
    #initialize the matches dictionary
    train_dict={}
    matches_dict={}
    for filepath2 in glob.iglob("Cropped License Plates/*.jpg"):
        #read image from licence plates
        img=cv.imread(filepath2)
        train = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        #detect blobs on second image
        train_keypoints,train_descriptors = sift.detectAndCompute(train, None)
        #find matches between the two images
        matches=matcher.knnMatch(query_descriptors,train_descriptors,k=2)
        #ratio test (David Lowe)
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        #store how many matches we have in a dictionary
        train_dict[filepath2]=len(good)
        result = cv.drawMatchesKnn(query,query_keypoints,train,train_keypoints,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        #store the drawn matches in another dictionary
        matches_dict[filepath2]=result
    print(train_dict)
    if(train_dict[max(train_dict,key=train_dict.get)]>10): #seuil =10
        #find the image that we have the most matches with
        query_dict=max(train_dict,key=train_dict.get) #query_dict is gonna be a path
        #get our drawn matches
        final_result=matches_dict[query_dict]
        #show images
        train=cv.imread(query_dict)
        cv.imshow("train",train)
        cv.moveWindow("train", 200, 0)
        cv.resizeWindow("train",200,200)
        cv.imshow("sift",final_result)
        cv.moveWindow("sift", 0, 200)
    else:
        error=cv.imread("error.png")
        cv.imshow("error",error)




def loadImage():
    global filename
    filename = fd.askopenfilename() #we use this to load an image
    #read the image and show it 
    image=cv.imread(filename)  
    img = Image.open(filename)   
    test = ImageTk.PhotoImage(img)

    label1 = Label(image=test)
    label1.image = test  
    label1.pack()
    match(image) #find the best match
    
#app window
window = Tk()
window.geometry("600x300")
window.title("License plate SIFT matcher")
#loading button
bt_load=Button(window,text='Load image',command=loadImage)
bt_load.pack()

window.mainloop()