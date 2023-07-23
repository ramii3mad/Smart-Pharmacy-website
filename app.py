#Import necessary libraries
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import tensorflow as tf
from keras_preprocessing.image import load_img
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from natsort import natsorted



#Pre processing functions
def thresholding(image):
    blurred_img = cv2.GaussianBlur(image.copy(), (17, 17), 0)
    gray_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    threshold_value = 255
    max_binary_value = cv2.THRESH_BINARY_INV
    threshold_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    block_size = 11
    constant = 2
    thresh_img = cv2.adaptiveThreshold(gray_img, threshold_value, threshold_type, max_binary_value, block_size, constant)
    return thresh_img

############################################################
def remove_header_footer(image):
    # Thresholding
    thresh_img = thresholding(image)

    # Find and Sort Contours for the full image
    contours, hierarchy = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])  # Sort vertically

    # Specific cutting contours
    img_width = image.shape[1]  # Width of the image
    contours_for_cut = []
    for ctr in sorted_contours:
        x, y, w, h = cv2.boundingRect(ctr)
        if w > (img_width * 0.80):
            contours_for_cut.append((x, y, w, h))

    # Crop contours from image
    cropped_img = image.copy()
    if len(contours_for_cut) > 1:
        x = 0
        w = cropped_img.shape[1]

        # Remove header
        h = cropped_img.shape[0]
        header_y = contours_for_cut[0][1]
        header_h = contours_for_cut[0][3]
        header = header_y + header_h
        y = header
        cropped_img = cropped_img[y:h, x:w]

        # Remove footer
        new_y = 0
        h_without_header = cropped_img.shape[0]
        footer_y = contours_for_cut[-1][1]
        footer_h = contours_for_cut[-1][3]
        footer = h - footer_y
        new_h = h_without_header - footer
        cropped_img = cropped_img[new_y:new_h, x:w]

        plt.imshow(cropped_img)
    
    return cropped_img

###############################################################

def mean_height_of_lines(sorted_contours_lines):
    sum_of_heights = 0
    for ctr in sorted_contours_lines:
        x, y, w, h = ctr
        sum_of_heights += h
    mean_of_heights = sum_of_heights / len(sorted_contours_lines)
    return mean_of_heights
##########################################################
def mean_space_between_lines(sorted_contours_lines):
    sum_of_spaces = 0
    i = 1
    for ctr in sorted_contours_lines:
        if i < len(sorted_contours_lines):
            x, y, w, h = ctr
            nx, ny, nw, nh = sorted_contours_lines[i]
            sum_of_spaces += ny - (y + h)
            i += 1
    mean_of_spaces = sum_of_spaces / (len(sorted_contours_lines) - 1)
    return mean_of_spaces
#########################################################
def contours_to_xywh(sorted_contours_lines_N):
  sorted_contours_lines = []
  for ctr in sorted_contours_lines_N:
    x,y,w,h = cv2.boundingRect(ctr)
    sorted_contours_lines.append((x,y,w,h))
  return(sorted_contours_lines)
#######################################
def remove_noise(img4, sorted_contours_lines, mean_of_heights):
  sorted_contours_lines_mean = []
  for ctr in sorted_contours_lines:
    x,y,w,h = ctr
    if(h < int(mean_of_heights/2)):
        continue
    cv2.rectangle(img4, (x,y), (x+w, y+h), (np.random.randint(255),np.random.randint(255),np.random.randint(255)), 5) #img, coordnate, area, color, bold
    sorted_contours_lines_mean.append((x,y,w,h))
  return sorted_contours_lines_mean
#########################################
def get_contours_line_segmentation(thresh_img):
    #line delation
  kernel = np.ones((5,200), np.uint8) #matrix of ones on shape 3*85 in dataType unsigned int 
  dilated = cv2.dilate(thresh_img, kernel, iterations = 1) #iteration is num of steps of kernal
  # plt.imshow(dilated, cmap='gray')

  #get contours
  (contours, heirarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  sorted_contours_lines_N = sorted(contours, key = lambda ctr : cv2.boundingRect(ctr)[1]) # (x, y, w, h) #1 mean sort vertically but 0 mean sort horizontally

  sorted_contours_lines = contours_to_xywh(sorted_contours_lines_N)

  return sorted_contours_lines
##########################################
def line_segmentation(thresh_img, cutted_img):

  sorted_contours_lines = get_contours_line_segmentation(thresh_img)
  #First Calling Of Mean Hight & Mean Space**(Before Delete Noise)
  mean_of_heights = mean_height_of_lines(sorted_contours_lines)
  mean_of_spaces = mean_space_between_lines(sorted_contours_lines)



  #remove noise and drow line segmentation on cutted_img
  sorted_contours_lines_mean = remove_noise(cutted_img, sorted_contours_lines, mean_of_heights)
  plt.imshow(cutted_img)

  return sorted_contours_lines_mean
####################################################
def spaceBetweenWords(word1, word2):
  x, y, w, h = cv2.boundingRect(word1)
  x2, y2, w2, h2 = cv2.boundingRect(word2)
  space=x2-(w+x)
  return space
###############################################
def average_x(sorted_contours_lines_mean):
  avgX = 0
  sumX = 0
  for line in sorted_contours_lines_mean:
    sumX += line[0]
  avgX = sumX/len(sorted_contours_lines_mean)
  return avgX
############################################
def word_segmentation(thresh_img, cutted_img, sorted_contours_lines_mean):
  kernel = np.ones((1,16), np.uint8)
  dilated2 = cv2.dilate(thresh_img, kernel, iterations = 4)
  plt.imshow(dilated2, cmap='gray')

  img3 =cutted_img.copy()
  img4 =cutted_img.copy()
  # words_list = []
  ln=0
  FistWordList = []
  FR=0
  avgX = average_x(sorted_contours_lines_mean)
  for line in sorted_contours_lines_mean:
    sorted_contour_words = []
    if line[0] > (avgX*0.5) :#line[0]=x #if exist more than word in exact line  لو الاكس كبيره يبقي السطر بادئ مش من اول السطر فمش عايزه
      continue
    FirstWordInLine=0
   
   
    x, y, w, h = line
    
    roi_line = dilated2[y:h+y, x:w+x]
   
    # draw contours on each word
    (cnt, heirarchy) = cv2.findContours(roi_line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contour_words_beforeFilter = sorted(cnt, key=lambda cntr : cv2.boundingRect(cntr)[0])#0 mean sort horizontally but 1 mean sort vertically  
    
    for word in sorted_contour_words_beforeFilter:
        if cv2.contourArea(word) < 400:
            continue
        sorted_contour_words.append(word)

    for word in sorted_contour_words:
        
        if FR==0 :
          firstR = word
          FR=1
        if (ln >= 0) and (FirstWordInLine == 0) and len(sorted_contour_words)>1:
            spaceBetweenWord=spaceBetweenWords(sorted_contour_words[0], sorted_contour_words[1])
            FistWordList.append(((x, y), word, spaceBetweenWord))
            FirstWordInLine = 1
            

        x2, y2, w2, h2 = cv2.boundingRect(word)
        # words_list.append([x+x2, y+y2, x+x2+w2, y+y2+h2])
        cv2.rectangle(img3, (x+x2, y+y2), (x+x2+w2, y+y2+h2), (np.random.randint(255),np.random.randint(255),np.random.randint(255)),6)
    ln+=1

  plt.imshow(img3)
  
  contours_R = []

  AreaR1 = cv2.contourArea(FistWordList[0][1]) #FistWordList[num of word][contours of word]##contour of first word in list

  if(AreaR1 < (cv2.contourArea(firstR)*3)) :
    for word in FistWordList:
      ##word -> word[0]= contours of line, word[1]=contour of word in the line, word[2]= space between first & second
      print(cv2.contourArea(word[1])) #word[1]=contour of word
      if (cv2.contourArea(word[1]) < (AreaR1+(AreaR1*1.5))  ):
        x2, y2, w2, h2 = cv2.boundingRect(word[1])
        #word is (x, y), word----->(x, y) of line of word ,,, word is contour of words 
        #word[1] = contour of word
        #word[0][0] = x of line,,,,word[0][1] = y of line
        cv2.rectangle(img4, (word[0][0]+x2, word[0][1]+y2), (word[0][0]+x2+w2, word[0][1]+y2+h2), (np.random.randint(255),np.random.randint(255),np.random.randint(255)),6)
        contours_R.append((word[0][0]+x2, word[0][1]+y2, w2, h2))
      
  plt.imshow(img4)
  return contours_R
################################################################
def avgRR(contors):
  avgR = 0
  sumR = 0  
  for R in contors:
    sumR += (R[0] + R[2]) #x+w
  avgR = sumR/len(contors)
  return avgR
##############################################
def remove_R(imgR, contours_R):
  avgR = avgRR(contours_R)
  x, y, w, h = contours_R[0]
  # if( (x+w) < avgR):
  # cv2.rectangle(imgR, (x, y), (x+w, y+h), (np.random.randint(255),np.random.randint(255),np.random.randint(255)),9)
  n_y=0
  n_h=imgR.shape[0]
  n_x=int(avgR)
  n_w = imgR.shape[1]
  im = imgR[n_y:n_h, n_x:n_w]

  plt.imshow(im)
  return im
###########################################
def final_word_detection(im):
  thresh_img = thresholding(im)
  kernel = np.ones((1,100), np.uint8)
  dilated5 = cv2.dilate(thresh_img, kernel, iterations = 1)
  plt.imshow(dilated5, cmap='gray')
  
  (contours, heirarchy) = cv2.findContours(dilated5.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  sorted_contours_Word_N = sorted(contours, key = lambda ctr : cv2.boundingRect(ctr)[1]) # (x, y, w, h) #1 mean sort vertically but 0 mean sort horizontally

  sorted_contours_Words = []
  for ctr in sorted_contours_Word_N:
    x,y,w,h = cv2.boundingRect(ctr)
    sorted_contours_Words.append((x,y,w,h))

  img4 = im.copy()
  sorted_contours_Words_mean = []
  for ctr in sorted_contours_Words:
    
    #ctr= sorted_contours_lines[4]
    x,y,w,h = ctr
    if (w * h) < 2000:
      continue
    
    cv2.rectangle(img4, (x,y), (x+w, y+h), (np.random.randint(255),np.random.randint(255),np.random.randint(255)), 5) #img, coordnate, area, color, bold
    sorted_contours_Words_mean.append((x,y,w,h))
  cv2.imwrite(f"static\Result.jpg", img4)
  return sorted_contours_Words_mean
#############################################
def run(original_img):
  cutted_img = remove_header_footer(original_img)
  thresh_img = thresholding(cutted_img)

  sorted_contours_lines_mean = line_segmentation(thresh_img.copy(), cutted_img.copy())
  #calc mean height and mean spaces after remove noise and line segmentation
  mean_of_heights = mean_height_of_lines(sorted_contours_lines_mean)
  mean_of_spaces = mean_space_between_lines(sorted_contours_lines_mean)

  contours_R = word_segmentation(thresh_img.copy(), cutted_img.copy(), sorted_contours_lines_mean)

  im = remove_R(cutted_img.copy(), contours_R)

  sorted_contours_Words_mean = final_word_detection(im.copy())


  crop_img = im.copy()
  i=0
#   %rm -rf /content/Words
  os.mkdir('/content/Words')
  for ctr in sorted_contours_Words_mean:
   
    x,y,w,h = ctr
    #x,y,w,h = cv2.boundingRect(ctr)
    word = crop_img[y:h+y, x:w+x]
    cv2.imwrite(f"/content/Words/Word_{i}.jpg", word)
    i+=1

#################################################################
def delete_folder(folder_path):
    try:
        # Delete the folder and its contents
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' deleted successfully.")
    except FileNotFoundError:
        print(f"Folder '{folder_path}' does not exist.")
    except Exception as e:
        print(f"Error occurred while deleting folder '{folder_path}': {e}")












# Create flask instance
app = Flask(__name__)

#Set Max size of file as 10MB.
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# Categories
Categories = ['cataflam', 'ketolac', 'brufen', 'panadol', 'Actos', 'Diclac', 'Insulin']

#Allow files with extension png, jpg and jpeg
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init():
    global graph
    graph = tf.compat.v1.get_default_graph()

# Function to load and prepare the image in right shape
from PIL import Image
# from keras.preprocessing.image import img_to_array

def read_image(filename):
    # Load the image
    img = Image.open(filename)
    img = img.resize((224, 224))  # Resize if necessary
    # Convert the image to array
    img = img_to_array(img)
    # Reshape the image into a sample of 1 channel
    img = img.reshape(1, 224, 224, 3)
    # Prepare it as pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img




@app.route("/", methods=['GET', 'POST'])
def home():

    return render_template('home.html')

@app.route("/predict", methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        try:
            print('in try')
            if file and allowed_file(file.filename):
                print('True if')
                filename = file.filename
                file_path = os.path.join('static\images', filename)
                print('file_path section = ', file_path)

                original_img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                print('true read')
                
               
                
                cutted_img = remove_header_footer(original_img)
                thresh_img = thresholding(cutted_img)
                sorted_contours_lines_mean = line_segmentation(thresh_img.copy(), cutted_img.copy())
                # sorted_contours_lines = get_contours_line_segmentation(thresh_img)
                #First Calling Of Mean Hight & Mean Space**(Before Delete Noise)
                mean_of_heights = mean_height_of_lines(sorted_contours_lines_mean)
                mean_of_spaces = mean_space_between_lines(sorted_contours_lines_mean)

                contours_R = word_segmentation(thresh_img.copy(), cutted_img.copy(), sorted_contours_lines_mean)
                
                im = remove_R(cutted_img.copy(), contours_R)

                sorted_contours_Words_mean = final_word_detection(im.copy())

                crop_img = im.copy()
                i=0
                # %rm -rf /content/Words
                delete_folder('static\Words')
                delete_folder('static\Result.jpg')
                os.mkdir('static\Words')
                print('os done')
                for ctr in sorted_contours_Words_mean:
                
                  x,y,w,h = ctr
                  #x,y,w,h = cv2.boundingRect(ctr)
                  word = crop_img[y:h+y, x:w+x]
                  cv2.imwrite(f"static\Words\Word_{i}.jpg", word)
                  i+=1


            
            




                file.save(file_path)
                print('success saving img')

                with graph.as_default():
                  model1 = load_model('model.h5')
                  print('success loading model')
                  label=[]
                  for images in natsorted(os.listdir('static\Words')):
                     print(images)
                     img = read_image(f"static\Words\{images}")
                     print('success read img')
                     class_prediction = model1.predict(img)
                     print('class prediction: ',class_prediction[0])
                     predictions = np.argmax(class_prediction, axis=-1)
                     label.append(Categories[predictions[0]])
                 
                
                  return render_template('predict.html', product=label, user_image=file_path, user_image1='static\Result.jpg')

        except Exception as e:
            return "Unable to read the file. Please check if the file extension is correct."

    return render_template('predict.html')

if __name__ == "__main__":
    init()
    app.run()