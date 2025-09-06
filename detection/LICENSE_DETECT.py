import os
import cv2
import numpy as np
import sys
import glob
import random
import importlib.util
from tflite_runtime.interpreter import Interpreter


import matplotlib
import matplotlib.pyplot as plt
import time
import csv

def tflite_detect_images(modelpath,lblpath,imgpath,min_conf,savepath=os.path.join('SaveImg','LICENCE','cpu2'),txt_only='False'):
    images=glob.glob(imgpath + './*jpg')
    labels=['licence']

   # with open(lblpath,'r') as f:
        
    #    labels=[line.strip() for line in f.readlines()]
    interpreter=Interpreter(model_path=modelpath)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    float_input=(input_details[0]['dtype']==np.float32)
    input_mean=127.5
    input_std=127.5
    k=0
    #images_to_test = random.sample(images, num_test_images)
    for m in range(179):
        k=k+1
        image = cv2.imread(imgpath+'/'+str(k)+'.jpg')
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imH, imW, _ = image.shape
        image_resized = cv2.resize(image_rgb, (width, height))
        input_data = np.expand_dims(image_resized, axis=0)
        if float_input:
            input_data = (np.float32(input_data) - input_mean) / input_std
        init=time.time()
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()
        time1=time.time()-init
        boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
        #print(boxes)
        classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects
        detections=[]
        for i in range(len(scores)):
            if ((scores[i] > min_conf) and (scores[i] <= 1.0)):

              # Get bounding box coordinates and draw box
              # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

                detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])

                # "text_only" controls whether we want to display the image results or just save them in .txt files
#         # Get filenames and paths
                image_fn = os.path.basename(imgpath+'/'+str(k)+'.jpg')
                base_fn, ext = os.path.splitext(image_fn)
                txt_result_fn = base_fn +'.txt'
                txt_savepath = os.path.join(savepath+'/'+str(k)+'.txt')
        with open(os.path.join(savepath,'SLicence.csv'),'a') as file:
            for detection in detections:
                writer=csv.writer(file)
                writer.writerow([k,detection[0],detection[1],time1])
                break
            
        # Write results to text file
        # (Using format defined by https://github.com/Cartucho/mAP, which will make it easy to calculate mAP)
        with open(txt_savepath,'w') as f:
             
            for detection in detections:
                f.write('%s %.4f %d %d %d %d\n' % (detection[0], detection[1], detection[2], detection[3], detection[4], detection[5]))        
#         image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        #plt.figure(figsize=(12,16))
        #plt.imshow(image)
        #plt.savefig(os.path.join("/home/jetson/test/tflite1/mAP/SaveImg/LICENCE",str(k)+"_cpu"+".jpg"),bbox_inches="tight")
        #plt.show()
    return


savepath=os.path.join('SaveImg','LICENCE','cpu2')
PATH_TO_IMAGES=os.path.join('scripts','extra','FINAL_TEST')
PATH_TO_MODEL=os.path.join('tflite','Licence_FINAL_16.tflite')
PATH_TO_LABEL=os.path.join('tfliteL.txt')
min_conf_threshold=0.0
#images_to_test=10
tflite_detect_images(PATH_TO_MODEL,PATH_TO_LABEL,PATH_TO_IMAGES,min_conf_threshold)








