import numpy as np
import matplotlib.pyplot as plt
import pytesseract as pyt
import cv2
import time
import re
'''
LP_char=pyt.image_to_string(canny, config=custom_config)
final_char=re.sub(r'\W+', '', LP_char)
final_char
'''
'''
# code for displaying multiple images in one figure 
#import libraries 
import cv2 
from matplotlib import pyplot as plt 
  
# create figure 
fig = plt.figure(figsize=(16,8)) 
  
# setting values to rows and column variables 
rows = 1
columns = 4
image = cv2.imread('LP2.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = thresholding(gray)
opening = opening(gray)
canny = canny(gray)
# reading images 
Image1 = cv2.imread('LP2.png') 
custom_config = r'--oem 3 --psm 6'
# Adds a subplot at the 1st position 
fig.add_subplot(rows, columns, 1) 

LP_char=pyt.image_to_string(gray, config=custom_config)
final_char=re.sub(r'\W+', '', LP_char)
  
# showing image 
plt.imshow(gray,'gray') 
plt.axis('off') 
plt.title("Gray Conversion \n LPR result = 'SN66XMZ'",fontname="Times New Roman") 
  
# Adds a subplot at the 2nd position 
fig.add_subplot(rows, columns, 2) 
  
# showing image 
plt.imshow(thresh,'gray') 
plt.axis('off') 
plt.title("Binarization \n LPR result = 'SN66XMZ'") 
  
# Adds a subplot at the 3rd position 
fig.add_subplot(rows, columns, 3) 
  
# showing image 
plt.imshow(opening,'gray') 
plt.axis('off') 
plt.title("Dilation \n LPR result = 'SN66XHZ'") 
  
# Adds a subplot at the 4th position 
fig.add_subplot(rows, columns, 4) 
  
# showing image 
plt.imshow(canny,'gray') 
plt.axis('off') 
plt.title("Canny Edge Detection \n LPR result = 'SiGe' ")

plt.savefig('LPR_results.pdf', bbox_inches='tight')
'''
import cv2
import numpy as np

img = cv2.imread('LP2.png')

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)


#openn=opening(gray)

image = cv2.imread('LP2.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = thresholding(gray)
opening = opening(gray)
canny = canny(gray)

fig = plt.figure(figsize=(8,6))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(gray,cmap='gray')
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(opening,cmap='gray')
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(thresh,cmap='gray')
ax4 = fig.add_subplot(2,2,4)
ax4.imshow(canny,cmap='gray')

# Adding custom options
custom_config = r'--oem 3 --psm 6'
t1=time.time()
LP_char=pyt.image_to_string(gray, config=custom_config)
#txt=pyt.image_to_string(gray, config=custom_config)
final_char=re.sub(r'\W+', '', LP_char)
t2=time.time()
print('LPR result = ', final_char)
print('time taken = ', t2-t1)

plt.imshow(gray)

fig = plt.figure(figsize=(8,6))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(gray,cmap='gray')

ax3 = fig.add_subplot(2,2,3)
ax3.imshow(thresh,cmap='gray')

# Adding custom options
custom_config = r'--oem 3 --psm 6'
t1=time.time()
LP_char=pyt.image_to_string(gray, config=custom_config)
#txt=pyt.image_to_string(gray, config=custom_config)
final_char=re.sub(r'\W+', '', LP_char)
t2=time.time()
print('LPR result = ', final_char)
print('time taken = ', t2-t1)

fig = plt.figure(figsize=(8,6))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(gray,cmap='gray')

ax3 = fig.add_subplot(2,2,3)
ax3.imshow(thresh,cmap='gray')

# Adding custom options
custom_config = r'--oem 3 --psm 6'
t1=time.time()
LP_char=pyt.image_to_string(gray, config=custom_config)
#txt=pyt.image_to_string(gray, config=custom_config)
final_char=re.sub(r'\W+', '', LP_char)
t2=time.time()
print('LPR result = ', final_char)
print('time taken = ', t2-t1)


image = cv2.imread('LP3.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = thresholding(gray)
plt.imshow(gray)
# Adding custom options
custom_config = r'--oem 3 --psm 6'
t1=time.time()
LP_char=pyt.image_to_string(gray, config=custom_config)
#txt=pyt.image_to_string(gray, config=custom_config)
final_char=re.sub(r'\W+', '', LP_char)
t2=time.time()
print('LPR result = ', final_char)
print('time taken = ', t2-t1)

image = cv2.imread('LP8.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = thresholding(gray)
plt.imshow(gray)
# Adding custom options
custom_config = r'--oem 3 --psm 6'
t1=time.time()
LP_char=pyt.image_to_string(gray, config=custom_config)
#txt=pyt.image_to_string(gray, config=custom_config)
final_char=re.sub(r'\W+', '', LP_char)
t2=time.time()
print('LPR result = ', final_char)
print('time taken = ', t2-t1)


'''
# code for displaying multiple images in one figure 
  
#import libraries 
import cv2 
from matplotlib import pyplot as plt 
  
# create figure 
fig = plt.figure(figsize=(10,6)) 
  
# setting values to rows and column variables 
rows = 1
columns = 4
image = cv2.imread('LP2.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = thresholding(gray)
opening = opening(gray)
canny = canny(gray)
# reading images 
Image1 = cv2.imread('LP2.png') 

# Adds a subplot at the 1st position 
fig.add_subplot(rows, columns, 1) 
  
# showing image 
plt.imshow(gray) 
plt.axis('off') 
plt.title("First") 
  
# Adds a subplot at the 2nd position 
fig.add_subplot(rows, columns, 2) 
  
# showing image 
plt.imshow(thresh) 
plt.axis('off') 
plt.title("Second") 
  
# Adds a subplot at the 3rd position 
fig.add_subplot(rows, columns, 3) 
  
# showing image 
plt.imshow(opening) 
plt.axis('off') 
plt.title("Third") 
  
# Adds a subplot at the 4th position 
fig.add_subplot(rows, columns, 4) 
  
# showing image 
plt.imshow(canny) 
plt.axis('off') 
plt.title("Fourth")
'''
'''
image = cv2.imread('LP2.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = thresholding(gray)
#opening = opening(gray)
#canny = canny(gray)
'''
