#%%
import cv2
import pytesseract
from PIL import Image
text = pytesseract.image_to_string(Image.open('img1.png'))
print(text)
#%%
