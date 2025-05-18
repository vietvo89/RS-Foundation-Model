import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
from utils import *
import math

def image_sharpen(image):
    # 4. Sharpening (Reduces blur, can amplify noise)
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    image = cv2.filter2D(image, -1, kernel)
    return image

def center_crop(img, dim):
	"""Returns center cropped image
	Args:
	img: image to be center cropped
	dim: dimensions (width, height) to be cropped
	"""
	width, height = img.shape[1], img.shape[0]

	# process crop width and height for max available dimension
	crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
	crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
	mid_x, mid_y = int(width/2), int(height/2)
	cw2, ch2 = int(crop_width/2), int(crop_height/2) 
	crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
	return crop_img

def scale_image(img, factor=1):
	"""Returns resize image by scale factor.
	This helps to retain resolution ratio while resizing.
	Args:
	img: image to be scaled
	factor: scale factor to resize
	"""
	return cv2.resize(img,(int(img.shape[1]*factor), int(img.shape[0]*factor)))

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    """
    Adjusts the brightness and contrast of an image.

    Args:
        image (numpy.ndarray): The input image.
        brightness (int): Brightness adjustment value (negative or positive).
        contrast (int): Contrast adjustment value (0.0 to 3.0, typically).

    Returns:
         numpy.ndarray: The adjusted image.
    """
    adjusted_image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    return adjusted_image

def display_images_with_bounding_box(img_nm=None, df=None, txt=False,zoom=1, id_label_dict=None):

    img_pth = str(image_fldr / img_nm)
    im = cv2.imread(img_pth)    # tried using the flag cv2.IMREAD_ANYDEPTH flag, because images are 24 bit, but that removed the color channels.
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    init_h, init_w, _ = im.shape
    zoom_h, zoom_w, = round(init_h/zoom), round(init_w/zoom)
    im = im[0:zoom_h, 0:zoom_w]
    dh, dw, _ = im.shape
    fig, axs = plt.subplots(figsize=(int(dh/200), int(dw/200)))
    selected_df = df[(df['IMAGE_ID']==img_nm)]
    
    # create a bounding box with the data & draw it
    for i, row in selected_df.iterrows():
        box_id, x_min, y_min, x_max, y_max = row['TYPE_ID'],row['XMIN'],row['YMIN'],row['XMAX'],row['YMAX']
        x_min, y_min, x_max, y_max = int(x_min), int(y_max), int(x_max), int(y_min)
        print(i, box_id,id_label_dict[box_id],x_min, y_min, x_max, y_max, x_max-x_min, y_min-y_max)
        cv2.rectangle(im, (x_min, y_min), (x_max, y_max), (0,255,0), thickness=math.ceil(3/zoom))
        if txt:
            font_size = math.ceil(2/zoom)
            font_thickness = math.ceil(2/zoom)
            x_p = x_min if x_min>0 else x_max
            y_p = y_max-10 if y_max-10>10 else y_min+50
            cv2.putText(im, f'{box_id}', (x_p, y_p), cv2.FONT_HERSHEY_SIMPLEX, font_size, (36,255,12), font_thickness )

    # Show image with bboxes
    axs.set_title(f"Image {img_nm}", fontsize = 12)
    axs.imshow(im)
    axs.set_axis_off()

    # Display all the images
    plt.tight_layout()
    plt.show()