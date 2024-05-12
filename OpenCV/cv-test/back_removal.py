import cv2
import numpy as np

def remove_background_with_grabcut(image_path, new_background_path, output_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Create a mask using GrabCut
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    rect = (50, 50, image.shape[1]-50, image.shape[0]-50)  # Define a rectangle around the person
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
    # Create a new mask where the probable foreground is marked as white
    new_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # Apply the mask to the original image
    segmented_person = image * new_mask[:, :, np.newaxis]
    
    # Load the new background image
    background = cv2.imread(new_background_path)
    
    # Resize the background to match the size of the segmented person
    background = cv2.resize(background, (image.shape[1], image.shape[0]))
    
    # Combine the person and the background
    final_image = background * (1 - new_mask[:, :, np.newaxis]) + segmented_person
    
    # Save the final image
    cv2.imwrite(output_path, final_image)

# Paths to input image, new background, and output image
input_image_path = 'images/1.jpg'
new_background_path = 'result/new_background.jpg'
output_image_path = 'result/output_image.jpg'

remove_background_with_grabcut(input_image_path, new_background_path, output_image_path)