import cv2

def normalize_image_orientation(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_corrected = cv2.flip(img, 0)
    return img_corrected

def preprocess_image(image_path):
    normalized_image = normalize_image_orientation(image_path)
    target_width, target_height = 150, 150
    normalized_resized_image = cv2.resize(normalized_image, (target_width, target_height))
    return normalized_resized_image
