import cv2
import numpy as np
import os

def build_histogram(images_folder):
    histogram = np.zeros((256, 256))
    for filename in os.listdir(images_folder):
        image_path = os.path.join(images_folder, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image: {image_path}")
            continue
        image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        cb = image_ycrcb[:,:,1]
        cr = image_ycrcb[:,:,2]
        hist, _, _ = np.histogram2d(cb.ravel(), cr.ravel(), bins=(256, 256), range=[[0, 256], [0, 256]])
        histogram += hist
    return histogram

def bayesian_classifier(test_images_folder, skin_histogram, non_skin_histogram):
    skin_prior = len(os.listdir(r'C:\\Users\\RAFAT\\Desktop\\MV06\\Skin_small\\Train\\images')) / (len(os.listdir(r'C:\\Users\\RAFAT\\Desktop\\MV06\\Skin_small\\Train\\images')) + len(os.listdir(r'C:\\Users\\RAFAT\\Desktop\\MV06\\Skin_small\\Train\\masks')))
    non_skin_prior = 1 - skin_prior
    skin_likelihood = skin_histogram / np.sum(skin_histogram)
    non_skin_likelihood = non_skin_histogram / np.sum(non_skin_histogram)

    correct_skin = 0
    correct_non_skin = 0
    total_images = 0

    for filename in os.listdir(test_images_folder):
        image_path = os.path.join(test_images_folder, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image: {image_path}")
            continue
        image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        cb = image_ycrcb[:,:,1]
        cr = image_ycrcb[:,:,2]

        skin_posterior = skin_prior * skin_likelihood[cb.astype(int), cr.astype(int)]
        non_skin_posterior = non_skin_prior * non_skin_likelihood[cb.astype(int), cr.astype(int)]

        predicted_class = 'skin' if np.sum(skin_posterior) > np.sum(non_skin_posterior) else 'non_skin'
        actual_class = 'skin' if 'skin' in image_path else 'non_skin'

        if predicted_class == 'skin' and actual_class == 'skin':
            correct_skin += 1
        elif predicted_class == 'non_skin' and actual_class == 'non_skin':
            correct_non_skin += 1

        total_images += 1

    accuracy_skin = correct_skin / total_images * 100 if total_images > 0 else 0
    accuracy_non_skin = correct_non_skin / total_images * 100 if total_images > 0 else 0
    overall_accuracy = (correct_skin + correct_non_skin) / total_images * 100 if total_images > 0 else 0

    print(f"Accuracy for skin: {accuracy_skin:.2f}%")
    print(f"Accuracy for non-skin: {accuracy_non_skin:.2f}%")
    print(f"Overall accuracy: {overall_accuracy:.2f}%")

# Assuming train images are located in 'train' folder
skin_hist = build_histogram(r'C:\\Users\\RAFAT\\Desktop\\MV06\\Skin_small\\Train\\images')
non_skin_hist = build_histogram(r'C:\\Users\\RAFAT\\Desktop\\MV06\\Skin_small\\Train\\masks')
bayesian_classifier(r'C:\\Users\\RAFAT\\Desktop\\MV06\\Skin_small\\Test\\images', skin_hist, non_skin_hist)

