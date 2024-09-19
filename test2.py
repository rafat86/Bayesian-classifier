import os
import cv2
import numpy as np
import random
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

def convert_to_ycbcr(images):
    ycbcr_images = [cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) for img in images]
    return ycbcr_images

def ensure_binary_mask(mask):
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary_mask

def build_histograms(images, masks, bins=256):
    skin_hist = np.zeros((bins, bins), dtype=np.float32)
    non_skin_hist = np.zeros((bins, bins), dtype=np.float32)
    
    for img, mask in zip(images, masks):
        mask = ensure_binary_mask(mask)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                cb = img[i, j, 1]
                cr = img[i, j, 2]
                if mask[i, j] > 0:
                    skin_hist[cb, cr] += 1
                else:
                    non_skin_hist[cb, cr] += 1

    skin_hist /= np.sum(skin_hist)
    non_skin_hist /= np.sum(non_skin_hist)
    
    return skin_hist, non_skin_hist

def classify_pixel(cb, cr, skin_hist, non_skin_hist):
    if skin_hist[cb, cr] > non_skin_hist[cb, cr]:
        return 1
    else:
        return 0

def classify_image(image, skin_hist, non_skin_hist):
    classified_img = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            cb = image[i, j, 1]
            cr = image[i, j, 2]
            classified_img[i, j] = classify_pixel(cb, cr, skin_hist, non_skin_hist)
    return classified_img

def compute_accuracies(true_masks, pred_masks):
    skin_accuracies = []
    non_skin_accuracies = []
    overall_accuracies = []
    
    for true_mask, pred_mask in zip(true_masks, pred_masks):
        true_mask = ensure_binary_mask(true_mask)
        skin_true = (true_mask.flatten() == 255)
        non_skin_true = (true_mask.flatten() == 0)
        
        skin_pred = (pred_mask.flatten() == 1)
        non_skin_pred = (pred_mask.flatten() == 0)
        
        skin_acc = accuracy_score(skin_true, skin_pred)
        non_skin_acc = accuracy_score(non_skin_true, non_skin_pred)
        overall_acc = accuracy_score(true_mask.flatten(), pred_mask.flatten())
        
        skin_accuracies.append(skin_acc)
        non_skin_accuracies.append(non_skin_acc)
        overall_accuracies.append(overall_acc)
        
    return np.mean(skin_accuracies), np.mean(non_skin_accuracies), np.mean(overall_accuracies)

# Paths to your training and testing data
train_images_folder = r'C:\\Users\\RAFAT\\Desktop\\MV06\\Skin_small\\Train\\images'
train_masks_folder = r'C:\\Users\\RAFAT\\Desktop\\MV06\\Skin_small\\Train\\masks'
test_images_folder = r'C:\\Users\\RAFAT\\Desktop\\MV06\\Skin_small\\Test\\images'
test_masks_folder = r'C:\\Users\\RAFAT\\Desktop\\MV06\\Skin_small\\Test\\masks'

# Load images and masks
train_images = load_images(train_images_folder)
train_masks = load_images(train_masks_folder)
test_images = load_images(test_images_folder)
test_masks = load_images(test_masks_folder)

# Ensure all masks are binary and match dimensions of corresponding images
train_masks = [ensure_binary_mask(cv2.resize(mask, (img.shape[1], img.shape[0]))) for img, mask in zip(train_images, train_masks)]
test_masks = [ensure_binary_mask(cv2.resize(mask, (img.shape[1], img.shape[0]))) for img, mask in zip(test_images, test_masks)]

# Convert images to YCbCr
train_ycbcr_images = convert_to_ycbcr(train_images)
test_ycbcr_images = convert_to_ycbcr(test_images)

# Build histograms from training data
skin_hist, non_skin_hist = build_histograms(train_ycbcr_images, train_masks)

# Classify training images
train_pred_masks = [classify_image(img, skin_hist, non_skin_hist) for img in train_ycbcr_images]

# Classify testing images
test_pred_masks = [classify_image(img, skin_hist, non_skin_hist) for img in test_ycbcr_images]

# Compute accuracies
train_skin_acc, train_non_skin_acc, train_overall_acc = compute_accuracies(train_masks, train_pred_masks)
test_skin_acc, test_non_skin_acc, test_overall_acc = compute_accuracies(test_masks, test_pred_masks)

# Print accuracies
print(f'Training Skin Accuracy: {train_skin_acc:.2f}')
print(f'Training Non-Skin Accuracy: {train_non_skin_acc:.2f}')
print(f'Training Overall Accuracy: {train_overall_acc:.2f}')
print(f'Testing Skin Accuracy: {test_skin_acc:.2f}')
print(f'Testing Non-Skin Accuracy: {test_non_skin_acc:.2f}')
print(f'Testing Overall Accuracy: {test_overall_acc:.2f}')

# Display a random test image along with its mask and classified image
random_index = random.randint(0, len(test_images) - 1)
random_test_image = test_images[random_index]
random_test_mask = test_masks[random_index]
random_pred_mask = test_pred_masks[random_index]

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(random_test_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Ground Truth Mask')
plt.imshow(random_test_mask, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Predicted Mask')
plt.imshow(random_pred_mask, cmap='gray')
plt.axis('off')

plt.show()
