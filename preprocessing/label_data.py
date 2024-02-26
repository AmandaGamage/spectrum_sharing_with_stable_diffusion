import numpy as np
#import convert_to_png 
#import datasets
#from datasets import load_dataset
#from accelerate.utils import write_basic_config
import cv2
import os
import random
import shutil
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#from skimage.metrics import structural_similarity as ssim

#write_basic_config()


#chirp 1
#file_path_chirp2 = 'E://Msc//Lab//sd//spectrogram_numpy//spectrogram//Chirp_1.npy'
#output_directory = 'E:\\Msc\\Lab\\sd\\spectrogram_numpy\\spectrogram\\spectrogram images new'


'''def convert_to_png_function(file_path,output):
    #convert to png
    convert_to_png_obj=convert_to_png.convert_png()
    convert_to_png_obj.converting_and_saving_pngs(file_path,output)

    dataset = load_dataset("imagefolder", data_dir=output)

    return datasets




def radar_check_function(image,templates):
    pattern_found = False

    # Iterate over each template
    for template_path in templates:
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        
        if image.shape[-1] != template.shape[-1]:
            # If the number of channels is different, convert the template to grayscale
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        h, w = template.shape
        image = cv2.resize(image, (w, h))

        # Perform template matching
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

        # Set a threshold to determine if the pattern is present
        threshold = 0.8
        locations = cv2.findNonZero((result >= threshold).astype(np.uint8))

        # If any locations are found, set the flag to True
        if len(locations[0]) > 0:
            pattern_found = True
            break  # Exit the loop once any pattern is found


    return pattern_found


def extract_features(img):
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()  # Normalize for comparison
    return hist


def compare_features(features1, features2):

    if isinstance(features1, np.ndarray) and isinstance(features2, np.ndarray):
        distance = np.linalg.norm(features1 - features2)
        similarity = 1 - (distance / np.sqrt(2))  
        return similarity

    if isinstance(features1, np.ndarray) and np.all(features1 >= 0):  
        similarity = np.sum(np.minimum(features1, features2))
        return similarity
    else:
        raise ValueError("Unsupported feature types for comparison.")

# Specify paths
#image_folder = "E:\\Msc\\Lab\\sd\\spectrogram_numpy\\spectrogram\\data_dir"
#output_file = "E:\\Msc\\Lab\\sd\\spectrogram_numpy\\spectrogram\\labels_1.txt"

# Load sample images for comparison
#radar_samples = [cv2.imread(f"E:\\Msc\\Lab\\sd\\spectrogram_numpy\\spectrogram\\data_dir_samples\\radar_sample_{i}.png") for i in range(1, 26)]  
#empty_samples = [cv2.imread(f"E:\\Msc\\Lab\\sd\\spectrogram_numpy\\spectrogram\\data_dir_samples\\empty_sample_{i}.png") for i in range(1,  23)]  

# Create an empty list to store labels
#labels = []

def sample_feature_extractor(samples):
    sample_features=[]
    for sample in samples:
        features=extract_features(sample)
        sample_features.append(features)
    return sample_features

#radar_sample_features=sample_feature_extractor(radar_samples)
#empty_sample_features=sample_feature_extractor(empty_samples)

# Iterate through images in the folder
for filename in os.listdir(image_folder):
    if filename.endswith(".png"):  # Check for png extension
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)

        # Extract features based on your chosen approach
        features = extract_features(img)  # Replace with your implementation

        # Calculate average similarity scores across all samples
        similarity_score_radar=[]
        for sample_features in radar_sample_features:
            similarity_score_radar.append(compare_features(features, sample_features))
        similarity_to_radar=sum(similarity_score_radar)/len(similarity_score_radar)

        similarity_score_empty=[]
        for sample_e in empty_sample_features:
            similarity_score_empty.append(compare_features(features, sample_e))
        similarity_to_empty=sum(similarity_score_empty)/len(similarity_score_empty)

        # Determine label based on average similarity scores
        if similarity_to_radar > similarity_to_empty:
            label = "radar"
        else:
            label = "empty"

        # Append label to the list
        labels.append(f"{filename} {label}\n")'''

# Save labels to the text file
#with open(output_file, "w") as f:
 #   f.writelines(labels)

#print(f"Labels saved to {output_file}")


# Path to the folder containing class folders
root_folder = 'E:\\Msc\\Lab\\spectrum_sharing_data\\labeled_data'
# Path to the folder where images will be moved
#output_folder = 'E:\\Msc\\Lab\\spectrum_sharing_data\\all_data'
# Path to the output text file for labels
label_file = 'E:\\Msc\\Lab\\spectrum_sharing_data\\labels.txt'

def move_and_rename_images(root_folder, output_folder, label_file):
    with open(label_file, 'w') as f:
        image_counter = 1
        for class_folder in os.listdir(root_folder):
            class_path = os.path.join(root_folder, class_folder)
            if os.path.isdir(class_path):
                images_in_class = os.listdir(class_path)
                random.shuffle(images_in_class)  # Shuffle images in each class folder
                for img_file in images_in_class:
                    img_path = os.path.join(class_path, img_file)
                    # Rename image with prefix of class folder name and image count
                    new_filename = f"{class_folder}_{image_counter}.jpg"
                    new_img_path = os.path.join(output_folder, new_filename)
                    shutil.copyfile(img_path, new_img_path)
                    # Write label to text file
                    f.write(f" {class_folder}\n")
                    image_counter += 1


# Create the output folder if it doesn't exist
#if not os.path.exists(output_folder):
 #   os.makedirs(output_folder)

# Move and rename images and write labels
#move_and_rename_images(root_folder, output_folder, label_file)
#########################################################################################
#shuffling the images

'''labels_file='E:\\Msc\\Lab\\spectrum_sharing_data\\labels.txt'
output_folder = 'E:\\Msc\\Lab\\spectrum_sharing_data\\all_shuffled_data'
images_folder= 'E:\\Msc\\Lab\\spectrum_sharing_data\\all_data'


# Read labels
with open(labels_file, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Generate image filenames based on the number of labels
image_filenames = [f'image_{i+1}.jpg' for i in range(len(labels))]

# Shuffle images and labels
combined = list(zip(image_filenames, labels))
random.shuffle(combined)
shuffled_image_filenames, shuffled_labels = zip(*combined)

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Write shuffled labels
shuffled_labels_file = os.path.join(output_folder, 'shuffled_labels.txt')
with open(shuffled_labels_file, 'w') as f:
    for label in shuffled_labels:
        f.write(f"{label}\n")

# Copy and rename shuffled images
for i, image_filename in enumerate(shuffled_image_filenames):
    image_path = os.path.join(images_folder, random.choice(os.listdir(images_folder))) # pick a random image from the folder
    new_image_path = os.path.join(output_folder, image_filename)
    shutil.copyfile(image_path, new_image_path)

print("Images and labels shuffled successfully!")'''
####################################################################################
image_folder = "E:\\Msc\\Lab\\spectrum_sharing_data\\all_shuffled_data"

# Path to the labels.txt file
labels_file = "E:\\Msc\\Lab\\spectrum_sharing_data\\shuffled_labels.txt"

# New file to save modified labels
new_labels_file = "E:\\Msc\\Lab\\spectrum_sharing_data\\all_shuffled_data\\new_shuffled_labels.txt"
'''
# Read labels from the existing file
with open(labels_file, 'r') as file:
    labels = file.readlines()

# Strip newline characters and split by space
labels = [label.strip().split() for label in labels]

# Get list of image file names in the image folder
image_files = os.listdir(image_folder)

# Combine image paths with labels
label_with_paths = [(os.path.join(image_folder, image_files[i]), labels[i][0]) for i in range(len(image_files))]

# Write combined labels to new file
with open(new_labels_file, 'w') as file:
    for label, path in label_with_paths:
        file.write(f"{path} {label}\n")

print("New labels file generated successfully!")
    


'''
import os
import random

# Path to the directory containing the class folders
data_dir = "E:\\Msc\\Lab\\spectrum_sharing_data\\one_channel_labeled_data"

# Path to the directory where you want to move all images
output_dir = "E:\\Msc\\Lab\\spectrum_sharing_data\\one_channel_all_labeled_data"

# Initialize an empty list to store image paths and corresponding labels
image_label_pairs = []

# Iterate over each class folder
for class_name in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_name)
    
    # Skip if the entry is not a directory
    if not os.path.isdir(class_dir):
        continue
    
    # Iterate over each image in the class folder
    for filename in os.listdir(class_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Assuming images are either JPG or PNG
            # Store the image path and corresponding label (class name)
            image_path = os.path.join(class_dir, filename)
            image_label_pairs.append((image_path, class_name))

# Shuffle the image_label_pairs list
random.shuffle(image_label_pairs)

# Move the images to the output directory, rename them, and write labels to a text file
with open(os.path.join(output_dir, "labels.txt"), "w") as f:
    for idx, (image_path, label) in enumerate(image_label_pairs, start=1):
        # Move the image to the output directory and rename it
        dst = os.path.join(output_dir, f"image_{idx}.jpg")
        os.rename(image_path, dst)
        
        # Write the label and new image path to the labels file
        f.write(f"{label} image_{idx}.jpg\n")
