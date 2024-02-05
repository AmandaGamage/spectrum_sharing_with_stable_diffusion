import numpy as np
import matplotlib.pyplot as plt
import convert_to_png 
import datasets
from datasets import load_dataset
import accelerate
from accelerate.utils import write_basic_config
import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from skimage.metrics import structural_similarity as ssim

write_basic_config()


#chirp 1
file_path_chirp2 = 'E://Msc//Lab//sd//spectrogram_numpy//spectrogram//Chirp_1.npy'
output_directory = 'E:\\Msc\\Lab\\sd\\spectrogram_numpy\\spectrogram\\spectrogram images new'


def convert_to_png_function(file_path,output):
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
image_folder = "E:\\Msc\\Lab\\sd\\spectrogram_numpy\\spectrogram\\data_dir"
output_file = "E:\\Msc\\Lab\\sd\\spectrogram_numpy\\spectrogram\\labels_1.txt"

# Load sample images for comparison
radar_samples = [cv2.imread(f"E:\\Msc\\Lab\\sd\\spectrogram_numpy\\spectrogram\\data_dir_samples\\radar_sample_{i}.png") for i in range(1, 26)]  
empty_samples = [cv2.imread(f"E:\\Msc\\Lab\\sd\\spectrogram_numpy\\spectrogram\\data_dir_samples\\empty_sample_{i}.png") for i in range(1,  23)]  

# Create an empty list to store labels
labels = []

def sample_feature_extractor(samples):
    sample_features=[]
    for sample in samples:
        features=extract_features(sample)
        sample_features.append(features)
    return sample_features

radar_sample_features=sample_feature_extractor(radar_samples)
empty_sample_features=sample_feature_extractor(empty_samples)

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
        labels.append(f"{filename} {label}\n")

# Save labels to the text file
with open(output_file, "w") as f:
    f.writelines(labels)

print(f"Labels saved to {output_file}")



    


