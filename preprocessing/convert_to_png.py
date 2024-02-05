import numpy as np
import os
import matplotlib.pyplot as plt
import torch

class convert_png:
    def converting_and_saving_pngs(self,file_path,output_directory):
        loaded_array = np.load(file_path)
        os.makedirs(output_directory, exist_ok=True)

        for image_no, image_array in enumerate(loaded_array, start=1):
            plt.imshow(image_array, cmap='Greys')
            plt.axis('off')
            output_png_path = os.path.join(output_directory, f'chirp_image_{image_no}.png')
            plt.savefig(output_png_path, bbox_inches='tight', pad_inches=0)
            plt.clf()  # Clear the current figure to release memory

        plt.close()
        #file_path = 'E://Msc//Lab//sd//spectrogram_numpy//spectrogram//Chirp_1.npy'
        #output_directory = 'E:\\Msc\\Lab\\sd\\spectrogram_numpy\\spectrogram\\spectrogram images'
        
#test cuda available        
print(torch.cuda.is_available())