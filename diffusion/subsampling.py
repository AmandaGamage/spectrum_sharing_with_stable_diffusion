
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from preprocessing import split_data as sd

class Populate(Dataset):
    def dataloader(root,labels_file_path):
        data_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
        dataset = sd.DatasetMapper(root, labels_file_path, transform=data_transform)
        return dataset


# Usage example:

#dataset=Populate.dataloader(root="E:\\Msc\\Lab\\sd\\spectrogram_numpy\\spectrogram\\data_dir",labels_file_path="E:\\Msc\\Lab\\sd\\spectrogram_numpy\\spectrogram\\labels_1.txt")
#dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=32)
#for batch in dataloader:
 #   pass



