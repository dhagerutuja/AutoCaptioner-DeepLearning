import torch
from torch.utils.data import Dataset
import h5py
import json
import os
import torchvision.transforms as transforms


class ImageCaptioningDataset(Dataset):
    """
    PyTorch Dataset class for Image Captioning.
    """

    def __init__(self, data_directory, dataset_name, split_set, transform=None):
        """
        :param data_directory: Path where preprocessed data is stored
        :param dataset_name: Base name of processed datasets (e.g., 'coco_5_cap_per_img_5_min_word_freq')
        :param split_set: Data split - 'TRAIN', 'VAL', or 'TEST'
        :param transform: Transformations for image preprocessing
        """
        self.split = split_set.upper()
        assert self.split in {'TRAIN', 'VAL', 'TEST'}, "Invalid split name! Choose from 'TRAIN', 'VAL', or 'TEST'."

        # Open HDF5 file for images
        h5_file_path = os.path.join(data_directory, f"{self.split}_IMAGES_{dataset_name}.hdf5")
        self.h5_file = h5py.File(h5_file_path, 'r', libver='latest')
        self.images = self.h5_file['images']

        # Load captions and caption lengths
        with open(os.path.join(data_directory, f"{self.split}_CAPTIONS_{dataset_name}.json"), 'r') as j:
            self.captions = json.load(j)

        with open(os.path.join(data_directory, f"{self.split}_CAPLENS_{dataset_name}.json"), 'r') as j:
            self.caption_lengths = json.load(j)

        self.transform = transform if transform else self.default_transform()
        self.num_samples = len(self.captions)

    def __getitem__(self, index):
        """
        Fetches an image-caption pair.
        """
        image = torch.FloatTensor(self.images[index // len(self.captions[0])] / 255.0)
        if self.transform:
            image = self.transform(image)

        caption = torch.LongTensor(self.captions[index])
        caption_length = torch.LongTensor([self.caption_lengths[index]])

        if self.split == 'TRAIN':
            return image, caption, caption_length
        else:
            all_captions = torch.LongTensor(
                self.captions[((index // len(self.captions[0])) * len(self.captions[0])):
                              (((index // len(self.captions[0])) * len(self.captions[0])) + len(self.captions[0]))])
            return image, caption, caption_length, all_captions

    def __len__(self):
        return self.num_samples

    @staticmethod
    def default_transform():
        """
        Returns a default set of transformations for preprocessing.
        """
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),  # Augmentation for better generalization
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
