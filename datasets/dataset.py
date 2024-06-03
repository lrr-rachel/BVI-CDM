import os
import torch
import torch.utils.data
from torchvision.transforms.functional import to_tensor
import PIL
from PIL import Image
import re
from datasets.data_augment import PairCompose, PairRandomCrop, PairToTensor
import numpy as np
import cv2


class LLdataset:
    def __init__(self, config):
        self.config = config

    def get_loaders(self):
        print("=> LLdataset using dataset '{}'".format(self.config.data.val_dataset))
        train_dataset = AllWeatherDataset(self.config.data.train_data_dir,
                                          patch_size=self.config.data.patch_size,
                                          filelist=os.path.join(self.config.data.file_list_path, '{}_train.txt'.format(self.config.data.train_dataset)),
                                          train=True
                                          )
        val_dataset = AllWeatherDataset(self.config.data.val_data_dir,
                                        patch_size=self.config.data.patch_size,
                                        filelist=os.path.join(self.config.data.file_list_path,
                                                              '{}_val.txt'.format(self.config.data.val_dataset)),
                                        train=False)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                                 num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader


class AllWeatherDataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, filelist=None, train=True):
        super().__init__()

        self.dir = dir
        self.train = train
        self.file_list = filelist
        self.train_list = self.file_list
        with open(self.train_list) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents]
            gt_names = [i.strip() for i in contents]

        self.input_names = input_names
        self.gt_names = gt_names
        self.patch_size = patch_size
        if self.train:
            self.transforms = PairCompose([
                PairRandomCrop(self.patch_size),
                PairToTensor()
            ])
        else:
            self.transforms = PairCompose([
                PairToTensor()
            ])

    def get_images(self, index):
        input_name = self.input_names[index].replace('\n', '')
        gt_name = self.gt_names[index].replace('\n', '')
        img_id = re.split('/', input_name)[-1][:-4]
        input_img = Image.open(os.path.join(self.dir, input_name)) if self.dir else PIL.Image.open(input_name)
        gt_img = Image.open(os.path.join(self.dir, gt_name)) if self.dir else PIL.Image.open(gt_name)

        input_img, gt_img = self.transforms(input_img, gt_img)

        return torch.cat([input_img, gt_img], dim=0), img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)

class VIDEOdataset:
    def __init__(self, config):
        self.config = config

    def get_loaders(self):
        print("=> VIDEOdataset using dataset '{}'".format(self.config.data.val_dataset))
        supporting_frames = self.config.data.supporting_frames
        train_dataset = LowLightVideoDataset(self.config.data.data_dir,
                                             supporting_frames,
                                             patch_size=self.config.data.patch_size,
                                             filelist=os.path.join(self.config.data.file_list_path, '{}_input.txt'.format(self.config.data.train_dataset)),
                                             train=True
                                          )
        val_dataset = LowLightVideoDataset(self.config.data.data_dir, supporting_frames,
                                        patch_size=self.config.data.patch_size,
                                        filelist=os.path.join(self.config.data.file_list_path,
                                                              '{}_test.txt'.format(self.config.data.val_dataset)),
                                        train=False)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                                 num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader

class LowLightVideoDataset(torch.utils.data.Dataset):
    def __init__(self, dir, supporting_frames, patch_size, filelist=None, train=True):
        super().__init__()

        self.dir = dir
        self.train = train
        self.file_list = filelist
        self.train_list = self.file_list
        self.supporting_frames = supporting_frames
        with open(self.train_list) as f:
            contents = f.readlines()
            input_names = [i for i in contents]
        self.input_names = input_names
        self.patch_size = patch_size
        if self.train:
            self.transforms = PairCompose([
                PairRandomCrop(self.patch_size),
                # PairToTensor()
            ])
        else:
            self.transforms = PairCompose([
                # PairToTensor()
            ])

    def get_images(self, index):
        # Get the full path of the current frame from input names
        current_input_path = self.input_names[index].replace('\n', '')
        current_input_path = os.path.join(self.dir, current_input_path)

        # Extract the directory, filename, and frame number for the input image
        input_directory, input_filename = os.path.split(current_input_path)
        frame_number = int(re.search(r'(\d+).png', input_filename).group(1))
        input_frames = []

        # Generate filenames for the two frames before, the current frame, and two frames after
        for i in range(frame_number - 2, frame_number + 3):  # From frame_number-2 to frame_number+2
            # Construct the new filename with adjusted frame number, preserving leading zeros
            new_frame_number = f"{i:05d}"  # New frame number, zero-padded
            new_input_filename = f"{new_frame_number}.png"
            # Construct the full path for the new input filename
            new_input_path = os.path.join(input_directory, new_input_filename)
            # Load the input image
            input_img = Image.open(new_input_path)
            # input_he = self.histogram_equalization_colour(input_img, self.calculate_average_brightness(np.array(input_img)))
            input_frames.append(input_img)
        # Get the full path of the current frame from GT names
        current_gt_path = current_input_path.replace('input', 'gt').replace('low_light_10', 'normal_light_10').replace('low_light_20', 'normal_light_20')
        # Load the GT image
        gt_img = Image.open(current_gt_path)

        # Stack all input frames along the channel dimension
        input_frames = [to_tensor(frame) for frame in input_frames]
        # Apply transformations to each input frame and the GT image (assuming 'self.transforms' is defined)
        input_imgs = torch.cat(input_frames, dim=0)
        input_imgs, gt_img = self.transforms(input_imgs, gt_img)

        # Stack the stacked input frames with the GT image
        gt_img = to_tensor(gt_img)
        stacked_input_gt = torch.cat([input_imgs, gt_img], dim=0)

        # Extract the image ID from the input filename
        img_id = input_filename[:-4]
        # stacked_input_gt: (c, h, w) - (18, h, w)
        H, W = stacked_input_gt.shape[1], stacked_input_gt.shape[2]  # Extract height and width
        # If the height can't be divided by 128, crop to the largest possible size
        if H % 128 != 0:
            new_H = H // 128 * 128  # Calculate the new height
            stacked_input_gt = stacked_input_gt[:, :new_H, :]  # Apply cropping

        # If the width can't be divided by 128, crop to the largest possible size
        if W % 128 != 0:
            new_W = W // 128 * 128  # Calculate the new width
            stacked_input_gt = stacked_input_gt[:, :, :new_W]  # Apply cropping

        return stacked_input_gt, img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)