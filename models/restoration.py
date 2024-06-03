import torch
import numpy as np
import utils
import os
import torch.nn.functional as F
import time

def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class DiffusiveRestoration:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion

        if os.path.isfile(args.resume):
            self.diffusion.load_ddm_ckpt(args.resume, ema=True)
            self.diffusion.model.eval()
        else:
            print('Pre-trained diffusion model path is missing!')

    def restore(self, val_loader):
        image_folder = os.path.join(self.args.image_folder, self.config.data.val_dataset)
        processing_times = []  # List to store processing times for each image
        filenames_and_times = []  # List to store filenames and their processing times

        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                start_time = time.time()  # Start timing
                x_cond = x[:, :self.config.data.channels, :, :].to(self.diffusion.device)
                b, c, h, w = x_cond.shape
                img_h_32 = int(32 * np.ceil(h / 32.0))
                img_w_32 = int(32 * np.ceil(w / 32.0))
                x_cond = F.pad(x_cond, (0, img_w_32 - w, 0, img_h_32 - h), 'reflect')
                x_output = self.diffusive_restoration(x_cond)
                x_output = x_output[:, :, :h, :w]
                filename = f"{y[0]}.png"
                utils.logging.save_image(x_output, os.path.join(image_folder, f"{y[0]}.png"))
                end_time = time.time()  # End timing
                processing_time = end_time - start_time  # Calculate processing time
                processing_times.append(processing_time)  # Store processing time
                filenames_and_times.append((filename, processing_time))  # Store filename and processing time
                print(f"Processing image {filename} took {processing_time:.2f} seconds")
        # Calculate average, min, and max image inference times
        average_processing_time = sum(processing_times) / len(processing_times)
        min_processing_time, min_filename = min(filenames_and_times, key=lambda x: x[1])
        max_processing_time, max_filename = max(filenames_and_times, key=lambda x: x[1])

        print(f"Average processing time: {average_processing_time:.2f} seconds")
        print(f"Minimum processing time: {min_processing_time:.2f} seconds (Image: {min_filename})")
        print(f"Maximum processing time: {max_processing_time:.2f} seconds (Image: {max_filename})")

        # Save the details to a file
        with open("processing_times_details.txt", "w") as file:
            file.write(f"Average processing time: {average_processing_time:.2f} seconds\n")
            file.write(f"Minimum processing time: {min_processing_time:.2f} seconds (Image: {min_filename})\n")
            file.write(f"Maximum processing time: {max_processing_time:.2f} seconds (Image: {max_filename})\n")

    def diffusive_restoration(self, x_cond):
        x_output = self.diffusion.model(x_cond)
        return x_output["pred_x"]

