import os
import glob
import h5py
from datetime import datetime, timedelta
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
# Make sure to adjust the import paths according to your project structure
from ldcast.models.autoenc import autoenc, encoder

def read_data(
    data_dir="../data/demo/20210622",
    t0=datetime(2021,6,22,18,35),
    interval=timedelta(minutes=5),
    past_timesteps=4,
    crop_box=((128,480), (160,608))
):
    cb = crop_box
    R_past = []
    t = t0 - (past_timesteps-1) * interval
    for i in range(past_timesteps):
        timestamp = t.strftime("%y%j%H%M")
        fn = f"RZC{timestamp}VL.801.h5"
        fn = os.path.join(data_dir, fn)
        found_files = glob.glob(fn)
        if found_files:
            fn = found_files[0]
        else:
            raise FileNotFoundError(f"Unable to find data file {fn}.")
        with h5py.File(fn, 'r') as f:
            R = f["dataset1"]["data1"]["data"][:]
        R = R[cb[0][0]:cb[0][1], cb[1][0]:cb[1][1]]
        R_past.append(R)
        t += interval

    R_past = np.stack(R_past, axis=0)
    return R_past

def load_autoencoder(autoenc_weights_fn, encoded_channels=64, hidden_width=32):
    # Assuming SimpleConvEncoder and SimpleConvDecoder are defined in your_project_path.models.autoenc
    enc = encoder.SimpleConvEncoder()  # Adjust as per your actual encoder structure
    dec = encoder.SimpleConvDecoder()  # Adjust as per your actual decoder structure
    autoencoder = autoenc.AutoencoderKL(encoder=enc, decoder=dec, encoded_channels=encoded_channels, hidden_width=hidden_width)
    autoencoder.load_state_dict(torch.load(autoenc_weights_fn, map_location=torch.device('cpu')))
    # autoencoder.eval()
    return autoencoder

def visualize_images(autoencoder, images):
    for i, img in enumerate(images):
        img_tensor = torch.Tensor(img)[None, None, ...]  # Add batch and channel dimensions
        with torch.no_grad():
            print(img_tensor.shape)
            decoded, mean, log_var = autoencoder(img_tensor)
            # For visualization, you might need to adjust dimensions
            z = mean  # Using mean as a representative of the latent space

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(img.squeeze(), cmap='gray')
        axs[0].set_title('Input Image')
        axs[0].axis('off')

        encoded_img = z.squeeze().detach().numpy()
        axs[1].imshow(encoded_img, cmap='gray', aspect='auto')
        axs[1].set_title('Latent Encoding')
        axs[1].axis('off')

        decoded_img = decoded.squeeze().detach().numpy()
        axs[2].imshow(decoded_img, cmap='gray')
        axs[2].set_title('Reconstructed Image')
        axs[2].axis('off')

        plt.show()

if __name__ == "__main__":
    data_dir = r"D:\Documents\UNI\Master\THESIS weather forecasting\Models\ldcast\data\demo\20210622" 
    autoenc_weights_fn = r"D:\Documents\UNI\Master\THESIS weather forecasting\Models\ldcast\models\autoenc\autoenc-32-0.01.pt"
    images = read_data(data_dir=data_dir, t0=datetime(2021,6,22,18,35), past_timesteps=4)
    autoencoder = load_autoencoder(autoenc_weights_fn)
    visualize_images(autoencoder, images)
