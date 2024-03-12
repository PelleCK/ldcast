
#TODO: imports
from datetime import datetime, timedelta
import glob
import os

from fire import Fire
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from ldcast.features.transform import Antialiasing
from ldcast.visualization import plots
from ldcast.models.autoenc import autoenc, encoder

def read_data(
        data_dir, 
        t0, 
        interval, 
        past_timesteps, 
        crop_box
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
    
def transform_precip(
        R,
        R_min_value=0.1,
        R_zero_value=0.02,
        log_R_mean=-0.051,
        log_R_std=0.528,
        device='cpu'
        ):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = R.copy()
    x[~(x >= R_min_value)] = R_zero_value
    x = np.log10(x)
    x -= log_R_mean
    x /= log_R_std
    x = x.reshape((1,) + x.shape)

    antialiasing = Antialiasing()
    x = antialiasing(x)
    x = x.reshape((1,) + x.shape)
    return torch.Tensor(x).to(device=device)

def inv_transform_precip(
        x,
        R_min_output=0.1,
        R_max_output=118.428,
        log_R_mean=-0.051,
        log_R_std=0.528,
        ):
    x *= log_R_std
    x += log_R_mean
    R = torch.pow(10, x)
    if R_min_output:        
        R[R < R_min_output] = 0.0
    if R_max_output is not None:
        R[R > R_max_output] = R_max_output
    R = R[:,0,...]
    return R.to(device='cpu').numpy()

#TODO: plot border function
def plot_border(ax, crop_box):
    pass

#TODO: plot frame function
def plot_frame(R, fn, draw_border, t, label):
    pass

#TODO: plot latent function
def plot_latent(latent, fn):
    pass

def plot_vae_analysis(images, latent_representations, reconstructed_images, out_dir, draw_border=True, labels=None, crop_box=None):
    """
    Plots the original images, their latent space representations, and the reconstructed images side by side.
    
    :param images: Array of original images.
    :param latent_representations: Array of latent representations.
    :param reconstructed_images: Array of reconstructed images.
    :param out_dir: Output directory to save the plots.
    :param draw_border: Whether to draw the border on the plots.
    :param labels: Optional list of labels for the images.
    """
    os.makedirs(out_dir, exist_ok=True)

    n = images.shape[0]  # Assuming images is a numpy array of shape [n_images, height, width]
    
    for i in range(n):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=150)  # Create a row of 3 subplots
        
        # Plot original image
        axs[0].imshow(images[i], cmap='gray')
        axs[0].set_title('Input Image' if not labels else f'Input: {labels[i]}')
        axs[0].axis('off')
        if draw_border:
            plot_border(axs[0], crop_box)
        
        # Plot latent representation
        # This is a simplification. Adjust this based on the actual shape and content of your latent representations.
        if latent_representations[i].ndim > 2:
            latent_img = latent_representations[i].reshape(-1, latent_representations[i].shape[-1])  # Reshape if needed
        else:
            latent_img = latent_representations[i]
        axs[1].imshow(latent_img, cmap='gray')
        axs[1].set_title('Latent Space')
        axs[1].axis('off')

        # Plot reconstructed image
        axs[2].imshow(reconstructed_images[i], cmap='gray')
        axs[2].set_title('Reconstructed Image')
        axs[2].axis('off')
        if draw_border:
            plot_border(axs[2], crop_box)

        # Save the figure
        fig.savefig(os.path.join(out_dir, f'VAE_Analysis_{i:02d}.png'), bbox_inches='tight')
        plt.close(fig)

def init_vae(vae_weights_fn):
    enc = encoder.SimpleConvEncoder()
    dec = encoder.SimpleConvDecoder()
    vae = autoenc.AutoencoderKL(enc, dec)
    vae.load_state_dict(torch.load(vae_weights_fn))
    return vae

def visualize_vae(
        data_dir="./data/demo/20210622", 
        vae_weights="./models/autoenc/autoenc-32-0.01.pt", 
        out_dir="./figures/vae_visualisation/", 
        t0=datetime(2021,6,22,18,35),
        interval=timedelta(minutes=5),
        past_timesteps=4,
        crop_box=((128,480), (160,608)),
        draw_border=True
    ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    R_past = read_data(
        data_dir=data_dir, t0=t0, interval=interval,
        past_timesteps=past_timesteps, crop_box=crop_box
    )
    x = transform_precip(R_past, device=device)
    print(x.shape)

    vae = init_vae(vae_weights).to(device)
    print(vae)

    # z, y = vae.encode(x), vae.decode(z)
    with torch.no_grad():
        for i in range(x.shape[0]):
            y, z, _ = vae(x[i,...].unsqueeze(0), sample_posterior=False)
            print(x[i,...].unsqueeze(0).shape, y.shape, z.shape)
            if i == 0:
                z_all = z
                y_all = y
            else:
                z_all = torch.cat((z_all, z), dim=0)
                y_all = torch.cat((y_all, y), dim=0)
        # y, z, _ = vae(x)
        # z = z.to(device='cpu').numpy()
        z = z_all.to(device='cpu').numpy()
        # y = y_all.to(device='cpu').numpy()

    R_pred = inv_transform_precip(y)[0, ...]
    print(x.shape, R_past.shape, z.shape, R_pred.shape)

    # remove first two empty dimensions of R_pred


    plot_vae_analysis(R_past, z, R_pred, out_dir, draw_border=draw_border, crop_box=crop_box)

    # #TODO: change to plot all frames in a loop
    # for k in range(R_past.shape[0]):
    #     fn = os.path.join(out_dir, f"R_past-{k:02d}.png")
    #     t = t0 - (R_past.shape[0]-k-1) * interval
    #     plot_frame(R_past[k,:,:], fn, draw_border=draw_border,
    #         t=t, label="Input image")

    # # plot_frame(R_past, f"{out_dir}/input.png", draw_border, t0, "input")

    # #TODO: change to plot all frames in a loop
    # if plot_latent:
    #     for k in range(z.shape[0]):
    #         fn = os.path.join(out_dir, f"latent-{k:02d}.png")
    #         plot_latent(z[k,...], fn)
    #     # plot_latent(z, f"{out_dir}/latent.png")

    # #TODO: change to plot all frames in a loop
    # for k in range(R_pred.shape[0]):
    #     fn = os.path.join(out_dir, f"R_pred-{k:02d}.png")
    #     t = t0 + (k+1)*interval
    #     plot_frame(R_pred[k,:,:], fn, draw_border=draw_border,
    #         t=t, label="Output image")
    # # plot_frame(R_pred, f"{out_dir}/output.png", draw_border, t0, "output")
        
    # # plot the input, latent and output images in a single loop
    # # and for each image, plot the input, latent and output images in a subplot
    # # and save the plot to a file
    # # note that there is no future prediction, but just a reconstruction of the input image
    # # so the output image has the same time step as the input image
    # for k in range(R_past.shape[0]):
    #     fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    #     t = t0 - (R_past.shape[0]-k-1) * interval
    #     plot_frame(R_past[k,:,:], fn, draw_border=draw_border,
    #         t=t, label="Input image")
    #     plot_latent(z[k,...], fn)
    #     t = t0 - (R_past.shape[0]-k-1) * interval
    #     plot_frame(R_pred[k,:,:], fn, draw_border=draw_border,
    #         t=t, label="Reconstruction")
    #     plt.savefig(f"{out_dir}/R-{k:02d}.png")
    #     plt.close()



if __name__ == "__main__":
    Fire(visualize_vae)
