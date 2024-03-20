
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
from ldcast.models.distributions import sample_from_standard_normal

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

# def read_knmi_data(
#         data_dir
#         ):
#     all_files = os.listdir(data_dir)

#     # Filter out only files (excluding directories)
#     files_only = [file for file in all_files if os.path.isfile(os.path.join(data_dir, file))]

#     # Take the first four files
#     first_four_files = files_only[:4]

#     R_past = []
#     # Print the first four files
#     for file in first_four_files:
#         print(file)
#         with h5py.File(os.path.join(data_dir, file), 'r') as f:
            
    
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
    return R.detach().numpy()

#TODO: plot border function
def plot_border(ax, crop_box=((128,480), (160,608))):    
    import shapefile
    border = shapefile.Reader("./data/Border_CH.shp")
    shapes = list(border.shapeRecords())
    for shape in shapes:
        x = np.array([i[0]/1000. for i in shape.shape.points[:]])
        y = np.array([i[1]/1000. for i in shape.shape.points[:]])
        ax.plot(
            x-crop_box[1][0]-255, 480-y-crop_box[0][0],
            'k', linewidth=1.0
        )

#TODO: plot frame function
def plot_frame(R, ax, draw_border=True, t=None, label=None):
    # fig = plt.figure(dpi=150)
    # ax = fig.add_subplot()
    plots.plot_precip_image(ax, R)
    ax.set_title(label)
    if draw_border:
        plot_border(ax)
    if t is not None:
        timestamp = "%Y-%m-%d %H:%M UTC"
        ax.text(
            0.02, 0.98, t.strftime(timestamp),
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes            
        )

# #TODO: plot latent function
# def plot_latent(latent):
#     channel = i*8 + j
#     axs[i][j].imshow(latents[0, channel, 0, ...], cmap='gray')
#     axs[i][j].set_title(f'Channel {channel}')
#     axs[i][j].axis('off')

def plot_vae_analysis(inputs, xs, latents, ys, reconstructions, ts, out_dir, draw_border=True, labels=None, crop_box=None):
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

    n = inputs.shape[0]  # Assuming images is a numpy array of shape [n_images, height, width]
    
    fig, axs = plt.subplots(n, 4, figsize=(15, 5*n), dpi=150)  # Create a row of 3 subplots
    print('latent shape:', latents.shape)
    for i in range(n):
        # Plot original image
        plot_frame(inputs[i], axs[i][0], draw_border=True, t=ts[i], label="Original")

        # plot input precipitation values
        axs[i][1].imshow(xs[i], cmap='gray')
        axs[i][1].set_title('Transformed input')
        axs[i][1].axis('off')

        # plot output precipitation values
        axs[i][2].imshow(ys[i], cmap='gray')
        axs[i][2].set_title('Model output')
        axs[i][2].axis('off')
        
        # Plot reconstructed image
        plot_frame(reconstructions[i], axs[i][3], draw_border=True, t=ts[i], label="Reconstructed (inverse transformed output)")

    # Save the figure
    fig.savefig(os.path.join(out_dir, f'VAE_Analysis.png'), bbox_inches='tight')
    plt.close(fig)

    # plot latent space: (1, 32, 1, 88, 112)
    # discard first dimension, and plot all 32 channels in 4 rows of 8 channels each
    fig, axs = plt.subplots(4, 8, figsize=(15, 10), dpi=150)
    for i in range(4):
        for j in range(8):
            channel = i*8 + j
            axs[i][j].imshow(latents[0, channel, 0, ...], cmap='gray')
            axs[i][j].set_title(f'Channel {channel}')
            axs[i][j].axis('off')

    fig.savefig(os.path.join(out_dir, f'Latent_Space Posterior Sample.png'), bbox_inches='tight')
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
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    
    R_past = read_data(
        data_dir=data_dir, t0=t0, interval=interval,
        past_timesteps=past_timesteps, crop_box=crop_box
    )
    
    x = transform_precip(R_past, device=device)

    vae = init_vae(vae_weights).to(device)

    y, mean, log_var = vae(x, sample_posterior=False)
    z = sample_from_standard_normal(mean, log_var)
    z = z.detach().numpy()
    mean = mean.detach().numpy()

    R_pred = inv_transform_precip(y)[0, ...]
    print(x.shape, R_past.shape, z.shape, R_pred.shape)

    # t = t0 - (R_past.shape[0]-k-1) * interval
    # compute ts in advance for all frames
    xs = x[0, 0, ...].detach().numpy()
    ys = y[0, 0, ...].detach().numpy()
    ts = [t0 - (R_past.shape[0]-k-1) * interval for k in range(R_past.shape[0])]
    plot_vae_analysis(R_past, xs, z, ys, R_pred, ts, out_dir, draw_border=draw_border, crop_box=crop_box)



if __name__ == "__main__":
    Fire(visualize_vae)
