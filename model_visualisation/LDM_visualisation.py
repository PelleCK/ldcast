from datetime import datetime, timedelta
import glob
import os

from fire import Fire
import h5py
from matplotlib import pyplot as plt
import numpy as np
import torch

from ldcast import forecast
from ldcast.visualization import plots


def read_data(
    data_dir="./data/demo/20210622",
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


def plot_frame(R, fn, draw_border=True, t=None, label=None):
    fig = plt.figure(dpi=150)
    ax = fig.add_subplot()
    plots.plot_precip_image(ax, R)
    if draw_border:
        plot_border(ax)
    if t is not None:
        timestamp = "%Y-%m-%d %H:%M UTC"
        if label is not None:
            timestamp += f" ({label})"
        ax.text(
            0.02, 0.98, t.strftime(timestamp),
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes            
        )
    
    fig.savefig(fn, bbox_inches='tight')
    plt.close(fig)


def plot_feature_maps(feature_maps, out_dir):
    for k, v in feature_maps.items():
        channels = v.shape[1]
        fig, axs = plt.subplots(3, channels, figsize=(15, 10), dpi=150)

        # Ensure axs is always 2-dimensional (3, channels)
        if channels == 1:
            axs = np.expand_dims(axs, axis=-1)  # Expand the dimensions of axs to make it 2-dimensional

        for channel in range(channels):
            v_channel = v[0, channel, ...]
            v_channel_mean = v_channel.mean(axis=-1)
            v_channel_max = v_channel.max(axis=-1)
            # Using numpy's linalg.norm for norm calculation
            v_channel_norm = np.linalg.norm(v_channel, ord=2, axis=-1)

            axs[0, channel].imshow(v_channel_mean, cmap='gray')
            axs[0, channel].set_title(f'Mean of channel {channel}')
            axs[0, channel].axis('off')

            axs[1, channel].imshow(v_channel_max, cmap='gray')
            axs[1, channel].set_title(f'Max of channel {channel}')
            axs[1, channel].axis('off')

            im_norm = axs[2, channel].imshow(v_channel_norm, cmap='viridis', vmin=0, vmax=100)
            axs[2, channel].set_title(f'Embedding norm of channel {channel}')
            axs[2, channel].axis('off')
            fig.colorbar(im_norm, ax=axs[2, channel], fraction=0.046, pad=0.04)

        # suptitle for entire figure
        fig.suptitle(f'Mean, max, and norm of embeddings after {k} block')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for suptitle
        plt.show()
        fig.savefig(os.path.join(out_dir, f'emb_mean_max_norm_after_{k}.png'), bbox_inches='tight')
        plt.close(fig)


def forecast_demo(
    ldm_weights_fn="./models/genforecast/genforecast-radaronly-256x256-20step.pt",
    autoenc_weights_fn="./models/autoenc/autoenc-32-0.01.pt",
    num_diffusion_iters=1, # 50
    out_dir="./figures/forecaster_visualisation/",
    data_dir="./data/demo/20210622",
    t0=datetime(2021,6,22,18,35),
    interval=timedelta(minutes=5),
    past_timesteps=4,
    crop_box=((128,480), (160,608)),
    draw_border=True,
    ensemble_members=1,
):
    R_past = read_data(
        data_dir=data_dir, t0=t0, interval=interval,
        past_timesteps=past_timesteps, crop_box=crop_box
    )

    if ensemble_members == 1:
        print("Using single model")
        fc = forecast.Forecast(
            ldm_weights_fn=ldm_weights_fn,
            autoenc_weights_fn=autoenc_weights_fn
        )

        feature_maps = {}

        def get_hook_fn(key):
            def hook_fn(module, input, output):
                feature_maps[key] = output.detach().cpu().numpy()  # Detach the output to avoid saving computation graph
            return hook_fn

        # def hook_fn(module, input, output):
        #     feature_maps[module] = output

        fc.ldm.context_encoder.analysis[0][-1].register_forward_hook(get_hook_fn("analysis"))
        fc.ldm.context_encoder.temporal_transformer[-1].register_forward_hook(get_hook_fn("temporal_transformer"))
        fc.ldm.context_encoder.fusion.register_forward_hook(get_hook_fn("fusion"))
        fc.ldm.context_encoder.forecast[-1].register_forward_hook(get_hook_fn("forecast"))

        R_pred = fc(
            R_past,
            num_diffusion_iters=num_diffusion_iters
        )

        # print('feature maps keys:', feature_maps.keys())
        print('feature maps shapes:', {k: v.shape for k, v in feature_maps.items()}) # k.__class__.__name__

    elif ensemble_members > 1:
        fc = forecast.ForecastDistributed(
            ldm_weights_fn=ldm_weights_fn,
            autoenc_weights_fn=autoenc_weights_fn,
        )
        R_past = R_past.reshape((1,) + R_past.shape)
        R_pred = fc(
            R_past,
            num_diffusion_iters=num_diffusion_iters,
            ensemble_members=ensemble_members    
        )
        R_past = R_past[0,...]
        R_pred = R_pred[0,...].mean(axis=-1) # compute ensemble mean
    else:
        raise ValueError("ensemble_members must be > 0")

    os.makedirs(out_dir, exist_ok=True)
    
    # plot feature maps from dictionary
    # take the mean over the embedding dimension (last dimension, length 128)
    # then plot the 32 channels with each channel as a subplot
    plot_feature_maps(feature_maps, out_dir)
            

    # for k, v in feature_maps.items():
    #     print(k, v.shape)
    #     if k == 'analysis':
    #         # no channels for analysis output, so single plot
    #         v_mean = v.mean(axis=-1)
    #         v_max = v.max(axis=-1)

    #         fig = plt.figure(dpi=150)

    #         ax1 = fig.add_subplot(1, 2, 1)
    #         ax1.imshow(v_mean[0, 0, ...], cmap='gray')
    #         ax1.set_title(f'Mean of embedding after {k} block')
    #         ax1.axis('off')

    #         ax2 = fig.add_subplot(1, 2, 2)
    #         ax2.imshow(v_max[0, 0, ...], cmap='gray')
    #         ax2.set_title(f'Max of embedding after {k} block')
    #         ax2.axis('off')

    #         fig.savefig(os.path.join(out_dir, f'emb_mean_max_after_{k}.png'), bbox_inches='tight')
    #         plt.close(fig)
    #     else:
    #         channels = v.shape[1]
    #         fig, axs = plt.subplots(2, channels, figsize=(15, 10), dpi=150)
    #         for channel in range(channels):
    #             v_channel_mean = v[0, channel, ...].mean(axis=-1)
    #             v_channel_max = v[0, channel, ...].max(axis=-1)

    #             axs[0][channel].imshow(v_channel_mean, cmap='gray')
    #             axs[0][channel].set_title(f'Mean of channel {channel} after {k}')
    #             axs[0][channel].axis('off')

    #             axs[1][channel].imshow(v_channel_max, cmap='gray')
    #             axs[1][channel].set_title(f'Max of channel {channel} after {k}')
    #             axs[1][channel].axis('off')

    #         # suptitle for entire figure
    #         fig.suptitle(f'Mean and max of embeddings after {k} block')
    #         fig.tight_layout()
    #         fig.savefig(os.path.join(out_dir, f'emb_mean_max_after_{k}.png'), bbox_inches='tight')
    #         plt.close(fig)


if __name__ == "__main__":
        Fire(forecast_demo)
