import os

import torch
import wandb
from omegaconf import OmegaConf

from torch_3dgs.data import read_data
from torch_3dgs.trainer import Trainer
from torch_3dgs.model import GaussianModel
from torch_3dgs.point import get_point_clouds
from torch_3dgs.utils import dict_to_device

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def visualize_point_cloud(point_cloud, max_points=1000000):
    """
    Visualize a 3D point cloud using matplotlib.

    Args:
        point_cloud: A PointCloud object with .coords and optional .channels (R, G, B).
        max_points: Limit the number of points for performance.
    """
    coords = point_cloud.coords  # Nx3
    if coords.shape[0] > max_points:
        idx = np.random.choice(coords.shape[0], max_points, replace=False)
        coords = coords[idx]
        if 'R' in point_cloud.channels:
            r = point_cloud.channels['R'][idx]
            g = point_cloud.channels['G'][idx]
            b = point_cloud.channels['B'][idx]
    else:
        if 'R' in point_cloud.channels:
            r = point_cloud.channels['R']
            g = point_cloud.channels['G']
            b = point_cloud.channels['B']

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    if 'R' in point_cloud.channels:
        # Normalize colors
        rgb = np.stack([r, g, b], axis=1)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=rgb, s=0.5)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c='gray', s=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("3D Point Cloud Visualization (Matplotlib)")
    plt.show()

if __name__ == "__main__":
    config = OmegaConf.load("config.yaml")
    os.makedirs(config.output_folder, exist_ok=True)
    device = torch.device(config.device)

    data = read_data(config.data_folder, resize_scale=config.resize_scale)
    data = dict_to_device(data, device)

    points = get_point_clouds(
        data["camera"],
        data["depth"],
        data["alpha"],
        data["rgb"],
    )
    raw_points = points.random_sample(config.num_points)
    # visualize_point_cloud(raw_points)

    model = GaussianModel(sh_degree=4, debug=False)
    model.create_from_pcd(pcd=raw_points)

    wandb.init(
        project="EV-HW1",
        config=OmegaConf.to_container(config, resolve=True),
    )

    trainer = Trainer(
        data=data,
        model=model, 
        device=device,
        num_steps=config.num_steps,
        eval_interval=config.eval_interval,
        l1_weight=config.l1_weight,
        dssim_weight=config.dssim_weight,
        depth_weight=config.depth_weight,
        lr=config.lr,
        results_folder=config.output_folder,
        render_kwargs={
            "tile_size": config.render.tile_size,
        },
        logger=wandb,
    )
    trainer.train()