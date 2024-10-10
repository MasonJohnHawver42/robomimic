"""
This file contains utility functions for visualizing image observations in the training pipeline.
These functions can be a useful debugging tool.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils


def image_tensor_to_numpy(image):
    """
    Converts processed image tensors to numpy so that they can be saved to disk or video.
    A useful utility function for visualizing images in the middle of training.

    Args:
        image (torch.Tensor): images of shape [..., C, H, W]

    Returns:
        image (np.array): converted images of shape [..., H, W, C] and type uint8
    """
    return TensorUtils.to_numpy(
            ObsUtils.unprocess_image(image)
        ).astype(np.uint8)


def image_to_disk(image, fname):
    """
    Writes an image to disk.

    Args:
        image (np.array): image of shape [H, W, 3]
        fname (str): path to save image to
    """
    image = Image.fromarray(image)
    image.save(fname)


def image_tensor_to_disk(image, fname):
    """
    Writes an image tensor to disk. Any leading batch dimensions are indexed out
    with the first element.

    Args:
        image (torch.Tensor): image of shape [..., C, H, W]. All leading dimensions
            will be indexed out with the first element
        fname (str): path to save image to
    """
    # index out all leading dimensions before [C, H, W]
    num_leading_dims = len(image.shape[:-3])
    for _ in range(num_leading_dims):
        image = image[0]
    image = image_tensor_to_numpy(image)
    image_to_disk(image, fname)


def visualize_image_randomizer(original_image, randomized_image, randomizer_name=None):
    """
    A function that visualizes the before and after of an image-based input randomizer
    Args:
        original_image: batch of original image shaped [B, H, W, 3]
        randomized_image: randomized image shaped [B, N, H, W, 3]. N is the number of randomization per input sample
        randomizer_name: (Optional) name of the randomizer
    Returns:
        None
    """

    B, N, H, W, C = randomized_image.shape

    # Create a grid of subplots with B rows and N+1 columns (1 for the original image, N for the randomized images)
    fig, axes = plt.subplots(B, N + 1, figsize=(4 * (N + 1), 4 * B))

    for i in range(B):
        # Display the original image in the first column of each row
        axes[i, 0].imshow(original_image[i])
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        # Display the randomized images in the remaining columns of each row
        for j in range(N):
            axes[i, j + 1].imshow(randomized_image[i, j])
            axes[i, j + 1].axis("off")

    title = randomizer_name if randomizer_name is not None else "Randomized"
    fig.suptitle(title, fontsize=16)

    # Adjust the space between subplots for better visualization
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    # Show the entire grid of subplots
    plt.show()


def depth_to_rgb(depth_map, depth_min=None, depth_max=None):
    """
    Convert depth map to rgb array by computing normalized depth values in [0, 1].
    """
    # normalize depth map into [0, 1]
    if depth_min is None:
        depth_min = depth_map.min()
    if depth_max is None:
        depth_max = depth_map.max()
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    # depth_map = np.clip(depth_map / 3., 0., 1.)
    if len(depth_map.shape) == 3:
        assert depth_map.shape[-1] == 1
        depth_map = depth_map[..., 0]
    assert len(depth_map.shape) == 2 # [H, W]
    return (255. * cm.plasma(depth_map, 3)).astype(np.uint8)[..., :3]

# Define global palettes
GLOCAL_PALLETES = [
    np.array([[8, 156, 255], [52, 0, 40], [0, 255, 0], [240, 8, 225], [255, 214, 54]]),
    np.array([[15, 53, 254], [209, 130, 194], [135, 0, 0], [126, 255, 32], [255, 79, 9], [255, 207, 76],
              [1, 175, 174], [0, 5, 50], [122, 119, 79], [154, 5, 245]]),
    np.array([[221, 157, 147], [98, 241, 25], [60, 1, 189], [143, 125, 77], [19, 11, 118], [149, 108, 140], 
              [253, 225, 211], [108, 201, 187], [233, 14, 202], [203, 57, 249], [217, 3, 16], [84, 240, 247], 
              [218, 191, 31], [133, 37, 32], [2, 164, 83], [74, 67, 125], [100, 233, 115], [200, 59, 71], 
              [21, 228, 129], [225, 141, 230]])
    # You can add more palettes with different sizes as needed.
]

# Function to automatically select a palette based on N
def select_palette(N):
    # Look for a palette with at least N colors
    for palette in GLOCAL_PALLETES:
        print(palette, len(palette))
        if len(palette) >= N:
            return palette[:N]  # Return the first N colors from the selected palette
    
    # If no palette is large enough, repeat the largest palette
    largest_palette = GLOCAL_PALLETES[-1]
    return np.tile(largest_palette, (N // len(largest_palette) + 1, 1))[:N]

# Function to map segment indices to RGB color map
def seg_to_rgb(seg_map, N=None):
    """
    Convert a segmentation map to an RGB image using an automatically selected palette.
    
    Parameters:
    seg_map: 2D numpy array with integer values representing segment labels.
    N: Number of unique segments (optional). If None, it's inferred from seg_map.
    
    Returns:
    rgb_map: An RGB image of the same spatial size as seg_map.
    """
    
    if N is None:
        N = seg_map.max() + 1  # Infer the number of segments

    # Automatically select a palette with at least N colors
    palette = select_palette(N)
    
    seg_map = seg_map.squeeze(-1)

    # Create an empty RGB map with the same height and width as seg_map
    h, w = seg_map.shape
    rgb_map = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Map each segment index to the corresponding color in the palette
    for segment_id in range(N):
        rgb_map[seg_map == segment_id] = palette[segment_id]
    
    return rgb_map.astype(np.uint8)[..., :3]