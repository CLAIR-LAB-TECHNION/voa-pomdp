import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from matplotlib.colors import ListedColormap

from modeling.belief.block_position_belief import UnnormalizedBlocksPositionsBelief, BlockPosDist


def plot_block_belief(block_pos_belief: UnnormalizedBlocksPositionsBelief,
                      block_id,
                      actual_state=None,
                      positive_sensing_points=None,
                      negative_sensing_points=None,
                      pickup_attempt_points=None,
                      grid_size=100,
                      n_levels=50,
                      ret_as_image=False):
    """
    Plot the belief of the block position as a heatmap on a 2D plane.
    :param block_pos_belief: BlocksPositionsBelief that contains the belief to plot.
    :param block_id: Block id to plot the belief for.
    :param actual_state: Actual position of block to plot as a blue square if provided
    :param grid_size: Number of points to use in the grid for plotting.
    :param n_levels: Number of levels to use in the contour plot.
    """
    return plot_block_distribution(block_pos_belief.block_beliefs[block_id],
                                   block_pos_belief.ws_x_lims,
                                   block_pos_belief.ws_y_lims,
                                   actual_state=actual_state,
                                   positive_sensing_points=positive_sensing_points,
                                   negative_sensing_points=negative_sensing_points,
                                   pickup_attempt_points=pickup_attempt_points,
                                   grid_size=grid_size,
                                   n_levels=n_levels,
                                   ret_as_image=ret_as_image)


def plot_all_blocks_beliefs(block_pos_belief: UnnormalizedBlocksPositionsBelief,
                            actual_states=None,
                            positive_sensing_points=None,
                            negative_sensing_points=None,
                            pickup_attempt_points=None,
                            per_block_observed_mus_and_sigmas=None,
                            grid_size=100,
                            n_levels=50,
                            ret_as_image=False):
    """
    Plot the belief of all blocks in the belief as a heatmap on a 2D plane.
    parmeters are similar to plot_block_belief. per block observed mus and sigmas can be provided
    as a list, where if a block doesn't have observation, sigmas are set to -1.
    """
    if actual_states is None:
        actual_states = [None] * block_pos_belief.n_blocks_orig

    cmaps = ['Reds', 'Oranges', 'Purples', 'pink_r', 'hot_r',]
    images = []
    for i, block_belief in enumerate(block_pos_belief.block_beliefs_original_position):
        cmap = cmaps[i % len(cmaps)]
        if block_belief is None:
            # there is no block anymore, just an empty image
            im = None

        else:
            observed_mus_and_sigmas = None
            if per_block_observed_mus_and_sigmas is not None and len(per_block_observed_mus_and_sigmas) > i:
                if not per_block_observed_mus_and_sigmas[i][1][0] == -1:
                    observed_mus_and_sigmas = per_block_observed_mus_and_sigmas[i]

            im = plot_block_distribution(block_belief,
                                         block_pos_belief.ws_x_lims,
                                         block_pos_belief.ws_y_lims,
                                         actual_state=actual_states[i],
                                         positive_sensing_points=positive_sensing_points,
                                         negative_sensing_points=negative_sensing_points,
                                         pickup_attempt_points=pickup_attempt_points,
                                         observed_mus_and_sigmas=observed_mus_and_sigmas,
                                         grid_size=grid_size,
                                         n_levels=n_levels,
                                         ret_as_image=True,
                                         color_map=cmap)
        # remove whitespace:
        # im = im[10:-10, 10:-10, :]
        images.append(im)

    # if all images are blank, return empty image
    if all([im is None for im in images]):
        return np.zeros((5, 5, 3), dtype=np.uint8)

    # otherwise, create blank image in the same size of others instead of Nones:
    non_blank_images = [im for im in images if im is not None]
    blank_image = np.zeros_like(non_blank_images[0])
    images = [im if im is not None else blank_image for im in images]

    # plot all images in a column, first stack them on y to get a single image
    final_image = np.vstack(images)

    if ret_as_image:
        return final_image

    plt.figure(figsize=(8, 6*block_pos_belief.n_blocks_orig))
    plt.imshow(final_image)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_block_distribution(block_pos_dist: BlockPosDist,
                            x_lims,
                            y_lims,
                            actual_state=None,
                            positive_sensing_points=None,
                            negative_sensing_points=None,
                            pickup_attempt_points=None,
                            observed_mus_and_sigmas=None,
                            grid_size=100,
                            n_levels=50,
                            color_map='Reds',
                            ret_as_image=False):
    """
    Plot the block position distribution on a 2D plane as a grid with color intensity.
    :param block_pos_dist: BlocksPositionsBelief that contains the distribution to plot on x and y axis.
    :param x_lims: x-axis limits for the plot.
    :param y_lims: y-axis limits for the plot.
    :param actual_state: Actual position of block to plot as a blue square if provided
    :param grid_size: Number of points to use in the grid for plotting.
    :param n_levels: Number of levels to use in the contour plot.
    """
    x = np.linspace(x_lims[0], x_lims[1], grid_size)
    y = np.linspace(y_lims[0], y_lims[1], grid_size)
    dx = (x[1] - x[0]) / 2
    dy = (y[1] - y[0]) / 2

    # Shift x and y to midpoints for calculation, this is better for calculating the sum of the distribution
    x_mid = x[:-1] + dx
    y_mid = y[:-1] + dy

    xx, yy = np.meshgrid(x_mid, y_mid)
    z = block_pos_dist.pdf(np.stack([xx.ravel(), yy.ravel()], axis=1)).reshape(xx.shape)

    if z.max() == 0:
        # no point with probability found. Try finer grid:
        x = np.linspace(x_lims[0], x_lims[1], grid_size*4)
        y = np.linspace(y_lims[0], y_lims[1], grid_size*4)
        dx = (x[1] - x[0]) / 2
        dy = (y[1] - y[0]) / 2
        y_mid = y[:-1] + dy
        x_mid = x[:-1] + dx
        xx, yy = np.meshgrid(x_mid, y_mid)
        z = block_pos_dist.pdf(np.stack([xx.ravel(), yy.ravel()], axis=1)).reshape(xx.shape)

    if z.max() == 0:
        # no point with probability found. Just return empty image
        return np.zeros((800, 600, 3), dtype=np.uint8)

    levels = np.linspace(0, np.max(z), n_levels)
    alpha = 1 if observed_mus_and_sigmas is not None else 1

    plt.figure(figsize=(8, 6))

    plt.contourf(xx, yy, z, levels=levels, cmap=color_map, alpha=alpha)
    plt.colorbar(label='Probability Density')

    if actual_state is not None:
        # plot 0.04x0.04 square around the actual state
        actual_x, actual_y = actual_state
        rect = patches.Rectangle((actual_x - 0.02, actual_y - 0.02), 0.04, 0.04, linewidth=1,
                                 edgecolor='none', facecolor='blue', alpha=0.5)
        plt.gca().add_patch(rect)
        plt.plot(actual_x, actual_y, 'bo', markersize=3, )

    if positive_sensing_points is not None:
        for point in positive_sensing_points:
            plt.plot(point[0], point[1], 'g+', markersize=9, )

    if negative_sensing_points is not None:
        for point in negative_sensing_points:
            plt.plot(point[0], point[1], 'g_', markersize=9, )

    if pickup_attempt_points is not None:
        for point in pickup_attempt_points:
            plt.plot(point[0], point[1], 'go', markersize=9, )

    if observed_mus_and_sigmas is not None:
        # Create a custom colormap with transparency at lower values
        cmap = plt.get_cmap('Grays')
        colors = cmap(np.linspace(0, 1, 256))
        colors[:, -1] = np.linspace(0, 1, 256) ** 1 # Alpha channel
        custom_cmap = ListedColormap(colors)

        mus = observed_mus_and_sigmas[0]
        sigmas = observed_mus_and_sigmas[1]
        dist = BlockPosDist(x_lims, y_lims, mus[0], sigmas[0], mus[1], sigmas[1])
        z = dist.pdf(np.stack([xx.ravel(), yy.ravel()], axis=1)).reshape(xx.shape)
        levels = np.linspace(0, np.max(z), 20)
        plt.contourf(xx, yy, z, levels=levels, cmap=custom_cmap, antialiased=True, )

    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(x_lims)
    plt.ylim(y_lims)
    plt.gca().set_aspect('equal')
    plt.tight_layout()

    if ret_as_image:
        # instead of showing, return as image:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        img_np = np.array(img)
        plt.close()
        return img_np

    plt.show()

def plot_distribution(distribution, lower, upper, samples=2000):
    x = np.linspace(lower, upper, samples)
    y = distribution.pdf(x)

    plt.plot(x, y)
    plt.fill_between(x, y, )
    plt.show()
