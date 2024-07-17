import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL.Image import Image
from modeling.block_position_belief import BlocksPositionsBelief, BlockPosDist


def plot_block_belief(block_pos_belief: BlocksPositionsBelief,
                      block_id,
                      actual_state=None,
                      positive_sensing_points=None,
                      negative_sensing_points=None,
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
                                   grid_size=grid_size,
                                   n_levels=n_levels,
                                   ret_as_image=ret_as_image)


def plot_block_distribution(block_pos_dist: BlockPosDist,
                            x_lims,
                            y_lims,
                            actual_state=None,
                            positive_sensing_points=None,
                            negative_sensing_points=None,
                            grid_size=100,
                            n_levels=50,
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

    sum_z_mid = np.sum(z) * (x[1] - x[0]) * (y[1] - y[0])
    print(f"Sum of z using midpoints: {sum_z_mid}")

    levels = np.linspace(0, np.max(z), n_levels)

    # Plot the heatmap
    plt.figure(figsize=(8, 6))

    plt.contourf(xx, yy, z, levels=levels, cmap='Reds')
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

    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(x_lims)
    plt.ylim(y_lims)
    plt.gca().set_aspect('equal')

    if ret_as_image:
        # instead of showing, return as image:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        img_np = np.array(img)
        return img_np

    plt.show()


def plot_distribution(distribution, lower, upper, samples=2000):
    x = np.linspace(lower, upper, samples)
    y = distribution.pdf(x)

    plt.plot(x, y)
    plt.fill_between(x, y, )
    plt.show()

