import numpy as np
import torch
import matplotlib.pyplot as plt

from mushroom_rl.utils.dataset import parse_dataset, compute_episodes_length, get_init_states
from mushroom_rl.utils.torch import to_float_tensor


def compute_V0(dataset, pi_net, Q, action_dim):
    init_states_deter = get_init_states(dataset)
    prev_a = torch.zeros((init_states_deter.shape[0], action_dim))
    a, next_hidden = pi_net(to_float_tensor(init_states_deter), prev_a)
    V = Q(to_float_tensor(init_states_deter), to_float_tensor(a), to_float_tensor(next_hidden), prev_a)
    return V


def compute_mean_Q_across_trajectories(dataset, pi_net, Q):

    state, action, _, _, _, _ = parse_dataset(dataset)
    L = compute_episodes_length(dataset)
    L_cumulated = [np.sum(L[:i + 1]) for i in range(len(L))]

    # calculate the prev_a
    action_traj = np.split(action, L_cumulated[:-1])
    prev_a = [np.concatenate([np.expand_dims(np.zeros_like(a[0]), axis=0), a[:-1]], ) for a in action_traj]
    prev_a = np.vstack(prev_a)

    # calculate the Qs
    a, next_hidden = pi_net(to_float_tensor(state), to_float_tensor(prev_a))
    Qs = Q(to_float_tensor(state), to_float_tensor(a), to_float_tensor(next_hidden), prev_a)
    Qs = np.split(Qs, L_cumulated[:-1])

    # calculate the mean across trajectories. We use masked array as the trajectories can have unequal lengths
    max_L = np.max(L)
    masked_Q = np.ma.empty((len(Qs), max_L))
    masked_Q.mask = True
    for i in range(len(Qs)):
        q_i = Qs[i]
        masked_Q[i, :len(Qs[i])] = q_i

    mean_Q = np.mean(masked_Q, axis=0)

    return mean_Q


def plot_to_tensorboard(writer, fig, step, tag):
    """
    Takes a matplotlib figure handle and converts it using
    canvas and string-casts to a numpy array that can be
    visualized in TensorBoard using the add_image function

    Parameters:
        writer (tensorboard.SummaryWriter): TensorBoard SummaryWriter instance.
        fig (matplotlib.pyplot.fig): Matplotlib figure handle.
        step (int): counter usually specifying steps/epochs/time.
    """

    # Draw figure on canvas
    fig.canvas.draw()

    # Convert the figure to numpy array, read the pixel values and reshape the array
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Normalize into 0-1 range for TensorBoard(X). Swap axes for newer versions where API expects colors in first dim
    img = img / 255.0
    img = np.fliplr(img)
    img = np.rot90(img)
    img = np.swapaxes(img, 0, 2) # if your TensorFlow + TensorBoard version are >= 1.8

    # Add figure in numpy "image" to TensorBoard writer
    writer.add_image(tag, img, step)
    plt.close(fig)
