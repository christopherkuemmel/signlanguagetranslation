"""Contains the Plotter class for saving loss curves."""
import os
from typing import List, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
# plt.switch_backend('agg')


def save_plot(points, output_path: str = './output'):
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)  # put ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig(os.path.join(output_path, 'loss.png'))
    plt.close()


class Plotter():
    """The Plotter will store, plot and save training and validation scores."""

    def __init__(self, output_path: str) -> None:
        """
        Args:
            output_path (str): The path where to save the plots.

        """
        sns.set(style="darkgrid")
        self.output_path = output_path
        self.loss_df = pd.DataFrame(columns=['timestep', 'loss', 'value'])

    def __call__(self, timesteps: Union[float, List[float]], training_loss: Union[float, List[float]], validation_loss: Union[float, List[float]]) -> None:
        """Update the stored timesteps, training- and validation losses and plot them.

        Args:
            timesteps (Union[float, List[float]]): The timestep (or list of timesteps).
            training_loss (Union[float, List[float]]): The training loss (or list of timesteps) corresponding to the timesteps.
            validation_loss (Union[float, List[float]]): The validation loss (or list of timesteps) corresponding to the timesteps
        """
        if isinstance(timesteps, float) and isinstance(training_loss, float) and isinstance(validation_loss, float):
            self.loss_df = self.loss_df.append(
                {
                    'timestep': timesteps,
                    'loss': 'train',
                    'value': training_loss
                },
                ignore_index=True,
            )
            self.loss_df = self.loss_df.append(
                {
                    'timestep': timesteps,
                    'loss': 'val',
                    'value': validation_loss
                },
                ignore_index=True,
            )
        else:
            assert len(timesteps) == len(training_loss) == len(validation_loss)
            self.loss_df = self.loss_df.append(
                [{
                    'timestep': timesteps[idx],
                    'loss': 'train',
                    'value': training_loss[idx]
                } for idx in range(len(timesteps))],
                ignore_index=True,
            )
            self.loss_df = self.loss_df.append(
                [{
                    'timestep': timesteps[idx],
                    'loss': 'val',
                    'value': validation_loss[idx]
                } for idx in range(len(timesteps))],
                ignore_index=True,
            )

        # TODO: plot showing doubled labels (several times loss..)
        plot = sns.lineplot(data=self.loss_df, x='timestep', y='value', hue='loss')
        plot = plot.get_figure()
        plot.savefig(os.path.join(self.output_path, 'loss.png'))
