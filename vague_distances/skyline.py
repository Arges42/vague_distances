"""Module to compute the skyline for a set of given events,
using vague distance measures.

Author: Jan Greulich
Date: 09.02.2018
"""

import pypref as p
import pandas as pd

from .util import ProgressBar
from .vague_representation import Error

class InitializationError(Error):
    """Exception raised if something has not been initialized.

    Attributes:
        expression -- input expression in which the error occured
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message
        super().__init__()


class Skyline:
    """Construct a skyline using the given events.
    The query can be changed later.
    """

    def __init__(self, events=None, **kwargs):
        """

        Args:
            temporal_intervals (list): List of tuples of temporal and spatial informations
        """

        if isinstance(events, dict):
            events = events.values()
        self.events = events
        self.spatial_metric = kwargs.get("spatial_metric", "euclidean")

        self.distances = None
        self.skyline = None

    def build_distances(self, query, verbose=False):
        """Compute the distance from the query to each event.
        This method will be automatically be called by find_skyline.

        Attributes:
            query -- Event
            verbose -- If True, show the progress
        """

        distances = pd.DataFrame(columns=["temp", "spatial"])

        if verbose:
            progress = ProgressBar(len(self.events), fmt=ProgressBar.FULL)

        for event in self.events:
            temporal = query.vague_interval.vague_distance(
                event.vague_interval)
            spatial = query.vague_area.vague_distance3(
                event.vague_area, metric=self.spatial_metric)
            distances.loc[event.uid, "temp"] = temporal
            distances.loc[event.uid, "spatial"] = spatial
            if verbose:
                progress.current += 1
                progress()

        if verbose:
            progress.done()

        return distances

    def plot_skyline(self, ax):
        """Plot the skyline.

        Attributes:
            ax -- Matplotlib.pyplot.Axis
        """

        if self.distances is None:
            raise InitializationError(self.plot_skyline, "Distances have not been computed.")

        if self.skyline is None:
            raise InitializationError(self.plot_skyline, "Skyline has not been computed.")

        ax.plot(self.distances["temp"],
                self.distances["spatial"], 'bo', fillstyle="none")
        ax.plot(self.skyline["temp"], self.skyline["spatial"], 'bo')
        ax.set_xlabel("Temporal Distance")
        ax.set_ylabel("Spatial Distance")

        return ax

    def find_skyline(self, query, precomputed=False, verbose=False):
        """Find the skyline for the given event.

        Attributes:
            query -- Event
            precomputed -- If True, do not compute the distance for the input query.
                           Only use this, if the build_distances has been called before.
            verbose -- If True show progress of distance computation
        """
        if not precomputed or self.distances is None:
            self.distances = self.build_distances(query, verbose=verbose)

        pref = p.low("temp") * p.low("spatial")
        sky = pref.psel(self.distances)
        self.skyline = sky

        return sky
