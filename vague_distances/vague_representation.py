"""
This  module provides the access to VagueIntervals and VagueAreas.
All introduced distance measures are implemented here.

Author: Jan Greulich
Date: 09.02.2018
"""

from itertools import product

import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy import integrate
from pyemd import emd

from .distances import hausdorff_origin, hausdorff_dist, circle_dist

EPS = 0.00001


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class InputError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occured
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message
        super().__init__()


class VagueInterval:
    """Vague interval

    """
    uid = 0

    def __init__(self, start_date, end_date, pmf_start, pmf_length):
        self._start = start_date
        self._end = end_date
        self._pmf_start = pmf_start
        self._pmf_length = pmf_length

        self.uid = VagueInterval.uid
        VagueInterval.uid += 1

    def fixed_origin_distance(self, other, n_bins=20):
        """Compute the distance between self and other.
        This uses the fixed origin distance, see Definition 3.10.

        Attributes:
            other -- VagueInterval
            n_bins -- Number of maximum bins for the distance histogram to the origin
        """

        self._check_input(other, self.fixed_origin_distance)

        dist_hist1 = self.distance_histogram_origin(
            hausdorff_origin, n_bins=n_bins)
        dist_hist2 = other.distance_histogram_origin(
            hausdorff_origin, n_bins=n_bins)
        h1 = np.hstack((dist_hist1[:, 1], np.zeros(len(dist_hist2))))
        h2 = np.hstack((np.zeros(len(dist_hist1)), dist_hist2[:, 1]))
        D = distance.squareform(
            distance.pdist(
                np.hstack((dist_hist1[:, 0], dist_hist2[:, 0])).reshape(-1, 1)
            )
        )

        e = emd(h1, h2, D)

        return e

    def density_vague_distance(self, other):
        """Compute the distance between self and other.
        This uses the pmf based vague distance, see Definition 3.7

        Attributes:
            other -- VagueInterval
        """

        self._check_input(other, self.density_vague_distance)

        sig1 = self.signature()
        sig2 = other.signature()

        D = distance.squareform(distance.pdist(
            np.vstack((sig1[:, :2], sig2[:, :2])), metric=hausdorff_dist))
        e = emd(np.hstack((sig1[:, 2], np.zeros(sig2[:, 2].shape[0]))),
                np.hstack((np.zeros(sig1[:, 2].shape[0]), sig2[:, 2])),
                D
               )
        return e

    def rv_vague_distance(self, other, n_samples=1000):
        """Compute the distance histogram between self and other.
        This used the rv based vague distance, see Definition 3.9

        Attributes:
            other -- VagueInterval
            n_samples -- Number of samples to draw from the distributions

        """
        self._check_input(other, self.rv_vague_distance)

        samples1 = self.sample_interval(n_samples=n_samples)
        samples2 = other.sample_interval(n_samples=n_samples)

        def inner_hausdorff(row):
            return hausdorff_dist(row[:2], row[2:])

        return np.apply_along_axis(inner_hausdorff, 1, np.hstack((samples1, samples2)))

    def sample_interval(self, n_samples=1):
        """Draw n sampled intervals from the vague interval.

        Attributes:
            n_samples -- Number of samples
        """

        s = self._pmf_start.sample(n_samples=n_samples)
        l = self._pmf_length.sample(n_samples=n_samples)

        return np.stack((s, l), axis=-1)

    def signature(self):
        """Compute the signature of the VagueInterval.
        See Definition 3.6
        """

        days = np.arange(self._start, self._end)
        lengths = np.arange(self._end-self._start)
        xv, yv = np.meshgrid(days, lengths)
        intervals = np.stack((xv, xv+yv), axis=-1).reshape(-1, 2)
        prob = self._pmf(xv, yv).reshape(-1, 1)
        prob = prob/prob.sum()
        return np.hstack((intervals, prob))

    def distance_histogram_origin(self, dist, n_bins=20):
        """Compute the distance histogram to the origin.
        See Definition 3.8

        Attributes:
            dist -- Distance function that takes two Nx2 arrays as input
            n_bins -- Maximum number of bins for the distance histogram
        """

        combs = list(product(range(self._start, self._end+1),
                             range(0, self._end-self._start+1)))
        intervals = np.array([(s, l) for s, l in combs if s+l <= self._end])
        probs = self._pmf2(intervals)

        dists = dist(intervals, np.array([[0, 0]]))
        distances = np.stack((dists, probs/probs.sum()), axis=-1)

        min_dist = distances[:, 0].min()
        max_dist = distances[:, 0].max()

        if max_dist-min_dist < n_bins:
            n_bins = max_dist-min_dist+1

        bins = np.linspace(min_dist, max_dist, n_bins)
        bin_idx = np.abs(
            np.repeat(distances[:, 0].reshape(-1, 1), n_bins, axis=1)-bins).argmin(1)
        histogram = pd.DataFrame(
            np.array([bin_idx, distances[:, 1]]).T, columns=["bin", "p"])
        histogram = histogram.groupby("bin").sum()

        assert abs(sum(histogram.values)-1) < EPS

        return np.hstack((bins.reshape(-1, 1), histogram.values))

    @property
    def intervals(self):
        """Return all subintervals of the VagueInterval."""

        combs = list(product(range(self._start, self._end+1),
                             range(0, self._end-self._start+1)))
        intervals = np.array([(s, l) for s, l in combs if s+l <= self._end])
        return intervals

    def _pmf2(self, interval):
        # TODO: Rename this
        idx = interval.sum(1) < self._end
        prob = self._pmf_start(interval[:, 0])*self._pmf_length(interval[:, 1])
        return np.where(idx, prob, 0)

    def _pmf(self, day, length):
        return np.where(day+length < self._end, self._pmf_start(day)*self._pmf_length(length), 0)

    def _check_input(self, other, expression):
        """Convenience function to check the input."""

        if not isinstance(other, VagueInterval):
            raise InputError(
                expression, "other needs to be an instance of {}".format(self.__class__))

    def __repr__(self):
        return "VagueInterval(s:{},e:{}) {}".format(self._start, self._end, self.uid)


class VagueArea:
    """Vague Area

    """
    # TODO: Iplement width/height variation
    uid = 0

    def __init__(self, lower_left, upper_right, pdf_xy, **kwargs):
        self.x = lower_left[0]
        self.y = lower_left[1]
        self.width = upper_right[0] - lower_left[0]
        self.height = upper_right[1] - lower_left[1]

        self.use_points = True
        self.pdf_xy = pdf_xy
        self.pdf_x = self._marginal_pdf("x")
        self.pdf_y = self._marginal_pdf("y")

        self.uid = VagueArea.uid
        VagueArea.uid += 1

        self.name = kwargs.get("name", None)

    def rv_vague_distance(self, other, n_samples=100, metric="circle_dist"):
        """Compute the distance distribution between self and other.
        See Definition 4.5

        Attributes:
            other -- VagueArea
            n_samples -- Nuber of samples to draw from the distributions
            metric -- Base metric, signature metric(p1, p2), with p1,p2 being 1x2
                      Or one of the following strings: circle_dist, euclidean
        """

        self._check_input(other, self.rv_vague_distance)

        if metric == "circle_dist":
            metric = circle_dist
        elif metric == "euclidean":
            metric = distance.euclidean

        samples1 = self.sample_points(n_samples=n_samples)
        samples2 = other.sample_points(n_samples=n_samples)
        stacked_samples = np.hstack((samples1, samples2))

        dist = np.apply_along_axis(lambda r: metric(
            r[:2], r[2:]), 1, stacked_samples)

        return dist

    def density_vague_distance(self, other, n_samples=10, metric='circle_dist'):
        """Compute the pdf based vague distance between self and other.
        See Definition 4.4

        Attributes:
            other -- VagueArea
            n_samples -- Number of equally spaced samples to draw from one dimension
                         There will be n_samples**2 points for each vague area
            metric -- Base metric, signature metric(p1, p2), with p1,p2 being 1x2
                      Or one of the following strings: circle_dist, euclidean
        """
        self._check_input(other, self.density_vague_distance)

        grid1 = self.sample_grid(n_samples=n_samples)
        grid2 = other.sample_grid(n_samples=n_samples)

        if metric == "circle_dist":
            metric = circle_dist
        elif metric == "euclidean":
            metric = distance.euclidean

        D = distance.squareform(
            distance.pdist(
                np.vstack((grid1[:, :2], grid2[:, :2])),
                metric=metric
            )
        )

        dist1 = np.hstack((grid1[:, 2], np.zeros(grid2[:, 2].shape[0])))
        dist2 = np.hstack((np.zeros(grid1[:, 2].shape[0]), grid2[:, 2]))

        e = emd(dist1, dist2, D)

        return e

    def fixed_origin_distance(self, other, n_samples=100, metric='euclidean'):
        """Fixed origin distance between self and other.
        See Definition 4.7

        Attributes:
            other -- VagueArea
            n_samples -- Nuber of samples to draw from the combined x/y dimension
            metric -- Base metric, signature metric(p1, p2), with p1,p2 being 1x2
                      Or one of the following strings: circle_dist, euclidean
        """

        self._check_input(other, self.fixed_origin_distance)

        if self.x+self.width > other.x:
            sample_space_x = np.linspace(
                self.x, self.x + self.width, n_samples)
            sample_space_x2 = np.linspace(
                other.x, other.x + other.width, n_samples)
            sample_space_x = np.hstack((sample_space_x2, sample_space_x))

        elif self.x+self.width < other.x:
            sample_space_x = np.linspace(
                self.x, self.x + self.width, n_samples)
            sample_space_x2 = np.linspace(
                other.x, other.x + other.width, n_samples)
            sample_space_x = np.hstack((sample_space_x, sample_space_x2))

        else:
            sample_space_x = np.linspace(
                min(self.x, other.x),
                max(self.x + self.width, other.x + other.width),
                n_samples
            )

        if self.y+self.height > other.y:
            sample_space_y = np.linspace(
                self.y, self.y + self.height, n_samples)
            sample_space_y2 = np.linspace(
                other.y, other.y + other.height, n_samples)
            sample_space_y = np.hstack((sample_space_y2, sample_space_y))

        elif self.y+self.height < other.y:
            sample_space_y = np.linspace(
                self.y, self.y + self.height, n_samples)
            sample_space_y2 = np.linspace(
                other.y, other.y + other.height, n_samples)
            sample_space_y = np.hstack((sample_space_y, sample_space_y2))

        else:
            sample_space_y = np.linspace(
                min(self.y, other.y),
                max(self.y + self.height, other.y + other.height),
                n_samples
            )

        bin_dist_x = distance.squareform(
            distance.pdist(
                sample_space_x.reshape(-1, 1),
                metric=metric
            )
        )

        bin_dist_y = distance.squareform(
            distance.pdist(
                sample_space_y.reshape(-1, 1),
                metric=metric
            )
        )

        samples_x1 = self.pdf_x(sample_space_x)
        samples_x1 /= np.sum(samples_x1)
        samples_x2 = other.pdf_x(sample_space_x)
        samples_x2 /= np.sum(samples_x2)

        samples_y1 = self.pdf_y(sample_space_y)
        samples_y1 /= np.sum(samples_y1)
        samples_y2 = other.pdf_y(sample_space_y)
        samples_y2 /= np.sum(samples_y2)

        emd_x = emd(samples_x1, samples_x2, bin_dist_x)
        emd_y = emd(samples_y1, samples_y2, bin_dist_y)

        return (emd_x+emd_y)/2.

    def sample_points(self, n_samples=1):
        """Sample points from the vague area

        """
        return self.pdf_xy.sample(n_samples=n_samples)

    def sample_grid(self, n_samples=100):
        sample_space_x = np.linspace(self.x, self.x + self.width, n_samples)
        sample_space_y = np.linspace(self.y, self.y + self.height, n_samples)
        xx, yy = np.meshgrid(sample_space_x, sample_space_y)
        points = np.stack((xx, yy), axis=-1).reshape(-1, 2)

        prob = self.pdf_xy(xx, yy)
        prob = prob.reshape(-1, 1)
        prob = prob/prob.sum()

        return np.hstack((points, prob))

    def pdf(self, x, y):
        return self.pdf_xy(x, y)

    def display_pdf(self, axes):
        x, y = np.meshgrid(
            np.linspace(self.x, self.x+self.width, 1000),
            np.linspace(self.y, self.y+self.height, 1000)
        )
        z = self.pdf(x, y)
        z /= np.sum(z)
        cs = axes.contourf(x, y, z)

        return cs

    @property
    def outline(self):
        return (self.x, self.y, self.x+self.width, self.y+self.height)

    def _marginal_pdf(self, dim):
        if dim == "y":
            def func(y):
                return integrate.quad(
                    lambda x: self.pdf_xy(x, y),
                    self.pdf_xy.x1,
                    self.pdf_xy.x2
                )[0]
            return np.vectorize(func)
        elif dim == "x":
            def func(x):
                return integrate.quad(
                    lambda y: self.pdf_xy(x, y),
                    self.pdf_xy.y1,
                    self.pdf_xy.y2
                )[0]
            return np.vectorize(func)

    def _check_input(self, other, expression):
        """Convenience function to check the input."""

        if not isinstance(other, VagueArea):
            raise InputError(
                expression, "other needs to be an instance of {}".format(self.__class__))

    def __repr__(self):
        return "VagueArea(x:{:.2f},y:{:.2f},w:{:.2f},h:{:.2f}) {}".format(
            self.x, self.y, self.width, self.height, self.uid
        )


class Event:
    """Wrapper class for VagueAreas and VagueIntervals.
    Enables direct computation of spatial and temporal distances.
    """
    uid = 0

    def __init__(self, vague_interval, vague_area, **kwargs):
        """
        Attributes:
            vague_interval -- VagueInterval
            vague_area -- VagueArea

            _id -- Additional identifier
        """
        self.vague_interval = vague_interval
        self.vague_area = vague_area
        self.uid = Event.uid
        Event.uid += 1

        self._id = kwargs.get("_id", None)

    def distance(self, other, dist="fixed_origin_distance"):
        """Compute the temporl and spatial distance from self to other.

        Attributes:
            other -- Event
            dist -- String to indicate the function, which should be used.
                    Instance of fixed_origin_distance, density_vague_distance or
                    rv_vague_distance
        """

        if not isinstance(other, Event):
            raise InputError(
                self.distance, "other needs to be an instance of {}".format(self.__class__))

        if dist == "fixed_origin_distance":
            temp_dist = self.vague_interval.fixed_origin_distance(
                other.vague_interval)
            spatial_dist = self.vague_area.fixed_vague_distance(
                other.vague_area)
        elif dist == "density_vague_distance":
            temp_dist = self.vague_interval.density_vague_distance(
                other.vague_interval)
            spatial_dist = self.vague_area.density_vague_distance(
                other.vague_area)
        elif dist == "rv_vague_distance":
            temp_dist = self.vague_interval.rv_vague_distance(
                other.vague_interval)
            spatial_dist = self.vague_area.rv_vague_distance(other.vague_area)
        else:
            raise NotImplementedError("Distance function is not supported")

        return temp_dist, spatial_dist

    def __repr__(self):
        if self.vague_area.name:
            name = self.vague_area.name
        else:
            name = self.vague_area
        return "Event {}: {} during {}".format(self.uid, name, self.vague_interval)
