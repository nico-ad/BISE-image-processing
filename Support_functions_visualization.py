# %% Import librairies

import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, Circle
import mplcursors
import pandas as pd
from collections import defaultdict
import csv
import cv2
from PIL import Image, ImageOps, ImageDraw, ImageFont
import pathlib
from pathlib import Path, PurePath, PureWindowsPath, PurePosixPath
from tkinter import Tcl
import time
import json
import os
import datetime

from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
import seaborn as sns

from inspect import signature
from functools import partial

import statsmodels.api as sm

from skimage import io, measure, morphology, filters
from skimage.measure import (
    label,
    regionprops,
    regionprops_table,
    moments,
    moments_central,
    moments_normalized,
    moments_hu,
)
from skimage.feature import peak_local_max, canny
from skimage.segmentation import watershed, clear_border
from skimage.exposure import equalize_hist, equalize_adapthist
from skimage.filters import threshold_sauvola, threshold_niblack

import scipy
from scipy.ndimage import (
    label,
    distance_transform_edt,
    gaussian_filter,
    binary_fill_holes,
)
from scipy.spatial import distance, distance_matrix, cKDTree, Delaunay, Voronoi
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.optimize import linear_sum_assignment, curve_fit, OptimizeWarning
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline, griddata
from scipy import io

from shapely.geometry import Polygon, box, Point

from lap import lapjv

from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

from multiprocessing import Manager
import multiprocessing as mp
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue as pyqueue

import pyarrow as pa
import pyarrow.parquet as pq

import warnings

warnings.filterwarnings("ignore", category=OptimizeWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from Support_functions_preview import FitFunction

# %% Create class

# class KalmanFilter:
#     def __init__(self, F, B, H, Q, R, x0, P0):
#         self.F = F
#         self.B = B
#         self.H = H
#         self.Q = Q
#         self.R = R
#         self.x = x0
#         self.P = P0

#     def predict(self, u):
#         """ predict position """

#         self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
#         self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
#         return self.x

#     def update(self, z):
#         """ Update position """

#         S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R # residual covariance
#         K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S)) # Kalman gain
#         y = z - np.dot(self.H, self.x)
#         self.x = self.x + np.dot(K, y)
#         I = np.eye(self.P.shape[0])
#         self.P = np.dot(I - np.dot(K, self.H), self.P)
#         return self.x


class VisualizationFunctions:
    def __init__(self, parent=None):
        pass

    # # def _compute_target_particles(self, group, length=1024):
    # #     positions = group[["x", "y"]].to_numpy()
    # #     diameters = group["diameter_mean"].to_numpy()

    # #     dx = positions[:, 0][:, np.newaxis] - positions[:, 0][np.newaxis, :]
    # #     dy = positions[:, 1][:, np.newaxis] - positions[:, 1][np.newaxis, :]

    # #     mask = (dx >= 0) & (dx <= length) & (np.abs(dy) < diameters[:, np.newaxis] / 2)
    # #     np.fill_diagonal(mask, False)

    # #     counts = np.sum(mask, axis=1)

    # #     rectangles = []
    # #     for (x, y), d in zip(positions, diameters):
    # #         half_d = d / 2
    # #         xmin = y - half_d
    # #         xmax = y + half_d
    # #         ymin = x
    # #         ymax = x + length
    # #         rectangles.append((xmin, ymin, xmax, ymax))
    # #     return counts, np.array(rectangles)

    # def _compute_target_particles(self, group):

    #     positions = group[["x", "y"]].to_numpy()
    #     diameters = group["diameter_mean"].to_numpy()

    #     counts, rectangles = [], []
    #     for (x, y), d in zip(positions, diameters):
    #         r = d / 2.0
    #         xmin = x - r
    #         xmax = x + r
    #         ymin = y
    #         ymax = 0
    #         rectangles.append((xmin, ymin, xmax, ymax))
    #         coords = ((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin,ymax), (xmin, ymin))
    #         rect = Polygon(coords)
    #         points_inside = [p for p in positions if Point(p).within(rect)]
    #         counts.append(len(points_inside) / (rect.area*d))
    #     return np.array(counts), np.array(rectangles)

    def _load_image(
        self,
        name: str | Path = None,
        invert: bool = True,
        rotate_image: int = None,
        crop: list = None,
        flip_left_right: bool = False,
    ) -> np.ndarray:

        img = Image.open(name).convert("L")

        if invert:
            img = ImageOps.invert(img)
        img = np.array(img, dtype=np.float32)

        if crop is not None:
            img = img[
                self.cropping_image[0] : self.cropping_image[1],
                self.cropping_image[2] : self.cropping_image[3],
            ]

        if rotate_image is not None:
            img = Image.fromarray(img)
            if rotate_image == 90:
                img = img.transpose(Image.ROTATE_90)
            elif rotate_image == 180:
                img = img.transpose(Image.ROTATE_180)
            elif rotate_image == 270:
                img = img.transpose(Image.ROTATE_270)

        return np.array(img, dtype=np.uint8)

    def _compute_coordination_number(
        self,
        df: pd.DataFrame = None,
        do_plot: bool = False,
        eps=0,
        ratio_frame=1,
    ) -> pd.DataFrame:
        """Compute coordination number for each particle"""

        df = df.copy()
        df["coordination"] = None

        for fr, group in df.groupby("frame"):

            # if fr % 2 != 0:
            #     print(f"BAD : {fr}, {fr % 2}")
            #     continue
            # print(f"GOOD : {fr}, {fr % 2}")

            pts = group[["x", "y"]].values
            d = group["diameter_mean"].values

            tree = cKDTree(pts)
            coord = np.zeros(len(group), dtype=int)

            # define max radius
            r_max = d.max()
            r_max += 0.1 * r_max

            for i in range(len(group)):
                neighbors = tree.query_ball_point(pts[i], r_max)
                neighbors.remove(i)
                neighbors = [
                    j
                    for j in neighbors
                    if np.linalg.norm((pts[j] - pts[i])) >= (d[i] + eps) / 2
                ]
                coord[i] = len(neighbors)
            df.loc[group.index, "coordination"] = coord

        return df

    def Visualize_num_part_per_frames(
        self,
        dataframe: pd.DataFrame,
        velocity: pd.DataFrame = None,
        pixel_size: float = None,
        frames: int | list | np.ndarray = None,
        labels: int | list | np.ndarray = None,
        time_interval: float = 1 / 8000,
        x_unit: str = "time",
        y_unit: str = "number",
        rolling_mean: int = None,
        initial_amount: bool = True,
        final_amout: bool = True,
        language: str = "en",
    ):

        if velocity is None:
            velocity = [None] * len(dataframe)

        # if unit not in ["frame", "time", "velocity", "Reynolds_duct", "Reynolds_friction"]:
        #     msg = f"'unit' must be frame or time or velocity or Reynolds_duct or Reynolds_friction, not {unit}"
        #     raise ValueError(msg)

        _, ax = plt.subplots()

        results = []

        for run, (data, vel) in enumerate(zip(dataframe, velocity)):
            mask_keys = ["frame", "main_path", "name", "time", "label"]

            if isinstance(frames, int):
                frames = [frames]
            if frames is None:
                frames = [dataframe["frame"].unique()]
            if not isinstance(frames, (int, np.ndarray, list)):
                msg = "'frames' must be int, list or ndarray of int"
                raise TypeError(msg)

            labels = data["label"].unique()

            mask_frames = data["frame"].isin(frames)
            mask_labels = data["frame"].isin(frames)
            mask = mask_frames & mask_labels

            sub_df = data.loc[mask, mask_keys].copy()

            if sub_df.empty:
                continue

            unit_factor_x = {
                "frame": 1,
                "time": time_interval,
            }[x_unit]

            unit_factor_y = {
                "number": 1,
                "ratio": 1,
            }[y_unit]

            # ----- group data
            grouped = sub_df.groupby("frame")

            data_dict = {
                "curves": True,
                "x": sub_df["frame"].unique(),
                "y": [len(group["label"].unique()) for _, group in grouped],
                "run": [run],
                "x_log": False,
                "label_curve": "Number of different labels in each frames",
                "x_unit": unit_factor_x,
                "y_unit": unit_factor_y,
                "x_label": "Time [s]" if unit_factor_x == "time" else "Frames",
                "y_label": "Number of particules",
            }

            results.append(data_dict)

        return results

        # mask_keys_vel = [
        #     "frame", "timestamp", "voltage", "velocity",
        #     ]

        # idx = frames * self.time_interval
        # mask = velocity["timestamp"].isin(idx)

        # sub_df = velocity.loc[
        #     mask,
        #     keys,
        # ].copy()

        # label_per_frame = sub_df["frame"].value_counts().sort_index()
        # labels_first_frame = set(sub_df[sub_df["frame"] == min(sub_df["frame"].unique())]["label"].to_numpy())
        # labels_last_frame = set(sub_df[sub_df["frame"] == max(sub_df["frame"].unique())]["label"].to_numpy())
        # print(f"Number of particles detached between first and last frame : {len(labels_last_frame) - len(labels_first_frame)}")
        # print(f"{len(labels_last_frame - labels_first_frame)}")

        # if rolling_mean is None:
        #     rolling_mean = len(label_per_frame)

        # if initial_amount:
        #     first_frame = sub_df["frame"].unique().min()
        #     initial_labels = sub_df[sub_df["frame"] == first_frame]["label"].to_numpy()

        # if final_amout:
        #     last_frame = sub_df["frame"].unique().min()
        #     final_labels = sub_df[sub_df["frame"] == last_frame]["label"].to_numpy()

        # if initial_amount and final_amout:
        #     ratio = []
        #     for _, group in sub_df.groupby("frame"):
        #         ratio.append(len(initial_labels) / len(group))

        #     ax_twin = ax.twinx()
        #     ax_twin.plot(
        #         frames,
        #         ratio,
        #         color="tab:green",
        #         label="Ratio of remaining particles" if language == "en" else "Proportion de particules restantes",
        #     )

    def Visualize_mean_inter_particle_distance(
        self,
        path_dataframe: str | list = None,
        frames: int | list | np.ndarray = None,
        max_distance: float = None,
        divisor="mean_diameter_per_frame",
        do_plot=False,
        unit: str = None,
    ):

        # check if columns exists
        if not isinstance(path_dataframe, (list, str, Path)):
            msg = "dataframe must be list or pd.DataFrame type"
            raise TypeError(msg)
        if not isinstance(path_dataframe, list):
            path_dataframe = [path_dataframe]

        for i, path_data in enumerate(path_dataframe):
            print(path_data)

            dataframe = self._load_dataframe(path_data)

            keys = [
                "frame",
                "main_path",
                "name",
                "time",
                "label",
                "x",
                "y",
                "diameter_mean",
                "diameter_std",
                # "collision",
            ]
            self._check_keys(dataframe, keys)

            if isinstance(frames, int):
                frames = [frames]
            if frames is None:
                frames = [dataframe["frame"].unique()]
            if not isinstance(frames, (int, np.ndarray, list)):
                msg = "frames variable must be int, list of ndarray of int"
                raise TypeError(msg)

            mask = dataframe["frame"].isin(frames)

            sub_df = dataframe.loc[
                mask,
                keys,
            ].copy()

            for i, (frame_id, group) in enumerate(sub_df.groupby("frame")):
                self.mean_dist = self._compute_mean_distance(df=sub_df, divisor=divisor)

                if do_plot:
                    fig, ax = plt.subplots()
                    ax.plot(
                        self.mean_dist[:, 0],
                        self.mean_dist[:, 1],
                    )

                    x_ticks = ax.get_xticks()[1:-1]
                    ax.set_xticks(x_ticks)
                    if unit == "time":
                        ax.set_xticklabels(
                            [
                                f"{x_value * self.time_interval:.2f}"
                                for x_value in np.array(ax.get_xticks())
                            ],
                            fontsize=16,
                        )
                        ax.set_xlabel("Time $[s]$", fontsize=20)
                    elif unit == "frame":
                        ax.set_xticklabels(
                            [f"{x_value:.0f}" for x_value in np.array(ax.get_xticks())],
                            fontsize=16,
                        )
                        ax.set_xlabel("Frame number", fontsize=20)

                    y_ticks = ax.get_yticks()[1:-1]
                    ax.set_yticks(y_ticks)
                    ax.set_yticklabels(
                        [f"{y_value:.2f}" for y_value in np.array(ax.get_yticks())],
                        fontsize=16,
                    )
                    ax.set_ylabel(
                        "Inter particle distance per frame : $\\dfrac{{L}}{{d}}$",
                        fontsize=18,
                    )

                    plt.show()
        else:
            msg = "'x', 'y' variable not in dataframe"
            raise KeyError(msg)

    def Visualize_max_label_per_frames(self):
        sub_df = self.dataframe.copy()
        max_label_per_frame = sub_df.groupby("frame")["label"].max()
        fig, ax = plt.subplots()
        ax.bar(max_label_per_frame.index, max_label_per_frame.values)
        ax.hlines(
            y=max_label_per_frame.values.mean(),
            xmin=max_label_per_frame.index.min(),
            xmax=max_label_per_frame.index.max(),
            color="tab:red",
        )
        plt.show()

    def _compute_mean_free_path(
        self,
        df: pd.DataFrame = None,
        frames: int | list | np.ndarray = None,
    ) -> list:
        """ Compute mean free path """

        if frames is None:
            frames = df["frame"].to_numpy()
        elif isinstance(frames, int):
            frames = [frames]
        elif not isinstance(frames, (list, np.ndarray)):
            msg = "frames must be list or ndarray"
            raise TypeError(msg)
        
        # ----- compute density
        density = self._compute_surface_concentration(
            df=df, frames=frames,
            img_size=[df["image_width"].unique(), df["image_height"].unique()],
            cut=(1, 1),
        )  # [#/mm-2]

        # ----- compute mean diameter
        mean_diameter = [
            group.loc[group["diameter_mean"] > 0.0, "diameter_mean"].mean()
            for _, group in df.groupby("frame")
        ]  # [mm]

        # ----- compute mean free path
        if all(density) > 0 and all(mean_diameter) > 0:
            mean_free_path = 1.0 / (density * mean_diameter)  # [mm]
        else:
            mean_free_path = 0.0
        
        return mean_free_path  # [mm]

    def _compute_mean_distance(
        self, df: pd.DataFrame, divisor: str, max_distance: float = None
    ) -> np.ndarray:
        results = []

        if divisor == "mean_diameter_total":
            global_mean_diameter = df["diameter"].mean()  # [mm]
        elif divisor == "d_50":
            global_median_diameter = np.median(df["diameter"].unique())  # [mm]

        for frame_id, group in df.groupby("frame"):
            if max_distance is None:
                max_distance = np.inf
            else:
                max_distance = 2 * group["diameter"].max()
            coords = group[["x", "y"]].to_numpy()  # [mm]
            kd_tree = scipy.spatial.KDTree(coords)
            pairs = kd_tree.query_pairs(r=max_distance)
            pair_list = list(pairs)

            if pair_list:
                indices_i, indices_j = np.array(pair_list).T
                distances = np.linalg.norm(
                    coords[indices_i] - coords[indices_j], axis=1
                )  # [mm]
                mean_distance = distances.mean()  # [mm]
            else:
                mean_distance = 0.0  # [mm]
            if divisor == "mean_diameter_per_frame":
                div = group["diameter_mean"]  # [mm]
            elif divisor == "mean_diameter_total":
                div = global_mean_diameter  # [mm]
            elif divisor == "d_50":
                div = global_median_diameter  # [mm]

            normalized_distance = mean_distance / div if div != 0 else 0.0
            results.append((frame_id, normalized_distance))  # [#], [/]

        return np.array(results)  # [#], [/]

    def _compute_collision_frequency(
        self,
        velocity: list | np.ndarray = None,
        mean_free_path: np.ndarray = None,
    ) -> list:

        freq_coll = []
        if (mean_free_path.all() != 0):
            freq_coll = velocity / (mean_free_path)

        return list(freq_coll)

        # results = []

        # for frame_id, group in df.groupby("frame"):
        #     coll_mask = group["collision"]
        #     velocity = group.loc[coll_mask, "velocity"].to_numpy()

        #     match_idx = np.where(mean_free_path[:, 0] == frame_id)[0]
        #     if len(match_idx) == 0 or velocity.size == 0:
        #         freq_mean = 0.0
        #     else:
        #         path = mean_free_path[match_idx[0], 1]
        #         if path != 0:
        #             freq_mean = (velocity / path).mean()
        #         else:
        #             freq_mean = 0.0

        #     results.append((frame_id, freq_mean))

        # return np.array(results, dtype=float)

    # def _compute_surface_concentration(self, df:pd.DataFrame, frames:int|list|np.ndarray, img_size:list|np.ndarray=[1024, 512], cut: tuple = (1, 1)):

    #     if not isinstance(df, pd.DataFrame):
    #         msg = "df must be dataframe"
    #         raise TypeError(msg)

    #     if frames is None:
    #         frames = self.dataframe["frame"].to_numpy()
    #     elif isinstance(frames, int):
    #         frames = [frames]
    #     elif not isinstance(frames, (list, np.ndarray)):
    #         msg = "frames must be list or ndarray"
    #         raise TypeError(msg)

    #     if not isinstance(img_size, (list|np.ndarray)):
    #         msg = "img_size must be list or array"
    #         raise TypeError(msg)

    #     if not isinstance(cut, tuple):
    #         msg = "cut must be tuple"
    #         raise TypeError(msg)

    #     num_labels = len(df["label"].unique())

    #     # compute total surface concentration
    #     total_surface_concentration = num_labels / (
    #         np.prod(img_size) * (self.pixel_size) ** 2
    #     ) # [#/mm^2]

    #     if cut == (1, 1) or cut is None:

    #         if len(frames) == 1:
    #             total_surface_concentration = []
    #             c = len(df["label"]) / (np.prod(img_size) * (self.pixel_size) ** 2)
    #             total_surface_concentration.append((frames[0], c))
    #         else:
    #             total_surface_concentration = []
    #             for frame_id, group in df.groupby("frame"):
    #                 c = len(group["label"]) / (np.prod(img_size) * (self.pixel_size) ** 2)
    #                 total_surface_concentration.append((frame_id, c))
    #         return total_surface_concentration # [#/mm^2]

    #     # cut surface image in cut
    #     cut_h = [0] + [
    #         ii
    #         for ii in range(
    #             int(img_size[0] / cut[0]), img_size[0] + 1, int(img_size[0] / cut[0])
    #         )
    #     ]

    #     cut_v = [0] + [
    #         ii
    #         for ii in range(
    #             int(img_size[1] / cut[1]), img_size[1] + 1, int(img_size[1] / cut[1])
    #         )
    #     ]

    #     # compute surface concentration per zone
    #     def find_zone(x, y, cut_h, cut_v):
    #         return next(
    #             (
    #                 (i, j)
    #                 for i in range(len(cut_h) - 1)
    #                 for j in range(len(cut_v) - 1)
    #                 if cut_h[i] <= y < cut_h[i + 1] and cut_v[j] <= x < cut_v[j + 1]
    #             ),
    #             None,
    #         )

    #     zone_counts = {}
    #     zones = df.apply(
    #         lambda row: find_zone(
    #             row["x"] / self.pixel_size, # [px]
    #             row["y"] / self.pixel_size, # [px]
    #             cut_h, cut_v,
    #         ),
    #         axis=1,
    #     )
    #     zone_counts_frame = zones.value_counts().to_dict()
    #     zone_counts[frame] = zone_counts_frame
    #     concentrations = []
    #     for ii, (y1, y2) in enumerate(zip(cut_h[:-1], cut_h[1:])):
    #         for jj, (x1, x2) in enumerate(zip(cut_v[:-1], cut_v[1:])):
    #             zone_id = (ii, jj)
    #             # center = ((x1 + x2) // 2, (y1 + y2) // 2)
    #             surface = (
    #                 (cut_h[ii + 1] - cut_h[ii])
    #                 * (cut_v[jj + 1] - cut_v[jj])
    #                 * (self.pixel_size) ** 2
    #             ) # [mm^2]
    #             count = sum(zone_counts[zone].get(zone_id, 0) for zone in zone_counts) # [#]
    #             concentrations.append(count / surface) # [#/mm^2]
    #     return total_surface_concentration, concentrations, cut_v, cut_h, zone_counts # [#/mm^2], [#/mm^2], [px], [px]

    # def _rotate_coords(
    #     self,
    #     points,
    #     width,
    #     height,
    #     angle,
    #     ):

    #     rotated = []

    #     for x, y in points:

    #         x_norm = x / width
    #         y_norm = y / height

    #         if angle == 90:
    #             x_new_norm = y_norm
    #             y_new_norm = 1 - x_norm
    #             new_w, new_h = height, width

    #         elif angle == 180:
    #             x_new_norm = 1 - x_norm
    #             y_new_norm = 1 - y_norm
    #             new_w, new_h = width, height

    #         elif angle == 270:
    #             x_new_norm = 1 - y_norm
    #             y_new_norm = x_norm
    #             new_w, new_h = height, width

    #         else:
    #             msg = "Angle must be 90, 180 or 270 degrees"
    #             raise ValueError(msg)

    #         x_new = x_new_norm * new_w
    #         y_new = y_new_norm * new_h

    #         rotated.append((x_new, y_new))

    #     return np.array(rotated)

    def _rotate_coords(
        self,
        points: list | np.ndarray = None,
        width: int = None,
        height: int = None,
        angle: int = None,
        center: list | np.ndarray = None,
    ) -> np.ndarray:

        if angle in [90, 270]:
            new_w, new_h = height, width
        elif angle == 180:
            new_w, new_h = width, height
        else:
            msg = "Angle must be 90, 180 or 270 degrees"
            raise ValueError(msg)

        if points.shape[0] == 0:
            return np.empty((0, 2))

        if center is None:
            cx, cy = width / 2, height / 2
        else:
            cx, cy = center

        new_cx, new_cy = new_w / 2, new_h / 2

        theta = np.radians(-angle)

        rotated = []
        for x, y in points:
            # shift
            x_c, y_c = x - cx, y - cy

            # rotate
            x_r = x_c * np.cos(theta) - y_c * np.sin(theta)
            y_r = x_c * np.sin(theta) + y_c * np.cos(theta)

            # repositionning
            x_new = x_r + new_cx + new_w
            y_new = y_r + new_cy

            rotated.append((x_new, y_new))

        return np.array(rotated)

    def _compute_surface_concentration(
        self,
        df: pd.DataFrame,
        frames: int | list | np.ndarray = None,
        pixel_size: float = 1.0,
        img_size: list | np.ndarray = [1024, 512],
        cut: tuple = (1, 1),
    ):

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a DataFrame")

        if frames is None:
            frames = df["frame"].unique()
        elif isinstance(frames, int):
            frames = [frames]
        elif not isinstance(frames, (list, np.ndarray)):
            raise TypeError("'frames' must be int, list or ndarray")

        if not isinstance(img_size, (list, np.ndarray)):
            raise TypeError("'img_size' must be a list or array")

        if not isinstance(cut, tuple):
            raise TypeError("'cut' must be a tuple")

        df = df[df["frame"].isin(frames)]

        if cut == (1, 1) or cut is None:
            surface = np.prod(img_size) * pixel_size**2
            num_labels = df["frame"].value_counts().sort_index()
            density = (num_labels / surface).to_numpy()  # [#/mm^2]
            if density.size == 1:
                return float(density[0])
            else:
                return density

        cut_h = [0] + [int(i) for i in np.linspace(img_size[0], img_size[0], cut[0])]
        cut_v = [0] + [int(i) for i in np.linspace(img_size[1], img_size[1], cut[1])]

        def find_zone(x_px, y_px, cut_h, cut_v):
            for i in range(len(cut_h) - 1):
                for j in range(len(cut_v) - 1):
                    if (
                        cut_h[i] <= y_px < cut_h[i + 1]
                        and cut_v[j] <= x_px < cut_v[j + 1]
                    ):
                        return (i, j)
            return None

        df = df.copy()
        df["zone"] = df.apply(
            lambda row: find_zone(
                row["x"] / pixel_size,
                row["y"] / pixel_size,
                cut_h,
                cut_v,
            ),
            axis=1,
        )

        results = []
        for frame_id, group in df.groupby("frame"):
            zone_counts = group["zone"].value_counts().to_dict()

            for i, (y1, y2) in enumerate(zip(cut_h[:-1], cut_h[1:])):
                for j, (x1, x2) in enumerate(zip(cut_v[:-1], cut_v[1:])):
                    zone_id = (i, j)
                    count = zone_counts.get(zone_id, 0)
                    surface = (y2 - y1) * (x2 - x1) * pixel_size**2
                    concentration = count / surface if surface > 0 else 0.0
                    results.append((frame_id, zone_id, concentration))

        return results

    def Visualize_mean_free_path(
        self,
        path_dataframe: pd.DataFrame = None,
        frames: int | list | np.ndarray = None,
        mode: str = "separate",
        unit: str = "time",
        language: str = "en",
        do_plot=False,
    ):

        if not isinstance(path_dataframe, (list, str, Path)):
            msg = "dataframe must be list or pd.DataFrame type"
            raise TypeError(msg)
        if not isinstance(path_dataframe, list):
            path_dataframe = [path_dataframe]

        if mode not in ["together", "separate"]:
            msg = f"'mode' can be together or separate, not {mode}"
            raise ValueError(msg)

        if mode == "together":
            fig, ax = plt.subplots()

        for i, path_data in enumerate(path_dataframe):
            print(path_data)

            keys = [
                "frame",
                "main_path",
                "name",
                "time",
                "label",
                "image_size",
            ]

            dataframe = self._load_dataframe(
                path_data,
                usecols=keys,
            )
            self._check_keys(dataframe, keys)

            if isinstance(frames, int):
                frames = [frames]
            if frames is None:
                frames = [dataframe["frame"].unique()]
            if not isinstance(frames, (int, np.ndarray, list)):
                msg = "'frames' must be int, list or ndarray of int"
                raise TypeError(msg)

            mask = dataframe["frame"].isin(frames)

            sub_df = dataframe.loc[
                mask,
                keys,
            ].copy()

            mean_free_path = self._compute_mean_free_path(df=sub_df)
            print(frames)
            print(mean_free_path)
            print(len(frames), len(mean_free_path))

            if mode == "separate":
                fig, ax = plt.subplots()

            ax.plot(
                frames,
                mean_free_path,
                color="tab:blue",
            )

        x_ticks = ax.get_xticks()[1:-1]
        y_ticks = ax.get_yticks()[1:]
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)

        if unit == "time":
            if language == "en":
                ax.set_xlabel("Time $[s]$", fontsize=self.dict_fontsize["label"])
            if language == "fr":
                ax.set_xlabel("Temps $[s]$", fontsize=self.dict_fontsize["label"])
            ax.set_xticklabels(
                [f"{x_tick * self.time_interval:.0f}" for x_tick in x_ticks],
                fontsize=self.dict_fontsize["ticks"],
            )

        elif unit == "frame":
            if language == "en":
                ax.set_xlabel("Frame number", fontsize=self.dict_fontsize["label"])
            if language == "fr":
                ax.set_xlabel("Image", fontsize=self.dict_fontsize["label"])
            ax.set_xticklabels(
                [f"{x_tick:.0f}" for x_tick in x_ticks],
                fontsize=self.dict_fontsize["ticks"],
            )

        if language == "en":
            ax.set_ylabel(
                "Mean free path $\\lambda_m$ $[mm]$",
                fontsize=self.dict_fontsize["label"],
            )
        if language == "fr":
            ax.set_ylabel(
                "Libre parcours moyen $\\lambda_m$ $[mm]$",
                fontsize=self.dict_fontsize["label"],
            )
        ax.set_yticklabels(
            [f"{y_tick:.2f}" for y_tick in y_ticks],
            fontsize=self.dict_fontsize["ticks"],
        )

        plt.subplots_adjust(**self.dict_fontsize["subplots"])

        plt.show()

    def Visualize_collision_frequency(
        self,
        dataframe: list | str = None,
        velocity: list | str = None,
        pixel_size: float = 1.0,
        frames: list | int = None,
        labels: list | int = None,
        x_unit: str = "time",
        y_unit: str = "frequency",
    ):

        if isinstance(frames, int):
            frames = [frames]

        result = []

        for run, (data, vel) in enumerate(zip(dataframe, velocity)):
            mask_keys = [
                "frame",
                "label",
                "dt", "diameter_mean",
                "image_height", "image_width",
            ]

            if isinstance(frames, int):
                frames = [frames]
            if frames is None:
                frames = data["frame"].unique()
            if not isinstance(frames, (int, np.ndarray, list)):
                msg = "'frames' must be int, list or ndarray of int"
                raise TypeError(msg)

            mask_frames = data["frame"].isin(frames)
            mask = mask_frames
            sub_df = data.loc[mask, mask_keys].copy()
            frames = sub_df["frame"].unique()

            if vel is not None:
                keys = [
                    "frame",
                    "timestamp",
                    "voltage",
                    "velocity",
                ]

                mask = vel["frame"].isin(frames)

                vel = vel.loc[mask, keys].copy()

                if "friction" not in vel:
                    vel["friction"] = 0.0564 * vel["velocity"] ** (7 / 8)  # m/s

            mean_free_path = self._compute_mean_free_path(df=sub_df)

            collision_frequency = self._compute_collision_frequency(
                vel["friction"], mean_free_path,
            )

            dt = sub_df["dt"].unique()

            unit_factor_x = {
                "frames": 1,
                "time": dt,
                "fric_velocity": 1.0,
                "flow_velocity": 1.0,
                "reynolds": 1.0,
            }[x_unit]

            unit_label_x = {
                "frames": "Frames",
                "time": "Time $[s]$",
                "fric_velocity": "Friction velocity [m/s]",
                "flow_velocity": "Flow velocity [m/s]",
                "reynolds": "Reynold number",
            }[x_unit]

            min_decimals_x = {
                "frames": 0,
                "time": 3,
                "fric_velocity": 3,
                "flow_velocity": 3,
                "reynolds": 3,
            }[x_unit]

            unit_factor_y = {
                "frequency_s": 1,
                "frequency_ks": 1e3,
                "frequency_ms": 1e6,
            }[y_unit]

            unit_label_y = {
                "frequency_s": "Frequency $[{{s^-1}}]$",
                "frequency_ks": "Frequency $[{{10^3 s^-1}}]$",
                "frequency_ms": "Frequency $[{{10^6 s^-1}}]$",
            }[y_unit]

            min_decimals_y = {
                "frequency_s": 3,
                "frequency_ks": 3,
                "frequency_ms": 3,
            }[y_unit]

            df = pd.DataFrame(columns=["frame", "coll_freq"])
            df["frame"] = frames
            df["coll_freq"] = collision_frequency

            grouped_frame = df.groupby("frame")

            data_dict = {
                "curves": True,
                "x": [group["frame"].to_numpy() for _, group in grouped_frame],
                "y": [group["coll_freq"].to_numpy() for _, group in grouped_frame],
                "run": [run],
                # "fit": [fit_func],
                "x_log": False,
                "x_unit": unit_factor_x,
                "y_unit": unit_factor_y,
                "x_label": unit_label_x,
                "y_label": unit_label_y,
                "min_decimals_x": min_decimals_x,
                "min_decimals_y": min_decimals_y,
            }

            result.append(data_dict)

        return result

    def Visualize_coordination_number(
        self,
        dataframe: pd.DataFrame = None,
        frames: int | list | np.ndarray = None,
        labels: int | list | np.ndarray = None,
        time_interval: float = 1 / 8000,
        pixel_size: float = None,
        x_unit: str = "coord_num",
        y_unit: str = "frequency",
        z_unit: str = "frames",
        normalize: bool = True,
        velocity: pd.DataFrame = None,
    ) -> list:

        result = []

        for i, data in enumerate(dataframe):
            mask_keys = [
                "frame",
                "time",
                "dt",
                "label",
                "coordination",
                "x",
                "y",
                "diameter_mean",
                "main_path",
                "name",
            ]

            # ----- frames
            if frames is None:
                selected_frames = data["frame"].unique()
            elif isinstance(frames, int):
                selected_frames = [frames]
            elif isinstance(frames, (list, np.ndarray)):
                selected_frames = frames
            else:
                msg = "'frames' must be int, list or np.ndarray of int"
                raise TypeError(msg)

            # ----- labels
            if labels == "all" or labels is None:
                selected_labels = sorted(data["label"].unique())
            else:
                selected_labels = labels

            # ----- filtering
            mask = data["frame"].isin(selected_frames) & data["label"].isin(
                selected_labels
            )

            sub_df = data.loc[
                mask,
                mask_keys,
            ].copy()

            if sub_df.empty:
                continue

            sub_df = self._compute_coordination_number(df=sub_df, eps=15, ratio_frame=1)
            # sub_df.dropna(inplace=True)
            time_interval = sub_df["dt"].unique()

            # ----- unit factor
            unit_factor_x = {
                "coord_num": 1,
            }[x_unit]

            unit_factor_y = {
                "count": 1,
                "frequency": 1,
            }[y_unit]

            unit_factor_z = {
                "frames": 1,
                "time_s": time_interval,
                "time_ms": time_interval * 1000,
            }[z_unit]

            # ----- unit label
            unit_label_x = {
                "coord_num": "Coordination number",
            }[x_unit]

            unit_label_y = {
                "count": "Counts",
                "frequency": "Frequency",
            }[y_unit]

            unit_label_z = {
                "frames": "Frames",
                "time_s": "Time $[s]$",
                "time_ms": "Time $[ms]$",
            }[z_unit]

            df = sub_df.sort_values(by="frame")
            grouped_frames = df.groupby("frame")

            max_coord_number = np.max(df["coordination"].unique())
            # print(f"Max number of coordination number : {max_coord_number}")
            data_dict = {
                "hist_3d": True,
                "x": [
                    np.histogram(
                        group["coordination"],
                        bins=np.arange(0, max_coord_number + 2, 1, dtype=int),
                        density=False,
                    )[1]
                    for _, group in grouped_frames
                ],
                "y": [
                    np.histogram(
                        group["coordination"],
                        bins=np.arange(0, max_coord_number + 2, 1, dtype=int),
                        density=False,
                    )[0]
                    for _, group in grouped_frames
                ],
                "z": [group["frame"].unique() for _, group in grouped_frames],
                "run": [i],
                # "fit": [fit_func],
                "x_log": False,
                "nomalize": True,
                "x_unit": unit_factor_x,
                "y_unit": unit_factor_y,
                "z_unit": unit_factor_z,
                "x_label": unit_label_x,
                "y_label": unit_label_y,
                "z_label": unit_label_z,
            }

            # # display
            # if True:

            #     _, ax = plt.subplots()

            #     name = Path(
            #     df['main_path'].iloc[0],
            #     df['name'].iloc[0]
            #     )
            #     img = Image.open(name).convert("L")
            #     img = np.array(img, dtype=np.float32)
            #     ax.imshow(img, cmap="gray")

            #     # plot coords + radius
            #     for _, group in df.groupby("frame"):
            #         x = group["x"].values
            #         y = group["y"].values
            #         di = group["diameter_mean"].values
            #         r_max = di.max()
            #         tol = r_max

            #         for i, j, d in zip(x, y, di):
            #             ax.scatter(x, y)
            #             ax.add_patch(plt.Circle((i, j), d/2, color="b", fill=False))

            #             # plot research radius
            #             ax.add_patch(plt.Circle((i, j), r_max, color="k", fill=False))
            #             ax.add_patch(plt.Circle((i, j), r_max+tol, color="r", fill=False))

            #         plt.show()

            result.append(data_dict)

        return result

    def Concentration_vs_number_collision(
        self,
        frames: int | list | np.ndarray = None,
        do_plot=False,
        do_save: bool = False,
        path_save: Path | str = None,
    ):

        if frames is None:
            frames = self.dataframe["frame"].to_numpy()
        elif isinstance(frames, int):
            frames = [frames]
        elif not isinstance(frames, (list, np.ndarray)):
            msg = "frames must be list or ndarray"
            raise TypeError(msg)
        if not all(
            key in self.dataframe.keys()
            for key in [
                "frame",
                "label",
                "x",
                "y",
                "diameter",
                "velocity",
                "collision",
                "collision_id",
            ]
        ):
            msg = "Missing key"
            raise KeyError(msg)

        if path_save is None and do_save:
            msg = "Must specify a path to save picture"
            raise TypeError(msg)
        if path_save is not None and do_save is None:
            do_save = True

        sub_df = self.dataframe[
            [
                "frame",
                "label",
                "x",
                "y",
                "diameter",
                "velocity",
                "collision",
                "collision_id",
            ]
        ].copy()

        concentration = self._compute_surface_concentration(
            df=sub_df, frames=None, img_size=[1024, 512], cut=(1, 1)
        )
        mean_free_path = self._compute_mean_free_path(sub_df, frames=None)
        collision_frequency = self._compute_collision_frequency(sub_df, mean_free_path)

        conc_values = concentration[:, 1]
        freq_values = collision_frequency[:, 1]

        fig, ax = plt.subplots()

        # ax.hist(x=freq_values, bins=conc_values)
        ax.scatter(conc_values, freq_values)

        x_ticks = ax.get_xticks()[1:-1]
        y_ticks = ax.get_yticks()[1:]
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels([f"{x_tick:.3f}" for x_tick in x_ticks], fontsize=16)
        ax.set_yticklabels([f"{y_tick:.0f}" for y_tick in y_ticks], fontsize=16)
        ax.set_xlabel("Concentration [#/$mm^2$]", fontsize=18)
        ax.set_ylabel("Collision frequency $\\nu_c$", fontsize=18)

        # # histogram
        # axs[1].hist(x=conc_values, bins=np.linspace(min(freq_values), max(freq_values), 10))

        # x_ticks = axs[1].get_xticks()[1:-1]
        # y_ticks = axs[1].get_yticks()[1:]
        # axs[1].set_xticks(x_ticks)
        # axs[1].set_yticks(y_ticks)
        # axs[1].set_xticklabels([f"{x_tick:.3f}" for x_tick in x_ticks], fontsize=16)
        # axs[1].set_yticklabels([f"{y_tick:.0f}" for y_tick in y_ticks], fontsize=16)
        # axs[1].set_xlabel("Collision frequency $\\nu_c$", fontsize=18)
        # axs[1].set_ylabel("Concentration [#/$mm^2$]", fontsize=18)

        plt.show()

    def Visualize_concentration_vs_time(
        self,
        path_dataframe: str | Path = None,
        curve_names: str | list = None,
        frames: int | list | np.ndarray = None,
        unit: str = "time",
        path_velocity: str | Path = None,
        do_smooth: bool = True,
        do_save: bool = False,
        path_save: Path | str = None,
        mode: str = "together",
        language: str = "en",
    ):

        if not isinstance(path_dataframe, (list, str, Path)):
            msg = "dataframe must be list or pd.DataFrame type"
            raise TypeError(msg)
        if not isinstance(path_dataframe, list):
            path_dataframe = [path_dataframe]

        if path_velocity is not None:
            if not isinstance(path_velocity, (list, str, Path)):
                msg = "dataframe must be list or pd.DataFrame type"
                raise TypeError(msg)
            if not isinstance(path_velocity, list):
                path_velocity = [path_velocity]
        else:
            path_velocity = [None] * len(path_dataframe)

        if mode not in ["together", "separate"]:
            msg = f"'mode' can be together or separate, not {mode}"
            raise ValueError(msg)

        if mode == "together":
            fig, ax = plt.subplots()

        if len(path_dataframe) > 10:
            colors = cm.get_cmap("tab20")
        else:
            colors = cm.get_cmap("tab10")

        for i, (path_data, path_vel) in enumerate(zip(path_dataframe, path_velocity)):
            print(path_data)

            keys = [
                "frame",
                "main_path",
                "name",
                "time",
                "label",
                "diameter_mean",
            ]

            dataframe = self._load_dataframe(
                path_data,
                usecols=keys,
            )
            self._check_keys(dataframe, keys)

            if isinstance(frames, int):
                frames = [frames]
            if frames is None:
                frames = [dataframe["frame"].unique()]
            if not isinstance(frames, (int, np.ndarray, list)):
                msg = "'frames' must be int, list or ndarray of int"
                raise TypeError(msg)

            mask = dataframe["frame"].isin(frames)

            sub_df = dataframe.loc[
                mask,
                keys,
            ].copy()

            if not path_vel == None:
                keys = [
                    "frame",
                    "timestamp",
                    "voltage",
                    "velocity",
                ]

                velocity = self._load_velocity(
                    path_vel,
                )
                self._check_keys(velocity, keys)

                mask = velocity["frame"].isin(frames)

                velocity = velocity.loc[
                    mask,
                    keys,
                ].copy()

            if unit == "time":
                x_values = sub_df["frame"].unique() * self.time_interval
            elif unit == "frame":
                x_values = sub_df["frame"].unique()
            elif unit == "velocity":
                x_values = velocity
            elif unit == "Reynolds":
                x_values = velocity * sub_df["diameter"].mean() / self.nu_air

            concentration = self._compute_surface_concentration(sub_df)[:, 1]
            print(f"Concentration at t=0 ms : {concentration[0]:.2f} mm^-2")

            if unit == "velocity":
                x_values, concentration = x_values[::200], concentration[::200]
                ax.errorbar(
                    x_values,
                    concentration,
                    xerr=100,
                    color=colors(i),
                    label=f"Essai {i + 1}",
                )
            else:
                ax.plot(
                    x_values,
                    concentration,
                    color=colors(i),
                    label=f"Essai {i + 1}" if curve_names is None else curve_names[i],
                )

        x_ticks = ax.get_xticks()[1:-1]
        ax.set_xticks(x_ticks)

        if unit == "time":
            if language == "en":
                ax.set_xlabel("Time $[s]$", fontsize=self.dict_fontsize["label"])
            if language == "fr":
                ax.set_xlabel("Temps $[s]$", fontsize=self.dict_fontsize["label"])
            ax.set_xticklabels(
                [f"{x_tick:.2f}" for x_tick in x_ticks],
                fontsize=self.dict_fontsize["ticks"],
            )

        elif unit == "frame":
            if language == "en":
                ax.set_xlabel("Frames", fontsize=self.dict_fontsize["label"])
            if language == "fr":
                ax.set_xlabel("Images", fontsize=self.dict_fontsize["label"])
            ax.set_xticklabels(
                [f"{x_tick:.0f}" for x_tick in x_ticks],
                fontsize=self.dict_fontsize["ticks"],
            )

        elif unit == "velocity":
            if language == "en":
                ax.set_xlabel(
                    "Velocity in the middle of the duct $[m.s^{-1}]$",
                    fontsize=self.dict_fontsize["label"],
                )
            if language == "fr":
                ax.set_xlabel(
                    "Vitesse au centre de la veine $[m.s^{-1}]$",
                    fontsize=self.dict_fontsize["label"],
                )
            ax.set_xticklabels(
                [f"{x_tick:.2f}" for x_tick in x_ticks],
                fontsize=self.dict_fontsize["ticks"],
            )

        elif unit == "Reynolds":
            if language == "en":
                ax.set_xlabel(
                    "Reynolds number of the flow", fontsize=self.dict_fontsize["label"]
                )
            if language == "fr":
                ax.set_xlabel(
                    "Nombre de Reynolds de l'écoulement",
                    fontsize=self.dict_fontsize["label"],
                )
            ax.set_xticklabels(
                [f"{x_tick:.0f}" for x_tick in x_ticks],
                fontsize=self.dict_fontsize["ticks"],
            )

        y_ticks = ax.get_yticks()[1:]
        ax.set_yticks(y_ticks)
        if language == "en":
            ax.set_ylabel("Density $[mm^{-2}]$", fontsize=self.dict_fontsize["label"])
        if language == "fr":
            ax.set_ylabel(
                "Concentration surfacique $[mm^{-2}]$",
                fontsize=self.dict_fontsize["label"],
            )
        ax.set_yticklabels(
            [f"{y_tick:.2f}" for y_tick in y_ticks],
            fontsize=self.dict_fontsize["ticks"],
        )

        if len(path_dataframe) > 1:
            plt.legend(fontsize=self.dict_fontsize["legend"])

        plt.subplots_adjust(**self.dict_fontsize["subplots"])

        plt.show()

    def Visualize_concentration_initial_final_resuspended_fraction(
        self,
        path_dataframe: str | Path = None,
        curve_names: (str | list) = None,
        frames: int | list | np.ndarray = None,
        do_smooth: bool = True,
        do_save=False,
        path_save: Path | str = None,
    ):

        if not isinstance(path_dataframe, (list, str, Path)):
            msg = "dataframe must be list or pd.DataFrame type"
            raise TypeError(msg)
        if not isinstance(path_dataframe, list):
            path_dataframe = [path_dataframe]

        # _, ax = plt.subplots()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        colors = cm.get_cmap("tab10")

        initial_concentration, final_concentration, resuspension_fraction = [], [], []

        for i, path_data in enumerate(path_dataframe):
            print(path_data)

            dataframe = self._load_dataframe(path_data)

            if isinstance(frames, int):
                frames = [frames]
            if isinstance(frames, str) and frames == "all":
                frames = self.dataframe["frame"].unique()
            elif not isinstance(frames, (np.ndarray, list)):
                msg = "frames variable must be int or np.ndarray"
                raise TypeError(msg)

            initial_concentration.append(
                *self._compute_surface_concentration(
                    dataframe[dataframe["frame"] == frames[0]]
                )[:, 1]
            )
            final_concentration.append(
                *self._compute_surface_concentration(
                    dataframe[dataframe["frame"] == frames[-1]]
                )[:, 1]
            )
            resuspension_fraction.append(
                1
                - len(dataframe[dataframe["frame"] == frames[-1]])
                / len(dataframe[dataframe["frame"] == frames[0]])
            )

            ax.scatter(
                initial_concentration,
                final_concentration,
                resuspension_fraction,
                color=colors(i),
                label=f"Essai {i + 1}" if curve_names is None else curve_names[i],
            )

        print(initial_concentration)
        print(final_concentration)
        print(resuspension_fraction)

        A = np.c_[
            initial_concentration,
            final_concentration,
            np.ones(np.array(initial_concentration).shape),
        ]
        Z = resuspension_fraction
        fit, residual, _, _ = scipy.linalg.lstsq(A, Z)  # (A.T * A).I * A.T * b
        print(fit)
        error = np.linalg.norm(residual)
        a, b, c = fit

        print(f"Solution : {a:.2f} X + {b:.2f} Y + {c:.2f} Z = 0")
        print(f"Residuals : {residual:.2}")

        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        X, Y = np.meshgrid(
            np.arange(min(initial_concentration), max(initial_concentration)),
            np.arange(min(final_concentration), max(final_concentration)),
        )
        Z = np.zeros(X.shape)
        for r in range(X.shape[0]):
            for c in range(X.shape[1]):
                Z[r, c] = fit[0] * X[r, c] + fit[1] * Y[r, c] + fit[2]

        ax.plot_surface(
            X,
            Y,
            Z,
            cmap=cm.coolwarm,
            alpha=0.3,
        )

        x_ticks = ax.get_xticks()[1:-1]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{x_tick:.2f}" for x_tick in x_ticks], fontsize=16)
        ax.set_xlabel("Initial concentration $[mm^{-2}]$", fontsize=18)

        y_ticks = ax.get_yticks()[1:]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{y_tick:.2f}" for y_tick in y_ticks], fontsize=16)
        ax.set_ylabel("Final concentration $[mm^{-2}]$", fontsize=18)

        # z_ticks = ax.get_zticks()[1:]
        # ax.set_zticks(z_ticks)
        # ax.set_zticklabels([f"{y_tick:.2f}" for y_tick in y_ticks], fontsize=16)
        # ax.set_zlabel("Final resuspension fraction", fontsize=18)

        if len(path_dataframe) > 1:
            plt.legend(fontsize=12)

        plt.show()

    def Visualize_resuspension_vs_concentration(
        self,
        path_dataframe: str | Path = None,
        curve_names: str | list = None,
        frames: int | list | np.ndarray = None,
        do_fit: bool = True,
        do_save: bool = False,
        path_save: Path | str = None,
        mode: str = "together",
        language: str = "en",
    ):

        if not isinstance(path_dataframe, (list, str, Path)):
            msg = "dataframe must be list or pd.DataFrame type"
            raise TypeError(msg)
        if not isinstance(path_dataframe, list):
            path_dataframe = [path_dataframe]

        if mode not in ["together", "separate"]:
            msg = f"'mode' can be together or separate, not {mode}"
            raise ValueError(msg)

        if mode == "together":
            fig, ax = plt.subplots()

        if len(path_dataframe) > 10:
            colors = cm.get_cmap("tab20")
        else:
            colors = cm.get_cmap("tab10")

        lines = []

        for i, path_data in enumerate(path_dataframe):
            print(path_data)

            keys = [
                "frame",
                "main_path",
                "name",
                "time",
                "label",
                "diameter_mean",
            ]

            dataframe = self._load_dataframe(
                path_data,
                usecols=keys,
            )
            self._check_keys(dataframe, keys)

            if isinstance(frames, int):
                frames = [frames]
            if frames is None:
                frames = [dataframe["frame"].unique()]
            if not isinstance(frames, (int, np.ndarray, list)):
                msg = "'frames' must be int, list or ndarray of int"
                raise TypeError(msg)

            mask = dataframe["frame"].isin(frames)

            sub_df = dataframe.loc[
                mask,
                keys,
            ].copy()
            frames = sub_df["frame"].unique()

            initial_density = self._compute_surface_concentration(
                sub_df[sub_df["frame"] == 0]
            )
            print(f"Density {initial_density:.2f}mm^-2")

            Kr_init, Kr_fin = [], []
            Kr_init.append(
                1.0
                - (
                    len(sub_df[sub_df["frame"] == frames[0]])
                    / len(sub_df[sub_df["frame"] == frames[0]])
                )
            )
            Kr_fin.append(
                1.0
                - (
                    len(sub_df[sub_df["frame"] == frames[-1]])
                    / len(sub_df[sub_df["frame"] == frames[0]])
                )
            )

            ax.scatter(
                initial_density,
                Kr_fin,
                color=colors(i),
                label=f"{initial_density:.1f} $[mm^{{-2}}]$",
            )

        if do_fit:
            coef, _, fit = self._fit_curve(
                np.array(initial_density), np.array(Kr_fin), func_base="poly_1st_order"
            )
            print(coef)
            ax.plot(
                initial_density,
                fit,
                color="black",
                label=f"fit poly. {coef[0]:.2e}x $\\times$ {coef[1]:.2e}",
            )

        x_ticks = ax.get_xticks()[1:-1]
        y_ticks = ax.get_yticks()[1:]

        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)

        if language == "en":
            ax.set_xlabel(
                "Initial concentration $[mm^{{-2}}]$",
                fontsize=self.dict_fontsize["label"],
            )
        elif language == "fr":
            ax.set_xlabel(
                "Concentration surfacique initiale $[mm^{{-2}}]$",
                fontsize=self.dict_fontsize["label"],
            )
        ax.set_xticklabels(
            [f"{x_tick:.2f}" for x_tick in x_ticks],
            fontsize=self.dict_fontsize["ticks"],
        )

        if language == "en":
            ax.set_ylabel(
                "Final resuspensded fraction", fontsize=self.dict_fontsize["label"]
            )
        elif language == "fr":
            ax.set_ylabel(
                "Fraction de mise en suspension finale",
                fontsize=self.dict_fontsize["label"],
            )
        ax.set_yticklabels(
            [f"{y_tick:.2f}" for y_tick in y_ticks],
            fontsize=self.dict_fontsize["ticks"],
        )

        ncol = 2 if len(path_dataframe) > 10 else 1
        plt.legend(ncol=ncol, fontsize=self.dict_fontsize["legend"])

        plt.show()

    def Visualize_particle_on_frame(
        self,
        frames: int | list | np.ndarray = None,
        unit: str = "time",
        do_save=False,
        path_save: Path | str = None,
    ):

        if isinstance(frames, int):
            frames = [frames]
        if isinstance(frames, str) and frames == "all":
            frames = self.dataframe["frame"].unique()
        elif not isinstance(frames, (np.ndarray, list)):
            msg = "frames variable must be int or np.ndarray"
            raise TypeError(msg)

        if path_save is None and do_save:
            msg = "Must specify a path to save picture"
            raise TypeError(msg)
        if path_save is not None and do_save is None:
            do_save = True

        if not ((unit == "px") or (unit == "meter")):
            msg = "units varible is not px or meter"
            raise ValueError(msg)

        n = len(frames)

        for frame in frames:
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / cols))

            fig, ax = plt.subplots(figsize=(8, 6))

            img = np.array(
                Image.open(self.list_images[frame]).convert("L"), dtype=np.uint8
            )
            ax.imshow(img, cmap="gray")

            if unit == "px":
                x_ticks = np.linspace(
                    0, int(img.shape[1] / 1000) * 1000, int(img.shape[1] / 1000) + 1
                )
                y_ticks = np.linspace(
                    0, int(img.shape[0] / 1000) * 1000, int(img.shape[0] / 1000) + 1
                )

                ax.set_xticks(x_ticks)
                ax.set_yticks(y_ticks)

                ax.set_xticklabels(x_ticks.astype(int), fontsize=18)
                ax.set_yticklabels(y_ticks.astype(int), fontsize=18)

                ax.set_xlabel("x $[px]$", fontsize=20)
                ax.set_ylabel("y $[px]$", fontsize=20)

            if unit == "meter":  # a revoir, probleme de valeurs
                x_ticks = np.linspace(
                    0, int(img.shape[1] / 1000) * 1000, int(img.shape[1] / 1000) + 1
                )
                y_ticks = np.linspace(
                    0, int(img.shape[0] / 1000) * 1000, int(img.shape[0] / 1000) + 1
                )

                ax.set_xticks(x_ticks / self.pixel_size)  # [px]
                ax.set_yticks(y_ticks / self.pixel_size)  # [px]

                ax.set_xticklabels(x_ticks.astype(float), fontsize=18)
                ax.set_yticklabels(y_ticks.astype(float), fontsize=18)

                ax.set_xlabel("x $[mm]$", fontsize=20)
                ax.set_ylabel("y $[mm]$", fontsize=20)

            plt.legend(fontsize=12)
            plt.subplots_adjust(0.13, 0.13, 1.0, 0.96, 0.0, 0.0)
            if do_save:
                name = Path(path_save) / Path(
                    f"Visualize_frame_"
                    + PurePath(path_save).parts[-1]
                    + "_on_plate_"
                    + str(frame + 1)
                    + ".png"
                )
                print(name)
                plt.savefig(name, dpi=120)
            else:
                plt.show()

    def Visualize_particle_detection(
        self,
        path_dataframe: str | Path = None,
        change_main_path_images: str | list = None,
        frames: int | list | np.ndarray = None,
        path_save: Path | str = None,
        do_save: bool = False,
        units: str = "px",
        rotate_image: int = None,
        crop: list | np.ndarray = None,
        language: str = "en",
    ):

        # check if columns exists
        if not isinstance(path_dataframe, (list, str, Path)):
            msg = "dataframe must be list or pd.DataFrame type"
            raise TypeError(msg)
        if not isinstance(path_dataframe, list):
            path_dataframe = [path_dataframe]

        for i, path_data in enumerate(path_dataframe):
            print(path_data)

            keys = [
                "frame",
                "main_path",
                "name",
                "time",
                "label",
                "x",
                "y",
                "diameter_mean",
                "diameter_std",
            ]

            dataframe = self._load_dataframe(
                path_data,
                # change_main_path_images=change_main_path_images[i],
                usecols=keys,
            )
            self._check_keys(dataframe, keys)

            if isinstance(frames, int):
                frames = [frames]
            if frames is None:
                frames = [dataframe["frame"].unique()]
            if not isinstance(frames, (int, np.ndarray, list)):
                msg = "frames variable must be int, list of ndarray of int"
                raise TypeError(msg)

            mask = dataframe["frame"].isin(frames)

            sub_df = dataframe.loc[
                mask,
                keys,
            ].copy()

            for i, (frame_id, group) in enumerate(sub_df.groupby("frame")):
                _, ax = plt.subplots()

                path_image = Path(group["main_path"].unique()[0]) / Path(
                    group["name"].unique()[0]
                )
                img = Image.open(path_image).convert("L")
                img = self._load_image(
                    path_image, invert=False, rotate_image=rotate_image, crop=crop
                )

                if rotate_image is not None:
                    if rotate_image == 90:
                        img = img.transpose(Image.ROTATE_90)
                    elif rotate_image == 180:
                        img = img.transpose(Image.ROTATE_180)
                    elif rotate_image == 270:
                        img = img.transpose(Image.ROTATE_270)

                if rotate_image in [90, 180, 270]:
                    coords = self._rotate_coords(
                        group[["x", "y"]].to_numpy(),
                        np.array(img).shape[1],
                        np.array(img).shape[0],
                        rotate_image,
                    )
                    x_converted, y_converted = coords[:, 0], coords[:, 1]
                elif rotate_image == 0:
                    x_converted, y_converted = (
                        group["x"].to_numpy(),
                        group["y"].to_numpy(),
                    )

                img = np.array(img, dtype=np.uint8)

                ax.imshow(img, cmap="gray")
                ax.scatter(
                    x_converted,  # [px|meter|milli]
                    y_converted,  # [px|meter|milli]
                    marker="o",
                    color="tab:orange",
                    alpha=0.7,
                    # s=sub_df["diameter_mean"], # [px]
                )

                x_ticks = ax.get_xticks()[1:-1]
                y_ticks = ax.get_yticks()[1:-1]
                ax.set_xticks(x_ticks)
                ax.set_yticks(y_ticks)

                if units == "px":
                    ax.set_xticklabels(
                        [f"{x_tick:.0f}" for x_tick in x_ticks],
                        fontsize=self.dict_fontsize["ticks"],
                    )
                    ax.set_yticklabels(
                        [f"{y_tick:.0f}" for y_tick in y_ticks],
                        fontsize=self.dict_fontsize["ticks"],
                    )
                    ax.set_xlabel("x $[px]$", fontsize=self.dict_fontsize["label"])
                    ax.set_ylabel("y $[px]$", fontsize=self.dict_fontsize["label"])

                if units == "meter":
                    ax.set_xticklabels(
                        [
                            f"{x_tick * self.pixel_size / 1000:.2f}"
                            for x_tick in x_ticks
                        ],
                        fontsize=self.dict_fontsize["ticks"],
                    )
                    ax.set_yticklabels(
                        [
                            f"{y_tick * self.pixel_size / 1000:.2f}"
                            for y_tick in y_ticks
                        ],
                        fontsize=self.dict_fontsize["ticks"],
                    )
                    ax.set_xlabel("x $[m]$", fontsize=self.dict_fontsize["label"])
                    ax.set_ylabel("y $[m]$", fontsize=self.dict_fontsize["label"])

                if units == "milli":
                    ax.set_xticklabels(
                        [f"{x_tick * self.pixel_size:.2f}" for x_tick in x_ticks],
                        fontsize=self.dict_fontsize["ticks"],
                    )
                    ax.set_yticklabels(
                        [f"{y_tick * self.pixel_size:.2f}" for y_tick in y_ticks],
                        fontsize=self.dict_fontsize["ticks"],
                    )
                    ax.set_xlabel("x $[mm]$", fontsize=self.dict_fontsize["label"])
                    ax.set_ylabel("y $[mm]$", fontsize=self.dict_fontsize["label"])

                # plt.subplots_adjust(self.dict_fontsize["legend"])

                if do_save:
                    name = Path(path_save) / Path(
                        f"Visualize_{len(group):d}_particles_"
                        + PurePath(path_save).parts[-1]
                        + "_on_plate_"
                        + str(frame_id)
                        + ".png"
                    )
                    plt.savefig(name, dpi=120)
                else:
                    plt.show()

    # def Visualize_coordination_number(
    #         self,
    #         path_dataframe: str|list|np.ndarray = None,
    #         frames: int|list|np.ndarray = None,
    #         n_bins: int = 7,
    #         y_unit: str = "time",
    #         language: str = "en",
    # ):

    #     if not isinstance(path_dataframe, (list, str, Path)):
    #         msg = "dataframe must be list or pd.DataFrame type"
    #         raise TypeError(msg)
    #     if not isinstance(path_dataframe, list):
    #         path_dataframe = [path_dataframe]

    #     for i, path_data in enumerate(path_dataframe):

    #         print(path_data)

    #         keys = [
    #             "frame", "main_path", "name", "time", "label", "x", "y",
    #             "coordination"
    #             ]

    #         dataframe = self._load_dataframe(
    #             path_data,
    #             usecols=keys,
    #             )
    #         self._check_keys(dataframe, keys)

    #         if isinstance(frames, int):
    #             frames = [frames]
    #         if frames is None:
    #             frames = [dataframe["frame"].unique()]
    #         if not isinstance(frames, (int, np.ndarray, list)):
    #             msg = "'frames' must be int, list or ndarray of int"
    #             raise TypeError(msg)

    #         mask = dataframe["frame"].isin(frames)

    #         sub_df = dataframe.loc[
    #             mask,
    #             keys,
    #         ].copy()

    #         frames = sorted(sub_df["frame"].unique())
    #         # max_coord = sub_df["coordination"].max()
    #         max_coord = n_bins

    #         norm = Normalize(vmin=frames[0], vmax=frames[-1])
    #         cmap = cm.plasma

    #         fig = plt.figure()
    #         ax = fig.add_subplot(111, projection="3d")

    #         for i, f in enumerate(frames):

    #             data = sub_df[sub_df["frame"] == f]["coordination"].values
    #             color = cmap(norm(f))

    #             hist, bins = np.histogram(data, bins=np.arange(max_coord+2))

    #             x = bins[:-1] - 0.5
    #             y = np.full_like(x, f) - 0.5
    #             z = np.zeros_like(x)
    #             dx = np.ones_like(x) * 0.8
    #             dy = np.ones_like(x) * 0.8
    #             dz = hist

    #             ax.bar3d(
    #                 x, y, z,
    #                 dx, dy, dz,
    #                 color=color,
    #                 shade=True,
    #             )

    #         x_ticks = ax.get_xticks()[1:-1]
    #         y_ticks = ax.get_yticks()[1:-1]
    #         z_ticks = ax.get_zticks()[1:-1]

    #         ax.set_xticks(x_ticks)
    #         ax.set_yticks(y_ticks)
    #         ax.set_zticks(z_ticks)

    #         if language == "en":
    #             ax.set_xlabel("Coordination number", fontsize=self.dict_fontsize["label"])
    #         elif language == "fr":
    #             ax.set_xlabel("Nombre de coordination", fontsize=self.dict_fontsize["label"])
    #         ax.set_xticklabels([f"{x_tick:.0f}" for x_tick in x_ticks], fontsize=self.dict_fontsize["ticks"])

    #         if y_unit == "frame":
    #             if language == "en":
    #                 ax.set_ylabel("Frames", fontsize=self.dict_fontsize["label"])
    #             elif language == "fr":
    #                 ax.set_ylabel("Images", fontsize=self.dict_fontsize["label"])
    #             ax.set_yticklabels([f"{y_tick:.0f}" for y_tick in y_ticks], fontsize=self.dict_fontsize["ticks"])
    #         elif y_unit == "time":
    #             if language == "en":
    #                 ax.set_ylabel("Time $[s]$", fontsize=self.dict_fontsize["label"])
    #             elif language == "fr":
    #                 ax.set_ylabel("Temps $[s]$", fontsize=self.dict_fontsize["label"])
    #             ax.set_yticklabels([f"{y_tick*self.time_interval:.2f}" for y_tick in y_ticks], fontsize=self.dict_fontsize["ticks"])

    #         if language == "en":
    #             ax.set_zlabel("Occurence", fontsize=self.dict_fontsize["label"])
    #         elif language == "fr":
    #             ax.set_zlabel("Occurence", fontsize=self.dict_fontsize["label"])
    #         ax.set_zticklabels([f"{z_tick:.0f}" for z_tick in z_ticks], fontsize=self.dict_fontsize["ticks"])

    #         plt.subplots_adjust(**self.dict_fontsize["subplots"])

    #         # if do_save:
    #         #     name = Path(path_save) / Path(
    #         #         f"Visualize_{len(group):d}_particles_"
    #         #         + PurePath(path_save).parts[-1]
    #         #         + "_on_plate_"
    #         #         + str(frame_id)
    #         #         + ".png"
    #         #     )
    #         #     plt.savefig(name, dpi=120)
    #         # else:
    #         #     plt.show()

    #         # mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    #         # mappable.set_array([])
    #         # fig.colorbar(mappable, ax=ax)

    #         plt.show()

    # def Visualize_Voronoi_triangulation(
    #     self,
    #     path_dataframe: str|Path = None,
    #     frames: int|list|np.ndarray = None,
    #     path_save: Path|str = None,
    #     do_save: bool = False,
    #     units: str = "px",
    #     rotate_image: int = None,
    #     crop: list|np.ndarray = None,
    #     change_main_path_images: list|str|Path = None,
    #     do_density_map: bool = True,
    #     min_max_value_colorbar: list = None,
    #     mode: str = "separate",
    #     language: str = "en",
    #     ):

    #     if not isinstance(path_dataframe, (list, str, Path)):
    #         msg = "dataframe must be list or pd.DataFrame type"
    #         raise TypeError(msg)
    #     if not isinstance(path_dataframe, list):
    #         path_dataframe = [path_dataframe]

    #     if mode not in ["together", "separate"]:
    #         msg = f"'mode' can be together or separate, not {mode}"
    #         raise ValueError(msg)

    #     if mode == "together":
    #         fig, ax = plt.subplots()

    #     for i, path_data in enumerate(path_dataframe):

    #         print(path_data)

    #         keys = [
    #             "frame", "main_path", "name", "time", "label", "diameter_mean",
    #             "x", "y"
    #             ]

    #         dataframe = self._load_dataframe(
    #             path_data,
    #             usecols=keys,
    #             )
    #         self._check_keys(dataframe, keys)

    #         if isinstance(frames, int):
    #             frames = [frames]
    #         if frames is None:
    #             frames = [dataframe["frame"].unique()]
    #         if not isinstance(frames, (int, np.ndarray, list)):
    #             msg = "'frames' must be int, list or ndarray of int"
    #             raise TypeError(msg)

    #         mask = dataframe["frame"].isin(frames)

    #         sub_df = dataframe.loc[
    #             mask,
    #             keys,
    #         ].copy()

    #         for i, (frame_id, group) in enumerate(sub_df.groupby("frame")):

    #             if mode == "separate":
    #                 fig, ax = plt.subplots()

    #             group["x"] = group["x"] * self.pixel_size
    #             group["y"] = group["y"] * self.pixel_size

    #             # load and display image
    #             path_image = Path(group["main_path"].unique()[0]) / Path(group["name"].unique()[0])
    #             img = Image.open(path_image).convert("L")

    #             if rotate_image is not None:
    #                 if rotate_image == 90:
    #                     img = img.transpose(Image.ROTATE_90)
    #                 elif rotate_image == 180:
    #                     img = img.transpose(Image.ROTATE_180)
    #                 elif rotate_image == 270:
    #                     img = img.transpose(Image.ROTATE_270)

    #                 # coords = self._rotate_coords(group[["x", "y"]].to_numpy(), np.array(img).shape[1], np.array(img).shape[0], rotate_image)
    #                 # x_converted, y_converted = coords[:, 0], coords[:, 1]

    #             x_converted, y_converted = group["x"]/self.pixel_size, group["y"]/self.pixel_size

    #             img = np.array(img, dtype=np.uint8)
    #             ax.imshow(img, cmap="gray")

    #             ax.scatter(
    #                 x_converted, # [px|meter|milli]
    #                 y_converted, # [px|meter|milli]
    #                 marker="o",
    #                 # s=sub_df["diameter_mean"], # [px]
    #                 alpha=0.7,
    #                 color="tab:orange",
    #             )

    #             # compute Voronoi triangulation
    #             points = np.array(list(zip(x_converted, y_converted)))
    #             voronoi_tri = Voronoi(points)

    #             x_max = img.shape[1]
    #             y_max = img.shape[0]

    #             for ridge in voronoi_tri.ridge_vertices:
    #                 if -1 not in ridge:
    #                     v0, v1 = voronoi_tri.vertices[ridge]
    #                     ax.plot([v0[0], v1[0]], [v0[1], v1[1]], color="gray")
    #             ax.set_xlim([0, x_max])
    #             ax.set_ylim([0, y_max])

    #             if do_density_map:

    #                 bounding_polygon = Polygon([
    #                     (0, 0), (2*x_max, 0),
    #                     (2*x_max, 2*y_max), (0, 2*y_max)
    #                 ])
    #                 densities, cells = [], []
    #                 for region_index in voronoi_tri.point_region:
    #                     region = voronoi_tri.regions[region_index]
    #                     if not region or -1 in region:
    #                         densities.append(0)
    #                         cells.append(None)
    #                         continue

    #                     polygon = Polygon(voronoi_tri.vertices[region])
    #                     clipped = polygon.intersection(bounding_polygon)

    #                     if clipped.is_empty:
    #                         densities.append(0)
    #                         cells.append(None)
    #                     else:
    #                         area = clipped.area
    #                         rho = 1.0 / area if area > 0 else 0
    #                         densities.append(rho)
    #                         cells.append(clipped)

    #                 densities = np.array(densities)
    #                 densities = np.clip(densities, 0, np.percentile(densities, 99))
    #                 densities_norm = (densities - densities.min()) / (densities.max() - densities.min() + 1e-9)

    #                 # create mesh
    #                 nx, ny = img.shape
    #                 x_grid = np.linspace(points[:, 0].min(), points[:, 0].max(), nx)
    #                 y_grid = np.linspace(points[:, 1].min(), points[:, 1].max(), ny)
    #                 X, Y = np.meshgrid(x_grid, y_grid)

    #                 # interpolate
    #                 Z = scipy.interpolate.griddata(points, densities, (X, Y), method="cubic")

    #                 # plot
    #                 cmap = plt.cm.coolwarm
    #                 ax.imshow(Z, origin="lower", extent=(points[:, 0].min(), points[:, 0].max(), points[:, 1].min(), points[:, 1].max()),
    #                           cmap=cmap, aspect="auto", alpha=0.5)
    #                 if min_max_value_colorbar is not None:
    #                     min_max_value_colorbar = np.array(min_max_value_colorbar)
    #                     sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(
    #                         vmin=min_max_value_colorbar.min(),
    #                         vmax=min_max_value_colorbar.max()
    #                         ))
    #                 else:
    #                     sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(
    #                         vmin=densities.min(),
    #                         vmax=densities.max()
    #                         ))
    #                 sm.set_array([])
    #                 color_bar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.01)
    #                 x_ticks = color_bar.get_ticks()[:-1]
    #                 color_bar.set_ticks(x_ticks)
    #                 if language == "en":
    #                     color_bar.set_label("Local density $\\rho_p$ $[mm^{{-2}}]$", fontsize=self.dict_fontsize["label"])
    #                 elif language == "fr":
    #                     color_bar.set_label("Densité locale $\\rho_p$ $[mm^{{-2}}]$", fontsize=self.dict_fontsize["label"])
    #                 color_bar.set_ticklabels([f"{color_value:.1e}" for color_value in np.array(color_bar.get_ticks())], fontsize=self.dict_fontsize["label"])

    #             x_ticks = ax.get_xticks()[1:-1]
    #             y_ticks = ax.get_yticks()[1:-1]
    #             ax.set_xticks(x_ticks)
    #             ax.set_yticks(y_ticks)

    #             if units == "px":
    #                 ax.set_xlabel("x $[px]$", fontsize=self.dict_fontsize["ticks"])
    #                 ax.set_ylabel("y $[px]$", fontsize=self.dict_fontsize["ticks"])
    #                 ax.set_xticklabels([f"{x_value:.0f}" for x_value in np.array(ax.get_xticks())], fontsize=self.dict_fontsize["label"])
    #                 ax.set_yticklabels([f"{y_value:.0f}" for y_value in np.array(ax.get_yticks())], fontsize=self.dict_fontsize["label"])

    #             if units == "meter":
    #                 ax.set_xlabel("x $[m]$", fontsize=self.dict_fontsize["ticks"])
    #                 ax.set_ylabel("y $[m]$", fontsize=self.dict_fontsize["ticks"])
    #                 ax.set_xticklabels([f"{x_value * self.pixel_size / 1000:.2f}" for x_value in np.array(ax.get_xticks())], fontsize=self.dict_fontsize["label"])
    #                 ax.set_yticklabels([f"{y_value * self.pixel_size / 1000:.2f}" for y_value in np.array(ax.get_yticks())], fontsize=self.dict_fontsize["label"])

    #             if units == "milli":
    #                 ax.set_xlabel("x $[mm]$", fontsize=self.dict_fontsize["ticks"])
    #                 ax.set_ylabel("y $[mm]$", fontsize=self.dict_fontsize["ticks"])
    #                 ax.set_xticklabels([f"{x_value * self.pixel_size:.2f}" for x_value in np.array(ax.get_xticks())], fontsize=self.dict_fontsize["label"])
    #                 ax.set_yticklabels([f"{y_value * self.pixel_size:.2f}" for y_value in np.array(ax.get_yticks())], fontsize=self.dict_fontsize["label"])

    #             plt.subplots_adjust(**self.dict_fontsize["subplots"])

    #             if do_save:
    #                 name = Path(path_save) / Path(
    #                     f"Visualize_{len(group):d}_particles_"
    #                     + PurePath(path_save).parts[-1]
    #                     + "_on_plate_"
    #                     + str(frame_id)
    #                     + ".png"
    #                 )
    #                 plt.savefig(name, dpi=120)
    #             else:
    #                 plt.show()

    def Visualize_Voronoi_triangulation(
        self,
        dataframe: str | Path = None,
        frames: int | list | np.ndarray = None,
        labels: int | list | np.ndarray = None,
        pixel_size: float = None,
        time_interval: float = None,
        x_unit: str = "mm",
        y_unit: str = "mm",
        rotate: int = None,
        do_density_map: bool = False,
        velocity = None,
    ) -> list:

        results = []

        for data in dataframe:
            # ----- frames
            if frames is None:
                selected_frames = data["frame"].unique()
            elif isinstance(frames, int):
                selected_frames = [frames]
            elif isinstance(frames, (list, np.ndarray)):
                selected_frames = frames
            else:
                msg = "'frames' must be int, list or np.ndarray of int"
                raise TypeError(msg)

            # ----- labels
            if labels == "all" or labels is None:
                selected_labels = sorted(data["label"].unique())
            else:
                selected_labels = labels

            # ----- filtering
            mask = data["frame"].isin(selected_frames) & data["label"].isin(
                selected_labels
            )

            sub_df = data.loc[
                mask,
                [
                    "frame",
                    "main_path",
                    "name",
                    "label",
                    "time",
                    "x",
                    "y",
                    "diameter_mean",
                ],
            ].copy()

            if sub_df.empty:
                continue

            unit_factor = {"px": 1, "mm": pixel_size, "m": pixel_size / 1000}[x_unit]

            unit_label = {
                "px": "px",
                "mm": "mm",
                "m": "m",
            }[y_unit]

            # sub_df["x"] *= unit_factor
            # sub_df["y"] *= unit_factor

            # ----- load image
            name = Path(data["main_path"].iloc[0], data["name"].iloc[0])
            img = self._load_image(name, invert=False, rotate_image=rotate)

            # ----- group data
            grouped_frames = sub_df.groupby("frame")

            # ----- compute Voronoi triangulation
            vors = []
            density_maps = []

            if do_density_map:
                bounding_polygon = Polygon(
                    [
                        (0, 0),
                        (2 * img.shape[1], 0),
                        (2 * img.shape[1], 2 * img.shape[0]),
                        (0, 2 * img.shape[0]),
                    ]
                )

            for frame, g in grouped_frames:
                pts = np.unique(g[["x", "y"]].values, axis=0)
                if len(pts) < 3:
                    vors.append(None)
                    density_maps.append(None)
                    continue

                # voronoi
                vor = Voronoi(pts)
                vors.append(vor)

                ridges = [
                    vor.vertices[r]
                    for r in vor.ridge_vertices
                    if len(r) == 2 and -1 not in r
                ]

                # ----- density map
                densities = []
                if do_density_map:
                    for region_index in vor.point_region:
                        region = vor.regions[region_index]

                        if not region or -1 in region:
                            density_maps.append(0)
                            continue

                        polygon = Polygon(vor.vertices[region])
                        clipped = polygon.intersection(bounding_polygon)

                        if clipped.is_empty:
                            densities.append(0)
                        else:
                            area = clipped.area
                            rho = 1.0 / area if area > 0 else 0
                            densities.append(rho)

                    densities = np.array(densities)

                    if len(densities) != len(vor.points):
                        density_maps.append(None)
                    densities = np.clip(densities, 0, np.percentile(densities, 99))
                    # densities_norm = (densities - densities.min()) / (densities.max() - densities.min() + 1e-9)

                    # ----- create mesh
                    pts = vor.points
                    nx, ny = img.shape[1], img.shape[0]

                    x = np.linspace(pts[:, 0].min(), pts[:, 0].max(), nx)
                    y = np.linspace(pts[:, 1].min(), pts[:, 1].max(), ny)
                    X, Y = np.meshgrid(x, y)

                    Z = griddata(pts, densities, (X, Y), method="cubic")
                    density_maps.append(Z)

            data_dict = {
                "image": img,
                "voronoi": True,
                "x": [group["x"].to_numpy() for _, group in grouped_frames],
                "y": [group["y"].to_numpy() for _, group in grouped_frames],
                "frames": [group["frame"].to_numpy() for _, group in grouped_frames],
                "vor": vors,
                # "voronoi_vertices": voronoi_vertices,
                "density_map": density_maps if do_density_map else None,
                "unit": unit_factor,
                "x_label": f"X [{unit_label}]",
                "y_label": f"Y [{unit_label}]",
                "density_label": f"Density [\\rho]" if do_density_map else None,
            }
            results.append(data_dict)

        return results

    def Visualize_surface_concentration(
        self,
        path_dataframe: str | Path = None,
        frames: int | list | np.ndarray = None,
        cut: tuple = (1, 1),
        rotate_image: int = 0,
        do_scale_colorbar: bool = True,
        do_color_map: bool = True,
        path_save: str | Path = None,
        do_save: bool = False,
        do_save_csv: bool = True,
        units: str = "px",
        mode: str = "separate",
        language: str = "en",
    ):

        if not isinstance(cut, tuple):
            msg = "cut variable must be tuple"
            raise TypeError(msg)

        if path_save is None and do_save:
            msg = "Must specify a path to save picture"
            raise TypeError(msg)
        if path_save is not None and do_save is None:
            do_save = True

        if path_save is None and do_save:
            msg = "Must specify a path to save picture"
            raise TypeError(msg)
        if path_save is not None and do_save is None:
            do_save = True

        if mode not in ["together", "separate"]:
            msg = f"'mode' can be together or separate, not {mode}"
            raise ValueError(msg)

        if mode == "together":
            fig, ax = plt.subplots()

        for i, path_data in enumerate(path_dataframe):
            print(path_data)

            keys = [
                "frame",
                "main_path",
                "name",
                "time",
                "label",
                "diameter_mean",
                "x",
                "y",
            ]

            dataframe = self._load_dataframe(
                path_data,
                usecols=keys,
            )
            self._check_keys(dataframe, keys)

            if isinstance(frames, int):
                frames = [frames]
            if frames is None:
                frames = [dataframe["frame"].unique()]
            if not isinstance(frames, (int, np.ndarray, list)):
                msg = "'frames' must be int, list or ndarray of int"
                raise TypeError(msg)

            mask = dataframe["frame"].isin(frames)

            sub_df = dataframe.loc[
                mask,
                keys,
            ].copy()

            for i, (frame_id, group) in enumerate(sub_df.groupby("frame")):
                if mode == "separate":
                    fig, ax = plt.subplots()

                # load and display image
                path_image = Path(group["main_path"].unique()[0]) / Path(
                    group["name"].unique()[0]
                )
                img = Image.open(path_image).convert("L")

                if rotate_image is not None:
                    if rotate_image == 90:
                        img = img.transpose(Image.ROTATE_90)
                    elif rotate_image == 180:
                        img = img.transpose(Image.ROTATE_180)
                    elif rotate_image == 270:
                        img = img.transpose(Image.ROTATE_270)

                    # coords = self._rotate_coords(group[["x", "y"]].to_numpy(), np.array(img).shape[1], np.array(img).shape[0], rotate_image)
                    # x_converted, y_converted = coords[:, 0], coords[:, 1]

                x_converted, y_converted = (
                    group["x"] / self.pixel_size,
                    group["y"] / self.pixel_size,
                )

                img = np.array(img, dtype=np.uint8)
                ax.imshow(img, cmap="gray")

                if cut == (1, 1):
                    total_density = self._compute_surface_concentration(
                        df=sub_df,
                        frames=frames,
                        img_size=list(img.shape),
                        cut=tuple(cut),
                    )
                else:
                    total_density, concentrations, cut_v, cut_h, zone_counts = (
                        self._compute_surface_concentration(
                            df=sub_df,
                            frames=frames,
                            img_size=list(img.shape),
                            cut=tuple(cut),
                        )
                    )

                # plot concentration per zone
                if cut != (1, 1):
                    for ii, (y1, y2) in enumerate(zip(cut_h[:-1], cut_h[1:])):
                        for jj, (x1, x2) in enumerate(zip(cut_v[:-1], cut_v[1:])):
                            zone_id = (ii, jj)
                            center = ((x1 + x2) // 2, (y1 + y2) // 2)
                            surface = (
                                (cut_h[ii + 1] - cut_h[ii])
                                * (cut_v[jj + 1] - cut_v[jj])
                                * (self.pixel_size * 1000) ** 2
                            )
                            count = sum(
                                zone_counts[zone].get(zone_id, 0)
                                for zone in zone_counts
                            )
                            ax.text(
                                center[0],
                                center[1],
                                f"{(ii * len(cut_v[:-1]) + jj) + 1}",
                                ha="center",
                                va="bottom",
                            )
                            ax.text(
                                center[0],
                                center[1],
                                f"{count / surface:.1f} $mm^{{-2}}$",
                                ha="center",
                                va="top",
                            )

            if do_color_map is not None and do_color_map:
                cmap = plt.cm.get_cmap("autumn_r")
                if do_scale_colorbar:
                    norm = Normalize(
                        vmin=min(dict["concentrations"]),
                        vmax=max(dict["concentrations"]),
                    )
                else:
                    norm = Normalize(vmin=min(concentrations), vmax=max(concentrations))
                rectangles = [
                    {
                        "x": cut_v[jj],
                        "y": cut_h[ii],
                        "width": cut_v[jj + 1] - cut_v[jj],
                        "height": cut_h[ii + 1] - cut_h[ii],
                        "concentration": concentrations[ii * (len(cut_v) - 1) + jj],
                    }
                    for ii in range(len(cut_h) - 1)
                    for jj in range(len(cut_v) - 1)
                ]

                for rect in rectangles:
                    ax.add_patch(
                        patches.Rectangle(
                            (rect["x"], rect["y"]),
                            rect["width"],
                            rect["height"],
                            linewidth=1,
                            facecolor=cmap(norm(rect["concentration"])),
                        )
                    )
                sm = cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax, orientation="vertical")
                cbar.set_label(
                    "Densities" if language == "en" else "Concentrations surfaciques",
                    rotation="vertical",
                    fontsize=18,
                )

            x_ticks = ax.get_xticks()
            y_ticks = ax.get_yticks()

            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)

            if units == "px":
                ax.set_xlabel("x $[px]$", fontsize=self.dict_fontsize["label"])
                ax.set_ylabel("y $[px]$", fontsize=self.dict_fontsize["label"])
                ax.set_xticklabels(
                    [f"{x_tick:.0f}" for x_tick in x_ticks],
                    fontsize=self.dict_fontsize["ticks"],
                )
                ax.set_yticklabels(
                    [f"{x_tick:.0f}" for x_tick in x_ticks],
                    fontsize=self.dict_fontsize["ticks"],
                )

            elif units == "mm":
                ax.set_xlabel("x $[mm]$", fontsize=self.dict_fontsize["label"])
                ax.set_ylabel("y $[mm]$", fontsize=self.dict_fontsize["label"])
                ax.set_xticklabels(
                    [f"{x_tick:.2f}" for x_tick in x_ticks],
                    fontsize=self.dict_fontsize["ticks"],
                )
                ax.set_yticklabels(
                    [f"{x_tick:.2f}" for x_tick in x_ticks],
                    fontsize=self.dict_fontsize["ticks"],
                )

            plt.subplots_adjust(**self.dict_fontsize["subplots"])

            # if do_save:
            #     name = Path(path_save) / Path(
            #         f"Surface_concentration_{total_surface_concentration:.2e}_mm2_"
            #         + PurePath(path_save).parts[-1]
            #         + f"_plate_{frame+1}.png"
            #     )
            #     plt.savefig(name)
            # else:
            #     plt.show()

    def Visualize_multi_plates_concentration(
        self,
        cut=None,
        path_load=None,
        path_save=None,
        files=None,
        do_color_map=True,
        do_save=False,
        units="px",
    ):

        if path_load is None:
            msg = "Must specify a path to load images"
            raise TypeError(msg)
        # if isinstance(path_load, (str, pathlib.WindowsPath)):
        #     path_load = [path_load]

        if path_save is None and do_save:
            msg = "Must specify a path to save images"
            raise TypeError(msg)
        if path_save is not None and do_save is None:
            do_save = True

        if not ((units == "px") or (units == "meter")):
            msg = "units varible is not px or meter"
            raise ValueError(msg)

        imgs = []
        files = [path_load / file for file in files]
        for file in files:
            img = Image.open(Path(file)).convert("L")
            width, height = img.size
            img = img.resize((width // 4, height // 4))
            imgs.append(np.array(img))

        fig = plt.figure(figsize=(6, 6))

        gs = GridSpec(
            3,
            4,
            figure=fig,
            wspace=0.001,
            hspace=0.1,
            width_ratios=[1, 1, 1, 1],
            height_ratios=[1, 3.5, 1],
        )

        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        ax4 = fig.add_subplot(gs[1, 2])
        ax5 = fig.add_subplot(gs[1, 3])
        ax6 = fig.add_subplot(gs[2, :])

        axes = [ax1, ax2, ax3, ax4, ax5, ax6]
        total_concentrations = []
        for idx, (ax, img) in enumerate(zip(axes, imgs)):
            if idx in (0, 5):
                img = np.rot90(img, k=1)
                ax.imshow(img, cmap="gray")
                (
                    total_surface_concentration,
                    concentrations,
                    cut_v,
                    cut_h,
                    zone_counts,
                ) = self._compute_surface_concentration(img, idx, cut[::-1])
                total_concentrations.append(concentrations)
            else:
                ax.imshow(img, cmap="gray")
                (
                    total_surface_concentration,
                    concentrations,
                    cut_v,
                    cut_h,
                    zone_counts,
                ) = self._compute_surface_concentration(img, idx, cut)
                total_concentrations.append(concentrations)

            # # plot concentration per zone
            # for ii, (y1, y2) in enumerate(zip(cut_h[:-1], cut_h[1:])):
            #     for jj, (x1, x2) in enumerate(zip(cut_v[:-1], cut_v[1:])):
            #         zone_id = (ii, jj)
            #         center = ((x1 + x2) // 2, (y1 + y2) // 2)
            #         surface = (cut_h[ii + 1] - cut_h[ii]) * (cut_v[jj + 1] - cut_v[jj]) * (self.pixel_size * 1000)**2
            #         count = sum(zone_counts[zone].get(zone_id, 0) for zone in zone_counts)
            #         ax.text(center[0], center[1],
            #                 f'{(ii * len(cut_v[:-1]) + jj) + 1}',
            #                 ha='center', va='bottom')
            #         ax.text(center[0], center[1],
            #                 f'{count / surface:.1f}{r'$mm^{-2}$'}',
            #                 ha='center', va='top')

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)

            if do_color_map is not None and do_color_map:
                norm = Normalize(
                    vmin=np.min(total_concentrations), vmax=np.max(total_concentrations)
                )
                cmap = plt.cm.autumn

                for img in (
                    imgs
                ):  # Supposons que vous ayez une liste de données pour chaque image
                    rectangles = [
                        {
                            "x": cut_v[jj],
                            "y": cut_h[ii],
                            "width": cut_v[jj + 1] - cut_v[jj],
                            "height": cut_h[ii + 1] - cut_h[ii],
                            "concentration": concentrations[ii * (len(cut_v) - 1) + jj],
                        }
                        for ii in range(len(cut_h) - 1)
                        for jj in range(len(cut_v) - 1)
                    ]
                for rect in rectangles:
                    ax.add_patch(
                        patches.Rectangle(
                            (rect["x"], rect["y"]),
                            rect["width"],
                            rect["height"],
                            linewidth=1,
                            facecolor=cmap(norm(rect["concentration"])),
                        )
                    )

        norm = Normalize(
            vmin=np.min(total_concentrations), vmax=np.max(total_concentrations)
        )
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax5, orientation="vertical")
        cbar.set_label(
            f"Concentrations {r'$[#/mm^2]$'}", rotation="vertical", fontsize=18
        )

        if do_save:
            name = path_save / Path(
                f"Particle_concentration_over_{len(imgs)}_plates.png"
            )
            plt.savefig(str(name), dpi=120)
        else:
            plt.show()

    def Smooth_trajectories(
        self,
        dataframe: pd.DataFrame = None,
        pixel_size: float = None,
        time_interval: float = 1 / 8000,
        frames: list | np.ndarray = None,
        labels: list | np.ndarray = None,
        path_save: str = None,
        do_save: bool = False,
        x_unit: str = "time",
        y_unit: str = "fraction",
    ):

        # def kalman_track_2d(
        #         positions, dt=1.0,
        #         process_noise=1e-3,
        #         measurement_noise=2.0,
        #         static_threshold=0.1,
        #         freeze_static=True
        #         ):
        #     positions = np.array(positions)
        #     n = len(positions)

        #     # state : [x, y, vx, vy]
        #     # x = np.zeros((4, 1))
        #     # x[:2] = positions[0:2].reshape(2, -1)

        #     # Matrices
        #     F = np.array([
        #         [1, 0, dt, 0],
        #         [0, 1, 0, dt],
        #         [0, 0, 1, 0],
        #         [0, 0, 0, 1]
        #     ])

        #     H = np.array([
        #         [1, 0, 0, 0],
        #         [0, 1, 0, 0]
        #     ])

        #     P = np.eye(4) * 500
        #     Q = np.eye(4) * process_noise
        #     R = np.eye(2) * measurement_noise
        #     I = np.eye(4)

        #     filtered = []

        #     for z in positions:

        #         print(f"Initial z : {z}")

        #         # Predict
        #         z = F @ z
        #         print(f"Predicted z : {z}")
        #         P = F @ P @ F.T + Q

        #         # Update
        #         y = z - H @ z
        #         S = H @ P @ H.T + R
        #         K = P @ H.T @ np.linalg.inv(S)

        #         z = z + K @ y
        #         P = (I - K @ H) @ P

        #         filtered.append(x[:2].flatten())

        #     filtered = np.array(filtered)

        #     # detect static particles
        #     velocities = np.diff(filtered, axis=0)
        #     speed = np.linalg.norm(velocities, axis=1)

        #     mean_speed = np.mean(speed)

        #     if freeze_static and mean_speed < static_threshold:
        #         mean_pos = np.mean(filtered, axis=0)
        #         filtered[:] = mean_pos

        #     return filtered

        def kalman_xy(
            x,
            P,
            measurement,
            R,
            motion=np.matrix("0. 0. 0. 0.").T,
            Q=np.matrix(np.eye(4)),
        ):
            """
            Parameters:
            x: initial state 4-tuple of location and velocity: (x0, x1, x0_dot, x1_dot)
            P: initial uncertainty convariance matrix
            measurement: observed position
            R: measurement noise
            motion: external motion added to state vector x
            Q: motion noise (same shape as P)
            """
            return kalman(
                x,
                P,
                measurement,
                R,
                motion,
                Q,
                F=np.matrix("""
                            1. 0. 1. 0.;
                            0. 1. 0. 1.;
                            0. 0. 1. 0.;
                            0. 0. 0. 1.
                            """),
                H=np.matrix("""
                            1. 0. 0. 0.;
                            0. 1. 0. 0."""),
            )

        def kalman(x, P, measurement, R, motion, Q, F, H):
            """
            Parameters:
            x: initial state
            P: initial uncertainty convariance matrix
            measurement: observed position (same shape as H*x)
            R: measurement noise (same shape as H)
            motion: external motion added to state vector x
            Q: motion noise (same shape as P)
            F: next state function: x_prime = F*x
            H: measurement function: position = H*x

            Return: the updated and predicted new values for (x, P)

            See also http://en.wikipedia.org/wiki/Kalman_filter

            This version of kalman can be applied to many different situations by
            appropriately defining F and H
            """
            # UPDATE x, P based on measurement m
            # distance between measured and current position-belief
            y = np.matrix(measurement).T - H * x
            S = H * P * H.T + R  # residual convariance
            K = P * H.T * S.I  # Kalman gain
            x = x + K * y
            I = np.matrix(np.eye(F.shape[0]))  # identity matrix
            P = (I - K * H) * P

            # PREDICT x, P based on motion
            x = F * x + motion
            P = F * P * F.T + Q

            return x, P

        for run, data in enumerate(dataframe):
            # ----- frames
            if frames is None:
                selected_frames = data["frame"].unique()
            elif isinstance(frames, int):
                selected_frames = [frames]
            elif isinstance(frames, (list, np.ndarray)):
                selected_frames = frames
            else:
                msg = "'frames' must be int, list or np.ndarray of int"
                raise TypeError(msg)

            # ----- labels
            if labels == "all" or labels is None:
                selected_labels = sorted(data["label"].unique())
            else:
                selected_labels = labels

            # ------ filtering
            mask = data["frame"].isin(selected_frames) & data["label"].isin(
                selected_labels
            )

            sub_df = data.loc[
                mask, ["frame", "label", "x", "y", "vx", "vy", "main_path", "name"]
            ].copy()

            if sub_df.empty:
                continue

            _, ax = plt.subplots()

            # ----- load image
            name = Path(data["main_path"].iloc[0], data["name"].iloc[0])
            img = self._load_image(name, invert=False)

            for lbl, group in sub_df.groupby("label"):
                print(len(group))

                raw_traj = group[["x", "y", "vx", "vy"]]

                x = np.matrix("0. 0. 0. 0.").T
                P = np.matrix(np.eye(4)) * 1000

                res = []
                R = 0.01**2

                for meas in raw_traj.values:
                    meas = meas[:2]
                    x, P = kalman_xy(x, P, meas, R)
                    res.append((x[:2]).tolist())
                kalman_x, kalman_y = zip(*res)

                ax.imshow(img, cmap="gray")

                ax.scatter(raw_traj["x"], raw_traj["y"], color="b")
                ax.scatter(kalman_x, kalman_y, color="r")

            plt.show()
        return None

    def Visualize_resuspended_fraction(
        self,
        dataframe: pd.DataFrame = None,
        pixel_size: float = None,
        time_interval: float = 1 / 8000,
        frames: list | np.ndarray = None,
        labels: list | np.ndarray = None,
        path_save: str = None,
        do_save: bool = False,
        x_unit: str = "time",
        y_unit: str = "fraction",
        curve_names: str | list = None,
        use_fit: bool = True,
        fit_function: str = "sigmoid",
        use_mean: bool = False,
        kernel_size: int = None,
        velocity: pd.DataFrame = None,
    ) -> list:

        if velocity is None:
            velocity = [None] * len(dataframe)

        if path_save is None and do_save:
            msg = "Must specify a path to save picture"
            raise TypeError(msg)
        if path_save is not None and do_save is None:
            do_save = True

        results = []

        for run, data in enumerate(dataframe):
            # ----- frames
            if frames is None:
                selected_frames = data["frame"].unique()
            elif isinstance(frames, int):
                selected_frames = [frames]
            elif isinstance(frames, (list, np.ndarray)):
                selected_frames = frames
            else:
                msg = "'frames' must be int, list or np.ndarray of int"
                raise TypeError(msg)

            # ----- labels
            if labels == "all" or labels is None:
                selected_labels = sorted(data["label"].unique())
            else:
                selected_labels = labels

            # ------ filtering
            mask = data["frame"].isin(selected_frames) & data["label"].isin(
                selected_labels
            )

            sub_df = data.loc[mask, ["frame", "diameter_mean"]].copy()

            if sub_df.empty:
                continue

            # ----- select unit factor
            unit_factor_x = {
                "frames": 1,
                "time": time_interval,
                "f_velocity": 1.0,  # friction velocity
                "m_velocity"  # middle duct velocity
                "reynolds": 1.0,
            }[x_unit]

            unit_label_x = {
                "frames": "Frames",
                "time": "Time $[s]$",
                "f_velocity": "Friction velocity $[m/s]$",  # friction velocity
                "m_velocity": "Middle duct velocity $[m/s]$",  # middle duct velocity
                "reynolds": "Reynolds number",
            }[x_unit]

            unit_factor_y = {
                "fraction": 1,
            }[y_unit]

            unit_label_y = {
                "fraction": "$K_{{res}}$",
            }[y_unit]

            # # ----- velocity
            # if velocity is not None:
            #     mask = (
            #         velocity["frame"].isin(selected_frames)
            #     )
            #     velocity = velocity.loc[
            #         mask,
            #         ["frame", "velocity"]
            #     ].copy()
            #     velocity["friction"] = 0.0564 * velocity["velocity"] ** (7/8)

            #     if velocity.empty:
            #         continue

            # ----- group data
            grouped = sub_df.groupby("frame")
            initial_num_parts = len(
                sub_df[sub_df["frame"] == min(sub_df["frame"].unique())]
            )

            # ----- fill data_dict
            data_dict = {
                "curves": True,
                "x": [sub_df["frame"].unique()],
                "y": [[(1 - (len(group) / initial_num_parts)) for _, group in grouped]],
                "run": [run],
                "x_unit": unit_factor_x,
                "y_unit": unit_factor_y,
                "x_label": unit_label_x,
                "y_label": unit_label_y,
            }
            results.append(data_dict)

        return results

    def Study_brutal_resuspension(
        self,
        dataframe_path: str | Path = None,
        change_main_path_images: str | list | Path = None,
        frames: int | list = None,
        normalize_distance: bool = True,
        path_save=None,
        do_save=False,
        units="time",
        curve_names: (str | list) = None,
        use_fit=True,
        fit_function: str = "sigmoid",
        use_mean: bool = False,
        kernel_size: int = None,
        path_velocity: str | Path = None,
        velocity_fit_coef: list = None,
    ):

        def line(x, slope, x1, y1):
            return slope * (x - x1) + y1

        if not isinstance(dataframe_path, (list, str, Path)):
            msg = "dataframe_path must be list, str or Path"
            raise TypeError(msg)

        if not isinstance(dataframe_path, list):
            dataframe_path = [dataframe_path]

        if path_save is None and do_save:
            msg = "Must specify a path to save picture"
            raise TypeError(msg)
        if path_save is not None and do_save is None:
            do_save = True

        if not (
            (units == "time")
            or (units == "frame")
            or (units == "velocity")
            or (units == "Reynolds")
        ):
            msg = "units varible is not time, frame, velocity or Reynolds"
            raise ValueError(msg)

        if path_velocity is not None:
            data_velocity = self._load_velocity(path_velocity)

            # _, ax = plt.subplots()
            # colors = cm.get_cmap("tab10")
            for i, data in enumerate(data_velocity):
                timestamp = data["timestamp"].to_numpy()
                velocity = data["velocity"].to_numpy()
                popt_velocity = None
                while popt_velocity is None:
                    popt_velocity, _, fit, r2 = self._fit_curve(
                        x=timestamp,
                        y=velocity,
                        func_base="poly_order_1",
                        max_retries=10,
                    )
                fit = self.functions_fitting["_poly_order"](
                    timestamp, *velocity_fit_coef
                )
            #     print(f"r2 = {r2:.2f}")
            #     print(popt_velocity)

            #     ax.plot(timestamp, velocity, color=colors(i), alpha=0.3, label="$U_{{raw}}$")
            #     ax.plot(timestamp, fit, color=colors(i), label=f"{popt_velocity[0]:.2f} $\\times U_{{raw}}$+{popt_velocity[1]:.2f}")
            # x_ticks = ax.get_xticks()[1:-1]
            # y_ticks = ax.get_yticks()[1:]
            # ax.set_xticks(x_ticks)
            # ax.set_yticks(y_ticks)
            # ax.set_xticklabels([f"{x_value:.2f}" for x_value in np.array(ax.get_xticks())], fontsize=16)
            # ax.set_yticklabels([f"{y_value:.2f}" for y_value in np.array(ax.get_yticks())], fontsize=16)
            # ax.set_xlabel("Temps [$s$]", fontsize=20)
            # ax.set_ylabel("Vitesse de l'écoulement [$m.s^{{-1}}$]", fontsize=20)
            # plt.legend(fontsize=12)
            # plt.show()

        fig, ax = plt.subplots()
        colors = cm.get_cmap("tab10")
        for i, data_path in enumerate(dataframe_path):
            keys = ["frame", "time", "main_path", "name", "x", "y", "diameter_mean"]

            dataframe = self._load_dataframe(
                data_path,
                usecols=keys,
                change_main_path_images=change_main_path_images[i],
            )

            print(data_path)

            if isinstance(frames, int):
                frames = [frames]
            if frames is None:
                frames = [dataframe["frame"].unique()]
            if not isinstance(frames, (int, np.ndarray, list)):
                msg = "frames variable must be int, list of ndarray of int"
                raise TypeError(msg)

            init_density = self._compute_surface_concentration(dataframe, frames=0)
            print(f"Initial density is {init_density:.2f}mm^-2")

            density = self._compute_surface_concentration(dataframe, frames=frames)
            print(f"Density before-after is {density[0]:.2f} - {density[-1]:.2f}mm^-2")

            mask = dataframe["frame"].isin(frames)

            sub_df = dataframe.loc[
                mask,
                keys,
            ].copy()

            if units == "time":
                x_values = sub_df["time"].unique() * sub_df["frame"].unique()
            elif units == "frame":
                x_values = sub_df["frame"].unique()

            # compute resuspension fraction
            fraction = 1 - np.array(
                [len(group["frame"]) for _, group in sub_df.groupby("frame")]
            ) / len(dataframe[dataframe["frame"] == 0])
            _, _, fit, r2 = self._fit_curve(
                x_values,
                fraction,
                func_base=fit_function,
            )
            print(f"r^2 = {r2:.2f}")

            # # compute nearest-neighbor distance from Voronoi
            # distances_nn = []
            # for _, group in sub_df.groupby("frame"):
            #     points = group[["x", "y"]].to_numpy()
            #     if len(points) > 2:
            #         vor = Voronoi(points)
            #         valid_pairs = []
            #         for (p1, p2), verts in zip(vor.ridge_points, vor.ridge_vertices):
            #             if all(v >= 0 for v in verts):
            #                 valid_pairs.append((p1, p2))
            #         valid_pairs = np.unique(np.sort(valid_pairs, axis=1), axis=0)
            #         dists = np.linalg.norm(points[valid_pairs[:, 0]] - points[valid_pairs[:, 1]], axis=1)
            #         if normalize_distance:
            #             dists = dists / np.mean(dists)
            #         distances_nn.append(dists.tolist())
            #     else:
            #         distances_nn.append([])

            # fig, ax = plt.subplots()
            # colors_hist = cm.get_cmap("plasma")
            # colors_fit = cm.get_cmap("viridis")
            # dist_fit = []
            # for i, dist in enumerate(distances_nn):
            #     hist, bins = np.histogram(dist, bins=100)
            #     centered_bins = (bins[:-1] + bins[1:]) / 2
            #     popt, _, fit_lognormal, r2_log_normal = self._fit_curve(
            #         x=centered_bins, y=hist, func_base="log_normal", n_components=1,
            #     )
            #     dist_fit.append(np.exp(popt[1]))
            #     ax.stairs(hist, bins, color=colors_hist(i / max(1, len(distances_nn) - 1)))
            #     ax.plot(centered_bins, fit_lognormal, color=colors_fit(i / max(1, len(distances_nn) - 1)))
            # x_ticks = ax.get_xticks()[1:]
            # y_ticks = ax.get_yticks()[1:]
            # ax.set_xticks(x_ticks)
            # ax.set_yticks(y_ticks)
            # ax.set_xticklabels([f"{x_value:.2f}" for x_value in np.array(ax.get_xticks())], fontsize=16)
            # ax.set_yticklabels([f"{y_value:.2f}" for y_value in np.array(ax.get_yticks())], fontsize=16)
            # ax.set_xlabel("Distance D entre les plus proches voisins [mm]", fontsize=20)
            # ax.set_ylabel("$\\dfrac{{dN}}{{d log(D)}}$", fontsize=20)

            # # plot nearest neighbor distance
            # fig, ax = plt.subplots()
            # ax.plot(x_values,
            #         dist_fit,
            #         color="tab:blue",
            #         )
            # x_ticks = ax.get_xticks()[1:]
            # y_ticks = ax.get_yticks()[1:]
            # ax.set_xticks(x_ticks)
            # ax.set_yticks(y_ticks)
            # ax.set_xticklabels([f"{x_value:.2f}" for x_value in np.array(ax.get_xticks())], fontsize=16)
            # ax.set_yticklabels([f"{y_value:.2f}" for y_value in np.array(ax.get_yticks())], fontsize=16)
            # ax.set_xlabel("Temps [s]", fontsize=20)
            # ax.set_ylabel("Distance D entre les plus proches voisins [mm]", fontsize=20)
            # plt.show()

            # # compute targed particles
            # for _, group in sub_df.groupby("frame"):
            #     counts, rects = self._compute_target_particles(group[["x", "y", "diameter_mean"]] / self.pixel_size)
            #     # fig, ax = plt.subplots()
            #     # path_image = Path(group["main_path"].unique()[0]) / Path(group["name"].unique()[0])
            #     # print(path_image)
            #     # img = self._load_image(path_image)
            #     # ax.imshow(img, cmap="gray")
            #     # ax.scatter(group["x"].to_numpy()/self.pixel_size, group["y"].to_numpy()/self.pixel_size, color="tab:orange")

            #     # for i in range(len(rects)):
            #     #     xmin, ymin = rects[i, 0], rects[i, 1]
            #     #     xmax, ymax = rects[i, 2], rects[i, 3]
            #     #     width, height = xmax - xmin, ymax - ymin
            #     #     rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor="r", alpha=0.3)
            #     #     ax.add_patch(rect)
            #     # plt.show()

            #     sigma, n = np.pi*group["diameter_mean"].to_numpy()**2/4, counts
            #     mean_free_path = 1 / np.mean((sigma * n))
            #     print(f"Mean free path = {mean_free_path:.2e} um")

            # fig, ax = plt.subplots()
            # hist, bins = np.histogram(counts, bins=100)
            # ax.stairs(hist, bins)
            # plt.show()

            # compute mean free path
            mean_free_path = (
                self._compute_mean_free_path(df=sub_df, frames=frames) / 1000
            )  # m
            # fig, ax = plt.subplots()
            # ax.plot(frames, mean_free_path)
            # x_ticks = ax.get_xticks()[1:-1]
            # ax.set_xticks(x_ticks)
            # ax.set_xticklabels([f"{x_value*self.time_interval:.2f}" for x_value in np.array(ax.get_xticks())], fontsize=16)
            # ax.set_xlabel("Temps [s]", fontsize=20)
            # y_ticks = ax.get_yticks()[1:]
            # ax.set_yticks(y_ticks)
            # ax.set_yticklabels([f"{y_value:.2f}" for y_value in np.array(ax.get_yticks())], fontsize=16)
            # ax.set_ylabel("Libre parcours moyen [$mm$]", fontsize=20)
            # plt.show()

            if velocity_fit_coef is not None:
                velocity = self.functions_fitting["_poly_order"](
                    timestamp, *velocity_fit_coef
                )
            else:
                velocity = self.functions_fitting["_poly_order"](
                    timestamp, *popt_velocity
                )
            coll_frequency = (
                0.0625 * velocity[frames] ** (7 / 8) / mean_free_path
            )  # m/s / m
            ax.plot(
                x_values,
                coll_frequency,
                color=colors(i),
                label=f"$C_0=${init_density:.2f} $mm^{{2}}$",
            )

        x_ticks = ax.get_xticks()[1:-1]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(
            [f"{x_value:.2f}" for x_value in np.array(ax.get_xticks())], fontsize=16
        )
        ax.set_xlabel("Temps [s]", fontsize=20)
        y_ticks = ax.get_yticks()[1:]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(
            [f"{y_value:.2f}" for y_value in np.array(ax.get_yticks())], fontsize=16
        )
        ax.set_ylabel("Fréquence de collisions [$s^{{-1}}$]", fontsize=20)
        fig.subplots_adjust()

        plt.legend(fontsize=12)
        if do_save:
            name = Path(path_save, "Collision_frequency.png")
            plt.savefig(name, dpi=120)
        else:
            plt.show()

            # # compute derivative
            # spl = UnivariateSpline(x_values, fit, s=0)

            # # find argmax of derivative
            # fprime = spl.derivative()(x_values)
            # fprime_arg_max = np.argmax(fprime)

            # # find derivative value of dicrete values
            # derivative_loc = [0.1, 0.25, 0.5, 0.75, 0.9]
            # fprime_arg = [int(p * (len(x_values) - 1)) for p in derivative_loc]
            # fprime = [spl.derivative()(min(x_values) + int((max(x_values) - min(x_values)) * p)) for p in derivative_loc]
            # print(f"Max value of derivative is {spl.derivative()(x_values[fprime_arg_max]):.2e} at {x_values[fprime_arg_max]:.2f}")
            # for arg, p in zip(fprime_arg, derivative_loc):
            #     print(f"Value of derivative at {x_values[arg]:.2f} s ({p*100:.0f}%) is {spl.derivative()(x_values[arg]):.2e} s^-1")

            # # plot fraction
            # ax.plot(
            #     x_values, fraction,
            #     color="tab:blue", label="Resuspended fraction",
            #     )

            # # plot fit
            # ax.plot(
            #     x_values, fit,
            #     color="tab:red", label=f"Fit resuspended fraction with $r^2$={r2:.2f}",
            #     )

            # x_center = x_values[fprime_arg_max]
            # delta = 0.1 * (max(x_values) - min(x_values))
            # x_range = np.linspace(x_center - delta, x_center + delta, 100)
            # # ax.plot(
            # #     x_range,
            # #     line(x_range, spl.derivative()(fprime_arg_max), x_center, fit[fprime_arg_max]),
            # #     color="black", label="Maximum rate of resuspension",
            # # )
            # ax.scatter(
            #     x_center, fit[fprime_arg_max], color="black",
            # )

            # # plot slope
            # for arg in fprime_arg:
            #     slope = spl.derivative()(x_values[arg])
            #     x_center, y_center = x_values[arg], fit[arg]
            #     delta = 0.1 * (max(x_values) - min(x_values))
            #     x_range = np.linspace(x_center - delta, x_center + delta, 100)
            #     ax.plot(
            #         x_range,
            #         line(x_range, slope, x_center, y_center),
            #         color="gray", label="Fit 1st order polynomial",
            #     )
            #     ax.scatter(
            #         x_center, y_center, color="gray",
            #     )

            # # plot coef dir.
            # step = 100
            # x_values_step = np.array([np.mean(x_values[i:i+step]) for i in range(0, len(x_values), step)])
            # derivative_batch = np.array([np.mean(spl.derivative()(x_values[i:i+step])) for i in range(0, len(x_values), step)])
            # ax_twin = ax.twinx()
            # ax_twin.plot(
            #     x_values_step,
            #     derivative_batch,
            #     color="tab:green", alpha=0.5,
            # )

            # if path_velocity is not None:
            #     ax_twin = ax.twinx()
            #     ax_twin.plot(
            #         x_values, velocity[frames],
            #         label="Vitesse",
            #         color="black", alpha=0.5,
            #         )

            #     ax_twin_y_ticks = ax_twin.get_yticks()[1:-1]
            #     ax_twin.set_yticks(ax_twin_y_ticks)
            #     ax_twin.set_yticklabels([f"{y_value:.2f}" for y_value in np.array(ax_twin.get_yticks())], fontsize=16)
            #     ax_twin.set_ylabel("Vitesse au centre de la veine", fontsize=18, color="black")
            #     ax_twin.tick_params(axis="y", colors="black")

            # x_ticks = ax.get_xticks()[1:-1]
            # ax.set_xticks(x_ticks)
            # if units == "time":
            #     ax.set_xticklabels([f"{x_value:.2f}" for x_value in np.array(ax.get_xticks())], fontsize=16)
            #     ax.set_xlabel("Temps $[s]$", fontsize=18)
            # elif units == "frame":
            #     ax.set_xticklabels([f"{x_value:.0f}" for x_value in np.array(ax.get_xticks())], fontsize=16)
            #     ax.set_xlabel("Frame number", fontsize=18)
            # elif units == "velocity":
            #     ax.set_xticklabels([f"{x_value:.2f}" for x_value in np.array(ax.get_xticks())], fontsize=16)
            #     ax.set_xlabel("Velocity $[m/s]$", fontsize=18)
            # elif units == "Reynolds":
            #     ax.set_xticklabels([f"{x_value:.2f}" for x_value in np.array(ax.get_xticks())], fontsize=16)
            #     ax.set_xlabel("Reynolds", fontsize=18)

            # y_ticks = ax.get_yticks()[1:]
            # ax.set_yticks(y_ticks)
            # ax.set_yticklabels([f"{y_value:.2f}" for y_value in np.array(ax.get_yticks())], fontsize=16)
            # ax.set_ylabel("Fraction de remise en suspension", fontsize=20)

            # if do_save:
            #     name = Path(path_save, "Visualize_resuspended_fraction.png")
            #     plt.savefig(name, dpi=120)
            # else:
            #     plt.show()

    def Study_CoefSlope_Density(
        self,
        dataframe_path: str | Path = None,
        change_main_path_images: str | list | Path = None,
        frames: int | list = None,
        fit_function: str = "sigmoid",
        path_save=None,
        do_save=False,
        units="time",
        curve_names: (str | list) = None,
    ):

        if not isinstance(dataframe_path, (list, str, Path)):
            msg = "dataframe_path must be list, str or Path"
            raise TypeError(msg)

        if not isinstance(dataframe_path, list):
            dataframe_path = [dataframe_path]

        if path_save is None and do_save:
            msg = "Must specify a path to save picture"
            raise TypeError(msg)
        if path_save is not None and do_save is None:
            do_save = True

        if not (
            (units == "time")
            or (units == "frame")
            or (units == "velocity")
            or (units == "Reynolds")
        ):
            msg = "units varible is not time, frame, velocity or Reynolds"
            raise ValueError(msg)

        _, ax = plt.subplots(1, 1, figsize=(8, 6))

        for i, data_path in enumerate(dataframe_path):
            keys = ["frame", "time", "main_path", "name", "x", "y", "diameter_mean"]

            dataframe = self._load_dataframe(
                data_path,
                usecols=keys,
                change_main_path_images=change_main_path_images[i],
            )

            print(data_path)

            if isinstance(frames, int):
                frames = [frames]
            if frames is None:
                frames = dataframe["frame"].unique()
            if not isinstance(frames, (int, np.ndarray, list)):
                msg = "frames variable must be int, list of ndarray of int"
                raise TypeError(msg)

            mask = dataframe["frame"].isin(frames)

            sub_df = dataframe.loc[
                mask,
                keys,
            ].copy()

            if units == "time":
                x_values = sub_df["time"].unique() * sub_df["frame"].unique()
            elif units == "frame":
                x_values = sub_df["frame"].unique()

            # conpute density
            density = self._compute_surface_concentration(sub_df, frames=frames)[:, 1]

            # compute resuspension fraction
            fraction = 1 - np.array(
                [len(group["frame"]) for _, group in sub_df.groupby("frame")]
            ) / len(dataframe[dataframe["frame"] == 0])

            # compute fit
            _, _, fit, r2 = self._fit_curve(
                x_values,
                fraction,
                func_base=fit_function,
            )
            print(f"r^2 = {r2:.2f}")

            # compute derivative
            spl = UnivariateSpline(x_values, fit, s=0)

            # derivative
            fprime = spl.derivative()(x_values)

            # # plot fraction
            # ax.plot(
            #     x_values, fraction,
            #     color="tab:blue", label="Resuspended fraction fitting",
            #     )

            # # plot density
            # ax.plot(
            #     x_values, density,
            #     color="tab:orange", label="Density",
            # )

            # plot fprime vs density stepped
            step = 100
            density_stepped = np.array(
                [np.mean(density[i : i + step]) for i in range(0, len(density), step)]
            )
            fprime_stepped = np.array(
                [
                    np.mean(spl.derivative()(x_values[i : i + step]))
                    for i in range(0, len(fprime), step)
                ]
            )
            ax.plot(
                density_stepped,
                fprime_stepped,
                color="tab:red",
                label=f"Fit density with $r^2$={r2:.2f}",
            )

            y_ticks = ax.get_yticks()[1:-1]
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(
                [f"{y_value:.2f}" for y_value in np.array(ax.get_yticks())], fontsize=16
            )
            ax.set_ylabel("Density stepped", fontsize=18, color="black")
            ax.tick_params(axis="y", colors="black")

            x_ticks = ax.get_xticks()[1:-1]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(
                [f"{x_value:.2f}" for x_value in np.array(ax.get_xticks())], fontsize=16
            )
            ax.set_xlabel("Temps $[s]$", fontsize=18)

            if do_save:
                name = Path(path_save, "Study_of_density_vs_slope_density.png")
                plt.savefig(name, dpi=120)
            else:
                plt.show()

    def Visualize_remaining_fraction(
        self,
        dataframe: pd.DataFrame = None,
        pixel_size: float = None,
        time_interval: float = 1 / 8000,
        frames: list | np.ndarray = None,
        labels: list | np.ndarray = None,
        change_main_path_images: str | list | Path = None,
        path_save: str = None,
        do_save: bool = False,
        x_unit: str = "time",
        y_unit: str = "fraction",
        curve_names: str | list = None,
        use_fit: bool = True,
        fit_function: str = "sigmoid",
        use_mean: bool = False,
        kernel_size: int = None,
        velocity: str | Path = None,
    ):

        if velocity is None:
            velocity = [None] * len(dataframe)

        results = []

        for run, data in enumerate(dataframe):
            # ----- frames
            if frames is None:
                selected_frames = data["frame"].unique()
            elif isinstance(frames, int):
                selected_frames = [frames]
            elif isinstance(frames, (list, np.ndarray)):
                selected_frames = frames
            else:
                msg = "'frames' must be int, list or np.ndarray of int"
                raise TypeError(msg)

            # ----- labels
            if labels == "all" or labels is None:
                selected_labels = sorted(data["label"].unique())
            else:
                selected_labels = labels

            # ------ filtering
            mask = data["frame"].isin(selected_frames) & data["label"].isin(
                selected_labels
            )

            sub_df = data.loc[mask, ["frame", "diameter_mean"]].copy()

            if sub_df.empty:
                continue

            # ----- select unit factor
            unit_factor_x = {
                "frames": 1,
                "time": time_interval,
                "f_velocity": 1.0,  # friction velocity
                "m_velocity"  # middle duct velocity
                "reynolds": 1.0,
            }[x_unit]

            unit_label_x = {
                "frames": "Frames",
                "time": f"Time $[s]$",
                "f_velocity": f"Friction velocity $[m/s]$",  # friction velocity
                "m_velocity": f"Middle duct velocity $[m/s]$",  # middle duct velocity
                "reynolds": "Reynolds number",
            }[x_unit]

            unit_factor_y = {
                "fraction": 1,
            }[y_unit]

            unit_label_y = {
                "fraction": f"$K_{{rem}}$",
            }[y_unit]

            # # ----- velocity
            # if velocity is not None:
            #     mask = (
            #         velocity["frame"].isin(selected_frames)
            #     )
            #     velocity = velocity.loc[
            #         mask,
            #         ["frame", "velocity"]
            #     ].copy()
            #     velocity["friction"] = 0.0564 * velocity["velocity"] ** (7/8)

            #     if velocity.empty:
            #         continue

            # ----- group data
            grouped = sub_df.groupby("frame")
            initial_num_parts = len(
                sub_df[sub_df["frame"] == min(sub_df["frame"].unique())]
            )

            # ----- fill data_dict
            data_dict = {
                "curves": True,
                "x": [sub_df["frame"].unique()],
                "y": [[(len(group) / initial_num_parts) for _, group in grouped]],
                "run": [run],
                "x_unit": unit_factor_x,
                "y_unit": unit_factor_y,
                "x_label": unit_label_x,
                "y_label": unit_label_y,
            }
            results.append(data_dict)

        return results

    def Visualize_remaining_fraction_multiple(
        self,
        dataframes_path=None,
        path_save=None,
        do_save=False,
        units="time",
        use_fit=True,
    ):

        if dataframes_path is None:
            msg = "Must specify dataframes path"
            raise ValueError(msg)

        if path_save is None and do_save:
            msg = "Must specify a path to save picture"
            raise TypeError(msg)
        if path_save is not None and do_save is None:
            do_save = True

        if not ((units == "time") or (units == "frame")):
            msg = "units varible is not time or frame"
            raise ValueError(msg)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        colors = ["tab:blue", "tab:orange", "tab:green"]

        for i, df_path in enumerate(dataframes_path):
            df = pd.DataFrame(pd.read_csv(df_path, index_col=[0]))

            fraction = 1 - np.array(
                [len(group["frame"]) for _, group in df.groupby("frame")]
            ) / len(df[df["frame"] == 0])

            if units == "time":
                ax.plot(
                    df["time"].unique(),
                    fraction,
                    color=colors[i],  # label="Raw remaining fraction",
                )

            elif units == "frame":
                ax.plot(
                    df["frame"].unique(),
                    fraction,
                    color=colors[i],  # label="Raw remaining fraction",
                )

        if units == "time":
            x_ticks = np.linspace(
                0,
                int(np.max(df["time"].unique()) / 1000) * 1000,
                int(np.max(df["time"].unique()) / 1000) + 1,
            )
            y_ticks = np.linspace(0, 1, 5)

            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)

            ax.set_xticklabels(x_ticks.astype(float), fontsize=18)
            ax.set_yticklabels(y_ticks.astype(float), fontsize=18)

            ax.set_xlabel("Time $[s]$", fontsize=20)
            ax.set_ylabel("Fraction of detached particles", fontsize=20)

        elif units == "frame":
            x_ticks = np.linspace(
                0,
                int(np.max(df["time"].unique()) / 1000) * 1000,
                int(df["time"].unique() / 1000) + 1,
            )
            y_ticks = np.linspace(
                0,
                int(df["time"].unique() / 1000) * 1000,
                int(df["time"].unique() / 1000) + 1,
            )

            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)

            ax.set_xticklabels(x_ticks.astype(float), fontsize=18)
            ax.set_yticklabels(y_ticks.astype(float), fontsize=18)

            ax.set_xlabel("frame [frame #]", fontsize=20)
            ax.set_ylabel("Resuspended fraction [frame #]", fontsize=20)

        plt.subplots_adjust()
        if use_fit:
            plt.legend(fontsize=12)
        if do_save:
            name = Path(path_save, "Visualize_remaining_fraction.png")
            plt.savefig(name, dpi=120)
        else:
            plt.show()

    def Visualize_resuspended_particle_frame_0(
        self, path_save=None, do_save=False, units="time"
    ):

        labels_frame_0 = self.dataframe[self.dataframe["frame"] == 0]["label"]

        fraction = []
        for _, group in self.dataframe.groupby("frame"):
            labels_frame_n = group["label"]
            same = np.array([lbl for lbl in labels_frame_0 if lbl in labels_frame_n])
            fraction.append(1 - (same / len(labels_frame_0)))

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        df = self.dataframe.copy()

        if units == "time":
            ax.plot(
                df["time"].unique(),
                fraction,
                color="tab:blue",
                label="Raw remaining fraction",
            )

        elif units == "frame":
            ax.plot(
                df["frame"].unique(),
                fraction,
                color="tab:blue",
                label="Raw remaining fraction",
            )

        if units == "time":
            x_ticks = np.linspace(
                0,
                int(np.max(df["time"].unique()) / 1000) * 1000,
                int(np.max(df["time"].unique()) / 1000) + 1,
            )
            y_ticks = np.linspace(0, 1, 5)

            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)

            ax.set_xticklabels(x_ticks.astype(float), fontsize=18)
            ax.set_yticklabels(y_ticks.astype(float), fontsize=18)

            ax.set_xlabel("Time [s]}", fontsize=20)
            ax.set_ylabel("Resuspended fraction", fontsize=20)

        elif units == "frame":
            x_ticks = np.linspace(
                0,
                int(np.max(df["time"].unique()) / 1000) * 1000,
                int(df["time"].unique() / 1000) + 1,
            )
            y_ticks = np.linspace(
                0,
                int(df["time"].unique() / 1000) * 1000,
                int(df["time"].unique() / 1000) + 1,
            )

            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)

            ax.set_xticklabels(x_ticks.astype(float), fontsize=18)
            ax.set_yticklabels(y_ticks.astype(float), fontsize=18)

            ax.set_xlabel(f"frame {r'[frame #]'}", fontsize=20)
            ax.set_ylabel(f"Resuspended fraction {r'[frame #]'}", fontsize=20)

        plt.subplots_adjust()
        if do_save:
            name = Path(path_save, "Visualize_remaining_fraction.png")
            plt.savefig(name, dpi=120)
        else:
            plt.show()

    def Visualize_mean_diameter_vs_velocity(
        self,
        path_save=None,
        do_save=False,
        units="max_velocity",
        max_velocity_measurements_file=None,
    ):

        if path_save is None and do_save:
            msg = "Must specify a path to save picture"
            raise TypeError(msg)
        if path_save is not None and do_save is None:
            do_save = True

        if not (
            (units == "max_velocity")
            or (units == "friction_velocity")
            or (units == "reynolds")
        ):
            msg = "units varible is not max_velocity or friction_velocity or reynolds"
            raise ValueError(msg)

        if max_velocity_measurements_file is None:
            msg = "max_velocity_measurements variable must  be fed"
            raise TypeError(msg)
        elif not isinstance(max_velocity_measurements_file, (Path, str)):
            msg = "max_velocity_measurements varible must be a path"
            raise TypeError(msg)

        frame_sizes = np.array(self.dataframe.groupby("frame").size())
        fraction = np.array(frame_sizes) / frame_sizes[0]

        if max_velocity_measurements_file is not None:
            time_stamp, voltage = self.Read_velocity_file(
                max_velocity_measurements_file
            )
            velocity_converted = 0.0
            for power, coeff in enumerate(
                self.polynomial_coefficient_tension_to_velocity
            ):
                velocity_converted += coeff * voltage**power

        mean_diameters = [
            group["diameter"].mean() for _, group in self.dataframe.groupby("frame")
        ]
        # mean_diameters = np.linspace(0, len(velocity_converted), len(velocity_converted))

        # fig, ax = plt.subplots()
        # ax.plot(
        #     time_stamp, voltage,
        # )
        # plt.show()
        # fig, ax = plt.subplots()
        # ax.plot(
        #     time_stamp, velocity_converted,
        # )
        # plt.show()
        # sys.exit(0)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        if units == "max_velocity":
            ax.plot(velocity_converted, mean_diameters, color="tab:blue")
            x_ticks = np.linspace(
                min(velocity_converted),
                max(velocity_converted),
                len(velocity_converted),
            )

        elif units == "friction_velocity":
            friction_velocity = 0.0625 * velocity_converted ** (7 / 8)
            ax.plot(friction_velocity, mean_diameters, color="tab:blue")
            x_ticks = np.linspace(
                min(friction_velocity), max(friction_velocity), len(friction_velocity)
            )

        elif units == "reynolds":
            friction_velocity = 0.0625 * velocity_converted ** (7 / 8)
            reynolds = friction_velocity * mean_diameters / self.nu_air
            ax.plot(reynolds, mean_diameters, color="tab:blue")
            x_ticks = np.linspace(min(reynolds), max(reynolds), len(reynolds))

        y_ticks = np.linspace(
            min(mean_diameters), max(mean_diameters), len(mean_diameters)
        )

        # print(x_ticks)
        # print(y_ticks)
        # # sys.exit(0)

        # ax.set_xticks(x_ticks)
        # ax.set_yticks(y_ticks)

        # ax.set_xticklabels(x_ticks.astype(float), fontsize=18)
        # ax.set_yticklabels(y_ticks.astype(float), fontsize=18)

        if units == "max_velocity":
            ax.set_xticklabels(ax.get_xticks(), fontsize=18)
            ax.set_yticklabels(ax.get_yticks() / self.pixel_size, fontsize=18)
            ax.set_xlabel("Velocity max $[m/s]$", fontsize=20)
            ax.set_ylabel("Mean diameter $[\\mu m]$", fontsize=20)

        elif units == "friction_velocity":
            ax.set_xticklabels(np.array(ax.get_xticks(), dtype=np.uint16), fontsize=18)
            ax.set_yticklabels(
                np.array(ax.get_yticks() / self.pixel_size, dtype=np.uint16),
                fontsize=18,
            )
            ax.set_xlabel("Friction velocity $[m/s]$", fontsize=20)
            ax.set_ylabel("Mean diameter $[\\mu m]$", fontsize=20)

        elif units == "reynolds":
            ax.set_xticklabels(np.array(ax.get_xticks(), dtype=np.uint16), fontsize=18)
            ax.set_yticklabels(
                np.array(ax.get_yticks() / self.pixel_size, dtype=np.uint16),
                fontsize=18,
            )
            ax.set_xlabel("Reynolds", fontsize=20)
            ax.set_ylabel("Mean diameter $[\\mu m]$", fontsize=20)

        plt.subplots_adjust()
        if do_save:
            name = Path(path_save, f"Visualize_mean_diameter_vs_{units}.png")
            plt.savefig(name, dpi=120)
        else:
            plt.show()

    def Visualize_velocity_flow(
        self,
        frames: list | int = None,
        labels: list | int = None,
        dataframe: pd.DataFrame = None,
        pixel_size: float = None,
        time_interval: float = None,
        x_unit: str = "frames",
        y_unit: str = "m/s",
        min_decimal_x = None,
        min_decimal_y = None,
        do_save: bool = False,
        path_save: str | Path = None,
        velocity: pd.DataFrame = None,
    ):

        if path_save is None and do_save:
            msg = "Must specify a path to save picture"
            raise TypeError(msg)
        if path_save is not None and do_save is None:
            do_save = True

        uncertainties = 0.03

        results = []

        for ii, vel in enumerate(velocity):

            # ----- frames
            if frames is None:
                selected_frames = vel["frame"].unique()
            elif isinstance(frames, int):
                selected_frames = [frames]
            elif isinstance(frames, (list, np.ndarray)):
                selected_frames = frames
            else:
                msg = "'frames' must be int, list or np.ndarray of int"
                raise TypeError(msg)

            # ----- filtering
            mask = vel["frame"].isin(selected_frames)

            sub_df = vel.loc[
                mask, ["frame", "voltage", "velocity"]
            ].copy()

            if sub_df.empty:
                continue

            unit_factor_x = {
                "frames": 1.0,
                "time": time_interval,
            }[x_unit]

            unit_label_x = {
                "frames": "",
                "time": "s",
            }[x_unit]
            
            min_decimals_x = {
                "frames": 0,
                "time": 3,
            }[x_unit]

            unit_factor_y = {
                "m/s": 1,
                "mm/s": 1000,
            }[y_unit]

            unit_label_y = {
                "m/s": "m/s",
                "mm/s": "mm/s",
            }[y_unit]
            
            min_decimals_y = {
                "m/s": 3,
                "mm/s": 3,
            }[y_unit]

            # compute linear regression + uncertainties on fit
            X = sm.add_constant(sub_df["frame"].to_numpy() * unit_factor_x)
            model = sm.OLS(sub_df["velocity"].to_numpy() * unit_factor_y, X) # Ordinary Least Squares
            fitting = model.fit()
            b, a = fitting.params
            b_err, a_err = fitting.bse # standart error
            ci = fitting.conf_int(alpha=0.05)
            # print(f"a = {a:.3e} ($\pm$ {a_err:.3e}), with confidence interval at 95% : {ci[1, 0]:.3f} at {ci[1, 1]:.3f}")
            # print(f"b = {b:.3e} ($\pm$ {b_err:.3e}), with confidence interval at 95% : {ci[0, 0]:.3f} at {ci[0, 1]:.3f}")
            # print(f"r2 = {fitting.rsquared:.2f}")

            data_dict = {
                "curves": True,
                "x": [sub_df["frame"]],
                "y": [sub_df["velocity"]],
                "run": [ii],
                "x_log": False,
                "label_curve": "Velocity in middle of the duct",
                "x_unit": unit_factor_x,
                "y_unit": unit_factor_y,
                "x_label": f"Time [{unit_label_x}]",
                "y_label": f"Velocity [{unit_label_y}]",
                "min_decimals_x": min_decimals_x,
                "min_decimals_y": min_decimals_y,
                "x_ticks_sci": False,
                "y_ticks_sci": False,
                "uncertainties": uncertainties,
                "fit_params": [[a, b, a_err, b_err]],
            }
            results.append(data_dict)

        return results
    
    def Visualize_friction_velocity(
        self,
        frames: list | int = None,
        labels: list | int = None,
        dataframe: pd.DataFrame = None,
        pixel_size: float = None,
        time_interval: float = None,
        x_unit: str = "frames",
        y_unit: str = "m/s",
        min_decimal_x = None,
        min_decimal_y = None,
        do_save: bool = False,
        path_save: str | Path = None,
        velocity: pd.DataFrame = None,
    ):

        if path_save is None and do_save:
            msg = "Must specify a path to save picture"
            raise TypeError(msg)
        if path_save is not None and do_save is None:
            do_save = True

        uncertainties = 0.03

        results = []

        for ii, vel in enumerate(velocity):

            # ----- frames
            if frames is None:
                selected_frames = vel["frame"].unique()
            elif isinstance(frames, int):
                selected_frames = [frames]
            elif isinstance(frames, (list, np.ndarray)):
                selected_frames = frames
            else:
                msg = "'frames' must be int, list or np.ndarray of int"
                raise TypeError(msg)

            # ----- filtering
            mask = vel["frame"].isin(selected_frames)

            sub_df = vel.loc[
                mask, ["frame", "voltage", "velocity"]
            ].copy()

            if sub_df.empty:
                continue

            if "friction" not in sub_df:
                sub_df["friction"] = 0.0564 * sub_df["velocity"] ** (7 / 8)  # m/s

            unit_factor_x = {
                "frames": 1.0,
                "time": time_interval,
            }[x_unit]

            unit_label_x = {
                "frames": "",
                "time": "s",
            }[x_unit]
            
            min_decimals_x = {
                "frames": 0,
                "time": 3,
            }[x_unit]

            unit_factor_y = {
                "m/s": 1,
                "mm/s": 1000,
            }[y_unit]

            unit_label_y = {
                "m/s": "m/s",
                "mm/s": "mm/s",
            }[y_unit]
            
            min_decimals_y = {
                "m/s": 3,
                "mm/s": 3,
            }[y_unit]

            # compute linear regression + uncertainties on fit
            X = sm.add_constant(sub_df["frame"].to_numpy() * unit_factor_x)
            model = sm.OLS(sub_df["friction"].to_numpy() * unit_factor_y, X) # Ordinary Least Squares
            fitting = model.fit()
            b, a = fitting.params
            b_err, a_err = fitting.bse # standart error
            ci = fitting.conf_int(alpha=0.05)
            print(f"a = {a:.3e} ($\pm$ {a_err:.3e}), with confidence interval at 95% : {ci[1, 0]:.3f} at {ci[1, 1]:.3f}")
            print(f"b = {b:.3e} ($\pm$ {b_err:.3e}), with confidence interval at 95% : {ci[0, 0]:.3f} at {ci[0, 1]:.3f}")

            data_dict = {
                "curves": True,
                "x": [sub_df["frame"]],
                "y": [sub_df["friction"]],
                "run": [ii],
                "x_log": False,
                "label_curve": "Velocity in middle of the duct",
                "x_unit": unit_factor_x,
                "y_unit": unit_factor_y,
                "x_label": f"Time [{unit_label_x}]",
                "y_label": f"Velocity [{unit_label_y}]",
                "min_decimals_x": min_decimals_x,
                "min_decimals_y": min_decimals_y,
                "x_ticks_sci": False,
                "y_ticks_sci": False,
                "uncertainties": uncertainties,
            }
            results.append(data_dict)

        return results

    def Visualize_histogram_diameters(
        self,
        dataframe: pd.DataFrame = None,
        pixel_size: float = None,
        frames: int | list | np.ndarray = None,
        labels: int | list | np.ndarray = None,
        fit_function: str = None,
        n_components: int = 1,
        d_50: int = None,
        n_bins: int = 50,
        do_diameter_cut_off: bool = True,
        density: bool = True,
        path_save: str | Path = None,
        do_save: bool = False,
        x_unit: str = "um",
        y_unit: str = "number",
    ):

        if not isinstance(n_components, int):
            msg = "'n_components' must be integer"
            raise TypeError(msg)
        if n_components < 1:
            msg = "n_components variable must be integer greater or equal to 1"
            raise ValueError(msg)

        if d_50 is not None:
            if isinstance(n_bins, int):
                msg = "'d_50' must be float"
                raise TypeError(msg)

        if not isinstance(n_bins, int):
            msg = "'n_bins must be integer"
            raise TypeError(msg)

        if path_save is None and do_save:
            msg = "Must specify a path to save picture"
            raise TypeError(msg)
        if path_save is not None and do_save is None:
            do_save = True

        colors = cm.get_cmap("tab10")

        results = []

        for data in dataframe:
            # ----- frames
            if frames is None:
                selected_frames = data["frame"].unique()
            elif isinstance(frames, int):
                selected_frames = [frames]
            elif isinstance(frames, (list, np.ndarray)):
                selected_frames = frames
            else:
                msg = "'frames' must be int, list or np.ndarray of int"
                raise TypeError(msg)

            mask_keys = ["frame", "diameter_mean"]

            # ----- filtering
            mask = data["frame"].isin(frames)

            sub_df = data.loc[mask, mask_keys].copy()

            if sub_df.empty:
                continue

            unit_factor_x = {"px": 1, "um": pixel_size, "mm": pixel_size / 1000}[x_unit]

            unit_factor_y = {
                "number": 1,
                "density": 1.0,
            }[y_unit]

            sub_df["diameter_mean"] *= unit_factor_x

            # ----- group data
            grouped = sub_df.groupby("frame")

            data_dict = {
                "histogram": True,
                "bins": [
                    np.histogram(
                        group["diameter_mean"],
                        bins=n_bins,
                        density=True if y_unit == "density" else False,
                    )[1]
                    for _, group in grouped
                ],
                "hist": [
                    np.histogram(
                        group["diameter_mean"],
                        bins=n_bins,
                        density=True if y_unit == "density" else False,
                    )[0]
                    for _, group in grouped
                ],
                "frame": [frame for frame, _ in grouped],
                "fit": True,
                "x_log": False,
                "d_50": np.median(
                    [
                        np.histogram(group["diameter_mean"], bins=n_bins)[1]
                        for _, group in grouped
                    ]
                ),
                "label_curve": "Particle sizing distribution",
                "x_unit": unit_factor_x,
                "y_unit": unit_factor_y,
                "x_label": "Equivalent diameters $D_p$ $[\\mu m]$",
                "y_label": "$\\dfrac{{dN}}{{d \\, log(D_p)}}$",
            }
            results.append(data_dict)

        return results

        # data_dict = {}

        # for data in dataframe:

        #     if isinstance(frames, int):
        #         frames = [frames]
        #     if frames is None:
        #         frames = data["frame"].unique()
        #     if not isinstance(frames, (int, np.ndarray, list)):
        #         msg = "'frames' must be int, list or ndarray of int"
        #         raise TypeError(msg)

        #     mask_frames = data["frame"].isin(frames)

        #     mask_keys = [
        #         "frame", "main_path", "name", "time", "label", "diameter_mean",
        #         "x", "y"
        #         ]

        #     sub_df = data.loc[mask_frames][mask_keys].copy()

        #     data_dict = {
        #         "histogram": True,
        #         "frame": [],
        #         "bins": [],
        #         "hist": [],
        #         "fit": True,
        #         "x_log": True,
        #         "d_50": [],
        #         "color": "tab:blue",
        #         "edgecolor": "black",
        #         "fill": True,
        #         "label": "Particle sizing distribution",
        #         "x_unit": 1000,
        #         "y_unit": 1,
        #         "x_label": "Equivalent diameters $D_p$ $[\\mu m]$",
        #         "y_label": "$\\dfrac{{dN}}{{d \\, log(D_p)}}$",
        #     }

        #     for frame_id, group in sub_df.groupby("frame"):

        #         diameters = group["diameter_mean"].to_numpy() * pixel_size
        #         hist, bins = np.histogram(diameters, bins=n_bins)

        #         data_dict["frame"].append([frame_id])
        #         data_dict["bins"].append(bins)
        #         data_dict["hist"].append(hist)
        #         data_dict["d_50"].append([np.median(diameters)])

        # return data_dict

    def Visualize_mean_diameter(
        self,
        dataframe: Path | str | list = None,
        velocity: Path | str | list = None,
        frames: list = None,
        labels: list = None,
        path_save: Path | list | str = None,
        do_save: bool = False,
        x_unit: str = "time",
        y_unit: str = "mm",
    ):

        if not isinstance(path_dataframe, (list, str, Path)):
            msg = "dataframe must be list or pd.DataFrame type"
            raise TypeError(msg)
        if not isinstance(path_dataframe, list):
            path_dataframe = [path_dataframe]

        if path_velocity is not None:
            if not isinstance(path_velocity, (list, str, Path)):
                msg = "dataframe must be list or pd.DataFrame type"
                raise TypeError(msg)
            if not isinstance(path_velocity, list):
                path_velocity = [path_velocity]
        else:
            path_velocity = [None] * len(path_dataframe)

        if x_unit not in [
            "time",
            "frame",
            "velocity",
            "Reynolds_duct",
            "Reynolds_friction",
        ]:
            msg = f"'units' varible is not time, frame, velocity or Reynolds_duct or Reynolds_friction. Not {x_unit}"
            raise ValueError(msg)

        results = []

        for i, (data, vel) in enumerate(zip(dataframe, velocity)):
            keys = [
                "frame",
                "main_path",
                "name",
                "time",
                "label",
                "diameter_mean",
            ]

            if isinstance(frames, int):
                frames = [frames]
            if frames is None:
                frames = [dataframe["frame"].unique()]
            if not isinstance(frames, (int, np.ndarray, list)):
                msg = "'frames' must be int, list or ndarray of int"
                raise TypeError(msg)

            mask_frames = dataframe["frame"].isin(frames)

            sub_df = dataframe.loc[
                mask,
                keys,
            ].copy()
            frames = sub_df["frame"].unique()

            mean_diameter_per_frame = sub_df.groupby("frame")["diameter_mean"].mean()
            std_diameter_per_frame = (
                sub_df.groupby("frame")["diameter_mean"].std() / 2.0
            )

            # if use_mean is not None and kernel_size is not None:
            #     kernel = [1] * kernel_size
            #     mean_diameter_per_frame = scipy.signal.convolve(mean_diameter_per_frame, kernel, mode="same")
            #     conv = scipy.signal.convolve(mean_diameter_per_frame, kernel, mode="same")
            #     mean_diameter_per_frame = conv/max(conv) * max(mean_diameter_per_frame)

            initial_density = self._compute_surface_concentration(
                sub_df, int(frames[0])
            )

            ax.plot(
                frames,
                mean_diameter_per_frame,
                color="tab:blue",
                label=f"Mean diameter for $C_0={initial_density:.2f} mm^{{-2}}$",
            )

            ax.fill_between(
                frames,
                y1=mean_diameter_per_frame - std_diameter_per_frame,
                y2=mean_diameter_per_frame + std_diameter_per_frame,
                color="tab:blue",
                alpha=0.2,
            )

            ax.plot(
                frames,
                mean_diameter_per_frame,
                color="tab:blue",
            )

            ax.fill_between(
                frames,
                y1=mean_diameter_per_frame - std_diameter_per_frame,
                y2=mean_diameter_per_frame + std_diameter_per_frame,
                color="tab:blue",
                alpha=0.2,
            )

        if use_fit:
            _, _, fit_func, _ = self._fit_curve(frames, mean_diameter_per_frame)
            ax.plot(
                frames,
                fit_func,
                color="tab:red",
                label="Fit : $A + \\frac{{K-A}}{{(C + Q e^{{-Bt}})^{{1/\\nu}}}}$",
            )

        x_ticks = ax.get_xticks()[1:-1]
        y_ticks = ax.get_yticks()[1:-1]
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)

        ax.set_yticklabels(
            [f"{y_tick * 1000:.0f}" for y_tick in np.array(y_ticks)], fontsize=16
        )

        if x_unit == "time":
            if language == "en":
                ax.set_xlabel("Time $[s]$", fontsize=self.dict_fontsize["label"])
            if language == "fr":
                ax.set_xlabel("Temps $[s]$", fontsize=self.dict_fontsize["label"])
            ax.set_xticklabels(
                [f"{x_tick * self.time_interval:.1f}" for x_tick in x_ticks],
                fontsize=16,
            )

        elif x_unit == "frame":
            if language == "en":
                ax.set_xlabel("Frame", fontsize=self.dict_fontsize["label"])
            if language == "fr":
                ax.set_xlabel("Image", fontsize=self.dict_fontsize["label"])
            ax.set_xticklabels([f"{x_tick:.1f}" for x_tick in x_ticks], fontsize=16)

        if y_unit == "um":
            if language == "en":
                ax.set_ylabel(
                    "Mean diameters [$\\mu m$]", fontsize=self.dict_fontsize["label"]
                )
            if language == "fr":
                ax.set_ylabel(
                    "Diamètres moyens [$\\mu m$]", fontsize=self.dict_fontsize["label"]
                )
            ax.set_xticklabels(
                [f"{y_tick:.1f}" for y_tick in y_ticks],
                fontsize=self.dict_fontsize["ticks"],
            )

        elif y_unit == "mm":
            if language == "en":
                ax.set_ylabel(
                    "Mean diameters [$mm$]", fontsize=self.dict_fontsize["label"]
                )
            if language == "fr":
                ax.set_ylabel(
                    "Diamètres moyens [$mm$]", fontsize=self.dict_fontsize["label"]
                )
            ax.set_xticklabels(
                [f"{y_tick / 1000:.1f}" for y_tick in y_ticks],
                fontsize=self.dict_fontsize["ticks"],
            )

        plt.subplots_adjust(**self.dict_fontsize["subplots"])

        # if do_save:
        #     name = Path(path_save, "Visualize_mean_diameter.png")
        #     plt.savefig(name, dpi=120)
        # else:
        #     plt.show()

        plt.legend(fontsize=self.dict_fontsize["legend"])

        plt.show()

    def Visualize_labels(
        self,
        dataframe: str | Path = None,
        pixel_size=None,
        frames: int | list | np.ndarray = None,
        labels: int | list | np.ndarray = None,
        change_main_path_images: str | list = None,
        x_unit: str = "px",
        y_unit: str = "px",
        path_save: Path | str = None,
        do_save: bool = False,
        unit: str = "px",
        rotate_image: int = None,
        rotate_coords: int = None,
    ):

        for i, data in enumerate(dataframe):
            mask_keys = [
                "frame",
                "main_path",
                "name",
                "time",
                "label",
                "x",
                "y",
                "coords_pixels",
                "diameter_mean",
                "diameter_std",
                "mass_mean",
                "mass_std",
                "velocity",
                "acceleration",
                "momentum",
                "kinetic",
                # "cluster_id", "cluster_label",
                # "collision",
            ]

            if isinstance(frames, int):
                frames = [frames]
            if frames is None:
                frames = [data["frame"].unique()]
            if not isinstance(frames, (int, np.ndarray, list)):
                msg = "frames variable must be int, list of ndarray of int"
                raise TypeError(msg)

            labels = data["label"].unique()

            mask_frames = data["frame"].isin(frames)
            mask_labels = data["label"].isin(labels)

            sub_df = data.loc[mask_frames, mask_keys].copy()

            name = Path(
                f"{sub_df['main_path'].unique()[0]}\\{sub_df['name'].unique()[0]}"
            )
            img = self._load_image(name, invert=False, rotate_image=rotate_image)

            data_dict = {
                "histogram": False,
                "image": img,
                "x": [],
                "y": [],
                "time": [],
                "coords_pixels": [],
                "label": [],
                "diameter_mean": [],
                "mass_mean": [],
                "velocity": [],
                "acceleration": [],
                "momentum": [],
                "kinetic": [],
                # "collision": [],
                # "cluster_id": [],
                "x_log": False,
                "x_label": "X $[px]$",
                "y_label": "Y $[px]$",
                "x_unit": 1.0,
                "y_unit": 1.0,
                "color": [],
            }

        for _, lbl in enumerate(labels):
            df = sub_df[sub_df["label"] == lbl].copy()

            if x_unit == "px":
                x = data["x"]
            elif x_unit == "mm":
                x = data["x"] * pixel_size
            elif x_unit == "m":
                x = data["x"] * pixel_size / 1000
            else:
                msg = "Unknown 'x_unit'"
                raise TypeError(msg)

            if y_unit == "px":
                y = data["y"]
            elif y_unit == "mm":
                y = data["y"] * pixel_size
            elif y_unit == "m":
                y = data["y"] * pixel_size / 1000
            else:
                msg = "Unknown 'y_unit'"
                raise TypeError(msg)

            data_dict["x"].append(x)
            data_dict["y"].append(y)
            data_dict["time"].append(data["time"])
            data_dict["coords_pixels"].append(data["coords_pixels"])
            data_dict["diameter_mean"].append(data["diameter_mean"])
            data_dict["mass_mean"].append(data["mass_mean"])
            data_dict["velocity"].append(data["velocity"])
            data_dict["acceleration"].append(data["acceleration"])
            data_dict["momentum"].append(data["momentum"])
            data_dict["kinetic"].append(data["kinetic"])
            # data_dict["collision"].append(data["collision"])
            # data_dict["cluster_id"].append(data["cluster_id"])
            data_dict["label"].append(lbl)

        return data_dict

        # _, ax = plt.subplots(figsize=(8, 6))

        # path_image = Path(group["main_path"].unique()[0]) / Path(group["name"].unique()[0])
        # img = Image.open(path_image).convert("L")

        # if rotate_image is not None:
        #     if rotate_image == 90: img = img.transpose(Image.ROTATE_90)
        #     elif rotate_image == 180: img = img.transpose(Image.ROTATE_180)
        #     elif rotate_image == 270: img = img.transpose(Image.ROTATE_270)

        # if rotate_coords in [90, 180, 270]:
        #     coords = self._rotate_coords(group[["x", "y"]].to_numpy(), np.array(img).shape[1], np.array(img).shape[0], rotate_coords)
        #     x_converted, y_converted = coords[:, 0], coords[:, 1]
        # elif rotate_coords == 0:
        #     x_converted, y_converted = group["x"].to_numpy()/self.pixel_size, group["y"].to_numpy()/self.pixel_size

        # img = np.array(img, dtype=np.uint8)

        # ax.imshow(img, cmap="gray")

        # # x_converted = (
        # #     group["x"] * self.pixel_size if unit == "milli"
        # #     else group["x"] if unit == "px"
        # #     else group["x"] * self.pixel_size / 1000.0 if unit == "meter"
        # #     else None
        # # )

        # # y_converted = (
        # #     group["y"] * self.pixel_size if unit == "milli"
        # #     else group["y"] if unit == "px"
        # #     else group["y"] * self.pixel_size / 1000.0 if unit == "meter"
        # #     else None
        # # )

        # # diameter_converted = (
        # #     group["diameter_mean"] if unit == "milli"
        # #     else group["diameter_mean"] if unit == "px"
        # #     else group["diameter_mean"] * self.pixel_size / 1000.0 if unit == "meter"
        # #     else None
        # # )

        # scatter = ax.scatter(
        #     x_converted,  # [milli, meter, px]
        #     y_converted,  # [milli, meter, px]
        #     color="tab:blue",
        #     # s=diameter_converted, # [milli, meter, px]
        #     # alpha=(i + 1) / len(frames),
        # )

        # if unit == "milli":
        #     x_ticks = ax.get_xticks()[1:-1]
        #     y_ticks = ax.get_yticks()[1:-1]
        #     ax.set_xticks(x_ticks)
        #     ax.set_yticks(y_ticks)
        #     ax.set_xticklabels([f"{x_tick * self.pixel_size:.2f}" for x_tick in x_ticks], fontsize=16)
        #     ax.set_yticklabels([f"{y_tick * self.pixel_size:.2f}" for y_tick in y_ticks], fontsize=16)
        #     ax.set_xlabel("x $[mm]$", fontsize=20)
        #     ax.set_ylabel("y $[mm]$", fontsize=20)

        # if unit == "px":
        #     x_ticks = ax.get_xticks()[1:-1]
        #     y_ticks = ax.get_yticks()[1:-1]
        #     ax.set_xticks(x_ticks)
        #     ax.set_yticks(y_ticks)
        #     ax.set_xticklabels([f"{x_tick:.0f}" for x_tick in x_ticks], fontsize=16)
        #     ax.set_yticklabels([f"{y_tick:.0f}" for y_tick in y_ticks], fontsize=16)
        #     ax.set_xlabel("x $[px]$", fontsize=20)
        #     ax.set_ylabel("y $[px]$", fontsize=20)

        # elif unit == "meter":
        #     x_ticks = ax.get_xticks()[1:-1]
        #     y_ticks = ax.get_yticks()[1:-1]
        #     ax.set_xticks(x_ticks)
        #     ax.set_yticks(y_ticks)
        #     ax.set_xticklabels([f"{x_tick * self.pixel_size:.2f}" for x_tick in x_ticks], fontsize=16)
        #     ax.set_yticklabels([f"{y_tick * self.pixel_size:.2f}" for y_tick in y_ticks], fontsize=16)
        #     ax.set_xlabel("x $[m]$", fontsize=20)
        #     ax.set_ylabel("y $[m]$", fontsize=20)

        # cursor = mplcursors.cursor(scatter, hover=True)

        # @cursor.connect("add")
        # def on_add(sel):
        #     index = sel.index
        #     sel.annotation.set_text(
        #         f"Image number : {frame_id:d}, t : {sub_df['time'].iloc[index]*1000:.3f} $ms$\n"
        #         f"Label : {sub_df['label'].iloc[index]}\n"
        #         f"X : {sub_df['x'].iloc[index]:.2f} $m$, Y : {sub_df['y'].iloc[index]:.2f} $m$\n"
        #         f"Diameter {sub_df['diameter_mean'].iloc[index]:.2f} $m$\n"
        #         f"m : {sub_df['mass_mean'].iloc[index]:.2e} kg \u00b1 {sub_df['mass_std'].iloc[index]:.2e}\n"
        #         f"v : {sub_df['velocity'].iloc[index]:.2e} $m.s^{{-1}}$\n"
        #         f"a : {sub_df['acceleration'].iloc[index]:.2e} $m.s^{{-2}}$\n"
        #         f"p : {sub_df['momentum'].iloc[index]:.2e} $kg m.s^{{-1}}$\n"
        #         f"K : {sub_df['kinetic'].iloc[index]:.2e} $J$\n"
        #         # f"Collision : {sub_df['collision'].iloc[index]}"
        #         # f"cluster_id : {sub_df['cluster_id'].iloc[index]}"
        #     )
        #     sel.annotation.get_bbox_patch().set(alpha=0.8, color="lightblue")
        #     sel.annotation.arrow_patch.set(
        #         arrowstyle="simple", fc="white", alpha=0.5
        #     )

    def Visualize_num_labels(
        self,
        dataframe: str | Path = None,
        frames: int | list | np.ndarray = None,
        labels: int | list | np.ndarray = None,
        n_bins: int = 50,
        x_unit: str = "label",
        y_unit: str = "number",
        pixel_size: float = 1.0,
        time_interval: float = 1 / 8000,
        path_save: Path | str = None,
        do_save: bool = False,
        language: str = "en",
    ):

        colors = cm.get_cmap("tab10")

        results = []

        for data in dataframe:
            mask_keys = ["frame", "main_path", "name", "time", "label"]

            if isinstance(frames, int):
                frames = [frames]
            if frames is None:
                frames = [data["frame"].unique()]
            if not isinstance(frames, (int, np.ndarray, list)):
                msg = "'frames' must be int, list or ndarray of int"
                raise TypeError(msg)

            labels = data["label"].unique()

            mask_frames = data["frame"].isin(frames)
            mask_labels = data["label"].isin(labels)
            mask = mask_frames & mask_labels

            sub_df = data.loc[mask, mask_keys].copy()

            if sub_df.empty:
                continue

            unit_factor_x = {
                "label": 1,
            }[x_unit]

            unit_factor_y = {
                "number": 1,
                "ratio": 1,
            }[y_unit]

            # ----- group data
            grouped = sub_df.groupby("frame")

            data_dict = {
                "histogram": True,
                "bins": [
                    np.histogram(
                        group["label"],
                        bins=len(group["label"].unique()) - 1,
                        density=False,
                    )[1]
                    for _, group in grouped
                ],
                "hist": [
                    np.histogram(
                        group["label"],
                        bins=len(group["label"].unique()) - 1,
                        density=False,
                    )[0]
                    for _, group in grouped
                ],
                "x_log": False,
                "label_curve": "Labels",
                "x_unit": unit_factor_x,
                "y_unit": unit_factor_y,
                "x_label": "Labels",
                "y_label": "Number of labels",
            }

            results.append(data_dict)

        return results

        # # plot histogram
        # n_bins = n_bins if n_bins is not None else max(group["label"])
        # hist, edges = np.histogram(group["label"], bins=n_bins, density=False)
        # bin_centers = (edges[:-1] + edges[1:]) / 2

        # hist, bins, patches = ax.hist(
        #     group["label"], bins=n_bins,
        #     color="tab:blue", edgecolor="black", density=False,
        # )

    def Track_particles_velocity(
        self,
        dataframe: pd.DataFrame = None,
        velocity: pd.DataFrame = None,
        pixel_size: float = 1.0,
        time_interval: float =  None,
        frames: list | int = None,
        labels: list | int = None,
        x_unit: str = "mm",
        y_unit: str = "mm",
        path_save: str = None,
        do_save: bool = False,
        rotate: int = 0,
        crop: tuple = (1, 1),
    ):

        if isinstance(frames, int):
            frames = [frames]

        if path_save is None and do_save:
            msg = "Must specify a path to save picture"
            raise TypeError(msg)
        if path_save is not None and do_save is None:
            do_save = True

        results = []

        for data in dataframe:
            # ----- frames
            if frames is None:
                selected_frames = data["frame"].unique()
            elif isinstance(frames, int):
                selected_frames = [frames]
            elif isinstance(frames, (list, np.ndarray)):
                selected_frames = frames
            else:
                msg = "'frames' must be int, list or np.ndarray of int"
                raise TypeError(msg)

            # ----- labels
            if labels == "all" or labels is None:
                selected_labels = sorted(data["label"].unique())
            else:
                selected_labels = labels

            # ----- filtering
            mask = data["frame"].isin(selected_frames) & data["label"].isin(
                selected_labels
            )

            sub_df = data.loc[
                mask, ["frame", "main_path", "name", "label", "x", "y", "dt"]
            ].copy()

            if sub_df.empty:
                continue

            x_unit_factor = {"px": 1, "mm": pixel_size, "m": pixel_size / 1000}[x_unit]

            x_unit_label = {
                "px": "X [px]",
                "mm": "X [mm]",
                "m": "X [m]",
            }[x_unit]

            min_decimals_x = {
                "px": 0,
                "mm": 1,
                "m": 3,
            }[x_unit]

            y_unit_factor = {"px": 1, "mm": pixel_size, "m": pixel_size / 1000}[y_unit]

            y_unit_label = {
                "px": "Y [px]",
                "mm": "Y [mm]",
                "m": "Y [m]",
            }[y_unit]

            min_decimals_y = {
                "px": 0,
                "mm": 1,
                "m": 3,
            }[y_unit]

            # ----- load image
            name = Path(
                data["main_path"].iloc[0],
                data["name"].iloc[0],
                # Path("D:\\BISE_experiments\\Essai_7\\8000Hz\\4x10mm3\\3\\Images"),
            )
            img = self._load_image(name, invert=False, rotate_image=rotate)

            # ----- compute velocity
            df = sub_df.sort_values(by=["label", "frame"])
            dt = sub_df["dt"].unique()

            df["dx"] = df.groupby("label")["x"].diff().fillna(0.0)  # [px, m, mm]
            df["dy"] = df.groupby("label")["y"].diff().fillna(0.0)  # [px, m, mm]

            df["vx"] = df.groupby("label")["dx"].transform(
                lambda x: x / 1.0
            )  # [px/s, m/s, mm/s]
            df["vy"] = df.groupby("label")["dy"].transform(
                lambda x: x / 1.0
            )  # [px/s, m/s, mm/s]

            df["disp"] = np.sqrt(
                df.groupby("label")["dx"].transform(lambda x: x**2)
                + df.groupby("label")["dy"].transform(lambda x: x**2)
            )  # [px, m, mm]
            df["velocity"] = df.groupby("label")["disp"].transform(
                lambda x: x / 1.0
            )  # [px/s, m/s, mm/s]

            grouped_label = df.groupby("label")
            data_dict = {
                "image": img,
                "x": [group["x"].to_numpy() for _, group in grouped_label],
                "y": [group["y"].to_numpy() for _, group in grouped_label],
                "vx": [group["vx"].to_numpy() for _, group in grouped_label],
                "vy": [group["vy"].to_numpy() for _, group in grouped_label],
                "label": list(group["label"].unique() for _, group in grouped_label),
                "x_unit": x_unit_factor,
                "y_unit": y_unit_factor,
                "x_label": f"{x_unit_label}",
                "y_label": f"{y_unit_label}",
                "min_decimals_x": min_decimals_x,
                "min_decimals_y": min_decimals_y,
            }
            results.append(data_dict)

        return results

        # colors = cm.get_cmap("tab10")

        # for i, data in enumerate(dataframe):

        #     mask_keys = [
        #         "frame", "main_path", "name", "time", "label",
        #         "x", "y", "vx", "vy", "diameter_mean",
        #         ]

        #     if isinstance(frames, int):
        #         frames = [frames]
        #     if frames is None:
        #         frames = [dataframe["frame"].unique()]
        #     if not isinstance(frames, (int, np.ndarray, list)):
        #         msg = "'frames' must be int, list or ndarray of int"
        #         raise TypeError(msg)

        #     if labels == "all":
        #         labels = sorted(data["label"].unique())

        #     mask_frames = data["frame"].isin(frames)
        #     mask_labels = data["frame"].isin(labels)
        #     sub_df = data.loc[mask_frames & mask_labels, mask_keys].copy()
        #     frames = sub_df["frame"].unique()

        #     name = Path(f"{sub_df['main_path'].unique()[0]}\\{sub_df['name'].unique()[0]}")
        #     img = self._load_image(name, invert=False, rotate_image=rotate)

        #     # data_dict = {
        #     #     "image": img,
        #     #     "x": [[]],
        #     #     "y": [[]],
        #     #     "vx": [[]],
        #     #     "vy": [[]],
        #     #     "diameter_mean": [[]],
        #     #     "label": [[]],
        #     #     "x_unit": "px",
        #     #     "y_unit": "px",
        #     #     "x_label": "X $[px]$",
        #     #     "y_label": "Y $[px]$",
        #     # }

        #     data_dict = []

        #     for j, lbl in enumerate(labels):

        #         df = sub_df[sub_df["label"] == lbl].copy()

        #         data_dict.append(df)

        #         # if x_unit == "px":
        #         #     x = df["x"]
        #         # elif x_unit == "mm":
        #         #     x = df["x"] * pixel_size
        #         # elif x_unit == "m":
        #         #     x = df["x"] * pixel_size / 1000

        #         # if y_unit == "px":
        #         #     y = df["y"]
        #         # elif y_unit == "mm":
        #         #     y = df["y"] * pixel_size
        #         # elif y_unit == "m":
        #         #     y = df["y"] * pixel_size / 1000

        #         # if x_unit == "px":
        #         #     vx = df["vx"]
        #         # elif x_unit == "mm":
        #         #     vx = df["vx"] * pixel_size
        #         # elif x_unit == "m":
        #         #     vx = df["vx"] * pixel_size / 1000

        #         # if y_unit == "px":
        #         #     vy = df["y"]
        #         # elif y_unit == "mm":
        #         #     vy = df["vy"] * pixel_size
        #         # elif y_unit == "m":
        #         #     vy = df["vy"] * pixel_size / 1000

        #         # if y_unit == "px":
        #         #     d = df["diameter_mean"]
        #         # elif y_unit == "mm":
        #         #     d = df["diameter_mean"] * pixel_size
        #         # elif y_unit == "m":
        #         #     d = df["diameter_mean"] * pixel_size / 1000

        #         # data_dict["x"].append(x)
        #         # data_dict["y"].append(y)
        #         # data_dict["vx"].append(vx)
        #         # data_dict["vy"].append(vy)
        #         # data_dict["label"].append(lbl)
        #         # data_dict["diameter_mean"].append(d)

        # return data_dict

    def Visualize_mass(
        self,
        frames: int | list | np.ndarray = None,
        labels: int | list | np.ndarray = None,
        path_save=None,
        do_save: str | Path = False,
        x_unit: str = "time",
        y_unit: str = "kg",
        do_plot_error: bool = False,
        do_smooth: bool = False,
    ):

        if not "frame" in self.dataframe.keys():
            msg = "frame not in dataframe"
            raise KeyError(msg)
        if isinstance(frames, int):
            frames = [frames]
        if frames is None:
            frames = [self.dataframe["frame"].unique()]
        if not isinstance(frames, (int, np.ndarray, list)):
            msg = "frames variable must be int, list of ndarray of int"
            raise TypeError(msg)

        if not "label" in self.dataframe.keys():
            msg = "label not in dataframe"
            raise KeyError(msg)
        if isinstance(labels, int):
            labels = [labels]
        if labels is None:
            labels = [self.dataframe["label"].unique()]
        if not isinstance(labels, (int, np.ndarray, list)):
            msg = "labels variable must be int, list of ndarray of int"
            raise TypeError(msg)

        mask_frames = self.dataframe["frame"].isin(*frames)
        mask_labels = self.dataframe["label"].isin(labels)
        mask = mask_frames & mask_labels

        sub_df = self.dataframe.loc[
            mask,
            [
                "frame",
                "time",
                "label",
                "diameter",
                "mass",
                "mass_mean",
                "mass_min",
                "mass_max",
            ],
        ].copy()

        print(sub_df[sub_df["frame"] == 7005])

        fig, ax = plt.subplots()

        for label_id, group in sub_df.groupby("label"):
            x_converted = (
                group["time"]
                if x_unit == "time"
                else group["frame"]
                if x_unit == "frame"
                else None
            )

            y_converted = (
                group["mass"]
                if y_unit == "kg"
                else group["mass"] * 1e3
                if y_unit == "g"
                else None
            )

            if do_smooth:
                k = 3
                kernel = np.ones((k,))
                y_converted = scipy.signal.convolve(
                    y_converted, kernel, mode="same", method="auto"
                )

            ax.plot(
                x_converted,
                y_converted,
                label=f"label {int(label_id)}",
            )

            y_converted = (
                group["mass_mean"]
                if y_unit == "kg"
                else group["mass_mean"] * 1e3
                if y_unit == "g"
                else None
            )

            if do_smooth:
                k = 3
                kernel = np.ones((k,))
                y_converted = scipy.signal.convolve(
                    y_converted, kernel, mode="same", method="auto"
                )

            ax.plot(
                x_converted,
                y_converted,
                label=f"mass mean for label {int(label_id)}",
            )

            if do_plot_error:
                y_converted_min = (
                    group["mass_min"]
                    if y_unit == "kg m/s"
                    else group["mass_min"] * 1e3
                    if y_unit == "kg mm/s"
                    else None
                )
                if do_smooth:
                    k = 3
                    kernel = np.ones((k,))
                    y_converted_min = scipy.signal.convolve(
                        y_converted_min, kernel, mode="same", method="auto"
                    )

                y_converted_max = (
                    group["mass_max"]
                    if y_unit == "kg"
                    else group["mass_max"] * 1e3
                    if y_unit == "g"
                    else None
                )
                if do_smooth:
                    k = 3
                    kernel = np.ones((k,))
                    y_converted_max = scipy.signal.convolve(
                        y_converted_max, kernel, mode="same", method="auto"
                    )

                ax.fill_between(
                    x_converted,
                    y1=y_converted_min,
                    y2=y_converted_max,
                    alpha=0.2,
                )

        print(max(y_converted))
        x_ticks = ax.get_xticks()[1:-1]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(
            [f"{x_tick:.2f}" for x_tick in x_ticks], fontsize=18
        )  # [s, frame]
        if x_unit == "time":
            ax.set_xlabel("Time $[s]$", fontsize=20)
        elif x_unit == "frame":
            ax.set_xlabel("Frame $[#]$", fontsize=20)

        y_ticks = ax.get_yticks()[1:-1]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(
            [f"{y_tick:.2e}" for y_tick in y_ticks], fontsize=18
        )  # [m/s, mm/s]
        if y_unit == "kg":
            ax.set_ylabel("Mass $\\left[kg\\right]$", fontsize=20)
        if y_unit == "g":
            ax.set_ylabel("Mass $\\left[g\\right]$", fontsize=20)

        plt.subplots_adjust(0.12, 0.1, 0.96, 0.96, 0.0, 0.0)
        plt.legend(fontsize=12)
        if do_save:
            name = Path(path_save) / Path(f"Visualize_mass_of labels_{labels}" + ".png")
            plt.savefig(name, dpi=120)
        else:
            plt.show()

    def Visualize_velocity(
        self,
        dataframe: pd.DataFrame = None,
        velocity: str | list | Path = None,
        frames: int | list | np.ndarray = None,
        labels: int | list | np.ndarray = None,
        time_interval: float = None,
        pixel_size: float = None,
        path_save: str | list = None,
        do_save: str | Path = False,
        x_unit: str = "time",
        y_unit: str = "m/s",
        min_decimal_x = None,
        min_decimal_y = None,
        use_subpixel: bool = False,
        do_plot_error: bool = False,
        do_smooth: bool = False,
        mode: str = "together",
    ):

        vel_altitude = []
        if velocity is not None:
            if not isinstance(velocity, (list, str, Path)):
                msg = "dataframe must be list or pd.DataFrame type"
                raise TypeError(msg)
            if not isinstance(velocity, list):
                velocity = [velocity]
        else:
            velocity = [None] * len(dataframe)

        if len(dataframe) > 10:
            colors = cm.get_cmap("tab20")
        else:
            colors = cm.get_cmap("tab10")

        result = []

        for run, (data, vel) in enumerate(zip(dataframe, velocity)):
            mask_keys = [
                "frame",
                "time",
                "label",
                "dt",
                "diameter_mean",
                "x",
                "y",  # "x_subpixel", "y_subpixel",
                "velocity",
                "vx",
                "vy",
            ]

            if isinstance(frames, int):
                frames = [frames]
            if frames is None:
                frames = data["frame"].unique()
            if not isinstance(frames, (int, np.ndarray, list)):
                msg = "'frames' must be int, list or ndarray of int"
                raise TypeError(msg)

            if labels == "all" or labels is None:
                labels = sorted(data["label"].unique())

            mask_frames = data["frame"].isin(frames)
            mask_labels = data["label"].isin(labels)
            mask = mask_frames & mask_labels
            sub_df = data.loc[mask, mask_keys].copy()
            frames = sub_df["frame"].unique()

            if vel is not None:
                keys = [
                    "frame",
                    "timestamp",
                    "voltage",
                    "velocity",
                ]

                mask = vel["frame"].isin(frames)

                vel = vel.loc[mask, keys].copy()

                # compute velocity gradient
                self.nu_air = 1.56e-5  # m2/s
                if "friction" not in vel:
                    vel["friction"] = 0.0564 * vel["velocity"] ** (7 / 8)  # m/s
                vel_grad = vel["friction"].mean() ** 2 / self.nu_air  # /s

            dt = sub_df["dt"].unique()

            print(x_unit)

            unit_factor_x = {
                "frames": 1,
                "time": time_interval,
                "fric_velocity": 1.0,
                "flow_velocity": 1.0,
                "reynolds": 1.0,
            }[x_unit]

            unit_label_x = {
                "frames": "Frames",
                "time": "Time $[s]$",
                "fric_velocity": "Friction velocity [m/s]",
                "flow_velocity": "Flow velocity [m/s]",
                "reynolds": "Reynold number",
            }[x_unit]

            min_decimals_x = {
                "frames": 0,
                "time": 3,
                "fric_velocity": 3,
                "flow_velocity": 3,
                "reynolds": 3,
            }[x_unit]

            unit_factor_y = {
                "mm/s": pixel_size,
                "m/s": pixel_size / 1000,
            }[y_unit]

            unit_label_y = {
                "mm/s": "Velocity $[mm/s]$",
                "m/s": "Velocity $[m/s]$",
            }[y_unit]

            min_decimals_y = {
                "mm/s": 0,
                "m/s": 2,
            }[y_unit]

            unit_factor_y_2 = {
                "mm/s": 1e3,
                "m/s": 1.0,
            }[y_unit]

            df = sub_df.sort_values(by=["label", "frame"])
            dt = sub_df["dt"].unique()

            if use_subpixel:
                print("USE SUBPIXEL")
                df["dx"] = (
                    df.groupby("label")["x_subpixel"].diff().fillna(0.0)
                )  # .where(lambda x: x.abs() >= 1.0, 0.0)
                df["dy"] = (
                    df.groupby("label")["y_subpixel"].diff().fillna(0.0)
                )  # .where(lambda x: x.abs() >= 1.0, 0.0)
            else:
                df["dx"] = (
                    df.groupby("label")["x"].diff().fillna(0.0)
                )  # .where(lambda x: x.abs() >= 1.0, 0.0)
                df["dy"] = (
                    df.groupby("label")["y"].diff().fillna(0.0)
                )  # .where(lambda x: x.abs() >= 1.0, 0.0)

            df["vx"] = df.groupby("label")["dx"].transform(lambda x: x / dt)
            df["vy"] = df.groupby("label")["dy"].transform(lambda x: x / dt)

            df["disp"] = np.sqrt(
                df.groupby("label")["dx"].transform(lambda x: x**2)
                + df.groupby("label")["dy"].transform(lambda x: x**2)
            )
            df["velocity"] = df.groupby("label")["disp"].transform(lambda x: x / dt)

            # diameter = sub_df.groupby("label")["diameter_mean"].unique().values
            df["diameter"] = df["label"].map(
                sub_df.groupby("label")["diameter_mean"].unique()
            )

            # compute velocity at d_p / 2
            if vel is not None:
                df["vel_altitude"] = df.groupby("label")["diameter_mean"].transform(
                    lambda x: ((x / 2) * pixel_size) * vel_grad
                )  # m/s

            # if len(df["velocity"]) > 3:
            #     func_base = FitFunction()._get_fitting_function()["exp"]
            #     for _, group in df.groupby("label"):
            #         popt, _, fit_func, r2 = FitFunction()._fit_curve(group["frame"], group["velocity"], func_base=func_base)

            grouped_label = df.groupby("label")

            data_dict = {
                "curves": True,
                "x": [group["frame"].to_numpy() for _, group in grouped_label],
                "y": [group["velocity"].to_numpy() for _, group in grouped_label],
                "label": [group["label"].to_numpy() for _, group in grouped_label],
                # "fit": [fit_func],
                "x_log": False,
                "x_unit": unit_factor_x,
                "y_unit": unit_factor_y,
                "x_label": unit_label_x,
                "y_label": unit_label_y,
                "min_decimals_x": min_decimals_x,
                "min_decimals_y": min_decimals_y,
                "x_ticks_sci": False,
                "y_ticks_sci": True,
                "velocity_f": [group["vel_altitude"].to_numpy() for _, group in grouped_label] if vel is not None else None,
                "y_unit_2": unit_factor_y_2,
            }

            for lbl, group in grouped_label:
                output_file = f"/home/abad-ale/Documents/Images_analysis/GUI/Velocity_run_{4}_label_{lbl}.csv"
                output_file = Path(output_file)
                df = pd.DataFrame(
                    {
                        "frame": group["frame"].to_numpy(),
                        "velocity_p [px/s]": group["velocity"].to_numpy(),
                        "velocity_p [mm/s]": group["velocity"].to_numpy() * pixel_size,
                        "diameter_p [px]": group["diameter"].to_numpy(),
                        "diameter_p [mm]": group["diameter"].to_numpy() * pixel_size,
                        "velocity_f [m/s]": group["vel_altitude"] if vel is not None else None,
                    }
                )

                table = pa.Table.from_pandas(df)
                pq.write_table(table, output_file)

                table = pq.read_table(output_file)
                df = table.to_pandas()
                df.to_csv(output_file, index=False, encoding="utf-8")

            result.append(data_dict)

        return result

    def Visualize_acceleration(
        self,
        dataframe: pd.DataFrame = None,
        velocity: str | list | Path = None,
        frames: int | list | np.ndarray = None,
        labels: int | list | np.ndarray = None,
        time_interval: float = None,
        pixel_size: float = None,
        path_save: str | list = None,
        do_save: str | Path = False,
        x_unit: str = "time",
        y_unit: str = "m/s",
        min_decimal_x = None,
        min_decimal_y = None,
        use_subpixel: bool = False,
        do_plot_error: bool = False,
        do_smooth: bool = False,
        mode: str = "together",
    ):

        if len(dataframe) > 10:
            colors = cm.get_cmap("tab20")
        else:
            colors = cm.get_cmap("tab10")

        result = []

        for run, data in enumerate(dataframe):
            mask_keys = [
                "frame",
                "time",
                "label",
                "dt",
                "diameter_mean",
                "x",
                "y",  # "x_subpixel", "y_subpixel",
                "velocity",
                "vx",
                "vy",
            ]

            if isinstance(frames, int):
                frames = [frames]
            if frames is None:
                frames = data["frame"].unique()
            if not isinstance(frames, (int, np.ndarray, list)):
                msg = "'frames' must be int, list or ndarray of int"
                raise TypeError(msg)

            if labels == "all" or labels is None:
                labels = sorted(data["label"].unique())

            mask_frames = data["frame"].isin(frames)
            mask_labels = data["label"].isin(labels)
            mask = mask_frames & mask_labels
            sub_df = data.loc[mask, mask_keys].copy()
            frames = sub_df["frame"].unique()

            dt = sub_df["dt"].unique()

            unit_factor_x = {
                "frames": 1,
                "time": time_interval,
            }[x_unit]

            unit_label_x = {
                "frames": "Frames",
                "time": "Time $[s]$",
            }[x_unit]

            min_decimals_x = {
                "frames": 0,
                "time": 3,
            }[x_unit]

            unit_factor_y = {
                "mm/s2": pixel_size,
                "m/s2": pixel_size / 1000,
            }[y_unit]

            unit_label_y = {
                "mm/s2": "Acceleration $[mm/s]$",
                "m/s2": "Acceleration $[m/s]$",
            }[y_unit]

            min_decimals_y = {
                "mm/s2": 0,
                "m/s2": 2,
            }[y_unit]

            df = sub_df.sort_values(by=["label", "frame"])
            dt = sub_df["dt"].unique()

            if use_subpixel:
                print("USE SUBPIXEL")
                df["dx"] = (
                    df.groupby("label")["x_subpixel"].diff().fillna(0.0)
                )  # .where(lambda x: x.abs() >= 1.0, 0.0)
                df["dy"] = (
                    df.groupby("label")["y_subpixel"].diff().fillna(0.0)
                )  # .where(lambda x: x.abs() >= 1.0, 0.0)
            else:
                df["dx"] = (
                    df.groupby("label")["x"].diff().fillna(0.0)
                )  # .where(lambda x: x.abs() >= 1.0, 0.0)
                df["dy"] = (
                    df.groupby("label")["y"].diff().fillna(0.0)
                )  # .where(lambda x: x.abs() >= 1.0, 0.0)

            df["vx"] = df.groupby("label")["dx"].transform(lambda x: x / dt)
            df["vy"] = df.groupby("label")["dy"].transform(lambda x: x / dt)

            df["disp"] = np.sqrt(
                df.groupby("label")["dx"].transform(lambda x: x**2)
                + df.groupby("label")["dy"].transform(lambda x: x**2)
            )
            df["velocity"] = df.groupby("label")["disp"].transform(lambda x: x / dt)

            # diameter = sub_df.groupby("label")["diameter_mean"].unique().values
            df["diameter"] = df["label"].map(
                sub_df.groupby("label")["diameter_mean"].unique()
            )

            df["ax"] = df.groupby("label")["vx"].transform(lambda x: x / dt)
            df["ay"] = df.groupby("label")["vy"].transform(lambda x: x / dt)
            df["acceleration"] = df.groupby("label")["velocity"].transform(
                lambda x: x / dt
            )

            # if len(df["acceleration"]) > 3:
            #     func_base = FitFunction()._get_fitting_function()["exp"]
            #     for _, group in df.groupby("label"):
            #         popt, _, fit_func, r2 = FitFunction()._fit_curve(group["frame"], group["acceleration"], func_base=func_base)

            grouped_label = df.groupby("label")

            data_dict = {
                "curves": True,
                "x": [group["frame"].to_numpy() for _, group in grouped_label],
                "y": [group["acceleration"].to_numpy() for _, group in grouped_label],
                "label": [group["label"].to_numpy() for _, group in grouped_label],
                # "fit": [fit_func],
                "x_log": False,
                "x_unit": unit_factor_x,
                "y_unit": unit_factor_y,
                "x_label": unit_label_x,
                "y_label": unit_label_y,
                "min_decimals_x": min_decimals_x,
                "min_decimals_y": min_decimals_y,
                "x_ticks_sci": False,
                "y_ticks_sci": True,
            }

            for lbl, group in grouped_label:
                output_file = f"/home/abad-ale/Documents/Images_analysis/GUI/Acceleration_run_{4}_label_{lbl}.csv"
                output_file = Path(output_file)
                df = pd.DataFrame(
                    {
                        "frame": group["frame"].to_numpy(),
                        "acceleration_p [px/s]": group["acceleration"].to_numpy(),
                        "acceleration_p [mm/s]": group["acceleration"].to_numpy() * pixel_size,
                        "diameter_p [px]": group["diameter"].to_numpy(),
                        "diameter_p [mm]": group["diameter"].to_numpy() * pixel_size,
                    }
                )

                table = pa.Table.from_pandas(df)
                pq.write_table(table, output_file)

                table = pq.read_table(output_file)
                df = table.to_pandas()
                df.to_csv(output_file, index=False, encoding="utf-8")

            result.append(data_dict)

        return result

    def Visualize_momentum(
        self,
        dataframe: pd.DataFrame = None,
        velocity: str | list | Path = None,
        frames: int | list | np.ndarray = None,
        labels: int | list | np.ndarray = None,
        time_interval: float = None,
        pixel_size: float = None,
        path_save: str | list = None,
        do_save: str | Path = False,
        x_unit: str = "time",
        y_unit: str = "m/s",
        min_decimal_x = None,
        min_decimal_y = None,
        use_subpixel: bool = False,
        do_plot_error: bool = False,
        do_smooth: bool = False,
        mode: str = "together",
    ):

        if len(dataframe) > 10:
            colors = cm.get_cmap("tab20")
        else:
            colors = cm.get_cmap("tab10")

        result = []

        for run, data in enumerate(dataframe):
            mask_keys = [
                "frame",
                "time",
                "label",
                "dt",
                "diameter_mean",
                "x",
                "y",  # "x_subpixel", "y_subpixel",
                "velocity",
                "vx",
                "vy",
                "mass_mean",
            ]

            if isinstance(frames, int):
                frames = [frames]
            if frames is None:
                frames = data["frame"].unique()
            if not isinstance(frames, (int, np.ndarray, list)):
                msg = "'frames' must be int, list or ndarray of int"
                raise TypeError(msg)

            if labels == "all" or labels is None:
                labels = sorted(data["label"].unique())

            mask_frames = data["frame"].isin(frames)
            mask_labels = data["label"].isin(labels)
            mask = mask_frames & mask_labels
            sub_df = data.loc[mask, mask_keys].copy()
            frames = sub_df["frame"].unique()

            dt = sub_df["dt"].unique()

            unit_factor_x = {
                "frames": 1,
                "time": time_interval,
            }[x_unit]

            unit_label_x = {
                "frames": "Frames",
                "time": "Time $[s]$",
            }[x_unit]

            min_decimals_x = {
                "frames": 0,
                "time": 3,
            }[x_unit]

            unit_factor_y = {
                "kg.mm/s": pixel_size,
                "kg.m/s": pixel_size / 1000,
            }[y_unit]

            unit_label_y = {
                "kg.mm/s": "Momentum $[kg \, mm/s]$",
                "kg.m/s": "Momentum $[kg \, m/s]$",
            }[y_unit]

            min_decimals_y = {
                "kg.mm/s": 0,
                "kg.m/s": 2,
            }[y_unit]

            df = sub_df.sort_values(by=["label", "frame"])
            dt = sub_df["dt"].unique()

            if use_subpixel:
                print("USE SUBPIXEL")
                df["dx"] = (
                    df.groupby("label")["x_subpixel"].diff().fillna(0.0)
                )  # .where(lambda x: x.abs() >= 1.0, 0.0)
                df["dy"] = (
                    df.groupby("label")["y_subpixel"].diff().fillna(0.0)
                )  # .where(lambda x: x.abs() >= 1.0, 0.0)
            else:
                df["dx"] = (
                    df.groupby("label")["x"].diff().fillna(0.0)
                )  # .where(lambda x: x.abs() >= 1.0, 0.0)
                df["dy"] = (
                    df.groupby("label")["y"].diff().fillna(0.0)
                )  # .where(lambda x: x.abs() >= 1.0, 0.0)

            df["vx"] = df.groupby("label")["dx"].transform(lambda x: x / dt)
            df["vy"] = df.groupby("label")["dy"].transform(lambda x: x / dt)

            df["disp"] = np.sqrt(
                df.groupby("label")["dx"].transform(lambda x: x**2)
                + df.groupby("label")["dy"].transform(lambda x: x**2)
            )
            df["velocity"] = df.groupby("label")["disp"].transform(lambda x: x / dt)

            # diameter = sub_df.groupby("label")["diameter_mean"].unique().values
            df["diameter"] = df["label"].map(
                sub_df.groupby("label")["diameter_mean"].unique()
            )

            df["momentum"] = (
                df.groupby("label")["mass_mean"].transform(lambda x: x * (pixel_size / 1000)**3)
                * df.groupby("label")["velocity"].transform(lambda x: x * pixel_size / 1000)
            )

            # if len(df["momentum"]) > 3:
            #     func_base = FitFunction()._get_fitting_function()["exp"]
            #     for _, group in df.groupby("label"):
            #         popt, _, fit_func, r2 = FitFunction()._fit_curve(group["frame"], group["momentum"], func_base=func_base)

            grouped_label = df.groupby("label")

            data_dict = {
                "curves": True,
                "x": [group["frame"].to_numpy() for _, group in grouped_label],
                "y": [group["momentum"].to_numpy() for _, group in grouped_label],
                "label": [group["label"].to_numpy() for _, group in grouped_label],
                # "fit": [fit_func],
                "x_log": False,
                "x_unit": unit_factor_x,
                "y_unit": unit_factor_y,
                "x_label": unit_label_x,
                "y_label": unit_label_y,
                "min_decimals_x": min_decimals_x,
                "min_decimals_y": min_decimals_y,
                "x_ticks_sci": False,
                "y_ticks_sci": True,
            }

            for lbl, group in grouped_label:
                output_file = f"/home/abad-ale/Documents/Images_analysis/GUI/Momentum_run_{4}_label_{lbl}.csv"
                output_file = Path(output_file)
                df = pd.DataFrame(
                    {
                        "frame": group["frame"].to_numpy(),
                        "momentum_p [px/s]": group["momentum"].to_numpy(),
                        "momentum_p [mm/s]": group["momentum"].to_numpy() * pixel_size,
                        "diameter_p [px]": group["diameter"].to_numpy(),
                        "diameter_p [mm]": group["diameter"].to_numpy() * pixel_size,
                    }
                )

                table = pa.Table.from_pandas(df)
                pq.write_table(table, output_file)

                table = pq.read_table(output_file)
                df = table.to_pandas()
                df.to_csv(output_file, index=False, encoding="utf-8")

            result.append(data_dict)

        return result

    def Visualize_kinetic_energy(
        self,
        dataframe: pd.DataFrame = None,
        velocity: str | list | Path = None,
        frames: int | list | np.ndarray = None,
        labels: int | list | np.ndarray = None,
        time_interval: float = None,
        pixel_size: float = None,
        path_save: str | list = None,
        do_save: str | Path = False,
        x_unit: str = "time",
        y_unit: str = "J",
        min_decimal_x = None,
        min_decimal_y = None,
        use_subpixel: bool = False,
        do_plot_error: bool = False,
        do_smooth: bool = False,
        mode: str = "together",
    ):

        if len(dataframe) > 10:
            colors = cm.get_cmap("tab20")
        else:
            colors = cm.get_cmap("tab10")

        result = []

        for run, data in enumerate(dataframe):
            mask_keys = [
                "frame",
                "time",
                "label",
                "dt",
                "diameter_mean",
                "x",
                "y",  # "x_subpixel", "y_subpixel",
                "velocity",
                "vx",
                "vy",
                "mass_mean",
            ]

            if isinstance(frames, int):
                frames = [frames]
            if frames is None:
                frames = data["frame"].unique()
            if not isinstance(frames, (int, np.ndarray, list)):
                msg = "'frames' must be int, list or ndarray of int"
                raise TypeError(msg)

            if labels == "all" or labels is None:
                labels = sorted(data["label"].unique())

            mask_frames = data["frame"].isin(frames)
            mask_labels = data["label"].isin(labels)
            mask = mask_frames & mask_labels
            sub_df = data.loc[mask, mask_keys].copy()
            frames = sub_df["frame"].unique()

            dt = sub_df["dt"].unique()

            unit_factor_x = {
                "frames": 1,
                "time": time_interval,
            }[x_unit]

            unit_label_x = {
                "frames": "Frames",
                "time": "Time $[s]$",
            }[x_unit]

            min_decimals_x = {
                "frames": 0,
                "time": 3,
            }[x_unit]

            unit_factor_y = {
                "J": 1.0,
                "uJ": 1e-6,
                "nJ": 1e-9,
                "pJ": 1e-12,
            }[y_unit]

            unit_label_y = {
                "J": "Kinetic $[J]$",
                "uJ": "Kinetic $[\mu J]$",
                "nJ": "Kinetic $[nJ]$",
                "pJ": "Kinetic $[pJ]$"
            }[y_unit]

            min_decimals_y = {
                "J": 3,
                "uJ": 0,
                "nJ": 2,
                "pJ": 2,
            }[y_unit]

            df = sub_df.sort_values(by=["label", "frame"])
            dt = sub_df["dt"].unique()

            if use_subpixel:
                print("USE SUBPIXEL")
                df["dx"] = (
                    df.groupby("label")["x_subpixel"].diff().fillna(0.0)
                )  # .where(lambda x: x.abs() >= 1.0, 0.0)
                df["dy"] = (
                    df.groupby("label")["y_subpixel"].diff().fillna(0.0)
                )  # .where(lambda x: x.abs() >= 1.0, 0.0)
            else:
                df["dx"] = (
                    df.groupby("label")["x"].diff().fillna(0.0)
                )  # .where(lambda x: x.abs() >= 1.0, 0.0)
                df["dy"] = (
                    df.groupby("label")["y"].diff().fillna(0.0)
                )  # .where(lambda x: x.abs() >= 1.0, 0.0)

            df["vx"] = df.groupby("label")["dx"].transform(lambda x: x / dt)
            df["vy"] = df.groupby("label")["dy"].transform(lambda x: x / dt)

            df["disp"] = np.sqrt(
                df.groupby("label")["dx"].transform(lambda x: x**2)
                + df.groupby("label")["dy"].transform(lambda x: x**2)
            )
            df["velocity"] = df.groupby("label")["disp"].transform(lambda x: x / dt)

            # diameter = sub_df.groupby("label")["diameter_mean"].unique().values
            df["diameter"] = df["label"].map(
                sub_df.groupby("label")["diameter_mean"].unique()
            )

            df["kinetic"] = (
                1 / 2
                * df.groupby("label")["mass_mean"].transform(lambda x: x * (pixel_size / 1000)**3)
                * df.groupby("label")["velocity"].transform(lambda x: (x * pixel_size / 1000)**2)
            )  # J

            # if len(df["kinetic"]) > 3:
            #     func_base = FitFunction()._get_fitting_function()["exp"]
            #     for _, group in df.groupby("label"):
            #         popt, _, fit_func, r2 = FitFunction()._fit_curve(group["frame"], group["kinetic"], func_base=func_base)

            grouped_label = df.groupby("label")

            data_dict = {
                "curves": True,
                "x": [group["frame"].to_numpy() for _, group in grouped_label],
                "y": [group["kinetic"].to_numpy() for _, group in grouped_label],
                "label": [group["label"].to_numpy() for _, group in grouped_label],
                # "fit": [fit_func],
                "x_log": False,
                "x_unit": unit_factor_x,
                "y_unit": unit_factor_y,
                "x_label": unit_label_x,
                "y_label": unit_label_y,
                "min_decimals_x": min_decimals_x,
                "min_decimals_y": min_decimals_y,
                "x_ticks_sci": False,
                "y_ticks_sci": True,
            }

            for lbl, group in grouped_label:
                output_file = f"/home/abad-ale/Documents/Images_analysis/GUI/Kinetic_run_{4}_label_{lbl}.csv"
                output_file = Path(output_file)
                df = pd.DataFrame(
                    {
                        "frame": group["frame"].to_numpy(),
                        "kinetic_p [px/s]": group["kinetic"].to_numpy(),
                        "kinetic_p [mm/s]": group["kinetic"].to_numpy() * pixel_size,
                        "diameter_p [px]": group["diameter"].to_numpy(),
                        "diameter_p [mm]": group["diameter"].to_numpy() * pixel_size,
                    }
                )

                table = pa.Table.from_pandas(df)
                pq.write_table(table, output_file)

                table = pq.read_table(output_file)
                df = table.to_pandas()
                df.to_csv(output_file, index=False, encoding="utf-8")

            result.append(data_dict)

        return result

    def Visualize_cluster(
        self,
        pixel_size: float = 1.0,
        path_dataframe: str | Path = None,
        frames: int | list = None,
        path_save: str | Path = None,
        do_save: bool = False,
        units: str = "px",
        rotate: int = None,
    ):

        # check if columns exists
        if not isinstance(path_dataframe, (list, str, Path)):
            msg = "dataframe must be list or pd.DataFrame type"
            raise TypeError(msg)
        if not isinstance(path_dataframe, list):
            path_dataframe = [path_dataframe]

        for i, path_data in enumerate(path_dataframe):
            print(path_data)

            dataframe = self._load_dataframe(path_data)

            keys = [
                "frame",
                "main_path",
                "name",
                "time",
                "label",
                "x",
                "y",
                "diameter_mean",
                "diameter_std",
                "cluster_id",
                "cluster_label",
            ]
            self._check_keys(dataframe, keys)

            if isinstance(frames, int):
                frames = [frames]
            if frames is None:
                frames = [dataframe["frame"].unique()]
            if not isinstance(frames, (int, np.ndarray, list)):
                msg = "frames variable must be int, list of ndarray of int"
                raise TypeError(msg)

            mask = dataframe["frame"].isin(frames)

            sub_df = dataframe.loc[
                mask,
                keys,
            ].copy()

            for i, (frame_id, group) in enumerate(sub_df.groupby("frame")):
                # if rotate in [90, 270]:
                #     _, ax = plt.subplots(figsize=(6, 8))
                # elif rotate == 180:
                #     _, ax = plt.subplots(figsize=(8, 6))

                _, ax = plt.subplots(figsize=(8, 6))

                path_image = Path(sub_df["main_path"].unique()[0]) / Path(
                    sub_df["name"].unique()[0]
                )
                img = Image.open(path_image).convert("L")
                if rotate is not None:
                    if rotate == 90:
                        img = img.transpose(Image.ROTATE_90)
                    elif rotate == 180:
                        img = img.transpose(Image.ROTATE_180)
                    elif rotate == 270:
                        img = img.transpose(Image.ROTATE_270)
                    coords = self._rotate_coords(
                        sub_df[["x", "y"]].to_numpy() / self.pixel_size,
                        np.array(img).shape[1],
                        np.array(img).shape[0],
                        rotate,
                    )
                    x_converted, y_converted = coords[:, 0], coords[:, 1]

                else:
                    x_converted, y_converted = sub_df["x"], sub_df["y"]

                img = np.array(img, dtype=np.uint8)

                ax.imshow(img, cmap="gray")

                number_of_clusters = len(sub_df[sub_df["cluster_id"] != -1])

                clusters = [c for c in sub_df["cluster_id"].unique() if c != -1]
                colors = {
                    c: cm.get_cmap("hsv", len(clusters))(i)
                    for i, c in enumerate(clusters)
                }

                for _, row in sub_df.iterrows():
                    if row["cluster_id"] != -1:
                        ax.scatter(
                            x_converted / self.pixel_size,  # [px]
                            y_converted / self.pixel_size,  # [px]
                            marker="o",
                            s=row["diameter_mean"] * 10 / self.pixel_size,  # [px]
                            alpha=0.7,
                            color=colors[row["cluster_id"]],
                            label=f"Cluster {row['cluster_id']}",
                        )

                if units == "milli":
                    x_ticks = ax.get_xticks()[1:-1]
                    y_ticks = ax.get_yticks()[1:-1]
                    ax.set_xticks(x_ticks)
                    ax.set_yticks(y_ticks)
                    ax.set_xticklabels(
                        [f"{x_tick * self.pixel_size:.2f}" for x_tick in x_ticks],
                        fontsize=16,
                    )
                    ax.set_yticklabels(
                        [f"{y_tick * self.pixel_size:.2f}" for y_tick in y_ticks],
                        fontsize=16,
                    )
                    ax.set_xlabel("x $[mm]$", fontsize=20)
                    ax.set_ylabel("y $[mm]$", fontsize=20)

                if units == "px":
                    x_ticks = ax.get_xticks()[1:-1]
                    y_ticks = ax.get_yticks()[1:-1]
                    ax.set_xticks(x_ticks)
                    ax.set_yticks(y_ticks)
                    ax.set_xticklabels(
                        [f"{x_tick:.0f}" for x_tick in x_ticks], fontsize=16
                    )
                    ax.set_yticklabels(
                        [f"{y_tick:.0f}" for y_tick in y_ticks], fontsize=16
                    )
                    ax.set_xlabel("x $[px]$", fontsize=20)
                    ax.set_ylabel("y $[px]$", fontsize=20)

                elif units == "meter":
                    x_ticks = ax.get_xticks()[1:-1]
                    y_ticks = ax.get_yticks()[1:-1]
                    ax.set_xticks(x_ticks)
                    ax.set_yticks(y_ticks)
                    ax.set_xticklabels(
                        [f"{x_tick * self.pixel_size:.2f}" for x_tick in x_ticks],
                        fontsize=16,
                    )
                    ax.set_yticklabels(
                        [f"{y_tick * self.pixel_size:.2f}" for y_tick in y_ticks],
                        fontsize=16,
                    )
                    ax.set_xlabel("x $[m]$", fontsize=20)
                    ax.set_ylabel("y $[m]$", fontsize=20)

            if len(sub_df[sub_df["cluster_id"] != -1]) <= 10:
                handles, labels = ax.get_legend_handles_labels()
                plt.legend(
                    dict(zip(labels, handles)).values(),
                    dict(zip(labels, handles)).keys(),
                )

            plt.subplots_adjust(0.13, 0.13, 1.0, 0.96, 0.0, 0.0)
            if do_save:
                name = Path(path_save) / Path(
                    f"Visualize_{number_of_clusters:d}_clusters_"
                    + PurePath(path_save).parts[-1]
                    + "_on_plate_"
                    + str(frames + 1)
                    + ".png"
                )
                plt.savefig(name, dpi=120)
            else:
                plt.show()

    def Visualize_fraction_aggregates(
        self,
        frames: int | list | np.ndarray = None,
        n_bins: int = 20,
        do_print_fractions: bool = False,
        path_save: str | Path = None,
        do_save: bool = False,
    ):

        # # if ("cluster_id", "cluster_label") not in self.dataframe.keys():
        # if not (("cluster_id", "cluster_label") in self.dataframe.keys()):
        #     msg = "cluster_id and/or cluster_label is not in dataframe"
        #     raise KeyError(msg)

        for frame in frames:
            sub_df = self.dataframe[self.dataframe["frame"] == frame]
            nb_particles = len(sub_df)

            nb_particles_aggregate = len(sub_df[sub_df["cluster_id"] != -1])
            fraction_agg = nb_particles_aggregate / nb_particles

            nb_particles_alone = len(sub_df[sub_df["cluster_id"] == -1])
            fraction_alone = nb_particles_alone / nb_particles

            if do_print_fractions:
                print(f"Fraction of isolated particles is {fraction_alone:.2f}")
                print(f"Fraction of aggregated particles is {fraction_agg:.2f}")
                print(f"Sum of both fraction is {fraction_alone + fraction_agg:.2f}")
                print()

            aggregates_id = sub_df[sub_df["cluster_id"] != -1]["cluster_id"].unique()
            print(f"Number of aggregates : {len(aggregates_id)}")

            nb_particle_per_cluster = np.array(
                [len(sub_df[sub_df["cluster_id"] == id]) for id in aggregates_id]
            )
            # print(nb_particle_per_cluster)
            # print(np.mean(nb_particle_per_cluster), nb_particle_per_cluster.std)

            fig, ax = plt.subplots()

            ax.plot(
                aggregates_id,
                nb_particle_per_cluster,
                label="Number of particles per aggregate",
            )

            ax.hlines(
                y=np.mean(nb_particle_per_cluster),
                xmin=np.min(aggregates_id),
                xmax=np.max(aggregates_id),
                color="tab:orange",
                label="mean",
            )

            ax.hlines(
                y=np.mean(nb_particle_per_cluster)
                + np.std(nb_particle_per_cluster) / 2,
                xmin=np.min(aggregates_id),
                xmax=np.max(aggregates_id),
                linestyles="--",
                color="tab:orange",
                label="mean + std / 2",
            )

            ax.hlines(
                y=np.mean(nb_particle_per_cluster)
                - np.std(nb_particle_per_cluster) / 2,
                xmin=np.min(aggregates_id),
                xmax=np.max(aggregates_id),
                linestyles="--",
                color="tab:orange",
                label="mean - std / 2",
            )

            # ax.hist(
            #     nb_particle_per_cluster,
            #     bins=n_bins,
            #     color="tab:blue",
            #     edgecolor="black",
            #     density=False,
            # )

            ax.set_xlabel("Aggregate number", fontsize=20)
            ax.set_ylabel("Number of particles per aggregate", fontsize=20)

            plt.legend(fontsize=12)
            plt.subplots_adjust(0.13, 0.13, 1.0, 0.96, 0.0, 0.0)
            if do_save:
                name = Path(path_save) / Path(
                    "Visualize_histogram_fraction_aggegates_"
                    + PurePath(path_save).parts[-1]
                    + "_on_plate_"
                    + str(frame + 1)
                    + ".png"
                )
                plt.savefig(name, dpi=120)
            else:
                plt.show()

    def Track_particles_position(
        self,
        dataframe: pd.DataFrame = None,
        velocity: pd.DataFrame = None,
        pixel_size: float = 1.0,
        time_interval: float =  None,
        frames: list | int = None,
        labels: list | int = None,
        x_unit: str = "mm",
        y_unit: str = "mm",
        path_save: str = None,
        do_save: bool = False,
        rotate: int = 0,
        crop: tuple = (1, 1),
    ):

        if isinstance(frames, int):
            frames = [frames]

        if path_save is None and do_save:
            msg = "Must specify a path to save picture"
            raise TypeError(msg)
        if path_save is not None and do_save is None:
            do_save = True

        results = []

        for data in dataframe:
            # ----- frames
            if frames is None:
                selected_frames = data["frame"].unique()
            elif isinstance(frames, int):
                selected_frames = [frames]
            elif isinstance(frames, (list, np.ndarray)):
                selected_frames = frames
            else:
                msg = "'frames' must be int, list or np.ndarray of int"
                raise TypeError(msg)

            # ----- labels
            if labels == "all" or labels is None:
                selected_labels = sorted(data["label"].unique())
            else:
                selected_labels = labels

            # ----- filtering
            mask = data["frame"].isin(selected_frames) & data["label"].isin(
                selected_labels
            )

            sub_df = data.loc[
                mask, ["frame", "main_path", "name", "label", "x", "y"]
            ].copy()

            if sub_df.empty:
                continue

            x_unit_factor = {"px": 1, "mm": pixel_size, "m": pixel_size / 1000}[x_unit]

            x_unit_label = {
                "px": "X [px]",
                "mm": "X [mm]",
                "m": "X [m]",
            }[x_unit]

            min_decimals_x = {
                "px": 0,
                "mm": 1,
                "m": 3,
            }[x_unit]

            y_unit_factor = {"px": 1, "mm": pixel_size, "m": pixel_size / 1000}[y_unit]

            y_unit_label = {
                "px": "Y [px]",
                "mm": "Y [mm]",
                "m": "Y [m]",
            }[y_unit]

            min_decimals_y = {
                "px": 0,
                "mm": 1,
                "m": 3,
            }[y_unit]

            # ----- load image
            name = Path(
                data["main_path"].iloc[0],
                data["name"].iloc[0],
                # Path("D:\\BISE_experiments\\Essai_7\\8000Hz\\4x10mm3\\3\\Images"),
            )
            img = self._load_image(name, invert=False, rotate_image=rotate)

            # ----- group data
            grouped = sub_df.groupby("label")

            data_dict = {
                "image": img,
                "x": [group["x"].to_numpy() for _, group in grouped],
                "y": [group["y"].to_numpy() for _, group in grouped],
                "label": [group["label"].unique() for _, group in grouped],
                "x_unit": x_unit_factor,
                "y_unit": y_unit_factor,
                "x_label": f"{x_unit_label}",
                "y_label": f"{y_unit_label}",
                "min_decimals_x": min_decimals_x,
                "min_decimals_y": min_decimals_y,
            }
            results.append(data_dict)

            # compute distance
            # distances = [
            #     np.sqrt(
            #         (group["x"].to_numpy()[0] - group["x"].to_numpy()[-1])**2 +
            #         (group["y"].to_numpy()[0] - group["y"].to_numpy()[-1])**2
            #         ) * pixel_size
            #     for _, group in grouped
            # ]
            # print(distances)

            # sub_df["dx"] = (
            #     sub_df.groupby("label")["x"].diff().fillna(0.0)
            # ) * pixel_size
            # sub_df["dy"] = (
            #     sub_df.groupby("label")["y"].diff().fillna(0.0)
            # ) * pixel_size

            # sub_df["disp"] = np.sqrt(
            #     sub_df.groupby("label")["dx"].transform(lambda x: x**2)
            #     + sub_df.groupby("label")["dy"].transform(lambda x: x**2)
            # )
            # print(sub_df["disp"].shape, sub_df["disp"])
            
        return results

    def Visualize_collisions(
        self,
        frames: int | list | np.ndarray,
        path_save: Path | str = None,
        do_save: bool = False,
        unit: str = "milli",
        img_before_after: bool = True,
    ):

        if isinstance(frames, int):
            if frames in self.dataframe["frame"].unique():
                frames = [frames]
            else:
                msg = "frames variable is not in available frames"
                raise ValueError(msg)
        if isinstance(frames, str) and frames == "all":
            if frames in self.dataframe["frame"].unique():
                frames = self.dataframe["frame"].unique()
            else:
                msg = "frames variable is not in available frames"
                raise ValueError(msg)
        # if self.dataframe["frame"].isin(frames).all():
        #     print(
        #         set(frames).symmetric_difference(set(self.dataframe["frame"].unique()))
        #     )
        #     msg = "Not all frames are in dataframe"
        #     raise ValueError(msg)

        if path_save is None and do_save:
            msg = "Must specify a path to save picture"
            raise TypeError(msg)
        if path_save is not None and do_save is None:
            do_save = True

        if not ((unit == "px") or (unit == "meter") or (unit == "milli")):
            msg = "units varible is not px or meter or milli"
            raise ValueError(msg)

        mask = self.dataframe["frame"].isin(*frames)

        sub_df = self.dataframe.loc[
            mask, ["frame", "name", "label", "x", "y", "diameter_mean", "collision"]
        ].copy()

        for frame_id, group in sub_df.groupby("frame"):
            fig, ax = plt.subplots(figsize=(8, 6))

            img_name = sub_df[sub_df["frame"] == frame_id]["name"].unique()[0]
            img = np.array(
                Image.open(
                    f"{sub_df['main_path'].unique()[0]}\\{sub_df['name'].unique()[0]}"
                ).convert("L"),
                dtype=np.uint8,
            )
            ax.imshow(img, cmap="gray")

            # if (
            #     img_before_after
            #     and (frame != np.min(self.dataframe["frame"].unique()))
            #     and (frame != np.max(self.dataframe["frame"].unique()))
            # ):
            #     img_name = self.dataframe[self.dataframe["frame"] == frame - 1][
            #         "name"
            #     ].unique()[0]
            #     img = np.array(Image.open(img_name).convert("L"), dtype=np.uint8)
            #     ax.imshow(img, cmap="gray", alpha=0.5)
            #     img_name = self.dataframe[self.dataframe["frame"] == frame + 1][
            #         "name"
            #     ].unique()[0]
            #     img = np.array(Image.open(img_name).convert("L"), dtype=np.uint8)
            #     ax.imshow(img, cmap="gray", alpha=0.5)

            number_of_collisions = len(group[group["collision"]]) // 2
            print("Number of collision", number_of_collisions)

            for _, row in group.iterrows():
                if row["collision"]:
                    x_converted = (
                        row["x"] / self.pixel_size
                        if unit == "milli"
                        else row["x"] / self.pixel_size / 1000.0
                        if unit == "meter"
                        else row["x"]
                        if unit == "px"
                        else None
                    )

                    y_converted = (
                        row["y"] / self.pixel_size
                        if unit == "milli"
                        else row["y"] / self.pixel_size / 1000.0
                        if unit == "meter"
                        else row["y"]
                        if unit == "px"
                        else None
                    )

                    diameter_converted = (
                        row["diameter_mean"] / self.pixel_size
                        if unit == "milli"
                        else row["diameter_mean"] / self.pixel_size / 1000.0
                        if unit == "meter"
                        else row["diameter_mean"]
                        if unit == "px"
                        else None
                    )

                    ax.scatter(
                        x_converted,  # [milli, meter, px]
                        y_converted,  # [milli, meter, px]
                        marker="*",
                        # s=diameter_converted, # [milli, meter, px]
                        alpha=0.7,
                        color="tab:green",
                    )

                    # previous_point = [
                    #     group['x'].shift(periods=1, fill_value=0).to_numpy()[0]/self.pixel_size,
                    #     group['y'].shift(periods=1, fill_value=0).to_numpy()[0]/self.pixel_size,
                    # ]
                    # current_point = [
                    #     group['x'].to_numpy()[0]/self.pixel_size,
                    #     group['y'].to_numpy()[0]/self.pixel_size,
                    # ]

                    # ax.plot(
                    #     [previous_point[0], current_point[0]], [previous_point[1], current_point[1]],
                    #     color='tab:gray', alpha=0.7
                    # )
                    # ax.text(
                    #     x=(previous_point[0] + current_point[0])/2, y=(previous_point[1] + current_point[1])/2,
                    #     s=str(i),
                    # )

            x_ticks = ax.get_xticks()[1:-1]
            y_ticks = ax.get_yticks()[1:-1]
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)

            if unit == "milli":
                ax.set_xticklabels(
                    [f"{x_tick * self.pixel_size:.2f}" for x_tick in x_ticks],
                    fontsize=18,
                )  # [mm]
                ax.set_yticklabels(
                    [f"{y_tick * self.pixel_size:.2f}" for y_tick in y_ticks],
                    fontsize=18,
                )  # [mm]
                ax.set_xlabel("x $[mm]$", fontsize=20)
                ax.set_ylabel("y $[mm]$", fontsize=20)

            elif unit == "meter":
                ax.set_xticklabels(
                    [f"{x_tick * self.pixel_size / 1000.0:.2f}" for x_tick in x_ticks],
                    fontsize=18,
                )  # [m]
                ax.set_yticklabels(
                    [f"{y_tick * self.pixel_size / 1000.0:.2f}" for y_tick in y_ticks],
                    fontsize=18,
                )  # [m]
                ax.set_xlabel("x $[mm]$", fontsize=20)
                ax.set_ylabel("y $[mm]$", fontsize=20)

            elif unit == "px":
                ax.set_xticklabels(
                    [f"{x_tick:.2f}" for x_tick in x_ticks], fontsize=18
                )  # [px]
                ax.set_yticklabels(
                    [f"{y_tick:.2f}" for y_tick in y_ticks], fontsize=18
                )  # [px]
                ax.set_xlabel("x $[px]$", fontsize=20)
                ax.set_ylabel("y $[px]$", fontsize=20)

            if number_of_collisions > 0:
                plt.legend(fontsize=12)
            plt.subplots_adjust(0.08, 0.08, 0.96, 0.96, 0.0, 0.0)

            if do_save:
                name = Path(path_save) / Path(
                    f"Visualize_{number_of_collisions:d}_collisions_"
                    + PurePath(path_save).parts[-1]
                    + "_on_plate_"
                    + str(frame_id + 1)
                    + ".png"
                )
                plt.savefig(name, dpi=120)
            else:
                plt.show()

    def Visualize_collisions_over_time(
        self,
        frames: int | list | np.ndarray = None,
        labels: int | list | np.ndarray = None,
        path_save: Path | str = None,
        do_save: bool = False,
        x_unit: str = "s",
        occurences_mode: str = "number",
    ):

        # check if columns exists
        if not "frame" in self.dataframe.columns:
            msg = "frame not in dataframe"
            raise KeyError(msg)
        if not "time" in self.dataframe.columns:
            msg = "time not in dataframe"
            raise KeyError(msg)
        if not "label" in self.dataframe.columns:
            msg = "label not in dataframe"
            raise KeyError(msg)
        if not "collision" in self.dataframe.columns:
            msg = "collision not in dataframe"
            raise KeyError(msg)

        # format inputs
        if frames is None:
            frames = self.dataframe["frame"].unique()
        if isinstance(frames, int):
            frames = [frames]
        if not isinstance(frames, np.ndarray):
            frames = frames.tolist()
        if not isinstance(frames, list):
            if len(frames) == 1 and isinstance(frames[0], list, np.ndarray):
                frames = list(frames[0])
        else:
            msg = "frames variable must be int, list or ndarray or int"
            raise TypeError(msg)

        if labels is None:
            labels = self.dataframe["label"].unique()
        if isinstance(labels, int):
            labels = [labels]
        if not isinstance(labels, np.ndarray):
            labels = labels.tolist()
        if not isinstance(labels, list):
            if len(labels) == 1 and isinstance(labels[0], list, np.ndarray):
                labels = list(labels[0])
        else:
            msg = "labels variable must be int, list or ndarray or int"
            raise TypeError(msg)

        if path_save is None and do_save:
            msg = "Must specify a path to save figure"
            raise TypeError(msg)
        if path_save is not None and do_save is None:
            do_save = True

        if x_unit not in ["frame", "s", "ms"]:
            msg = "units variable is not frame or s or ms"
            raise ValueError(msg)
        if occurences_mode not in ["number", "fraction", "percent"]:
            msg = "units variable is not frame or s or ms"
            raise ValueError(msg)

        mask_frames = self.dataframe["frame"].isin(frames)
        mask_labels = self.dataframe["label"].isin(labels)
        mask = mask_frames & mask_labels

        sub_df = self.dataframe.loc[
            mask, ["frame", "time", "label", "collision"]
        ].copy()

        collisions_per_frame = sub_df.groupby("frame")["collision"].sum()
        if occurences_mode == "fraction":
            collisions_per_frame /= collisions_per_frame.sum()
        if occurences_mode == "percent":
            collisions_per_frame /= collisions_per_frame.sum() * 100
        print(collisions_per_frame.sum(), len(collisions_per_frame))

        fig, ax = plt.subplots()

        x_converted = (
            sub_df["time"].unique()
            if x_unit == "s"
            else sub_df["frame"].unique()
            if x_unit == "frame"
            else None
        )

        ax.plot(x_converted, collisions_per_frame, marker="o")

        x_ticks = ax.get_xticks()[1:-1]
        y_ticks = ax.get_yticks()[1:-1]
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels([f"{x_tick:.1f}" for x_tick in x_ticks], fontsize=16)
        ax.set_yticklabels([f"{y_tick:.0f}" for y_tick in y_ticks], fontsize=16)

        if x_unit == "s":
            ax.set_xlabel("Time [s]", fontsize=18)
        if x_unit == "ms":
            ax.set_xlabel("Time [ms]", fontsize=18)
        if x_unit == "frames":
            ax.set_xlabel("Frame [#]", fontsize=18)

        if occurences_mode == "percent":
            ax.set_ylabel("Percentage of occurences [%]", fontsize=18)
        elif occurences_mode == "fraction":
            ax.set_ylabel("Fraction of occurences", fontsize=18)
        elif occurences_mode == "number":
            ax.set_ylabel("Occurences [#]", fontsize=18)

        plt.show()

    def Spatial_correlations(
        self,
        correlation: str | str | str = ("position", "velocityposition_velocity"),
        labels=None,
        path_save=None,
        do_save=False,
        units="meter",
    ):

        if not isinstance(correlation, str):
            msg = "correlation variable must be string"
            raise TypeError(msg)

        if labels is None:
            msg = "Must specify a labels"
            raise TypeError(msg)
        if isinstance(labels, int):
            labels = [labels]
        if not isinstance(labels, (np.ndarray, list)):
            msg = "labels variable must be np.ndarray or list"
            raise TypeError(msg)

        def compute_position_correlation(positions, bin_width=0.5, max_distance=5.0):
            positions = positions.to_numpy()
            # compute distances
            distances = []
            for i, pos1 in enumerate(positions):
                for j, pos2 in enumerate(positions):
                    if i < j:
                        distances.append(np.linalg.norm(pos1 - pos2))
            # compute histogram
            distances = np.array(distances)
            bins = np.arange(0, max_distance + bin_width, bin_width)
            hist, edges = np.histogram(distances, bins=bins)
            # compute normalised density
            bin_centers = 0.5 * (edges[:-1] + edges[1:])
            area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
            density = len(positions) / (np.max(positions[:, 0]) ** 2)
            g_r = hist / (area * density)
            return bin_centers, g_r

        def compute_velocity_correlation(velocities, bin_width=0.5, max_distance=5.0):
            velocities = velocities.to_numpy()
            velocity_corr = []
            distance_bins = []
            for i, vel1 in enumerate(velocities[:, 0]):
                for j, vel2 in enumerate(velocities[:, 1]):
                    if i < j:
                        distance = np.linalg.norm(vel1 - vel2)
                        velocity_dot = np.dot(velocities[i], velocities[j])
                        velocity_corr.append((distance, velocity_dot))
            velocity_corr = np.array(velocity_corr)
            distances = velocity_corr[:, 0]
            velocity_products = velocity_corr[:, 1]
            bins = np.arange(0, max_distance + bin_width, bin_width)
            digitized = np.digitize(distances, bins)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            corr_values = [
                np.mean(velocity_products[digitized == i]) for i in range(1, len(bins))
            ]
            return bin_centers, corr_values

        def compute_position_velocity_correlation(
            positions, velocities, bin_width=0.5, max_distance=5.0
        ):
            positions = positions.to_numpy()
            velocities = velocities.to_numpy()
            pos_vel_corr = []
            distance_bins = []
            for i, pos1 in enumerate(positions):
                for j, pos2 in enumerate(positions):
                    if i < j:
                        distance = np.linalg.norm(pos1 - pos2)
                        rel_velocity = np.linalg.norm(velocities[i] - velocities[j])
                        pos_vel_corr.append((distance, rel_velocity))
            # Organiser par distance et créer des moyennes
            pos_vel_corr = np.array(pos_vel_corr)
            distances = pos_vel_corr[:, 0]
            velocity_differences = pos_vel_corr[:, 1]
            bins = np.arange(0, max_distance + bin_width, bin_width)
            digitized = np.digitize(distances, bins)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            corr_values = [
                np.mean(velocity_differences[digitized == i])
                for i in range(1, len(bins))
            ]
            return bin_centers, corr_values

        sub_df = self.dataframe[self.dataframe["label"].isin(labels)]

        if not all(key in sub_df.keys() for key in ("x", "y", "vx", "vy")):
            msg = "keys x and/or y and/or vx and/or vy not exists"
            raise KeyError(msg)

        if correlation == "position":
            bins, g = compute_position_correlation(positions=sub_df[["x", "y"]])
            label_plot = "Position correlation"
            if units not in ("meter", "px"):
                if units == "meter":
                    x_ticks = np.linspace(0, max(bins) * self.pixel_size, len(bins) + 1)
                    y_ticks = np.linspace(0, max(g), len(g) + 1)
                if units == "px":  # a finir
                    x_ticks = np.linspace(0, max(bins) * self.pixel_size, len(bins) + 1)
                    y_ticks = np.linspace(0, max(g), len(g) + 1)

        elif correlation == "velocity":
            bins, g = compute_velocity_correlation(velocities=sub_df[["vx", "vy"]])
            label_plot = "Velocity correlation"
            x_ticks = np.linspace(0, max(bins) * self.pixel_size, len(bins) + 1)
            y_ticks = np.linspace(0, max(g), len(g) + 1)

        elif correlation == "position_velocity":
            bins, g = compute_position_velocity_correlation(
                positions=sub_df[["x", "y"]], velocities=sub_df[["vx", "vy"]]
            )
            label_plot = "Position-Velocity correlation"
            x_ticks = np.linspace(0, max(bins) * self.pixel_size, len(bins) + 1)
            y_ticks = np.linspace(0, max(g), len(g) + 1)

        fig, ax = plt.subplots()

        ax.plot(
            bins,
            g,
            color="tab:blue",
            linestyle="-",
            label=label_plot,
        )

        # if units == 'meter':

        #     ax.set_xticks(x_ticks)
        #     ax.set_yticks(y_ticks)

        #     ax.set_xticklabels(x_ticks.astype(int), fontsize=18)
        #     ax.set_yticklabels(y_ticks.astype(int), fontsize=18)
        #     ax.set_xlabel(f'x {r'[px]'}', fontsize=20)
        #     ax.set_ylabel(f'y {r'[px]'}', fontsize=20)

        # elif units == 'px':

        #     x_ticks = np.linspace(0, max(bins), len(bins)+1)
        #     y_ticks = np.linspace(0, max(g), len(g)+1)

        #     ax.set_xticks(x_ticks)
        #     ax.set_yticks(y_ticks)

        #     ax.set_xticklabels(x_ticks.astype(int), fontsize=18)
        #     ax.set_yticklabels(y_ticks.astype(int), fontsize=18)
        #     ax.set_xlabel(f'x {r'[px]'}', fontsize=20)
        #     ax.set_ylabel(f'y {r'[px]'}', fontsize=20)

        # plt.legend()
        plt.subplots_adjust(0.08, 0.08, 0.96, 0.96, 0.0, 0.0)
        if do_save:
            name = Path(path_save) / Path(
                f"Visualize_{label_plot:s}_"
                + PurePath(path_save).parts[-1]
                + "_on_plate_"
                + str(labels)
                + ".png"
            )
            plt.savefig(name, dpi=120)
        else:
            plt.show()

    def Mean_square_displacement(
        self,
        path_dataframe: (str | list | Path) = None,
        frames: list = None,
        labels: list = None,
        path_save: str = None,
        do_save: bool = False,
        units: str = "frames",
    ):

        def msd(df):

            if not isinstance(df, pd.DataFrame):
                msg = "trajectories variable must be pd.DataFrame"
                raise TypeError(msg)

            max_lag = len(df["frame"].unique())
            msd_dict = {lag: [] for lag in range(1, max_lag)}

            for _, group in df.groupby("label"):
                group = group.sort_values(by="frame")
                x = group["x"].to_numpy()
                y = group["y"].to_numpy()
                n_frames = len(x)

                for lag in range(1, n_frames):
                    dx = x[lag:] - x[:-lag]
                    dy = y[lag:] - y[:-lag]
                    squared_displacement = dx**2 + dy**2
                    msd_dict[lag].extend(squared_displacement)

            total_msd, lag_msd, mean_msd, std_msd = [], [], [], []
            for lag, values in msd_dict.items():
                if len(values) > 0:
                    mean_msd.append(np.mean(values))
                    std_msd.append(np.std(values))
                    lag_msd.append(lag)
                    total_msd.append(values)

            return total_msd, np.array(lag_msd), np.array(mean_msd), np.array(std_msd)

        if isinstance(frames, int):
            frames = [frames]
        if isinstance(frames, str) and frames == "all":
            frames = self.dataframe["frame"].unique()
        if not isinstance(frames, (list, np.ndarray)):
            msg = "frames variable must be list or array"
            raise TypeError(msg)

        if path_save is None and do_save:
            msg = "Must specify a path to save picture"
            raise TypeError(msg)
        if path_save is not None and do_save is None:
            do_save = True

        if not ((units == "px") or (units == "meter") or (units == "milli")):
            msg = "units variable is not px or meter or milli"
            raise ValueError(msg)

        for path in path_dataframe:
            print(path)

            dataframe = self._load_dataframe(path)

            keys = [
                "frame",
                "time",
                "label",
                "x",
                "y",
            ]
            self._check_keys(dataframe, keys)

            mask = dataframe["frame"].isin(frames)

            sub_df = dataframe.loc[
                mask,
                keys,
            ].copy()

            _, ax = plt.subplots()

            for lbl in labels:
                sub_df = sub_df[sub_df["label"] == lbl].copy()
                print(sub_df.shape)
                print(sub_df)

                displacement, lag_msd, mean_msd, std_msd = msd(
                    sub_df.sort_values(by="frame")
                )

                print(len(displacement), len(lag_msd), len(mean_msd), len(std_msd))

                ax.plot(
                    lag_msd,
                    mean_msd,
                    color="tab:blue",
                    linestyle="-",
                    label="Mean MSD",
                )

                ax.fill_between(
                    lag_msd,
                    y1=mean_msd - std_msd / 2,
                    y2=mean_msd + std_msd / 2,
                    color="tab:blue",
                    alpha=0.2,
                    label="Standart deviation of MSD",
                )

                if units == "frames":
                    ax.set_xlabel("Frames", fontsize=20)
                elif units == "time":
                    ax.set_xlabel("Time [s]", fontsize=20)
                ax.set_ylabel("Mean square displacement (MSD) [mm]", fontsize=20)

            plt.legend(fontsize=12)
            plt.show()

    def Velocity_autocorrelation_function(
        self,
        path_save: str | Path = None,
        do_save: bool = False,
        use_fit: bool = True,
        plot_collisions: bool = True,
        normalisation: str = "max_vacf",
        max_velocity: float = None,
    ):

        if not isinstance(normalisation, str):
            msg = "normalisation must be str type"
            raise TypeError(msg)
        if normalisation not in ("max_vacf", "max_velocity", None):
            msg = "normalisation must be max_vacf or max_velocity"
            raise ValueError(msg)
        if normalisation == "max_velocity":
            if not isinstance(max_velocity, float):
                msg = "max_velocity must be float type"
                raise TypeError(msg)
            if max_velocity < 0.0:
                msg = "max_velocity must be greater than 0.0"
                raise ValueError(msg)

        def vacf(df):
            if not isinstance(df, pd.DataFrame):
                msg = "trajectories variable must be pd.DataFrame"
                raise TypeError(msg)

            df["velocity"].fillna(0.0, inplace=True)
            if any(df["velocity"].isna()):
                print("nan in velocity")
                # sys.exit(0)

            df = df.sort_values(by=["label", "frame"])
            vacf_list = []
            for label in df["label"].unique():
                particle_data = df[df["label"] == label]
                velocities = np.array(list(particle_data["velocity"]))
                n_steps = len(velocities)
                vacf = np.zeros(n_steps)
                for lag in range(n_steps):
                    dot_products = np.sum(
                        velocities[: n_steps - lag] * velocities[lag:], axis=0
                    )
                    vacf[lag] = np.mean(dot_products)
                vacf_list.append(vacf)

            max_length = max(len(vacf) for vacf in vacf_list)
            vacf_padded = [
                np.pad(vacf, (0, max_length - len(vacf)), mode="constant")
                for vacf in vacf_list
            ]
            vacf_avg = np.mean(vacf_padded, axis=0)
            return np.arange(len(vacf_avg)), vacf_avg

        def exponential_fit(x, a, b, c):
            return a * np.exp(-b * (x - c))

        sub_df = self.dataframe.copy()

        lag, correlation = vacf(sub_df[["frame", "label", "velocity"]])
        if normalisation == "max_vacf":
            correlation /= max(correlation)
        elif normalisation == "max_velocity":
            correlation /= max_velocity
        # print(len(lag), len(correlation))

        fig, ax = plt.subplots(1, 3)

        # autocorrelation
        ax[0].plot(
            lag,
            correlation,
            color="tab:blue",
            linestyle="--",
            label="$\\mathcal{{R}}_{{vv}}(t)$",
        )

        local_max_min = (
            np.diff(np.sign(np.diff(correlation))).nonzero()[0] + 1
        )  # local min+max
        local_min = (np.diff(np.sign(np.diff(correlation))) > 0).nonzero()[
            0
        ] + 1  # local min
        local_max = (np.diff(np.sign(np.diff(correlation))) < 0).nonzero()[
            0
        ] + 1  # local max

        ax[0].scatter(
            local_max,
            correlation[local_max],
            color="tab:orange",
            marker="x",
            label="local peaks",
        )

        if use_fit:
            fit, _ = self.Fit_curve(
                x=lag,
                data=correlation,
                func=exponential_fit,
                p0=[1, 1, lag.min()],
                bounds=[0, np.inf],
            )

            fitted = exponential_fit(lag, *fit)
            fitted /= max(fitted)

            ax[0].plot(
                lag,
                fitted,
                color="tab:red",
                linestyle="-",
                label=f"$Z_{{fit}}=exp(-t / \\tau)$ | $\\tau$ = {1 / fit[1]:.2f}",
            )
            if normalisation is None:
                ax[0].set_yscale("log")

        if plot_collisions:
            collision_index = sub_df.index[sub_df["collision"]].to_list()
            collision_frames = (
                sub_df.loc[collision_index, "frame"].unique() - sub_df["frame"].min()
            )
            ax[0].vlines(
                x=collision_frames,
                ymin=min(fitted),
                ymax=max(fitted),
                colors="tab:green",
                linestyle="--",
            )

        # x_ticks = np.linspace(0, max(lag), len(lag)+1)
        # y_ticks = np.linspace(0, 1, len()+1)

        # ax.set_xticks(x_ticks)
        # ax.set_yticks(y_ticks)

        # ax.set_xticklabels(x_ticks.astype(int), fontsize=18)
        # ax.set_yticklabels(y_ticks.astype(int), fontsize=18)
        ax[0].set_xlabel("Frames number", fontsize=20)
        ax[0].set_ylabel("Normalized $\\mathcal{{R}}_{{vv}}(t)$", fontsize=20)

        # Fourier transform of autocorrelation
        def fourier_transform(func):
            if not isinstance(func, np.ndarray):
                func = np.array(func)
            freq_signal = scipy.fft.fft(func)
            freqs = np.fft.fftfreq(len(func))
            power_signal = np.abs(freq_signal) ** 2
            return freqs, power_signal, freq_signal

        fourier_correlation = fourier_transform(correlation)

        ax[1].plot(
            fourier_correlation[0],
            fourier_correlation[1],
            color="tab:red",
            linestyle="-",
            label="Power",
        )
        # ax[1].plot(
        #     fourier_correlation[0], np.real(fourier_correlation[2]),
        #     color='tab:orange', linestyle='-', label='Real part',
        # )
        # ax[1].plot(
        #     fourier_correlation[0], np.imag(fourier_correlation[2]),
        #     color='tab:green', linestyle='-', label='Imaginary part',
        # )

        local_min_max = (
            np.diff(np.sign(np.diff(np.log10(fourier_correlation[1])))).nonzero()[0] + 1
        )  # local min+max
        local_min = (
            np.diff(np.sign(np.diff(np.log10(fourier_correlation[1])))) > 0
        ).nonzero()[0] + 1  # local min
        local_max = (
            np.diff(np.sign(np.diff(np.log10(fourier_correlation[1])))) < 0
        ).nonzero()[0] + 1  # local max

        ax[1].scatter(
            fourier_correlation[0][local_max],
            fourier_correlation[1][local_max],
            color="tab:green",
            marker="x",
        )
        ax[1].set_yscale("log")
        ax[1].set_xlabel("$\\omega$", fontsize=20)
        ax[1].set_ylabel(
            "$\\mathcal{{F}}[\\mathcal{{R}}_{{vv}}(t)](\\omega)$", fontsize=20
        )

        # invert Fourier transform of autocorrelation
        def invert_fourier_transform(func):
            if not isinstance(func, np.ndarray):
                func = np.array(func)
            signal = scipy.fft.ifft(func)
            time = np.linspace(0, len(signal), len(signal))
            power_signal = np.abs(signal) ** 2
            return time, power_signal, signal

        invert_fourier_correlation = invert_fourier_transform(
            np.sqrt(fourier_correlation[1])
        )

        ax[2].plot(
            invert_fourier_correlation[0],
            invert_fourier_correlation[1],
            color="tab:red",
            linestyle="-",
            label="Power",
        )
        ax[2].set_yscale("log")
        ax[2].set_xlabel("t", fontsize=20)
        ax[2].set_ylabel(
            "$\\mathcal{{F}}^{{-1}}\\left[\\sqrt{{\\mathcal{{F}}(\\mathcal{{R}}_{{vv}}(t))(\\omega)}}\\right]$",
            fontsize=20,
        )

        plt.legend(fontsize=12)
        plt.subplots_adjust(
            left=0.05, bottom=0.1, right=0.99, top=0.88, wspace=0.25, hspace=0.2
        )
        plt.show()

    def Spectral_analysis_of_trajectories(
        self, labels, path_save=None, do_save=False, do_smooth=True
    ):

        if isinstance(labels, int):
            labels = [labels]
        if not isinstance(labels, (list, np.ndarray)):
            msg = "labels varible must be ndarray or list"
            raise TypeError(msg)

        def fourier_transform(trajectories, d_unit):
            if not isinstance(trajectories, np.ndarray):
                trajectories = np.array(trajectories)
            freq_x = scipy.fft.fft(trajectories[:, 0])
            freq_y = scipy.fft.fft(trajectories[:, 1])
            freqs = scipy.fft.fftfreq(len(trajectories))
            freqs = freqs / d_unit
            power_x = np.abs(freq_x) ** 2
            power_y = np.abs(freq_y) ** 2
            return freqs, power_x, power_y

        sub_df = self.dataframe[self.dataframe["label"].isin(labels)]
        trajectories = sub_df[["x", "y"]]
        print(np.mean(trajectories["x"].diff()))
        print(np.mean(trajectories["y"].diff()))
        freqs, power_x, power_y = fourier_transform(
            trajectories.to_numpy(), self.pixel_size
        )
        freqs, power_x, power_y = zip(*sorted(zip(freqs, power_x, power_y)))

        fig, ax = plt.subplots()

        ax.plot(
            freqs,
            power_x,
            color="tab:blue",
            linestyle="-",
            label="x-axis",
            alpha=0.5,
        )
        ax.plot(
            freqs,
            power_y,
            color="tab:orange",
            linestyle="-",
            label="y-axis",
            alpha=0.5,
        )

        if do_smooth:
            k = 3
            kernel = np.ones((k,))
            power_x_smoothed = scipy.signal.convolve(
                power_x, kernel, mode="same", method="auto"
            )
            power_y_smoothed = scipy.signal.convolve(
                power_y, kernel, mode="same", method="auto"
            )

            ax.plot(
                freqs,
                power_x_smoothed,
                color="tab:blue",
                linestyle="-",
                label="x-axis smoothed",
            )
            ax.plot(
                freqs,
                power_y_smoothed,
                color="tab:orange",
                linestyle="-",
                label="y-axis smoothed",
            )

        # get frequencies of n max peaks
        n_max = 5
        n_max_peaks_x = np.argsort(power_x_smoothed)[-n_max:][::-1].astype(np.int64)
        n_max_peaks_y = np.argsort(power_y_smoothed)[-n_max:][::-1].astype(np.int64)
        print([freqs[ind] for ind in n_max_peaks_x])
        print([freqs[ind] for ind in n_max_peaks_y])

        ax.scatter(
            [freqs[ind] for ind in n_max_peaks_x],
            power_x_smoothed[n_max_peaks_x],
            color="tab:blue",
            marker="x",
        )
        ax.scatter(
            [freqs[ind] for ind in n_max_peaks_y],
            power_y_smoothed[n_max_peaks_y],
            color="tab:orange",
            marker="x",
        )

        # x_ticks = freqs
        # y_ticks = power_x

        # ax.set_xticks(x_ticks)
        # ax.set_yticks(y_ticks)

        ax.set_yscale("log", base=10)

        # ax.set_xticklabels(x_ticks.astype(int), fontsize=18)
        # ax.set_yticklabels(y_ticks.astype(int), fontsize=18)

        ax.set_xlabel("$\\lambda [mm^{{-1}}]$", fontsize=20)
        ax.set_ylabel("$\\mathcal{{F}}[P](\\lambda)$", fontsize=20)

        plt.legend(fontsize=12)
        plt.subplots_adjust(0.08, 0.08, 0.96, 0.96, 0.0, 0.0)
        if do_save:
            name = Path(path_save) / Path(
                f"Visualize_{1:s}_"
                + PurePath(path_save).parts[-1]
                + "_on_plate_"
                + str(labels)
                + ".png"
            )
            plt.savefig(name, dpi=120)
        else:
            plt.show()

    def Collision_rate(self, frames: tuple = (9000, 10000), u_mean: float = 10):

        def compute_concentration(
            df, frames: tuple = (9000, 10000), ratio: float = 1.0
        ):
            df = df[
                df["frame"].isin(
                    np.linspace(frames[0], frames[1], frames[1] - frames[0] + 1).astype(
                        int
                    )
                )
            ].copy()
            concentration = []
            for _, group in df.groupby("frame"):
                concentration.append(len(group) / (1024 * 512 * ratio * ratio))
            return concentration

        label_initial_frame = self.dataframe[
            self.dataframe["frame"] == self.dataframe["frame"].unique().min()
        ]["label"]
        # ratio_fluxes =

        concentration = compute_concentration(self.dataframe, frames, self.pixel_size)
        d_mean = self.dataframe.groupby("frame")["diameter"].mean()
        u_star = np.sqrt(u_mean * 2 * self.nu_air / d_mean)
        freq_coll = u_star * d_mean * concentration

        fig, ax = plt.subplots(2, 2, sharex=True)

        ax[0, 0].plot(
            self.dataframe["time"].unique(),
            freq_coll,
            color="tab:blue",
            linestyle="-",
            label="Collision frequency",
        )
        ax[0, 0].set_ylabel("$\\nu_c$ $[s^{{-1}}]$", fontsize=20)
        ax[0, 0].legend()

        ax[1, 0].plot(
            self.dataframe["time"].unique(),
            1 / freq_coll,
            color="tab:blue",
            linestyle="-",
            label="Collision rate",
        )
        ax[1, 0].set_xlabel("Time $[s]$", fontsize=20)
        ax[1, 0].set_ylabel("$\\tau_c$ $[s]$", fontsize=20)
        ax[1, 0].legend()

        colors = matplotlib.colormaps["jet"]
        for i, freq in enumerate(freq_coll):
            ax[0, 1].plot(
                self.dataframe["time"].unique(),
                np.exp(-self.dataframe["time"].unique() * freq),
                color=colors(i / len(self.dataframe["time"].unique())),
                linestyle="-",
            )
        ax[0, 1].set_xlabel("Time $[s]$", fontsize=20)
        ax[0, 1].set_ylabel("Exponential decay", fontsize=20)
        ax[0, 1].legend()

        plt.legend()
        plt.show()

    def Fluxes_variation(
        self, frames_bounds: np.ndarray | list, height: float = 100e-6, x_units="time"
    ):

        if not isinstance(frames_bounds, (np.ndarray, list)):
            frames_bounds = np.array(frames_bounds)
        if frames_bounds[0] == frames_bounds[1]:
            msg = "frames bounds must be different"
            raise ValueError(msg)
        if not (frames_bounds[0] < frames_bounds[1]):
            msg = "frames bounds min must be smaller than frames bounds max"
            raise ValueError(msg)

        def compute_flux(df, frames, x_lim, key):
            df = df[df["frame"].isin(frames)]
            detection, flux = [], []
            if key == "sup":
                for frame_idx, group in df.groupby("frame"):
                    filtered = group[group["x"] > x_lim]
                    if len(filtered) > 0:
                        detection.append(filtered["label"].tolist())
                        flux.append(len(filtered["label"]))
                    else:
                        detection.append(np.nan)
                        flux.append(0)
            if key == "inf":
                for _, group in df.groupby("frame"):
                    filtered = group[group["x"] < x_lim]
                    if len(filtered) > 0:
                        detection.append(filtered["label"].tolist())
                        flux.append(len(filtered["label"]))
                    else:
                        detection.append(np.nan)
                        flux.append(0)
            return detection, flux

        frames = np.linspace(
            frames_bounds[0], frames_bounds[1], frames_bounds[1] - frames_bounds[0] + 1
        ).astype(int)

        inlet_label, inlet_flux = compute_flux(
            self.dataframe, frames, x_lim=1010 * self.pixel_size, key="sup"
        )
        inlet_flux /= (
            512
            * self.pixel_size
            * height
            * (
                self.dataframe["time"].unique().max()
                - self.dataframe["time"].unique().min()
            )
        )

        outlet_label, outlet_flux = compute_flux(
            self.dataframe, frames, x_lim=24 * self.pixel_size, key="inf"
        )
        outlet_flux /= (
            512
            * self.pixel_size
            * height
            * (
                self.dataframe["time"].unique().max()
                - self.dataframe["time"].unique().min()
            )
        )

        fig, ax = plt.subplots()

        ax.plot(
            self.dataframe["time"].unique(),
            inlet_flux,
            color="tab:blue",
            linestyle="-",
            label="Inlet flux",
        )

        ax.plot(
            self.dataframe["time"].unique(),
            outlet_flux,
            color="tab:orange",
            linestyle="-",
            label="Outlet flux",
        )
        ax.set_xlabel("Time $[s]$", fontsize=20)
        ax.set_ylabel("$\\phi$ $[m^{{-2}} s^{{-1}}]$", fontsize=20)
        plt.legend(fontsize=12)
        plt.show()

    def Position_Diameters(self):
        df = self.dataframe[self.dataframe["frame"] == 9000]
        df = df[["x", "y", "diameter"]].copy()
        df.to_csv("Essai_2/X_Y_diameter_9000.csv")

    def Get_velocity_on_mouvement(
        self,
        dataframe_path: (str | list | Path) = None,
        frames: (list | np.ndarray) = None,
        num_bins: int = 10,
        bins_log: bool = False,
        velocity_min_max: (list | np.ndarray) = None,
        occurences_mode: str = "normal",
        do_save: bool = True,
        path_save: (Path | str) = None,
    ):

        if not isinstance(num_bins, int):
            msg = f"num_bins must be integer, not {type(num_bins)}"
            raise TypeError(msg)

        if isinstance(velocity_min_max, list):
            velocity_min_max = np.array(velocity_min_max)
        if (velocity_min_max is not None) and (
            not isinstance(velocity_min_max[0], float)
        ):
            msg = f"Lower bound velocity must be float type, not {type(velocity_min_max[0])}"
        if (velocity_min_max is not None) and (
            not isinstance(velocity_min_max[1], float)
        ):
            msg = f"Upper bound velocity must be float type, not {type(velocity_min_max[1])}"
        if velocity_min_max is not None:
            if velocity_min_max[0] > velocity_min_max[1]:
                msg = "Lower bound velocity must be less than upper bound velocity"
                raise ValueError(msg)

        def find_first_positive(group, min_frames: int = 1):
            # if len(group["frame"].unique()) < min_frames:
            #     return None
            first_positive = group[
                (group["velocity"] > 0) & (group["frame"] != 0)
            ].sort_values(by="frame")
            if not first_positive.empty:
                return first_positive.iloc[0]["velocity"]
            else:
                return None

        for path in dataframe_path:
            print(path)

            fig, ax = plt.subplots()

            dataframe = self._load_dataframe(path)

            required = {"frame", "label"}
            missing = required - set(dataframe.columns)
            if not required.issubset(dataframe.columns):
                msg = f"In file {Path(path).name}, missing required column(s) {missing}"
                raise TypeError(msg)

            sub_df = dataframe[["frame", "label", "velocity"]]

            print(f"\nNumber of frames : {len(sub_df['frame'].unique())}")
            print(f"Number of labels : {len(sub_df['label'].unique())}")

            if frames is None:
                frames = sub_df["frame"].to_numpy()
            else:
                if isinstance(frames, (int, np.ndarray)):
                    frames = [frames]
                sub_df = sub_df[sub_df["frame"].isin(frames)]

            print(f"{sub_df[sub_df['velocity'] > 0.0]}")

            # compute starting velocity
            velocity_starting = sub_df.groupby("frame").apply(
                find_first_positive, min_frames=5
            )

            # print(f"Number of detected particles : {len(velocity_starting)}")
            print(velocity_starting)

            if velocity_min_max is not None:
                vmin, vmax = velocity_min_max
                velocity_starting = velocity_starting[
                    (velocity_starting >= vmin) & (velocity_starting <= vmax)
                ]

            if bins_log:
                bins = np.logspace(
                    np.log10(min(velocity_starting)),
                    np.log10(max(velocity_starting)),
                    num=num_bins,
                )
            else:
                bins = np.linspace(
                    min(velocity_starting), max(velocity_starting), num=num_bins
                )

            # plot histogram of velocity
            counts, bins = np.histogram(a=velocity_starting, bins=bins)
            print(counts, bins)

            if occurences_mode == "percent":
                counts = counts / np.sum(counts) * 100
            elif occurences_mode == "fraction":
                counts = counts / np.sum(counts)
            # bin_centers = 0.5 * (bins[:-1] + bins[1:])

            ax.stairs(
                values=counts,
                edges=bins,
                fill=True,
                color="tab:blue",
                edgecolor="black",
                linewidth=1.5,
                alpha=0.7,
            )

            if bins_log:
                ax.set_xscale("log")
            ax.vlines(x=0.0, ymin=0, ymax=max(counts), color="tab:red")

            x_ticks = ax[0].get_xticks()
            y_ticks = ax[0].get_yticks()
            ax[0].set_xticks(x_ticks)
            ax[0].set_yticks(y_ticks)
            ax[0].set_xticklabels(
                [f"{bin_val:.0f}" for bin_val in x_ticks], fontsize=16
            )
            ax[0].set_yticklabels(
                [f"{hist_val:.2f}" for hist_val in y_ticks], fontsize=16
            )

            ax[0].set_xlabel("Starting velocity [mm/s]", fontsize=18)
            if occurences_mode == "percent":
                ax[0].set_ylabel("Percentage of occurences [%]", fontsize=18)
            elif occurences_mode == "fraction":
                ax[0].set_ylabel("Fraction of occurences", fontsize=18)
            elif occurences_mode == "number":
                ax[0].set_ylabel("Occurences [#]", fontsize=18)

            if do_save:
                name = Path(path_save) / Path(
                    path_save.split("/")[-2]
                    + "_creep_motion_"
                    + path_save.split("/")[-1]
                    + ".png"
                )
                print(name)
                # plt.savefig(name, dpi=120)
            else:
                plt.show()

    def Make_video(
        self,
        load_images,
        path_save,
        video_name,
        images_range,
        frame_rate=30,
        format_video=".avi",
        do_display_time: bool = True,
        rotate: int = 0,
    ):
        matplotlib.use("Agg")

        if len(images_range) != 2:
            msg = f"length of image_range is different from 2 : {len(images_range)}"
            raise TypeError(msg)

        if not Path(load_images).exists:
            msg = "File to load images do not exists"
            raise FileExistsError(msg)

        if not Path(path_save).exists:
            msg = "File to save video do not exists"
            raise FileExistsError(msg)

        list_images = [
            file for file in Path(load_images).iterdir() if Path(file).is_file()
        ]
        list_images = list(Tcl().call("lsort", "-dict", list_images))
        list_images = list_images[images_range[0] : images_range[1]]

        img = np.array(Image.open(Path(list_images[0])))
        if img.ndim == 3:  # Convert color image to inverted grayscale
            img = np.array(
                ImageOps.invert(
                    Image.fromarray(img).convert("L").rotate(rotate, expand=True)
                ),
                dtype=np.uint8,
            )
        height, width = img.shape

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        # if video_name is None:
        #     video_name = f"{str(path_save).split('\\')[-2]}_{str(path_save).split('\\')[-1]}_{str(load_images).split('\\')[-1]}_images_{images_range[0]}_{images_range[1]}.avi"

        video_writer = cv2.VideoWriter(
            Path(path_save) / Path(video_name), fourcc, frame_rate, (width, height)
        )

        min, max = images_range
        font = ImageFont.truetype("arial.ttf", size=42)

        for curr_img, img_name in zip(
            list(np.linspace(min, max + 1, max - min + 2, dtype=np.int16)), list_images
        ):
            pil_img = (
                Image.open(Path(img_name)).convert("RGB").rotate(rotate, expand=True)
            )

            if do_display_time:
                draw = ImageDraw.Draw(pil_img)
                text = f"t = {curr_img * 250e-6:2f} ms"
                draw.text((0, 0), text, font=font, fill=(0, 0, 128))

            img = np.array(pil_img)

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video_writer.write(img)

        video_writer.release()

        print("Videos created")


# %%
