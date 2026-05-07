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

from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import QSizePolicy, QLabel

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
import seaborn as sns

import pyarrow as pa
import pyarrow.parquet as pq

from inspect import signature
from functools import partial

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
from scipy.interpolate import UnivariateSpline
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
from queue import Empty

import time

from typing import Union, Optional, List

from PyQt5.QtCore import QObject, QThread, pyqtSignal

import traceback

import pickle


def _pickable(f):
    try:
        pickle.dumps(f)
        return True
    except Exception:
        return False


def _worker_wrapper(args):
    """Wrap process image function"""

    index, imag_path, params = args

    df = _process_image_core(img_path=imag_path, params=params)

    return index, df


def _load_image_core(image_path, invert, rotate_angle=0, crop=None) -> np.ndarray:
    """Image loader"""

    image_path = Path(image_path)
    if not image_path.exists():
        msg = f"Image {image_path} does not exists"
        raise FileNotFoundError(msg)

    img = Image.open(image_path).convert("L")
    if invert:
        img = ImageOps.invert(img)

    if rotate_angle != 0:
        img = img.rotate(rotate_angle, expand=True)

    if crop is not None:
        x_start, y_start, x_end, y_end = crop
        img = img.crop(x_start, y_start, x_end, y_end)

    return np.array(img, dtype=np.uint8)


def _process_image_core(img_path: str, params: dict) -> pd.DataFrame:
    """Analyse image patch"""

    all_data = []
    img = _load_image_core(
        img_path,
        invert=params["invert_grayscale"],
        rotate_angle=params["rotate_angle"],
        crop=params["crop"],
    )

    def _bilateral_filtering(img, sigma_s, sigma_r):
        """Compute bilateral filtering"""

        img = img.astype(np.float64)

        # spatial component
        img_smooth = gaussian_filter(img, sigma=sigma_s)

        # intensity component
        diff = img[:, :, None] - img[:, None, :]
        weights = np.exp(-(diff**2) / (2 * sigma_r**2))

        # normalisation
        weights_sum = np.sum(weights, axis=2)
        img_filtered = np.sum(weights * img[:, :, None], axis=2) / weights_sum

        return img_filtered

    def _convert_coordinates(min_row, min_col, max_row, max_col):
        x = min_row
        y = min_col
        height = max_row - min_row
        width = max_col - min_col
        return [x, height, y, width]

    def _gaussian_2d(xy, x0, y0, amplitude, sigma_x, sigma_y, offset):
        x, y = xy
        g = offset + amplitude * np.exp(
            -(((x - x0) ** 2) / (2 * sigma_x**2) + ((y - y0) ** 2) / (2 * sigma_y**2))
        )
        return g.ravel()

    def _refine_coordinates(x, y, img, bb):
        refined_coords = []
        region = img[bb[0] : bb[0] + bb[1], bb[2] : bb[2] + bb[3]]
        x_grid, y_grid = np.meshgrid(
            np.arange(bb[0], bb[0] + bb[1]), np.arange(bb[2], bb[2] + bb[3])
        )

        # initial value
        initial_guess = (x, y, region.max(), 1, 1, region.min())

        # use fit to adjust 2D gaussian model
        try:
            params, _ = curve_fit(
                _gaussian_2d,
                (x_grid.ravel(), y_grid.ravel()),
                region.ravel(),
                p0=initial_guess,
            )

            refined_x, refined_y = params[0], params[1]

            if not (
                bb[2] <= refined_x < bb[2] + bb[3]
                and bb[0] <= refined_y < bb[0] + bb[1]
            ):
                refined_x, refined_y = x, y

            refined_coords.append((refined_x, refined_y))

        except RuntimeError:
            refined_coords.append((x, y))

        return refined_coords

    # filter = 0.1
    # bilateral = _bilateral_filtering(img, filter, filter)
    # img = bilateral.copy()

    # compute thresholding
    threshold = filters.threshold_otsu(img)
    binary = img > threshold

    # compute fill holes
    binary = binary_fill_holes(binary)

    # compute small objects removing and clear bordering
    if params["small_objects"] is not None:
        binary = morphology.remove_small_objects(
            binary, min_size=params["small_objects"]
        )
    binary = clear_border(binary)

    distance = distance_transform_edt(binary)

    mask = morphology.h_maxima(distance, h=0.01)

    markers = measure.label(mask)

    labeled_image = watershed(-distance, markers, mask=binary)

    if params["circ_thresh"] is None:
        mean_circularity = np.mean(
            prop.circularity for prop in regionprops(labeled_image)
        )
        circ_thresh = (
            mean_circularity * 0.8,
            mean_circularity * 1.2,
        )
    else:
        circ_thresh = params["circ_thresh"]

    # extract particle properties
    for prop in regionprops(labeled_image, intensity_image=img):
        if prop.perimeter == 0 or prop.area == 0:
            continue

        # circularity = 4 * np.pi * prop.area / (prop.perimeter**2)
        circularity = prop.axis_minor_length / prop.axis_major_length
        bb = _convert_coordinates(*prop.bbox)

        if (bb[1] < 10000) and (bb[3] < 10000):
            mask = (labeled_image == prop.label).astype(np.float16)[
                bb[0] : bb[0] + bb[1], bb[2] : bb[2] + bb[3]
            ]
            if mask.size == 0:
                continue

            start_frame = None
            if start_frame is None:
                curr_frame = int(str(Path(img_path).name).split(".")[2].split("-")[0])

            if circ_thresh[0] <= circularity <= circ_thresh[1]:
                centroid = prop.centroid
                x, y = int(centroid[1]), int(centroid[0])
                # initial_center = (int(centroid[1]), int(centroid[0]))
                # refined_center = self._refine_center(prop, initial_center)
                # x, y = refined_center[0] + x_offset, refined_center[1] + y_offset
                diameter = 2 * np.sqrt(prop.area / np.pi)

                subpixel_coords = _refine_coordinates(x, y, img, bb)
                x_subpixel, y_subpixel = subpixel_coords[0]

                time = 0.0

                all_data.append(
                    {
                        "frame": curr_frame,
                        "main_path": str(Path(img_path).parent),
                        "name": str(Path(img_path).name),
                        "time": time,  # [s]
                        "bound_box": bb,
                        "coords_pixels": json.dumps(prop.coords.tolist())
                        if isinstance(prop.coords, np.ndarray)
                        else str(prop.coords),  # coords of pixels that define particle
                        "x": x,  # [px]
                        "y": y,  # [px]
                        "x_subpixel": x_subpixel,  # [px]
                        "y_subpixel": y_subpixel,  # [px]
                        "diameter": diameter,  # [px]
                        "perimeter": prop.perimeter,  # [px]
                        "area": prop.area,  # [px^2]
                        "circularity": circularity,
                        "intensity_max": prop.intensity_max,
                        "intensity_min": prop.intensity_min,
                        "intensity_mean": prop.intensity_mean,
                        "intensity_std": prop.intensity_std,
                        "equivalent_diameter_area": prop.equivalent_diameter_area,  # [px]
                        "axis_major_length": prop.axis_major_length,  # [px]
                        "axis_minor_length": prop.axis_minor_length,  # [px]
                        "image_width": int(img.shape[0]),
                        "image_height": int(img.shape[1]),
                    }
                )

                # if start_frame is None:
                #     curr_frame += 1

    return pd.DataFrame(all_data)  # .to_dict(orient="records")


class ParticleAnalyser(QObject):
    display_signal = pyqtSignal(int)
    progress_signal = pyqtSignal(int, int, int)
    finished_signal = pyqtSignal()

    def __init__(
        self,
        parent,
        display_every: int,
        image_paths: List[Union[str, Path]],
        name_save_files: List[str],
        do_analysis: bool = False,
        change_main_path_images: Optional[Union[Path | str]] = None,
        video_sequence: bool = True,
        min_max_img: Optional[Union[List[int], np.ndarray]] = None,
        crop_image: Optional[Union[List[int], np.ndarray]] = None,
        format: Optional[Union[List[str]]] = None,
        invert_grayscale: bool = True,
        use_threshold: bool = True,
        threshold: Optional[int] = None,
        adaptative_threshold: bool = False,
        use_canny: bool = True,
        use_noise: bool = False,
        fill_holes: bool = True,
        dataframe_path: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        start_frame: Optional[int] = None,
        is_subpixel_detection: bool = False,
        pixel_size: float = 1.0 / 146.0,  # [mm/px]
        circularity_threshold: List[float] = [0.5, 1.5],
        small_objects: int = 10,
        time_interval: Optional[float] = None,
        density: float = 2230.0,  # [kg/m^3]
        nu_air: float = 1.5 * 10 ** (-6),  # [m^2/s]
        polynomial_coefficient_tension_to_velocity: Optional[List[float]] = None,
        # [3784, -12628, 17465, -12806, 5245.1, -1136.7, 102.01, ],  # 3784 - 12628*x +17465*x^2 - 12806*x^3 + 5245*x^4 - 1136.7*x^5 + 102.01*x^6
        do_save_data: Optional[bool] = None,
        save_name: Optional[Union[str, Path]] = None,
        progress_bar_disappear: bool = True,
        total_num_workers: int = 10,
        num_worker_per_image: int = 1,
    ):
        super().__init__()

        self.parent = parent
        self.display_every = display_every  # display image during analysis
        self._is_running = True

        self.dict_fontsize = {
            "label": 18,
            "ticks": 18,
            "legend": 16,
            "subplots": {
                "left": 0.075,
                "bottom": 0.090,
                "right": 0.930,
                "top": 0.97,
                "wspace": 0.0,
                "hspace": 0.0,
            },
        }

        if (image_paths is None) or len(image_paths) == 0:
            msg = "Path of images is recquired and cannot be empty"
            raise ValueError(msg)
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        # if (image_paths is None) and (name_save_files is None):
        #     msg = "Must specify file name"
        #     raise AttributeError(msg)

        # if image_paths is not None:
        #     if not all(isinstance(path, (str, list)) and os.path.exists(path) for path in image_paths) or (dataframe_path is None):
        #         msg = 'One path is not valid. Must specify valid path'
        #         raise FileExistsError(msg)
        #     else:
        #         self.is_file_or_dir = False
        # if not all(isinstance(path, str) and os.path.isfile(path) for path in image_paths):
        #     msg = 'One file path is not valid. Must specify valid file path'
        #     # raise FileExistsError(msg)
        # else:
        #     self.is_file_or_dir = True

        self.image_paths = image_paths

        self.name_save_files = name_save_files

        self.do_analysis = do_analysis

        self.change_main_path_images = change_main_path_images

        self.video_sequence = video_sequence

        self.min_max_img = min_max_img
        if self.min_max_img is not None:
            if len(self.min_max_img) != 2:
                msg = "min_max_img must be length 2"
                raise TabError(msg)
            if self.min_max_img[0] > self.min_max_img[1]:
                msg = "Values of 'min_max_img' must be in ascending order"
                raise ValueError(msg)

        self.crop_image = crop_image

        self.format = format

        self.invert_grayscale = invert_grayscale

        self.use_threshold = use_threshold

        self.adaptative_threshold = adaptative_threshold

        self.use_canny = use_canny

        self.use_noise = use_noise

        self.fill_holes = fill_holes

        self.progress_bar_disappear = not progress_bar_disappear

        if dataframe_path is not None:
            if isinstance(dataframe_path, str):
                dataframe_path = []
            for path in dataframe_path:
                if os.path.exists(path):
                    print("Dataframe exists")
                else:
                    print(f"Actual dataframe path : {dataframe_path}")
                    msg = "Dataframe path not exists. Must specify a valid path"
                    raise FileExistsError(msg)
        self.dataframe_path = dataframe_path

        self.start_frame = start_frame

        self.is_subpixel_detection = is_subpixel_detection

        self.pixel_size = pixel_size

        if not isinstance(self.pixel_size, float):
            msg = f"'pixel_size' must be a float, not {type(self.pixel_size)}"
        if self.pixel_size == 0.0:
            msg = "'pixel_size variable can t be 0.0'"
            raise ValueError(msg)

        self.time_interval = time_interval

        # physical constants:
        self.nu_air = nu_air

        self.polynomial_coefficient_tension_to_velocity = (
            polynomial_coefficient_tension_to_velocity
        )

        self.do_save_data = do_save_data
        # if save_name is not None and dataframe_path is None:
        #     if isinstance(save_name, list):
        #         if len(save_name) != len(image_paths):
        #             print(len(save_name), len(image_paths))
        #             msg = "Files to save must have same length as number of folders"
        #             raise ValueError(msg)
        #         else:
        #             print(len(save_name), len(image_paths))
        #             msg = "File to save must have a name"
        #             raise ValueError(msg)
        self.save_name = save_name

        if not isinstance(progress_bar_disappear, bool):
            msg = "'progress_bar_disappear' must a bool type"
            raise TypeError(msg)

        if not isinstance(total_num_workers, int):
            msg = (
                f"'total_num_workers' must be a integer, not {type(total_num_workers)}"
            )

        # check for cores availability
        available_workers = mp.cpu_count()
        if total_num_workers > 1 and self.dataframe_path is None:
            if total_num_workers > available_workers:
                total_num_workers = available_workers - 2
                msg = (
                    f"\n"
                    "/!\\/!\\/!\\/!\\/!\\\n"
                    f"   Number of cores required is greater than number of available cores (max cores is {available_workers:d})\n"
                    f"   Number of cores is set to {total_num_workers:d}.\n"
                    f"/!\\/!\\/!\\/!\\/!\\"
                    f"\n"
                )
                print(msg)
            else:
                print(
                    f"Number of requested cores are available : {total_num_workers:d} cores"
                )
        self.total_num_workers = total_num_workers

        if not isinstance(num_worker_per_image, int):
            msg = f"'num_worker_per_image' must be a integer, not {type(num_worker_per_image)}"

        # check for num_worker_per_image availability
        if num_worker_per_image > 1:
            while self.total_num_workers * num_worker_per_image > available_workers:
                num_worker_per_image -= 1
                if num_worker_per_image < 1:
                    num_worker_per_image = 1
                    break
            print(f"Number of worker to analyse each image is {num_worker_per_image:d}")
        self.num_worker_per_image = num_worker_per_image

        if threshold is not None:
            if not isinstance(threshold, int):
                msg = f"threshold variable must be an integer not a {type(threshold)}"
        self.threshold = threshold

        if not isinstance(small_objects, int):
            msg = f"'small_objects' variable must be an integer not a {type(small_objects)}"
        self.small_objects = small_objects

        if not all(circ_thresh > 0.0 for circ_thresh in circularity_threshold):
            msg = "'circularity_threshold' variable must be greater than 0.0"
        self.circularity_threshold = circularity_threshold

        # if not all(hu_thresh > 0.0 for hu_thresh in hu_moments_threshold):
        #     msg = "'hu_moments_threshold' variable must be greater than 0.0"
        # self.hu_moments_threshold = hu_moments_threshold

        if density < 0.0:
            msg = "density variable must be positive"
            raise ValueError(msg)
        self.density = density

        self.max_distance = 10.0

        self.memory = 5

    def _get_worker_state(self):
        return {
            "video_sequence": self.video_sequence,
            "max_distance": self.max_distance,
            "memory": self.memory,
        }

    @classmethod
    def _restore_worker_state(cls, state):
        cls.video_sequence = state["video_sequence"]
        cls.max_distance = state["max_distance"]
        cls.memory = state["memory"]

    def _stop(self):
        self._is_running = False

    def Process_images(self):

        if (
            self.dataframe_path is None
            and self.name_save_files is None
            and self.do_analysis
        ):
            msg = "must specify value for 'name_save_file' or 'dataframe_path'"
            raise TypeError(msg)

        if self.do_analysis:
            if self.dataframe_path is not None or self.name_save_files is not None:
                dataframe = self._process_images_from_folders()
        else:
            dataframe = self._load_existing_dataframe()

            return dataframe

    def _process_images_from_folders(self):

        all_results = []
        print()

        for ii, folder in enumerate(self.image_paths):
            output_path = Path(self.name_save_files)

            # check os
            if os.name == "nt":
                folder = Path(PureWindowsPath(folder))
                print("Operating System is Windows")
            else:
                folder = Path(PurePosixPath(folder))
                print("Operating System is Linux or MacOS")

            # check image format
            if Path(folder).exists():
                if self.format is None:
                    list_images = [
                        file for file in Path(folder).iterdir() if file.is_file()
                    ]
                else:
                    valid_formats = self.Process_formats(self.format)
                    list_images = [
                        file
                        for file in Path(folder).iterdir()
                        if file.is_file and file.suffix[1:] in valid_formats
                    ]
            else:
                msg = f"Folder '{folder}' do not exists"
                raise FileNotFoundError(msg)

            # sorting and selection
            list_images = list(Tcl().call("lsort", "-dict", list_images))
            if self.min_max_img is None:
                self.min_max_img = [0, len(list_images)]
            if isinstance(self.min_max_img[ii], list):
                list_images = list_images[
                    int(self.min_max_img[ii][0]) : int(self.min_max_img[ii][1])
                ]
            if self.min_max_img is not None:
                list_images = list_images[
                    int(self.min_max_img[0]) : int(self.min_max_img[1])
                ]

            if isinstance(list_images, str):
                list_images = [list_images]

            self.list_images = list_images

            if output_path.exists():
                print(f"File {output_path.name} already exists in {output_path.parent}")
                self.progress_signal.emit(
                    0, len(self.list_images), len(self.list_images)
                )
                # self.finished_detection_signal.emit()
                continue

            print(
                f"Number of images to process for '{folder}' is {len(self.list_images)}\n"
            )

            # analyse images
            self.results = [None] * len(self.list_images)

            if self.total_num_workers > 1:
                print("Use multi processing ...")
                self.Process_images_parallel(output_file=output_path)

            else:
                for idx, path_image in enumerate(
                    tqdm(
                        self.list_images,
                        desc="Analyse images",
                        bar_format="{l_bar}{bar:40}{r_bar}",
                        colour="white",
                        unit="image",
                        leave=True,
                    )
                ):
                    res = self.Process_image(image_path=path_image)
                    self.results[idx] = pd.DataFrame(res)
                    if self.start_frame is not None:
                        self.start_frame += 1
                    self.progress.emit(idx, len(self.list_images))

                # fuse and save
                if self.results and any(r is not None for r in self.results):
                    df = pd.concat(self.results, ignore_index=True)
                    all_results.append(df)

                    if isinstance(self.save_name, list):
                        save_file = self.save_name[ii]
                        df.to_csv(save_file, index=False)
                        print(f"Data saved for file {save_file} !\n")
                    else:
                        df.to_csv(save_file, index=False)
                        print("Data saved")

        # run next functions : label -> velocity + acceleration + kinetic energy + ...
        print("PIPELINE ...", end=" ")
        analyser_class = self.__class__
        worker_state = self._get_worker_state()
        self._run_pipeline(
            folders=output_path,
            analyser_class=analyser_class,  # self
            worker_class=worker_state,
            n_workers=self.total_num_workers,
        )
        print("DONE")

        # return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

    def Process_images_parallel(
        self,
        chunksize: int = 20,
        save_every: int = 1000,
        output_file: Path | str = None,
    ):
        """Analyse images with multiporcessing"""

        params = {
            "pixel_size": self.pixel_size,
            "bilateral_filter_sigma": 0.1,
            "threshold": self.threshold,
            "small_objects": self.small_objects,
            "circ_thresh": self.circularity_threshold,
            "invert_grayscale": self.invert_grayscale,
            "num_worker_per_image": self.num_worker_per_image,
            "dict_fontsize": self.dict_fontsize,
            "rotate_angle": 0,
            "crop": None,
        }

        args_list = [
            (i, img_name, params) for i, img_name in enumerate(self.list_images)
        ]

        output_file = Path(output_file)
        output_file = output_file.parent / output_file.name
        output_file.parent.mkdir(parents=True, exist_ok=True)

        processed = 0

        # ==========
        # multiprocessing with context manager
        # ==========

        with mp.Pool(
            processes=self.total_num_workers,
            maxtasksperchild=500,
        ) as pool:
            writer = None

            for index, df in pool.imap_unordered(
                _worker_wrapper,
                args_list,
                chunksize=chunksize,
            ):
                # user interruption
                if not getattr(self, "_is_running", True):
                    pool.terminate()
                    break

                # write parquet stream
                table = pa.Table.from_pandas(df)

                if writer is None:
                    writer = pq.ParquetWriter(output_file, table.schema)

                writer.write_table(table)

                processed += 1

                # progress signal to GUI
                self.progress_signal.emit(
                    0,
                    processed,
                    len(self.list_images),
                )

                if processed % self.display_every == 0:
                    print(f"Signal emit : {processed}")
                    self.display_signal.emit(processed)

        if writer is not None:
            writer.close()

        table = pq.read_table(output_file)
        df = table.to_pandas()
        df.to_csv(output_file, index=False)

        self.finished_signal.emit()
        print(f"Detection of all images is over : {processed}")

    def Process_image(
        self, image_path: Path | str = None, do_plot: bool = False
    ) -> pd.DataFrame:
        """
        Splits the image into n horizontal patch and processes each patch independently.
        Merge the results into a single dataframe
        """

        # load image
        img = self._load_image(image_path, invert=False, crop=None, rotate_image=270)
        self.img_size = img.shape

        if do_plot:
            _, ax = plt.subplots()
            ax.imshow(img, cmap="gray")
            x_ticks, y_ticks = ax.get_xticks()[1:-1], ax.get_yticks()[1:-1]
            ax.set_xticks(x_ticks), ax.set_yticks(y_ticks)
            ax.set_xticklabels(
                [f"{x_tick * self.pixel_size:.0f}" for x_tick in x_ticks],
                fontsize=self.dict_fontsize["ticks"],
            )
            ax.set_yticklabels(
                [f"{y_tick * self.pixel_size:.0f}" for y_tick in y_ticks],
                fontsize=self.dict_fontsize["ticks"],
            )
            ax.set_xlabel("x $[mm]$", fontsize=self.dict_fontsize["label"])
            ax.set_ylabel("y $[mm]$", fontsize=self.dict_fontsize["label"])
            ax.set_title("Original")
            plt.show()

        if self.invert_grayscale:
            img = np.array(
                ImageOps.invert(
                    Image.fromarray(img).convert("L")  # .transpose(Image.ROTATE_270)
                ),
                dtype=np.uint8,
            )
        else:
            img = np.array(Image.fromarray(img).convert("L"), dtype=np.uint8)

        n_patches = self.num_worker_per_image
        height = img.shape[0]
        h_patch = height // n_patches
        patches = []
        for i in range(n_patches):
            y_start = i * h_patch
            y_end = height if i == n_patches - 1 else (i + 1) * h_patch
            patch = img[y_start:y_end, :]
            patches.append((patch, (y_start, 0), i, image_path))

        # multi-process
        args_list = [(self, p[0], p[1], p[2], p[3]) for p in patches]
        if mp.current_process().daemon:  # multi CPU AND multi CPU per images
            return pd.concat([self._process_patch_wrapper(*args) for args in args_list])
        with mp.Pool(processes=self.num_worker_per_image) as pool:
            res = pool.starmap(self._process_patch_wrapper, args_list)
            # mono CPU AND multi CPU per images | mono CPU AND mono CPU per image
        return pd.concat(res) if len(res) > 1 else pd.DataFrame(res[0])

    def _load_image(
        self,
        name: str | Path,
        invert: bool = True,
        rotate_image: int = None,
        crop: list = None,
    ):

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

    def _run_pipeline(self, folders, analyser_class, worker_class, n_workers=4):

        manager = Manager()
        queue = manager.Queue()
        bars = {}

        if not isinstance(folders, list):
            folders = [folders]

        results = [
            analyser_class._worker_pipeline(
                path, analyser_class, worker_class, queue, task_id
            )
            for task_id, path in enumerate(folders)
        ]

        # with ProcessPoolExecutor(max_workers=1) as ex:
        #     futures = {
        #         ex.submit(analyser_class._worker_pipeline, path, analyser_class, worker_class, queue, task_id): path
        #             for task_id, path in enumerate(folders)
        #     }

        #     finished = 0
        #     while finished < len(folders):

        #         try:
        #             kind, *msg = queue.get()
        #         except Empty:
        #             continue

        #         kind, *msg = queue.get()

        #         if kind == "init":
        #             task_id, total, label = msg
        #             bars[task_id] = tqdm(
        #                 total=total,
        #                 desc=f"CPU {task_id} : {label}",
        #                 position = task_id,
        #                 bar_format="{l_bar}{bar:40}{r_bar}",
        #                 unit="step",
        #                 leave=True,
        #             )

        #         elif kind == "tick":
        #             task_id, stage = msg
        #             bar = bars[task_id]
        #             bar.update(1)
        #             bar.set_postfix_str(stage)

        #         elif kind == "done":
        #             task_id, _ = msg
        #             finished += 1
        #             bars[task_id].close()

        #         elif kind == "error":
        #             task_id, err_msg = msg
        #             finished += 1
        #             print(f"CPU {task_id}, error : {err_msg}")
        #             bars[task_id].close()

        #     for f in futures:
        #         f.result()

        #   self.finished_analysis_signal.emit()

    # def _run_pipeline(self, files, analyser_class, worker_class, n_workers=4):

    #     manager = Manager()
    #     queue = manager.Queue()

    #     print(f"IN RUN PIPELINE")

    #     if not isinstance(files, list):
    #         files = [files]

    #     for task_id, path in enumerate(files):

    #         finished = 0
    #         while finished < len(files):
    #             print(f"{finished}")

    #             try:
    #                 kind, *msg = queue.get()
    #             except Empty:
    #                 continue

    #             kind, *msg = analyser_class._worker_pipeline(
    #                 path, analyser_class, worker_class, queue, task_id,
    #                 )

    #             if kind == "init":
    #                 task_id, total, label = msg

    #             elif kind == "tick":
    #                 task_id, stage = msg

    #             elif kind == "done":
    #                 task_id, _ = msg
    #                 finished += 1

    #             elif kind == "error":
    #                 task_id, err_msg = msg
    #                 finished += 1
    #                 print(f"CPU {task_id}, error : {err_msg}")

    #     # self.finished_analysis_signal.emit()

    def _get_worker_state(self):
        return {
            "video_sequence": self.video_sequence,
            "max_distance": self.max_distance,
            "memory": self.memory,
        }

    @classmethod
    def _restore_worker_state(cls, state):
        cls.video_sequence = state["video_sequence"]
        cls.max_distance = state["max_distance"]
        cls.memory = state["memory"]

    @staticmethod
    def _worker_pipeline(path, analyser_class, worker_state, queue, task_id):

        try:
            total_steps = 8
            label = Path(path).name
            queue.put(("init", task_id, total_steps, label))

            # restore worker state
            analyser = analyser_class
            analyser._restore_worker_state(worker_state)
            final_path = Path(path).with_suffix(".csv")

            print("Pipeline loop")
            df = pd.DataFrame(pd.read_csv(path))
            # df.drop(columns=["label", "local_label", "roi_id"])

            print("ASSIGN ID ...", end=" ")
            df = analyser.Assign_ID_ROI(dataframe=df)
            df.to_csv(final_path)
            # queue.put(("tick", task_id, "Assign_ID"))
            # print("Assign_ID_ROI done !!!")
            print("done")

            print("UPDATE DIAMETERS ...", end=" ")
            # df = pd.DataFrame(pd.read_csv(path))
            df = analyser.UpdateDiameters(dataframe=df)
            df.to_csv(final_path)
            # queue.put(("tick", task_id, "UpdateDiameters"))
            # print("UpdateDiameters done !!!")
            print("done")

            # print("CLUSTER ...", end=" ")
            # # df = pd.DataFrame(pd.read_csv(path))
            # df = analyser_class.Cluster(dataframe=df)
            # df.to_csv(final_path)
            # # queue.put(("tick", task_id, "Cluster"))
            # # print("Cluster done !!!")
            # print("done")

            print("MASS ...", end=" ")
            # df = pd.DataFrame(pd.read_csv(path))
            df = analyser_class.Mass(dataframe=df, density=2230.0)
            df.to_csv(final_path)
            # queue.put(("tick", task_id, "Mass"))
            # print("Mass done !!!")
            print("done")

            # print("VELOCITY ...", end=" ")
            # # df = pd.DataFrame(pd.read_csv(path))
            # df = analyser_class.Velocity(dataframe=df, time_interval=1 / 8000)
            # df.to_csv(final_path)
            # # queue.put(("tick", task_id, "Velocity"))
            # # print("Velocity done !!!")
            # print(" done")

            # print("ACCELERATION ...", end=" ")
            # # df = pd.DataFrame(pd.read_csv(path))
            # df = analyser_class.Acceleration(dataframe=df)
            # df.to_csv(final_path)
            # # queue.put(("tick", task_id, "Acceleration"))
            # # print("Acceleration done !!!")
            # print(" done")

            # print("MOMENTUM ...", end=" ")
            # # df = pd.DataFrame(pd.read_csv(path))
            # df = analyser_class.Momentum(dataframe=df)
            # df.to_csv(final_path)
            # # queue.put(("tick", task_id, "Momentum"))
            # # print("Momentum done !!!")
            # print(" done")

            # print("KINETIC ...", end=" ")
            # # df = pd.DataFrame(pd.read_csv(path))
            # df = analyser_class.KineticEnergy(dataframe=df)
            # df.to_csv(final_path)
            # # queue.put(("tick", task_id, "KineticEnergy"))
            # # print("KineticEnergy done !!!")
            # print("done")

            # df = pd.DataFrame(pd.read_csv(path))
            # df = analyser_class.Collision(dataframe=df)
            # df.to_csv(final_path)
            # queue.put(("tick", task_id, "Collision"))

            df.to_csv(final_path)
            # queue.put(("done", task_id, str(final_path)))

        except Exception as e:
            print(f"Error occured : {e}")
            traceback.print_exc()

        print(final_path)
        return str(final_path)

        # except Exception as e:
        #     queue.put(("error", task_id, str(e)))
        #     return None

    # def Assign_ID_ROI(
    #     dataframe: pd.DataFrame = None,
    #     max_dist: int = 10,  # px
    #     do_plot: bool = False,
    #     language: str = "en",
    #     unit: str = "px",
    #     labels_used: str = "global",
    # ):

    #     def _generate_roi(
    #         x_min: int,
    #         x_max: int,
    #         y_min: int,
    #         y_max: int,
    #         roi_w: int,
    #         roi_h: int,
    #         overlap: int,
    #     ):
    #         rois = []
    #         step_x = roi_w - overlap
    #         step_y = roi_h - overlap

    #         x0 = x_min
    #         while x0 + roi_w <= x_max:
    #             y0 = y_min
    #             while y0 + roi_h <= y_max:
    #                 rois.append((x0, y0, x0 + roi_w, y0 + roi_h))
    #                 y0 += step_y
    #             x0 += step_x
    #         return rois

    #     def _track_all_rois_mp(
    #         df: pd.DataFrame = None,
    #         rois: list = None,
    #         max_dist: int = None,
    #         nproc: int = None,
    #     ):

    #         if nproc < 1:
    #             msg = f"'nproc' must be greater than 0, current value is {nproc}"
    #             raise ValueError(msg)

    #         args = [(i, roi, df, max_dist) for i, roi in enumerate(rois)]

    #         with mp.Pool(processes=nproc) as pool:
    #             results = pool.map(ParticleAnalyser._track_one_roi, args)

    #         results = [r for r in results if r is not None]
    #         return pd.concat(results).sort_index()

    #     def _assign_global_labels(
    #         df: pd.DataFrame = None,
    #         max_dist: int = None,
    #         do_plot: bool = False,
    #     ):

    #         df = df.copy()
    #         df["global_label"] = -1

    #         next_gid = 0
    #         frames = sorted(df["frame"].unique())

    #         prev_pos = None
    #         prev_gid = None

    #         for _, df_t in df.groupby("frame"):

    #             curr_pos = df_t[["x", "y"]].to_numpy()
    #             n = len(df_t)

    #             if prev_pos is None:
    #                 df.loc[df_t.index, "global_label"] = np.arange(
    #                     next_gid, next_gid + n, dtype=int
    #                 )
    #                 prev_gid = df.loc[df_t.index, "global_label"].to_numpy()
    #                 prev_pos = curr_pos
    #                 next_gid += n
    #                 continue
    #             else:
    #                 cost = cdist(prev_pos, curr_pos)
    #                 cost[cost > max_dist] = 1e3

    #             # display cost matrix
    #             if do_plot:
    #                 _, ax = plt.subplots()
    #                 ax.imshow(cost)

    #                 n_labels = max(cost.shape)

    #                 if n_labels <= 10:
    #                     base_map = plt.get_cmap("tab10")
    #                 if 10 < n_labels <= 20:
    #                     base_map = plt.get_cmap("tab20")
    #                 elif n_labels > 20:
    #                     base_map = plt.get_cmap("plasma")

    #                 ncolors = int(np.max(cost) - np.min(cost) + 1)
    #                 colors = base_map(np.linspace(0, 1, ncolors))
    #                 cmap = mcolors.ListedColormap(colors)

    #                 ax.set_xlim(0, cost.shape[1])
    #                 ax.set_ylim(0, cost.shape[0])

    #                 x_ticks = ax.get_xticks()[:-1]
    #                 y_ticks = ax.get_yticks()[:-1]

    #                 ax.set_xticks(x_ticks)
    #                 ax.set_yticks(y_ticks)

    #                 ax.set_xticklabels(
    #                     [f"{x_tick:.0f}" for x_tick in x_ticks]
    #                 )  # , fontsize=ParticleAnalyser.dict_fontsize["ticks"])
    #                 if language == "fr":
    #                     ax.set_xlabel(
    #                         "Particles detectées à l'image précédente"
    #                     )  # , fontsize=ParticleAnalyser.dict_fontsize["label"])
    #                 if language == "en":
    #                     ax.set_xlabel(
    #                         "Detected particles in previous frame"
    #                     )  # , fontsize=ParticleAnalyser.dict_fontsize["label"])

    #                 ax.set_yticklabels(
    #                     [f"{y_tick:.0f}" for y_tick in y_ticks]
    #                 )  # , fontsize=ParticleAnalyser.dict_fontsize["ticks"])
    #                 if language == "fr":
    #                     ax.set_ylabel(
    #                         "Particles detectées à l'image courante"
    #                     )  # , fontsize=ParticleAnalyser.dict_fontsize["label"])
    #                 if language == "en":
    #                     ax.set_ylabel(
    #                         "Detected particles in current frame"
    #                     )  # , fontsize=ParticleAnalyser.dict_fontsize["label"])

    #                 norm = mcolors.BoundaryNorm(
    #                     np.arange(np.min(cost), np.max(cost) + 2, 1), ncolors=ncolors
    #                 )

    #                 norm = plt.Normalize(vmin=np.min(cost), vmax=np.max(cost))
    #                 sm = plt.cm.ScalarMappable(
    #                     cmap=cmap,
    #                     norm=norm,
    #                 )
    #                 sm.set_array([])
    #                 cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.01)
    #                 cbar_ticks = cbar.get_ticks()
    #                 cbar.set_ticks(cbar_ticks)
    #                 cbar.set_ticklabels(
    #                     [f"{np.abs(cbar_tick):.0f}" for cbar_tick in cbar_ticks]
    #                 )  # , fontsize=ParticleAnalyser.dict_fontsize["ticks"])
    #                 if language == "fr":
    #                     cbar.set_label(
    #                         "Coût"
    #                     )  # , fontsize=ParticleAnalyser.dict_fontsize["label"])
    #                 if language == "en":
    #                     cbar.set_label(
    #                         "Cost"
    #                     )  # , fontsize=ParticleAnalyser.dict_fontsize["label"])

    #             plt.show()
    #                 # plt.savefig(
    #                 #     f"/home/abad-ale/Documents/Images_Assign_ID/Cost_matrix_frame_{t}.png"
    #                 # )

    #             gids = np.full(n, -1, dtype=int)
    #             if not (
    #                 np.any(~np.isfinite(cost).all(axis=1))
    #                 or np.any(~np.isfinite(cost).all(axis=0))
    #             ):
    #                 row_ind, col_ind = linear_sum_assignment(cost)

    #                 for i, j in zip(row_ind, col_ind):
    #                     if np.isfinite(cost[i, j]):
    #                         gids[j] = prev_gid[i]

    #             for j in range(n):
    #                 if gids[j] == -1:
    #                     gids[j] = next_gid
    #                     next_gid += 1

    #             df.loc[df_t.index, "global_label"] = gids
    #             prev_pos = curr_pos
    #             prev_gid = gids

    #         return df

    #     def _full_tracking_pipeline(
    #         df: pd.DataFrame = None,
    #         roi_params: list = None,
    #         max_dist: int = None,
    #         merge_dist: int = None,
    #         nproc: int = 1,
    #         do_plot: bool = False,
    #     ):
    #         xmin, xmax, ymin, ymax, roi_w, roi_h, overlap = roi_params
    #         rois = _generate_roi(xmin, xmax, ymin, ymax, roi_w, roi_h, overlap)

    #         df_local = _track_all_rois_mp(df, rois, max_dist=max_dist, nproc=nproc)
    #         df_final = _assign_global_labels(df_local, max_dist, do_plot=do_plot)

    #         return df_final, rois

    #     dataframe = dataframe.copy()

    #     required_cols = {"frame", "main_path", "name", "x", "y"}
    #     missing = required_cols - set(dataframe.columns)
    #     if missing:
    #         msg = f"Missing requiered column(s) : {missing}"
    #         raise KeyError(msg)

    #     # get image shape
    #     img_size = _load_image_core(
    #         image_path=Path(
    #             dataframe["main_path"].unique()[0], dataframe["name"].unique()[0]
    #         ),
    #         invert=False,
    #     ).shape

    #     # compute tracking
    #     # if not all(col in dataframe.columns for col in ["label", "local_label", "roi_id"]):
    #     df_local, rois = _full_tracking_pipeline(
    #         dataframe,
    #         roi_params=(0, img_size[1], 0, img_size[0], img_size[1], img_size[0], 0),
    #         max_dist=max_dist,
    #         # merge_dist=7,
    #         nproc=1,
    #     )

    #     df_tracked = dataframe.copy()
    #     df_tracked["label"] = -1
    #     df_tracked["local_label"] = -1
    #     df_tracked["roi_id"] = -1

    #     if not df_local.empty:
    #         df_tracked.loc[df_local.index, "label"] = df_local["global_label"]
    #         df_tracked.loc[df_local.index, "local_label"] = df_local["local_label"]
    #         df_tracked.loc[df_local.index, "roi_id"] = df_local["roi_id"]

    #     # else:
    #     #     df_tracked = dataframe.copy()

    #     # display
    #     if do_plot:
    #         if labels_used == "global":
    #             labels = df_tracked["label"].unique()
    #         else:
    #             labels = df_tracked["local_label"].unique()
    #         n_labels = len(labels)
    #         print()
    #         print("Info labels :")
    #         print(f"    number of labels : {n_labels}")
    #         print(
    #             f"    min label : {df_tracked['label'].unique().min()}, max label : {df_tracked['label'].unique().max()}"
    #         )

    #         frames = df_tracked["frame"].unique()
    #         n_frames = len(frames)

    #         print(f"Number of frames : {n_frames}")

    #         if n_labels <= 10:
    #             base_map = plt.get_cmap("tab10")
    #         if 10 < n_labels <= 20:
    #             base_map = plt.get_cmap("tab20")
    #         elif n_labels > 20:
    #             base_map = plt.get_cmap("plasma")

    #         colors = base_map(np.linspace(0, 1, n_labels))
    #         cmap = mcolors.ListedColormap(colors)
    #         norm = mcolors.BoundaryNorm(np.arange(n_labels + 1) - 0.5, n_labels)

    #         for frame_id, group_frames in df_tracked.groupby("frame"):

    #             _, ax = plt.subplots()

    #             ax.set_title(f"Frame : {frame_id}")

    #             # ----- add image
    #             path_img = Path(
    #                 group_frames["main_path"].unique()[0],
    #                 group_frames["name"].unique()[0],
    #             )
    #             img = _load_image_core(path_img, invert=False)
    #             ax.imshow(img, cmap="gray")

    #             # ----- add ROIs
    #             for roi in rois:
    #                 x0, y0, x1, y1 = roi
    #                 width, height = x1 - x0, y1 - y0
    #                 rect = patches.Rectangle(
    #                     (x0, y0),
    #                     width,
    #                     height,
    #                     edgecolor="black",
    #                     facecolor="tab:red",
    #                     alpha=0.2,
    #                 )
    #                 ax.add_patch(rect)

    #             # ----- add labels
    #             for i, (_, group_labels) in enumerate(group_frames.groupby("label")):
    #                 # add particle coordinates
    #                 ax.scatter(
    #                     group_labels["x"],
    #                     group_labels["y"],
    #                     color=cmap(i),
    #                     marker="o",
    #                     alpha=1.0,
    #                 )

    #                 # ----- add circle per label
    #                 for x, y, d in zip(
    #                     group_labels["x"], group_labels["y"], group_labels["diameter"]
    #                 ):
    #                     # add equivalent diameter
    #                     ax.add_patch(plt.Circle((x, y), d / 2, color="b", fill=False))

    #                     # add neighboor distance
    #                     ax.add_patch(
    #                         plt.Circle((x, y), max_dist, color="r", fill=False)
    #                     )

    #             ax.set_xlim(0, img_size[1])
    #             ax.set_ylim(0, img_size[0])

    #             x_ticks = ax.get_xticks()[:-1]
    #             y_ticks = ax.get_yticks()[:-1]

    #             ax.set_xticks(x_ticks)
    #             ax.set_yticks(y_ticks)

    #             ax.set_xlabel("x [px]")  # , fontsize=self.dict_fontsize["label"])
    #             ax.set_ylabel("y [px]")  # , fontsize=self.dict_fontsize["label"])
    #             ax.set_xticklabels(
    #                 [f"{x_tick:.0f}" for x_tick in x_ticks]
    #             )  # , fontsize=self.dict_fontsize["ticks"])
    #             ax.set_yticklabels(
    #                 [f"{y_tick:.0f}" for y_tick in y_ticks]
    #             )  # , fontsize=self.dict_fontsize["ticks"])

    #             # add colorbar
    #             # sm = plt.cm.ScalarMappable(
    #             #     cmap=cmap,
    #             #     norm=norm,
    #             #     )
    #             # sm.set_array([])
    #             # cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.01)
    #             # cbar_ticks = cbar.get_ticks()
    #             # cbar.set_ticks(cbar_ticks)
    #             # if language == "fr":
    #             #     cbar.set_label("Labels")#, fontsize=self.dict_fontsize["label"])
    #             # if language == "en":
    #             #     cbar.set_label("Labels")#, fontsize=self.dict_fontsize["label"])
    #             # cbar.set_ticklabels([])

    #             # if labels_used == "global":
    #             #     cbar.set_ticklabels([f"{np.abs(cbar_tick)+0.5:.0f}" for cbar_tick in cbar_ticks])#, fontsize=self.dict_fontsize["ticks"])
    #             # else:
    #             #     for i, cbar_tick in enumerate(cbar_ticks):
    #             #         cbar.ax.text(
    #             #             0.5, cbar_tick - 0.5,
    #             #             f"{int(cbar_tick)}",
    #             #             ha="center", va="center",
    #             #             # fontsize=self.dict_fontsize["ticks"],
    #             #         )

    #             plt.show()
    #             # plt.savefig(
    #             #     f"/home/abad-ale/Documents/Images_Assign_ID/Assign_ID_frame_{frame_id}.png"
    #             # )

    #     return df_tracked

    def Assign_ID_ROI(
        dataframe: pd.DataFrame = None,
        max_dist: int = 10,  # px
        do_plot: bool = True,
    ):
        """ Assign IDs to particles based on a frame-to-frame tracking """

        df = dataframe.copy()

        # ----- check missing columns
        required_cols = {"frame", "main_path", "name", "x", "y"}
        missing = required_cols - set(df.columns)
        if missing:
            raise KeyError(f"Missing required column(s) : {missing}")
        
        # ----- initialize dataframe
        df["label"] = -1
        next_id = 0
        prev_pos = None
        prev_id = None

        # ----- analyse by frame
        for _, df_frame in df.groupby("frame"):
            curr_pos = df_frame[["x", "y"]].to_numpy()
            n = len(df_frame)

            # assign initial labels
            if prev_pos is None:
                df.loc[df_frame.index, "label"] = np.arange(next_id, next_id + n, dtype=int)
                prev_id = df.loc[df_frame.index, "label"].to_numpy()
                prev_pos = curr_pos
                next_id += n
                continue
                
            # compute cost matrix
            cost = cdist(prev_pos, curr_pos)
            cost[cost > max_dist] = 1e3

            ids = np.full(n, -1, dtype=int)
            if np.all(~np.isfinite(cost)):
                row_ind, col_ind = linear_sum_assignment(cost)
                for i, j in zip(row_ind, col_ind):
                    if np.isfinite(cost[i, j]):
                        ids[j] = prev_id[i]
            
            # assign new IDs to unliked particles
            for j in range(n):
                if ids[j] == -1:
                    ids[j] = next_id
                    next_id += 1
            
            df.loc[df_frame.index, "label"] = ids
            prev_pos = curr_pos
            prev_id = ids
    
        def _plot_assign_particles(df, max_dist):

            frames = sorted(df["frame"].unique())
            n_labels = df["label"].nunique()

            for f, group_frames in df.groupby("frame"):

                fig, ax = plt.subplots()

                ax.set_title(f"Frame : {f}")

                # ----- add image
                path_img = Path(
                    group_frames["main_path"].unique()[0],
                    group_frames["name"].unique()[0],
                )
                img = _load_image_core(path_img, invert=False)
                ax.imshow(img, cmap="gray")

                # ----- add labels
                for i, (_, group_labels) in enumerate(group_frames.groupby("label")):
                    # add particle coordinates
                    ax.scatter(
                        group_labels["x"],
                        group_labels["y"],
                        marker="o",
                        alpha=1.0,
                    )

                    # ----- add circle per label
                    for x, y, d in zip(
                        group_labels["x"], group_labels["y"], group_labels["diameter"]
                    ):
                        # add equivalent diameter
                        ax.add_patch(plt.Circle((x, y), d / 2, color="b", fill=False))

                        # add neighboor distance
                        ax.add_patch(
                            plt.Circle((x, y), max_dist, color="r", fill=False)
                        )

                # ax.set_xlim(0, df["image_height"])
                # ax.set_ylim(0, df["image_width"])

                x_ticks = ax.get_xticks()[:-1]
                y_ticks = ax.get_yticks()[:-1]

                ax.set_xticks(x_ticks)
                ax.set_yticks(y_ticks)

                ax.set_xlabel("x [px]")  # , fontsize=self.dict_fontsize["label"])
                ax.set_ylabel("y [px]")  # , fontsize=self.dict_fontsize["label"])
                ax.set_xticklabels(
                    [f"{x_tick:.0f}" for x_tick in x_ticks]
                )  # , fontsize=self.dict_fontsize["ticks"])
                ax.set_yticklabels(
                    [f"{y_tick:.0f}" for y_tick in y_ticks]
                )  # , fontsize=self.dict_fontsize["ticks"])

                plt.show()

        # ----- plot
        if do_plot:
            _plot_assign_particles(df, max_dist)
        
        return df

    @staticmethod
    def _select_roi(df: pd.DataFrame = None, roi: list = None) -> pd.DataFrame:
        """Select particles in defined ROI"""

        x0, y0, x1, y1 = roi
        return df[(df["x"] >= x0) & (df["x"] < x1) & (df["y"] >= y0) & (df["y"] < y1)][
            ["frame", "x", "y"]
        ].copy()

    @staticmethod
    def _lap_tracking(df: pd.DataFrame = None, max_dist: int = None) -> pd.DataFrame:
        """Apply lap tracking to assign label"""

        df = df.copy()

        # get frame min and frame max
        frames = sorted(df["frame"].unique())

        # assign ID for frame min
        mask = df["frame"] == frames[0]
        df.loc[mask, "local_label"] = np.arange(mask.sum())
        next_id = mask.sum()

        for f_prev, f_curr in zip(frames[:-1], frames[1:]):
            prev = df[df.frame == f_prev].copy()
            curr = df[df.frame == f_curr].copy()
            curr["local_label"] = -1

            if len(prev) == 0:
                curr["local_label"] = np.arange(next_id, next_id + len(curr))
                next_id += len(curr)
                df.loc[curr.index, "local_label"] = curr["local_label"]
                continue

            cost = cdist(prev[["x", "y"]], curr[["x", "y"]])
            cost[cost > max_dist] = 100 # 1e2

            # check if cost matrix is finite for all values
            if np.all(~np.isfinite(cost)):
                curr["label"] = np.arange(next_id, next_id + len(curr))
                next_id += len(curr)
                df.loc[curr.index, "local_label"] = curr["local_label"]
                continue

            row_ind, col_ind = linear_sum_assignment(cost)

            # from scipy.sparse import csr_array
            # from scipy.sparse.csgraph import min_weight_full_bipartite_matching
            # biadjacency = csr_array(linear_sum_assignment(cost))
            # row_ind, col_ind = min_weight_full_bipartite_matching(biadjacency)

            assigned_curr = set()
            for i, j in zip(row_ind, col_ind):
                if np.isfinite(cost[i, j]):
                    curr.loc[curr.index[j], "local_label"] = prev.iloc[i]["local_label"]
                    assigned_curr.add(j)

            for j in range(len(curr)):
                if j not in assigned_curr:
                    curr.loc[curr.index[j], "local_label"] = next_id
                    next_id += 1

            df.loc[curr.index, "local_label"] = curr["local_label"]

        return df

    @staticmethod
    def _track_one_roi(args: list = None) -> pd.DataFrame:
        """Assign label for particles in one ROI"""

        roi_id, roi, df_roi, max_dist = args
        df_roi = ParticleAnalyser._select_roi(df_roi, roi)
        if df_roi.empty:
            return None

        df_lab = ParticleAnalyser._lap_tracking(df_roi, max_dist)
        df_lab["roi_id"] = roi_id
        return df_lab

    def UpdateDiameters(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Compute mean diameter of each particles"""

        df = dataframe.copy()

        required_cols = {"label", "diameter"}
        missing = required_cols - set(dataframe.columns)
        if missing:
            msg = f"Missing requiered column(s) : {missing}"
            raise KeyError(msg)

        if not all(
            col in dataframe.columns for col in ["diameter_mean", "diameter_std"]
        ):
            dataframe["diameter_mean"] = dataframe.groupby("label")[
                "diameter"
            ].transform("mean")
            dataframe["diameter_std"] = dataframe.groupby("label")[
                "diameter"
            ].transform("std")

        return dataframe

    # @staticmethod
    def Cluster(
        dataframe: pd.DataFrame = None,
        eps_multiplier: float = 1.7,
        min_samples: int = 2,
        n_jobs: int = 1,
    ) -> pd.DataFrame:
        """Cluster object using DBSCAN"""

        required_cols = {"frame", "label", "diameter", "x", "y"}
        missing = required_cols - set(dataframe.columns)
        if missing:
            msg = f"Missing requiered column(s) : {missing}"
            raise KeyError(msg)

        frames_groups = list(dataframe.groupby("frame"))

        results = []

        if n_jobs > 1:
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(ParticleAnalyser._cluster_frame)(
                    frame_id, group, eps_multiplier, min_samples
                )
                for frame_id, group in frames_groups
            )
        else:
            results = [
                ParticleAnalyser._cluster_frame(
                    frame_id, group, eps_multiplier, min_samples
                )
                for frame_id, group in frames_groups
            ]

        # for frame_id, res in results:
        #     print(frame_id)
        #     dataframe.loc[res.index, ["cluster_id", "cluster_label", ]] = res[["cluster_id", "cluster_label"]].values

        all_res = pd.concat([res for _, res in results])
        dataframe.loc[all_res.index, "cluster_id"] = all_res["cluster_id"].values
        dataframe.loc[all_res.index, "cluster_label"] = all_res["cluster_label"]

        dataframe = ParticleAnalyser._compute_coordination_number(dataframe)

        return dataframe

    # @staticmethod
    def _cluster_frame(frame_id, group, eps_multiplier, min_samples) -> pd.DataFrame:
        """Compute cluster frame"""

        result = group.copy()
        n = len(result)
        result["cluster_id"] = -np.ones(n, dtype=int)
        result["cluster_label"] = [[] for _ in range(n)]

        if len(group) < min_samples:
            return frame_id, result

        max_distances = group["diameter"].values * eps_multiplier / 2.0
        eps = np.median(max_distances)

        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_ids = clustering.fit_predict(group[["x", "y"]].values)
        result["cluster_id"] = cluster_ids

        for c_id in np.unique(cluster_ids):
            if c_id == -1:
                continue

            # identify cluster c_id
            members_idx = np.where(cluster_ids == c_id)[0]

            # get label on cluster c_id
            labels_in_cluster = group["label"].values[members_idx]

            neighbors = [
                list(np.delete(labels_in_cluster, i))
                for i in range((len(labels_in_cluster)))
            ]
            for idx_pos, neigh in zip(members_idx, neighbors):
                result.at[result.index[idx_pos], "cluster_label"].extend(neigh)
        return (frame_id, result)

    # @staticmethod
    def _compute_coordination_number(
        df: pd.DataFrame = None, do_plot: bool = False
    ) -> pd.DataFrame:
        """Compute coordination number for each particle"""

        df = df.copy()
        df["coordination"] = 0

        for _, group in df.groupby("frame"):
            pts = group[["x", "y"]].values
            d = group["diameter_mean"].values

            tree = cKDTree(pts)
            coord = np.zeros(len(group), dtype=int)

            # define max raduis
            r_max = d.max() + 0.1 * d.max()

            for i in range(len(group)):
                neighbors = tree.query_ball_point(pts[i], r_max)
                neighbors.remove(i)
                neighbors = [
                    j
                    for j in neighbors
                    if np.linalg.norm(pts[j] - pts[i]) >= (d[i] / 2)
                ]
                coord[i] = len(neighbors)
            df.loc[group.index, "coordination"] = coord

            # display
            if do_plot:
                _, ax = plt.subplots()

                name = Path(df["main_path"].iloc[0], df["name"].iloc[0])
                img = _load_image_core(name, invert=False)
                ax.imshow(img, cmap="gray")

                # plot coords + radius
                for pt, di in zip(pts, d):
                    ax.plot(pt[0], pt[1])
                    ax.add_patch(
                        plt.Circle((pt[0], pt[1]), di / 2, color="b", fill=False)
                    )

                    # plot research radius
                    ax.add_patch(
                        plt.Circle((pt[0], pt[1]), r_max, color="k", fill=False)
                    )

                plt.show()

        return df

    # @staticmethod
    def Mass(dataframe: pd.DataFrame, density: float) -> pd.DataFrame:
        """Compute mass in kg for each particle"""

        df = dataframe.copy()

        required_cols = {"diameter", "diameter_mean", "diameter_std"}
        missing = required_cols - set(df.columns)
        if missing:
            msg = f"Missing requiered column(s) : {missing}"
            raise KeyError(msg)

        if not all(
            col in df.columns
            for col in ["mass", "mass_mean", "mass_std", "mass_min", "mass_max"]
        ):
            grouped = df.groupby("label")
            df["mass"] = grouped["diameter"].transform(
                lambda x: 4 / 3 * np.pi * (x / 2.0) ** 3 * density
            )  # kg
            df["mass_mean"] = grouped["diameter_mean"].transform(
                lambda x: 4 / 3 * np.pi * (x / 2.0) ** 3 * density
            )  # kg
            df["mass_std"] = grouped["diameter_std"].transform(
                lambda x: 4 / 3 * np.pi * (x / 2.0) ** 3 * density
            )  # kg
            # df["mass_min"] = grouped["diameter_mean"].transform(lambda x: 4 / 3 * np.pi * ((x - x.min / 2.0)) ** 3 * density) # kg
            # df["mass_max"] = grouped["diameter_mean"].transform(lambda x: 4 / 3 * np.pi * ((x + x.max / 2.0)) ** 3 * density) # kg

            # update dataframe
            cols_to_update = [
                "mass",
                "mass_mean",
                "mass_std",
            ]  # , "mass_min", "mass_max"]
            df[cols_to_update] = df[cols_to_update]  # .fillna(0.0)
            dataframe[cols_to_update] = df[cols_to_update].fillna(0.0)

        return dataframe

    # @staticmethod
    def Velocity(dataframe: pd.DataFrame, time_interval: float) -> pd.DataFrame:
        """Compute velocity in px/s for each particle at each frame"""

        df = dataframe.copy()

        required_cols = {"frame", "label", "time", "x", "y"}
        missing = required_cols - set(df.columns)
        if missing:
            msg = f"Missing requiered column(s) : {missing}"
            raise KeyError(msg)

        dataframe = dataframe.sort_values(by=["label", "frame"]).reset_index(drop=True)

        # initialise columns
        n = len(df)
        df["displacement"] = np.zeros(n)
        df["dx"] = np.zeros(n)
        df["dy"] = np.zeros(n)
        df["velocity"] = np.zeros(n)
        df["vx"] = np.zeros(n)
        df["vy"] = np.zeros(n)
        df["angle"] = np.zeros(n)
        df["dt"] = df.groupby("label")["time"].diff(periods=1).fillna(0.0)

        df["dt"] = [time_interval] * n

        if not all(
            col in dataframe.columns
            for col in [
                "dx",
                "dy",
                "displacement",
                "angle",
                "vx",
                "vy",
                "velocity",
                "dt",
            ]
        ):
            # compute dx, dy and displacement
            df["dx"] = df.groupby("label")["x"].diff(periods=1).fillna(0.0)  # [px]
            df["dy"] = df.groupby("label")["y"].diff(periods=1).fillna(0.0)  # [px]

            df["displacement"] = (
                df.groupby("label")[["dx", "dy"]]
                .apply(lambda g: np.hypot(g["dx"], g["dy"]))
                .reset_index(level=0, drop=True)
            )  # [px]
            # df["displacement"] = np.hypot(df.groupby("label")["dx"], df.groupby("label")["dy"]) # [px]

            # compute angle
            # df["angle"] = np.arctan2(df.groupby("label")["dy"], df.groupby("label")["dx"]) # [rad]
            df["angle"] = (
                df.groupby("label")[["dx", "dy"]]
                .apply(lambda g: np.arctan2(g["dy"], g["dx"]))
                .reset_index(level=0, drop=True)
            )  # [rad]

            # compute velocity
            df["vx"] = (
                df.groupby("label")[["dx", "dt"]]
                .apply(lambda g: g["dx"] / g["dt"])
                .reset_index(level=0, drop=True)
            )  # [px/s]
            df["vy"] = (
                df.groupby("label")[["dy", "dt"]]
                .apply(lambda g: g["dy"] / g["dt"])
                .reset_index(level=0, drop=True)
            )  # [px/s]
            df["velocity"] = (
                df.groupby("label")[["displacement", "dt"]]
                .apply(lambda g: g["displacement"] / g["dt"])
                .reset_index(level=0, drop=True)
            )  # [px/s]

            # df["vx"] = df.groupby("label")["dx"] / df.groupby("label")["dt"] # [px/s]
            # df["vy"] = df.groupby("label")["dy"] / df.groupby("label")["dt"] # [px/s]
            # df["velocity"] = df.groupby("label")["displacement"] / df.groupby("label")["dt"] # [px/s]

            # update dataframe
            cols_to_update = [
                "dx",
                "dy",
                "displacement",
                "angle",
                "vx",
                "vy",
                "velocity",
                "dt",
            ]
            dataframe[cols_to_update] = df[cols_to_update].fillna(0.0)

        # print(n)
        # print(len(dataframe["velocity"]), len(dataframe["velocity"]>0), len(dataframe["velocity"]>0)/len(dataframe["velocity"])*100)

        # _, axs = plt.subplots(1, 2)
        # counts, bins = np.histogram(dataframe["displacement"], bins=100)
        # axs[0].hist(counts, bins)
        # counts, bins = np.histogram(dataframe["velocity"], bins=100)
        # axs[1].stairs(counts, bins)
        # plt.show()

        return dataframe

    # @staticmethod
    def Momentum(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Compute momentum in kg * px/s for each particle"""

        df = dataframe.copy()

        required_cols = {"velocity", "mass", "mass_mean"}  # , "mass_min", "mass_max"}
        missing = required_cols - set(df.columns)
        if missing:
            msg = f"Missing requiered column(s) : {missing}"
            raise KeyError(msg)

        if not all(
            col in dataframe.columns
            for col in ["momentum", "momentum_mean", "momentum_min", "momentum_max"]
        ):
            df["momentum"] = (
                df.groupby("label")[["mass", "velocity"]]
                .apply(lambda g: g["mass"] * g["velocity"])
                .reset_index(level=0, drop=True)
            )  # kg * m / s
            df["momentum_mean"] = (
                df.groupby("label")[["mass_mean", "velocity"]]
                .apply(lambda g: g["mass_mean"] * g["velocity"])
                .reset_index(level=0, drop=True)
            )  # kg * m / s

            # df["momentum"] = df.groupby("label")["mass"] * df.grouby("label")["velocity"] # kg * m / s
            # df["momentum_mean"] = df.groupby("label")["mass_mean"] * df.groupby("label")["velocity"] # kg * m / s
            # df["momentum_min"] = df.groupby("label")["mass_min"] * df.groupby("label")["velocity"]  # kg * m / s
            # df["momentum_max"] = df.groupby("label")["mass_max"] * df.groupby("label")["velocity"]  # kg * m / s

            cols_to_update = [
                "momentum",
                "momentum_mean",
            ]  # , "momentum_min", "momentum_max"]
            dataframe[cols_to_update] = df[cols_to_update].fillna(0.0)

        return dataframe

    # @staticmethod
    def Acceleration(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Compute acceleration in px/s**2 for each particle"""

        df = dataframe.copy()

        required_cols = {"vx", "vy", "velocity", "time", "dt"}
        missing = required_cols - set(df.columns)
        if missing:
            msg = f"Missing read_csv(path))"
            # df = analyser_class.Collision(dataframe=df)
            # df.to_csv(final_path)
            # queue.put(("tick", task_id, equiered column(s) : {missing}"
            raise KeyError(msg)

        if not all(col in dataframe.columns for col in ["ax", "ay", "acceleration"]):
            df["ax"] = (
                df.groupby("label")[["vx", "dt"]]
                .apply(lambda g: g["vx"] / g["dt"])
                .reset_index(level=0, drop=True)
            )  # px / s^2
            df["ay"] = (
                df.groupby("label")[["vy", "dt"]]
                .apply(lambda g: g["vy"] / g["dt"])
                .reset_index(level=0, drop=True)
            )  # px / s^2
            df["acceleration"] = (
                df.groupby("label")[["velocity", "dt"]]
                .apply(lambda g: g["velocity"] / g["dt"])
                .reset_index(level=0, drop=True)
            )  # px / s^2

            # df["ax"] = df["vx"].diff(periods=1) / df["dt"] # px / s^2
            # df["ay"] = df["vy"].diff(periods=1) / df["dt"] # px / s^2
            # df["acceleration"] = df["velocity"].diff(periods=1) / df["dt"] # px / s^2

            cols_to_update = ["ax", "ay", "acceleration"]
            dataframe[cols_to_update] = df[cols_to_update].fillna(0.0)

        return dataframe

    # @staticmethod
    def KineticEnergy(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Compute kinetic energy in kg * (px/s)**2 for each particle"""

        df = dataframe.copy()

        required_cols = {"velocity", "mass", "mass_mean"}  # , "mass_min", "mass_max"}
        missing = required_cols - set(df.columns)
        if missing:
            msg = f"Missing requiered column(s) : {missing}"
            raise KeyError(msg)

        if not all(col in dataframe.columns for col in ["kinetic", "kinetic_mean"]):
            df["kinetic"] = (
                df.groupby("label")[["mass", "velocity"]]
                .apply(lambda g: 1 / 2 * g["mass"] * g["velocity"] ** 2)
                .reset_index(level=0, drop=True)
            )  # kg * px^2 / s^2
            df["kinetic_mean"] = (
                df.groupby("label")[["mass_mean", "velocity"]]
                .apply(lambda g: 1 / 2 * g["mass_mean"] * g["velocity"] ** 2)
                .reset_index(level=0, drop=True)
            )  # kg * px^2 / s^2

            # df["kinetic"] = 1 / 2 * df["mass"] * (df["velocity"] ** 2) # kg * px^2 / s^2
            # df["kinetic_mean"] = 1 / 2 * df["mass_mean"] * (df["velocity"] ** 2) # kg * px^2 / s^2
            # df["kinetic_min"] = 1 / 2 * df["mass_min"] * (df["velocity"] ** 2) # kg * px^2 / s^2
            # df["kinetic_max"] = 1 / 2 * df["mass_max"] * (df["velocity"] ** 2) # kg * px^2 / s^2

            cols_to_update = [
                "kinetic",
                "kinetic_mean",
            ]  # , "kinetic_min", "kinetic_max"]
            dataframe[cols_to_update] = df[cols_to_update].fillna(0.0)

        return dataframe
