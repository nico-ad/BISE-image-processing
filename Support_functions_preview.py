
from PyQt5 import uic
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QThread, QTimer
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QInputDialog, QAction
from PyQt5.QtWidgets import QTabWidget, QPushButton, QDial, QCheckBox, QLineEdit, QProgressBar, QComboBox, QLabel, QSpinBox
from PyQt5.QtWidgets import (
    QTableWidget,
    QTableWidgetItem,
    QDialog,
    QFileDialog,
    QTableView,
    QListView,
    QTreeView,
    QAbstractItemView,
    QSizePolicy,
    QToolBar,
    QDoubleSpinBox,
    QLineEdit,
    QColorDialog,
    QFrame,
)
from PyQt5.QtWidgets import (
    QStyledItemDelegate,
    QVBoxLayout,
    QHBoxLayout,
)
from PyQt5.QtGui import QIcon, QFont, QPixmap, QImage, QPainter, QColor

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
from scipy.ndimage import (
    label, distance_transform_edt, gaussian_filter, binary_fill_holes,
)
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
from skimage.measure import regionprops
from scipy.integrate import quad

from skimage import filters, morphology, measure, segmentation
from skimage.segmentation import watershed, clear_border

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.colors as mcolors
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import cv2

from tkinter import Tcl

class ParticleAnalyser:
    """ Image analysis from particles image """
    
    def __init__(self, parent=None):
        self.parent = parent

    def _apply_image_modifications(self, filename: str, params: dict, chkbx: dict):
        
        """
        Apply modification on image and extract particles
        
        Args:
            filename (str): image path
            params (dict): dictionnary of parameters
        
        Return:
            pd.DataFrame: particles properties
            np.ndarray: original image modified
            np.ndarray: segmented image
        """
        
        do_plot = False
        
        circ_thresh = params["circularity thresh"]
        is_subpixel = params["enable subpixel detection"]
        invert_grayscale = params["invert grayscale"]
        h_maxima_value = params["h maxima"]
        
        img_raw = Image.open(filename)
        img_raw = np.array(img_raw)
        if img_raw.ndim == 2:
            img_raw = np.stack([img_raw]*3, axis=-1)
        
        if do_plot:
            _, ax = plt.subplots()
            ax.imshow(img_raw, cmap="gray")
            ax.set_title("RAW")
            plt.show()
        
        img = img_raw
        if invert_grayscale:
            img_gray = ImageOps.invert(Image.fromarray(img).convert("L"))
        else: img_gray = Image.fromarray(img_raw).convert("L")
        img_gray = np.array(img_gray, dtype=np.uint8)
        if do_plot:
            _, ax = plt.subplots()
            ax.imshow(img_gray, cmap="gray")
            ax.set_title("GRAY")
            plt.show()
        
        # compute bilateral filtering
        filter = 0.1
        img_gray = self._bilateral_filtering(img_gray, filter, filter)
        # img_gray = np.array(img_gray, dtype=np.uint8)
        if do_plot:
            _, ax = plt.subplots()
            ax.imshow(img_gray, cmap="gray")
            ax.set_title("BILATERAL")
            plt.show()
        
        # binarisation
        binary = self._threshold_image(np.array(img_gray, dtype=np.uint8), chkbx)
        if do_plot:
            _, ax = plt.subplots()
            ax.imshow(binary, cmap="gray")
            ax.set_title("OTSU")
            plt.show()
        
        binary = self._postprocess_binary(binary, params, chkbx)
        if do_plot:
            _, ax = plt.subplots()
            ax.imshow(binary, cmap="gray")
            ax.set_title("BINARY")
            plt.show()
        
        # watershed segmentation
        distance = distance_transform_edt(binary)
        if do_plot:
            _, ax = plt.subplots()
            ax.imshow(distance, cmap="gray")
            ax.set_title("DISTANCE")
            plt.show()
        
        mask = morphology.h_maxima(distance, h=h_maxima_value)
        if do_plot:
            _, ax = plt.subplots()
            ax.imshow(mask, cmap="gray")
            ax.set_title("MARKERS")
            plt.show()
        
        markers = measure.label(mask)
        if do_plot:
            _, ax = plt.subplots()
            ax.imshow(markers, cmap="gray")
            ax.set_title("Markers")
            plt.show()
        
        labeled_image = watershed(-distance, markers, mask=binary)
        if do_plot:
            _, ax = plt.subplots()
            ax.imshow(labeled_image, cmap="gray")
            ax.set_title("WATERSHED")
            plt.show()
        
        all_data = self._extract_particle_data(labeled_image, img, circ_thresh, filename.parent, False)
        
        # visual identification
        if chkbx["identify_part"]:
            labeled_image = self._draw_particle_contours(labeled_image, img_raw)
        
        return pd.DataFrame(all_data), img, labeled_image
    
    # ----------
    # Helpers
    # ----------
    
    def _bilateral_filtering(self, img, sigma_s, sigma_r):
        """ bilateral filtering"""
        
        img = img.astype(np.float64)
        
        # create grid of spatial coordinates
        radius = int(3 * sigma_s)
        y, x = np.mgrid[-radius:radius+1, -radius:radius+1]
        
        # spatial gaussian weights
        spatial_gaussian = np.exp(-(x**2 + y**2) / (2 * sigma_s**2))
        output = np.zeros_like(img)
        normalizer = np.zeros_like(img)
        for i in range(radius, img.shape[0] - radius):
            for j in range(radius, img.shape[1] - radius):
                
                # extract local patch
                patch = img[
                    i-radius:i+radius+1,
                    j-radius:j+radius+1,
                    ]
                
                # compute intensity gaussian weights
                intensity_gaussian = np.exp(-((patch-img[i, j])**2) / (2 * sigma_r**2))
                weights = spatial_gaussian * intensity_gaussian
                
                # normalize
                weights /= weights.sum()
                
                # compute filtered pixel value
                output[i, j] = np.sum(patch * weights)
                normalizer[i, j] = weights.sum()
        return output
    
    def _threshold_image(self, img_gray, chkbx):
        """ Automatic or manual threshold """
        # if getattr(self, "thresh_method", None) and self.thresh_method_isChecked():
        if chkbx["threshold"]:
            thresh = filters.threshold_otsu(img_gray)
            return img_gray > thresh
        else:
            return img_gray > img_gray/2
        
    def _postprocess_binary(self, binary, params, chkbx):
        """ Apply binary cleaning : holes, border and small objects"""
        if chkbx["fill_holes"]:
            binary = binary_fill_holes(binary)
        if chkbx["remove_small"]:
            min_size = int(params.get("small objects"))
            binary = morphology.remove_small_objects(binary, min_size)
        if chkbx["clear_border"]:
            binary = clear_border(binary)
        return binary
    
    def _extract_particle_data(self, labeled_image, intensity_image, circ_thresh,
                               img_name, is_subpixel) -> pd.DataFrame:
        """ Extract particles properties from segmented image """
        
        self.start_frame = 0
        
        all_data = []
        
        # if no circularity
        if circ_thresh is None:
            mean_circularity = np.mean(
                prop.circularity for prop in regionprops(labeled_image)
            )
            circ_thresh = (
                mean_circularity * 0.8,
                mean_circularity * 1.2,
            )
            
        # extract particles properties
        for prop in regionprops(labeled_image, intensity_image=intensity_image):
            if prop.perimeter == 0 or prop.area == 0:
                continue
            
            # compute circularity
            circularity = prop.axis_minor_length / prop.axis_major_length
            bb = self._convert_coordinates(*prop.bbox)
            
            if (bb[1] < 10000) and (bb[3] < 10000):

                mask = (labeled_image == prop.label).astype(np.float16)[
                    bb[0] : bb[0] + bb[1], bb[2] : bb[2] + bb[3]
                ]
                if mask.size == 0:
                    continue
                
                if self.start_frame is None:
                    curr_frame = int(str(img_name).split(".")[2].split("-")[0])
                else:
                    curr_frame = 0

                if (
                    circ_thresh[0] <= circularity <= circ_thresh[1]
                    ):

                    centroid = self._compute_centroid(prop, is_subpixel, intensity_image)
                    x, y = int(centroid[1]), int(centroid[0])
                    diameter = 2 * np.sqrt(prop.area / np.pi)
                    
                    # time = self.time_interval or 0.0
                    time = 0.025

                    all_data.append(
                        {
                            "frame": curr_frame,
                            "main_path": Path(img_name).parent,
                            "name": Path(img_name).name,
                            "time": time,  # [s]
                            "bound_box": bb,
                            "coords_pixels": prop.coords,  # coords of pixels that define particle
                            "x": x, # [px]
                            "y": y, # [px]
                            # "x_subpixel": x_subpixel, # [px]
                            # "y_subpixel": y_subpixel, # [px]
                            "diameter": diameter, # [px]
                            "perimeter": prop.perimeter, # [px]
                            "area": prop.area, # [px^2]
                            "circularity": circularity,
                            "intensity_max": prop.intensity_max,
                            "intensity_min": prop.intensity_min,
                            "intensity_mean": prop.intensity_mean,
                            "intensity_std": prop.intensity_std,
                            "equivalent_diameter_area": prop.equivalent_diameter_area, # [px]
                            "axis_major_length": prop.axis_major_length, # [px]
                            "axis_minor_length": prop.axis_minor_length, # [px]
                        }
                    )

                    if self.start_frame is None:
                        curr_frame += 1
            
        return pd.DataFrame(all_data)
        
    def _convert_coordinates(self, min_row, min_col, max_row, max_col):
        """ Convert particle pixel to bound box"""
        
        x = min_row
        y = min_col
        height = max_row - min_row
        width = max_col - min_col
        return [x, height, y, width]
        
    def _compute_centroid(self, prop, subpixel, intensity_image):
        """ Compute refined center """
        
        centroid = np.array([prop.centroid[1], prop.centroid[0]])
        
        if not subpixel:
            return int(centroid[1]), int(centroid[0])
        
        # bounding box
        min_row, min_col, max_row, max_col = prop.bbox
        region = intensity_image[min_row:max_row, min_col:max_col]
        
        # grid for 
        
        y_grid, x_grid = np.mgrid[min_row:max_row, min_col:max_col]
        
        # 2D gaussian function
        def gaussian_2d(xy, x0, y0, A, sigma_x, sigma_y, offset):
            x, y = xy
            g = offset + A * np.exp(-(((x-x0)**2) / (2*sigma_x**2) + (y-y0) / (2*sigma_y**2)))
            return g.ravel()
        
        # guess initial
        p0 = (centroid[0], centroid[1], region.max(), 1, 1, region.min())
        
        try:
            popt, _ = curve_fit(gaussian_2d, (x_grid.ravel(), y_grid.ravel(), region.ravel()), p0=p0)
            refined_x, refined_y = popt[0], popt[1]
            
            # check limits
            if not (min_col <= refined_x < max_col) or not (min_row <= refined_y < max_row):
                refined_x, refined_y = centroid
        except RuntimeError:
            refined_x, refined_y = centroid
        
        return float(refined_x), float(refined_y)
    
    def _draw_particle_contours(self, labeled_image: np.ndarray, img_raw: np.ndarray):
        """ Draw contours particles on image"""
        contours = measure.find_contours(labeled_image, level=0.5)
        for contour in contours:
            for y, x in contour:
                y, x = int(round(y)), int(round(x))
                if 0 <= y < img_raw.shape[0] and 0 <= x < img_raw.shape[1]:
                    img_raw[y, x] = [255, 0, 0]
        return img_raw

class ImageViewer:
    def __init__(self, parent, pixel_size):
        
        self.parent = parent
        self.layout = self.parent.layout()

        self.pixel_size = pixel_size
        
        self.canvases = []
        self.toolbars = []
        
        self.dict_fontsize = {
            "label" : 18,
            "ticks" : 18,
            "legend": 16,
            "subplots": {
                "left": 0.075,
                "bottom": 0.090,
                "right": 0.930,
                "top": 0.97,
                "wspace": 0.0,
                "hspace": 0.0,
            }
        }
        
    # display coordinates on mouse
    def _on_move(self, event, ax, img, status_label):
        
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            status_label.setText("")
            return
        
        x_px = int(round(event.xdata))
        y_px = int(round(event.ydata))
        
        if 0 <= x_px < img.shape[1] and 0 <= y_px < img.shape[0]:
            val = img[y_px, x_px]
            status_label.setText(
                f"x={event.xdata:.2f} mm   "
                f"y={event.ydata:.2f} mm   "
                f"value={val}",
            )
    
    # def _on_move(self, event):
    #     """ Display coordinates of mouse on toolbar """
        
    #     if event.xdata is not None and event.ydata is not None:
    #         self.status_label.setText(
    #             f"x={event.xdata*self.pixel_size:.2f} mm   "
    #             f"y={event.ydata*self.pixel_size:.2f} mm   "
    #         )

    # zoom on scroll
    def on_scroll(self, event, ax, canvas):
        base_scale = 1.2
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        
        xdata = event.xdata
        ydata = event.ydata
        
        if xdata is None or ydata is None:
            return
        
        scale_factor = 1 / base_scale if event.button == "up" else base_scale
        
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        
        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
        
        ax.set_xlim(xdata - new_width * (1 - relx), xdata + new_width * relx)
        ax.set_ylim(ydata - new_height * (1 - rely), ydata + new_height * rely)
        canvas.draw_idle()
        
    def clear(self):
        for canvas in self.canvases:
            self.layout.removeWidget(canvas)
            canvas.setParent(None)
            canvas.deleteLater()
        
        for toolbar in self.toolbars:
            self.layout.removeWidget(toolbar)
            toolbar.setParent(None)
            toolbar.deleteLater()
        
        self.canvases.clear()
        self.toolbars.clear()

    def add_img(
        self,
        img: np.ndarray = None,
        data: pd.DataFrame = None,
    ):
        
        fig = Figure()
        ax = fig.add_subplot(111)
        
        ax.imshow(img, cmap="gray", aspect="auto")
        ax.set_aspect("equal", adjustable="box")

        ax.scatter(data["x"], data["y"], color="tab:orange")
        
        x_ticks = ax.get_xticks()[1:-1]
        y_ticks = ax.get_yticks()[1:-1]
        
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        
        ax.set_xlabel("X [mm]", fontsize=self.dict_fontsize["label"])
        ax.set_ylabel("Y [mm]", fontsize=self.dict_fontsize["label"])
        
        ax.set_xticklabels([f"{x_tick*self.pixel_size:.2f}" for x_tick in x_ticks], fontsize=self.dict_fontsize["ticks"])
        ax.set_yticklabels([f"{y_tick*self.pixel_size:.2f}" for y_tick in y_ticks], fontsize=self.dict_fontsize["ticks"])
        
        fig.tight_layout()
        
        canvas = FigureCanvas(fig)
        canvas.setMinimumSize(0, 0)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        canvas.updateGeometry()
        toolbar = NavigationToolbar(canvas, self.parent)
        
        status_label = QLabel()
        toolbar.addWidget(status_label)
        
        canvas.mpl_connect("motion_notify_event",
                           lambda event: self._on_move(event, ax, img, status_label),
        )
        # canvas.mpl_connect("scroll_event",
        #                    lambda event: self.on_scroll(event, ax, canvas),
        # )
        
        self.layout.addWidget(canvas)
        self.layout.addWidget(toolbar)
        self.canvases.append(canvas)
        self.toolbars.append(toolbar)
        
    def add_density(
        self,
        img: np.ndarray = None,
    ):
        
        fig = Figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        
        ax.imshow(img, cmap="gray", aspect="auto")
        
        x_ticks = ax.get_xticks()[1:-1]
        y_ticks = ax.get_yticks()[1:-1]
        
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        
        ax.set_xlabel("X [mm]", fontsize=self.dict_fontsize["label"])
        ax.set_ylabel("Y [mm]", fontsize=self.dict_fontsize["label"])
        
        ax.set_xticklabels([f"{x_tick*self.pixel_size:.2f}" for x_tick in x_ticks], fontsize=self.dict_fontsize["ticks"])
        ax.set_yticklabels([f"{y_tick*self.pixel_size:.2f}" for y_tick in y_ticks], fontsize=self.dict_fontsize["ticks"])
        
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, self.parent)
        status_label = QLabel()
        toolbar.addWidget(status_label)
        
        # canvas.mpl_connect("motion_notify_event",
        #                    lambda event: self.on_move(event, ax, img, status_label),
        # )
        # canvas.mpl_connect("scroll_event",
        #                    lambda event: self.on_scroll(event, ax, canvas),
        # )
        
        self.layout.addWidget(toolbar)
        self.layout.addWidget(canvas)
        self.canvases.append(canvas)
        self.toolbars.append(toolbar)

    # def add_histogram(
    #     self,
    #     df: pd.DataFrame,
    #     ):
        
    #     diameters = df["diameter"].to_numpy()
    #     hist, bins = np.histogram(diameters)
        
    #     fig = Figure()
    #     ax = fig.add_subplot(111)
        
    #     self._draw_histogram(ax, diameters)
        
    #     canvas = FigureCanvas(fig)
    #     toolbar = NavigationToolbar(canvas, self.parent)
        
    #     action_customize = QAction("Customize histogram", self.parent)
    #     action_customize.triggered.connect(lambda: self._customize_histogram(ax, diameters, canvas))
    #     toolbar.addAction(action_customize)
        
    #     self.layout.addWidget(toolbar)
    #     self.layout.addWidget(canvas)
    #     self.canvases.append(canvas)
    #     self.toolbars.append(toolbar)
    
    # def _draw_histogram(self, ax, diameters, bins=50, density=False, hist_style="bar"):
    #     """ Draw histogram """
        
    #     ax.clear()
    #     hist, bin_edges = np.histogram(diameters, bins=bins, density=density)
        
    #     ax.stairs(
    #         values=hist, edges=bin_edges,
    #         fill=True, color="tab:blue", edgecolor="black", linewidth=1.5, alpha=0.7
    #     )
        
    #     x_ticks = ax.get_xticks()[:-1]
    #     y_ticks = ax.get_yticks()[:-1]
        
    #     ax.set_xticks(x_ticks)
    #     ax.set_yticks(y_ticks)
        
    #     ax.set_xticklabels([f"{x_tick*self.pixel_size:.2f}" for x_tick in x_ticks], fontsize=self.dict_fontsize["ticks"])
    #     ax.set_yticklabels([f"{y_tick:.2f}" for y_tick in y_ticks], fontsize=self.dict_fontsize["ticks"])
    #     ax.set_xlabel("Diameters [mm]", fontsize=self.dict_fontsize["label"])
    #     ax.set_ylabel("Counts", fontsize=self.dict_fontsize["label"])
    
    # def _customize_histogram(self, ax, diameters, canvas):
    #     """ Histogram customization """
        
    #     bins, ok = QInputDialog.getInt(self.parent, "Histogram Bins", "Number of bins", value=50, min=1, max=200)
    #     if ok:
    #         self._draw_histogram(ax, diameters, bins=bins)
    #         canvas.draw_idle()

class ViewerWorker(QObject):
    
    finished = pyqtSignal(np.ndarray, pd.DataFrame)
    
    def __init__(self, image, data, rotation_value):
        super().__init__()
        
        self.image = image
        self.data = data
        self.rotation_value = rotation_value
        
    def run(self):
        
        img_rot = self._apply_rotation(self.image, self.rotation_value)
        self.finished.emit(img_rot, self.data)
    
    def _apply_rotation(self, img, rotation):
        """ Rotate image """
        
        rotations = {"NONE": 0, "ROTATE_90": 1, "ROTATE_180": 2, "ROTATE_270": 3}
        k = rotations.get(rotation, 0)
        return np.rot90(img, k)

class Spinner(QWidget):
    
    def __init__(self, parent, radius=12, line_length=6, line_width=3, speed=80):
        super().__init__(parent)
            
        self.radius = radius
        self.line_length = line_length
        self.line_width = line_width
        self.speed = speed
        self.angle = 0
        
        self.setFixedSize((radius + line_length) * 2, (radius + line_length) * 2)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._rotate)
        self.hide()
    
    def start(self):
        self.show()
        self.timer.start(self.speed)
    
    def stop(self):
        self.timer.stop()
        self.hide()
        
    def _rotate(self):
        self.angle = (self.angle + 30) % 360
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        for i in range(12):
            color = QColor(100, 100, 100)
            color.setAlpha(int((i + 1) / 12) * 255)
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            
            painter.save()
            painter.translate(self.width() / 2, self.height() / 2)
            painter.rotate(self.angle + i * 30)
            painter.drawRoundedRect(
                int(self.radius), int(-self.line_width / 2),
                int(self.line_length), int(self.line_width),
                self.line_width / 2, self.line_width / 2
            )
            painter.restore()

class LatexLabel(QWidget):
    def __init__(self, latex_string, fontsize=14, parent=None):
        super().__init__(parent)
        
        self.figure = Figure(figsize=(3, 1))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        self.ax.axis("off")
        self.ax.text(
            0.5, 0.5,
            latex_string,
            ha="center", va="center",
            fontsize=fontsize,
        )
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

class HistogramViewer(QWidget):
    
    def __init__(self, parent=None, pixel_size=float(1)):
        super().__init__(parent)
        
        self.uncertainties = Uncertainties
        
        self.parent = parent
        
        self.pixel_size = pixel_size
        
        self.data = None
        
        self.default_config = {
            "bins": 50,
            "density": False,
            "color": "tab:blue",
            "edgecolor": "black",
            "alpha": 0.7,
            "fill": True,
            "xlim_min": None,
            "xlim_max": None,
            "n_x_ticks": 5,
            "ylim_min": None,
            "ylim_max": None,
            "n_y_ticks": 5,
            "xlog": True,
            "fit": True,
            "fit_params": (None, None),
            "uncertainties_params" : (None, None),
        }
        
        self.dict_fontsize = {
            "label" : 18,
            "ticks" : 18,
            "legend": 16,
            "subplots": {
                "left": 0.075,
                "bottom": 0.090,
                "right": 0.930,
                "top": 0.97,
                "wspace": 0.0,
                "hspace": 0.0,
            }
        }
        
        self.hist_config = self.default_config.copy()
        
        self.layout = QVBoxLayout(self)
        self._create_figure()
        
        h_layout = QHBoxLayout()
        
        self.toolbar = NavigationToolbar(self.canvas, self)
        h_layout.addWidget(self.toolbar)
        
        self.custom_btn = QPushButton("Customize histogram")
        self.custom_btn.clicked.connect(self._open_hist_params)
        h_layout.addWidget(self.custom_btn)
        
        self.layout.addLayout(h_layout)
        
    def _create_figure(self):
        """ Create figure """
        
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)
        
    def _open_hist_params(self):
        """ Open dialog box to modify histogram parameters """
        
        if self.data is None:
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Customize Histogram")
        layout = QVBoxLayout(dialog)
        
        hist_parameters_layout = QHBoxLayout()
        
        label = QLabel("Histogram parameters")
        label.setAlignment(Qt.AlignLeft)
        font = label.font()
        font.setBold(True)
        label.setFont(font)
        layout.addWidget(label)
        
        # bins
        hist_parameters_layout.addWidget(QLabel("Number of bins : "))
        bins_spin = QSpinBox()
        bins_spin.setRange(1, 200)
        bins_spin.setValue(self.hist_config["bins"])
        hist_parameters_layout.addWidget(bins_spin)
        bins_spin.valueChanged.connect(lambda v: self._update_hist_config("bins", v))
        
        # density
        density_cb = QCheckBox("Density")
        density_cb.setChecked(self.hist_config["density"])
        hist_parameters_layout.addWidget(density_cb)
        density_cb.stateChanged.connect(lambda state: self._update_hist_config("density", state==2))
        
        # log mode
        xlog_cb = QCheckBox("Log X-axis")
        xlog_cb.setChecked(self.hist_config["xlog"])
        xlog_cb.toggled.connect(lambda v: self._update_hist_config("xlog", v))
        hist_parameters_layout.addWidget(xlog_cb)
        
        # fit function
        fit_cb = QCheckBox("Fit")
        fit_cb.setChecked(self.hist_config["fit"])
        fit_cb.toggled.connect(lambda v: self._update_hist_config("fit", v))
        hist_parameters_layout.addWidget(fit_cb)
        layout.addLayout(hist_parameters_layout)
        
        # label histogram style
        label = QLabel("Axes style")
        label.setAlignment(Qt.AlignLeft)
        font = label.font()
        font.setBold(True)
        label.setFont(font)
        layout.addWidget(label)
        
        # color
        self.color_btn = QPushButton("Select fill color")
        fill_color = self.hist_config["color"]
        fill_text_color = self._contrasting_text_color(fill_color)
        self.color_btn.setStyleSheet(f"QPushButton {{ background-color: {self.hist_config['color']}; color: {fill_text_color}  }}")
        layout.addWidget(self.color_btn)
        self.color_btn.clicked.connect(lambda _, b=self.color_btn: self._pick_color(b, "color"))
        
        # edges color
        self.edge_btn = QPushButton("Select edge color")
        edge_color = self.hist_config["color"]
        edge_color = self._contrasting_text_color(edge_color)
        self.edge_btn.setStyleSheet(f"QPushButton {{ background-color: {self.hist_config['edgecolor']}; color: {edge_color}  }}")
        layout.addWidget(self.edge_btn)
        self.edge_btn.clicked.connect(lambda _, b=self.edge_btn: self._pick_color(b, "edgecolor"))
        
        # alpha
        alpha_layout = QHBoxLayout()
        alpha_layout.addWidget(QLabel("Alpha : "))
        alpha_spin = QDoubleSpinBox()
        alpha_spin.setRange(0.0, 1.0)
        alpha_spin.setSingleStep(0.05)
        alpha_spin.setValue(self.hist_config["alpha"])
        alpha_layout.addWidget(alpha_spin)
        layout.addWidget(alpha_spin)
        alpha_spin.valueChanged.connect(lambda v: self._update_hist_config("alpha", v))
        
        # ----- label ax limits
        label = QLabel("Axes limits")
        label.setAlignment(Qt.AlignLeft)
        font = label.font()
        font.setBold(True)
        label.setFont(font)
        layout.addWidget(label)
        
        # x limits
        x_layout = QHBoxLayout()
        x_min_input = QLineEdit("")
        x_max_input = QLineEdit("")
        x_min_input.setText(str(self.hist_config["xlim_min"]) if self.hist_config["xlim_min"] is not None else "")
        x_max_input.setText(str(self.hist_config["xlim_max"]) if self.hist_config["xlim_max"] is not None else "")
        x_layout.addWidget(QLabel("X limits : "))
        x_layout.addWidget(x_min_input)
        x_layout.addWidget(x_max_input)
        x_min_input.editingFinished.connect(
            lambda: self._update_hist_config(
                "xlim_min", float(x_min_input.text()) if x_min_input.text() else None
                )
            )
        x_max_input.editingFinished.connect(
            lambda: self._update_hist_config(
                "xlim_max", float(x_max_input.text()) if x_max_input.text() else None
                )
            )
        xticks_spin = QSpinBox()
        xticks_spin.setRange(2, 50)
        xticks_spin.setValue(self.hist_config["n_x_ticks"])
        x_layout.addWidget(QLabel("Number of ticks : "))
        x_layout.addWidget(xticks_spin)
        xticks_spin.valueChanged.connect(lambda v: self._update_hist_config("n_x_ticks", v))
        layout.addLayout(x_layout)
        
        # y limits
        y_layout = QHBoxLayout()
        y_min_input = QLineEdit("")
        y_max_input = QLineEdit("")
        y_min_input.setText(str(self.hist_config["ylim_min"]) if self.hist_config["ylim_min"] is not None else "")
        y_max_input.setText(str(self.hist_config["ylim_max"]) if self.hist_config["ylim_max"] is not None else "")
        y_layout.addWidget(QLabel("Y limits : "))
        y_layout.addWidget(y_min_input)
        y_layout.addWidget(y_max_input)
        y_min_input.editingFinished.connect(
            lambda: self._update_hist_config(
                "ylim_min", float(y_min_input.text()) if y_min_input.text() else None
                )
            )
        y_max_input.editingFinished.connect(
            lambda: self._update_hist_config(
                "ylim_max", float(y_max_input.text()) if y_max_input.text() else None
                )
            )
        yticks_spin = QSpinBox()
        yticks_spin.setRange(2, 50)
        yticks_spin.setValue(self.hist_config["n_y_ticks"])
        y_layout.addWidget(QLabel("Number of ticks : "))
        y_layout.addWidget(yticks_spin)
        yticks_spin.valueChanged.connect(lambda v: self._update_hist_config("n_y_ticks", v ))
        layout.addLayout(y_layout)
        
        # ----- label fit parameters
        frame = QFrame()
        frame.setFrameShape(QFrame.Box)
        frame.setLineWidth(1)
        frame.setStyleSheet(
            "border: 1px solid black;"
            "border-radius: 5px;"
            "background-color: #fafafa;"
        )
        
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(10, 10, 10, 10)
        
        label = QLabel("Fit function parameters")
        label.setStyleSheet("border: none;")
        label.setAlignment(Qt.AlignLeft)
        font = label.font()
        font.setBold(True)
        # font.setFont(self.parent.font_content)
        label.setFont(font)
        frame_layout.addWidget(label)
        
        fit_function_layout = QHBoxLayout()
        fit_function_layout.setSpacing(15)
        
        fit_function_left_layout = QVBoxLayout()
        fit_function_left_layout.setSpacing(15)
        
        if self.hist_config["xlog"]:
            latex_widget_formula = LatexLabel(
                r"$\text{Fit}(d_p) = \dfrac{1}{d_p \, \sigma \, \sqrt{2 \, \pi}} "
                r"\, \exp \left( - \dfrac{(\ln(d_p) - \mu)^2}{2 \, \sigma^2} \right)$",
                fontsize=10,
                )
            latex_widget_mu = LatexLabel(
                r"$\mu_m = "f"{self.hist_config['fit_params'][0]:.2f}$",
                fontsize=10,
            )
            latex_widget_sigma = LatexLabel(
                r"$\sigma_g = "f"{self.hist_config['fit_params'][1]:.2f}$",
                fontsize=10,
            )
        else:
            latex_widget_formula = LatexLabel(
                r"$\text{Fit}(d_p) = \dfrac{1}{\sigma \, \sqrt{2 \, \pi}} "
                r"\, \exp \left( - \dfrac{(d_p - \mu)^2}{2 \, \sigma^2} \right)$ ",
                fontsize=10,
                )
            latex_widget_mu = LatexLabel(
                r"$\mu = "f"{self.hist_config['fit_params'][0]:.2f}$",
                fontsize=10,
            )
            latex_widget_sigma = LatexLabel(
                r"$\sigma = "f"{self.hist_config['fit_params'][1]:.2f}$",
                fontsize=10,
            )
        
        fit_function_left_layout.addWidget(latex_widget_formula)
        fit_function_left_layout.addWidget(latex_widget_mu)
        fit_function_left_layout.addWidget(latex_widget_sigma)
        
        fit_function_layout.addLayout(fit_function_left_layout)
        
        parameters_layout = QVBoxLayout()
        if self.hist_config["xlog"]:
            mu_m = np.exp(self.hist_config['fit_params'][0])
            latex_widget_mu = LatexLabel(
                r"$\mu_m = "f"{mu_m:.2f} (\\pm {self.hist_config['uncertainties_params'][0]:.2f}) \\, \\mu m$",
                fontsize=10,
                )
            # sigma = np.sqrt((np.exp(self.hist_config['fit_params'][1]**2) - 1) * np.exp(2 * self.hist_config['fit_params'][0] + self.hist_config['fit_params'][1]**2))
            sigma_g = np.exp(self.hist_config['fit_params'][1])
            latex_widget_sigma = LatexLabel(
                r"$\sigma_g = "f"{sigma_g:.2f} (\\pm {self.hist_config['uncertainties_params'][1]:.2f})$",
                fontsize=10,
                )
        else:
            latex_widget_mu = LatexLabel(
                r"$\mu = "f"{self.hist_config['fit_params'][0]:.2f}$",
                fontsize=10,
                )
            latex_widget_sigma = LatexLabel(
                r"$\sigma = "f"{self.hist_config['fit_params'][1]:.2f}$",
                fontsize=10,
                )
        latex_widget_integral = LatexLabel(
            r"$\int_\mathbb{R} \, Fit(d_p) \, dd_p= "f"{1.000:.2f}$",
            fontsize=10,
            )
        parameters_layout.addWidget(latex_widget_mu)
        parameters_layout.addWidget(latex_widget_sigma)
        parameters_layout.addWidget(latex_widget_integral)
        
        fit_function_layout.addLayout(parameters_layout)
        
        frame_layout.addLayout(fit_function_layout)
        
        layout.addWidget(frame)
        
        # reset
        btn_layout = QHBoxLayout()
        reset_btn = QPushButton("Reset")
        ok_btn = QPushButton("OK")
        btn_layout.addWidget(reset_btn)
        btn_layout.addWidget(ok_btn)
        layout.addLayout(btn_layout)
        reset_btn.clicked.connect(self._reset_defaults)
        ok_btn.clicked.connect(dialog.accept)
        
        if dialog.exec_():
            self._draw_histogram()
        
    def _create_toolbar(self):
        """ Create toolbar for customizing figure """
        
        self.toolbarQToolBar("Matplotlib toolbar", self)
        self.layout.addWidget(self.toolbar)
        
        customize_action = QAction("Customize histogram", self)
        customize_action.triggered.connect(self._open_hist_params)
        self.toolbar.addAction(customize_action)
    
    def _set_data(self, data: np.ndarray):
        """ Create histogram """
        
        self.data = data
        self._draw_histogram()
    
    def _pick_color(self, button, key):
        """ Select color from table """
        
        color = QColorDialog.getColor(QColor(self.hist_config[key]), self, "Select color")
        if color.isValid():
            hex_color = color.name()
            text_color = self._contrasting_text_color(hex_color)
            button.setStyleSheet(f"QPushButton {{background-color: {hex_color}; color: {text_color}; }}")
            self.hist_config[key] = hex_color
            self._draw_histogram()
    
    def _contrasting_text_color(self, color_value) -> str:
        """ Determine if color if closer to white or black """
        
        rgb = mcolors.to_rgb(color_value)
        r, g, b = [int(255 * x) for x in rgb]
        luminance = 0.299*r + 0.587*g + 0.114*b
        return "white" if luminance < 128 else "black"
    
    def _update_hist_config(self, key, value):
        """ Update histogram dict """
        
        self.hist_config[key] = value
        self._draw_histogram()
    
    def _update_limits_from_axes(self):
        """ Update limits from histogram axes """
        
        if self.ax is None:
            return
        
        self.hist_config["xlim_min"], self.hist_config["xlim_max"] = self.ax.get_xlim()
        self.hist_config["ylim_min"], self.hist_config["ylim_max"] = self.ax.get_ylim()
    
    def _draw_histogram(self):
        """ Draw histogram from histogram config """
        
        if self.data is None or len(self.data) == 0:
            self.ax.clear()
            self.canvas.draw_idle()
            return
        self.ax.clear()
        
        bins = self.hist_config["bins"]
        density = self.hist_config["density"]
        xlog = self.hist_config["xlog"]
        fit = self.hist_config["fit"]
        color = self.hist_config["color"]
        edge_color = self.hist_config["edgecolor"]
        alpha = self.hist_config["alpha"]
        fill = self.hist_config["fill"]
        
        xmin_conf = self.hist_config["xlim_min"]
        xmax_conf = self.hist_config["xlim_max"]
        ymin_conf = self.hist_config["ylim_min"]
        ymax_conf = self.hist_config["ylim_max"]
        
        n_x_ticks = self.hist_config["n_x_ticks"]
        n_y_ticks = self.hist_config["n_y_ticks"]
        
        # compute histogram
        hist_kwargs = {}
        scaled_data = self.data * self.pixel_size * 1000
        
        if self.hist_config["bins"] is not None:
            hist_kwargs["bins"] = self.hist_config["bins"]
        
        if self.hist_config["density"] is not None:
            hist_kwargs["density"] = self.hist_config["density"]
        
        hist, bin_edges = np.histogram(scaled_data, **hist_kwargs)
        # num_peaks, _ = self._count_hist_peaks(hist)
        # print(num_peaks)
        
        # display histogram
        stairs_kwargs = {
            "values": hist,
            "edges": bin_edges,
            }
        for key in ["fill", "color", "edgecolor", "alpha"]:
            value = self.hist_config[key]
            if value is not None:
                if isinstance(value, (float, int)):
                    value = round(value, 2)
                stairs_kwargs[key] = value
                
        self.ax.stairs(
            **stairs_kwargs,
            linewidth=1.5,
            label="Particle sizing")
        
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # set linear or log scale
        self.ax.set_xscale("log" if xlog else "linear")
        
        # set x ticks
        if xmin_conf is not None and xmax_conf is not None:
            self.ax.set_xlim(xmin_conf, xmax_conf)
            
        xmin, xmax = self.ax.get_xlim()
        
        x_ticks = None
        if n_x_ticks is not None and n_x_ticks > 1:
            if xlog:
                if xmin > 0 and xmax > 0:
                    x_ticks = np.logspace(np.log10(xmin), np.log10(xmax), n_x_ticks)
                else:
                    x_ticks = None
            else:
                x_ticks = np.linspace(xmin, xmax, n_x_ticks)
            x_ticks = [self._nearest_ten(x) for x in x_ticks]
            
        self.ax.set_xticks(x_ticks)
        
        # set y ticks
        if ymin_conf is not None and ymax_conf is not None:
            self.ax.set_ylim(ymin_conf, ymax_conf)
            
        ymin, ymax = self.ax.get_ylim()
        
        y_ticks = None
        if n_y_ticks is not None and n_y_ticks > 1:
            y_ticks = np.linspace(ymin, ymax, n_y_ticks)
            if not density:
                y_ticks = [self._nearest_ten(y) for y in y_ticks]
            self.ax.set_yticks(y_ticks)
        else:
            y_ticks = np.linspace(ymin, ymax, len(self.ax.get_yticks()))
            self.ax.set_yticks(y_ticks)
        
        if fit:
            fitting_function = FitFunction()
            func_base = fitting_function._get_fitting_function()
            if xlog:
                x_values = np.logspace(np.log10(xmin), np.log10(xmax), 500)
                func_base = func_base["log_normal"]
                hist, bin_edges = np.histogram(scaled_data, **hist_kwargs)
                centered_bins = (bin_edges[:-1] + bin_edges[1:]) / 2
                popt, _, fit_func, r2 = fitting_function._fit_curve(centered_bins, hist, func_base=func_base)
            
            else:
                x_values = np.linspace(xmin, xmax, 500)
                func_base = func_base["gaussian"]
                hist, bin_edges = np.histogram(scaled_data, **hist_kwargs)
                centered_bins = (bin_edges[:-1] + bin_edges[1:]) / 2
                popt, _, fit_func, r2 = fitting_function._fit_curve(centered_bins, hist, func_base=func_base)
            
            fit_func = func_base(x_values, *popt)
            self.hist_config["fit_params"] = (popt)
            self.hist_config["r2_coef"] = r2
            
            self.ax.plot(
                x_values, fit_func,
                color="tab:red", linewidth=1.5, alpha=alpha,
                label=f"Fit function {str(func_base.__name__).replace('_', ' ')}",
            )
        
        # display axes label
        if xlog:
            self.ax.set_xlabel("Equivalent diameters $log(D_p)$ $[\\mu m]$", fontsize=self.dict_fontsize["label"])
        else:
            self.ax.set_xlabel("Equivalent diameters $D_p$ $[\\mu m]$", fontsize=self.dict_fontsize["label"])
        self.ax.set_xticklabels([f"{x_tick:.0f}" for x_tick in x_ticks], fontsize=self.dict_fontsize["ticks"])
        
        if density:
            self.ax.set_ylabel("$\\dfrac{{dN}}{{d \\, log(D_p)}}$", fontsize=self.dict_fontsize["label"])
            self.ax.set_yticklabels([f"{y_tick:.3f}" for y_tick in y_ticks], fontsize=self.dict_fontsize["ticks"])
        else:
            self.ax.set_ylabel("Counts", fontsize=self.dict_fontsize["label"])
            self.ax.set_yticklabels([f"{y_tick:.0f}" for y_tick in y_ticks], fontsize=self.dict_fontsize["ticks"])
        
        if fit:
            self.ax.legend(fontsize=self.dict_fontsize["legend"])
        
        self.canvas.draw_idle()
        self._update_limits_from_axes()
        
        # compute uncertainties on fit parameters
        hist, bin_edges = np.histogram(scaled_data, hist_kwargs["bins"], density=False)
        centered_bins = (bin_edges[:-1] + bin_edges[1:]) / 2
        uncertainties = self.uncertainties(
            fit_func=func_base, params=popt,
            x_edges=centered_bins, hist=hist,
            uncertainty=self.pixel_size,
            )._compute_uncertainty()
        self.hist_config["uncertainties_params"] = uncertainties
    
    def _set_limit(self, key, text):
        """ Set axes limits """
        
        try:
            self.hist_config[key] = float(text) if text else None
        except ValueError:
            self.hist_config[key] = None
        self._draw_histogram()
        
    def _reset_defaults(self):
        """ Reset default parameters """
        
        self.hist_config = self.default_config.copy()
        self._draw_histogram()
    
    def _nearest_ten(self, number) -> int:
        """ Find closest tenth number """
        
        if 0 <= number < 10:
            return 0
        elif number >= 10:
            return int((number + 5) // 10) * 10
        else:
            return int((number - 5 ) // 10) * 10
    
    def _count_hist_peaks(self, hist_values) -> int:
        """ Count number of peaks """
        
        if not isinstance(hist_values, np.ndarray):
            hist_values = np.array(hist_values)
        
        if len(hist_values) < 3:
            return 0, np.array([])

        diff1 = np.diff(hist_values)
        sign_change = np.diff(np.sign(diff1))
        
        peaks_indices = np.where(sign_change < 0)[0] + 1
        
        n_peaks = len(peaks_indices)
        return n_peaks, peaks_indices

class FitFunction:
    def __init__(self):
        
        pass
    
    def _get_fitting_function(self):
        """ Fitting function choice """
        
        def poly_order(x, *coef):
            return sum(c * x**i for i, c in enumerate(reversed(coef)))
        
        def exp(x, a, b, x0, y_0):
            return a * np.exp(b * (x-x0)) + y_0
        
        def gaussian(x, mu, sigma, y_0):
            return 1 / (sigma*np.sqrt(2*np.pi)) * np.exp(-1/2*(x-mu)**2/(sigma**2)) + y_0
        
        def log(x, a, b, y_0):
            return a * np.log(b * x) + y_0
        
        def log_normal(x, mu, sigma, y_0):
            return 1 / (x * sigma * np.sqrt(2*np.pi)) * np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) + y_0
        
        def sigmoid(x, a, b, x0, y_0):
            return a / (1 + np.exp(-b * (x - x0))) + y_0
        
        def generalized_sigmoid(x, a, k, c, q, b, x0, nu):
            return a + (k - a) / ((c + q * np.exp(-b*(x - x0))))**(1/nu)
        
        return {
            "poly_order": poly_order,
            "exp": exp,
            "gaussian": gaussian,
            "log": log,
            "log_normal": log_normal,
            "sigmoid": sigmoid,
            "generalized_sigmoid": generalized_sigmoid,
        }
        
    def _validate_xy(self, x, y):
        """ Validate input before fitting """
        
        if x is None or y is None:
            msg = "'x' and 'y' must be provided as ndarray"
            raise TypeError(msg)

        x = np.array(x) if not isinstance(x, np.ndarray) else x
        y = np.array(y) if not isinstance(y, np.ndarray) else y
        
        return x, y
    
    def _init_params_bounds(self, func_base, x, y, n_components=1, p0=None, bounds=None):
        """ Parameters initilization before fitting"""
        
        eps = 1e-8
        params_size = {
            "exp": 4,
            "log": 2,
            "gaussian": 3,
            "log_normal": 3,
            "sigmoid": 3,
            "generalized_sigmoid": 7,
        }.get(func_base.__name__, len(p0) // n_components if p0 is not None else 1)
        
        if func_base.__name__ == "poly_order":
            degree = int(''.join(filter(str.isdigit, func_base.__name__.split("_")[-1])))
            p0 = np.ones(degree + 1) if p0 is not None else p0
        
        elif func_base.__name__ == "exp":
            if p0 is None or len(p0) == 0:
                A0 = max(y) - min(y)
                B0 = -0.1
                x0_0 = np.median(x)
                y0 = min(y)
                p0 = ([A0, B0, x0_0, y0] * n_components)
            else:
                p0 = np.ones(params_size * n_components)
            p0 = np.array(p0, dtype=float)
            if bounds is None:
                A_lower, A_upper = min(y), max(y)
                B_lower, B_upper = -10, 10
                x0_lower, x0_upper = min(x), max(x)
                y_lower, y_upper = min(y) - 0.5*abs(min(y)), max(y) + 0.5*abs(max(y))
                lower = np.tile([A_lower, B_lower, x0_lower, y_lower], n_components).tolist()
                upper = np.tile([A_upper, B_upper, x0_upper, y_upper], n_components).tolist()
                bounds = (np.array(lower, dtype=float), np.array(upper, dtype=float))

        elif func_base.__name__ == "gaussian":
            if p0 is None or len(p0) == 0:
                mu0 = np.median(x)
                sigma0 = max(np.std(x), eps)
                y0 = np.clip(min(y), min(y)-0.5*abs(min(y))+eps, max(y)+0.5*abs(max(y))-eps)
                p0 = ([mu0, sigma0, y0] * n_components)
            else:
                p0 = np.ones(params_size * n_components)
            p0 = np.array(p0, dtype=float)
            if bounds is None:
                mu_lower, mu_upper = min(x), max(x)
                sigma_lower, sigma_upper = eps, max(x)-min(x)
                y0_lower, y0_upper = min(y) - 0.5*abs(min(y)), max(y) + 0.5*abs(max(y))
                lower = np.tile([mu_lower, sigma_lower, y0_lower], n_components).tolist()
                upper = np.tile([mu_upper, sigma_upper, y0_upper], n_components).tolist()
                bounds = (np.array(lower, dtype=float), np.array(upper, dtype=float))
        
        elif func_base.__name__ == "log_normal":
            if p0 is None or len(p0) == 0:
                mu0 = np.clip(np.log(np.median(x[x>0])), np.log(min(x[x>0])+eps)+eps, np.log(max(x))-eps)
                sigma0 = max(1.0, eps)
                y0 = np.clip(min(y), min(y)-0.5*abs(min(y))+eps, max(y)+0.5*abs(max(y))-eps)
                p0 = [mu0, sigma0, y0] * n_components
            else:
                p0 = np.ones(params_size * n_components + 1)
            p0 = np.array(p0, dtype=float)
            if bounds is None:
                mu_lower, mu_upper = np.log(min(x[x>0]) + eps), np.log(max(x)) + eps
                sigma_lower, sigma_upper = eps, 10.0
                y0_lower = min(y) - 0.5*abs(min(y))
                y0_upper = max(y) + 0.5*abs(max(y))
                lower = np.tile([mu_lower, sigma_lower, y0_lower], n_components).tolist()
                upper = np.tile([mu_upper, sigma_upper, y0_upper], n_components).tolist()
                bounds = (np.array(lower, dtype=float), np.array(upper, dtype=float))
            
        elif func_base.__name__ == "log":
            if len(p0) == 0:
                p0 = np.ones(3)
        
        elif func_base.__name__ == "sigmoid":
            if p0 is None:
                p0 = [max(y)-min(y), 1, np.median(x), min(y)]
            if bounds is None:
                lower = [0, -10, min(x), min(y) - abs(min(y))]
                upper = [10*(max(y) - min(y)), 10, max(x), max(y) + abs(max(y))]
                bounds = (np.array(lower, dtype=float), np.array(upper, dtype=float))
        
        elif func_base.__name__ == "generalized_sigmoid":
            if len(p0) == 0:
                p0 = np.ones(7)
            
        # if bounds is None:
        #     lower = np.full(len(p0), -np.inf)
        #     upper = np.full(len(p0, np.inf))
        #     bounds = (lower, upper)
            
        return p0, bounds, params_size
        
    def _mixture_func(x, func, params, n_components=2, params_size=0, weights=None):
        """ Build mixture function """
        
        if weights is None:
            weights = np.ones(n_components) / n_components
        else:
            weights = np.array(weights, dtype=float)
            weights = weights / np.sum(weights)
            
        y0 = params[-1]
        
        results = np.zeros_like(x) + y0
        
        for i in range(n_components):
            start = i * params_size
            end = start + params_size
            results += weights[i] * func(x, *params[start:end])
        
        return results
    
    def _fit_curve(self, x, y, func_base, p0=None, bounds=None, n_components=1, weights=None, max_retries=5):
        """ Compute fit function """
        
        x, y = self._validate_xy(x, y)
        
        p0, bounds, params_size = self._init_params_bounds(func_base, x, y, n_components, p0, bounds)
        fit_func = func_base if n_components == 1 else lambda x, *params: self._mixture_func(x, func_base, params, n_components, params_size, weights)
        
        popt, pcov = None, None
        
        for attempt in range(max_retries):
            try:
                popt, pcov = curve_fit(fit_func, x, y, p0=p0, bounds=bounds, maxfev=50000)
                break
            except Exception:
                if attempt < max_retries - 1:
                    p0 = p0 * (1 + np.random.uniform(-0.5, 0.5, len(p0)))
                else:
                    msg = f"Fitting failed after {max_retries} attemps"
                    raise RuntimeError(msg)
        
        fit = fit_func(x, *popt)
        ss_res = np.sum((y - fit)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res / ss_tot
        
        # integral_value = self.verify_normalization(
        #     func=func_base, params=popt, weights=weights, domain=[min(x), max(x)],
        #     n_components=n_components, params_size=params_size,
        #     )
        # print(integral_value)
        
        return popt, pcov, fit, r2
    
    def verify_normalization(self, func, params, weights, domain, n_components, params_size):
        """ Verify integration """
        
        f = lambda x, *params: self._mixture_func(
            x, func, params,
            n_components=n_components,
            params_size=params_size,
            weights=weights,
            )
        integral, _ = quad(f, domain[0], domain[1])
        
        return integral
    
class Uncertainties():
    def __init__(self, fit_func, params, x_edges, hist, uncertainty):
        
        self.params = np.array(params, dtype=float)
        self.x_edges = np.array(x_edges)
        
        self.hist = hist
        self.fit_func = fit_func
        
        self.bin_width = np.diff(self.x_edges)
        
        self.uncertainty = uncertainty
        
        self._compute_uncertainty()
        
    def _compute_uncertainty(self):
        
        Ntot = np.sum(self.hist)
        
        # density error
        sigma_err = np.sqrt(self.hist)
        sigma_err[sigma_err == 0] = 1.0
        
        # weight matrix
        W = np.diag(1 / (sigma_err**2))
        
        # jacobian matrix
        J = self._compute_jacobian()
        
        # covariance matrix
        A = J.T @ W @ J
        
        cov_theta = np.linalg.pinv(A)
        
        err_mu = np.sqrt(cov_theta[0, 0])
        err_sigma = np.sqrt(cov_theta[1, 1])
        
        return (err_mu, err_sigma)
    
    def _compute_jacobian(self, eps=1e-6):
        """ Compute jacobian matrix """
        
        params = self.params.copy()
        
        f0 = self.fit_func(self.x_edges, *params)
        
        n_data = len(f0)
        n_params = len(self.params)
        
        J = np.zeros((n_data, n_params))
        
        for j in range(n_params):
            
            p_plus = self.params.copy()
            p_minus = self.params.copy()

            p_plus[j] += eps
            p_minus[j] -= eps
            
            f_plus = self.fit_func(self.x_edges, *p_plus)
            f_minus = self.fit_func(self.x_edges, *p_minus)
            
            J[:, j] = (f_plus - f_minus) / (2 * eps)
            
        return J

class VideoMaker(QObject):

    progress_video_creation = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, parent=None):
        super(VideoMaker, self).__init__(parent)
        self.parent = parent
        
    def Make_video(
        self,
        load_images,
        path_save,
        video_name,
        images_range: str|np.ndarray|list = [0, 34939],
        frame_rate=30,
        format_video=".avi",
        do_display_time:bool=True,
        rotate:int=0,
        freq: float = None,
    ):
        
        if not Path(load_images).exists():
            msg = "File to load images do not exists"
            raise FileExistsError(msg)
        
        if not Path(path_save).exists():
            msg = "File to save video do not exists"
            raise FileExistsError(msg)

        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        list_images = [
            file for file in Path(load_images).iterdir()
            if Path(file).is_file() and file.suffix.lower() in valid_extensions
        ]
        list_images = list(Tcl().call("lsort", "-dict", list_images))
        # list_images = list_images[images_range[0] : images_range[1]]

        total = len(list_images)
        print(f"Total images : {total}")

        img = np.array(Image.open(Path(list_images[0])))
        if img.ndim == 3:  # Convert color image to inverted grayscale
            img = np.array(
                # ImageOps.invert(Image.fromarray(img).convert("L").rotate(rotate, expand=True)), dtype=np.uint8
                Image.fromarray(img).convert("L").rotate(rotate, expand=True), dtype=np.uint8
            )
        height, width = img.shape

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # if video_name is None:
        #     video_name = f"{str(path_save).split('\\')[-2]}_{str(path_save).split('\\')[-1]}_{str(load_images).split('\\')[-1]}_images_{images_range[0]}_{images_range[1]}.avi"

        print(f"Path to save : {path_save}")
        video_writer = cv2.VideoWriter(
            Path(path_save) / Path(video_name), fourcc, frame_rate, (width, height)
        )
        
        min, max = images_range[0], images_range[1]
        # font = ImageFont.truetype("arial.ttf", size=42)
        import time
        import sys
        for i, (curr_img, img_name) in enumerate(zip(list(np.linspace(min, max+1, max-min+2, dtype=np.int16)), list_images)):
            
            pil_img = Image.open(Path(img_name)).convert("RGB").rotate(rotate, expand=True)
            
            if do_display_time:
                draw = ImageDraw.Draw(pil_img)
                text = f"t = {curr_img*1/freq:2f} s"
                draw.text((0, 0), text, font=18, fill=(0, 0, 128))
            
            img = np.array(pil_img)
            
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video_writer.write(img)

            percent = int(((i+1) / total) * 100)
            self.progress_video_creation.emit(percent)

        video_writer.release()
        self.finished.emit()

        print("Videos created")
