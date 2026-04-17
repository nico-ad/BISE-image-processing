#%% Import

import sys
from PyQt5 import uic
from PyQt5.QtCore import Qt, QThread, QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow
from PyQt5.QtWidgets import QTabWidget, QPushButton, QDial, QCheckBox, QLineEdit, QProgressBar, QComboBox, QLabel, QSpinBox
from PyQt5.QtWidgets import (
    QTableWidget,
    QTableWidgetItem,
    QFileDialog,
    QTableView,
    QListView,
    QTreeView,
    QAbstractItemView,
    QListWidget,
    QSplitter,
    QSizePolicy,
)
from PyQt5.QtWidgets import (
    QStyledItemDelegate,
    QVBoxLayout,
    QHBoxLayout,
)
from PyQt5.QtGui import QIcon, QFont, QPixmap, QImage

import pyqtgraph as pg

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from pathlib import Path
import csv

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.cm as cm
from matplotlib.colors import Normalize

import mplcursors

import Support_functions_visualization as Visualization
import Support_functions_preview as func_preview

class HelperTab4(QWidget):
    
    def __init__(self, parent):
        super().__init__()
        
        self.parent = parent
        
        self.visualization_func = Visualization.VisualizationFunctions(parent=self)
        self.fitting_func = func_preview.FitFunction()
        
        self.function_map = {
            "Histogram diameters": self.visualization_func.Visualize_histogram_diameters,
            "Number of particles": self.visualization_func.Visualize_num_part_per_frames,
            "Number of labels": self.visualization_func.Visualize_num_labels,
            "Mean diameter": self.visualization_func.Visualize_mean_diameter,
            "Mean inter-particle distance": self.visualization_func.Visualize_mean_inter_particle_distance,
            "Mean free path": self.visualization_func.Visualize_mean_free_path,
            "Coordination number": self.visualization_func.Visualize_coordination_number,
            "Surface concentration": self.visualization_func.Visualize_surface_concentration,
            "Resuspended fraction": self.visualization_func.Visualize_resuspended_fraction,
            "Collision frequency": self.visualization_func.Visualize_collision_frequency,
            "Remaining fraction": self.visualization_func.Visualize_remaining_fraction,
            "Velocity flow": self.visualization_func.Visualize_velocity_flow,
            "Velocity": self.visualization_func.Visualize_velocity,
            "Acceleration": self.visualization_func.Visualize_acceleration,
            "Momentum": self.visualization_func.Visualize_momentum,
            "Kinetic energy": self.visualization_func.Visualize_kinetic_energy,
            "Mean square displacement": self.visualization_func.Mean_square_displacement,
            "Particles dectection": self.visualization_func.Visualize_particle_detection,
            "Clusters": self.visualization_func.Visualize_cluster,
            "Position tracking particles": self.visualization_func.Track_particles_position,
            "Velocity tracking particles": self.visualization_func.Track_particles_velocity,
            "Visualize labels": self.visualization_func.Visualize_labels,
        }
        
        tab_4 = QWidget()
        tab_4_layout = QVBoxLayout(tab_4)
        
        # setup title page
        self.title_page_4 = QLineEdit(tab_4)
        self.title_page_4.setText("Plot graphs and images for interpretations")
        self.title_page_4.setReadOnly(True)
        self.title_page_4.setFont(self.parent.font_title)
        self.title_page_4.setAlignment(Qt.AlignCenter)
        tab_4_layout.addWidget(self.title_page_4)
        
        # ----- SPLITTER
        splitter = QSplitter(Qt.Horizontal)
        
        # ==========
        # LEFT PANEL
        # ==========
        
        # ----- create left panel
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # ----- folders table
        self.parent.parameters_table_tab_4 = QTableWidget(tab_4)
        self.parent.parameters_table_tab_4.setColumnCount(1)
        self.parent.parameters_table_tab_4.setHorizontalHeaderLabels(["Selected files"])
        self.parent.parameters_table_tab_4.horizontalHeader().setFont(self.parent.font_header)
        self.parent.parameters_table_tab_4.setColumnWidth(0, 300)
        self.parent.parameters_table_tab_4.setFixedWidth(300)
        left_layout.addWidget(self.parent.parameters_table_tab_4)
        
        # ----- add folder button
        self.add_folders = QPushButton(tab_4)
        self.add_folders.setText("Add folders")
        self.add_folders.setFont(self.parent.font_button)
        self.add_folders.setFixedHeight(50)
        self.add_folders.setFixedWidth(150)
        self.add_folders.clicked.connect(self._add_files)
        left_layout.addWidget(self.add_folders)
        
        # ----- remove folder button
        self.remove_folders = QPushButton(tab_4)
        self.remove_folders.setText("Remove folders")
        self.remove_folders.setFont(self.parent.font_button)
        self.remove_folders.setFixedHeight(50)
        self.remove_folders.setFixedWidth(150)
        self.remove_folders.clicked.connect(self._remove_files)
        left_layout.addWidget(self.remove_folders)
        
        # ----- function label
        self.function_label = QLabel("No function selected")
        self.function_label.setFont(self.parent.font_content)
        self.function_label.setAlignment(Qt.AlignCenter)
        self.function_label.setFixedWidth(250)
        left_layout.addWidget(self.function_label)
        
        # ----- function selector
        self.function_selector = QListWidget()
        self.function_selector.setFixedWidth(250)
        
        categories = {
            "1D display": [
                "Histogram diameters", "Number of particles", "Number of labels",
                "Mean inter-particle distance", "Mean free path", "Coordination number",
                "Collision frequency", "Resuspended fraction", "Remaining fraction",
                "Velocity flow", "Surface concentration", "Mean diameter",
                "Velocity", "Acceleration", "Momentum", "Kinetic energy",
                "Mean square displacement", "Visualize labels",
                ],
            "2D display": [
                "Particles detection", "Voronoi trangulation", "Labels",
                "Number of labels per frame", "Clusters", "Position tracking particles",
                "Velocity tracking particles",
                ],
        }
        
        for cat, funcs in categories.items():
            self.function_selector.addItem(f"--- {cat} ---")
            for f in funcs:
                self.function_selector.addItem(f)
        left_layout.addWidget(self.function_selector)
        self.function_selector.itemClicked.connect(self._execute_function)
        
        # ----- Frames
        frames_layout = QHBoxLayout()
        self.frames_label = QLabel("Frames")
        self.frames_label.setFont(self.parent.font_content)
        self.frames_label.setAlignment(Qt.AlignCenter)
        self.frames_label.setFixedWidth(80)
        frames_layout.addWidget(self.frames_label)
        
        self.frames_start = QLineEdit()
        self.frames_start.setPlaceholderText("Start")
        self.frames_start.setAlignment(Qt.AlignCenter)
        self.frames_start.setFixedWidth(80)
        frames_layout.addWidget(self.frames_start)
        
        self.frames_end = QLineEdit()
        self.frames_end.setPlaceholderText("End")
        self.frames_end.setAlignment(Qt.AlignCenter)
        self.frames_end.setFixedWidth(80)
        frames_layout.addWidget(self.frames_end)
        
        frames_layout.addStretch()
        left_layout.addLayout(frames_layout)
        
        # ----- Labels
        labels_layout = QHBoxLayout()
        self.labels_label = QLabel("Labels")
        self.labels_label.setFont(self.parent.font_content)
        self.labels_label.setAlignment(Qt.AlignCenter)
        self.labels_label.setFixedWidth(80)
        labels_layout.addWidget(self.labels_label)
        
        self.labels_textbox = QLineEdit()
        self.labels_textbox.setPlaceholderText("1;2;3 and/or 5-9 or all")
        self.labels_textbox.setFixedWidth(150)
        labels_layout.addWidget(self.labels_textbox)
        
        labels_layout.addStretch()
        left_layout.addLayout(labels_layout)
        
        left_layout.addStretch()
        # ==========
        # RIGHT PANEL
        # ==========
        
        # ----- plot area
        self.plot_container = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_container)
        
        self.viewer = ImageViewer(self.plot_container, pixel_size=1.0)
        
        # ----- INCLUDE TO SPLITTER
        
        splitter.addWidget(left_widget)
        splitter.addWidget(self.plot_container)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 7)
        
        tab_4_layout.addWidget(splitter)
        
        # add widget to layout
        self.parent.tabs.addTab(tab_4, "Plot analysis results")
    
    def _add_files(self):
        """" Add file for analysis"""
        
        # select files with dialog box
        dialog = QFileDialog(self.parent)
        dialog.setFileMode(QFileDialog.ExistingFiles)
        dialog.setNameFilter("CSV files (*.csv)")
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        
        if dialog.exec_():
            
            if not hasattr(self.parent, "folders_list_dataframe"):
                self.parent.folders_list_dataframe = []
            if not hasattr(self.parent, "files_list_dataframe"):
                self.parent.files_list_dataframe = []
            
            new_folders = []
            new_files = []
            
            for file_path in dialog.selectedFiles():
                
                file_path = Path(file_path)
                
                # check if file already loaded
                if file_path in self.parent.files_list_dataframe:
                    continue
                
                row = self.parent.parameters_table_tab_4.rowCount()
                self.parent.parameters_table_tab_4.insertRow(row)
                
                item = QTableWidgetItem()
                
                new_folders.append(file_path.parents[0])
                new_files.append(file_path.name)
                
                # truncate path
                file_path = str(file_path)
                if len(file_path) > 20:
                    display_text = f"{file_path[:20]}...{file_path[-20:]}"
                else:
                    display_text = file_path
                    
                # create item to store path folder
                item.setText(display_text)
                item.setData(Qt.UserRole, file_path)
                self.parent.parameters_table_tab_4.setItem(row, 0, item)
                
            self.parent.folders_list_dataframe.extend(new_folders)
            self.parent.files_list_dataframe.extend(new_files)
        
            self.parent.parameters_table_tab_4.viewport().update()
                
    def _remove_files(self):
        """ Remove file """
        
        line = self.parent.parameters_table_tab_4.currentRow()
        if line >= 0:
            self.parent.parameters_table_tab_4.removeRow(line)
        
        if not hasattr(self, "viewer"):
            self.viewer._clear()
    
    def _load_file(self, path: str|Path, usecols: str|list = None, change_main_path_images: str = None):
        """ Load data file """
        
        if path is None:
            msg = "path must not empty "
            raise TypeError(msg)
        
        dataframe = pd.DataFrame(pd.read_csv(path, usecols=usecols))
        if change_main_path_images:
            dataframe["main_path"] = [change_main_path_images][0] * len(dataframe["main_path"])
        if "Unnamed: 0" in dataframe.columns:
            dataframe = dataframe.drop(["Unnamed: 0"], axis=1)
        
        if change_main_path_images:
            dataframe["main_path"] = [change_main_path_images] * len(dataframe["main_path"])
        
        if "frame" in dataframe.columns:
            dataframe = dataframe.sort_values("frame")
        
        return dataframe
    
    def _execute_function(self, item):
        """ Launch function for visualization """
        
        name = item.text()
        if name.startswith("---"):
            return
        
        self.function_label.setText(f"Selected function : {name}")
        
        # load dataframes
        dataframe = [self._load_file(Path(p) / Path(f)) for p, f in zip(
            self.parent.folders_list_dataframe, self.parent.files_list_dataframe
            )]
        
        # read frames
        if self.frames_start.text() and self.frames_end.text():
            frames_start = int(self.frames_start.text()) if self.frames_start.text() else None
            frames_end = int(self.frames_end.text()) if self.frames_end.text() else None
            frames = np.linspace(frames_start, frames_end, frames_end-frames_start+1, dtype=int)
        else:
            frames = None
        
        # read labels
        if self.labels_textbox.text():
            labels_text = self.labels_textbox.text().strip().lower()
            labels = self._parse_labels(labels_text)
        else:
            labels = None
        
        # clear graph container
        self.viewer._clear()
        
        # load selected function and compute on thread
        if name in self.function_map:
            
            # func = self.function_map[name]
            # runner = ThreadRunner(
            #     func,
            #     dataframe=dataframe,
            #     pixel_size=1.0,
            #     frames=frames,
            #     labels=labels,
            #     on_results=self.viewer._display,
            #     # on_error=self.on_handle_error,
            # )
            # runner._start()
            
            func = self.function_map[name]
            print(func.__name__)
            if func:
                data_list = func(
                    dataframe=dataframe,
                    pixel_size=0.006,
                    frames=frames,
                    labels=labels,
                    )
        
        self.viewer._display(data_list)
    
    def _parse_labels(self, text):
        """ Parse labels textbox """
        
        if text == "all":
            return "all"
        
        result = set()
        if not text:
            return list(result)
        
        parts = text.split(";")
        for part in parts:
            part = part.strip()
            
            if "-" in part:
                
                start, end = part.split("-")
                start = int(start)
                end = int(end)
                
                if start <= end:
                    result.update(range(start, end + 1))
                else:
                    result.update(range(end, start + 1))
                    
            else:
                if part.isdigit():
                    result.add(int(part))
                    
        return sorted(list(result))

class ImageViewer(QWidget):

    def __init__(self, container_widget, pixel_size):
        super().__init__()
        
        self.container = container_widget
        
        if self.container.layout() is None:
            self.layout = QVBoxLayout(self.container)
        else:
            self.layout = self.container.layout()
        
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
                f"x={event.xdata*self.pixel_size:.2f} mm   "
                f"y={event.ydata*self.pixel_size:.2f} mm   "
                f"value={val}",
            )

    # zoom on scroll
    def _on_scroll(self, event, ax, canvas):
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
        
    def _clear(self):
        """ Clear canvas """
        
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

    def _order_of_magnitude(self, number, nearest_power_of_three=False):
        """ Give magnitude order of a number """

        if not isinstance(number, (int, float)):
            number = np.array(number)
        
        if number == 0:
            return 0
        
        abs_number = np.abs(number)
        exponent = np.floor(np.log10(abs_number))

        if nearest_power_of_three:
            if exponent < 0.0:
                return 10 ** (3 ** round((exponent - exponent%3) / 3))
            if exponent > 0.0:
                return 10 ** (3 ** round((exponent - exponent%3) / 3))
        
        return 10 ** exponent

    def _display(self, data_list: list, **kwargs):
        """ Diplay data even if 1dD or 2D data """
        
        self._clear()

        self.fig = Figure()

        for ii, data in enumerate(data_list):

            if not data:
                continue
            
            if "histogram" in data:
                self.plot_1d(data)
                
            elif "curves" in data:
                self.plot_1d(data)

            elif "hist_3d" in data:
                self.plot_1d(data)
            
            elif "image" in data:
                self.plot_2d(data)
            
            else:
                msg = "No key word detected"
                raise ValueError(msg)
            
    def plot_1d(self, data_dict: dict):
        """ Display 1d graph """
        
        # ----- draw histogram
        if data_dict.get("histogram"):
            
            self._plot_histogram(data_dict)

            if data_dict.get("fit"):
                self._plot_fit(data_dict)
            
            if data_dict.get("d_50"):
                self._plot_d_50(data_dict)
                
        # ----- draw curves
        elif data_dict.get("curves"):

            self._plot_curves(data_dict)

        # ----- draw histogram 3D
        elif data_dict.get("hist_3d"):
            self.plot_hist3d(data_dict)
        
        self._format_axes(data_dict)
        
        self.fig.tight_layout()
        
        canvas = FigureCanvas(self.fig)
        toolbar = NavigationToolbar(canvas, self.container)
        
        self.layout.addWidget(toolbar)
        self.layout.addWidget(canvas)
        
        self.canvases.append(canvas)
        self.toolbars.append(toolbar)

    def plot_2d(self, data_dict: dict):
        """ Display 2d graph """

        self.ax = self.fig.add_subplot(111)
        
        self.ax.imshow(data_dict["image"], origin="lower", cmap="gray", aspect="auto")
        self.ax.set_aspect("equal", adjustable="box")
        
        if data_dict["label"]:
            
            # --- color map
            if len(data_dict["label"]) <= 10:
                colors = cm.get_cmap("tab10")
                norm = None
            elif len(data_dict["label"]) <= 20:
                colors = cm.get_cmap("tab20")
                norm = None
            else:
                colors = cm.get_cmap("plasma")
            
            legend_elements = []

            # ----- scatter plot position
            for i, lbl in enumerate(data_dict["label"]):

                data_x = data_dict["x"][i]
                data_y = data_dict["y"][i]

                data_all = np.column_stack((data_x, data_y))
                _, indices = np.unique(data_all, axis=0, return_index=True)
                data_all = data_all[indices] # * data_dict["unit"]

                self.ax.scatter(
                    data_all[:, 0], data_all[:, 1],
                    label=f"Label {int(lbl)}",
                    )
        
                # ----- display vectors
                if "vx" in data_dict.keys() and "vy" in data_dict.keys():

                    from matplotlib.patches import FancyArrowPatch
                    
                    data_arrow_x = data_dict["vx"][i][indices] # * data_dict["unit"]
                    data_arrow_y = data_dict["vy"][i][indices] # * data_dict["unit"]

                    norm = np.sqrt((
                            data_dict["vx"][i][indices]**2 +
                            data_dict["vx"][i][indices]**2
                            ))
                    data_arrow_x = data_dict["vx"][i][indices] / norm
                    data_arrow_y = data_dict["vy"][i][indices] / norm

                    Q = self.ax.quiver(
                        data_all[:, 0], data_all[:, 1], # [m, mm, px]
                        data_arrow_x, data_arrow_y, # [m/s, mm/s, px/s]
                        angles="xy",
                        scale=1,
                        scale_units="xy",
                        color=colors(i),
                        alpha=0.6,
                        width=0.02,
                        # label=f"label {int(lbl)}",
                    )

                    mag_order = self._order_of_magnitude(max(norm))
                    print(max(norm), mag_order)
                    qk = self.ax.quiverkey(
                        Q, X=0.9, Y=0.9,
                        U=1*mag_order,
                        label=f"{1*mag_order} m/s",
                        # labelpos="E",
                        # color=colors(i),
                    )

                    legend_elements.append(
                        FancyArrowPatch((0.1, 0.5), (0.9, 0.8), color=colors(i),
                                   mutation_scale=100, arrowstyle="-|>")
                        )

        # sm = plt.cm.ScalarMappable(
        #     cmap=colors,
        #     norm=norm,
        #     )
        # sm.set_array([])
        # cbar = plt.colorbar(sm, ax=self.ax, fraction=0.046, pad=0.01)
        # cbar_ticks = cbar.get_ticks()
        # cbar.set_ticks(cbar_ticks)
        # cbar.set_ticklabels([f"{cbar_tick:.2f}" for cbar_tick in cbar_ticks], fontsize=self.dict_fontsize["ticks"])
        # cbar.set_label("Time displacement", fontsize=self.dict_fontsize["label"])
        
        x_ticks = self.ax.get_xticks()[1:-1]
        y_ticks = self.ax.get_yticks()[1:-1]
        
        self.ax.set_xticks(x_ticks)
        self.ax.set_yticks(y_ticks)
        
        self.ax.set_xlabel(data_dict["x_label"], fontsize=self.dict_fontsize["label"])
        self.ax.set_ylabel(data_dict["y_label"], fontsize=self.dict_fontsize["label"])
        
        self.ax.set_xticklabels([f"{x_tick*self.pixel_size:.1f}" for x_tick in x_ticks], fontsize=self.dict_fontsize["ticks"])
        self.ax.set_yticklabels([f"{y_tick*self.pixel_size:.3f}" for y_tick in y_ticks], fontsize=self.dict_fontsize["ticks"])
        
        self.ax.legend(
            handles=legend_elements, #[f"{1*mag_order} m/s"]*len(legend_elements),
            fontsize=self.dict_fontsize["legend"])
        
        self.fig.tight_layout()
        
        canvas = FigureCanvas(self.fig)
        toolbar = NavigationToolbar(canvas, self.container)
        
        self.layout.addWidget(toolbar)
        self.layout.addWidget(canvas)
        
        self.canvases.append(canvas)
        self.toolbars.append(toolbar)
        
        # canvas.mpl_connect("motion_notify_event",
        #                    lambda event: self.on_move(event, ax, img, status_label),
        # )
        # canvas.mpl_connect("scroll_event",
        #                    lambda event: self.on_scroll(event, ax, canvas),
        # )
    
    def plot_hist3d(self, data_dict: dict, inc = 0):
        """ Plot histogram in 3D """

        self.ax = self.fig.add_subplot(111, projection="3d")
        
        # collect data
        data_x_all = []
        data_y_all = []
        data_z_all = []
        
        for i in range(len(data_dict["x"])):
            
            x = np.array(data_dict["x"][i] * data_dict["x_unit"])[:-1] - 0.5
            y = np.array(data_dict["y"][i] * data_dict["y_unit"])
            z = np.array(data_dict["z"][i] * data_dict["z_unit"])
            
            data_x_all.append(x)
            data_y_all.append(y)
            data_z_all.append(np.full(len(x), z))
        
        # flatten
        data_x_all = np.concatenate(data_x_all)
        data_y_all = np.concatenate(data_y_all)
        data_z_all = np.concatenate(data_z_all)
        
        print("LOL")
        print(np.min(data_y_all), np.max(data_y_all))
        
        # down sample
        if len(data_y_all) > 100: down_sample = 5
        else: down_sample = 0
        data_x_all = data_x_all[::down_sample]
        data_y_all = np.array(data_y_all[::down_sample], dtype=float)
        data_z_all = np.array(data_z_all[::down_sample], dtype=float)

        # normalize counts
        num_coord_number = len(np.unique(data_x_all))
        if data_dict["nomalize"]:
            for i in range(0, len(data_z_all), num_coord_number):

                idx = np.arange(i, i+num_coord_number, dtype=int)
                total = np.sum(data_y_all[idx])
                
                if total != 0: data_y_all[idx] = data_y_all[idx] / total
                else: data_y_all[idx] = 0.0
        
        z0 = np.full_like(data_y_all, data_z_all)
        dx = np.ones_like(data_x_all) * 0.8
        dy = np.ones_like(data_y_all) * 1.0
        dz = np.ones_like(data_z_all) * (data_z_all[1] - data_z_all[0])
        
        print()
        print(np.min(data_x_all), np.max(data_x_all), data_dict["x_unit"])
        print(np.min(data_y_all), np.max(data_y_all), data_dict["y_unit"])
        print(np.min(data_z_all), np.max(data_z_all), data_dict["z_unit"])
        print()
        
        # # color by time
        # norm = Normalize(vmin=np.min(data_z_all), vmax=np.max(data_z_all))
        # colors = cm.plasma(norm(data_z_all))
        
        self.ax.bar3d(
            data_x_all, data_z_all, data_y_all,
            dx, dz, dy,
            shade=True,
            # color=colors, 
        )
        
        # # collect data
        # x_centers = []
        # dx_list = []
        # dy_list = []
        # z_list = []
        
        # for bin_list , occs_list, z_vals_list in zip(data_dict["x"], data_dict["y"], data_dict["z"]):
            
        #     bins_array = np.array(bin_list) * data_dict["x_unit"]
        #     if bins_array.ndim == 1:
        #         bins_array = np.array([bins_array, bins_array])
        #     x_centers.extend((bins_array[:, 0] + bins_array[:, 1]) / 2)
        #     dx_list.extend((bins_array[:, 1] - bins_array[:, 0]))
            
        #     dy_list.extend(np.array(occs_list) * data_dict["y_unit"])
            
        #     z_list.extend(np.array(z_vals_list) * data_dict["z_unit"])
        
        # x_centers = np.array(x_centers)
        # dx_list = np.array(dx_list)
        # dy_list = np.array(dy_list)
        # z_list = np.array(z_list)
        # y_base = np.zeros_like(x_centers)
        # dz = np.ones_like(x_centers) * 0.8
        
        # print(x_centers.shape, y_base.shape, z_list.shape)
        
        # # color by time
        # norm = Normalize(vmin=np.min(z_list), vmax=np.max(z_list))
        # colors = cm.plasma(norm(z_list))
        
        # self.ax.bar3d(
        #     x_centers, z_list, y_base,
        #     dx_list, dz, dy_list,
        #     color=colors, shade=True,
        # )

    def _plot_histogram(self, data_dict: dict, inc: int = 0):
        """ Plot histogram """

        self.ax = self.fig.add_subplot(111)

        self.ax.hist(
            data_dict["bins"][inc][:-1], bins=data_dict["bins"][inc],
            weights=data_dict["hist"][inc],
            color="tab:blue", edgecolor="k",
            label=f"{data_dict['label_curve']}",
        )
    
    def _plot_fit(self, data_list: list, inc: int = 0):
        """ Plot fit function """
        
        pass
    
        # if data_dict["fit"]:
        #     centered_bins = (data_dict["bins"][inc][:-1] + data_dict["bins"][inc][1:]) / 2
        #     func_base = func_preview.FitFunction._get_fitting_function(self)["log_normal"]
        #     popt, _, fit, r2 = func_preview.FitFunction._fit_curve(
        #         self,
        #         x=centered_bins, y=data_dict["hist"][inc],
        #         func_base=func_base,
        #         )
        #     self.ax.plot(
        #         centered_bins, fit,
        #         color="tab:red", label="Fitting function"
        #         )
        #     print(f"r2 = {r2:.2f}")
        #     median = np.exp(popt[0])
        #     std = np.sqrt((np.exp(popt[1]) - 1) * np.exp(2 * popt[0] + popt[1]))
        #     print(f"$\\mu$={median:.2e}, $\\sigma${std:.2e}")
    
    def _plot_d_50(self, data_dict: dict, inc: int = 0):
        """ Plot d_50 """
        
        if data_dict["d_50"][inc]:
            self.ax.axvline(
                x=data_dict["d_50"][inc],
                color="tab:purple",
                linewidth=4,
                label=f"Median diameter $d_{{50}}$={data_dict['d_50'][inc][0]:.2f} $\\mu m$",
            )
    
    def _plot_curves(self, data_dict: dict, inc: int = 0):
        """ Plot curves """

        self.ax = self.fig.add_subplot(111)
        
        if "label" in data_dict:
            if len(data_dict["label"]) <= 10: colors = cm.get_cmap("tab10")
            elif len(data_dict["label"]) <= 20: colors = cm.get_cmap("tab20")
            else: colors = cm.get_cmap("plasma")
            n = len(data_dict["label"])
        
        if "run" in data_dict:
            if len(data_dict["run"]) <= 10: colors = cm.get_cmap("tab10")
            elif len(data_dict["run"]) <= 20: colors = cm.get_cmap("tab20")
            else: colors = cm.get_cmap("plasma")
            n = len(data_dict["run"])

        for i in range(n):

            data_x = data_dict["x"][i]
            data_y = data_dict["y"][i]

            # # delete duplicates
            # data_x = np.unique(data_x, axis=0)
            # data_y = np.unique(data_y, axis=0)

            self.ax.plot(
                data_x * data_dict["x_unit"],
                data_y * data_dict["y_unit"],
                color=colors(i),
                label=f"Run {i}" if "run" in data_dict else f"Label {data_dict['label'][i][0]}",
            )

            if data_dict['label'][i][0] == 13:
                self.ax.plot(
                    data_x * data_dict["x_unit"],
                    [0.440] * len(data_x),
                    color="royalblue",
                )
                self.ax.plot(
                    data_x * data_dict["x_unit"],
                    [0.880] * len(data_x),
                    color="royalblue",
                )

            if data_dict['label'][i][0] == 23:
                self.ax.plot(
                    data_x * data_dict["x_unit"],
                    [0.368] * len(data_x),
                    color="darkorange",
                )
                self.ax.plot(
                    data_x * data_dict["x_unit"],
                    [0.796] * len(data_x),
                    color="darkorange",
                )

            if "fit" in data_dict:
                fit = data_dict["fit"][i]
                self.ax.plot(
                    data_x * data_dict["x_unit"],
                    fit * data_dict["y_unit"],
                    color="tab:red",
                    label=f"Fit label {i}",
                )

    def _plot_scatter(self, data_dict: dict):
        """ Plot scatter """

        self.ax = self.fig.add_subplot(111)

        if "label" in data_dict:
            if len(data_dict["label"]) <= 10: colors = cm.get_cmap("tab10")
            elif len(data_dict["label"]) <= 20: colors = cm.get_cmap("tab20")
            else: colors = cm.get_cmap("plasma")
            n = len(data_dict["label"])
        
        if "run" in data_dict:
            if len(data_dict["run"]) <= 10: colors = cm.get_cmap("tab10")
            elif len(data_dict["run"]) <= 20: colors = cm.get_cmap("tab20")
            else: colors = cm.get_cmap("plasma")
            n = len(data_dict["run"])
        
        for i, lbl in enumerate(data_dict["label"]):
            scatter = self.ax.scatter(
                data_dict["x"][i], data_dict["y"][i],
                color="tab:10",
                label=f"Label {lbl}",
                )
            
            
            cursor = mplcursors.cursor(scatter, hover=True)
            
            @cursor.connect("add")
            def on_add(sel):
                index = sel.index
                sel.annotation.set_text(
                    f"Image number : {lbl:d}, t : {data_dict['time'].iloc[index]*1000:.3f} $ms$\n"
                    f"Label : {data_dict['label'].iloc[index]}\n"
                    f"X : {data_dict['x'].iloc[index]:.2f} $m$, Y : {data_dict['y'].iloc[index]:.2f} $m$\n"
                    f"Number of particles pixels {data_dict['coords_pixels']:.0f} $px$\n"
                    f"Diameter {data_dict['diameter_mean'].iloc[index]:.2f} $m$\n"
                    f"m : {data_dict['mass_mean'].iloc[index]:.2e} kg \u00b1 {data_dict['mass_std'].iloc[index]:.2e}\n"
                    f"v : {data_dict['velocity'].iloc[index]:.2e} $m.s^{{-1}}$\n"
                    f"a : {data_dict['acceleration'].iloc[index]:.2e} $m.s^{{-2}}$\n"
                    f"p : {data_dict['momentum'].iloc[index]:.2e} $kg m.s^{{-1}}$\n"
                    f"K : {data_dict['kinetic'].iloc[index]:.2e} $J$\n"
                    # f"Collision : {sub_df['collision'].iloc[index]}"
                    # f"cluster_id : {sub_df['cluster_id'].iloc[index]}"
                )
                sel.annotation.get_bbox_patch().set(alpha=0.8, color="lightblue")
                sel.annotation.arrow_patch.set(
                    arrowstyle="simple", fc="white", alpha=0.5
                )
                
    def _format_axes(self, data_dict: dict):
        """ Plot axis """
        
        # x_axis_log
        if data_dict["x_log"]:
            self.ax.set_xscale("symlog")

        # ticks and labels
        # x_ticks = [0, 1]
        # self.ax.set_xticks(x_ticks)
        # self.ax.set_xticklabels([f"{x_tick:.2f}" for x_tick in x_ticks], fontsize=self.dict_fontsize["ticks"])
        # self.ax.set_xlabel(data_dict["x_label"], fontsize=self.dict_fontsize["label"])
        # print(x_ticks)
        
        # y_ticks = self.ax.get_yticks()[1:]
        # self.ax.set_yticks(y_ticks)
        # self.ax.set_yticklabels([f"{y_tick:.2f}" for y_tick in y_ticks], fontsize=self.dict_fontsize["ticks"])
        # self.ax.set_ylabel(data_dict["y_label"], fontsize=self.dict_fontsize["label"])
        # print(y_ticks)

        if "z_label" in data_dict:
            
            x_ticks = np.linspace(0, np.max(data_dict["x"])-1, np.max(data_dict["x"]), dtype=int) # self.ax.get_xticks()
            self.ax.set_xticks(x_ticks)
            self.ax.set_xticklabels([f"{x_tick:.0f}" for x_tick in x_ticks], fontsize=self.dict_fontsize["ticks"])
            self.ax.set_xlabel(data_dict["x_label"], fontsize=self.dict_fontsize["label"])
            print(x_ticks)
            
            y_ticks = self.ax.get_yticks()[1:-1]
            self.ax.set_yticks(y_ticks)
            self.ax.set_yticklabels([f"{y_tick:.2f}" for y_tick in y_ticks], fontsize=self.dict_fontsize["ticks"])
            self.ax.set_ylabel(data_dict["z_label"], fontsize=self.dict_fontsize["label"])
            print(y_ticks)

            z_ticks = self.ax.get_zticks()
            # if data_dict["nomalize"]:
            #     z_ticks = np.linspace(0, 1.0, 5, dtype=float)
            self.ax.set_zticks(z_ticks)
            self.ax.set_zticklabels([f"{z_tick:.2f}" for z_tick in z_ticks], fontsize=self.dict_fontsize["ticks"])
            self.ax.set_zlabel(data_dict["y_label"], fontsize=self.dict_fontsize["label"])
            print(z_ticks)
        
        else:

            x_ticks = self.ax.get_xticks()[1:]
            self.ax.set_xticks(x_ticks)
            self.ax.set_xticklabels([f"{x_tick:.3f}" for x_tick in x_ticks], fontsize=self.dict_fontsize["ticks"])
            self.ax.set_xlabel(data_dict["x_label"], fontsize=self.dict_fontsize["label"])
            print(x_ticks)
            
            y_ticks = self.ax.get_yticks()[1:]
            self.ax.set_yticks(y_ticks)
            self.ax.set_yticklabels([f"{y_tick:.2f}" for y_tick in y_ticks], fontsize=self.dict_fontsize["ticks"])
            self.ax.set_ylabel(data_dict["y_label"], fontsize=self.dict_fontsize["label"])
            print(y_ticks)

        
        _, labels_legend = self.ax.get_legend_handles_labels()
        if labels_legend:
            self.ax.legend(fontsize=self.dict_fontsize["legend"])

class SupportFunctions:
    def __init__(self):
        pass
    
    def _load_image(
        self,
        name:str|Path,
        invert:bool=True,
        rotate_image:int=None,
        crop:list=None,
        ):

            img = Image.open(name).convert("L")

            if invert:
                img = ImageOps.invert(img)
            img = np.array(img, dtype=np.float32)

            if crop is not None:
                img = img[
                    self.cropping_image[0]:self.cropping_image[1],
                    self.cropping_image[2]:self.cropping_image[3],
                ]
            
            if rotate_image is not None:
                    img = Image.fromarray(img)
                    if rotate_image == 90: img = img.transpose(Image.ROTATE_90)
                    elif rotate_image == 180: img = img.transpose(Image.ROTATE_180)
                    elif rotate_image == 270: img = img.transpose(Image.ROTATE_270)
            
            return np.array(img, dtype=np.uint8)
    
    def Read_velocity_file(self, path_files, num_velocity=100):

        list_files = [
            Path(file)
            for file in Path(path_files).iterdir()
            if file.is_file() and file.suffix[1:] in ("csv")
        ]
        time_stamp, voltage = [], []
        for file in list_files:
            with file.open("r", newline="") as f:
                reader = list(csv.reader(f))
                volt = []
                for line in range(1, num_velocity + 1):
                    volt.append(float(reader[-line][1]))
                volt = volt[::-1]
                time_stamp.append(float(file.name.split("-")[-1].split(".")[0]))
                voltage.append(np.mean(volt))
        return np.array(time_stamp), np.array(voltage)
    
    def _load_velocity(
        self,
        path:str|Path=None,
        ):
        
        if path is None:
            msg = "path must not empty "
            raise TypeError(msg)
        
        if isinstance(path, list|np.ndarray):
            data = []
            for p in path:
                if not Path(p).exists:
                    msg = f"file '{Path(p).name}' must exist"
                    raise FileExistsError(msg)
                data.append(pd.DataFrame(pd.read_csv(p)))
        elif isinstance(path, Path|str):
            if not Path(path).exists:
                msg = f"file '{Path(path).name}' must exist"
                raise FileExistsError(msg)
            data = pd.DataFrame(pd.read_csv(path))

        return data
    
    def _load_dataframe(
        self,
        path:str|Path,
        usecols:str|list=None,
        change_main_path_images:str=None,
        ):
        
        if path is None:
            msg = "path must not empty "
            raise TypeError(msg)
        
        if not Path(path).exists:
            msg = f"file {Path(path).name} must exist"
            raise FileExistsError(msg)
        
        dataframe = pd.DataFrame(pd.read_csv(path, usecols=usecols))
        if self.change_main_path_images:
            dataframe["main_path"] = [self.change_main_path_images][0] * len(dataframe["main_path"])
        if "Unnamed: 0" in dataframe.columns:
            dataframe = dataframe.drop(["Unnamed: 0"], axis=1)
        
        if change_main_path_images:
            dataframe["main_path"] = [change_main_path_images] * len(dataframe["main_path"])
        
        if "frame" in dataframe.columns:
            dataframe = dataframe.sort_values("frame")
        
        return dataframe
    
    def _check_keys(self, dataframe, keys):
        for key in keys:
            if not key in dataframe.columns:
                msg = f"{key} not un dataframe"
                raise KeyError(msg)

class ThreadWorker(QObject):
    
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self._interrupted = False
    
    def _stop(self):
        self.interrupted = True
        
    def _run(self):
        """ Run function with parameters """
        
        try:
            result = self.func(*self.args, **self.kwargs)
            
            if not self.interrupted:
                self.finished.emit(result)
            
        except Exception as e:
            self.error.emit(str(e))

class ThreadRunner:
    
    def __init__(self, func, *args, on_result=None, on_error=None, **kwargs):
        
        self.thread = QThread()
        self.worker = ThreadWorker(func, *args, **kwargs)
        
        self.worker.moveToThread(self.thread)
        
        self.thread.started.connect(self.worker._run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        
        if on_result:
            self.worker.finished.connect(on_result)
            
        if on_error:
            self.worker.finished.connect(on_error)
    
    def _start(self):
        """ Start thread """
        
        self.thread.start()
    
    def _stop(self):
        """ Stop thread """
        
        if self.thread.isRunning():
            self.thread.stop()
            self.thread.quit()
            self.thread.wait()