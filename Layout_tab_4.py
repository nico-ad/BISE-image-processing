# %% Import

import sys
import inspect
from PyQt5 import uic
from PyQt5.QtCore import Qt, QThread, QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow
from PyQt5.QtWidgets import (
    QTabWidget,
    QPushButton,
    QDial,
    QCheckBox,
    QLineEdit,
    QProgressBar,
    QComboBox,
    QLabel,
    QSpinBox,
)
from PyQt5.QtWidgets import (
    QTableWidget,
    QTableWidgetItem,
    QFileDialog,
    QTableView,
    QListView,
    QTreeView,
    QAbstractItemView,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QSizePolicy,
    QDialog,
    QHeaderView,
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
import os

import matplotlib as mpl
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

        if not hasattr(self.parent, "files_list_dataframe"):
            self.parent.files_list_dataframe = []

        if not hasattr(self.parent, "dataframe_cache"):
            self.parent.dataframe_cache = {}

        self.velocity_profile = []

        self.visualization_func = Visualization.VisualizationFunctions(parent=self)
        self.fitting_func = func_preview.FitFunction()

        self.function_map = {
            "Histogram diameters": {
                "func": self.visualization_func.Visualize_histogram_diameters,
                "settings": HistogramSettings(),
            },
            "Number of particles": {
                "func": self.visualization_func.Visualize_num_part_per_frames,
                "settings": NumberParticlesSettings(),
            },
            "Number of labels": {
                "func": self.visualization_func.Visualize_num_labels,
                "settings": NumberLabelsSettings(),
            },
            "Mean diameter": {
                "func": self.visualization_func.Visualize_mean_diameter,
                "settings": MeanDiameterSettings(),
            },
            "Mean inter-particle distance": {
                "func": self.visualization_func.Visualize_mean_inter_particle_distance,
                "settings": InterParticleDistanceSettings(),
            },
            "Mean free path": {
                "func": self.visualization_func.Visualize_mean_free_path,
                "settings": MeanFreePathParticleSettings(),
            },
            "Coordination number": {
                "func": self.visualization_func.Visualize_coordination_number,
                "settings": CoordinationNumberSettings(),
            },
            "Particle density": {
                "func": self.visualization_func.Visualize_density,
                "settings": DensitySettings(),
            },
            "Surface concentration": {
                "func": self.visualization_func.Visualize_surface_concentration,
                "settings": SurfaceConcentrationSettings(),
            },
            "Resuspended fraction": {
                "func": self.visualization_func.Visualize_resuspended_fraction,
                "settings": ResuspendedFractionSettings(),
            },
            "Collision frequency": {
                "func": self.visualization_func.Visualize_collision_frequency,
                "settings": CollisionFrequencySettings(),
            },
            "Remaining fraction": {
                "func": self.visualization_func.Visualize_remaining_fraction,
                "settings": RemainingFractionSettings(),
            },
            "Velocity flow": {
                "func": self.visualization_func.Visualize_velocity_flow,
                "settings": VelocityFlowSettings(),
            },
            "Friction velocity": {
                "func": self.visualization_func.Visualize_friction_velocity,
                "settings": FrictionVelocitySettings(),
            },
            "Velocity": {
                "func": self.visualization_func.Visualize_velocity,
                "settings": ParticlesVelocitySettings(),
            },
            "Acceleration": {
                "func": self.visualization_func.Visualize_acceleration,
                "settings": ParticleAccelerationSettings(),
            },
            "Momentum": {
                "func": self.visualization_func.Visualize_momentum,
                "settings": ParticleMomentumSettings(),
            },
            "Kinetic energy": {
                "func": self.visualization_func.Visualize_kinetic_energy,
                "settings": ParticleKineticEnergySettings(),
            },
            "Mean square displacement": {
                "func": self.visualization_func.Mean_square_displacement,
                "settings": ParticleMeanSquareDisplacementSettings(),
            },
            "Particles dectection": {
                "func": self.visualization_func.Visualize_particle_detection,
                "settings": ParticleDetectionSettings(),
            },
            "Clusters": {
                "func": self.visualization_func.Visualize_cluster,
                "settings": ParticleClusterSettings(),
            },
            "Position tracking particles": {
                "func": self.visualization_func.Track_particles_position,
                "settings": ParticleTrackingPositionSettings(),
            },
            "Velocity tracking particles": {
                "func": self.visualization_func.Track_particles_velocity,
                "settings": ParticleTrackingVelocitySettings(),
            },
            "Visualize labels": {
                "func": self.visualization_func.Visualize_labels,
                "settings": ParticleLabelsSettings(),
            },
            "Voronoi diagram": {
                "func": self.visualization_func.Visualize_Voronoi_triangulation,
                "settings": VoronoiDiagramSettings(),
            },
            "Smooth trajectories": {
                "func": self.visualization_func.Smooth_trajectories,
                "settings": ParticleSmoothTrajectoriesSettings(),
            },
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
        self.parent.parameters_table_tab_4.horizontalHeader().setFont(
            self.parent.font_header
        )
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
                "Smooth trajectories",
                "Histogram diameters",
                "Number of particles",
                "Number of labels",
                "Mean inter-particle distance",
                "Mean free path",
                "Coordination number",
                "Collision frequency",
                "Resuspended fraction",
                "Remaining fraction",
                "Velocity flow",
                "Friction velocity",
                "Particle density",
                "Surface concentration",
                "Mean diameter",
                "Velocity",
                "Acceleration",
                "Momentum",
                "Kinetic energy",
                "Mean square displacement",
                "Visualize labels",
            ],
            "2D display": [
                "Particles detection",
                "Voronoi diagram",
                "Labels",
                "Number of labels per frame",
                "Clusters",
                "Position tracking particles",
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
        
        # ----- Pixel size
        pixelSize_layout = QHBoxLayout()
        self.pixelSize_label = QLabel("Pixel size [mm/px]")
        self.pixelSize_label.setFont(self.parent.font_content)
        self.pixelSize_label.setAlignment(Qt.AlignCenter)
        self.pixelSize_label.setFixedWidth(80)
        pixelSize_layout.addWidget(self.pixelSize_label)

        self.pixelSize_textbox = QLineEdit()
        self.pixelSize_textbox.setPlaceholderText("1")
        self.pixelSize_textbox.setFixedWidth(150)
        pixelSize_layout.addWidget(self.pixelSize_textbox)

        pixelSize_layout.addStretch()
        left_layout.addLayout(pixelSize_layout)

        # ----- Acquisition frequency
        acqFreq_layout = QHBoxLayout()
        self.acqFreq_label = QLabel("Acquisition frequency [Hz]")
        self.acqFreq_label.setFont(self.parent.font_content)
        self.acqFreq_label.setAlignment(Qt.AlignCenter)
        self.acqFreq_label.setFixedWidth(80)
        acqFreq_layout.addWidget(self.acqFreq_label)

        self.acqFreq_textbox = QLineEdit()
        self.acqFreq_textbox.setPlaceholderText("1")
        self.acqFreq_textbox.setFixedWidth(150)
        acqFreq_layout.addWidget(self.acqFreq_textbox)

        acqFreq_layout.addStretch()
        left_layout.addLayout(acqFreq_layout)

        # ----- Options
        self.option_btn = QPushButton(tab_4)
        self.option_btn.setText("Options")
        self.option_btn.setFont(self.parent.font_button)
        self.option_btn.setFixedHeight(50)
        self.option_btn.setFixedWidth(150)
        self.option_btn.setEnabled(True)
        self.option_btn.clicked.connect(self._open_options)
        left_layout.addWidget(self.option_btn)

        left_layout.addStretch()

        # ==========
        # RIGHT PANEL
        # ==========

        # ----- plot area
        self.plot_container = QWidget()
        right_layout = QVBoxLayout(self.plot_container)

        # ----- data name to save
        save_layout = QHBoxLayout()
        self.graph_data_name_textbox = QLineEdit()
        self.graph_data_name_textbox.setPlaceholderText("folder_1/folder_2/graph_data_name.csv")
        self.graph_data_name_textbox.setFixedWidth(150)
        save_layout.addWidget(self.graph_data_name_textbox)

        self.show_legend_checkbox = QCheckBox("Show legend")
        self.show_legend_checkbox.setChecked(True)
        self.show_legend_checkbox.setFont(self.parent.font_content)
        self.show_legend_checkbox.toggled.connect(self._refresh_current_plot)
        save_layout.addWidget(self.show_legend_checkbox)

        save_layout.addStretch()
        right_layout.addLayout(save_layout)

        self.label_filter_title = QLabel("Particle labels")
        self.label_filter_title.setFont(self.parent.font_content)
        self.label_filter_title.setVisible(False)
        right_layout.addWidget(self.label_filter_title)

        self.label_filter_list = QListWidget()
        self.label_filter_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.label_filter_list.setMaximumHeight(180)
        self.label_filter_list.setVisible(False)
        self.label_filter_list.itemChanged.connect(self._refresh_current_plot)
        right_layout.addWidget(self.label_filter_list)

        self.viewer = ImageViewer(
            container_widget=self.plot_container,
            pixel_size=0.006,
        )

        # ----- INCLUDE TO SPLITTER

        splitter.addWidget(left_widget)
        splitter.addWidget(self.plot_container)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 7)

        tab_4_layout.addWidget(splitter)

        # add widget to layout
        self.parent.tabs.addTab(tab_4, "Plot analysis results")

    def _add_files(self):
        """ " Add file for analysis"""

        # select files with dialog box
        dialog = QFileDialog(self.parent)
        dialog.setFileMode(QFileDialog.ExistingFiles)
        dialog.setNameFilter("CSV files (*.csv)")
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)

        if dialog.exec_():
            for file_path in dialog.selectedFiles():
                file_path = Path(file_path)

                # check if file already loaded
                if file_path in self.parent.files_list_dataframe:
                    continue

                # add in table
                row = self.parent.parameters_table_tab_4.rowCount()
                self.parent.parameters_table_tab_4.insertRow(row)

                item = QTableWidgetItem()

                # truncate path display
                file_path_str = str(file_path)
                if len(file_path_str) > 20:
                    display_text = f"{file_path_str[:20]}...{file_path_str[-20:]}"
                else:
                    display_text = file_path

                # create item to store path folder
                item.setText(display_text)
                # store path
                item.setData(Qt.UserRole, file_path)
                self.parent.parameters_table_tab_4.setItem(row, 0, item)

                self.parent.files_list_dataframe.append(file_path)

            self.parent.parameters_table_tab_4.viewport().update()

    def _remove_files(self):
        """Remove file"""

        line = self.parent.parameters_table_tab_4.currentRow()

        if line >= 0:
            item = self.parent.parameters_table_tab_4.item(line, 0)
            file_path = item.data(Qt.UserRole)

            # remove from list
            if file_path in self.parent.files_list_dataframe:
                del self.parent.dataframe_cache[file_path]

            self.parent.parameters_table_tab_4.removeRow(line)

            if hasattr(self, "viewer"):
                self.viewer._clear()

    def _get_dataframe(self, file_path):
        """Return dataframe with intelligent cache"""

        cache = self.parent.dataframe_cache

        # date of modified file
        mtime = os.path.getmtime(str(file_path))

        if file_path not in cache or cache[file_path]["mtime"] != mtime:
            df = self._load_file(file_path)
            cache[file_path] = {
                "df": df,
                "mtime": mtime,
            }
        return cache[file_path]["df"]

    def _load_file(
        self,
        path: str | Path,
        usecols: str | list = None,
        change_main_path_images: str = None,
    ):
        """Load data file"""

        if path is None:
            msg = "path must not empty "
            raise TypeError(msg)

        dataframe = pd.DataFrame(pd.read_csv(path, usecols=usecols))
        if change_main_path_images:
            dataframe["main_path"] = [change_main_path_images][0] * len(
                dataframe["main_path"]
            )
        if "Unnamed: 0" in dataframe.columns:
            dataframe = dataframe.drop(["Unnamed: 0"], axis=1)

        if change_main_path_images:
            dataframe["main_path"] = [change_main_path_images] * len(
                dataframe["main_path"]
            )

        if "frame" in dataframe.columns:
            dataframe = dataframe.sort_values("frame")

        return dataframe

    def _open_options(self):
        """Open option window"""

        if not hasattr(self, "settings"):
            self.settings = None

        dialog = OptionDialog(self, settings=self.settings)

        def _handle_settings_applied(settings, velocity):

            self.velocity_profile = velocity if velocity else None

            # lambda settings:
            if settings:
                data_list = self._run_visualization_function(
                    include_save_data_name=False,
                )
                if data_list is not None:
                    self._display_current_plot(data_list)

        dialog.settings_applied.connect(_handle_settings_applied)
        dialog.exec_()

    def _execute_function(self, item):
        """Launch function for visualization"""

        name = item.text()
        if name.startswith("---"):
            return

        self.function_label.setText(f"Selected function : {name}")

        dataframe = [
            self._get_dataframe(path) for path in self.parent.files_list_dataframe
        ]

        # read frames
        if self.frames_start.text() and self.frames_end.text():
            frames_start = (
                int(self.frames_start.text()) if self.frames_start.text() else None
            )
            frames_end = int(self.frames_end.text()) if self.frames_end.text() else None
            frames = np.linspace(
                frames_start, frames_end, frames_end - frames_start + 1, dtype=int
            )
        else:
            frames = None

        # read labels
        if self.labels_textbox.text():
            labels_text = self.labels_textbox.text().strip().lower()
            labels = self._parse_labels(labels_text)
        else:
            labels = None
        
        # read pixel size
        if self.pixelSize_textbox.text():
            pixelSize = float(self.pixelSize_textbox.text())
        else: pixelSize = 1

        # read acquisition frequency
        if self.acqFreq_textbox.text():
            acqFreq = float(self.acqFreq_textbox.text())
        else: acqFreq = 1

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

            self.option_btn.setEnabled(False)

            self.data_to_plot = {
                "dataframe": dataframe,
                "pixel_size": pixelSize,
                "frames": frames,
                "labels": labels,
                "acquisition_freq": acqFreq,
            }

            # function informations
            self.func = self.function_map[name]["func"]
            self.settings = self.function_map[name]["settings"]

            print(self.func.__name__)
            # print(self.settings.x_axis, self.settings.y_axis)

            # import json
            # with open(self.parent.files_list_dataframe[0] / Path("Parameter_analysis.json"), "r") as file:
            #     parameters = json.load(file)

            if self.func:
                data_list = self._run_visualization_function()
                if data_list is not None:
                    self._display_current_plot(data_list)

        self.option_btn.setEnabled(True)

    def _display_current_plot(self, data_list: list):
        """Display the latest plot data and refresh the scrollable label list."""

        self.last_data_list = data_list
        self._sync_label_filter_list()
        self.viewer._display(
            data_list,
            show_legend=self.show_legend_checkbox.isChecked(),
            visible_labels=self._selected_labels_from_filter(),
        )

    def _sync_label_filter_list(self):
        """Populate the scrollable label list from the current 2D plot data."""

        data_dict = None
        if hasattr(self, "last_data_list"):
            for item in self.last_data_list:
                if item and "image" in item and "label" in item:
                    data_dict = item
                    break

        if not data_dict:
            self.label_filter_title.setVisible(False)
            self.label_filter_list.setVisible(False)
            self.label_filter_list.blockSignals(True)
            self.label_filter_list.clear()
            self.label_filter_list.blockSignals(False)
            return

        labels = sorted({int(label) for label in data_dict["label"]})
        current_selected = self._selected_labels_from_filter()

        self.label_filter_title.setVisible(True)
        self.label_filter_list.setVisible(True)
        self.label_filter_list.blockSignals(True)
        self.label_filter_list.clear()

        for label in labels:
            item = QListWidgetItem(f"Label {label}")
            item.setData(Qt.UserRole, label)
            item.setFlags(
                item.flags()
                | Qt.ItemIsUserCheckable
                | Qt.ItemIsEnabled
                | Qt.ItemIsSelectable
            )
            item.setCheckState(
                Qt.Checked if not current_selected or label in current_selected else Qt.Unchecked
            )
            self.label_filter_list.addItem(item)

        self.label_filter_list.blockSignals(False)

    def _selected_labels_from_filter(self):
        """Return the checked labels in the scrollable legend list."""

        if not hasattr(self, "label_filter_list"):
            return None

        selected = []
        for index in range(self.label_filter_list.count()):
            item = self.label_filter_list.item(index)
            if item.checkState() == Qt.Checked:
                selected.append(int(item.data(Qt.UserRole)))

        return selected

    def _refresh_current_plot(self, *args):
        """Re-render the last plot when the legend checkbox changes."""

        if not hasattr(self, "last_data_list") or not self.last_data_list:
            return

        self.viewer._display(
            self.last_data_list,
            show_legend=self.show_legend_checkbox.isChecked(),
            visible_labels=self._selected_labels_from_filter(),
        )

    def _run_visualization_function(self, include_save_data_name: bool = True):
        """Call the selected visualization with only the arguments it accepts."""

        if not hasattr(self, "func") or self.func is None:
            return None

        kwargs = self._build_visualization_kwargs(
            self.func,
            include_save_data_name=include_save_data_name,
        )

        try:
            return self.func(**kwargs)
        except TypeError as error:
            print(f"{self.func.__name__}: {error}")
            return None

    def _build_visualization_kwargs(
        self,
        func,
        include_save_data_name: bool = True,
    ):
        """Adapt tab inputs to the visualization function signature."""

        parameters = inspect.signature(func).parameters
        loaded_dataframes = [
            self._get_dataframe(path) for path in self.parent.files_list_dataframe
        ]

        kwargs = {}

        if "dataframe" in parameters:
            kwargs["dataframe"] = loaded_dataframes
        if "path_dataframe" in parameters:
            kwargs["path_dataframe"] = list(self.parent.files_list_dataframe)
        if "pixel_size" in parameters:
            kwargs["pixel_size"] = self.data_to_plot["pixel_size"]
        if "frames" in parameters:
            kwargs["frames"] = self.data_to_plot["frames"]
        if "labels" in parameters:
            kwargs["labels"] = self.data_to_plot["labels"]
        if "time_interval" in parameters:
            kwargs["time_interval"] = 1 / self.data_to_plot["acquisition_freq"]
        if "x_unit" in parameters and hasattr(self, "settings") and self.settings:
            kwargs["x_unit"] = self.settings.x_axis
        if "y_unit" in parameters and hasattr(self, "settings") and self.settings:
            kwargs["y_unit"] = self.settings.y_axis

        if "velocity" in parameters:
            if self.velocity_profile:
                kwargs["velocity"] = self.velocity_profile
            elif func.__name__ in {"Visualize_velocity_flow", "Visualize_friction_velocity"}:
                print(f"{func.__name__}: velocity profile is not configured")
                return None

        if include_save_data_name and "save_data_name" in parameters:
            save_name = self.graph_data_name_textbox.text().strip()
            if save_name:
                kwargs["save_data_name"] = save_name

        return kwargs

    def _parse_labels(self, text):
        """Parse labels textbox"""

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
                f"x={event.xdata * self.pixel_size:.2f} mm   "
                f"y={event.ydata * self.pixel_size:.2f} mm   "
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
        """Clear canvas"""

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
        """Give magnitude order of a number"""

        if not isinstance(number, (int, float)):
            number = np.array(number)

        if number == 0:
            return 0

        abs_number = np.abs(number)
        exponent = np.floor(np.log10(abs_number))

        if nearest_power_of_three:
            if exponent < 0.0:
                return 10 ** (3 ** round((exponent - exponent % 3) / 3))
            if exponent > 0.0:
                return 10 ** (3 ** round((exponent - exponent % 3) / 3))

        return 10**exponent

    def _display(self, data_list: list, **kwargs):
        """Diplay data even if 1dD or 2D data"""

        self._clear()
        self.last_data_list = data_list
        self.visible_labels = kwargs.get("visible_labels", None)

        self.show_legend = kwargs.get(
            "show_legend",
            self.show_legend_checkbox.isChecked() if hasattr(self, "show_legend_checkbox") else True,
        )

        self.fig = Figure(figsize=(4, 4), dpi=150)

        for ii, data in enumerate(data_list):
            if not data:
                continue

            if "histogram" in data:
                self.plot_1d(data)
                self._format_axes(data)

            elif "curves" in data:
                self.plot_1d(data)

            elif "hist_3d" in data:
                self.plot_1d(data)
                self._format_axes(data)

            elif "image" in data:
                self.plot_2d(data)

            else:
                msg = "No key word detected"
                raise ValueError(msg)

    def plot_1d(self, data_dict: dict):
        """Display 1d graph"""

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
        """Display 2d graph"""

        self.ax = self.fig.add_subplot(111)

        self.ax.imshow(data_dict["image"], origin="lower", cmap="gray", aspect="auto")
        self.ax.set_aspect("equal", adjustable="box")

        visible_labels = None if self.visible_labels is None else set(self.visible_labels)

        if "label" in data_dict:
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
                lbl_value = int(lbl)
                if visible_labels is not None and lbl_value not in visible_labels:
                    continue

                data_x = data_dict["x"][i]
                data_y = data_dict["y"][i]

                # remove same position to not overload graph
                data_all = np.column_stack((data_x, data_y))
                _, indices = np.unique(data_all, axis=0, return_index=True)
                data_all = data_all[indices]

                self.ax.scatter(
                    data_all[:, 0],
                    data_all[:, 1],
                    label=f"Label {lbl_value}",
                )

                # ----- display vectors
                if "vx" in data_dict.keys() and "vy" in data_dict.keys():
                    from matplotlib.patches import FancyArrowPatch

                    data_arrow_x = data_dict["vx"][i][indices]
                    data_arrow_y = data_dict["vy"][i][indices]

                    norm = np.sqrt(
                        (
                            data_dict["vx"][i][indices] ** 2
                            + data_dict["vx"][i][indices] ** 2
                        )
                    ) * 0.01
                    data_arrow_x = data_dict["vx"][i][indices] / norm
                    data_arrow_y = data_dict["vy"][i][indices] / norm

                    Q = self.ax.quiver(
                        data_all[:, 0],
                        data_all[:, 1],  # [m, mm, px]
                        data_arrow_x,
                        data_arrow_y,  # [m/s, mm/s, px/s]
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
                        Q,
                        X=0.9,
                        Y=0.9,
                        U=1 * mag_order,
                        label=f"{1 * mag_order} m/s",
                        # labelpos="E",
                        # color=colors(i),
                    )

                    legend_elements.append(
                        FancyArrowPatch(
                            (0.1, 0.5),
                            (0.9, 0.8),
                            color=colors(i),
                            mutation_scale=100,
                            arrowstyle="-|>",
                        )
                    )

        elif "voronoi" in data_dict:
            # ----- scatter plot position
            for i, fr in enumerate(data_dict["frames"]):
                data_x = data_dict["x"][i]
                data_y = data_dict["y"][i]

                data_all = np.column_stack((data_x, data_y))
                _, indices = np.unique(data_all, axis=0, return_index=True)
                data_all = data_all[indices]  # * data_dict["unit"]

                self.ax.scatter(
                    data_all[:, 0],
                    data_all[:, 1],
                )

                # ----- voronoi vertices
                vor = data_dict["vor"][i]
                for r in vor.ridge_vertices:
                    if len(r) == 2 and -1 not in r:
                        v0, v1 = vor.vertices[r]
                        self.ax.plot([v0[0], v1[0]], [v0[1], v1[1]], color="gray")
                x_max = data_dict["image"].shape[1]
                y_max = data_dict["image"].shape[0]
                self.ax.set_xlim([0, x_max])
                self.ax.set_ylim([0, y_max])

                # ------ density map
                if data_dict["density_map"] is not None:
                    cmap = plt.cm.coolwarm
                    self.ax.imshow(
                        data_dict["density_map"][i],
                        origin="lower",
                        extent=(data_x.min(), data_x.max(), data_y.min(), data_y.max()),
                        cmap=cmap,
                        aspect="auto",
                        alpha=0.5,
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
        
        number_ticks = 5
        self.ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(number_ticks))
        x_ticks = self.ax.get_xticks()[1:-1]
        self.ax.set_xticks(x_ticks)
        self.ax.set_xticklabels(
                self._number_of_ticks(x_ticks, factor=data_dict["x_unit"], min_decimal=data_dict["min_decimals_x"]),
                fontsize=self.dict_fontsize["ticks"],
            )
        self.ax.set_xlabel(
            data_dict["x_label"], fontsize=self.dict_fontsize["label"]
            )

        self.ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(number_ticks))
        y_ticks = self.ax.get_yticks()[1:-1]
        self.ax.set_yticks(y_ticks)
        self.ax.set_yticklabels(
                self._number_of_ticks(y_ticks, factor=data_dict["y_unit"], min_decimal=data_dict["min_decimals_y"]),
                fontsize=self.dict_fontsize["ticks"],
            )
        self.ax.set_ylabel(
            data_dict["y_label"], fontsize=self.dict_fontsize["label"]
            )

        if self.show_legend:
            self.ax.legend(
                # handles=legend_elements, #[f"{1*mag_order} m/s"]*len(legend_elements),
                fontsize=self.dict_fontsize["legend"],
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
            )
            self.fig.subplots_adjust(right=0.78)

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

    def plot_hist3d(self, data_dict: dict, inc=0):
        """Plot histogram in 3D"""

        self.ax = self.fig.add_subplot(111, projection="3d")

        # collect data
        data_x_all = []
        data_y_all = []
        data_z_all = []

        # print(f"Coord number {data_dict['x']}")
        # print(f"Occurence : {data_dict['y']} ({sum[data_dict['y']]})")
        # print(f"Time : {data_dict['z']}")

        for i in range(len(data_dict["x"])):
            x = np.array(data_dict["x"][i] * data_dict["x_unit"])[:-1] - 0.5
            y = np.array(data_dict["y"][i] * data_dict["y_unit"])
            z = np.array(data_dict["z"][i] * data_dict["z_unit"])

            data_x_all.append(x)
            data_y_all.append(y)
            data_z_all.append(np.full(len(x), z))

        # flatten
        data_x_all = np.concatenate(data_x_all).astype(float)  # coordination number
        data_y_all = np.concatenate(data_y_all).astype(
            float
        )  # number particule / proportion
        data_z_all = np.concatenate(data_z_all).astype(float)  # time / frame

        # number of different coordination numbers
        num_coord_number = len(np.unique(data_x_all))
        print(f"Num coor number {num_coord_number}")

        # # down sample
        # if len(data_y_all) > 100:
        #     downsample = 1
        #     l = np.inf
        #     while l < 100:
        #         l = len(data_x_all[::downsample*num_coord_number])
        #         downsample += 1
        # else: downsample = 1
        # data_x_all = data_x_all[::downsample*num_coord_number]
        # data_y_all = np.array(data_y_all[::downsample*num_coord_number], dtype=float)
        # data_z_all = np.array(data_z_all[::downsample*num_coord_number], dtype=float)
        # print(downsample)
        # print(len(data_x_all), len(data_y_all), len(data_z_all))

        # normalize counts
        if data_dict["nomalize"]:
            for i in range(0, len(data_z_all), num_coord_number):
                idx = np.arange(i, i + num_coord_number, dtype=int)

                data_y_all[idx] = np.array(data_y_all[idx]).astype(float)

                total = np.array([np.sum(data_y_all[idx])] * len(idx), dtype=float)

                if all(total) != 0:
                    data_y_all[idx] = data_y_all[idx] / total
                else:
                    data_y_all[idx] = [0.0] * len(idx)

        z0 = np.full_like(data_y_all, data_z_all)
        dx = np.ones_like(data_x_all) * 0.8
        dy = -data_y_all
        dz = np.ones_like(data_z_all) * (data_z_all[1] - data_z_all[0])

        # print()
        # print(np.min(data_x_all), np.max(data_x_all), data_dict["x_unit"])
        # print(np.min(data_y_all), np.max(data_y_all), data_dict["y_unit"])
        # print(np.min(data_z_all), np.max(data_z_all), data_dict["z_unit"])
        # print()

        # # color by time
        # norm = Normalize(vmin=np.min(data_z_all), vmax=np.max(data_z_all))
        # colors = cm.plasma(norm(data_z_all))

        self.ax.bar3d(
            data_x_all,
            data_z_all,
            data_y_all,
            dx,
            dz,
            dy,
            # shade=True,
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
        """Plot histogram"""

        self.ax = self.fig.add_subplot(111)

        self.ax.hist(
            data_dict["bins"][inc][:-1],
            bins=data_dict["bins"][inc],
            weights=data_dict["hist"][inc],
            color="tab:blue",
            edgecolor="k",
            label=f"{data_dict['label_curve']}",
        )

    def _plot_fit(self, data_list: list, inc: int = 0):
        """Plot fit function"""

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
        """Plot d_50"""

        if data_dict["d_50"][inc]:
            self.ax.axvline(
                x=data_dict["d_50"][inc],
                color="tab:purple",
                linewidth=4,
                label=f"Median diameter $d_{{50}}$={data_dict['d_50'][inc][0]:.2f} $\\mu m$",
            )

    def _plot_curves(self, data_dict: dict, inc: int = 0):
        """Plot curves"""

        self.ax = self.fig.add_subplot(111)

        if "label" in data_dict:
            if len(data_dict["label"]) <= 10:
                colors = cm.get_cmap("tab10")
            elif len(data_dict["label"]) <= 20:
                colors = cm.get_cmap("tab20")
            else:
                colors = cm.get_cmap("plasma")
            n = len(data_dict["label"])

        if "run" in data_dict:
            if len(data_dict["run"]) <= 10:
                colors = cm.get_cmap("tab10")
            elif len(data_dict["run"]) <= 20:
                colors = cm.get_cmap("tab20")
            else:
                colors = cm.get_cmap("plasma")
            n = len(data_dict["run"])

        for i in range(n):
            self.ax.plot(
                data_dict["x"][i],
                data_dict["y"][i],
                color=colors(i),
                linewidth=3,
                label=(
                    f"Run {i}"
                    if "run" in data_dict and len(data_dict["run"]) > 1
                    else f"Label {data_dict['label'][i][0]}"
                    if "label" in data_dict
                    else None
                )
            )

            if "fit" in data_dict:
                self.ax.plot(
                    data_dict["x"][i] * data_dict["x_unit"],
                    data_dict["fit"][i] * data_dict["y_unit"],
                    color="tab:red",
                    label=f"Fit label {i}",
                )

            if "velocity_f" in data_dict:
                if data_dict["velocity_f"] is not None:
                    self.ax.plot(
                        data_dict["x"][i],
                        data_dict["velocity_f"][i],
                        color=colors(i),
                        linewidth=3,
                        linestyle="dashed",
                        label=f"Fluid velocity at $r_p({data_dict['label'][i][0]})$",
                    )

            if "uncertainties" in data_dict:
                if "fit_params" in data_dict:
                    a, b, a_err, b_err = data_dict["fit_params"][i]
                else: a, b= np.polyfit(data_dict["x"][i], data_dict["y"][i], 1)
                print(a, b)
                print(f"r2 = {np.corrcoef(data_dict['x'][i], data_dict['y'][i])[0,1]**2:.3f}")

                fit = a * data_dict["x"][i] + b
                residus = data_dict["y"][i].to_numpy() - fit
                s = np.std(residus, ddof=2)
                y_sup = fit + 3*s # at 95%
                y_inf = fit - 3*s # at 95%
                
                self.ax.plot(
                        data_dict["x"][i],
                        a*data_dict["x"][i]+b,
                        color="tab:red",
                        linewidth=3,
                        label=f"Fit with parameters : a={a:.2e} and b={b:.2e}",
                        )

                self.ax.fill_between(
                    data_dict["x"][i],
                    y_sup, y_inf,
                    label="confidence interval at $\pm 3 \sigma$",
                    linewidth=3,
                    color="k",
                    alpha=0.3,
                    zorder=2,
                    )
                
                self.ax.plot(
                    data_dict["x"][i],
                    y_inf,
                    linewidth=3,
                    color="k",
                    alpha=0.5,
                    zorder=2,
                    )
                
                self.ax.plot(
                    data_dict["x"][i],
                    y_sup,
                    linewidth=3,
                    color="k",
                    alpha=0.5,
                    zorder=2,
                    )

    def _plot_scatter(self, data_dict: dict):
        """Plot scatter"""

        self.ax = self.fig.add_subplot(111)

        if "label" in data_dict:
            if len(data_dict["label"]) <= 10:
                colors = cm.get_cmap("tab10")
            elif len(data_dict["label"]) <= 20:
                colors = cm.get_cmap("tab20")
            else:
                colors = cm.get_cmap("plasma")
            n = len(data_dict["label"])

        if "run" in data_dict:
            if len(data_dict["run"]) <= 10:
                colors = cm.get_cmap("tab10")
            elif len(data_dict["run"]) <= 20:
                colors = cm.get_cmap("tab20")
            else:
                colors = cm.get_cmap("plasma")
            n = len(data_dict["run"])

        for i, lbl in enumerate(data_dict["label"]):
            scatter = self.ax.scatter(
                data_dict["x"][i],
                data_dict["y"][i],
                color="tab:10",
                label=f"Label {lbl}",
            )

            cursor = mplcursors.cursor(scatter, hover=True)

            @cursor.connect("add")
            def on_add(sel):
                index = sel.index
                sel.annotation.set_text(
                    f"Image number : {lbl:d}, t : {data_dict['time'].iloc[index] * 1000:.3f} $ms$\n"
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
        """Plot axis"""

        # x_axis_log
        if "x_log" in data_dict:
            if data_dict["x_log"]: self.ax.set_xscale("symlog")

        if "z_label" in data_dict:
            x_ticks = np.linspace(
                0, np.max(data_dict["x"]) - 1, np.max(data_dict["x"]), dtype=int
            )  # self.ax.get_xticks()
            self.ax.set_xticks(x_ticks)
            self.ax.set_xticklabels(
                [f"{x_tick:.0f}" for x_tick in x_ticks],
                fontsize=self.dict_fontsize["ticks"],
            )
            self.ax.set_xlabel(
                data_dict["x_label"], fontsize=self.dict_fontsize["label"]
            )
            # print(x_ticks)

            y_ticks = self.ax.get_yticks()[1:-1]
            self.ax.set_yticks(y_ticks)
            self.ax.set_yticklabels(
                [f"{y_tick:.2f}" for y_tick in y_ticks],
                fontsize=self.dict_fontsize["ticks"],
            )
            self.ax.set_ylabel(
                data_dict["z_label"], fontsize=self.dict_fontsize["label"]
            )
            # print(y_ticks)

            z_ticks = self.ax.get_zticks()
            if data_dict["nomalize"]:
                z_ticks = np.linspace(0, 1.0, 5, dtype=float)
            self.ax.set_zticks(z_ticks)
            self.ax.set_zticklabels(
                [f"{z_tick:.2f}" for z_tick in z_ticks],
                fontsize=self.dict_fontsize["ticks"],
            )
            self.ax.set_zlabel(
                data_dict["y_label"], fontsize=self.dict_fontsize["label"]
            )
            # print(z_ticks)

        else:
            number_ticks = 5
            self.ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(number_ticks))
            x_ticks = self.ax.get_xticks()[1:]
            self.ax.set_xticks(x_ticks)
            print(x_ticks, data_dict["x_unit"], data_dict["min_decimals_x"], data_dict["x_ticks_sci"])
            self.ax.set_xticklabels(
                self._number_of_ticks(x_ticks, factor=data_dict["x_unit"], min_decimal=data_dict["min_decimals_x"], sci=data_dict["x_ticks_sci"]),
                fontsize=self.dict_fontsize["ticks"],
            )
            self.ax.set_xlabel(
                data_dict["x_label"], fontsize=self.dict_fontsize["label"]
            )

            self.ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(number_ticks))
            y_ticks = self.ax.get_yticks()[1:-1]
            self.ax.set_yticks(y_ticks)
            self.ax.set_yticklabels(
                self._number_of_ticks(y_ticks, factor=data_dict["y_unit"], min_decimal=data_dict["min_decimals_y"], sci=data_dict["y_ticks_sci"]),
                fontsize=self.dict_fontsize["ticks"],
            )
            self.ax.set_ylabel(
                data_dict["y_label"], fontsize=self.dict_fontsize["label"]
            )

        _, labels_legend = self.ax.get_legend_handles_labels()
        if labels_legend and self.show_legend:
            self.ax.legend(fontsize=self.dict_fontsize["legend"])
        
        plt.subplots_adjust(0, 0, 1, 1)

    def _number_of_ticks(self, ticks, factor=1.0, min_decimal=0, number_ticks=5, sci=False):
        """ Adapt number of ticks """

        inc = min_decimal
        while True:
            
            if sci: temp_ticks = [f"{tick*factor:.{inc}e}" for tick in ticks]
            else: temp_ticks = [f"{tick*factor:.{inc}f}" for tick in ticks]

            if len(np.unique(temp_ticks)) == len(ticks):
                return temp_ticks
            
            inc += 1

class SupportFunctions:
    def __init__(self):
        pass

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
        path: str | Path = None,
    ):

        if path is None:
            msg = "path must not empty "
            raise TypeError(msg)

        if isinstance(path, list | np.ndarray):
            data = []
            for p in path:
                if not Path(p).exists:
                    msg = f"file '{Path(p).name}' must exist"
                    raise FileExistsError(msg)
                data.append(pd.DataFrame(pd.read_csv(p)))
        elif isinstance(path, Path | str):
            if not Path(path).exists:
                msg = f"file '{Path(path).name}' must exist"
                raise FileExistsError(msg)
            data = pd.DataFrame(pd.read_csv(path))

        return data

    def _load_dataframe(
        path: str | Path,
        usecols: str | list = None,
        change_main_path_images: str = None,
    ):

        if path is None:
            msg = "path must not empty "
            raise TypeError(msg)

        if not Path(path).exists:
            msg = f"file {Path(path).name} must exist"
            raise FileExistsError(msg)

        dataframe = pd.DataFrame(pd.read_csv(path, usecols=usecols))
        if change_main_path_images:
            dataframe["main_path"] = [change_main_path_images][0] * len(
                dataframe["main_path"]
            )
        if "Unnamed: 0" in dataframe.columns:
            dataframe = dataframe.drop(["Unnamed: 0"], axis=1)

        if change_main_path_images:
            dataframe["main_path"] = [change_main_path_images] * len(
                dataframe["main_path"]
            )

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
        """Run function with parameters"""

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
        """Start thread"""

        self.thread.start()

    def _stop(self):
        """Stop thread"""

        if self.thread.isRunning():
            self.thread.stop()
            self.thread.quit()
            self.thread.wait()


class OptionDialog(QDialog):
    # declare signal
    settings_applied = pyqtSignal(object, object)

    def __init__(self, parent=None, settings=None):
        super().__init__(parent)

        # plot settings
        self.settings = settings

        # ----- Dialog window size
        self.setWindowTitle("Graph options")
        self.resize(400, 300)

        main_layout = QVBoxLayout(self)

        # ----- tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # =========================
        # TAB 1 : Velocity files
        # =========================
        tab1 = QWidget()
        tab1_layout = QVBoxLayout(tab1)

        # velocity storage
        self.velocity_files = []

        # table
        self.table = QTableWidget()
        self.table.setColumnCount(1)
        self.table.setHorizontalHeaderLabels(["Velocity files"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        tab1_layout.addWidget(self.table)

        # load velocity button
        self.load_velocity_btn = QPushButton("Add velocity profile")
        # self.load_velocity_btn.setFont(self.parent.font_button)
        self.load_velocity_btn.setFixedSize(150, 50)
        self.load_velocity_btn.clicked.connect(self._add_velocity)
        tab1_layout.addWidget(self.load_velocity_btn)
        tab1_layout.addWidget(self.load_velocity_btn, alignment=Qt.AlignCenter)

        # load velocity button
        self.remove_velocity_btn = QPushButton("Remove velocity profile")
        # self.load_velocity_btn.setFont(self.parent.font_button)
        self.remove_velocity_btn.setFixedSize(150, 50)
        self.remove_velocity_btn.clicked.connect(self._remove_velocity)
        tab1_layout.addWidget(self.remove_velocity_btn)
        tab1_layout.addWidget(self.remove_velocity_btn, alignment=Qt.AlignCenter)

        # =========================
        # TAB 1 : Figure options
        # =========================
        tab2 = QWidget()
        tab2_layout = QVBoxLayout(tab2)

        if self.settings is not None:

            print(self.settings)

            # ----- X axis
            tab2_layout.addWidget(QLabel("X axis"))
            self.x_combo = QComboBox()
            # add item on combo box
            for label, value in self.settings.x_axis_options.items():
                self.x_combo.addItem(label, value)
            # set current data
            idx = self.x_combo.findData(self.settings.x_axis)
            self.x_combo.setCurrentIndex(idx)
            tab2_layout.addWidget(self.x_combo)

            # ----- Y axis
            tab2_layout.addWidget(QLabel("Y axis"))
            self.y_combo = QComboBox()
            # add item on combo box
            for label, value in self.settings.y_axis_options.items():
                self.y_combo.addItem(label, value)
            # set current data
            idx = self.y_combo.findData(self.settings.y_axis)
            self.y_combo.setCurrentIndex(idx)
            tab2_layout.addWidget(self.y_combo)

            # # ----- Z options
            # if self.settings.z_axis_options:
            #     tab2_layout.addWidget(QLabel("Z axis"))
            #     self.z_combo = QComboBox()
            #     # add item on combo box
            #     for label, value in self.settings.z_axis_options.items():
            #         self.z_combo.addItem(label, value)
            #     # set current data
            #     idx = self.z_combo.findData(self.settings.y_axis)
            #     self.z_combo.setCurrentIndex(idx)
            #     tab2_layout.addWidget(self.z_combo)

            self.x_combo.currentTextChanged.connect(self._update_x)
            self.y_combo.currentTextChanged.connect(self._update_y)

        # ----- Add tabs
        self.tabs.addTab(tab1, "Velocity file")
        self.tabs.addTab(tab2, "Figure")

        # Apply button
        layout = QVBoxLayout()
        self.apply_btn = QPushButton("Apply")
        # self.apply_btn.setFont(self.parent.font_button)
        self.apply_btn.setFixedSize(75, 25)
        self.apply_btn.clicked.connect(self._apply_settings)
        tab2_layout.addWidget(self.apply_btn)
        tab2_layout.addWidget(self.apply_btn, alignment=Qt.AlignRight)

        self.setLayout(main_layout)

    def _add_velocity(self):
        """ " Add one or multiple velocity profiles"""

        # select files with dialog box
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("CSV files (*.csv)")
        file_dialog.setOption(QFileDialog.DontUseNativeDialog, True)

        if file_dialog.exec_():
            for selected_files in file_dialog.selectedFiles():
                velocity_path = Path(selected_files)

                # load velocity profile
                data_velocity = SupportFunctions._load_dataframe(path=velocity_path)
                # if not ["frame", "timestamp", "voltage", "velocity"] in data_velocity.columns:
                #     print("Keyword is missong in veloctiy profile")

                if "friction" not in data_velocity.columns:
                    data_velocity["friction"] = 0.0564 * data_velocity["velocity"] ** (
                        7 / 8
                    )

                # check if file already loaded
                if velocity_path in self.velocity_files:
                    continue

                # add in table
                row = self.table.rowCount()
                self.table.insertRow(row)

                item = QTableWidgetItem()

                # truncate path display
                velocity_path_str = str(velocity_path)
                if len(velocity_path_str) > 30:
                    display_text = (
                        f"{velocity_path_str[:30]}...{velocity_path_str[-30:]}"
                    )
                else:
                    display_text = velocity_path_str

                # create item to store path folder
                item.setText(display_text)
                # store path
                item.setData(Qt.UserRole, velocity_path)
                self.table.setItem(row, 0, item)

                self.velocity_files.append(data_velocity)

            self.table.viewport().update()

        self.settings_applied.emit(self.settings, self.velocity_files)

    def _remove_velocity(self):
        """Remove selected velocity profile"""

        selected_row = self.table.selectionModel().selectedRows()

        if not selected_row:
            return

        for index in sorted(selected_row, reverse=True, key=lambda x: x.row()):
            row = index.row()

            item = self.table.item(row, 0)
            if item is not None:
                velocity_path = item.data(Qt.UserRole)

                if velocity_path in self.velocity_files:
                    self.velocity_files.remove(velocity_path)

            self.table.removeRow(row)

    def _apply_settings(self):
        """Validate figure modifications"""

        # pick up values from UI
        self.settings.x_axis = self.x_combo.currentData()
        self.settings.y_axis = self.y_combo.currentData()

        # redraw figure
        self.settings_applied.emit(self.settings, self.velocity_files)

    def _update_x(self):
        """Update unit in x axis"""
        self.settings.x_axis = self.x_combo.currentData()

    def _update_y(self):
        """Update unit in y axis"""
        self.settings.y_axis = self.y_combo.currentData()


class HistogramSettings:
    """Store configuration for particles histogram"""

    def __init__(self):

        # X axis
        self.x_axis_options = {
            "frames",
            "time",
            "fric_velocity",
            "flow_velocity",
            "reynolds",
        }

        # Y axis
        self.y_axis_options = {
            "px/s",
            "mm/s",
            "m/s",
        }

        # default selection
        self.x_axis = "time"
        self.y_axis = "mm/s"


class NumberParticlesSettings:
    """Store configuration for particles counting"""

    def __init__(self):
        # X axis
        self.x_axis_options = {
            "frames",
            "time",
        }
        # Y axis
        self.y_axis_options = {
            "number",
        }
        # default selection
        self.x_axis = "number"
        self.y_axis = "time"


class NumberLabelsSettings:
    """Store configuration for particles counting labels"""

    def __init__(self):
        # X axis
        self.x_axis_options = {
            "frames",
            "time",
        }
        # Y axis
        self.y_axis_options = {
            "number",
        }
        # default selection
        self.x_axis = "number"
        self.y_axis = "time"


class MeanDiameterSettings:
    """Store configuration for particles mean diameter"""

    def __init__(self):
        # X axis
        self.x_axis_options = {
            "frames",
            "time",
            "fric_velocity",
            "flow_velocity",
            "reynolds",
        }
        # Y axis
        self.y_axis_options = {
            "um",
            "mm",
        }
        # default selection
        self.x_axis = "time"
        self.y_axis = "um"


class InterParticleDistanceSettings:
    """Store configuration for particles inter distance"""

    def __init__(self):
        # X axis
        self.x_axis_options = {
            "px",
            "mm",
            "m",
        }
        # Y axis
        self.y_axis_options = {
            "number",
            "frequency",
        }
        # default selection
        self.x_axis = "mm"
        self.y_axis = "number"


class MeanFreePathParticleSettings:
    """Store configuration for particles mean free path"""

    def __init__(self):
        # X axis
        self.x_axis_options = {
            "px",
            "mm",
            "m",
        }
        # Y axis
        self.y_axis_options = {
            "number",
            "frequency",
        }
        # default selection
        self.x_axis = "mm"
        self.y_axis = "number"


class CoordinationNumberSettings:
    """Store configuration for particles mean free path"""

    def __init__(self):
        # X axis
        self.x_axis_options = {
            "coord_num",
        }
        # Y axis
        self.y_axis_options = {
            "count",
            "frequency",
        }
        # Z axis
        self.z_axis_options = {
            "frames",
            "time_s",
            "time_ms",
        }
        # default selection
        self.x_axis = "coord_num"
        self.y_axis = "frequency"
        self.z_axis = "time_s"

class DensitySettings:
    """Store configuration for particles density"""

    def __init__(self):
        # X axis
        self.x_axis_options = {
            "frames",
            "time",
            "fric_velocity",
            "flow_velocity",
            "reynolds",
        }
        # Y axis
        self.y_axis_options = {
            "/mm2",
            "/m2",
        }
        # default selection
        self.x_axis = "time"
        self.y_axis = "/mm2"

class SurfaceConcentrationSettings:
    """Store configuration for particles surface concentration"""

    def __init__(self):
        # X axis
        self.x_axis_options = {
            "frames",
            "time",
            "fric_velocity",
            "flow_velocity",
            "reynolds",
        }
        # Y axis
        self.y_axis_options = {
            "particle/px2",
            "particle/mm2",
            "particle/m2",
        }
        # default selection
        self.x_axis = "time"
        self.y_axis = "particle/mm2"


class ResuspendedFractionSettings:
    """Store configuration for particles resuspended fraction"""

    def __init__(self):
        # X axis
        self.x_axis_options = {
            "frames",
            "time",
            "fric_velocity",
            "flow_velocity",
            "reynolds",
        }
        # Y axis
        self.y_axis_options = {
            "fraction",
        }
        # default selection
        self.x_axis = "time"
        self.y_axis = "fraction"


class CollisionFrequencySettings:
    """Store configuration for particles collision frequency"""

    def __init__(self):
        # X axis
        self.x_axis_options = {
            "frames",
            "time",
            "fric_velocity",
            "flow_velocity",
            "reynolds",
        }
        # Y axis
        self.y_axis_options = {
            "frequency_s",
            "frequency_ks",
            "frequency_ms",
        }
        # default selection
        self.x_axis = "time"
        self.y_axis = "frequency_ms"


class RemainingFractionSettings:
    """Store configuration for particles remaining fraction"""

    def __init__(self):
        # X axis
        self.x_axis_options = {
            "frames",
            "time",
            "fric_velocity",
            "flow_velocity",
            "reynolds",
        }
        # Y axis
        self.y_axis_options = {
            "fraction",
        }
        # default selection
        self.x_axis = "time"
        self.y_axis = "fraction"


class VelocityFlowSettings:
    """Store configuration for flow velocity"""

    def __init__(self):
        # X axis
        self.x_axis_options = {
            "frames",
            "time",
            "fric_velocity",
            "flow_velocity",
            "reynolds",
        }
        # Y axis
        self.y_axis_options = {
            "mm/s",
            "m/s",
        }
        # default selection
        self.x_axis = "time"
        self.y_axis = "m/s"

class FrictionVelocitySettings:
    """Store configuration for frcition velocity"""

    def __init__(self):
        # X axis
        self.x_axis_options = {
            "frames",
            "time",
            "fric_velocity",
        }
        # Y axis
        self.y_axis_options = {
            "mm/s",
            "m/s",
        }
        # default selection
        self.x_axis = "time"
        self.y_axis = "m/s"


class ParticlesVelocitySettings:
    """Store configuration for particles velocity"""

    def __init__(self):
        # X axis
        self.x_axis_options = {
            "frames": "frames",
            "time": "time",
            "fric_velocity": "fric_velocity",
            "flow_velocity": "flow_velocity",
            "reynolds": "reynolds",
        }
        # Y axis
        self.y_axis_options = {
            "mm/s": "mm/s",
            "m/s": "m/s",
        }
        # default selection
        self.x_axis = "time"
        self.y_axis = "mm/s"


class ParticleAccelerationSettings:
    """Store configuration for particle acceleration"""

    def __init__(self):
        # X axis
        self.x_axis_options = {
            "frames",
            "time",
            "fric_velocity",
            "flow_velocity",
            "reynolds",
        }
        # Y axis
        self.y_axis_options = {
            "px/s2",
            "mm/s2",
            "m/s2",
        }
        # default selection
        self.x_axis = "time"
        self.y_axis = "m/s2"


class ParticleMomentumSettings:
    """Store configuration for particle momentum"""

    def __init__(self):
        # X axis
        self.x_axis_options = {
            "frames",
            "time",
            "fric_velocity",
            "flow_velocity",
            "reynolds",
        }
        # Y axis
        self.y_axis_options = {
            "kg.mm/s",
            "kg.m/s",
        }
        # default selection
        self.x_axis = "time"
        self.y_axis = "kg.m/s"


class ParticleKineticEnergySettings:
    """Store configuration for particle kinetic energy"""

    def __init__(self):
        # X axis
        self.x_axis_options = {
            "frames",
            "time",
            "fric_velocity",
            "flow_velocity",
            "reynolds",
        }
        # Y axis
        self.y_axis_options = {
            "J",
            "uJ",
            "nJ",
            "pJ",
        }
        # default selection
        self.x_axis = "time"
        self.y_axis = "J"


class ParticleMeanSquareDisplacementSettings:
    """Store configuration for particle mean square displacement"""

    def __init__(self):
        # X axis
        self.x_axis_options = {
            "frames",
            "time",
            "fric_velocity",
            "flow_velocity",
            "reynolds",
        }
        # Y axis
        self.y_axis_options = {
            "px",
            "mm",
            "m",
        }
        # default selection
        self.x_axis = "time"
        self.y_axis = "mm"


class ParticleDetectionSettings:
    """Store configuration for particle detection"""

    def __init__(self):
        # X axis
        self.x_axis_options = {
            "frames",
            "time",
            "fric_velocity",
            "flow_velocity",
            "reynolds",
        }
        # Y axis
        self.y_axis_options = {
            "px/s",
            "mm/s",
            "m/s",
        }
        # default selection
        self.x_axis = "time"
        self.y_axis = "mm/s"


class ParticleClusterSettings:
    """Store configuration for particle clustering"""

    def __init__(self):
        # X axis
        self.x_axis_options = {
            "frames",
            "time",
            "fric_velocity",
            "flow_velocity",
            "reynolds",
        }
        # Y axis
        self.y_axis_options = {
            "px/s",
            "mm/s",
            "m/s",
        }
        # default selection
        self.x_axis = "time"
        self.y_axis = "mm/s"


class ParticleTrackingPositionSettings:
    """Store configuration for particle tracking position"""

    def __init__(self):
        # X axis
        self.x_axis_options = {
            "px",
            "mm",
            "m",
        }
        # Y axis
        self.y_axis_options = {
            "px",
            "mm",
            "m",
        }
        # default selection
        self.x_axis = "mm"
        self.y_axis = "mm"


class ParticleTrackingVelocitySettings:
    """Store configuration for particle tracking velocity"""

    def __init__(self):
        # X axis
        self.x_axis_options = {
            "px",
            "mm",
            "m",
        }
        # Y axis
        self.y_axis_options = {
            "px",
            "mm",
            "m",
        }
        # default selection
        self.x_axis = "mm"
        self.y_axis = "mm"


class ParticleLabelsSettings:
    """Store configuration for particle labels"""

    def __init__(self):
        # X axis
        self.x_axis_options = {
            "frames",
            "time",
            "fric_velocity",
            "flow_velocity",
            "reynolds",
        }
        # Y axis
        self.y_axis_options = {
            "px/s",
            "mm/s",
            "m/s",
        }
        # default selection
        self.x_axis = "time"
        self.y_axis = "mm/s"


class VoronoiDiagramSettings:
    """Store configuration for Voronoi diagram"""

    def __init__(self):
        # X axis
        self.x_axis_options = {
            "px",
            "mm",
            "m",
        }
        # Y axis
        self.y_axis_options = {
            "px",
            "mm",
            "m",
        }
        # default selection
        self.x_axis = "mm"
        self.y_axis = "mm"


class ParticleSmoothTrajectoriesSettings:
    """Store configuration for particle smooth trajecotries"""

    def __init__(self):
        # X axis
        self.x_axis_options = {
            "frames",
            "time",
            "fric_velocity",
            "flow_velocity",
            "reynolds",
        }
        # Y axis
        self.y_axis_options = {
            "px/s",
            "mm/s",
            "m/s",
        }
        # default selection
        self.x_axis = "time"
        self.y_axis = "mm/s"


# %%
