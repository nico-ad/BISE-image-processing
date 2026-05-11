
#%% Import

import sys
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
    QSplitter,
    QGroupBox,
    QGridLayout,
    QStackedWidget,
    QHeaderView,
)
from PyQt5.QtWidgets import (
    QStyledItemDelegate,
    QVBoxLayout,
    QHBoxLayout,
)
from PyQt5.QtGui import QIcon, QFont, QPixmap, QImage, QPainter, QColor

import pyqtgraph as pg

import numpy as np
import pandas as pd
from tkinter import Tcl
from pathlib import Path
from PIL import (
    Image,
    ImageOps,
    ImageQt,
)

import scipy
from scipy.ndimage import (
    label,
    distance_transform_edt,
    gaussian_filter,
    binary_fill_holes,
)
from skimage.measure import (
    regionprops,
)
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
from scipy.integrate import quad

from skimage import (
    filters,
    morphology,
    measure,
    segmentation,
)
from skimage.segmentation import (
    watershed,
    clear_border,
)

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import ast

from functools import partial

import Support_functions_preview as func_preview

import Layout_tab_2

class HelperTab1(QWidget):
    
    folders_changed = pyqtSignal(list)
    files_changed = pyqtSignal(list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.parent = parent
        
        self.analyser_class = func_preview.ParticleAnalyser(parent=self)
        self.video_Maker = func_preview.VideoMaker()
        self.spinner = func_preview.Spinner(parent=self)
        
        self.parent.dial_steps = [1, 10, 100, 1000, 10000]
        
        self.parent.image_extensions = (
            ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff",
        )
        
        self.parent.default_params = {
            "images range": ([0, 21837], ""),
            "frequency acquisition": (8000, "Hz"),
            "circularity thresh": ([0.2, 1.0], ""),
            "small objects": (20, "px"),
            "enable subpixel detection": (False, ""),
            "invert grayscale": (True, ""),
            "pixel size": (f"{1/146:.5e}", "mm/px"),
            "number of CPU": (10, ""),
            "number of CPU per image": (1, ""),
            "image format": (".jpg", ""),
            "video sequence": (True, ""),
            "rotate image": ("NONE", ""),
            "h maxima": (0.5, ""),
        }
        
        self.parent.files = []
        
        tab_1 = QWidget()
        main_layout = QVBoxLayout(tab_1)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)
        
        # ==========
        # TITLE
        # ==========
        self.title_page_1 = QLabel("Setup parameters for image analysis and preview particle's detection")
        self.title_page_1.setAlignment(Qt.AlignCenter)
        self.title_page_1.setFont(self.parent.font_title)
        self.title_page_1.setMinimumHeight(45)
        # self.addWidget(self.title_page_1)
        
        # ==========
        # MAIN SPLITTER (SIDEBAR + WORKSPACE)
        # ==========
        content_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(content_splitter, 1)
        
        # ==========
        # SIDEBAR (LEFT)
        # ==========
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setSpacing(12)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)
        
        sidebar.setMinimumWidth(240)
        sidebar.setMaximumWidth(300)
        
        # ----- Load folder table
        self.parent.load_folders_names = QTableWidget(tab_1)
        self.parent.load_folders_names.setColumnCount(1)
        self.parent.load_folders_names.setHorizontalHeaderLabels(["Select files"])
        self.parent.load_folders_names.horizontalHeader().setFont(self.parent.font_header)
        self.parent.load_folders_names.setColumnWidth(0, 250)
        
        # # ----- save files table
        # self.parent.save_name_files = QTableWidget(tab_1)
        # self.parent.save_name_files.setColumnCount(1)
        # self.parent.save_name_files.setHorizontalHeaderLabels(["Select files"])
        # self.parent.save_name_files.horizontalHeader().setFont(self.parent.font_header)
        # self.parent.save_name_files.setColumnWidth(0, 250)
        
        # ----- Data group
        data_group = QGroupBox("Data")
        data_group_layout = QVBoxLayout()
        data_group.setLayout(data_group_layout)
        
        # folder tables
        data_group_layout.addWidget(self.parent.load_folders_names)
        # data_group_layout.addWidget(self.parent.save_name_files)
        
        # buttons
        self.add_folders_btn = QPushButton("Add folders")
        self.add_folders_btn.setFont(self.parent.font_button)
        self.add_folders_btn.setMinimumHeight(40)
        self.add_folders_btn.clicked.connect(self._add_folders)
        
        self.remove_folders_btn = QPushButton("Remove folders")
        self.remove_folders_btn.setFont(self.parent.font_button)
        self.remove_folders_btn.setMinimumHeight(40)
        self.remove_folders_btn.clicked.connect(self._remove_folders)
        
        data_group_layout.addWidget(self.add_folders_btn)
        data_group_layout.addWidget(self.remove_folders_btn)
        
        # ----- CONTROL GROUP
        control_group = QGroupBox("Controls")
        control_layout = QVBoxLayout()
        control_group.setLayout(control_layout)
        
        self.option_btn = QPushButton("Options")
        self.option_btn.setFont(self.parent.font_button)
        self.option_btn.setMinimumHeight(40)
        control_layout.addWidget(self.option_btn)
        self.option_btn.clicked.connect(self._option_dialog)

        self.preview_btn = QPushButton("Preview")
        self.preview_btn.setFont(self.parent.font_button)
        self.preview_btn.setMinimumHeight(40)
        control_layout.addWidget(self.preview_btn)
        self.preview_btn.clicked.connect(self._previewAnalysis)
        
        self.change_folder_btn = QPushButton("Change folder")
        self.change_folder_btn.setFont(self.parent.font_button)
        self.change_folder_btn.setMinimumHeight(40)
        control_layout.addWidget(self.change_folder_btn)
        self.change_folder_btn.clicked.connect(self._changeFolder)
        
        nav_layout = QHBoxLayout()
        
        self.previous_btn = QPushButton("Previous")
        self.previous_btn.setFont(self.parent.font_button)
        self.previous_btn.setMinimumHeight(40)
        self.previous_btn.clicked.connect(self._previousImage)
        
        self.next_btn = QPushButton("Next")
        self.next_btn.setFont(self.parent.font_button)
        self.next_btn.setMinimumHeight(40)
        self.next_btn.clicked.connect(self._nextImage)
        
        nav_layout.addWidget(self.previous_btn)
        nav_layout.addWidget(self.next_btn)
        
        # video button
        self.video_btn = QPushButton("Video")
        self.video_btn.setFont(self.parent.font_button)
        self.video_btn.setMinimumHeight(40)
        control_layout.addWidget(self.video_btn)
        self.video_btn.clicked.connect(self._make_video)
        
        # progress bar video creation
        self.progress_video_creation = QProgressBar(self.video_btn)
        self.progress_video_creation.setGeometry(
            0, 0,
            self.video_btn.width(), self.video_btn.height(),
        )
        self.progress_video_creation.setStyleSheet("""
        QProgressBar {
            background: rgba(0, 0, 0, 120);
            color: white;
            border: none;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #05B8CC;
            }
        """)
        self.progress_video_creation.setValue(0)
        self.progress_video_creation.hide()

        # # overlay video button + progress bar video creation
        # overlay_widget = QWidget()
        # overlay_layout = QVBoxLayout(overlay_widget)
        # overlay_layout.addWidget(self.video_btn)
        # overlay_layout.addWidget(self.progress_video_creation)

        # self.stacked_widget = QStackedWidget()
        # self.stacked_widget.addWidget(overlay_widget)

        control_layout.addLayout(nav_layout)
        
        # dial control
        self.change_magnitude = QDial(tab_1)
        self.change_magnitude.setMinimum(0)
        self.change_magnitude.setMaximum(len((self.parent.dial_steps)) - 1)
        self.change_magnitude.setSingleStep(1)
        self.change_magnitude.setPageStep(1)
        self.change_magnitude.setNotchesVisible(True)
        self.change_magnitude.setWrapping(False)
        control_layout.addWidget(self.change_magnitude)
        
        self.dial_label = QLabel()
        self.dial_label.setAlignment(Qt.AlignCenter)
        self.dial_label.setFont(self.parent.font_button)
        control_layout.addWidget(self.dial_label)
        
        self.change_magnitude.valueChanged.connect(self._update_dial_label)
        self._update_dial_label(self.change_magnitude.value())
        
        # ----- STATS GROUP
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout()
        stats_group.setLayout(stats_layout)
        
        self.num_part_per_frame = QLineEdit("Number of particles detected")
        self.num_part_per_frame.setAlignment(Qt.AlignCenter)
        stats_layout.addWidget(self.num_part_per_frame)
        
        self.density_per_frame = QLineEdit("Number of particles per unit area")
        self.density_per_frame.setAlignment(Qt.AlignCenter)
        stats_layout.addWidget(self.density_per_frame)
        
        self.diameters_in_frame = QLineEdit("Min / Max diameters")
        self.diameters_in_frame.setAlignment(Qt.AlignCenter)
        stats_layout.addWidget(self.diameters_in_frame)
        
        # add to sidebar
        sidebar_layout.addWidget(data_group)
        sidebar_layout.addWidget(control_group)
        sidebar_layout.addWidget(stats_group)
        sidebar_layout.addStretch(1)
        
        # ==========
        # WORKSPACE (RIGHT SIDE)
        # ==========
        workspace = QWidget()
        workspace_layout = QVBoxLayout(workspace)
        # workspace.setLayout(workspace_layout)
        workspace_layout.setSpacing(15)
        
        # # ----- Parameters + checkbox side by side
        # params_checkbox_container = QWidget()
        # params_checkbox_layout = QHBoxLayout(params_checkbox_container)
        # params_checkbox_layout.setSpacing(20)
        
        # # ----- Parameters table
        # param_group = QGroupBox("Parameters")
        # param_layout = QVBoxLayout(param_group)
        # # param_group.setLayout(param_layout)
        
        # combo_columns = {
        #     4: [True, False],
        #     5: [True, False],
        #     10: [True, False],
        #     11: ["None", "ROTATE_90", "ROTATE_180", "ROTATE_270"],
        # }
        
        # self.parent.parameters_table_tab_1 = QTableWidget(tab_1)
        
        # n_rows = len(self.parent.default_params)
        # self.parent.parameters_table_tab_1.setColumnCount(3)
        # self.parent.parameters_table_tab_1.setRowCount(n_rows)
        # self.parent.parameters_table_tab_1.blockSignals(True)
        
        # self.parent.parameters_table_tab_1.setHorizontalHeaderLabels(
        #     ["Parameters", "values", "Units"]
        # )
        
        # for i, (key, (value, unit)) in enumerate(self.parent.default_params.items()):
        #     item_key = QTableWidgetItem(str(key))
        #     item_key.setFlags(item_key.flags() & ~Qt.ItemIsEditable)
        #     self.parent.parameters_table_tab_1.setItem(i, 0, item_key)
            
        #     if i in combo_columns:
        #         combo = QComboBox()
        #         combo.addItems([str(x) for x in combo_columns[i]])
        #         combo.setCurrentText(str(combo_columns[i][0]))
                
        #         combo.currentIndexChanged.connect(self._updatePreview)
        #         self.parent.parameters_table_tab_1.setCellWidget(i, 1, combo)
        #     else:
        #         item_value = QTableWidgetItem(str(value))
        #         item_value.setData(Qt.UserRole, value)
        #         self.parent.parameters_table_tab_1.setItem(i, 1, item_value)
            
        #     unit_display = "" if unit is None else unit
        #     item_unit = QTableWidgetItem(str(unit_display))
        #     item_unit.setFlags(item_unit.flags() & ~Qt.ItemIsEditable)
        #     self.parent.parameters_table_tab_1.setItem(i, 2, item_unit)
        
        # self.parent.parameters_table_tab_1.itemChanged.connect(self._on_item_changed)
        # self.parent.parameters_table_tab_1.blockSignals(False)
        
        # # self.parent.parameters_table_tab_1.horizontalHeader().setStretchLastSection(True)
        # # self.parent.parameters_table_tab_1.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # # self.parent.parameters_table_tab_1.setColumnWidth(0, 200)
        # # self.parent.parameters_table_tab_1.setColumnWidth(1, 200)
        # # self.parent.parameters_table_tab_1.setColumnWidth(2, 200)
        # param_layout.addWidget(self.parent.parameters_table_tab_1)
        
        # # ----- Checkbox grid
        # checkbox_group = QGroupBox("Processing options")
        # checkbox_layout = QGridLayout(checkbox_group)
        # # checkbox_group.setLayout(checkbox_layout)
        
        # self.thresh_method = QCheckBox("Enable threshold")
        # self.thresh_method.setChecked(True)
        # self.thresh_method.stateChanged.connect(self._updatePreview)
        
        # self.fill_holes = QCheckBox("Fill holes")
        # self.fill_holes.setChecked(True)
        # self.fill_holes.stateChanged.connect(self._updatePreview)
        
        # self.clear_border = QCheckBox("Clear border")
        # self.clear_border.setChecked(True)
        # self.clear_border.stateChanged.connect(self._updatePreview)
        
        # self.remove_small = QCheckBox("Remove small particles")
        # self.remove_small.setChecked(True)
        # self.remove_small.stateChanged.connect(self._updatePreview)
        
        # self.identify_particles = QCheckBox("Identify particles")
        # self.identify_particles.setChecked(True)
        # self.identify_particles.stateChanged.connect(self._updatePreview)
        
        # self.label_particles = QCheckBox("Label particles")
        # self.label_particles.setChecked(True)
        # self.label_particles.stateChanged.connect(self._updatePreview)
        
        # checkbox_layout.addWidget(self.thresh_method, 0, 0)
        # checkbox_layout.addWidget(self.fill_holes, 0, 1)
        # checkbox_layout.addWidget(self.clear_border, 1, 0)
        # checkbox_layout.addWidget(self.remove_small, 1, 1)
        # checkbox_layout.addWidget(self.identify_particles, 2, 0)
        # checkbox_layout.addWidget(self.label_particles, 2, 1)
        
        # checkbox_layout.setAlignment(Qt.AlignTop)
        # # checkbox_layout.setSpacing(10)
        
        # self.params_chkbx = {
        #     "threshold": self.thresh_method.isChecked(),
        #     "fill_holes":self.fill_holes.isChecked(),
        #     "clear_border": self.clear_border.isChecked(),
        #     "remove_small": self.remove_small.isChecked(),
        #     "identify_part": self.identify_particles.isChecked(),
        #     "label_part": self.label_particles.isChecked(),
        # }
        
        # params_checkbox_layout.addWidget(param_group, stretch=3)
        # params_checkbox_layout.addWidget(checkbox_group, stretch=1)
        # workspace_layout.addWidget(params_checkbox_container)
        
        # workspace_layout.addStretch()
        
        # ----- preview area
        preview_splitter = QSplitter(Qt.Horizontal)
        
        # image preview
        image_group = QGroupBox("Image preview")
        image_group_layout = QVBoxLayout()
        image_group.setLayout(image_group_layout)
        
        self.image_name = QLineEdit()
        self.image_name.setText("Image name")
        self.image_name.setReadOnly(True)
        self.image_name.setFont(self.parent.font_content)
        self.image_name.setAlignment(Qt.AlignCenter)
        self.image_name.setFixedHeight(35)
        
        self.image_preview_container = QWidget()
        self.image_layout = QVBoxLayout(self.image_preview_container)
        
        image_group_layout.addWidget(self.image_name)
        image_group_layout.addWidget(self.image_preview_container)
        
        # sizing preview
        sizing_group = QGroupBox("Grain sizing")
        self.sizing_preview_container = QWidget()
        sizing_group_layout = QVBoxLayout(self.sizing_preview_container)
        sizing_group.setLayout(sizing_group_layout)
        
        self.sizing_title = QLineEdit("Particle distribution")
        self.sizing_title.setReadOnly(True)
        self.sizing_title.setFont(self.parent.font_content)
        self.sizing_title.setAlignment(Qt.AlignCenter)
        self.sizing_title.setFixedHeight(35)
        
        self.sizing_preview_container = QWidget()
        self.sizing_preview_layout = QVBoxLayout(self.sizing_preview_container)
        
        sizing_group_layout.addWidget(self.sizing_title)
        sizing_group_layout.addWidget(self.sizing_preview_container)
        
        preview_splitter.addWidget(image_group)
        preview_splitter.addWidget(sizing_group)
        
        # # add workspace widgets
        # workspace_layout.addWidget(param_group)
        # workspace_layout.addWidget(checkbox_group)
        # workspace_layout.addWidget(preview_splitter, 1)
        
        # ==========
        # ADD TO MAIN SPLITTER
        # ==========
        content_splitter.addWidget(sidebar)
        content_splitter.addWidget(workspace)
        
        content_splitter.setStretchFactor(0, 0)
        content_splitter.setStretchFactor(1, 1)
        
        # ==========
        # SCIENTIFIC TYPE
        # ==========
        tab_1.setStyleSheet(
            """
            QGroupBox{
                font-weight: bold;
                border: 1px;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 10px;
            }
            
            QGroupBox::title{
                subcontrol-origin: margin;
                left: 12px;
            }
            """
        )
        
        # add widget to layout
        self.parent.tabs.addTab(tab_1, "Setup analysis")

    #%% Functions %%#
    
    def _add_folders(self):
        """ Add folders to list """
        
        # select multiple folders with dialog box
        dialog = QFileDialog(self.parent)
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        # enables multi-selection
        for view in dialog.findChildren((QListView, QTreeView)):
            view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        
        if not dialog.exec_():
            return
        
        if not hasattr(self.parent, "folders_list"):
            self.parent.folders_list = []
        if not hasattr(self.parent, "files_list"):
            self.parent.files_list = []
        
        new_folders = []
        new_files = []
        
        for folder in dialog.selectedFiles():
            
            # check if folder already in table
            if folder in self.parent.folders_list:
                    continue
                
            # check image extension
            images = [
                f for f in Path(folder).iterdir()
                if f.is_file() and f.suffix.lower() in self.parent.image_extensions
                ]
            
            # add in table
            row = self.parent.load_folders_names.rowCount()
            self.parent.load_folders_names.insertRow(row)
            item = QTableWidgetItem()
            
            # truncate path
            display_text = f"{folder[:20]}...{folder[-20:]}" if len(folder) > 20 else folder
                
            # create item to store path folder
            item.setText(display_text)
            item.setData(1000, folder)
            self.parent.load_folders_names.setItem(row, 0, item)
            self.parent.setFont(self.parent.font_content)
            
            new_folders.append(folder)
            new_files.append(images)
            
        self.parent.folders_list.extend(new_folders)
        self.parent.files_list.extend(new_files)
        
        self.parent.load_folders_names.viewport().update()
        
        self.folders_changed.emit(self.parent.folders_list)
        self.files_changed.emit(self.parent.files_list)
    
    def path_in_table(self, table: QTableWidget, path: str | Path) -> bool:
        """ Check path ni table already exists """
        
        for row in range(table.rowCount()):
            item = table.item(row, 0)
            if item and item.data(1000) == path:
                return True
        return False
        
    def _remove_folders(self):
        """" Remove folders from list"""
        
        line = self.parent.load_folders_names.currentRow()
        if line < 0:
            return
        
        if line < len(self.parent.folders_list):
            self.parent.folders_list.pop(line)
            
        if line < len(self.parent.files_list):
            self.parent.files_list.pop(line)
            
        self.parent.load_folders_names.removeRow(line)
        
        self.folders_changed.emit(self.parent.folders_list)
        self.files_changed.emit(self.parent.files_list)

    def _folder_exists(self, folder):
        """ Check if folders exists """
        
        for i in range(self.parent.load_folders_names.rowCount()):
            item = self.parent.load_folders_names.item(i, 0)
            if item and item.data(1000) == folder:
                return True
        return False

    def _load_folders_from_table(self):
        self.parent.folders = []
        for row in range(self.parent.load_folders_names.rowCount()):
            item = self.parent.load_folders_names.item(row, 0)
            if item:
                folder = item.data(1000)
                self.parent.folders.append(folder)
        self.num_folders = len(self.parent.folders)
    
    def _sync_with_table_tab_4(self, item):
        row, col = item.row(), item.column()
        text = item.text()
        
        self.parent.parameters_table_tab_4.blockSignals(True)
        self.parent.parameters_table_tab_4.setItem(row, col, QTableWidgetItem(text))
        self.parent.parameters_table_tab_4.blockSignals(False)

    def _option_dialog(self):
        """ open dialog window """

        dialog = OptionDialog(self, settings=None, fonts=self.parent.fonts)

        def _handle_settings_applied(parameters, options):
            if parameters and options:
                self.analysis_parameters = parameters
                self.preview_options = options

        dialog.settings_applied.connect(_handle_settings_applied)
        dialog.exec_()

    def _previewAnalysis(self):
        """ load and display preview image """
        
        self._load_folders_from_table()
        self.parent.current_folder_index = 0
        self.parent.current_image_index = 0
        self._display_current_folder_image()
    
    def _display_current_folder_image(self):
        """ Load, analyse and display current image of selected folder """
        
        if not bool(self.parent.folders):
            self._reset_image_ui()
            return
        
        if not self._load_current_folder_files():
            self._reset_image_preview()
            return
        
        if not self._load_current_image_path():
            self._reset_image_preview()
            return
        
        self.image_name.setText(self.parent.image_path.name)
        self.image_name.setAlignment(Qt.AlignCenter)
        
        params = self._read_parameters()
        self.pixel_size = params["pixel size"]
        
        self._analyse_image()
        self._update_viewer()
        self._update_statistics()
        self._update_buttons_state()
        
    def _reset_image_ui(self):
        """ Reset image UI """
        
        self.image_path.clear()
        self.image_preview.clear()
        self.preview_analysis.setEnabled(False)
        self.change_folder.setEnabled(False)
        self.next_image.setEnabled(False)
        self.previous_image.setEnabled(False)
        
    def _reset_image_preview(self):
        """ Reset image preview """
        
        self.image_preview.clear()
        self._update_buttons_state()
    
    def _load_current_folder_files(self):
        """ Load images names from folders """
        
        folder_path = Path(self.parent.folders[self.parent.current_folder_index])
        self.valid_format = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        
        self.parent.files = sorted(
            (f for f in folder_path.iterdir() if f.is_file() and f.suffix.lower() in self.valid_format),
            key=lambda p: p.name.lower(),
        )
        
        return bool(self.parent.files)
        
    def _load_current_image_path(self):
        """ Check if image exists """
        
        self.parent.current_image_index = min(self.parent.current_image_index, len(self.parent.files) - 1)
        self.parent.image_path = self.parent.files[self.parent.current_image_index]
        return self.parent.image_path.exists()
    
    def _analyse_image(self):
        """ Apply analysis on current image"""
        
        self.current_params = self._read_parameters()
        
        self.data, self.base_image, self.analysed_image = self.analyser_class._apply_image_modifications(
            filename=self.parent.image_path,
            params=self.current_params,
            chkbx=self.params_chkbx,
        )
    
    def _update_viewer(self):
        """ Process image preview on another thread """
        
        rotation_combo = self.parent.parameters_table_tab_1.cellWidget(10, 1)
        rotation_value = rotation_combo.currentText() if rotation_combo else "None"
        
        self.spinner.start()
        
        self._get_or_create_viewers()
        
        self.viewer_thread = QThread()
        self.viewer_worker = func_preview.ViewerWorker(self.analysed_image, self.data, rotation_value)
        self.viewer_worker.moveToThread(self.viewer_thread)
        
        self.viewer_thread.started.connect(self.viewer_worker.run)
        self.viewer_worker.finished.connect(self._on_viewer_ready)
        self.viewer_worker.finished.connect(self.viewer_thread.quit)
        self.viewer_worker.finished.connect(self.viewer_worker.deleteLater)
        self.viewer_thread.finished.connect(self.viewer_thread.deleteLater)
        
        self.viewer_thread.start()
    
    def _on_viewer_ready(self, img_rot, hist):
        """ Display images preview """
        
        self.spinner.stop()
        
        self.image_viewer.add_img(img_rot, self.data)
        
        if not hasattr(self, "histogram_viewer"):
            self.histogram_viewer = func_preview.HistogramViewer(parent=self.sizing_preview_container, pixel_size=self.pixel_size)
            self.sizing_preview_layout.addWidget(self.histogram_viewer)
        else:
            self.histogram_viewer._draw_histogram()
        diameters = self.data["diameter"].to_numpy()
        if diameters.size > 0:
            self.histogram_viewer._set_data(diameters)
        
    def _get_or_create_viewers(self):
        """ Display images """
        
        if not hasattr(self, "image_viewer"):
            self.image_viewer = func_preview.ImageViewer(self.image_preview_container, pixel_size=self.pixel_size)
        else:
            self.image_viewer.clear()
            
        if not hasattr(self, "sizing_viewer"):
            self.sizing_viewer = func_preview.ImageViewer(self.sizing_preview_container, pixel_size=self.pixel_size)
        else:
            self.sizing_viewer.clear()
    
    def _update_statistics(self):
        """ Update statistics """
        
        num_part = len(self.data)
        density = num_part / (np.prod(self.base_image.shape) * self.pixel_size**2)
        
        self.num_part_per_frame.setText(f"{num_part:d} particles detected")
        self.density_per_frame.setText(f"Particles density {density:.3f} mm^-2")
        
    def _update_buttons_state(self):
        self.previous_btn.setEnabled(self.parent.current_image_index > 0)
        self.next_btn.setEnabled(self.parent.current_image_index < len(self.parent.files) - 1)
        self.change_folder_btn.setEnabled(len(self.parent.files) > 1)

    def _changeFolder(self):
        if len(self.parent.files) > 1:
            self.parent.current_folder_index = (self.parent.current_folder_index + 1) % len(self.parent.folders)
            self.parent.current_image_index = 0
            self._display_current_folder_image()

    def _nextImage(self):
        step = self.parent.dial_steps[self.change_magnitude.value()]
        if self.parent.current_image_index + step < len(self.parent.files):
            self.parent.current_image_index += step
        else:
            self.parent.current_image_index = len(self.parent.files) - 1
        self._display_current_folder_image()

    def _previousImage(self):
        step = self.parent.dial_steps[self.change_magnitude.value()]
        if self.parent.current_image_index - step >= 0:
            self.parent.current_image_index -= step
        else:
            self.parent.current_image_index = 0
        self._display_current_folder_image()

    def _update_dial_label(self, value):
        step = self.parent.dial_steps[value]
        self.dial_label.setText(f"Step : {step}")

    def _updatePreview(self):
        
        if not hasattr(self, "image_viewer") or not hasattr(self, "density_viewer"):
            return
        
        rotation_combo = self.parent.parameters_table_tab_1.cellWidget(10, 1)
        rotation_value = rotation_combo.currentText() if rotation_combo else "None"
        
        self.analyser_class._apply_image_modifications(
            filename=Path(self.parent.folders_list[self.parent.current_folder_index]) / Path(self.parent.files_list[self.parent.current_folder_index][self.parent.current_image_index]),
            params=self.current_params, chkbx=self.params_chkbx,
        )
        # self.image_viewer.clear()
        # self.image_viewer.add_img(self._apply_rotation(self.analysed_image, rotation_value))
        
        # self.density_viewer.clear()
        # self.density_viewer.add_density(self._apply_rotation(self.surface_image, rotation_value))
        
        self._update_buttons_state()

    def _on_item_changed(self, item):
        if item.column() == 1:
            # self.parent.parameters_table_tab_1.blockSignals(True)
            # item.setData(Qt.UserRole, item.text())
            # self.parent.parameters_table_tab_1.blockSignals(False)
            self._updatePreview()

    def _read_parameters(self) -> dict:
        """ Read table of parameters for image analysis """
        
        current_params = {}
        for row in range(self.parent.parameters_table_tab_1.rowCount()):
            # first column
            key = self.parent.parameters_table_tab_1.item(row, 0).text()
            # second column
            widget = self.parent.parameters_table_tab_1.cellWidget(row, 1)
            if isinstance(widget, QComboBox):
                value = widget.currentText()
            else:
                item = self.parent.parameters_table_tab_1.item(row, 1)
                value = item.text() # item.data(Qt.UserRole)
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass
            
            current_params[key] = value
        return current_params

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_overlay()

    def update_overlay(self):
        """ Adjust progress bar to video button """

        self.progress_video_creation.setGeometry(
            0, 0,
            self.video_btn.width(), self.video_btn.height()
        )
    
    def _make_video(self):
        """ Create video from folders images """

        if len(self.parent.folders_list) == 0:
            return
        
        # desabled button
        self.video_btn.setEnabled(False)

        self.current_index = 0
        self.folders = self.parent.folders_list

        self._start_next_video()
    
    def _start_next_video(self):
        """ Start video creation """

        # set viceo frequency
        freq_acq = self._read_parameters()["frequency acquisition"]
        
        # desabled button
        if self.current_index >= len(self.folders):
            self.video_btn.setEnabled(True)
            return
        
        folder = self.folders[self.current_index]

        # show progress bar
        self.progress_video_creation.setValue(0)
        self.progress_video_creation.show()
        self.progress_video_creation.raise_()

        # creat thread for video creation
        self.thread = QThread()

        # creat VideoMaker instance
        # self.worker = self.video_Maker
        self.worker = func_preview.VideoMaker()

        # move worker to thread
        self.worker.moveToThread(self.thread)

        # start video creation
        self.thread.started.connect(lambda: self.worker.Make_video(
            load_images=Path(Path(folder)),
            path_save=Path(Path(folder)).parents[0],
            video_name="Video.avi",
            # images_range="all",
            do_display_time=False,
            freq=freq_acq,
            )
        )

        # connect signals
        self.worker.progress_video_creation.connect(self.progress_video_creation.setValue)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self._on_finished_video_creation)

        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
            
        # start thread
        self.thread.start()

    def _on_finished_video_creation(self):
        """ start new video """
        
        self.progress_video_creation.hide()

        self.thread.quit()
        self.thread.wait()

        self.current_index += 1
        self._start_next_video()
        

class OptionDialog(QDialog):

    # declare signal
    settings_applied = pyqtSignal(object, object)

    def __init__(self, parent=None, settings=None, fonts=None):
        super().__init__(parent)

        self.settings = settings
        self.fonts = fonts

        # initialize default parameters
        self.default_params = {
            "Images range": ([0, 21837], ""),
            "Acquisition frequency ": (8000, "Hz"),
            "Circularity thresh": ([0.2, 1.0], ""),
            "Small objects": (20, "px"),
            "Enable subpixel detection": (False, ""),
            "Invert grayscale": (True, ""),
            "Pixel size": (1/146, "mm/px"),
            "Number of CPU": (10, ""),
            "Number of CPU per image": (1, ""),
            "Image format": (".jpg", ""),
            "Video sequence": (True, ""),
            "Rotate image": ("NONE", ""),
            "H maxima": (0.5, ""),
        }

        # combo boxes
        combo_columns = {
            4: [True, False],
            5: [True, False],
            10: [True, False],
            11: ["None", "ROTATE_90", "ROTATE_180", "ROTATE_270"],
        }

        # ----- dialog layout
        self.setWindowTitle("Graph options")
        self.resize(400, 300)

        main_layout = QVBoxLayout(self)

        # ----- tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # =========================
        # TAB 1 : parameters
        # =========================
        tab1 = QWidget()
        tab1_layout = QVBoxLayout(tab1)

        # table for parameters
        self.table = QTableWidget(tab1)
        n_rows = len(self.default_params)
        self.table.setColumnCount(3)
        self.table.setRowCount(n_rows)
        self.table.blockSignals(True)

        self.table.setHorizontalHeaderLabels(["Parameters", "values", "Units"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        
        # set header type to bold
        for col in range(self.table.columnCount()):
            item = self.table.horizontalHeaderItem(col)
            if item is not None:
                item.setFont(self.fonts["header"])

        # fill table cells
        for i, (key, (value, unit)) in enumerate(self.default_params.items()):
            # parameter name
            item_key = QTableWidgetItem(str(key))
            item_key.setFlags(item_key.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(i, 0, item_key)
            
            # parameter value
            if i in combo_columns:
                combo = QComboBox()
                combo.addItems([str(x) for x in combo_columns[i]])
                combo.setCurrentText(str(value))
                # store value type in UserRole of combobox
                combo.setProperty("default_value", value)
                self.table.setCellWidget(i, 1, combo)
            else:
                # value formatting
                if isinstance(value, float) and value < 1:
                    display_text = f"{value:.3e}"
                else:
                    display_text = str(value)
                item_value = QTableWidgetItem(display_text)
                item_value.setData(Qt.UserRole, value)
                self.table.setItem(i, 1, item_value)
            
            # units
            unit_display = "" if unit is None else str(unit)
            item_unit = QTableWidgetItem(unit_display)
            item_unit.setFlags(item_unit.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(i, 2, item_unit)
        
        self.table.blockSignals(False)

        tab1_layout.addWidget(self.table)
        self.tabs.addTab(tab1, "Parameters")

        # =========================
        # TAB 2 : processing options
        # =========================
        tab2 = QWidget()
        tab2_layout = QVBoxLayout(tab2)

        # group box
        checkbox_group = QGroupBox("Processing options")
        params_chkbx_layout = QVBoxLayout(checkbox_group)

        # create checkbox
        self.thresh_method = QCheckBox("Threshold")
        self.thresh_method.setChecked(True)
        self.thresh_method.setToolTip("Apply binarization threshold on images to remove background")

        self.fill_holes = QCheckBox("Fill holes")
        self.fill_holes.setChecked(True)
        self.fill_holes.setToolTip("After binarization, fill holes inside objects")

        self.clear_border = QCheckBox("Clear border")
        self.clear_border.setChecked(True)
        self.clear_border.setToolTip("Remove objects that touches border")

        self.remove_small = QCheckBox("Remove small objects")
        self.remove_small.setChecked(True)
        self.remove_small.setToolTip("Remove objects that contains less than n pixels")

        self.identify_particles = QCheckBox("Identify particles")
        self.identify_particles.setChecked(True)
        self.identify_particles.setToolTip("Apply contour on detected particles and aggregates")

        self.label_particles = QCheckBox("Label particles")
        self.label_particles.setChecked(True)
        self.label_particles.setToolTip("Label particles")

        # add checkboxes
        for chk in [self.thresh_method, self.fill_holes,
                    self.clear_border, self.remove_small,
                    self.identify_particles, self.label_particles]:
            params_chkbx_layout.addWidget(chk)
        
        tab2_layout.addWidget(checkbox_group)
        self.tabs.addTab(tab2, "Processing options")
        
        # initialize params dictionnary
        self.params_chkbx = {
            "threshold": self.thresh_method.isChecked(),
            "fill_holes": self.fill_holes.isChecked(),
            "clear_border": self.clear_border.isChecked(),
            "remove_small": self.remove_small.isChecked(),
            "identify_part": self.identify_particles.isChecked(),
            "label_part": self.label_particles.isChecked(),
        }

        # =========================
        # Apply button
        # =========================

        self.apply_btn = QPushButton("Apply")
        main_layout.addWidget(self.apply_btn)
        self.apply_btn.clicked.connect(self._apply_settings)

    def _apply_settings(self):
        """ Set settings """

        # update checkbox dict
        self.params_chkbx = {
            "threshold": self.thresh_method.isChecked(),
            "fill_holes": self.fill_holes.isChecked(),
            "clear_border": self.clear_border.isChecked(),
            "remove_small": self.remove_small.isChecked(),
            "identify_part": self.identify_particles.isChecked(),
            "label_part": self.label_particles.isChecked(),
        }

        table_values = {}
        for row in range(self.table.rowCount()):
            key = self.table.item(row, 0).text()
            cell_widget = self.table.cellWidget(row, 1)
            if cell_widget:  # combobox
                default_value = cell_widget.property("default_value")
                value = cell_widget.currentText()
                if isinstance(default_value, bool):
                    value = value == "True"
            else:   # QTableWidgetItem
                value = self.table.item(row, 1).data(Qt.UserRole)
            
            table_values[key] = value
        
        # emit signal
        self.settings_applied.emit(table_values, self.params_chkbx)