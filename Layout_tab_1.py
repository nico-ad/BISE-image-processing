
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
    QGraphicsDropShadowEffect,
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
    analysis_parameters = pyqtSignal(list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.parent = parent

        self.settings = [dict, dict]
        
        self.analyser_class = func_preview.ParticleAnalyser(parent=self)
        self.video_Maker = func_preview.VideoMaker()
        # self.spinner = func_preview.Spinner(parent=self)
        
        self.parent.dial_steps = [1, 10, 100, 1000, 10000]
        
        self.parent.image_extensions = (
            ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff",
        )

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
        data_group = QGroupBox("Image file selection")
        data_group_layout = QVBoxLayout()
        data_group.setLayout(data_group_layout)
        data_group.setStyleSheet("""
            QGroupBox {
            background-color: rgba(255, 255, 255, 0.85);
            border: 2px solid #AAAAAA;
            border-radius: 15;
            margin-top: 10px;
            font-weight: bold;
            font-size: 14px;
            padding: 10px;
            }
            QGroupBow::title {
            subcontrol-origin: margin;
            subcontrol-position: top corner;
            padding: 0 3px;
            }
        """)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(12)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 80))
        data_group.setGraphicsEffect(shadow)
        
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
        control_group.setStyleSheet("""
            QGroupBox {
            background-color: rgba(255, 255, 255, 0.85);
            border: 2px solid #AAAAAA;
            border-radius: 15;
            margin-top: 10px;
            font-weight: bold;
            font-size: 14px;
            padding: 10px;
            }
            QGroupBow::title {
            subcontrol-origin: margin;
            subcontrol-position: top corner;
            padding: 0 3px;
            }
        """)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(12)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 80))
        control_group.setGraphicsEffect(shadow)
        
        self.option_btn = QPushButton("Options")
        self.option_btn.setFont(self.parent.font_button)
        self.option_btn.setMinimumHeight(40)
        control_layout.addWidget(self.option_btn)
        self.option_btn.clicked.connect(self._option_dialog)

        self.preview_btn = QPushButton("Preview")
        self.preview_btn.setFont(self.parent.font_button)
        self.preview_btn.setMinimumHeight(40)
        self.preview_btn.setEnabled(False)
        control_layout.addWidget(self.preview_btn)
        self.preview_btn.clicked.connect(self._previewAnalysis)
        
        self.change_folder_btn = QPushButton("Change folder")
        self.change_folder_btn.setFont(self.parent.font_button)
        self.change_folder_btn.setMinimumHeight(40)
        self.change_folder_btn.setEnabled(False)
        control_layout.addWidget(self.change_folder_btn)
        self.change_folder_btn.clicked.connect(self._changeFolder)
        
        nav_layout = QHBoxLayout()
        
        self.previous_btn = QPushButton("Previous")
        self.previous_btn.setFont(self.parent.font_button)
        self.previous_btn.setMinimumHeight(40)
        self.previous_btn.setEnabled(False)
        nav_layout.addWidget(self.previous_btn)
        self.previous_btn.clicked.connect(self._previousImage)
        
        self.next_btn = QPushButton("Next")
        self.next_btn.setFont(self.parent.font_button)
        self.next_btn.setMinimumHeight(40)
        self.next_btn.setEnabled(False)
        nav_layout.addWidget(self.next_btn)
        self.next_btn.clicked.connect(self._nextImage)
        
        # video button
        self.video_btn = QPushButton("Video")
        self.video_btn.setFont(self.parent.font_button)
        self.video_btn.setMinimumHeight(40)
        self.video_btn.setEnabled(False)
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
        stats_group.setStyleSheet("""
            QGroupBox {
            background-color: rgba(255, 255, 255, 0.85);
            border: 2px solid #AAAAAA;
            border-radius: 15;
            margin-top: 10px;
            font-weight: bold;
            font-size: 14px;
            padding: 10px;
            }
            QGroupBow::title {
            subcontrol-origin: margin;
            subcontrol-position: top corner;
            padding: 0 3px;
            }
        """)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(12)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 80))
        stats_group.setGraphicsEffect(shadow)
        
        self.num_part_per_frame = QLineEdit()
        self.num_part_per_frame.setPlaceholderText("Number of particles detected")
        self.num_part_per_frame.setReadOnly(True)
        self.num_part_per_frame.setAlignment(Qt.AlignCenter)
        stats_layout.addWidget(self.num_part_per_frame)
        
        self.density_per_frame = QLineEdit()
        self.density_per_frame.setPlaceholderText("Number of particles per unit area")
        self.density_per_frame.setReadOnly(True)
        self.density_per_frame.setAlignment(Qt.AlignCenter)
        stats_layout.addWidget(self.density_per_frame)
        
        self.diameters_in_frame = QLineEdit()
        self.diameters_in_frame.setPlaceholderText("Min / Max diameters")
        self.diameters_in_frame.setReadOnly(True)
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
        
        # ----- preview area
        preview_splitter = QSplitter(Qt.Horizontal)
        
        # image preview
        image_group = QGroupBox("Image preview")
        image_group_layout = QVBoxLayout()
        image_group.setLayout(image_group_layout)
        
        # image name
        self.image_name = QLineEdit()
        self.image_name.setText("Image name")
        self.image_name.setReadOnly(True)
        self.image_name.setFont(self.parent.font_content)
        self.image_name.setAlignment(Qt.AlignCenter)
        self.image_name.setFixedHeight(35)
        image_group_layout.addWidget(self.image_name)
        
        # image and spinner container
        self.image_preview_container = QWidget()
        self.image_layout = QVBoxLayout(self.image_preview_container)
        image_group_layout.addWidget(self.image_preview_container)

        # intanciate image_viewer
        self.image_viewer = func_preview.ImageViewer(
            self.image_preview_container,
            pixel_size=None,
            )

        # # spinner
        # self.spinner = func_preview.Spinner(parent=self.image_preview_container)
        # self.image_layout.addWidget(self.spinner, alignment=Qt.AlignCenter)  # attach spinner to container
        # self.spinner.hide()

        # ImageViewer
        self.viewer = None
        
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

        # ----- enable buttons
        has_folders = len(self.parent.folders_list) > 0
        self.preview_btn.setEnabled(has_folders)
        self.change_folder_btn.setEnabled(has_folders)
        
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

        # ----- enable buttons
        has_folders = len(self.parent.folders_list) > 0
        self.preview_btn.setEnabled(has_folders)
        self.change_folder_btn.setEnabled(has_folders)

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


    def _handle_parameters_and_display(self, settings):
        self.settings = settings
        # print("Updated settings")
        # print(f"    Table ssettings : {self.settings[0]}")
        # print(f"    Checkboxes ssettings : {self.settings[1]}")

    def _option_dialog(self):
        """ open dialog window """
        
        # self.helper2 = Layout_tab_2.HelperTab2(parent=self.parent)
        dialog = OptionDialog(self, settings=self.settings, fonts=self.parent.fonts)
        dialog.settings_applied.connect(self.parent.helper_tab_2._handle_settings)
        dialog.settings_applied.connect(self._handle_parameters_and_display)
        dialog.exec_()

    def _previewAnalysis(self):
        """ load and display preview image """
        
        if hasattr(self, "settings"):
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
        
        self.pixel_size = self.settings["table"]["pixel_size"]
        
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
        """ Apply analysis on current image """
        
        self.data, self.base_image, self.analysed_image = self.analyser_class._apply_image_modifications(
            filename=self.parent.image_path,
            params=self.settings["table"],
            chkbx=self.settings["checkboxes"],
        )
    
    def _update_viewer(self):
        """ Process image preview on another thread """
        
        rotation_value = self.settings["table"]["rotate"]
        image_to_process = self.analysed_image
        data_to_process = self.data
        
        # display spinner
        # self.spinner.start()
        # self.image_display.hide() # hide image during analysis
        
        self._get_or_create_viewers()
        
        self.viewer_thread = QThread()
        self.viewer_worker = func_preview.ViewerWorker(
            image_to_process, data_to_process, rotation=rotation_value,
            )
        self.viewer_worker.moveToThread(self.viewer_thread)
        
        self.viewer_thread.started.connect(self.viewer_worker.run)
        self.viewer_worker.finished.connect(self._on_viewer_ready)
        self.viewer_worker.finished.connect(self.viewer_thread.quit)
        self.viewer_worker.finished.connect(self.viewer_worker.deleteLater)
        self.viewer_thread.finished.connect(self.viewer_thread.deleteLater)
        
        self.viewer_thread.start()
    
    def _on_viewer_ready(self, img_rot, hist):
    # def _on_viewer_ready(self, pixmap_or_data):
        """ Display images preview """
        
        # if self.spinner.parent() is not None:
        #     self.spinner.stop()
        
        self.image_viewer.clear()
        self.image_viewer.add_img(img_rot, self.data)

        # img = pixmap_or_data.get("image", None)
        # data = pixmap_or_data.get("data", None)
        # if img is None and data is not None:
        #     self.viewer.add_img(img=img, data=data)
        
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
        self.image_viewer.clear()
        self.image_viewer.add_img(self._apply_rotation(self.analysed_image, rotation_value))
        
        # self.density_viewer.clear()
        # self.density_viewer.add_density(self._apply_rotation(self.surface_image, rotation_value))
        
        self._update_buttons_state()

    def _on_item_changed(self, item):
        if item.column() == 1:
            # self.parent.parameters_table_tab_1.blockSignals(True)
            # item.setData(Qt.UserRole, item.text())
            # self.parent.parameters_table_tab_1.blockSignals(False)
            self._updatePreview()

    # def _read_parameters(self) -> dict:
    #     """ Read table of parameters for image analysis """
        
    #     current_params = {}
    #     for row in range(self.parent.parameters_table_tab_1.rowCount()):
    #         # first column
    #         key = self.parent.parameters_table_tab_1.item(row, 0).text()
    #         # second column
    #         widget = self.parent.parameters_table_tab_1.cellWidget(row, 1)
    #         if isinstance(widget, QComboBox):
    #             value = widget.currentText()
    #         else:
    #             item = self.parent.parameters_table_tab_1.item(row, 1)
    #             value = item.text() # item.data(Qt.UserRole)
    #         try:
    #             value = ast.literal_eval(value)
    #         except (ValueError, SyntaxError):
    #             pass
            
    #         current_params[key] = value
    #     return current_params

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
    settings_applied = pyqtSignal(dict)

    def __init__(self, parent=None, settings=None, fonts=None):
        super().__init__(parent)

        if not isinstance(settings, list) or len(settings) != 2:
            settings = [{}, {}]
        for i in [0, 1]:
            if not isinstance(settings[i], dict):
                settings[i] = {}
        self.settings = settings

        self.fonts = fonts

        # initialize default parameters
        self.default_params = [
            ("Images range", "images_range", [0, 21837], "", None, list),
            ("Acquisition frequency", "acq_frequency", 8000, "Hz", None, int),
            ("Circularity thresh", "circularity_thresh", [0.2, 1.0], "", None, list),
            ("Small objects", "small_objects", 20, "px", None, int),
            ("Enable subpixel detection", "subpixel", False, "", [True, False], bool),
            ("Invert grayscale", "invert_gray", True, "", [True, False], bool),
            ("Pixel size", "pixel_size", 1/146, "mm/px", None, float),
            ("Number of CPU", "n_cpu", 10, "", None, int),
            ("Number of CPU per image", "n_cpu_img", 1, "", None, int),
            ("Image format", "img_format", ".jpg", "", None, str),
            ("Video sequence", "video_seq", True, "", [True, False], bool),
            ("Rotate image", "rotate", "0", "", ["0", "90", "180", "270"], int),
            ("H maxima", "h_max", 0.01, "", None, float),
        ]

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

        # ----- table for parameters
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

        # ----- fill table cells
        for i, (label, key, default_value, unit, options, _) in enumerate(self.default_params):
            
            # parameter name
            item_label = QTableWidgetItem(label)
            item_label.setFlags(item_label.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(i, 0, item_label)

            # use existing value
            existing_value = self.settings[0].get(key, default_value)
            
            # parameter value
            if options:
                combo = QComboBox()
                combo.addItems([str(opt) for opt in options])
                combo.setCurrentText(str(existing_value))
                combo.setProperty("default_value", existing_value)
                self.table.setCellWidget(i, 1, combo)
            else:
                # value formatting
                if isinstance(existing_value, float) and existing_value < 1:
                    display_text = f"{existing_value:.3e}"
                else:
                    display_text = str(existing_value)
                item_value = QTableWidgetItem(display_text)
                item_value.setData(Qt.UserRole, existing_value)
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
        checkboxes_info = [
            ("threshold", "Threshold", True, "Apply binarization threshold on images to remove background"),
            ("fill_holes", "Fill holes", True, "After binarization, fill holes inside objects"),
            ("clear_border", "Clear border", True, "Remove objects that touches border"),
            ("remove_small", "Remove small objects", True, "Remove objects that contains less than n pixels"),
            ("identify_part", "Identity particles", True, "Apply contour on detected particles and aggregates"),
            ("label_part", "Label particles", True, "Label particles"),
        ]

        # add checkboxes
        self.checkboxes = {}
        for key, label, default, toolTip in checkboxes_info:
            chk = QCheckBox(label, parent=checkbox_group)
            chk.setChecked(self.settings[1].get(key, default))
            chk.setToolTip(toolTip)
            self.checkboxes[key] = chk
            params_chkbx_layout.addWidget(chk)

        tab2_layout.addWidget(checkbox_group)
        self.tabs.addTab(tab2, "Processing options")

        # =========================
        # Apply button
        # =========================

        self.apply_btn = QPushButton("Apply")
        main_layout.addWidget(self.apply_btn)
        self.apply_btn.clicked.connect(self._apply_settings)

    def _convert_value(self, value, dtype, default):
        """ Convert value in appropriate type """

        if dtype is bool:
            return value == "True"
        
        elif dtype is int:
            try:
                return int(value)
            except ValueError:
                return default
            
        elif dtype is float:
            try:
                return float(value)
            except ValueError:
                return default
        
        elif dtype is list:
            try:
                value = value.strip("[]")
                if not value:
                    return []
                items = [it.strip() for it in value.split(",")]
                converted_items = []
                for it in items:
                    converted_items.append(float(it) if "." in it else int(it))
                return converted_items
            except Exception:
                return default
        return value

    def _apply_settings(self):
        """ Set settings and emit signal """

        # update checkbox dict
        params_chkbx = {key: chk.isChecked() for key, chk in self.checkboxes.items()}

        table_values = {}
        for row in range(self.table.rowCount()):

            key = self.default_params[row][1]
            raw_value = self.default_params[row][2]
            cell_widget = self.table.cellWidget(row, 1)
            dtype = self.default_params[row][5]
            
            table_values[key] = self._convert_value(raw_value, dtype, raw_value)
        
        # emit signal
        settings = {
            "table": table_values,
            "checkboxes": params_chkbx,
        }
        self.settings_applied.emit(settings)

        # close window
        self.accept()