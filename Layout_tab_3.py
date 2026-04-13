#%% Import

import sys
from PyQt5 import uic
from PyQt5.QtCore import Qt
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

import matplotlib.pyplot as plt

class HelperTab3(QWidget):
    
    def __init__(self, parent):
        
        self.parent = parent
        
        tab_3 = QWidget()
        tab_3_layout = QVBoxLayout(tab_3)
        
        # setup title page
        self.title_page_3 = QLineEdit(tab_3)
        self.title_page_3.setText("Visualize analysis through graph and 2D plots")
        self.title_page_3.setReadOnly(True)
        self.title_page_3.setFont(self.parent.font_title)
        self.title_page_3.setAlignment(Qt.AlignCenter)
        
        tab_3_layout.addWidget(self.title_page_3)
        
        self.parent.tabs.addTab(tab_3, "Visualize analysis")
        
        # tab_3 = QWidget()
        # tab_3_layout = QVBoxLayout(tab_3)
        
        # self.tabs.addTab(tab_3, "Visualize analysis")
        
        # # setup title page
        # self.title_page_3 = QLineEdit(tab_3)
        # self.title_page_3.setText("Visualize analysis through graph and 2D plots")
        # self.title_page_3.setReadOnly(True)
        # self.title_page_3.setFont(self.font_title)
        # self.title_page_3.setAlignment(Qt.AlignCenter)
        
        # tab_3_layout.addWidget(self.title_page_3)