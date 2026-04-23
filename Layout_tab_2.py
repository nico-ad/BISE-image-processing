#%% Import

import numpy as np
import sys
from PyQt5 import uic
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject, QTimer, QDateTime, QElapsedTimer
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow
from PyQt5.QtWidgets import QTabWidget, QPushButton, QDial, QCheckBox, QLineEdit, QProgressBar, QScrollArea, QComboBox, QLabel, QSpinBox
from PyQt5.QtWidgets import (
    QTableWidget,
    QTableWidgetItem,
    QFileDialog,
    QTableView,
    QListView,
    QTreeView,
    QAbstractItemView,
    QSizePolicy,
    QGroupBox,
    QGridLayout,
)
from PyQt5.QtWidgets import (
    QStyledItemDelegate,
    QVBoxLayout,
    QHBoxLayout,
)
from PyQt5.QtGui import QIcon, QFont, QPixmap, QImage

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import time

import ast

from pathlib import Path

import psutil
from collections import deque

import Support_functions_analysis as func_analysis

class HelperTab2(QWidget):
    
    def __init__(self, parent):
        super().__init__(parent)
        
        self.parent = parent
        
        self.count_display_signal = 0
        self.display_every = 1000
        
        self.graph_index = 0
        self.graph_data = []
        self.results = {}
        self.current_folder = None
        self.folder_start_times = {}
        
        if hasattr(self.parent, "helper_tab_1"):
            self.parent.helper_tab_1.folders_changed.connect(self._on_folders_changed)
        
        if hasattr(self.parent, "helper_tab_1"):
            self.parent.helper_tab_1.files_changed.connect(self._on_files_changed)
        
        tab_2 = QWidget()
        main_layout = QVBoxLayout(tab_2)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # ==========
        # TITLE
        # ==========
        self.title_page_2 = QLabel("Analyse images")
        self.title_page_2.setAlignment(Qt.AlignCenter)
        self. title_page_2.setFont(self.parent.font_title)
        self.title_page_2.setFixedHeight(50)
        main_layout.addWidget(self.title_page_2)
        
        # ==========
        # CENTRAL AREA (HORIZONTAL)
        # ==========
        central_layout = QHBoxLayout()
        central_layout.setSpacing(20)
        main_layout.addLayout(central_layout, 1)
        
        # ==========
        # LEFT PANEL (CONTROL PANEL)
        # ==========
        left_panel = QVBoxLayout()
        left_panel.setSpacing(15)
        central_layout.addLayout(left_panel, 1)
        
        # ----- RUN / STOP BUTTONS
        buttons_layout = QHBoxLayout()
        self.run_button = QPushButton("RUN")
        self.run_button.setMinimumHeight(45)
        self.run_button.setFont(self.parent.font_button)
        self.run_button.clicked.connect(self._start_analysis)
        
        self.stop_button = QPushButton("STOP")
        self.stop_button.setMinimumHeight(45)
        self.stop_button.setFont(self.parent.font_button)
        self.stop_button.clicked.connect(self._stop_analysis)
        
        buttons_layout.addWidget(self.run_button)
        buttons_layout.addWidget(self.stop_button)
        
        left_panel.addLayout(buttons_layout)
        
        # ----- Progress bars area
        self.scroll_progress_bar = QScrollArea()
        self.scroll_progress_bar.setWidgetResizable(True)
        self.scroll_progress_bar.setMinimumHeight(150)
        
        self.container_progress_bar = QWidget()
        self.vbox_progress_bar = QVBoxLayout(self.container_progress_bar)
        self.vbox_progress_bar.setAlignment(Qt.AlignTop)
        
        self.scroll_progress_bar.setWidget(self.container_progress_bar)
        
        left_panel.addWidget(self.scroll_progress_bar)
        
        # ----- Elapsed time
        self.elapsed_time_label = QLabel("Elapsed time : 00:00:00")
        self.elapsed_time_label.setFont(self.parent.font_content)
        self.elapsed_time_label.setAlignment(Qt.AlignCenter)
        self.elapsed_time_label.setFixedHeight(40)
        self.elapsed_time_label.setStyleSheet(
            """
            background-color: #e6f2ff;
            border-radius: 8px;
            padding: 6px;
            font-weight: bold;
            """
        )
        left_panel.addWidget(self.elapsed_time_label)
        
        # ----- System monitor
        self.system_monitor = SystemMonitorWidget(parent=self)
        self.system_monitor.setMinimumHeight(180)
        left_panel.addWidget(self.system_monitor)
        
        left_panel.addStretch()
        
        # ==========
        # RIGHT PANEL (IMAGE DISPLAY)
        # ==========
        right_panel = QVBoxLayout()
        central_layout.addLayout(right_panel, 2)
        
        # ----- Image viewer
        graph_container = QWidget()
        graph_layout = QVBoxLayout(graph_container)
        graph_layout.setContentsMargins(0, 0, 0, 0)
        
        self.image_viewer = ImageViewer(graph_container, pixel_size=1.0)
        placehoder = np.ones((100, 100)) * 126
        self.image_viewer.add_img(placehoder)
        
        graph_layout.addWidget(self.image_viewer)
        
        right_panel.addWidget(graph_container, 1)
        
        # ----- Navigation buttons
        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(10)
        
        self.prev_button = QPushButton("<== Previous")
        self.next_button = QPushButton("Next ==>")
        
        self.prev_button.setMinimumHeight(40)
        self.next_button.setMinimumHeight(40)
        
        nav_layout.addStretch()
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)
        nav_layout.addStretch()
        
        right_panel.addLayout(nav_layout)
        
        # -----
        self.parent.tabs.addTab(tab_2, "Run images analysis")
        
        n_folders = len(self.parent.folders) if hasattr(self.parent, "folders") else 0
        self._create_progress_bar(n=n_folders)
        
        self._update_buttons_state()
        
        self.start_time = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_chrono)
        self.timer.setInterval(1000)
        
        self._check_run_conditions()
    
    def _check_run_conditions(self):
        """ Check conditions before running image analysis """
        
        folders_ok = getattr(self.parent, "folders_list", None) and len(self.parent.folders_list) > 0
        files_ok = getattr(self.parent, "files_list", None) and len(self.parent.files_list) > 0
        # pixel_size_ok = getattr(self.parent, "pixel_size", None) and isinstance(self.parent.pixel_size, float) and self.parent.pixel_size > 0.0
        
        if folders_ok and files_ok:
            self.run_button.setEnabled(True)
            self.run_button.setStyleSheet("background-color: green; color: white;")
        else:
            self.run_button.setEnabled(False)
            self.run_button.setStyleSheet("background-color: lightgray; color: white;")
    
    def _start_analysis(self):
        """ Run image analysis """
        
        self._start_chrono()
        self.system_monitor._start()
        
        # change button visual
        self.run_button.setEnabled(False)
        self.run_button.setStyleSheet("background-color: lightgray;")
        self.stop_button.setEnabled(True)
        self.stop_button.setStyleSheet("background-color: red; color: white")
        
        self.current_analysis_index = 0
        self._run_next_folder()
    
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
    
    def _run_next_folder(self):
        """ Run analysis for the next folder """
        
        if self.current_analysis_index >= len(self.folders_to_analyse):
            return
        
        folder = self.folders_to_analyse[self.current_analysis_index]
        self.files = self.files_to_analyse[self.current_analysis_index]
        print(f"current folder index {self.current_analysis_index}")
        
        save_names = Path(folder).parent / Path(f"Particle_analysis_Essai_7_4x10mm3_{self.current_analysis_index+2}_8000Hz_512_640.csv")
        
        params = self._read_parameters()
        
        self.analyser = func_analysis.ParticleAnalyser(
            parent=self, display_every=self.display_every,
            image_paths=folder,
            name_save_files=save_names,
            do_analysis=True,
            time_interval=1/params["frequency acquisition"],
            circularity_threshold=params["circularity thresh"],
            small_objects=int(params["small objects"]),
            total_num_workers=int(params["number of CPU"]),
            num_worker_per_image=int(params["number of CPU per image"]),
            )
        self.worker = self.analyser
        
        # create thread
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        
        # connect pipes
        self.thread.started.connect(self.worker.Process_images)
        # self.worker.display_signal.connect(self._display_image_from_index)
        self.worker.progress_signal.connect(self._update_progress)
        
        self.worker.finished_signal.connect(self._folder_finished)
        
        self.thread.start()
    
    def _folder_finished(self):
        """ Quit thread and launch new folder analysis """
        
        self.thread.quit()
        self.thread.wait()
        
        self.worker = None
        self.thread = None
        self.current_analysis_index += 1
        self._run_next_folder()
    
    def _stop_analysis(self):
        """ Stop analysis """
        
        if hasattr(self, "worker") and self.worker:
            self.analyser._stop()
            self.stop_button.setStyleSheet("background-color: lightgray;")
            self.stop_button.setEnabled(False)
        
        self._stop_chrono()
        self.system_monitor._stop()
    
    def _analysis_finished(self):
        
        self._stop_chrono()
        
        self.run_button.setEnabled(True)
        self.run_button.setStyleSheet("")
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("")
        
        self.thread.quit()
        self.thread.wait()
        
        self.worker = None
        self.thread = None
        # self.worker.deleteLater()
        # self.thread.deleteLater()
    
    def _create_progress_bar(self, n):
        """ Create progress bars for image analysis """
        
        self.progress_bars = []
        self.progress_labels = []
        self.progress_etas = []
        
        # clear previous progress bars
        while self.vbox_progress_bar.count():
            w = self.vbox_progress_bar.takeAt(0).widget()
            if w:
                w.deleteLater()
                
        # create new progress bars
        for i in range(n):
            container = QWidget()
            layout = QVBoxLayout(container)
            layout.setContentsMargins(5, 5, 5, 5)
            
            label = QLabel(f"Folder {i+1}")
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setFormat("%p%")
            
            eta = QLabel("Estimated time : --")
            
            layout.addWidget(label)
            layout.addWidget(bar)
            layout.addWidget(eta)
            
            self.vbox_progress_bar.addWidget(container)
            
            self.progress_bars.append(bar)
            self.progress_labels.append(label)
            self.progress_etas.append(eta)
    
    def _store_chunk(self, folder_idx, chunk_result):
        """ Store chunk for navigation """
        
        if folder_idx not in self.results:
            self.results[folder_idx] = []
            
        self.results[folder_idx].append(chunk_result)
        # self.graph_index = len(self.results[folder_idx]) - 1
        self.current_folder = folder_idx
        # self._display_image_from_index()
        # self._update_buttons_state()
        
    def _display_image_from_index(self, index: int):
        """ Display image analysis """
        
        self.count_display_signal += 1
        if index < len(self.files):
            name = self.files[index]
        else:
            return
        
        # print(name)
        # img = func_analysis.ParticleAnalyser._load_image(name=name)
        # print(name, img.shape)
        
        if self.current_folder not in self.results:
            return
        
        if index >= len(self.results[self.current_folder]):
            return
        
        name = self.results[self.current_folder][index]
        img = func_analysis.ParticleAnalyser._load_image(name)
        print(name, img.shape)
        
        self.image_viewer.clear()
        self.image_viewer.add_img(img)
        
        self.graph_index = index
        self._update_buttons_state()
        
    def _next_graph(self):
        """ Visualize next image """
        
        if self.graph_index < len(self.graph_data) - 1:
            self.graph_index += 1
            self._display_image_from_index(self.graph_index)
        self._update_buttons_state()
    
    def _previous_graph(self):
        """ Visualize previous image """
        
        if self.graph_index > 0:
            self.graph_index -= 1
            self._display_image_from_index(self.graph_index)
        self._update_buttons_state()
    
    def _update_buttons_state(self):
        """ Colored arrows if cannot change """
        
        if self.current_folder not in self.results:
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)
            return
        
        max_index = len(self.results[self.current_folder]) - 1
        self.prev_button.setEnabled(self.graph_index > 0)
        self.next_button.setEnabled(self.graph_index < max_index)
    
    def _update_progress(self, folder_idx: int, done: int, total: int):
        """ Update progress bar with value """
        
        if not (0 <= folder_idx < len(self.progress_bars)):
            return
        
        bar = self.progress_bars[folder_idx]
        label = self.progress_labels[folder_idx]
        eta = self.progress_etas[folder_idx]
        
        percent = int(done / total * 100)
        bar.setValue(percent)
        bar.setFormat(f"{done:,} / {total:,} (%p%)")
        
        if folder_idx not in self.folder_start_times:
            timer = QElapsedTimer()
            timer.start()
            self.folder_start_times[folder_idx] = timer
        
        elapsed_ms = self.folder_start_times[folder_idx].elapsed()
        elapsed_s = elapsed_ms / 1000
        
        rate = done / elapsed_s if elapsed_ms else 0
        remaining = int((total - done) / rate) if rate else 0

        if remaining > 3600:
            remaining_str = f"{remaining // 3600}h {(remaining % 3600) // 60}min  {(remaining % 60)}s"
        elif remaining > 60:
            remaining_str = f"{remaining // 60}min {(remaining % 60)}s"
        else:
            remaining_str = f"{remaining}s"
        
        eta.setText(f"Remaining time : {remaining_str}")
        
        if percent == 100:
            bar.setStyleSheet("""
                QProgressBar:: chunk {
                    background-color: #2ecc71;
                    }
                """)
            eta.setText("Finished")
        
        if 0 <= folder_idx < len(self.progress_bars):
            self.progress_bars[folder_idx].setValue(percent)
        
        if all(bar.value() == 100 for bar in self.progress_bars):
            self.stop_button.setEnabled(False)
            self.run_button.setEnabled(True)
            self._stop_chrono()
    
    def _start_chrono(self):
        self.start_time = QDateTime.currentDateTime()
        self.timer.start()
    
    def _stop_chrono(self):
        self.timer.stop()
        self._update_chrono()
        
    def _update_chrono(self):
        """ Update elapsed time from start analysis """
        
        if self.start_time is None:
            return
        
        now = QDateTime.currentDateTime()
        elapsed_seconds = self.start_time.secsTo(now)
        
        hours, remainder = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        self.elapsed_time_label.setText(f"Elapsed time : {hours:02d}:{minutes:02d}:{seconds:02d}")
    
    def _on_folders_changed(self, folders):
        """ Set progress bar number and allows to run analysis """
        
        self.folders_to_analyse = folders
        if isinstance(self.folders_to_analyse, str):
            self.folders_to_analyse = [self.folders_to_analyse]
        
        if len(self.folders_to_analyse) == 1 and all(isinstance(f, str) for f in self.folders_to_analyse):
            self.folders_to_analyse = [self.folders_to_analyse]
        
        n_folders = len(folders)
        self._create_progress_bar(n_folders)
        self._check_run_conditions()

    def _on_files_changed(self, files):
        """ Get file names """
        
        self.files_to_analyse = files
        
        if isinstance(self.files_to_analyse, str):
            self.files_to_analyse = [self.files_to_analyse]
        
        if len(self.files_to_analyse) == 1 and all(isinstance(f, str) for f in self.files_to_analyse):
            self.files_to_analyse = [self.files_to_analyse]

class ImageViewer(QWidget):
    def __init__(self, parent, pixel_size):
        super().__init__(parent)
        
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
    def on_move(self, event, ax, img, status_label):
        
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
    ):
        
        fig = Figure()
        ax = fig.add_subplot(111)
        
        ax.imshow(img, origin="lower", cmap="gray", aspect="auto")
        ax.set_aspect("equal", adjustable="box")
        
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
        
        # canvas.mpl_connect("motion_notify_event",
        #                    lambda event: self.on_move(event, ax, img, status_label),
        # )
        # canvas.mpl_connect("scroll_event",
        #                    lambda event: self.on_scroll(event, ax, canvas),
        # )
        
        self.layout.addWidget(canvas)
        self.layout.addWidget(toolbar)
        self.canvases.append(canvas)
        self.toolbars.append(toolbar)

# class SystemMonitorWidget_old(QWidget):
#     def __init__(self, parent=None, history_length=60):
#         super().__init__(parent)
        
#         self.history_length = history_length
        
#         # data buffer
#         self.cpu_history = deque([0] * history_length, maxlen=history_length)
#         self.ram_history = deque([0] * history_length, maxlen=history_length)
        
#         # layout
#         layout = QVBoxLayout(self)
        
#         # matplotlib figure
#         self.figure = Figure()
#         self.canvas = FigureCanvas(self.figure)
#         self.ax = self.figure.add_subplot(111)
        
#         self.ax.set_xlim(0, history_length)
#         self.ax.set_ylim(0, 100)
#         # self.ax.set_xlabel("Time in s", fontsize=self.parent.dict_fontsize["label"])
#         # self.ax.set_ylabel("Consumption in %", fontsize=self.parent.dict_fontsize["label"])
        
#         self.cpu_line, = self.ax.plot([], [], label="CPU")
#         self.ram_line, = self.ax.plot([], [], label="RAM")
        
#         self.ax.legend(loc="upper right")
        
#         layout.addWidget(self.canvas)
        
#         self.timer = QTimer()
#         self.timer.setInterval(1000)
#         self.timer.timeout.connect(self._update_metrics)
        
#     def _start(self):
#         self.timer.start()
    
#     def _stop(self):
#         self.timer.stop()
    
#     def _update_metrics(self):
#         """ Update monitoring graph """
        
#         cpu = psutil.cpu_percent()
#         ram = psutil.virtual_memory().percent
        
#         self.cpu_history.append(cpu)
#         self.ram_history.append(ram)
        
#         self.cpu_line.set_data(range(self.history_length), list(self.cpu_history))
#         self.ram_line.set_data(range(self.history_length), list(self.ram_history))
        
#         self.canvas.draw_idle()
        
class SystemMonitorWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.num_cores = psutil.cpu_count(logical=True)
        
        main_layout = QVBoxLayout(self)
        
        title = QLabel("System monitor")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        main_layout.addWidget(title)
        
        # ==========
        # CPU CORES
        # ==========
        
        self.cpu_group = QGroupBox("CPU cores usage")
        cpu_layout = QGridLayout(self.cpu_group)
        
        self.core_bars = []
        self.core_labels = []
        
        for i in range(self.num_cores):
            label = QLabel(f"Cores {i}")
            label.setMinimumWidth(120)
            
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setTextVisible(True)
            bar.setFormat("%p%")
            
            row = i // 2
            col = (i % 2) * 2
            
            cpu_layout.addWidget(label, row, col)
            cpu_layout.addWidget(bar, row, col + 1)
            
            self.core_labels.append(label)
            self.core_bars.append(bar)
            
        main_layout.addWidget(self.cpu_group)
        
        # ==========
        # RAM
        # ==========
        
        self.ram_group = QGroupBox("RAM usage")
        
        ram_layout = QVBoxLayout(self.ram_group)
        
        self.ram_label = QLabel("RAM")
        self.ram_bar = QProgressBar()
        
        self.ram_bar.setRange(0, 100)
        self.ram_bar.setTextVisible(True)
        self.ram_bar.setFormat("RAM %p%")
        
        ram_layout.addWidget(self.ram_label)
        ram_layout.addWidget(self.ram_bar)
        
        main_layout.addWidget(self.ram_group)
        
        self.timer = QTimer()
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._update_metrics)
        
    def _start(self):
        self.timer.start()
    
    def _stop(self):
        self.timer.stop()
    
    def _update_metrics(self):
        """ Update CPU and RAM display"""

        cpu_cores = psutil.cpu_percent(percpu=True)
        
        # cpu usage
        for i, val in enumerate(cpu_cores):
            if i < len(self.core_bars):
                self.core_bars[i].setValue(int(val))
                self.core_labels[i].setText(f"Core {i}")
        
        # ram usage
        ram = psutil.virtual_memory().percent
        self.ram_bar.setValue(int(ram))