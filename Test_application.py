
import sys
import psutil
from pathlib import Path

# PyQt imports
from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout

# support function for each tab
import Layout_tab_1 as layout_tab_1
import Layout_tab_2 as layout_tab_2
# import Layout_tab_3 as layout_tab_3
import Layout_tab_4 as layout_tab_4

class ParticleAnalysisApp(QMainWindow):
    
    """ Main window of the Particle Analysis application """
    
    def __init__(self):
        super().__init__()

        # self.setWindowIcon(QIcon("Logo_BIRD_version_finale.png"))
        
        # set fonts
        self.font_tabs = QFont("Arial", 14)
        self.font_title = QFont("Arial", 18)
        self.font_header = QFont("Arial", 12)
        self.font_header.setBold(True)
        self.font_content = QFont("Arial", 11)
        self.font_button = QFont("Arial", 14)
        
        # graph font
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
            }
        }

        self.fonts = {
            "tabs": self.font_tabs,
            "title": self.font_title,
            "header": self.font_header,
            "content": self.font_content,
            "button": self.font_button,
            "dict_fontsize": self.dict_fontsize,
        }
        
        # variables
        self.folders = []
        self.current_folder_index = 0
        self.path_images = []
        self.current_image_index = 0
        
        # ui_path = Path(__file__).parent / "ParticlesAnalysis.ui"
        # uic.loadUi(ui_path, self)
        
        self.setWindowTitle("BIRD for BISE Image Recognition and Detection")
        
        # create global layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        self.tabs = QTabWidget()
        self.tabs.tabBar().setFont(self.font_tabs)
        layout.addWidget(self.tabs)
        
        # initialize helper for each tabs
        self.helper_tab_1 = layout_tab_1.HelperTab1(parent=self)
        self.helper_tab_2 = layout_tab_2.HelperTab2(parent=self)
        # self.helper_tab_3 = layout_tab_3.HelperTab3(parent=self)
        self.helper_tab_4 = layout_tab_4.HelperTab4(parent=self)
        
        self.add_tabs()
        
        # connect classes
        # self.helper_tab_1.num_folders.connect(self.helper_tab_2.create_progress_bar)
        # self.helper_tab_1._emit_folder_count()
    
    def add_tabs(self):
        """ Add tabs to the QtTable using helper classes """
        # self.tabs.addTab(self.helper_tab_1, "Tab 1")
        # self.tabs.addTab(self.helper_tab_2, "Tab 2")
        # self.tabs.addTab(self.helper_tab_3, "Tab 3")
        # self.tabs.addTab(self.helper_tab_4, "Tab 4")
    
def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("icons/Logo_BIRD_version_finale_icon.ico"))
    win = ParticleAnalysisApp()
    win.setWindowIcon(QIcon("icons/Logo_BIRD_version_finale_icon.ico"))
    win.showMaximized()
    sys.exit(app.exec_())
    
if __name__ == "__main__":
    
    main()
