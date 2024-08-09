# Import required modules
from PyQt5.QtWidgets import QSplashScreen, QApplication, QMainWindow, QDesktopWidget
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import sys

# Import additional modules
from form import Ui_MainWindow
from graphics import World
import threading
import time

class Sim(QMainWindow, Ui_MainWindow):
    def __init__(self, rbt):
        super().__init__()

        self.setupUi(self)

        self.world = World(self, rbt)
        self.setCentralWidget(self.world)

        self.setWindowTitle('stewart_platform_viz')
        self.statusbar.showMessage('Ready.')

        self.center()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

def init_graphics(rbt):
    #sim.show()
    #splash.finish(sim)
    app = QApplication(sys.argv)
    app.processEvents()

    sim = Sim(rbt)        
    sim.world.status_bar = sim.statusbar
    sim.show()

    exit_code = app.exec_()
    sys.exit(exit_code)

# s.show()
# time.sleep(2)
# s.close()

    
