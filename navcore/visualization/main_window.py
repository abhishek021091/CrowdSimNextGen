from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
)
from navcore.visualization.recorder import Recorder
from PySide6.QtGui import QAction
import sys


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        canvas = QWidget()
        self.setCentralWidget(canvas)
        self.setWindowTitle("NavCore")
        self.resize(1200, 800)
        self.statusBar().showMessage("Ready")
        menuBar = self.menuBar()
        fileMenu = menuBar.addMenu("File")
        simulationMenu = menuBar.addMenu("Simulation")
        viewMenu = menuBar.addMenu("View")
        helpMenu = menuBar.addMenu("Help")

        exitAction = QAction("Exit", self)
        exitAction.setShortcut("Ctrl+Q")
        fileMenu.addAction(exitAction)
        exitAction.triggered.connect(self.close)

        self.recordAction = QAction("Record", self)
        self.recordAction.setShortcut("Ctrl+R")
        self.recordAction.triggered.connect(self.toggle_recording)
        simulationMenu.addAction(self.recordAction)

    def toggle_recording(self):
        if not hasattr(self, "recorder"):
            self.recorder: Recorder = Recorder()

        if self.recorder.recording:
            self.recorder.stop_recording()
            self.statusBar().showMessage("Recording stopped")
            self.recordAction.setText("Start Recording")
        else:
            self.recorder.start_recording()
            self.statusBar().showMessage("Recording started")
            self.recordAction.setText("Stop Recording")


def main():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
