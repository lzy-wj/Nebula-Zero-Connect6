import sys
import os

# Add subdirectories to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'ui'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'tools'))

from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow

if __name__ == '__main__':
    # Ensure working directory is correct
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion') # Use Fusion style for better CSS support
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
