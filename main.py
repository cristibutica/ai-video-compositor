import sys
from PyQt6.QtWidgets import QApplication

# Import the window class from our custom ui module
from ui.main_window import MainWindow

def main():
    # Create the PyQt6 application object
    app = QApplication(sys.argv)
    
    # Instantiate and show our modular main window
    window = MainWindow()
    window.show()
    
    # Execute the application loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main()