import os
import sys
import webview
import threading
from pathlib import Path

from app import run_pipeline

def resource_path(relative_path):
    """
    Get the absolute path to a resource, works for dev and for PyInstaller
    """
    if hasattr(sys, "_MEIPASS"):
        # Running in bundled executable
        return os.path.join(sys._MEIPASS, relative_path)
    # Running in development
    return os.path.join(os.path.abspath("."), relative_path)
    
class Api:
    def run_model(self, train_file, test_file, ref_month, cost):
        """
        Called from JS. Runs the pipeline and returns results.
        """
        results = run_pipeline(train_file, test_file, ref_month, float(cost))
        return results


def start_app():
    api = Api()
    # Get the HTML file path using resource_path for PyInstaller compatibility
    html_file = resource_path(os.path.join("web_ui", "gui.html"))
    window = webview.create_window(
        "LTV Prediction",
        html_file,
        width=1000,
        height=700,
        js_api=api
    )
    webview.start(debug=True)


if __name__ == "__main__":
    start_app()

