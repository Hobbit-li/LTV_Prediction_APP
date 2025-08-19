import webview
import threading
from pathlib import Path
from app import run_pipeline


class Api:
    def run_model(self, train_file, test_file, ref_month, cost):
        """
        Called from JS. Runs the pipeline and returns results.
        """
        results = run_pipeline(train_file, test_file, ref_month, float(cost))
        return results


def start_app():
    api = Api()
    window = webview.create_window(
        "LTV Prediction",
        html=Path(__file__).parent / "web_ui" / "gui.html",
        width=1000,
        height=700,
        js_api=api
    )
    webview.start(debug=True)


if __name__ == "__main__":
    start_app()

