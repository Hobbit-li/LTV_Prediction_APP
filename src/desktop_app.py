import webview
from app import run_pipeline

class Api:
    def run(self, path_ref, path_pre, ref_month, cost):
        try:
            result = run_pipeline(path_ref, path_pre, ref_month, cost)
            return result
        except Exception as e:
            return {"error": str(e)}

if __name__ == "__main__":
    api = Api()
    window = webview.create_window(
        "📊 数据上传与计算工具",
        "gui.html",  # 这里是你的前端页面
        js_api=api,
        width=800,
        height=600,
        resizable=True
    )
    webview.start()
