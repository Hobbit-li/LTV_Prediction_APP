import os
import logging
import webview

from app import run_pipeline
from config_loader import resource_path

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# def resource_path(relative_path):
#     """
#     Get the absolute path to a resource, works for dev and for PyInstaller
#     """
#     if hasattr(sys, "_MEIPASS"):
#         # Running in bundled executable
#         base_path = sys._MEIPASS
#     else:
#         # Running in development
#         base_path = os.path.abspath(".")
#      # 规范化路径，确保使用正确的路径分隔符
#     normalized_path = os.path.normpath(os.path.join(base_path, relative_path))
#     logger.info(f"资源路径: {base_path} + {relative_path} = {normalized_path}")
#     return normalized_path


def validate_file_path(file_path):
    """
    验证文件路径是否存在且可访问
    """
    if not file_path:
        return False, "文件路径为空"

    # 尝试规范化路径（处理可能的相对路径）
    try:
        abs_path = os.path.abspath(file_path)
    except Exception as e:
        return False, f"路径格式错误: {str(e)}"

    # 检查文件是否存在
    if not os.path.exists(abs_path):
        return False, f"文件不存在: {abs_path}"

    # 检查是否为文件（不是目录）
    if not os.path.isfile(abs_path):
        return False, f"路径不是文件: {abs_path}"

    return True, abs_path


class Api:
    def run_model(self, train_file, test_file, ref_month, cost):
        """
        Called from JS. Runs the pipeline and returns results.
        """
        try:
            logger.info(
                f"接收到的参数 - train_file: {train_file}, test_file: {test_file}, ref_month: {ref_month}, cost: {cost}"
            )

            # 验证训练文件
            is_valid_train, train_path_or_error = validate_file_path(train_file)
            if not is_valid_train:
                return {"error": f"训练文件无效: {train_path_or_error}"}

            # 验证测试文件
            is_valid_test, test_path_or_error = validate_file_path(test_file)
            if not is_valid_test:
                return {"error": f"测试文件无效: {test_path_or_error}"}

            # 验证成本参数
            try:
                cost_value = float(cost)
                if cost_value <= 0:
                    return {"error": "成本必须大于0"}
            except ValueError:
                return {"error": "成本必须是有效的数字"}

            # 运行模型
            logger.info("开始运行模型...")
            results = run_pipeline(
                train_path_or_error, test_path_or_error, ref_month, cost_value
            )
            logger.info("模型运行完成")

            return results
        except Exception as e:
            error_msg = f"运行模型时发生错误: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"error": error_msg}


def start_app():
    api = Api()

    # 获取HTML文件路径
    html_file = resource_path(os.path.join("web_ui", "gui.html"))

    # 检查HTML文件是否存在
    if not os.path.exists(html_file):
        logger.error(f"HTML文件不存在: {html_file}")
        # 创建简单的错误页面
        html_content = """
        <html><body>
            <h1>错误: GUI文件未找到</h1>
            <p>无法找到界面文件，请确保应用正确安装。</p>
        </body></html>
        """
        window = webview.create_window(
            "LTV Prediction - 错误", html=html_content, width=600, height=400
        )
    else:
        # 创建正常窗口
        window = webview.create_window(
            "LTV Prediction", html_file, width=1000, height=700, js_api=api
        )

    # 启动应用
    webview.start(debug=True)


if __name__ == "__main__":
    start_app()
