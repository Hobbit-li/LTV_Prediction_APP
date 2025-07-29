import streamlit as st
import traceback
import pandas as pd
from config_loader import load_config
from data_utils import data_preprocess
from train import train_process
from predict import predict_process
from results_show import show_roas_ltv
from visual import compare_plot, evaluate_ltv, residual_plot
from utils_io import save_predictions, create_output_dir
pd.options.mode.chained_assignment = None  # 关闭 SettingWithCopyWarning

st.set_page_config(page_title="LTV模型预测工具", layout="wide")
st.title("📊 LTV 模型预测工具")

try:
    # 加载配置参数
    config = load_config()
    days_list = config["days_list"]

    # 1. 上传训练参考数据
    st.header("📂 第一步：上传历史参考数据（带LTV标签）")
    ref_file = st.file_uploader("上传CSV文件作为训练数据", type=["csv"], key="ref")

    # 2. 上传需要预测的数据
    st.header("📂 第二步：上传待预测数据")
    pred_file = st.file_uploader("上传CSV文件作为预测数据", type=["csv"], key="pred")

    # 运行主逻辑按钮
    if ref_file and pred_file and st.button("🚀 开始训练与预测"):
        with st.spinner("数据加载与预处理中..."):
            df_ref = pd.read_csv(ref_file).fillna(0)
            df_pred = pd.read_csv(pred_file).fillna(0)

            temp_result = data_preprocess(df_ref, config)
            temp_result_pred = data_preprocess(df_pred, config, train_data=False)

        with st.spinner("训练模型中..."):
            model_results = {}
            for day in days_list:
                x_train_nonpayer, y_train_nonpayer = temp_result["train"][day][
                    "nonpayer"
                ]
                x_train_payer, y_train_payer = temp_result["train"][day]["payer"]
                x_valid_nonpayer, y_valid_nonpayer = temp_result["valid"][day][
                    "nonpayer"
                ]
                x_valid_payer, y_valid_payer = temp_result["valid"][day]["payer"]

                model_results[day] = train_process(
                    x_train_nonpayer,
                    x_valid_nonpayer,
                    x_train_payer,
                    x_valid_payer,
                    y_train_nonpayer,
                    y_valid_nonpayer,
                    y_train_payer,
                    y_valid_payer,
                    config,
                )
                st.write("✅ 脚本已加载，无语法错误")

        with st.spinner("使用验证集重新训练中..."):
            model_test = {}
            params_clf = config["params_clf"]
            params_reg = config["params_reg"]

            for day, res in model_results.items():
                params_clf["num_iterations"] = res["model_clf"].best_iteration
                params_reg["num_iterations"] = res["model_reg"].best_iteration

                x_clf, y_clf = temp_result["valid"][day]["nonpayer"]
                x_reg, y_reg = temp_result["valid"][day]["payer"]

                model_test[day] = train_process(
                    x_clf, x_clf, x_reg, x_reg, y_clf, y_clf, y_reg, y_reg, config
                )
                st.write("✅ 脚本已加载，无语法错误")

        with st.spinner("生成预测中..."):
            preds_results = {}
            for day in days_list:
                _, _, id_test = temp_result_pred["train"][day]["all"]
                x_test_nonpayer, y_test_nonpayer = temp_result_pred["train"][day][
                    "nonpayer"
                ]
                x_test_payer, y_test_payer = temp_result_pred["train"][day]["payer"]

                preds_results[day] = predict_process(
                    x_test_nonpayer,
                    x_test_payer,
                    y_test_nonpayer,
                    y_test_payer,
                    id_test,
                    model_test[day]["model_clf"],
                    model_test[day]["model_reg"],
                    config,
                )

        st.success("✅ 模型预测完成！")

        # 保存预测结果
        output_dir = create_output_dir()
        output_path = f"{output_dir}/ltv_predictions.csv"
        save_predictions(preds_results, output_dir)

        with open(output_path, "rb") as f:
            st.download_button(
                "📥 点击下载预测结果", f, file_name="ltv_predictions.csv"
            )

        # 展示图表
        st.header("📈 模型可视化评估")

        st.subheader("📊 预测值 vs 实际值")
        fig1 = compare_plot(preds_results, config)
        st.pyplot(fig1)

        st.subheader("📉 残差分布图")
        fig2 = residual_plot(preds_results, config)
        st.pyplot(fig2)

        st.subheader("💡 LTV评估指标")
        evaluate_ltv(preds_results, config)
        show_roas_ltv(preds_results, config)

except Exception as e:
    st.error("❌ 发生错误，下面是详细信息：")
    st.code(traceback.format_exc())
