import streamlit as st
import os
import numpy as np
import pandas as pd

from core.eeg_processor import get_edf_channels, process_uploaded_edf, extract_true_labels, truncate_wake_periods
from core.inference import run_inference

# 🌟 导入合并后的绘图组件
from components.charts import (
    plot_hypnogram_dual,
    plot_probability_map,
    plot_eeg_and_fft_dual,
    plot_stage_proportions
)

st.set_page_config(page_title="EEG 睡眠分期系统", layout="wide", initial_sidebar_state="expanded")

if "analyzed" not in st.session_state:
    st.session_state.analyzed = False

st.title("睡眠健康分析---睡眠分期系统")
st.markdown("睡眠健康分析系统的分期模型研究与开发")
st.divider()

with st.sidebar:
    st.header("⚙ 控制面板")
    uploaded_edf = st.file_uploader("上传脑电信号文件 (.edf)", type=["edf"])
    uploaded_label = st.file_uploader("上传真实标签文件 (可选)", type=["edf"])

    selected_channel = None
    temp_edf_path = "data/temp_uploaded.edf"
    temp_label_path = "data/temp_label.edf"

    if uploaded_edf is not None:
        os.makedirs(os.path.dirname(temp_edf_path), exist_ok=True)
        with open(temp_edf_path, "wb") as f:
            f.write(uploaded_edf.getvalue())

        if uploaded_label is not None:
            with open(temp_label_path, "wb") as f:
                f.write(uploaded_label.getvalue())

        channels = get_edf_channels(temp_edf_path)
        if channels:
            st.subheader("参数设置")
            default_idx = channels.index("EEG Fpz-Cz") if "EEG Fpz-Cz" in channels else 0
            selected_channel = st.selectbox(" 请选择要分析的脑电通道", options=channels, index=default_idx)

    st.markdown("---")
    btn_disabled = uploaded_edf is None or selected_channel is None
    analyze_btn = st.button(" 开始睡眠分期辅助", type="primary", width="stretch", disabled=btn_disabled)

if analyze_btn:
    with st.spinner(f'正在对 {selected_channel} 通道进行滤波、推理与首尾截断，请稍候...'):
        try:
            # 1. 接收从处理脚本传过来的 numpy 数组 (摆脱了 tensor)
            numpy_input, signal_trimmed, actual_channel, raw_num_batches = process_uploaded_edf(
                temp_edf_path, target_channel=selected_channel
            )

            # 2. 🌟 关键点：路径后缀改成 .onnx！并传入 numpy_input
            predictions, probs_array = run_inference(numpy_input, weight_path="weights/best_model.onnx")

            true_labels = None
            if uploaded_label is not None:
                true_labels = extract_true_labels(temp_label_path, raw_num_batches)

            trunc_signal, trunc_preds, trunc_trues, final_batches = truncate_wake_periods(
                signal_trimmed, predictions, true_labels, keep_wake_mins=30
            )

            start_idx = 0
            if len(trunc_preds) < len(predictions):
                for i in range(len(predictions) - len(trunc_preds) + 1):
                    if np.array_equal(predictions[i:i + len(trunc_preds)], trunc_preds):
                        start_idx = i
                        break
            trunc_probs = probs_array[start_idx: start_idx + len(trunc_preds)]

            st.session_state.signal_trimmed = trunc_signal
            st.session_state.predictions = trunc_preds
            st.session_state.probs_array = trunc_probs
            st.session_state.actual_channel = actual_channel
            st.session_state.num_batches = final_batches
            st.session_state.raw_num_batches = raw_num_batches

            # 模式判定
            if trunc_trues is not None:
                st.session_state.mode = "Evaluation"  # 评估模式
                st.session_state.true_labels = trunc_trues
                st.session_state.expert_labels = None
            else:
                st.session_state.mode = "Expert"  # 专家辅助模式
                st.session_state.true_labels = None
                st.session_state.expert_labels = trunc_preds.copy()

            st.session_state.analyzed = True

        except Exception as e:
            st.error(f"处理过程中发生错误: {e}")
            st.stop()

# ==========================================
#  UI 渲染逻辑
# ==========================================
if st.session_state.analyzed:

    if "current_epoch" not in st.session_state:
        st.session_state.current_epoch = 0

    st.subheader("睡眠分期深度可视分析")

    # 渲染宏观结构图
    if st.session_state.mode == "Evaluation":
        fig_hypno = plot_hypnogram_dual(
            predictions=st.session_state.predictions,
            true_labels=st.session_state.true_labels,
            is_expert_mode=False
        )
    else:
        fig_hypno = plot_hypnogram_dual(
            predictions=st.session_state.predictions,
            true_labels=st.session_state.expert_labels,
            is_expert_mode=True
        )

    st.plotly_chart(fig_hypno, width="stretch", key="chart_hypno")

    fig_prob_map = plot_probability_map(st.session_state.predictions, st.session_state.probs_array)
    st.plotly_chart(fig_prob_map, width="stretch", key="chart_prob")

    st.divider()

    # ------------------------------------------
    # 第二段：局部波形审查
    # ------------------------------------------
    st.subheader("原始EEG信号审查")


    def update_slider():
        st.session_state.current_epoch = st.session_state.epoch_slider_widget


    current_epoch = st.slider(
        "滑动时间轴以审查细节 (单位: 30秒/Epoch)",
        min_value=0, max_value=st.session_state.num_batches - 1,
        value=st.session_state.current_epoch, step=1,
        key="epoch_slider_widget", on_change=update_slider
    )

    col_wave, col_edit = st.columns([4.5, 1.5])
    stage_map = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

    with col_wave:
        pred_l = st.session_state.predictions[current_epoch]
        ref_l = st.session_state.true_labels[current_epoch] if st.session_state.mode == "Evaluation" else \
        st.session_state.expert_labels[current_epoch]

        fig_micro = plot_eeg_and_fft_dual(
            st.session_state.signal_trimmed, current_epoch, pred_label=pred_l,
            true_label=ref_l, channel_name=st.session_state.actual_channel,
            is_expert_mode=(st.session_state.mode == "Expert")
        )
        st.plotly_chart(fig_micro, width="stretch")

    with col_edit:
        current_ai_idx = int(st.session_state.predictions[current_epoch])
        probs = st.session_state.probs_array[current_epoch]
        ai_conf = probs[current_ai_idx] * 100

        if st.session_state.mode == "Evaluation":
            st.markdown("### 📊 阶段诊断详情")
            st.markdown("---")
            current_gt_idx = int(st.session_state.true_labels[current_epoch])
            st.metric(label="🤖 模型预测", value=stage_map.get(current_ai_idx, "未知"),
                      delta=f"置信度: {ai_conf:.1f}%", delta_color="off")
            st.markdown("---")
            if current_gt_idx == current_ai_idx:
                st.success(f"👨‍⚕️ 真实标注: **{stage_map.get(current_gt_idx, '未知')}**\n\n✅ 预测正确")
            else:
                st.error(f"👨‍⚕️ 真实标注: **{stage_map.get(current_gt_idx, '未知')}**\n\n❌ 预测错误")
        else:
            st.markdown("### 判读面板")
            st.markdown("---")
            reverse_map = {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
            current_expert_idx = int(st.session_state.expert_labels[current_epoch])

            sorted_indices = np.argsort(probs)[::-1]
            second_best_idx = sorted_indices[1]
            second_conf = probs[second_best_idx] * 100

            st.metric(label="模型原始预测", value=stage_map.get(current_ai_idx, "未知"),
                      delta=f"确信度: {ai_conf:.1f}%", delta_color="off")
            st.caption(f"*次优可能: {stage_map[second_best_idx]} ({second_conf:.1f}%)*")
            st.markdown("---")

            new_label_str = st.selectbox(
                " 修改判读结果:",
                options=["Wake", "N1", "N2", "N3", "REM"],
                index=list(reverse_map.values()).index(current_expert_idx),
                key=f"edit_box_{current_epoch}"
            )

            if reverse_map[new_label_str] != current_expert_idx:
                st.session_state.expert_labels[current_epoch] = reverse_map[new_label_str]
                st.rerun()

        st.markdown("---")
        st.markdown("**快速导航**")
        btn_col1, btn_col2 = st.columns(2)


        # ==========================================
        # 🌟 修复 BUG：使用 on_click 回调函数处理状态同步
        # 回调函数会在页面重绘之前执行，完美避开组件修改冲突！
        # ==========================================
        def go_prev():
            new_epoch = max(0, st.session_state.current_epoch - 1)
            st.session_state.current_epoch = new_epoch
            st.session_state.epoch_slider_widget = new_epoch


        def go_next():
            new_epoch = min(st.session_state.num_batches - 1, st.session_state.current_epoch + 1)
            st.session_state.current_epoch = new_epoch
            st.session_state.epoch_slider_widget = new_epoch


        with btn_col1:
            # 注意：把逻辑绑定到 on_click 参数上，并删除了底下的 if 逻辑
            st.button("◀ 上一段", width="stretch", on_click=go_prev)
        with btn_col2:
            st.button("下一段 ▶", width="stretch", on_click=go_next)

    st.divider()

    # ------------------------------------------
    # 第三段：智能报告看板
    # ------------------------------------------
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("分析报告与指标计算")
        m1, m2 = st.columns(2)
        m3, m4 = st.columns(2)

        total_hours = st.session_state.num_batches * 30 / 3600
        m1.metric("总分析时长", f"{total_hours:.2f} 小时")
        m2.metric("有效分段 (Epochs)", f"{st.session_state.num_batches} 段")

        if st.session_state.mode == "Evaluation":
            valid_mask = st.session_state.true_labels != 5
            valid_trues = st.session_state.true_labels[valid_mask]
            valid_preds = st.session_state.predictions[valid_mask]

            if len(valid_trues) > 0:
                correct_epochs = np.sum(valid_trues == valid_preds)
                accuracy = correct_epochs / len(valid_trues) * 100
                error_epochs = len(valid_trues) - correct_epochs
                m3.metric("模型整体准确率 (Accuracy)", f"{accuracy:.2f} %")
                m4.metric("错误判定段数", f"{error_epochs} 段")
        else:
            m3.metric("评估状态", "临床辅助诊断中")
            m4.metric("人工介入段数", f"{np.sum(st.session_state.predictions != st.session_state.expert_labels)} 段")

        st.markdown("---")
        df_report = pd.DataFrame({
            "Epoch (30s)": range(len(st.session_state.predictions)),
            "Time (Min)": np.arange(len(st.session_state.predictions)) * 0.5,
            "AI_Prediction": [stage_map.get(int(x), "?") for x in st.session_state.predictions]
        })

        if st.session_state.mode == "Evaluation":
            df_report["Ground_Truth"] = [stage_map.get(int(x), "?") for x in st.session_state.true_labels]
            df_report["Is_Error"] = df_report["AI_Prediction"] != df_report["Ground_Truth"]
            csv_data = df_report.to_csv(index=False).encode('utf-8-sig')
            st.download_button(" 导出模型性能评估报告 (CSV)", data=csv_data, file_name="evaluation_report.csv",
                               mime="text/csv", type="primary", width="stretch")
        else:
            df_report["Expert_Final"] = [stage_map.get(int(x), "?") for x in st.session_state.expert_labels]
            df_report["Is_Modified"] = df_report["AI_Prediction"] != df_report["Expert_Final"]
            csv_data = df_report.to_csv(index=False).encode('utf-8-sig')
            st.download_button(" 导出临床最终判读报告 (CSV)", data=csv_data, file_name="clinical_sleep_report.csv",
                               mime="text/csv", type="primary", width="stretch")

    with col2:
        st.subheader("阶段睡眠时间占比")
        target_labels = st.session_state.true_labels if st.session_state.mode == "Evaluation" else st.session_state.expert_labels
        fig_pie = plot_stage_proportions(target_labels)
        st.plotly_chart(fig_pie, width="stretch")

else:
    if not analyze_btn:
        st.info(" 欢迎使用。请在左侧面板上传 EDF 数据文件以开始您的数据分析。")

#streamlit run D:\PyCharm\project\UI\app.py
