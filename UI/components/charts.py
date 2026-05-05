import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.fft import rfft, rfftfreq

# =================================================================
# 🌟 全局排版与字体约束
# =================================================================
FONT_FAMILY = "Times New Roman, SimSun, serif"

# 1. 全局标准字体 (坐标轴、图例、悬浮窗、文字标注) - 12号字
GLOBAL_FONT = dict(family=FONT_FAMILY, size=12, color="black")

# 2. 主标题字体 - 18号字，略大显眼
TITLE_FONT = dict(family=FONT_FAMILY, size=18, color="black")

# 3. 子图标题字体 - 14号字
SUBTITLE_FONT = dict(family=FONT_FAMILY, size=14, color="black")

# 4. 全局坐标轴统一样式 (DRY 原则，避免重复代码)
COMMON_AXIS_CFG = dict(
    showgrid=True, gridcolor='rgba(200, 200, 200, 0.2)',
    showline=True, linecolor='black', linewidth=1,
    ticks='outside', tickcolor='black',
    title_font=GLOBAL_FONT, tickfont=GLOBAL_FONT
)


def plot_hypnogram_dual(predictions, true_labels=None, is_expert_mode=False):
    """
    绘制睡眠结构图 (Hypnogram)
    """
    mapping = {0: 4, 4: 3, 1: 2, 2: 1, 3: 0, 5: -1}
    x_values = np.arange(len(predictions)) * 30 / 3600
    y_pred = [mapping.get(int(p), -1) for p in predictions]

    tickvals = [-1, 0, 1, 2, 3, 4]
    ticktext = ['?', 'N3', 'N2', 'N1', 'REM', 'Wake']

    # 提前计算一个 Epoch 的时间跨度 (小时)
    dx = 30 / 3600

    if is_expert_mode and true_labels is not None:
        y_expert = [mapping.get(int(t), -1) for t in true_labels]
        fig = go.Figure()

        text_ai = [f"Epoch {i} | AI原始预测: {ticktext[tickvals.index(y_pred[i])]}" for i in range(len(y_pred))]
        fig.add_trace(go.Scatter(
            x=x_values, y=y_pred, mode='lines',
            line=dict(color='rgba(46, 134, 193, 0.3)', width=2),
            name='AI原始预测', text=text_ai,
            hovertemplate="<b>%{text}</b><br>入睡时间: %{x:.2f} 小时<extra></extra>"
        ))

        text_expert = [f"Epoch {i} | 最终判读: {ticktext[tickvals.index(y_expert[i])]}" for i in range(len(y_pred))]
        fig.add_trace(go.Scatter(
            x=x_values, y=y_expert, mode='lines',
            line=dict(color='#2E86C1', width=1.5),
            line_shape='hv', name='专家当前判读',
            text=text_expert,
            hovertemplate="<b>%{text}</b><br>入睡时间: %{x:.2f} 小时<extra></extra>"
        ))

        red_x, red_y, marker_x, marker_y = [], [], [], []
        for i in range(len(predictions)):
            if int(predictions[i]) != 5 and int(true_labels[i]) != 5:
                if int(predictions[i]) != int(true_labels[i]):
                    y_prev = y_expert[i - 1] if i > 0 else y_expert[i]
                    y_curr = y_expert[i]
                    x_curr = x_values[i]
                    x_next = x_values[i + 1] if i < len(x_values) - 1 else x_values[i] + dx

                    red_x.extend([x_curr, x_curr, x_next, None])
                    red_y.extend([y_prev, y_curr, y_curr, None])
                    marker_x.append(x_curr)
                    marker_y.append(y_curr)

        if red_x:
            fig.add_trace(
                go.Scatter(x=red_x, y=red_y, mode='lines', line=dict(color='#E74C3C', width=2.5), hoverinfo='skip',
                           showlegend=False))
            fig.add_trace(go.Scatter(x=marker_x, y=marker_y, mode='markers',
                                     marker=dict(color='#E74C3C', size=6, symbol='circle'), hoverinfo='skip',
                                     showlegend=False))

        fig.update_layout(
            title=dict(text="临床睡眠结构判读 (红线标示专家修改轨迹)", font=TITLE_FONT),
            height=300
        )
        # 单图没有子图标题，如果有额外注释强制应用全局字体
        for annotation in fig.layout.annotations if fig.layout.annotations else []:
            annotation.font = GLOBAL_FONT

        fig.update_yaxes(tickmode='array', tickvals=tickvals, ticktext=ticktext, **COMMON_AXIS_CFG)
        fig.update_xaxes(title_text="入睡时间 (小时)", **COMMON_AXIS_CFG)

    elif not is_expert_mode and true_labels is not None:
        y_true = [mapping.get(int(t), -1) for t in true_labels]

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
            subplot_titles=("专家真实标注 (Ground Truth)", "模型预测 (Model Prediction)")
        )

        text_gt = [f"Epoch {i} | 真实标注: {ticktext[tickvals.index(y_true[i])]}" for i in range(len(y_pred))]
        fig.add_trace(go.Scatter(
            x=x_values, y=y_true, mode='lines', line=dict(color='#2CA02C', width=1.2),
            line_shape='hv', name='真实标注', text=text_gt,
            hovertemplate="<b>%{text}</b><br>入睡时间: %{x:.2f} 小时<extra></extra>"
        ), row=1, col=1)

        text_pred = [f"Epoch {i} | 模型预测: {ticktext[tickvals.index(y_pred[i])]}" for i in range(len(y_pred))]
        fig.add_trace(go.Scatter(
            x=x_values, y=y_pred, mode='lines', line=dict(color='#2E86C1', width=1.2),
            line_shape='hv', name='模型预测', text=text_pred,
            hovertemplate="<b>%{text}</b><br>入睡时间: %{x:.2f} 小时<extra></extra>"
        ), row=2, col=1)

        for i in range(len(predictions)):
            if int(predictions[i]) != 5 and int(true_labels[i]) != 5:
                if int(predictions[i]) != int(true_labels[i]):
                    fig.add_shape(
                        type="line", x0=x_values[i], y0=-1, x1=x_values[i], y1=4,
                        line=dict(color="red", width=1, dash="solid"),
                        opacity=0.6, layer="below", row=2, col=1
                    )

        fig.update_layout(height=400)

        # 强制子图标题使用统一的副标题字体
        for annotation in fig.layout.annotations:
            annotation.font = SUBTITLE_FONT

        for row_idx in [1, 2]:
            fig.update_yaxes(tickmode='array', tickvals=tickvals, ticktext=ticktext, row=row_idx, col=1,
                             **COMMON_AXIS_CFG)
            fig.update_xaxes(row=row_idx, col=1, **COMMON_AXIS_CFG)
        fig.update_xaxes(title_text="入睡时间 (小时)", row=2, col=1)

    # 全局 Layout 约束
    fig.update_layout(
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=40, r=40, t=50, b=40),
        font=GLOBAL_FONT, hoverlabel=dict(font=GLOBAL_FONT),
        hovermode="x unified", showlegend=False
    )
    return fig


def plot_probability_map(predictions, probs_array):
    x_values = np.arange(len(predictions)) * 30 / 3600
    prob_colors = {0: '#FFD54F', 1: '#B3E5FC', 2: '#29B6F6', 3: '#1565C0', 4: '#AB47BC'}
    stage_names = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}

    fig = go.Figure()
    for class_idx in range(5):
        text_prob = [f"Epoch {i} | {stage_names[class_idx]} Prob: {val:.2f}" for i, val in
                     enumerate(probs_array[:, class_idx])]
        fig.add_trace(go.Scatter(
            x=x_values, y=probs_array[:, class_idx], mode='none',
            fillcolor=prob_colors[class_idx], stackgroup='one',
            name=f'{stage_names[class_idx]} Prob',
            text=text_prob, hovertemplate="<b>%{text}</b><extra></extra>"
        ))

    max_probs = np.max(probs_array, axis=1)
    fig.add_trace(go.Scatter(
        x=x_values, y=max_probs, mode='lines',
        line=dict(color='black', width=1.5), name='置信度 (Confidence)'
    ))

    fig.update_layout(
        title=dict(text="全阶段概率分布与确信度", font=TITLE_FONT),
        height=350, plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=40, r=40, t=60, b=40), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=GLOBAL_FONT),
        font=GLOBAL_FONT, hoverlabel=dict(font=GLOBAL_FONT)
    )
    fig.update_yaxes(title_text="Probability", range=[0, 1.05], **COMMON_AXIS_CFG)
    fig.update_xaxes(title_text="入睡时间 (小时)", **COMMON_AXIS_CFG)
    return fig


def plot_eeg_and_fft_dual(signal_array, current_epoch, pred_label, true_label=None, sfreq=100, epoch_sec=30,
                          channel_name="EEG", is_expert_mode=False):
    """
    绘制微观审查图：上下双轨合并展示 (时域波形 + FFT 频域谱)
    """
    label_names = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM', 5: 'Unknown', -1: '?'}

    start_idx = current_epoch * sfreq * epoch_sec
    end_idx = (current_epoch + 1) * sfreq * epoch_sec
    segment = signal_array[start_idx:end_idx]

    start_time_min = (current_epoch * epoch_sec) / 60.0
    end_time_min = ((current_epoch + 1) * epoch_sec) / 60.0
    time_axis = np.linspace(start_time_min, end_time_min, len(segment))

    yf = rfft(segment)
    xf = rfftfreq(len(segment), 1 / sfreq)
    amplitude = np.abs(yf) / len(segment) * 2
    freq_mask = xf <= 35
    f_plot = xf[freq_mask]
    amp_plot = amplitude[freq_mask]

    pred_str = label_names.get(int(pred_label), '?')
    title_text = f"Epoch {current_epoch} (时间: {start_time_min:.1f} min) | 通道: {channel_name}"

    if true_label is not None:
        true_str = label_names.get(int(true_label), '?')
        if is_expert_mode:
            title_text += f"<br><span style='color:black; font-size:13px;'>专家当前判读: {true_str} | AI 原始预测: {pred_str}</span>"
        else:
            if true_str != 'Unknown' and pred_str != true_str:
                title_text += f"<br><span style='color:black; font-size:13px;'>专家真实标注: {true_str} | 模型预测: {pred_str} (❌ 错误)</span>"
            else:
                title_text += f"<br><span style='color:black; font-size:13px;'>专家真实标注: {true_str} | 模型预测: {pred_str} (✅ 正确)</span>"
    else:
        title_text += f"<br><span style='color:black; font-size:13px;'>模型预测: {pred_str}</span>"

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.15,
        subplot_titles=("时域波形", "FFT 频域谱")
    )

    fig.add_trace(
        go.Scatter(x=time_axis, y=segment, mode='lines', line=dict(color='#2E86C1', width=1.2), name='EEG Signal',
                   hoverinfo='skip'), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=f_plot, y=amp_plot, mode='lines', line=dict(color='#2E86C1', width=1.5), name='FFT Spectrum',
                   fill='tozeroy', fillcolor='rgba(46, 134, 193, 0.2)'), row=2, col=1)

    fig.update_layout(
        title=dict(text=title_text, font=TITLE_FONT),
        height=450, plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=40, r=40, t=80, b=40), showlegend=False,
        hovermode="x unified", font=GLOBAL_FONT, hoverlabel=dict(font=GLOBAL_FONT)
    )

    for annotation in fig.layout.annotations:
        annotation.font = SUBTITLE_FONT

    # X, Y 轴应用统一样式，并为个别轴增加额外设置(如zeroline)
    fig.update_xaxes(title_text="", range=[start_time_min, end_time_min], row=1, col=1, **COMMON_AXIS_CFG)
    fig.update_yaxes(title_text="Amplitude (μV)", zeroline=True, zerolinecolor='rgba(0,0,0,0.3)', zerolinewidth=1,
                     row=1, col=1, **COMMON_AXIS_CFG)

    fig.update_xaxes(title_text="Frequency (Hz)", range=[0, 35], row=2, col=1, **COMMON_AXIS_CFG)
    fig.update_yaxes(title_text="Magnitude", rangemode="tozero", row=2, col=1, **COMMON_AXIS_CFG)
    return fig


def plot_stage_proportions(predictions):
    valid_preds = [int(p) for p in predictions if int(p) != 5]
    counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for p in valid_preds:
        if p in counts: counts[p] += 1

    labels_map = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
    colors_map = {'Wake': '#FFE082', 'REM': '#CE93D8', 'N1': '#E1F5FE', 'N2': '#B3E5FC', 'N3': '#9FA8DA'}

    labels, values, colors = [], [], []
    for k, v in counts.items():
        if v > 0:
            stage_name = labels_map[k]
            labels.append(stage_name)
            values.append(v)
            colors.append(colors_map[stage_name])

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=0.4, marker_colors=colors,
        textinfo='percent+label',
        textfont=GLOBAL_FONT,  # 强制统一饼图内文字字体
        insidetextorientation='horizontal',
        hovertemplate='%{label}<br>数量: %{value} 段 (Epochs)<br>占比: %{percent}<extra></extra>',
        textposition='inside'
    )])

    fig.update_layout(
        showlegend=False, margin=dict(l=30, r=30, t=30, b=30),
        height=350, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=GLOBAL_FONT, hoverlabel=dict(font=GLOBAL_FONT)
    )
    return fig