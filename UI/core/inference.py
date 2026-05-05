import numpy as np
import onnxruntime as ort
import os


def run_inference(numpy_input, weight_path="weights/best_model.onnx", batch_size=64):
    """
    使用轻量级 ONNX Runtime 进行分批次推理，完全脱离 PyTorch
    """
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"找不到 ONNX 权重文件：{weight_path}")

    # 1. 确保输入是 numpy 的 float32 类型
    if not isinstance(numpy_input, np.ndarray):
        input_data = np.array(numpy_input, dtype=np.float32)
    else:
        input_data = numpy_input.astype(np.float32)

    # 2. 创建 ONNX 推理会话 (默认使用 CPUExecutionProvider，适配所有普通电脑)
    session = ort.InferenceSession(weight_path, providers=['CPUExecutionProvider'])

    # 获取模型的输入节点名称 (我们在导出时命名为了 'input')
    input_name = session.get_inputs()[0].name

    all_predictions = []
    all_probs = []

    # 3. 分批次推理 (保留批处理逻辑，保证处理整夜长录音时内存稳定)
    num_samples = input_data.shape[0]
    for i in range(0, num_samples, batch_size):
        batch_x = input_data[i:i + batch_size]

        # 执行纯 C++ 后端运算，速度极快
        outputs = session.run(None, {input_name: batch_x})
        logits = outputs[0]

        # 4. 用 NumPy 实现 Softmax 和 Argmax
        # 减去最大值防止指数爆炸
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        preds = np.argmax(probs, axis=1)

        all_predictions.extend(preds)
        all_probs.extend(probs)

    return np.array(all_predictions), np.array(all_probs)