import torch
import sys
import types
import os

# =================================================================
# 🪄 黑科技必须搬过来！欺骗 PyTorch 的 pickle 解码器
# 拦截加载 .pth 时对 'parse_config' 的强行导入，防止崩溃
# =================================================================
dummy_module = types.ModuleType('parse_config')


class DummyConfig:
    pass


dummy_module.ConfigParser = DummyConfig
sys.modules['parse_config'] = dummy_module
# =================================================================

# 🌟 导入你真实的网络结构类名
from core.model_builder import Sleep_Model


def convert_to_onnx():
    print("正在加载 PyTorch 模型...")

    # 1. 实例化模型
    model = Sleep_Model()

    # 2. 安全地加载并提取权重（照抄 inference.py 的安全逻辑）
    weight_path = "weights/best_model.pth"  # 请确保在 UI 目录下运行脚本
    if not os.path.exists(weight_path):
        # 兼容一下绝对路径，防止路径不对
        weight_path = r"D:\PyCharm\project\UI\weights\best_model.pth"

    checkpoint = torch.load(weight_path, map_location='cpu')

    # 剥离出纯净的权重矩阵
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("成功从 checkpoint 字典中提取 state_dict")
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # 将纯净的权重放进模型
    model.load_state_dict(state_dict, strict=False)
    model.eval()  # 🌟 必须切换到评估模式！

    # 3. 构造一个虚拟输入 (Dummy Input)
    # 假设输入维度是 (BatchSize, Channel, Length) -> (1, 1, 3000)
    dummy_input = torch.randn(1, 1, 3000, dtype=torch.float32)

    # 4. 导出 ONNX
    onnx_path = "best_model.onnx"
    if not os.path.exists("weights"):
        os.makedirs("weights")

    print("正在导出为 ONNX 格式，请稍候...")
    torch.onnx.export(
        model,  # 要转换的模型
        dummy_input,  # 虚拟输入
        onnx_path,  # 保存路径
        export_params=True,  # 将训练好的权重一并导出
        opset_version=11,  # ONNX 的算子版本
        do_constant_folding=True,  # 执行常量折叠优化
        input_names=['input'],  # 输入节点名称
        output_names=['output'],  # 输出节点名称
        dynamic_axes={  # 让 Batch Size 变成动态的
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"✅ 转换成功！ONNX 模型已保存至: {onnx_path}")
    print("🎉 现在你可以删掉 PyTorch，投入 ONNX Runtime 的怀抱了！")


if __name__ == "__main__":
    convert_to_onnx()