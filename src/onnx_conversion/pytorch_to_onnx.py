import os
import torch

from onnxruntime.transformers.quantize_helper import QuantizeHelper
from onnxruntime.transformers.gpt2_helper import Gpt2Helper

from model import get_model

from train_config import Config

print(f"python3 -m transformers.onnx --model={Config.huggingFace_model} --feature=causal-lm --atol=8e-05 {Config.onnx_model_path}/"
)
model, tokenizer = get_model()

models_path = f"{Config.load_weights_path}{Config.save_file_name}.pt"
model.load_state_dict(torch.load(models_path, map_location=torch.device('cpu'))["model_state_dict"])
model.save_pretrained(f"{Config.huggingFace_model}")
tokenizer.save_pretrained(f"{Config.huggingFace_tokenizer}")

"""
os.system(
    f"python3 -m transformers.onnx --model={Config.huggingFace_model} --feature=causal-lm --atol=8e-05 {Config.onnx_model_path}/"
)
"""

optimized_fp32_model_path = f"{Config.optimized_fp32_model_path}/gpt2_float32.onnx"
quantized_int8_model_path = f"{Config.optimized_int8_model_path}/gpt2_int8.onnx"
Gpt2Helper.optimize_onnx(f"{Config.onnx_model_path}/model.onnx", optimized_fp32_model_path, False, model.config.num_attention_heads, model.config.hidden_size)
QuantizeHelper.quantize_onnx_model(optimized_fp32_model_path, quantized_int8_model_path)