import torch

class Config:
  patience = 5

  batch_size = 32
  num_workers = 4
  
  lr = 0.00003
  n_epoches=100
  load_weights_path = "model/"
  save_file_name = "model_weights_gpt2"
  MODEL_NAME = "gpt2"

  huggingFace_model = "model/huggingFace"
  huggingFace_tokenizer = "model/huggingFace"

  onnx_model_path = "model/onnx_model"
  optimized_fp32_model_path = "model/onnx_model"
  optimized_int8_model_path = "model/onnx_model"

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
