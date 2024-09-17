import sys
sys.path.append('../')
from utils import load_config, getInputArguments
from datasets import HiRISEData, BlenderLunarHIRISEData
from model import MetricLDN
import torch
import onnx
import onnx_tf
import tensorflow as tf
import numpy as np
torch.set_float32_matmul_precision('high')


# --- Gather arguments
args = getInputArguments(sys.argv)
cfg = load_config(args.cfg)

# --- Create Dataset and Model
if args.dataset == 'hirise':
    dataset = HiRISEData(cfg['hirise_images'], cfg['hirise_labels'], 'crater', batch_size=args.batch_size)
elif args.dataset == 'lunar':
    dataset = BlenderLunarHIRISEData(cfg['lunar_images'], cfg['lunar_trajPkl'], batch_size=args.batch_size)
args.train_ds = dataset.train_dataset
model = MetricLDN(args=args).eval()

input_shape = (1, 3, 128, 128)
torch.onnx.export(model, torch.randn(input_shape), 'metric_ldn.onnx', opset_version=12, input_names=['input'], output_names=['output'])

# Load  ONNX model
onnx_model = onnx.load('metric_ldn.onnx')
# Convert ONNX model to TensorFlow format
tf_model = onnx_tf.backend.prepare(onnx_model)
# Export  TensorFlow  model 
tf_model.export_graph("metric_ldn.tf")

converter = tf.lite.TFLiteConverter.from_saved_model("metric_ldn.tf")
tflite_model = converter.convert()
open('metric_ldn.tflite', 'wb').write(tflite_model)
