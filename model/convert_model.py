#!/usr/bin/env python3
"""Convert mushrooms.pb (TF1 frozen MobileNet V1) to mushrooms.tflite.
Path: mushrooms.pb -> PyTorch model -> mushrooms.onnx -> mushrooms.tflite
"""

import struct
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Step 1: Parse TF frozen graph protobuf ──────────────────────────────────

def read_varint(data, pos):
    result, shift = 0, 0
    while True:
        b = data[pos]; pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result, pos

def read_field(data, pos):
    if pos >= len(data):
        return None, None, None, pos
    tag, pos = read_varint(data, pos)
    fn, wt = tag >> 3, tag & 0x7
    if wt == 0:
        v, pos = read_varint(data, pos)
    elif wt == 2:
        l, pos = read_varint(data, pos)
        v = data[pos:pos + l]; pos += l
    elif wt == 5:
        v = data[pos:pos + 4]; pos += 4
    elif wt == 1:
        v = data[pos:pos + 8]; pos += 8
    else:
        return None, None, None, pos
    return fn, wt, v, pos

def parse_tensor_shape(data):
    dims = []
    pos = 0
    while pos < len(data):
        fn, wt, v, pos = read_field(data, pos)
        if fn is None:
            break
        if fn == 2 and wt == 2:
            dpos = 0
            while dpos < len(v):
                dfn, _, dv, dpos = read_field(v, dpos)
                if dfn is None:
                    break
                if dfn == 1:
                    dims.append(dv)
    return dims

def parse_tensor_proto(data):
    dtype = None; shape = []; tensor_content = None; float_vals = []
    pos = 0
    while pos < len(data):
        fn, wt, v, pos = read_field(data, pos)
        if fn is None:
            break
        if fn == 1 and wt == 0:
            dtype = v
        elif fn == 2 and wt == 2:
            shape = parse_tensor_shape(v)
        elif fn == 4 and wt == 2:
            tensor_content = v
        elif fn == 5 and wt == 5:
            float_vals.append(struct.unpack('<f', v)[0])
    if dtype == 1:  # DT_FLOAT
        if tensor_content:
            arr = np.frombuffer(bytes(tensor_content), dtype=np.float32).copy()
        elif float_vals:
            arr = np.array(float_vals, dtype=np.float32)
        else:
            arr = np.array([], dtype=np.float32)
        # shape=[] means scalar; shape=[N,...] means reshape
        if shape:
            arr = arr.reshape(shape)
        elif arr.size == 0 and len(float_vals) == 0:
            # scalar stored in tensor_content with empty shape
            if tensor_content and len(tensor_content) == 4:
                arr = np.frombuffer(bytes(tensor_content), dtype=np.float32).copy()
        return arr
    return None

def tensor_from_attrvalue(data):
    pos = 0
    while pos < len(data):
        fn, wt, v, pos = read_field(data, pos)
        if fn is None:
            break
        if fn == 8 and wt == 2:  # AttrValue.tensor
            return parse_tensor_proto(v)
    return None

def parse_nodedef(data):
    n = {'name': '', 'op': '', 'attrs': {}}
    pos = 0
    while pos < len(data):
        fn, wt, v, pos = read_field(data, pos)
        if fn is None:
            break
        if fn == 1 and wt == 2:
            n['name'] = v.decode('utf-8', 'replace')
        elif fn == 2 and wt == 2:
            n['op'] = v.decode('utf-8', 'replace')
        elif fn == 5 and wt == 2:
            key = None; attr_data = None
            apos = 0
            while apos < len(v):
                afn, awt, av, apos = read_field(v, apos)
                if afn is None:
                    break
                if afn == 1 and awt == 2:
                    key = av.decode('utf-8', 'replace')
                elif afn == 2 and awt == 2:
                    attr_data = av
            if key:
                n['attrs'][key] = attr_data
    return n

print("Parsing frozen graph...")
with open('mushrooms.pb', 'rb') as f:
    pb_data = f.read()

const_tensors = {}
pos = 0
while pos < len(pb_data):
    fn, wt, v, pos = read_field(pb_data, pos)
    if fn is None:
        break
    if fn == 1 and wt == 2:
        nd = parse_nodedef(v)
        if nd['op'] == 'Const' and 'value' in nd['attrs']:
            arr = tensor_from_attrvalue(nd['attrs']['value'])
            if arr is not None:
                const_tensors[nd['name']] = arr

print(f"Extracted {len(const_tensors)} weight tensors")

def get_np(name):
    t = const_tensors.get(name)
    if t is None:
        raise KeyError(f"Tensor not found: {name}")
    return t

# Read BN epsilon from the graph (the small constant added to variance)
eps_key = 'MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/batchnorm/add/y'
try:
    eps = float(const_tensors[eps_key].flat[0])
except (KeyError, IndexError):
    eps = 0.001
print(f"BatchNorm epsilon: {eps}")

# Number of output classes
num_classes = const_tensors['final_training_ops/biases/final_biases'].shape[0]
print(f"Number of output classes: {num_classes}")

# ── Step 2: Build PyTorch MobileNet V1 ──────────────────────────────────────

class DSBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, bn_eps):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_ch, eps=bn_eps)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_ch, eps=bn_eps)

    def forward(self, x):
        x = F.relu6(self.dw_bn(self.dw(x)))
        return F.relu6(self.pw_bn(self.pw(x)))

class MushroomNet(nn.Module):
    # MobileNet V1 backbone + original ImageNet logits + retrained final layer
    BLOCK_CFG = [
        (32,   64,  1),   # block 1
        (64,  128,  2),   # block 2
        (128, 128,  1),   # block 3
        (128, 256,  2),   # block 4
        (256, 256,  1),   # block 5
        (256, 512,  2),   # block 6
        (512, 512,  1),   # block 7
        (512, 512,  1),   # block 8
        (512, 512,  1),   # block 9
        (512, 512,  1),   # block 10
        (512, 512,  1),   # block 11
        (512, 1024, 2),   # block 12
        (1024,1024, 1),   # block 13
    ]

    def __init__(self, num_classes, bn_eps):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.conv0_bn = nn.BatchNorm2d(32, eps=bn_eps)
        self.blocks = nn.ModuleList(
            [DSBlock(ic, oc, s, bn_eps) for ic, oc, s in self.BLOCK_CFG]
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # Original MobileNet V1 logits layer (kept intact in the retrained graph)
        self.logits_conv = nn.Conv2d(1024, 1001, 1, bias=True)
        # New retrained classifier
        self.fc = nn.Linear(1001, num_classes, bias=True)

    def forward(self, x):
        x = F.relu6(self.conv0_bn(self.conv0(x)))
        for block in self.blocks:
            x = block(x)
        x = self.avgpool(x)              # (N, 1024, 1, 1)
        x = self.logits_conv(x)          # (N, 1001, 1, 1)
        x = x.flatten(1)                 # (N, 1001)
        x = self.fc(x)                   # (N, num_classes)
        return F.softmax(x, dim=1)

model = MushroomNet(num_classes, eps)

# ── Step 3: Load weights from frozen graph ───────────────────────────────────

def load_conv_weight(module, tf_arr):
    # TF: [H, W, in, out] → PyTorch: [out, in, H, W]
    module.weight = nn.Parameter(torch.from_numpy(tf_arr.transpose(3, 2, 0, 1).copy()))

def load_dw_weight(module, tf_arr):
    # TF: [H, W, C, 1] → PyTorch: [C, 1, H, W]
    module.weight = nn.Parameter(torch.from_numpy(tf_arr.transpose(2, 3, 0, 1).copy()))

def load_bn(bn, gamma, beta, mean, var):
    bn.weight = nn.Parameter(torch.from_numpy(gamma.copy()))
    bn.bias = nn.Parameter(torch.from_numpy(beta.copy()))
    bn.running_mean.copy_(torch.from_numpy(mean.copy()))
    bn.running_var.copy_(torch.from_numpy(var.copy()))

def bn_params(prefix):
    return (
        get_np(f'{prefix}/BatchNorm/gamma'),
        get_np(f'{prefix}/BatchNorm/beta'),
        get_np(f'{prefix}/BatchNorm/moving_mean'),
        get_np(f'{prefix}/BatchNorm/moving_variance'),
    )

print("Loading weights...")

# Initial conv
load_conv_weight(model.conv0, get_np('MobilenetV1/Conv2d_0/weights'))
load_bn(model.conv0_bn, *bn_params('MobilenetV1/Conv2d_0'))

# DS blocks 1–13
for i, block in enumerate(model.blocks, 1):
    dw_pfx = f'MobilenetV1/Conv2d_{i}_depthwise'
    pw_pfx = f'MobilenetV1/Conv2d_{i}_pointwise'
    load_dw_weight(block.dw, get_np(f'{dw_pfx}/depthwise_weights'))
    load_bn(block.dw_bn, *bn_params(dw_pfx))
    load_conv_weight(block.pw, get_np(f'{pw_pfx}/weights'))
    load_bn(block.pw_bn, *bn_params(pw_pfx))

# Original logits conv (1024 → 1001)
load_conv_weight(model.logits_conv, get_np('MobilenetV1/Logits/Conv2d_1c_1x1/weights'))
model.logits_conv.bias = nn.Parameter(
    torch.from_numpy(get_np('MobilenetV1/Logits/Conv2d_1c_1x1/biases').copy())
)

# Retrained FC (1001 → num_classes): TF (1001, N) needs transpose for PyTorch (N, 1001)
model.fc.weight = nn.Parameter(
    torch.from_numpy(get_np('final_training_ops/weights/final_weights').T.copy())
)
model.fc.bias = nn.Parameter(
    torch.from_numpy(get_np('final_training_ops/biases/final_biases').copy())
)

print("Weights loaded. Running eval sanity check...")
model.eval()
with torch.no_grad():
    dummy = torch.zeros(1, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}, sum: {out.sum().item():.6f} (should be ~1.0)")

# ── Step 4: Export to ONNX ───────────────────────────────────────────────────

print("Exporting to ONNX...")
dummy_input = torch.zeros(1, 3, 224, 224)
# Use the legacy TorchScript-based exporter (avoids onnxscript dependency)
torch.onnx.export(
    model,
    dummy_input,
    'mushrooms.onnx',
    opset_version=13,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
    dynamo=False,
)
print("Saved mushrooms.onnx")

# ── Step 5: Convert ONNX → TFLite via onnx2tf ───────────────────────────────

print("Converting ONNX to TFLite with onnx2tf...")
try:
    import onnx2tf
    onnx2tf.convert(
        input_onnx_file_path='mushrooms.onnx',
        output_folder_path='mushrooms_tflite_tmp',
        output_tfite_file_path='mushrooms.tflite',
        non_verbose=True,
    )
    print("SUCCESS: mushrooms.tflite written")
except Exception as e:
    print(f"onnx2tf failed: {e}")
    print("Trying onnx2tf CLI fallback...")
    import subprocess
    result = subprocess.run(
        [sys.executable, '-m', 'onnx2tf',
         '-i', 'mushrooms.onnx',
         '-o', 'mushrooms_tflite_tmp'],
        capture_output=True, text=True
    )
    print(result.stdout)
    print(result.stderr)
