#!/usr/bin/env python3
"""Build mushrooms.tflite directly from mushrooms.pb (no TensorFlow required).
Strategy: parse TF frozen graph -> fold BN into conv -> write TFLite flatbuffer.
"""

import struct
import numpy as np
import flatbuffers
import tflite
from tflite import (
    BuiltinOperator as BOT,
    BuiltinOptions as BOp,
    TensorType as TT,
    Padding as PAD,
    ActivationFunctionType as AFT,
)

# ── Parse TF frozen graph ────────────────────────────────────────────────────

def read_varint(data, pos):
    result, shift = 0, 0
    while True:
        b = data[pos]; pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80): break
        shift += 7
    return result, pos

def read_field(data, pos):
    if pos >= len(data): return None, None, None, pos
    tag, pos = read_varint(data, pos)
    fn, wt = tag >> 3, tag & 0x7
    if wt == 0:   v, pos = read_varint(data, pos)
    elif wt == 2:
        l, pos = read_varint(data, pos); v = data[pos:pos+l]; pos += l
    elif wt == 5: v = data[pos:pos+4]; pos += 4
    elif wt == 1: v = data[pos:pos+8]; pos += 8
    else: return None, None, None, pos
    return fn, wt, v, pos

def parse_shape(data):
    dims = []
    pos = 0
    while pos < len(data):
        fn, wt, v, pos = read_field(data, pos)
        if fn is None: break
        if fn == 2 and wt == 2:
            dpos = 0
            while dpos < len(v):
                dfn, _, dv, dpos = read_field(v, dpos)
                if dfn is None: break
                if dfn == 1: dims.append(dv)
    return dims

def parse_tensor(data):
    dtype = None; shape = []; content = None; fvals = []
    pos = 0
    while pos < len(data):
        fn, wt, v, pos = read_field(data, pos)
        if fn is None: break
        if fn == 1 and wt == 0: dtype = v
        elif fn == 2 and wt == 2: shape = parse_shape(v)
        elif fn == 4 and wt == 2: content = v
        elif fn == 5 and wt == 5: fvals.append(struct.unpack('<f', v)[0])
    if dtype != 1: return None
    if content:
        arr = np.frombuffer(bytes(content), dtype=np.float32).copy()
    elif fvals:
        arr = np.array(fvals, dtype=np.float32)
    else:
        arr = np.array([], dtype=np.float32)
    if shape:
        try: arr = arr.reshape(shape)
        except: pass
    elif content and len(content) == 4:
        arr = np.frombuffer(bytes(content), dtype=np.float32).copy()
    return arr

def tensor_from_attr(data):
    pos = 0
    while pos < len(data):
        fn, wt, v, pos = read_field(data, pos)
        if fn is None: break
        if fn == 8 and wt == 2: return parse_tensor(v)
    return None

def parse_node(data):
    n = {'name': '', 'op': '', 'attrs': {}}
    pos = 0
    while pos < len(data):
        fn, wt, v, pos = read_field(data, pos)
        if fn is None: break
        if fn == 1 and wt == 2: n['name'] = v.decode('utf-8', 'replace')
        elif fn == 2 and wt == 2: n['op'] = v.decode('utf-8', 'replace')
        elif fn == 5 and wt == 2:
            key = None; aval = None; apos = 0
            while apos < len(v):
                afn, awt, av, apos = read_field(v, apos)
                if afn is None: break
                if afn == 1 and awt == 2: key = av.decode('utf-8', 'replace')
                elif afn == 2 and awt == 2: aval = av
            if key: n['attrs'][key] = aval
    return n

print("Parsing frozen graph...")
with open('mushrooms.pb', 'rb') as f:
    pb = f.read()

W = {}
pos = 0
while pos < len(pb):
    fn, wt, v, pos = read_field(pb, pos)
    if fn is None: break
    if fn == 1 and wt == 2:
        nd = parse_node(v)
        if nd['op'] == 'Const' and 'value' in nd['attrs']:
            arr = tensor_from_attr(nd['attrs']['value'])
            if arr is not None: W[nd['name']] = arr

print(f"Loaded {len(W)} weight tensors")

try:
    eps = float(W['MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/batchnorm/add/y'].flat[0])
except: eps = 0.001
print(f"BN epsilon: {eps}")

num_classes = int(W['final_training_ops/biases/final_biases'].shape[0])
print(f"Classes: {num_classes}")

# ── BN fusion ────────────────────────────────────────────────────────────────

def fuse_conv(tf_w, gamma, beta, mean, var):
    """TF conv [H,W,in,out] + BN -> TFLite [out,H,W,in], bias"""
    scale = gamma / np.sqrt(var + eps)
    w = tf_w * scale[np.newaxis, np.newaxis, np.newaxis, :]
    b = beta - gamma * mean / np.sqrt(var + eps)
    return w.transpose(3, 0, 1, 2).astype(np.float32), b.astype(np.float32)

def fuse_dw(tf_w, gamma, beta, mean, var):
    """TF dw [H,W,C,1] + BN -> TFLite [1,H,W,C], bias"""
    scale = gamma / np.sqrt(var + eps)
    w = tf_w * scale[np.newaxis, np.newaxis, :, np.newaxis]
    b = beta - gamma * mean / np.sqrt(var + eps)
    return w.transpose(3, 0, 1, 2).astype(np.float32), b.astype(np.float32)

def bn(prefix):
    return (W[f'{prefix}/BatchNorm/gamma'], W[f'{prefix}/BatchNorm/beta'],
            W[f'{prefix}/BatchNorm/moving_mean'], W[f'{prefix}/BatchNorm/moving_variance'])

print("Fusing BN into weights...")
cv0_w, cv0_b = fuse_conv(W['MobilenetV1/Conv2d_0/weights'], *bn('MobilenetV1/Conv2d_0'))

dw_ws, dw_bs, pw_ws, pw_bs = [], [], [], []
for i in range(1, 14):
    dw_w, dw_b = fuse_dw(W[f'MobilenetV1/Conv2d_{i}_depthwise/depthwise_weights'],
                          *bn(f'MobilenetV1/Conv2d_{i}_depthwise'))
    pw_w, pw_b = fuse_conv(W[f'MobilenetV1/Conv2d_{i}_pointwise/weights'],
                            *bn(f'MobilenetV1/Conv2d_{i}_pointwise'))
    dw_ws.append(dw_w); dw_bs.append(dw_b)
    pw_ws.append(pw_w); pw_bs.append(pw_b)

# Fuse logits (1024->1001) with FC (1001->num_classes) into single FC (1024->num_classes)
lw = W['MobilenetV1/Logits/Conv2d_1c_1x1/weights'].squeeze()  # (1024,1001)
lb = W['MobilenetV1/Logits/Conv2d_1c_1x1/biases']             # (1001,)
fw = W['final_training_ops/weights/final_weights']              # (1001,N)
fb = W['final_training_ops/biases/final_biases']               # (N,)
fc_w = (lw @ fw).T.astype(np.float32)    # (N, 1024)
fc_b = (lb @ fw + fb).astype(np.float32) # (N,)

print(f"Weights ready. FC: {fc_w.shape}")

# ── Buffer and tensor plan ────────────────────────────────────────────────────
# Each weight/bias gets its own buffer with raw float32 bytes.
# All activation tensors share buffer 0 (empty).

weight_buffers = []  # list of (buffer_idx, numpy_array)
_buf = 0

def alloc_buf(arr):
    global _buf
    _buf += 1
    weight_buffers.append((_buf, arr))
    return _buf

EMPTY = 0  # buffer 0 = empty (activation tensors)

b_cv0_w = alloc_buf(cv0_w);   b_cv0_b = alloc_buf(cv0_b)
b_dw_ws = [alloc_buf(w) for w in dw_ws]
b_dw_bs = [alloc_buf(bv) for bv in dw_bs]
b_pw_ws = [alloc_buf(w) for w in pw_ws]
b_pw_bs = [alloc_buf(bv) for bv in pw_bs]
b_fc_w  = alloc_buf(fc_w);   b_fc_b  = alloc_buf(fc_b)
b_rshp  = alloc_buf(np.array([1, 1024], dtype=np.int32))

NUM_BUFFERS = _buf + 1  # +1 for buffer 0

# ── Tensor index table ────────────────────────────────────────────────────────
# MobileNet V1 DW block strides for blocks 1-13
BLK_STRIDES = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1]

# Spatial sizes after each layer
sp = 112  # after conv0 (stride=2 on 224)
SPATIAL = [sp]
for st in BLK_STRIDES:
    sp //= st
    SPATIAL.append(sp)  # after block_i_dw
    SPATIAL.append(sp)  # after block_i_pw

# Pointwise output channels per block 1-13
BLK_PW_CH = [64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024]

# Tensor metadata list: (name, shape, ttype, buf_idx)
T_META = []

def add_t(name, shape, ttype=TT.FLOAT32, buf=EMPTY):
    T_META.append((name, list(shape), ttype, buf))
    return len(T_META) - 1

T_IN   = add_t("input",      [1, 224, 224, 3])
T_CV0  = add_t("cv0_out",    [1, SPATIAL[0], SPATIAL[0], 32])

T_BLKS = []
dw_in_ch = 32
for i in range(13):
    sp_dw = SPATIAL[1 + i*2]
    sp_pw = SPATIAL[2 + i*2]
    pw_ch = BLK_PW_CH[i]
    dw_idx = add_t(f"b{i+1}_dw", [1, sp_dw, sp_dw, dw_in_ch])
    pw_idx = add_t(f"b{i+1}_pw", [1, sp_pw, sp_pw, pw_ch])
    T_BLKS.append((dw_idx, pw_idx))
    dw_in_ch = pw_ch

T_AVG    = add_t("avgpool",     [1, 1, 1, 1024])
T_RSHP_S = add_t("rshp_shape", [2], ttype=TT.INT32, buf=b_rshp)
T_RSHP   = add_t("reshape",    [1, 1024])
T_FC     = add_t("fc_out",     [1, num_classes])
T_OUT    = add_t("output",     [1, num_classes])

def add_w(name, arr, buf): return add_t(name, arr.shape, TT.FLOAT32, buf)

T_CV0_W = add_w("cv0_w", cv0_w, b_cv0_w)
T_CV0_B = add_w("cv0_b", cv0_b, b_cv0_b)
T_DW_W  = [add_w(f"dw{i+1}_w", dw_ws[i], b_dw_ws[i]) for i in range(13)]
T_DW_B  = [add_w(f"dw{i+1}_b", dw_bs[i], b_dw_bs[i]) for i in range(13)]
T_PW_W  = [add_w(f"pw{i+1}_w", pw_ws[i], b_pw_ws[i]) for i in range(13)]
T_PW_B  = [add_w(f"pw{i+1}_b", pw_bs[i], b_pw_bs[i]) for i in range(13)]
T_FC_W  = add_w("fc_w", fc_w, b_fc_w)
T_FC_B  = add_w("fc_b", fc_b, b_fc_b)

NUM_TENSORS = len(T_META)
print(f"Tensors: {NUM_TENSORS}, Buffers: {NUM_BUFFERS}")

# ── Operator plan ─────────────────────────────────────────────────────────────
OPS = []

OPS.append(('CONV', [T_IN, T_CV0_W, T_CV0_B], [T_CV0],
            dict(stride_h=2, stride_w=2, padding=PAD.SAME, activation=AFT.RELU6, dil_h=1, dil_w=1)))

prev = T_CV0
for i in range(13):
    dw_t, pw_t = T_BLKS[i]
    st = BLK_STRIDES[i]
    OPS.append(('DW_CONV', [prev, T_DW_W[i], T_DW_B[i]], [dw_t],
                dict(stride_h=st, stride_w=st, padding=PAD.SAME,
                     activation=AFT.RELU6, depth_mult=1, dil_h=1, dil_w=1)))
    OPS.append(('CONV', [dw_t, T_PW_W[i], T_PW_B[i]], [pw_t],
                dict(stride_h=1, stride_w=1, padding=PAD.SAME, activation=AFT.RELU6, dil_h=1, dil_w=1)))
    prev = pw_t

OPS.append(('AVG_POOL', [prev], [T_AVG],
            dict(stride_h=1, stride_w=1, filter_h=7, filter_w=7, padding=PAD.VALID, activation=AFT.NONE)))
OPS.append(('RESHAPE', [T_AVG, T_RSHP_S], [T_RSHP], {}))
OPS.append(('FC', [T_RSHP, T_FC_W, T_FC_B], [T_FC],
            dict(activation=AFT.NONE, weights_format=0, keep_num_dims=False)))
OPS.append(('SOFTMAX', [T_FC], [T_OUT], dict(beta=1.0)))

# Opcode list (in order of first use) -> BuiltinOperator value + BuiltinOptions union type
OP_INFO = {
    'CONV':     (BOT.CONV_2D,            BOp.Conv2DOptions),
    'DW_CONV':  (BOT.DEPTHWISE_CONV_2D,  BOp.DepthwiseConv2DOptions),
    'AVG_POOL': (BOT.AVERAGE_POOL_2D,    BOp.Pool2DOptions),
    'RESHAPE':  (BOT.RESHAPE,             BOp.ReshapeOptions),
    'FC':       (BOT.FULLY_CONNECTED,     BOp.FullyConnectedOptions),
    'SOFTMAX':  (BOT.SOFTMAX,             BOp.SoftmaxOptions),
}
seen_kinds = []
for op_kind, _, _, _ in OPS:
    if op_kind not in seen_kinds: seen_kinds.append(op_kind)
OPCODE_INDEX = {kind: i for i, kind in enumerate(seen_kinds)}

# ── Flatbuffers construction ──────────────────────────────────────────────────

CAPACITY = 30 * 1024 * 1024
bldr = flatbuffers.Builder(CAPACITY)

def int32_vec(values):
    n = len(values)
    bldr.StartVector(4, n, 4)
    for v in reversed(values):
        bldr.PrependInt32(v)
    return bldr.EndVector()

def table_vec(offsets):
    n = len(offsets)
    bldr.StartVector(4, n, 4)
    for off in reversed(offsets):
        bldr.PrependUOffsetTRelative(off)
    return bldr.EndVector()

# 1. Buffers
print("Building buffers...")
buf_offsets = []

tflite.BufferStart(bldr)
buf_offsets.append(tflite.BufferEnd(bldr))  # buffer 0: empty

for _idx, arr in sorted(weight_buffers, key=lambda x: x[0]):
    raw = arr.tobytes()
    n = len(raw)
    tflite.BufferStartDataVector(bldr, n)
    for byte in reversed(raw):
        bldr.PrependByte(byte)
    data_vec = bldr.EndVector()
    tflite.BufferStart(bldr)
    tflite.BufferAddData(bldr, data_vec)
    buf_offsets.append(tflite.BufferEnd(bldr))

assert len(buf_offsets) == NUM_BUFFERS, f"{len(buf_offsets)} != {NUM_BUFFERS}"

# 2. Tensors
print("Building tensors...")
tensor_offsets = []
for (name, shape, ttype, buf_idx) in T_META:
    name_str  = bldr.CreateString(name)
    shape_vec = int32_vec(shape)
    tflite.TensorStart(bldr)
    tflite.TensorAddName(bldr, name_str)
    tflite.TensorAddShape(bldr, shape_vec)
    tflite.TensorAddType(bldr, ttype)
    tflite.TensorAddBuffer(bldr, buf_idx)
    tflite.TensorAddHasRank(bldr, True)
    tensor_offsets.append(tflite.TensorEnd(bldr))

# 3. Operators
print("Building operators...")
op_offsets = []
for (kind, inp_ts, out_ts, opts) in OPS:
    if kind == 'CONV':
        tflite.Conv2DOptionsStart(bldr)
        tflite.Conv2DOptionsAddPadding(bldr, opts['padding'])
        tflite.Conv2DOptionsAddStrideH(bldr, opts['stride_h'])
        tflite.Conv2DOptionsAddStrideW(bldr, opts['stride_w'])
        tflite.Conv2DOptionsAddFusedActivationFunction(bldr, opts['activation'])
        tflite.Conv2DOptionsAddDilationHFactor(bldr, opts['dil_h'])
        tflite.Conv2DOptionsAddDilationWFactor(bldr, opts['dil_w'])
        opts_off = tflite.Conv2DOptionsEnd(bldr)

    elif kind == 'DW_CONV':
        tflite.DepthwiseConv2DOptionsStart(bldr)
        tflite.DepthwiseConv2DOptionsAddPadding(bldr, opts['padding'])
        tflite.DepthwiseConv2DOptionsAddStrideH(bldr, opts['stride_h'])
        tflite.DepthwiseConv2DOptionsAddStrideW(bldr, opts['stride_w'])
        tflite.DepthwiseConv2DOptionsAddDepthMultiplier(bldr, opts['depth_mult'])
        tflite.DepthwiseConv2DOptionsAddFusedActivationFunction(bldr, opts['activation'])
        tflite.DepthwiseConv2DOptionsAddDilationHFactor(bldr, opts['dil_h'])
        tflite.DepthwiseConv2DOptionsAddDilationWFactor(bldr, opts['dil_w'])
        opts_off = tflite.DepthwiseConv2DOptionsEnd(bldr)

    elif kind == 'AVG_POOL':
        tflite.Pool2DOptionsStart(bldr)
        tflite.Pool2DOptionsAddPadding(bldr, opts['padding'])
        tflite.Pool2DOptionsAddStrideH(bldr, opts['stride_h'])
        tflite.Pool2DOptionsAddStrideW(bldr, opts['stride_w'])
        tflite.Pool2DOptionsAddFilterHeight(bldr, opts['filter_h'])
        tflite.Pool2DOptionsAddFilterWidth(bldr, opts['filter_w'])
        tflite.Pool2DOptionsAddFusedActivationFunction(bldr, opts['activation'])
        opts_off = tflite.Pool2DOptionsEnd(bldr)

    elif kind == 'RESHAPE':
        shape_r = int32_vec([1, 1024])
        tflite.ReshapeOptionsStart(bldr)
        tflite.ReshapeOptionsAddNewShape(bldr, shape_r)
        opts_off = tflite.ReshapeOptionsEnd(bldr)

    elif kind == 'FC':
        tflite.FullyConnectedOptionsStart(bldr)
        tflite.FullyConnectedOptionsAddFusedActivationFunction(bldr, opts['activation'])
        tflite.FullyConnectedOptionsAddWeightsFormat(bldr, opts['weights_format'])
        tflite.FullyConnectedOptionsAddKeepNumDims(bldr, opts['keep_num_dims'])
        opts_off = tflite.FullyConnectedOptionsEnd(bldr)

    elif kind == 'SOFTMAX':
        tflite.SoftmaxOptionsStart(bldr)
        tflite.SoftmaxOptionsAddBeta(bldr, opts['beta'])
        opts_off = tflite.SoftmaxOptionsEnd(bldr)

    inp_v = int32_vec(inp_ts)
    out_v = int32_vec(out_ts)
    tflite.OperatorStart(bldr)
    tflite.OperatorAddOpcodeIndex(bldr, OPCODE_INDEX[kind])
    tflite.OperatorAddInputs(bldr, inp_v)
    tflite.OperatorAddOutputs(bldr, out_v)
    tflite.OperatorAddBuiltinOptionsType(bldr, OP_INFO[kind][1])
    tflite.OperatorAddBuiltinOptions(bldr, opts_off)
    op_offsets.append(tflite.OperatorEnd(bldr))

# 4. OperatorCodes
opcode_offsets = []
for kind in seen_kinds:
    tflite.OperatorCodeStart(bldr)
    tflite.OperatorCodeAddBuiltinCode(bldr, OP_INFO[kind][0])
    opcode_offsets.append(tflite.OperatorCodeEnd(bldr))

# 5. SubGraph
print("Assembling subgraph and model...")
tensors_vec   = table_vec(tensor_offsets)
operators_vec = table_vec(op_offsets)
inputs_idx    = int32_vec([T_IN])
outputs_idx   = int32_vec([T_OUT])
sg_name       = bldr.CreateString("main")

tflite.SubGraphStart(bldr)
tflite.SubGraphAddName(bldr, sg_name)
tflite.SubGraphAddTensors(bldr, tensors_vec)
tflite.SubGraphAddInputs(bldr, inputs_idx)
tflite.SubGraphAddOutputs(bldr, outputs_idx)
tflite.SubGraphAddOperators(bldr, operators_vec)
sg_off = tflite.SubGraphEnd(bldr)

# 6. Model
opcodes_vec   = table_vec(opcode_offsets)
subgraphs_vec = table_vec([sg_off])
buffers_vec   = table_vec(buf_offsets)
desc_str      = bldr.CreateString("MushroomHunter MobileNetV1")

tflite.ModelStart(bldr)
tflite.ModelAddVersion(bldr, 3)
tflite.ModelAddOperatorCodes(bldr, opcodes_vec)
tflite.ModelAddSubgraphs(bldr, subgraphs_vec)
tflite.ModelAddDescription(bldr, desc_str)
tflite.ModelAddBuffers(bldr, buffers_vec)
model_off = tflite.ModelEnd(bldr)

bldr.Finish(model_off, b"TFL3")
data = bytes(bldr.Output())

with open('mushrooms.tflite', 'wb') as f:
    f.write(data)

print(f"Written mushrooms.tflite ({len(data):,} bytes)")

# ── Sanity check ──────────────────────────────────────────────────────────────
from tflite.Model import Model
m = Model.GetRootAs(bytearray(data), 0)
print(f"TFLite version: {m.Version()}")
sg = m.Subgraphs(0)
print(f"Subgraph: {sg.TensorsLength()} tensors, {sg.OperatorsLength()} ops")
inp = sg.Tensors(T_IN)
out = sg.Tensors(T_OUT)
print(f"Input shape:  {[inp.Shape(j) for j in range(inp.ShapeLength())]}")
print(f"Output shape: {[out.Shape(j) for j in range(out.ShapeLength())]}")
print("SUCCESS")
