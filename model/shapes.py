import struct
import numpy as np

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
    if wt == 0: v, pos = read_varint(data, pos)
    elif wt == 2:
        l, pos = read_varint(data, pos); v = data[pos:pos+l]; pos += l
    elif wt == 5: v = data[pos:pos+4]; pos += 4
    elif wt == 1: v = data[pos:pos+8]; pos += 8
    else: return None, None, None, pos
    return fn, wt, v, pos

def parse_tensor_shape(data):
    dims = []
    pos = 0
    while pos < len(data):
        fn, wt, v, pos = read_field(data, pos)
        if fn is None: break
        if fn == 2 and wt == 2:
            dpos = 0
            while dpos < len(v):
                dfn, dwt, dv, dpos = read_field(v, dpos)
                if dfn is None: break
                if dfn == 1 and dwt == 0: dims.append(dv)
    return dims

def parse_tensor_proto_bytes(data):
    """Parse TensorProto bytes -> numpy array"""
    dtype = None; shape = []; tensor_content = None; float_vals = []
    pos = 0
    while pos < len(data):
        fn, wt, v, pos = read_field(data, pos)
        if fn is None: break
        if fn == 1 and wt == 0: dtype = v         # DataType enum
        elif fn == 2 and wt == 2: shape = parse_tensor_shape(v)  # TensorShapeProto
        elif fn == 4 and wt == 2: tensor_content = v  # tensor_content (raw bytes)
        elif fn == 5 and wt == 5:                   # float_val (repeated float, wire=5)
            float_vals.append(struct.unpack('<f', v)[0])
    if dtype == 1:  # DT_FLOAT
        if tensor_content:
            arr = np.frombuffer(bytes(tensor_content), dtype=np.float32).copy()
        elif float_vals:
            arr = np.array(float_vals, dtype=np.float32)
        else:
            arr = np.array([], dtype=np.float32)
        if shape:
            try: arr = arr.reshape(shape)
            except: pass
        return arr
    return None

def get_tensor_from_attr_value(data):
    """Parse AttrValue bytes -> tensor (field 8 = TensorProto)"""
    pos = 0
    while pos < len(data):
        fn, wt, v, pos = read_field(data, pos)
        if fn is None: break
        if fn == 8 and wt == 2:  # AttrValue.tensor (field 8)
            return parse_tensor_proto_bytes(v)
    return None

def parse_nodedef(data):
    n = {'name': '', 'op': '', 'inputs': [], 'attrs': {}}
    pos = 0
    while pos < len(data):
        fn, wt, v, pos = read_field(data, pos)
        if fn is None: break
        if fn == 1 and wt == 2: n['name'] = v.decode('utf-8', 'replace')
        elif fn == 2 and wt == 2: n['op'] = v.decode('utf-8', 'replace')
        elif fn == 3 and wt == 2: n['inputs'].append(v.decode('utf-8', 'replace'))
        elif fn == 5 and wt == 2:  # attr map entry
            key = None; attr_data = None
            apos = 0
            while apos < len(v):
                afn, awt, av, apos = read_field(v, apos)
                if afn is None: break
                if afn == 1 and awt == 2: key = av.decode('utf-8', 'replace')
                elif afn == 2 and awt == 2: attr_data = av
            if key: n['attrs'][key] = attr_data
    return n

with open('mushrooms.pb', 'rb') as f:
    data = f.read()

nodes = {}
pos = 0
while pos < len(data):
    fn, wt, v, pos = read_field(data, pos)
    if fn is None: break
    if fn == 1 and wt == 2:
        nd = parse_nodedef(v)
        nodes[nd['name']] = nd

keys_of_interest = [
    'MobilenetV1/Conv2d_0/weights',
    'MobilenetV1/Conv2d_1_depthwise/depthwise_weights',
    'MobilenetV1/Conv2d_1_pointwise/weights',
    'MobilenetV1/Logits/Conv2d_1c_1x1/weights',
    'MobilenetV1/Logits/Conv2d_1c_1x1/biases',
    'final_training_ops/weights/final_weights',
    'final_training_ops/biases/final_biases',
    'MobilenetV1/Conv2d_13_pointwise/weights',
    'MobilenetV1/Conv2d_0/BatchNorm/gamma',
]

for key in keys_of_interest:
    nd = nodes.get(key)
    if nd and 'value' in nd['attrs']:
        arr = get_tensor_from_attr_value(nd['attrs']['value'])
        print(f"{key}: shape={arr.shape if arr is not None else None}")
    else:
        print(f"{key}: NOT FOUND (op={nd['op'] if nd else 'missing'})")
