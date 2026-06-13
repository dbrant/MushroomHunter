import struct

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
    elif wt == 5: v = struct.unpack('<I', data[pos:pos+4])[0]; pos += 4
    elif wt == 1: v = struct.unpack('<Q', data[pos:pos+8])[0]; pos += 8
    else: return None, None, None, pos
    return fn, wt, v, pos

def parse_nodedef(data):
    n = {'name': '', 'op': '', 'inputs': [], 'attrs': {}}
    pos = 0
    while pos < len(data):
        fn, wt, v, pos = read_field(data, pos)
        if fn is None: break
        if fn == 1 and wt == 2: n['name'] = v.decode('utf-8', 'replace')
        elif fn == 2 and wt == 2: n['op'] = v.decode('utf-8', 'replace')
        elif fn == 3 and wt == 2: n['inputs'].append(v.decode('utf-8', 'replace'))
        elif fn == 5 and wt == 2:
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

def trace_back(name, depth=0, visited=None):
    if visited is None: visited = set()
    if name in visited or depth > 15: return
    visited.add(name)
    base = name.split(':')[0]
    nd = nodes.get(base)
    if nd:
        print('  ' * depth + nd['op'] + ': ' + nd['name'] + ' <- ' + str(nd['inputs']))
        if nd['op'] not in ('Const', 'Placeholder', 'PlaceholderWithDefault'):
            for inp in nd['inputs']:
                trace_back(inp, depth + 1, visited)

trace_back('MobilenetV1/Predictions/Reshape')
