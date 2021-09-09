from __future__ import print_function
import sys
import caffe
import onnx
import numpy as np
from caffe.proto import caffe_pb2
caffe.set_mode_cpu()
from onnx2caffe._transformers import ConvAddFuser,ConstantsToInitializers
from onnx2caffe._graph import Graph

import onnx2caffe._operators as cvt
import onnx2caffe._weightloader as wlr
from onnx2caffe._error_utils import ErrorHandling
from collections import OrderedDict
from onnx import shape_inference
import importlib

transformers = [
    ConstantsToInitializers(),
    ConvAddFuser(),
]

def isPoolingSizeUnmatched(shape_dict, node):
    do_slice_h = False
    do_slice_w = False

    intput_name = node.inputs[0]
    output_name = node.outputs[0]
    input_shape = shape_dict[intput_name]
    output_shape = shape_dict[output_name]
    pad_h = node.attrs["pads"][2]
    pad_w = node.attrs["pads"][3]
    kernel_h = node.attrs["kernel_shape"][0]
    kernel_w = node.attrs["kernel_shape"][1]
    stride_h = node.attrs["strides"][0]
    stride_w = node.attrs["strides"][1]
    pooling_h = int(np.ceil((input_shape[2] + 2 * pad_h - kernel_h) / stride_h + 1))
    pooling_w = int(np.ceil((input_shape[3] + 2 * pad_w - kernel_w) / stride_w + 1))
    if pooling_h != output_shape[2]:
        do_slice_h = True
    if pooling_w != output_shape[3]:
        do_slice_w = True

    return do_slice_h, do_slice_w

def convertToCaffe(graph, prototxt_save_path, caffe_model_save_path):

    exist_edges = []
    layers = []
    exist_nodes = []
    err = ErrorHandling()
    to_be_replaced = {}

    for i in graph.inputs:
        edge_name = i[0]
        input_layer = cvt.make_input(i)
        layers.append(input_layer)
        exist_edges.append(i[0])
        graph.channel_dims[edge_name] = graph.shape_dict[edge_name][1]


    for id, node in enumerate(graph.nodes):
        node_name = node.name
        op_type = node.op_type
        inputs = node.inputs
        inputs_tensor = node.input_tensors
        input_non_exist_flag = False

        for inp in inputs:
            if inp not in exist_edges and inp not in inputs_tensor:
                input_non_exist_flag = True
                break
        if input_non_exist_flag:
            continue

        if op_type not in cvt._ONNX_NODE_REGISTRY:
            err.unsupported_op(node)
            continue
        converter_fn = cvt._ONNX_NODE_REGISTRY[op_type]
        layer = converter_fn(node,graph,err)

        # replace the input which connect the origin polling layer
        if len(to_be_replaced) > 0:
            if type(layer)==tuple:
                for l in layer:
                    for i, input_layer in enumerate(l.inputs):
                        if input_layer in to_be_replaced:
                            l.inputs[i] = to_be_replaced[input_layer]
            else:
                for i, input_layer in enumerate(layer.inputs):
                    if input_layer in to_be_replaced:
                        layer.inputs[i]=to_be_replaced[input_layer]

        if type(layer)==tuple:
            for l in layer:
                layers.append(l)
        else:
            layers.append(layer)

        # append slice layer after pooling if shapesize not matched
        if op_type == "MaxPool" or op_type == "AveragePool":
            do_slice_h, do_slice_w = isPoolingSizeUnmatched(graph.shape_dict, node)
            if do_slice_h:
                layer = cvt._append_slice_after_pooling(node.outputs[0], 2, graph.shape_dict[node.outputs[0]][2])
                layers.append(layer)
            if do_slice_w:
                if do_slice_h:
                    layer = cvt._append_slice_after_pooling(layers[-1].outputs[0], 3, graph.shape_dict[node.outputs[0]][3])
                else:
                    layer = cvt._append_slice_after_pooling(node.outputs[0], 3, graph.shape_dict[node.outputs[0]][3])
                layers.append(layer)
            if do_slice_h or do_slice_w:
                to_be_replaced[node.outputs[0]] = layers[-1].outputs[0]

        outs = node.outputs
        for out in outs:
            exist_edges.append(out)

    net = caffe_pb2.NetParameter()
    for id,layer in enumerate(layers):
        layers[id] = layer._to_proto()
    net.layer.extend(layers)

    with open(prototxt_save_path, 'w') as f:
        print(net,file=f)

    caffe.set_mode_cpu()
    deploy = prototxt_save_path
    net = caffe.Net(deploy,
                    caffe.TEST)

    for id, node in enumerate(graph.nodes):
        node_name = node.name
        op_type = node.op_type
        inputs = node.inputs
        inputs_tensor = node.input_tensors
        input_non_exist_flag = False
        if op_type not in wlr._ONNX_NODE_REGISTRY:
            err.unsupported_op(node)
            continue
        converter_fn = wlr._ONNX_NODE_REGISTRY[op_type]
        converter_fn(net, node, graph, err)

    net.save(caffe_model_save_path)
    return net

def getGraph(onnx_path):
    model = onnx.load(onnx_path)
    model = shape_inference.infer_shapes(model)
    model_graph = model.graph
    graph = Graph.from_onnx(model_graph)
    graph = graph.transformed(transformers)
    graph.channel_dims = {}

    return graph

if __name__ == "__main__":
    onnx_path = sys.argv[1]
    prototxt_path = sys.argv[2]
    caffemodel_path = sys.argv[3]
    graph = getGraph(onnx_path)
    convertToCaffe(graph, prototxt_path, caffemodel_path)

