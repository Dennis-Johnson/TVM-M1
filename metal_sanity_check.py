#!/opt/homebrew/opt/python@3.8/bin/python3.8
import numpy as np
import math

import os
import tvm
from tvm import relay, auto_scheduler
from tvm import testing
from tvm.contrib import utils, xcode, coreml_runtime, graph_runtime

target = tvm.target.Target("metal", host="llvm -mcpu=apple-m1 -mtriple=arm64-apple-darwin21.3.0")

def _get_model(shape, dtype, var_names):
    """Return a model and any parameters it may have."""
    a = relay.var(next(var_names), shape=shape, dtype=dtype)
    out = relay.op.reduce.mean(a, 0)
    params = {}
    return out, params


def converter(shape):
    print("Shape: {}".format(shape))
    dtype = "float32"
    inputs = {"data": tvm.nd.array(np.random.uniform(-128, 127, shape).astype(dtype))}

    mod, params = _get_model(shape, dtype, iter(inputs))
    if isinstance(mod, tvm.relay.expr.Call):
        mod = tvm.IRModule.from_expr(mod)
    print('mod: ', mod)

    with tvm.transform.PassContext(opt_level=3):
        graph_module = relay.build(mod['main'], target=target, params=params)

    with auto_scheduler.ApplyHistoryBest("my_mean_model_metal"):
       with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
           graph_module = relay.build(mod['main'], target=target, params=params)
    return graph_module


def run(graph_module):
    ctx = tvm.metal(0)
    m = graph_runtime.graph_executor.GraphModule(graph_module["default"](ctx))
    m.run()


if __name__ == "__main__":
    shape = (2, 1)
    gm = converter(shape)
    run(gm)