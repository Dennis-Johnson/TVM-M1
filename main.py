import tvm
import numpy as np
import tvm.testing
from tvm import relay, te
from tvm.relay import testing
from tvm.contrib import graph_executor

# Define a ResNet-18 with relay python frontend
batch_size = 1
num_class  = 1000
image_shape = (3, 224, 224)
data_shape  = (batch_size,) + image_shape
out_shape   = (batch_size, num_class)

mod, params = relay.testing.resnet.get_workload(
    num_layers=18, batch_size=batch_size, image_shape=image_shape
)

print(mod.astext(show_meta_data=False))

# Compile the model using the Relay/TVM pipeline for M1 aarch64
opt_level   = 3
target = tvm.target.Target(target="metal", host="llvm -mcpu=apple-latest -mtriple=arm64-apple-macos")

with tvm.transform.PassContext(opt_level=opt_level):
    lib = relay.build(mod, target=target, params=params)

# M1 GPU go brr"
dev = tvm.metal()
data = np.random.uniform(-1, 1, size=data_shape).astype("float32")

# Create module
module = graph_executor.GraphModule(lib["default"](dev))

# set input and parameters
module.set_input("data", data)
module.run()

out = module.get_output(0, tvm.nd.empty(out_shape)).numpy()
print(out.flatten()[0:10])