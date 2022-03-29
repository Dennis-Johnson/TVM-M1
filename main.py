import tvm
import numpy as np
import tvm.relay.testing
from PIL import Image
from tvm import relay, auto_scheduler
from utils import softmax, get_network
from tvm.contrib import graph_executor

#################################################################
# Define a Network and Compilation Target
# ----------------
network    = "resnet-18"
batch_size = 1
layout     = "NCHW"
dtype      = "float32"
shape_dict = {"data": (batch_size, 3, 224, 224)}

target     =  tvm.target.Target("metal", host="llvm -mcpu=apple-m1 -mtriple=arm64-apple-darwin21.3.0")
log_file   = "./logs/%s-%s-B%d-%s.json" % (network, layout, batch_size, target.kind.name)

# Read and resize input image to 224x224.
img_path = "./assets/kitten.jpeg"
resized_image = Image.open(img_path).resize((224, 224))
img_data = np.asarray(resized_image).astype("float32")

# Input image is in HWC layout while ONNX expects CHW input.
img_data = np.transpose(img_data, (2, 0, 1))

# Normalize according to the ImageNet input specification.
imagenet_mean   = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
imagenet_stddev = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
norm_img_data   = (img_data / 255 - imagenet_mean) / imagenet_stddev

# Add the batch dimension, since we expect NCHW.
img_data = np.expand_dims(norm_img_data, axis=0)

#################################################################
# Extract Search Tasks from the Network
# --------------------
print("Extract tasks...")
mod, params, input_shape, output_shape = get_network(network, batch_size, layout, dtype=dtype)
tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

for idx, task in enumerate(tasks):
    print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
    print(task.compute_dag)

#################################################################
# Begin Tuning with XGBoost.
# --------------------
def run_tuning():
    print("Begin tuning...")
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10)

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials = 900 * len(tasks),
        runner=measure_ctx.runner,
        early_stopping = 100,
        verbose = 1,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)])

    tuner.tune(tune_option)

# run_tuning()

######################################################################
# Compile and Evaluate
# --------------------
print("Compiling model...")
with auto_scheduler.ApplyHistoryBest(log_file):
    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        lib = relay.build(mod, target=target, params=params)

# Create graph executor
dev    = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))

module.set_input("data", img_data)
module.run()
tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()

# Benchmark inference performance
print("Benchmarking inference time ...")
print(module.benchmark(dev, repeat=10, min_repeat_ms=500))