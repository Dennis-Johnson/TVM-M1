#!/opt/homebrew/opt/python@3.8/bin/python3.8
import onnx
import timeit
from PIL import Image
import numpy as np
import tvm
from tvm import autotvm, runtime
from tvm.contrib import graph_executor
from tvm.autotvm.tuner import XGBTuner
import tvm.auto_scheduler as auto_scheduler
import tvm.relay as relay

from scipy.special import softmax

# Load a ResNet-18 for classification.
# Got this from https://github.com/onnx/models/blob/main/vision/classification/resnet/model/resne18-v2-7.onnx
model_path = "./assets/resnet18-v2-7.onnx"
onnx_model = onnx.load(model_path)

# Read and resize input image to 224x224.
img_path = "./assets/kitten.jpeg"
resized_image = Image.open(img_path).resize((224, 224))
img_data = np.asarray(resized_image).astype("float32")

# Input image is in HWC layout while ONNX expects CHW input.
img_data = np.transpose(img_data, (2, 0, 1))

# Normalize according to the ImageNet input specification.
imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
imagenet_stddev = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
norm_img_data = (img_data / 255 - imagenet_mean) / imagenet_stddev

# Add the batch dimension, as we are expecting 4-dimensional input: NCHW.
img_data = np.expand_dims(norm_img_data, axis=0)

### Compile the model with Relay ###
input_name = "data"
shape_dict = {input_name: img_data.shape}
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
target = tvm.target.Target("metal", host="llvm -mcpu=apple-latest -mtriple=arm64-apple-macos")

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))

### Execute on the TVM Runtime ###
dtype = "float32"
module.set_input(input_name, img_data)
module.run()
output_shape = (1, 1000)
tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()

### Collect unoptimised performance data if you want ###
benchmark = False
if benchmark:
    timing_number = 10
    timing_repeat = 10
    unoptimized = (
        np.array(timeit.Timer(lambda: module.run()).repeat(repeat=timing_repeat, number=timing_number))
        / timing_number
    )
    unoptimized = {
        "mean": np.mean(unoptimized),
        "median": np.median(unoptimized),
        "std": np.std(unoptimized),
    }

    print(unoptimized)

### Results ###
# Load a list of class labels
labels_path = "assets/synset.txt"
with open(labels_path, "r") as f:
    labels = [l.rstrip() for l in f]

# Open the output and read the output tensor
scores = softmax(tvm_output)
scores = np.squeeze(scores)
ranks = np.argsort(scores)[::-1]
for rank in ranks[0:5]:
    print("class='%s' with probability=%f" % (labels[rank], scores[rank]))

# Tune the ResNet model with autoTVM
number = 10
repeat = 1
min_repeat_ms = 1   
timeout       = 100  # in seconds

# create a TVM runner
runner = autotvm.LocalRunner(
    number=number,
    repeat=repeat,
    timeout=timeout,
    min_repeat_ms=min_repeat_ms,
    enable_cpu_cache_flush=True,
)

# Tuning options for the XGBoost algorithm that guides the search.
tuning_option = {
    "tuner": "xgb",
    "trials": 1000,
    "early_stopping": 100,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(build_func="default"), runner=runner
    ),
    "tuning_records": "logs/resnet-50-v2-autotuning.json",
}

# begin by extracting the tasks from the onnx model
tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)

# Tune the extracted tasks sequentially.
for i, task in enumerate(tasks):
    prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
    tuner_obj = XGBTuner(task, loss_type="rank")
    tuner_obj.tune(
        n_trial=min(tuning_option["trials"], len(task.config_space)),
        early_stopping=tuning_option["early_stopping"],
        measure_option=tuning_option["measure_option"],
        callbacks=[
            autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
            autotvm.callback.log_to_file(tuning_option["tuning_records"]),
        ],
    )