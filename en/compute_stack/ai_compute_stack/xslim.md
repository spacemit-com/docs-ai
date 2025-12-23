sidebar_position: 4

# XSlim

**XSlim** is a **Post-Training Quantization (PTQ)** tool released by **SpacemiT**.  
It integrates chip-optimized quantization strategies and provides a unified quantization interface driven by a JSON configuration file.  
XSlim is fully open-sourced on GitHub: [github-xslim](https://github.com/spacemit-com/xslim).

## QuickStart

- **Install**

```bash
pip install xslim
```

- **Python API**

``` python
import xslim

demo_json = dict()
# Fill demo_json with required configuration fields

demo_json_path = "./demo_json.json"

# Use a Python dictionary
xslim.quantize_onnx_model(demo_json)

# Use a JSON configuration file
xslim.quantize_onnx_model(demo_json_path)

# The API supports passing either a model path or an ONNX ModelProto
# xslim.quantize_onnx_model("resnet18.json", "/home/share/modelzoo/classification/resnet18/resnet18.onnx")

# xslim.quantize_onnx_model(
#    "resnet18.json", "/home/share/modelzoo/classification/resnet18/resnet18.onnx", "resnet18_output.onnx"
# )

# import onnx
# onnx_model = onnx.load("/home/share/modelzoo/classification/resnet18/resnet18.onnx")
# quantized_onnx_model = xslim.quantize_onnx_model("resnet18.json", onnx_model)
```

- **Command Line**
``` bash
python -m xslim --config ./demo_json.json
# Specify input and output model paths
python -m xslim -c ./demo_json.json -i demo.onnx -o demo.q.onnx
# Dynamic quantization (no JSON configuration required)
python -m xslim -i demo.onnx -o demo.q.onnx --dynq
# Convert model to FP16 (no JSON configuration required)
python -m xslim -i demo.onnx -o demo.q.onnx --fp16
# Model slimming only (no quantization, no JSON configuration required)
python -m xslim -i demo.onnx -o demo.q.onnx
```

## Quantization Configuration

- **JSON Configuration Example**

```
{
    "model_parameters": {
        "onnx_model": "",                 // Required: Path to the ONNX model file
        "output_prefix": "",              // Optional: Prefix for the output quantized model file name
        "working_dir": "",                // Optional: Working directory for storing output files and intermediate quantization artifacts
        "skip_onnxsim": false             // Optional: Skip model simplification using onnxsim (default: false)
    },
    "calibration_parameters": {
        "calibration_step": 100,          // Optional: Maximum number of calibration samples to use (default: 100)
        "calibration_device": "cpu",      // Optional: Device for calibration ("cpu" or "cuda"; auto-detected if not specified, falls back to "cpu")
        "calibration_type": "default",    // Optional: Calibration method (default: "default"; options: "kl", "minmax", "percentile", "mse")
        "input_parameters": [             // List of input tensor configurations
            {
                "input_name": "data",     // Optional: Input tensor name (auto-detected from model if omitted)
                "input_shape": [1, 3, 224, 224], // Optional: Input shape (auto-detected from model if omitted)
                "dtype": "float32",       // Optional: Data type of input (auto-detected from model if omitted)
                "file_type": "img",       // Optional: Type of calibration data (default: "img"; options: "img", "npy", "raw")
                "color_format": "bgr",    // Optional: Color format for image data (default: "bgr")
                "mean_value": [103.94, 116.78, 123.68], // Optional: Per-channel mean values for preprocessing (default: none)
                "std_value": [57, 57, 57], // Optional: Per-channel standard deviation values for preprocessing (default: none)
                "preprocess_file": "",    // Optional: Path to custom Python preprocessing script; built-in presets: "PT_IMAGENET" or "IMAGENET"
                "data_list_path": ""      // Required: Path to the text file listing calibration data files
            }
            // Additional input entries can be added here for models with multiple inputs
        ]
    },
    // The following sections are all optional
    "quantization_parameters": {
        "analysis_enable": true,          // Optional: Enable post-quantization accuracy analysis (default: true)
        "precision_level": 0,             // Optional: Global precision control level (higher values may improve accuracy)
        "finetune_level": 1,              // Optional: Fine-tuning level during quantization (default: 1; options: 0, 1, 2, 3)
        "max_percentile": 0.9999,         // Optional: Percentile threshold used in percentile-based calibration
        "custom_setting": [               // Optional: Per-operator custom quantization settings
            {
                "input_names": ["aaa", "bbb"], // Input tensor names (constants may be omitted)
                "output_names": ["ccc"],  // Output tensor name(s)
                "max_percentile": 0.999,  // Custom percentile threshold for this subgraph
                "precision_level": 2,     // Custom precision level
                "calibration_type": "default" // Custom calibration method
            }
            // Additional custom settings can be added here
        ],
        "truncate_var_names": ["/Concat_5_output_0", "/Transpose_6_output_0"] // Optional: List of tensor names to truncate the model at
    }
}
```

### Optional Fields

The following fields are **optional**. If not specified, XSlim will apply the default behavior described below.

| Field Name  | Default Value  | Allowed Values | Notes  |
| -------- | --------------- | ------------ | ---------- |
| `output_prefix`      | Input ONNX filename (suffix removed), output ends with `.q.onnx` | —                                              |                                          |
| `working_dir`        | Directory of the input ONNX model                                | —                                              |                                          |
| `calibration_step`   | 100                                                              | —                                              | Recommended range: **100–1000**                                                                                       |
| `calibration_device` | `cuda`                                                           | `cuda`, `cpu`                                  | Automatically detected if not specified                                                                               |
| `calibration_type`   | `default`                                                        | `default`, `kl`, `minmax`, `percentile`, `mse` | Recommended to start with `default`, then try `percentile` or `minmax` if needed                                      |
| `input_name`         | Read from ONNX model                                             | —                                              |                                                                                                                       |
| `input_shape`        | Read from ONNX model                                             | —                                              | Shape must contain only integers; symbolic batch is supported and defaults to `1`                                     |
| `dtype`              | Read from ONNX model                                             | `float32`, `int8`, `uint8`, `int16`            | Currently, only `float32` is supported                                                                             |
| `file_type`          | `img`                                                            | `img`, `npy`, `raw`                            | For `raw`, only data matching `dtype` is supported (default: `float32`)                                               |
| `preprocess_file`    | `None`                                                           | `PT_IMAGENET`, `IMAGENET`                      | Two ImageNet standard preprocessing methods are built in                                                              |
| `finetune_level`     | 1                                                                | `0`, `1`, `2`, `3`                             | `0`: no aggressive tuning<br>`1`: light static tuning<br>`2+`: block-wise tuning based on quantization loss           |
| `precision_level`    | 0                                                                | `0`, `1`, `2`, `3`, `4`                        | `0`: full INT8 only<br>`1–2`: partial INT8 (suitable for most Transformers)<br>`3`: dynamic quantization<br>`4`: FP16 |
| `max_percentile`     | 0.9999                                                           | —                                              | Truncation range for `percentile` calibration (minimum: `0.99`)                                                       |
| `custom_setting`     | `None`                                                           | —                                              |                                          |
| `truncate_var_names` | `[]`                                                             | —                                              | Graph is split based on tensor names; validity is checked or an error is raised                                       |

**Calibration Data List Format**

Each line in `img_list.txt` represents one calibration data file path.
Both relative paths (relative to `img_list.txt`) and absolute paths are supported.
For multi-input models, ensure that the file lists are aligned in the same order.

```text
QuantZoo/Data/Imagenet/Calib/n01440764/ILSVRC2012_val_00002138.JPEG
QuantZoo/Data/Imagenet/Calib/n01443537/ILSVRC2012_val_00000994.JPEG
QuantZoo/Data/Imagenet/Calib/n01484850/ILSVRC2012_val_00014467.JPEG
QuantZoo/Data/Imagenet/Calib/n01491361/ILSVRC2012_val_00003204.JPEG
QuantZoo/Data/Imagenet/Calib/n01494475/ILSVRC2012_val_00015545.JPEG
QuantZoo/Data/Imagenet/Calib/n01496331/ILSVRC2012_val_00008640.JPEG
```

- **`preprocess_file` Usage**

For a custom preprocessing script (e.g. `custom_preprocess.py`), set
`preprocess_file` to `custom_preprocess.py:preprocess_impl` to reference the specific function.
For multi-input models, the same preprocessing logic can be reused if applicable.


``` python
from typing import Sequence
import torch
import cv2
import numpy as np

def preprocess_impl(path_list: Sequence[str], input_parametr: dict) -> torch.Tensor:
    """
    Read files from path_list, preprocess them according to input_parametr,
    and return a torch.Tensor.

    Args:
        path_list (Sequence[str]): List of files for one calibration batch
        input_parametr (dict): Corresponds to
            calibration_parameters.input_parametres[idx]

    Returns:
        torch.Tensor: A batch of calibration data
    """
    batch_list = []
    mean_value = input_parametr["mean_value"]
    std_value = input_parametr["std_value"]
    input_shape = input_parametr["input_shape"]
    for file_path in path_list:
        img = cv2.imread(file_path)
        img = cv2.resize(img, (input_shape[-1], input_shape[-2]), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32)
        img = (img - mean_value) / std_value
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img)
        img = torch.unsqueeze(img, 0)
        batch_list.append(img)
    return torch.cat(batch_list, dim=0)
```

## Quantization Accuracy Tuning

> TBD

## Change Log

For detailed release notes, refer to
[github-xslim-releases](https://github.com/spacemit-com/xslim/releases)
