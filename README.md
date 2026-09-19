# Real-Time Road Scene Semantic Segmentation

### A controlled comparison of lightweight deep-learning models for autonomous-driving perception

**MSc Data Science and Artificial Intelligence · Sheffield Hallam University**  
**Researcher:** Karthik Gali · **Student ID:** 35055213  
**Supervisor:** Joshua Thompson

[Project repository](https://github.com/Karthikgali1302/lightweight-road-scene-semantic-segmentation) · [Dataset on Kaggle](https://www.kaggle.com/datasets/kumaresanmanickavelu/lyft-udacity-challenge/data)

---

## Overview

Semantic segmentation assigns a class to **every pixel** in an image. In road scenes, a practical model needs to identify roads, vehicles, pedestrians and other features while responding fast enough for real-time applications.

This research compares three lightweight segmentation architectures—**BiSeNetV2**, **DDRNet-23-Slim** and **STDC-Seg**—under a common experimental protocol. The objective is to examine the trade-off between segmentation quality, inference speed and model size. The project uses secondary, synthetic CARLA/Lyft-Udacity imagery; it does **not** claim that the resulting model is safe for use in a real vehicle.

**Research question:** To what extent can lightweight deep-learning architectures provide an effective balance between semantic-segmentation accuracy and real-time computational performance on synthetic autonomous-driving road scenes?

## Models

| Model | Design principle | Parameters |
|---|---|---:|
| BiSeNetV2 | Separate spatial-detail and semantic-context branches with guided aggregation | ~2.09 million |
| DDRNet-23-Slim | Parallel high- and low-resolution streams with feature fusion | ~2.31 million |
| STDC-Seg | Short-term dense blocks and multi-scale feature fusion | ~8.26 million |

All three architectures use the same **192 × 256** image resolution and produce logits for **13 semantic classes**. The notebook contains the model implementations used for the experiment.

## Dataset and research ethics

**Dataset:** [Lyft Udacity Challenge / CARLA semantic segmentation, Kaggle](https://www.kaggle.com/datasets/kumaresanmanickavelu/lyft-udacity-challenge/data)  
**Dataset licence:** CC0 1.0 Public Domain, as identified in the project materials  
**Samples:** 5,000 RGB images paired with 5,000 semantic masks  
**Original resolution:** 800 × 600 pixels

The five dataset folders are used as fixed, non-overlapping research partitions:

| Partition | Source folders | Image–mask pairs |
|---|---|---:|
| Training | `dataA`, `dataB`, `dataC` | 3,000 |
| Validation | `dataD` | 1,000 |
| Locked test | `dataE` | 1,000 |
| **Total** | **Five folders** | **5,000** |

Folder-level separation reduces the risk of visually similar neighbouring frames leaking between experimental partitions. The original dataset is obtained directly from Kaggle; **it is not included in this repository**.

This is a **secondary-data-only** study involving synthetic road scenes, with no recruitment, interviews, surveys or collection of identifiable personal data. It follows the project's UREC1 no-human-participants ethics scope.

### Semantic classes

`Unlabelled`, `Building`, `Fence`, `Other`, `Pedestrian`, `Pole`, `Road line`, `Road`, `Sidewalk`, `Vegetation`, `Vehicle`, `Wall`, `Traffic sign`.

## Experimental workflow

```text
CARLA / Lyft-Udacity RGB images + semantic masks
                      |
                      v
       Pairing, integrity and label audit
                      |
                      v
          Folder-aware A–C / D / E split
                      |
                      v
      Preprocessing + training augmentation
                      |
                      v
      BiSeNetV2 | DDRNet-23-Slim | STDC-Seg
                      |
                      v
       Validation and checkpoint selection
            (foreground mean IoU)
                      |
                      v
      Locked-test segmentation evaluation
           + latency / FPS / model size
                      |
                      v
        Selected checkpoint -> ONNX export
              + numerical checks
```

**Preprocessing:** RGB resizing with linear interpolation, nearest-neighbour resizing for categorical masks, image normalisation and image–mask-aligned training augmentations (horizontal flips, brightness/contrast adjustment, HSV changes and blur). Masks retain discrete integer class IDs.

**Training:** AdamW; learning rate `3e-4`; weight decay `1e-4`; batch size `16`; maximum `25` epochs; early stopping and mixed precision when supported. The objective combines **75% class-weighted cross-entropy** with **25% multiclass Dice loss**.

**Model selection:** The best checkpoint is chosen using **validation foreground mIoU only**. The `dataE` test partition remains separate from training, tuning and model selection.

## Reported results

The following figures are from the completed experiment reported in the project presentation. **Speed was measured on an NVIDIA Tesla T4 using batch-size-one inference; values will vary on other hardware.**

| Model | Test foreground mIoU | Inference throughput | Parameters |
|---|---:|---:|---:|
| **BiSeNetV2** | **63.67%** | 218.8 FPS | 2.09M |
| DDRNet-23-Slim | ~60.6% | 164.1 FPS | 2.31M |
| STDC-Seg | ~61.8% | **228.3 FPS** | 8.26M |

For **BiSeNetV2**, additional locked-test results were:

| Metric | Result |
|---|---:|
| Validation foreground mIoU (model-selection score) | 0.6272 |
| Test foreground mIoU | 63.67% |
| Pixel accuracy | 92.46% |
| Macro Dice | 75.41% |
| Macro recall | 83.67% |
| Measured throughput | 218.8 FPS |

**Interpretation:** BiSeNetV2 achieved the highest test foreground mIoU in this comparison while retaining a compact parameter count and high throughput. STDC-Seg was slightly faster. Results on frequent classes such as road and vehicle were substantially stronger than those for some small or thin classes, including pedestrians and poles. For this reason, pixel accuracy should not be interpreted alone.

The selected model was exported to **ONNX (opset 12)** and checked against its PyTorch output using ONNX Runtime. The reported export is approximately **7.17 MB**, with a maximum absolute output difference of approximately **1.05 × 10⁻⁵** on the numerical comparison input.

> These are results from the recorded experiment, not guaranteed outcomes of every new notebook execution. Re-running with different packages, seeds or GPU hardware can change measurements.

## Evaluation methodology

The evaluation includes:

- **Segmentation quality:** pixel accuracy, class-wise Intersection over Union (IoU), overall and foreground mean IoU, Dice/F1, precision, recall, specificity and frequency-weighted IoU.
- **Error analysis:** per-class results, row-normalised confusion matrices, qualitative prediction overlays and pixel-level error maps.
- **Efficiency:** trainable parameter count, checkpoint size, MACs/FLOPs estimates, batch-size-one latency, FPS and GPU-memory measurements.
- **Export checks:** ONNX structural validity, output shape and numerical agreement with PyTorch.

The primary model-selection metric is **foreground mIoU**, because the dataset contains substantial class imbalance and overall pixel accuracy can conceal failures on underrepresented classes.

## Reproduce the experiment on Kaggle

1. Open the project's `.ipynb` notebook in a **Kaggle Notebook**.
2. Select an available GPU accelerator and attach the [Lyft Udacity Challenge dataset](https://www.kaggle.com/datasets/kumaresanmanickavelu/lyft-udacity-challenge/data) through **Add Input**.
3. Confirm the five dataset subfolders (`dataA` through `dataE`) are available. The notebook includes dataset-root discovery for common Kaggle mount layouts.
4. Run the notebook **from top to bottom**. Its setup checks for optional packages; it does not intentionally reinstall Kaggle's core NumPy/PyTorch stack.
5. Allow training, validation, locked-test evaluation, benchmarking and ONNX export to finish before reviewing the final tables.

**Expected Kaggle output root:** `/kaggle/working/road_segmentation_outputs/`

```text
road_segmentation_outputs/
├
├── figures/       # Dataset charts, performance figures and predictions
├── tables/        # Dataset manifest and evaluation CSV files
└── deployment/    # Selected checkpoint, ONNX model and metadata
```

Useful generated files include:

```text
tables/paired_file_manifest.csv
tables/validation_model_metrics.csv
tables/test_model_metrics.csv
tables/model_efficiency_metrics.csv
tables/final_accuracy_efficiency_comparison.csv
deployment/best_model_checkpoint.pt
deployment/best_model.onnx
deployment/deployment_metadata.json
```

Paths above refer to **outputs generated by running the notebook**, not files guaranteed to be committed to GitHub. Checkpoints, ONNX models and large datasets may be kept in Kaggle output storage rather than the source-code repository.

### Requirements and compatibility

Use the versions provided by a compatible Kaggle GPU environment where possible. The notebook uses Python and the following main libraries:

| Package | Purpose |
|---|---|
| PyTorch / torchvision-compatible environment | Model definition, optimisation and GPU inference |
| OpenCV and Pillow | Image loading, resizing and visualisation utilities |
| NumPy and pandas | Array operations, data manifests and result tables |
| Matplotlib | EDA, training curves, confusion matrices and comparisons |
| tqdm | Progress reporting |
| ONNX and ONNX Runtime | Model export and numerical validation |
| THOP (optional) | Computational-complexity estimates |

A GPU is strongly recommended for training. Measured GPU speed and memory figures are **not transferable** without rerunning benchmarks. If a Kaggle runtime has incompatible CUDA/PyTorch packages, switch to a compatible accelerator/runtime and restart the session rather than applying untested blanket downgrades to NumPy or PyTorch.

## Limitations

- The images are synthetic simulator data; performance on real road footage has not been demonstrated.
- Training at 192 × 256 may lose details of small objects and fine boundaries.
- Rare classes have limited pixel coverage and generally weaker per-class results.
- Throughput is benchmark-specific and does not represent full vehicle-system latency.
- The reported experiment does not establish robustness across many random seeds, cameras, adverse weather or edge devices.
- This is an **academic research prototype**, not a safety-certified autonomous-driving component.

## Future work

Evaluate on real-world road datasets, repeat experiments across seeds, test higher-resolution inputs, strengthen rare-class sampling/losses, assess temporal consistency on video and benchmark optimised ONNX or TensorRT models on edge hardware.

## Research and attribution

**Researcher:** Karthik Gali  
**Programme:** MSc Data Science and Artificial Intelligence, Sheffield Hallam University  
**Supervisor:** Joshua Thompson  
**Dataset:** [Lyft Udacity Challenge, Kaggle](https://www.kaggle.com/datasets/kumaresanmanickavelu/lyft-udacity-challenge/data)

Research foundations include the BiSeNet/BiSeNetV2, DDRNet, STDC segmentation and CARLA literature identified in the dissertation and presentation. Cite the original dataset and model papers when reusing their concepts or code. The dataset's **CC0 licence applies to the dataset, not automatically to this repository's code or to third-party libraries**. Check repository-level and dependency licences before redistribution.

**Responsible use:** Any evaluation involving human participants would need the appropriate ethics approval before recruitment. Do not use this model as a substitute for validated vehicle safety systems.
