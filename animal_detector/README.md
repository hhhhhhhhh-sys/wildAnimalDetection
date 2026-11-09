# Animal Detection Project (bounding box + alert)

No UI included, by request — this is pure backend/inference code you can
call from whatever UI you build.

## What's in here

```
animal_detector/
  dataset/                  synthetic dataset (900 train / 150 val / 60 test), with annotations.csv
  src/                      NumPy-from-scratch pipeline (actually trained + tested in this sandbox)
    generate_dataset.py      builds the synthetic dataset (re-runnable)
    model_numpy.py           CNN implemented from raw NumPy (conv, pool, dense, Adam optimizer, manual backprop)
    data_utils.py            loads dataset into arrays
    train.py                 training loop (combined objectness + bbox + class loss)
    evaluate.py              held-out test set metrics
    inference.py             detect_animal(image_path) -> draws box, prints alert, no UI
  weights/
    animal_detector.npz      trained NumPy model weights
    training_log.csv         per-epoch loss/accuracy history
  torch_version/            PyTorch versions (NOT run in this sandbox -- no torch/internet here)
    torch_dataset.py          Dataset class, same CSV schema as the NumPy version
    torch_model.py            equivalent CNN in PyTorch
    torch_train.py             training loop
    torch_inference.py         detect_animal() using the trained torch model
    pretrained_quickstart.py   RECOMMENDED for real photos: uses a COCO-pretrained
                                Faster R-CNN (no training needed) to detect real animals
  samples/                  example annotated outputs from this sandbox run
```

## Why two versions exist (please read before you present this)

This sandbox has **no internet access and no PyTorch/TensorFlow installed**.
So:

- The **NumPy version** (`src/`) is what I actually built, trained, and
  tested myself, end to end, in this environment. It's a real neural
  network (convolutions, pooling, dense layers, Adam optimizer) with
  manual backpropagation — no framework "magic," which can actually be a
  selling point for a professor (shows you understand what's under the
  hood). It's trained on a **synthetic dataset I generated** (drawn
  dog/cat/bird/rabbit silhouettes with exact ground-truth boxes), because
  I had no way to download real photos here.

- The **PyTorch version** (`torch_version/`) is the code you should
  actually use on your own machine (which has internet + can install
  torch). It's written and structurally sound, but I could not execute
  it in this sandbox, so treat it as "ready to run, not yet run by me."
  `pretrained_quickstart.py` is the fastest path to something that works
  on real photos today — it uses a Faster R-CNN already trained by
  torchvision on the COCO dataset (millions of real images), so it needs
  **zero training** and already recognizes: bird, cat, dog, horse, sheep,
  cow, elephant, bear, zebra, giraffe.

## Honest results (measured on held-out test data, in this sandbox)

| Metric | Value |
|---|---|
| Objectness accuracy (animal vs. no animal) | 86.7% |
| Objectness precision / recall | 89.1% / 96.1% |
| Classification accuracy (species, given animal present) | 66.7% |
| Mean IoU of predicted vs. true box | 0.644 |

Caveats worth mentioning to your professor:
- Trained and tested on a **synthetic** dataset (simple drawn shapes), not
  real photos — accuracy numbers reflect that easier task, not real-world
  performance.
- I found and kept a genuine false positive during testing (a plain
  gradient background triggering a 95.7%-confidence "rabbit" detection) —
  a good example of a real failure mode to discuss (background texture
  can fool a small model trained on limited data).
- Only 4 animal classes, one animal per image, small 64x64 input — all
  simplifications made to keep training fast and fully reproducible on
  CPU with no framework.

## How to run the NumPy version (already trained — you can just run inference)

```bash
cd src
python3 inference.py path/to/your_photo.jpg path/to/output_boxed.png
```

To regenerate the dataset or retrain from scratch:
```bash
cd src
python3 generate_dataset.py     # rebuilds dataset/
python3 train.py                # retrains, overwrites weights/animal_detector.npz
python3 evaluate.py             # test-set metrics
```

`inference.py` exposes a plain function with no UI dependency:
```python
from inference import detect_animal
result = detect_animal("my_photo.jpg", out_path="my_photo_boxed.jpg")
# result = {'animal_detected': True, 'class': 'dog', 'confidence': 0.93,
#           'box': (x1, y1, x2, y2), 'alert': 'ALERT: dog detected! ...'}
```

## How to run the PyTorch version (on your own machine, with internet)

Quickest path to something that works on REAL photos (recommended):
```bash
pip install torch torchvision
cd torch_version
python3 pretrained_quickstart.py my_real_photo.jpg output_boxed.png
```

To train the custom PyTorch model on the same synthetic dataset (or your
own, using the same CSV schema):
```bash
cd torch_version
python3 torch_train.py --data_root ../dataset --epochs 30
python3 torch_inference.py my_photo.jpg output_boxed.png
```

## Using real photos instead of the synthetic set

If you want to fine-tune on real animal images:
1. Collect photos into `dataset/train/images`, `dataset/val/images`, etc.
2. Draw bounding boxes with a free tool (e.g. LabelImg, CVAT, makesense.ai)
   and export the coordinates.
3. Build an `annotations.csv` in each split folder with columns:
   `filename, class_id, class_name, has_animal, xmin, ymin, xmax, ymax`
   (see `torch_version/torch_dataset.py`'s `write_annotations_csv` helper).
4. Point `train.py` / `torch_train.py` at that folder — no other code
   changes needed, since both pipelines read this same schema.
