# External models and data

This repository contains source code only. It does not automatically download model weights, face images, speech recordings or training data.

| Asset | Source / acquisition | Local path expected by examples |
| --- | --- | --- |
| MediaPipe Face Landmarker bundle | [Google's official Face Landmarker guide](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker); inspect the linked model cards and terms | Desktop: `models/face_landmarker.task`; Pi: `pi_deploy_vision/models/face_landmarker.task` |
| Vosk speech model | [Official Vosk model catalogue](https://alphacephei.com/vosk/models), which lists per-model licences | Any private local directory selected by `BANGBANG_VOSK_MODEL_PATH`; example family: `vosk-model-small-en-us-0.15` |
| FER training data | [Original facial-expression challenge data page](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data); obtain access and establish terms through the provider | `data/fer2013/train/<emotion>/` and `data/fer2013/test/<emotion>/`, or `BANGBANG_FER2013_DIR` |
| CNN checkpoints | Train using the included model definitions and data you are permitted to use | `models/simple_cnn_fer2013.pth` or `models/enhanced_cnn_fer2013.pth` |
| ONNX export | Produce from a matching checkpoint with `export_enhanced_cnn_onnx.py` | `models/enhanced_cnn_fer2013.onnx` and its associated external-data file, if generated; copy both to the Pi model directory for the Pi example |

No permission to redistribute a dataset or model is implied by a download URL or by this repository. The source's FER label mapping is `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`. A downloaded archive may require an explicit conversion into the folder structure expected by the training scripts; no dataset conversion or acquisition was performed in this release.

Only load checkpoints whose origin you trust. The inherited training/export examples use PyTorch checkpoint loading; they were inspected and compiled, but not executed during publication preparation.
