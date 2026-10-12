# Bangbang V2 — perception source

Facial landmarks, action-unit heuristics, emotion-model training, and a Raspberry Pi vision workflow from a collaborative robot-perception project. Siheng Ye contributed to the face/emotion perception pipeline and its embedded integration.

[Project overview](https://philia17.github.io/#bangbang-title) · [Source provenance](SOURCE_PROVENANCE.md) · [External assets](ASSETS.md)

The related paper, *Bangbang V2: On-Device Facial Emotion and Bilingual Speech Recognition for a Raspberry Pi-Based Tele-Operated Avatar Robot*, was accepted for presentation at **HRFEST 2026, Paper 230**. Camera-ready preparation is in progress; proceedings publication and indexing are not confirmed. This source snapshot is a set of development components, not a reproduction package for every component described in the paper.

## Included components

| Area | Source | What is available |
| --- | --- | --- |
| Face landmarks | [src/vision/face_mesh.py](src/vision/face_mesh.py) | MediaPipe Tasks integration, landmark measurements and visualisation helpers |
| Action-unit emotion heuristics | [src/vision/affect_engine_v2.py](src/vision/affect_engine_v2.py) | Temporal buffers, calibration and action-unit-to-emotion scoring |
| Operator feedback | [src/vision/au_feedback_learner.py](src/vision/au_feedback_learner.py) | Local updates to AU weights following operator corrections |
| Trainable models | [src/models/enhanced_cnn.py](src/models/enhanced_cnn.py), [src/models/emotion_lstm.py](src/models/emotion_lstm.py) | CNN/attention and temporal model definitions; training and ONNX export scripts |
| Raspberry Pi vision | [pi_deploy_vision/](pi_deploy_vision/) | Camera reader, AU processing and manual ONNX/camera examples with separate dependencies |
| Early speech/vision prototype | [legacy_prototype/src/main.py](legacy_prototype/src/main.py) | Stationary threaded prototype with Vosk speech recognition, fuzzy keyword matching, facial heuristics and template responses |

The later vision code and the earlier stationary prototype come from different source branches. They have not been merged into a single application. The included speech implementation uses Vosk with an English example vocabulary. It does **not** provide a complete bilingual cloud-speech or tele-operation implementation.

Outputs are advisory signals for an operator. Facial-expression and historical “pain” labels are heuristic categories, not validated measurements of a person's internal state or clinical findings. Response templates do not call services or operate a robot.

## Start with source inspection

Use Python 3.11 for the original dependency families. The release checks parse and compile source without importing it, opening devices or loading models:

```sh
python3 -m compileall -q src pi_deploy_vision legacy_prototype demo_*.py train_*.py export_enhanced_cnn_onnx.py
```

All 23 included Python files passed AST parsing and bytecode compilation during release preparation. Dependency installation, camera access, microphone access, inference, training, ONNX export and Raspberry Pi deployment were not re-run. Historical latency and frame-rate targets in the source are targets, not newly reproduced results.

Programs named `demo_*.py` are manual camera/device examples. They were renamed from upstream `test_*.py` so they are not mistaken for an offline automated test suite. The earlier branch's placeholder tests copied algorithms inside the tests rather than importing production code; they are not included as verification evidence.

## Desktop vision example

Create a separate environment and install the source-derived dependencies:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-vision.txt
```

Obtain the MediaPipe face-landmarker bundle through the [official model guide](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker), review its model terms, and place it at `models/face_landmarker.task`. It is not bundled or downloaded automatically.

From the repository root, this command explicitly opens a camera and starts the AU example:

```sh
python demo_affect_engine_v2.py
```

Other desktop examples include `demo_facemesh.py`, `demo_affect_engine.py`, and the `demo_*cnn*.py` scripts. CNN examples additionally need weights that match the included model definitions. No pretrained weights are supplied.

## Raspberry Pi example

Use a separate Pi environment and [pi_deploy_vision/requirements_pi.txt](pi_deploy_vision/requirements_pi.txt). The inherited dependency constraints are preserved; their installation on a fresh device has not been verified in this release.

Place the externally obtained face-landmarker bundle under `pi_deploy_vision/models/face_landmarker.task`, then run the manual example from that directory:

```sh
cd pi_deploy_vision
python demo_au_emotion_engine.py
```

The camera reader supports `VIDEO_DEVICE` as a device-index/path override. The ONNX example also needs `models/enhanced_cnn_fer2013.onnx` and any external-data file produced with that export; see [ASSETS.md](ASSETS.md). Source availability does not establish hardware readiness.

## Training and export

[train_simple_cnn.py](train_simple_cnn.py) and [train_enhanced_cnn.py](train_enhanced_cnn.py) expect an independently obtained dataset in `train/<emotion>/` and `test/<emotion>/` directories. They use `BANGBANG_FER2013_DIR` (default `data/fer2013`) and `BANGBANG_MODEL_DIR` (default `models`). The seven labels are angry, disgust, fear, happy, neutral, sad and surprise.

Training dependencies are listed in [requirements-training.txt](requirements-training.txt). [export_enhanced_cnn_onnx.py](export_enhanced_cnn_onnx.py) reads the matching local EnhancedCNN state dictionary and exports ONNX. Train and inspect your own weights or use weights with an established source and permission; this repository distributes neither datasets nor checkpoints.

## Early speech prototype

The independently preserved [legacy prototype](legacy_prototype/src/main.py) reads [default.example.yaml](legacy_prototype/config/default.example.yaml). Its code implements Vosk only; unimplemented cloud-credential configuration was removed from this release. No cloud API credentials are needed or included.

Install [legacy_prototype/requirements.txt](legacy_prototype/requirements.txt) in a separate environment. PyAudio also needs the platform's audio system/PortAudio support. Obtain a Vosk model from the [official catalogue](https://alphacephei.com/vosk/models) and select its local directory with `BANGBANG_VOSK_MODEL_PATH`. `BANGBANG_CONFIG` can select a private local configuration file.

Running the following command explicitly starts camera/microphone processing:

```sh
python legacy_prototype/src/main.py
```

The `simulate_hardware` setting should not be treated as an offline-test guarantee: the historical controller can still initialise device components. This release did not run it.

## Source and rights

The snapshot preserves the selected project source and its existing attribution. It contains no original Git history, personal records, participant media, datasets or model binaries. [SOURCE_PROVENANCE.md](SOURCE_PROVENANCE.md) identifies the exact source revisions and publication changes. [NOTICE.md](NOTICE.md) records the scope of attribution and third-party assets; no new blanket software licence has been assigned.
