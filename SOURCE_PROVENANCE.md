# Source provenance

Prepared from two tracked source revisions, with no original Git history included:

- Later vision branch: `master`, commit `1bc10b060904445e63a0cf33eaa42eb0b5d34873`.
- Earlier stationary prototype: `main`, commit `d1192bb0446156d5142d753723e5ec66107eaa25`.

No files were taken from an uncommitted local checkout. The branches contain different development workflows. This snapshot does not claim that the combined paper system can be reproduced from a single entry point.

## Included source mapping

| Destination | Source branch / path | Changes |
| --- | --- | --- |
| `src/models/emotion_lstm.py` | `master:src/models/emotion_lstm.py` | Unchanged source. |
| `src/models/enhanced_cnn.py` | `master:src/models/enhanced_cnn.py` | Unchanged source. |
| `src/vision/affect_engine.py` | `master:src/vision/affect_engine.py` | Unchanged source. |
| `src/vision/affect_engine_v2.py` | `master:src/vision/affect_engine_v2.py` | Unchanged source. |
| `src/vision/au_feedback_learner.py` | `master:src/vision/au_feedback_learner.py` | Unchanged source. |
| `src/vision/face_mesh.py` | `master:src/vision/face_mesh.py` | Unchanged source. |
| `train_simple_cnn.py` | `master:train_simple_cnn.py` | Replaced personal training/data output paths with environment settings and portable local defaults. |
| `train_enhanced_cnn.py` | `master:train_enhanced_cnn.py` | Replaced personal training/data output paths with environment settings and portable local defaults. |
| `export_enhanced_cnn_onnx.py` | `master:export_enhanced_cnn_onnx.py` | Unchanged source. |
| `pi_deploy_vision/au_emotion_engine.py` | `master:pi_deploy_vision/au_emotion_engine.py` | Unchanged source. |
| `pi_deploy_vision/au_feedback_learner.py` | `master:pi_deploy_vision/au_feedback_learner.py` | Unchanged source. |
| `pi_deploy_vision/camera.py` | `master:pi_deploy_vision/camera.py` | Unchanged source. |
| `pi_deploy_vision/face_mesh.py` | `master:pi_deploy_vision/face_mesh.py` | Unchanged source. |
| `pi_deploy_vision/requirements_pi.txt` | `master:pi_deploy_vision/requirements_pi.txt` | Unchanged source. |
| `pi_deploy_vision/demo_au_emotion_engine.py` | `master:pi_deploy_vision/test_au_emotion_engine.py` | Renamed camera/manual example from test_ to demo_ to prevent accidental test discovery. |
| `pi_deploy_vision/demo_camera.py` | `master:pi_deploy_vision/test_camera.py` | Renamed camera/manual example from test_ to demo_ to prevent accidental test discovery. |
| `pi_deploy_vision/demo_enhanced_cnn_onnx.py` | `master:pi_deploy_vision/test_enhanced_cnn_onnx.py` | Renamed camera/manual example from test_ to demo_ to prevent accidental test discovery. |
| `demo_affect_engine.py` | `master:test_affect_engine.py` | Renamed camera/manual example from test_ to demo_ to prevent accidental test discovery. |
| `demo_affect_engine_v2.py` | `master:test_affect_engine_v2.py` | Renamed camera/manual example from test_ to demo_ to prevent accidental test discovery. |
| `demo_cnn_realtime.py` | `master:test_cnn_realtime.py` | Renamed camera/manual example from test_ to demo_ to prevent accidental test discovery. |
| `demo_enhanced_cnn.py` | `master:test_enhanced_cnn.py` | Renamed camera/manual example from test_ to demo_ to prevent accidental test discovery. |
| `demo_enhanced_cnn_onnx.py` | `master:test_enhanced_cnn_onnx.py` | Renamed camera/manual example from test_ to demo_ to prevent accidental test discovery. |
| `demo_facemesh.py` | `master:test_facemesh.py` | Renamed camera/manual example from test_ to demo_ to prevent accidental test discovery. Removed personal absolute path; imports use the included src namespace and model path is relative to repository root. |
| `legacy_prototype/src/main.py` | `main:src/main.py` | Select non-secret example configuration by default; BANGBANG_CONFIG and BANGBANG_VOSK_MODEL_PATH override local paths. Clarify archival heuristic/operator scope. |
| `legacy_prototype/config/default.example.yaml` | `main:config/default.yaml` | Removed unimplemented cloud credential fields; retained Vosk-only example and changed response templates to operator assistance without claiming automatic calls. |

## Release documentation

The README, asset guide, attribution notice, ignore rules and source-derived desktop/training/legacy dependency lists were written for this snapshot. Pi dependency constraints remain those of the source branch. No licence, model-distribution permission, hardware validation or performance result is inferred from those lists.

Source changes are limited to portable local paths, explicit private configuration selection, truthful operator-response wording, removal of unimplemented cloud-credential fields, and manual-example naming. The algorithms were not retrained or tuned in this preparation.
