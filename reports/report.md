# Closed-loop benchmark report
Runtime: 29.4s | imagine loss 0.056 | distill loss 0.0001
## World-model horizon errors
| horizon | latent MSE | BEV IoU |
|---|---|---|
| 1 | 0.3152 | 0.000 |
| 5 | 0.2710 | 0.000 |
| 15 | 0.2003 | 0.000 |

## Closed-loop (held-out seeds 100-101)
| policy | route↑ | collision↓ | off-road↓ | jerk↓ | progress↑ |
|---|---|---|---|---|---|
| expert | 0.46 | 0.00 | 0.20 | 0.14 | 83.4 |
| bc | 0.40 | 0.00 | 1.00 | 0.11 | 71.4 |
| cem | 0.17 | 0.00 | 1.00 | 3.92 | 26.1 |
| dreamer | 0.03 | 0.00 | 1.00 | 0.25 | 4.5 |
| student | 0.35 | 0.00 | 0.70 | 0.03 | 63.5 |

## Per-scenario failure breakdown
| policy | scenario | route | collision | off-road |
|---|---|---|---|---|
| expert | cruise | 0.48 | 0.00 | 0.00 |
| expert | merge | 0.48 | 0.00 | 0.00 |
| expert | cutin | 0.46 | 0.00 | 0.00 |
| expert | pedestrian | 0.41 | 0.00 | 0.00 |
| expert | unstructured | 0.48 | 0.00 | 1.00 |
| bc | cruise | 0.39 | 0.00 | 1.00 |
| bc | merge | 0.40 | 0.00 | 1.00 |
| bc | cutin | 0.40 | 0.00 | 1.00 |
| bc | pedestrian | 0.39 | 0.00 | 1.00 |
| bc | unstructured | 0.40 | 0.00 | 1.00 |
| cem | cruise | 0.19 | 0.00 | 1.00 |
| cem | merge | 0.00 | 0.00 | 1.00 |
| cem | cutin | 0.01 | 0.00 | 1.00 |
| cem | pedestrian | 0.36 | 0.00 | 1.00 |
| cem | unstructured | 0.27 | 0.00 | 1.00 |
| dreamer | cruise | 0.02 | 0.00 | 1.00 |
| dreamer | merge | 0.03 | 0.00 | 1.00 |
| dreamer | cutin | 0.02 | 0.00 | 1.00 |
| dreamer | pedestrian | 0.03 | 0.00 | 1.00 |
| dreamer | unstructured | 0.02 | 0.00 | 1.00 |

## Distillation / latency
```json
{
  "error": "No module named 'onnxruntime'"
}
```

Student route completion: 0.35 (vs BC above). Rollout strip: `rollout_strip.png`.
