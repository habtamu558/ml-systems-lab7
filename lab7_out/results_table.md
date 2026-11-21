| Pillar | Metric | Baseline | Unstruct. 90% (global) | Struct. 50% filters | Student plain | Student distilled |
|---|---|---|---|---|---|---|
| DATA | Training data / schedule | 60000 img / 10 ep | +3 ep fine-tune | +3 ep fine-tune | 60000 img / 15 ep | 60000 img / 15 ep + teacher soft labels |
| MODEL | Test accuracy (%) | 92.53 | 93.12 | 93.07 | 91.68 | 91.52 |
| MODEL | Parameters (total) | 997,434 | 997,434 | 476,474 | 106,954 | 106,954 |
| MODEL | Non-zero parameters | 997,434 | 100,084 | 476,474 | 106,954 | 106,954 |
| MODEL | Size, dense file (KB) | 3900 | 3900 | 1865 | 421 | 421 |
| MODEL | Size, sparse CSR file (KB) | 7799 | 788 | 3729 | 842 | 842 |
| COMPUTE | Dense MACs / image (M) | 30.03 | 30.03 | 7.79 | 1.96 | 1.96 |
| COMPUTE | Effective (non-zero) MACs (M) | 30.03 | 8.49 | 7.79 | 1.96 | 1.96 |
| COMPUTE | CPU latency batch=1 (ms) | 4.694 | 5.522 | 1.590 | 0.899 | 0.873 |
| COMPUTE | CPU latency batch=64 (ms) | 210.07 | 246.63 | 77.32 | 31.00 | 32.13 |
| ENERGY | Energy proxy (uJ/inference) | 777 | 103 | 341 | 77 | 77 |

Student accuracy: plain 91.68 +/- 0.29, distilled 91.52 +/- 0.08 over seeds [0, 1, 2].
Energy proxy = eff_MACs*4.6 pJ + non-zero weights*640 pJ (Horowitz 2014 numbers); an estimate, not a measurement.