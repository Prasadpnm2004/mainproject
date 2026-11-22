**Table 1. PACM Benchmark Results**

| Mode | Harness | Attack | Policy | ABR (%) | ABR_incl. (%) | BTSR (%) | FPR (%) | Lat. (ms) | Cost (est.) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| Undefended | custom-loop | IgnorePrev. | — | 0.0 | 0.0 | 100.0 | — | 0.7 | $0.00 |
| Undefended | custom-loop | InjecAgent | — | 0.0 | 0.0 | 100.0 | — | 0.8 | $0.00 |
| Undefended | langgraph | IgnorePrev. | — | 0.0 | 0.0 | 100.0 | — | 5.0 | $0.00 |
| Undefended | langgraph | InjecAgent | — | 0.0 | 0.0 | 100.0 | — | 3.8 | $0.00 |
| Defended | custom-loop | IgnorePrev. | Aggressive | 100.0 | 100.0 | 100.0 | 0.0 | 0.7 | $0.00 |
| Defended | custom-loop | IgnorePrev. | Conservative | 0.0 | 100.0 | 100.0 | 0.0 | 0.8 | $0.00 |
| Defended | custom-loop | IgnorePrev. | Standard | 0.0 | 100.0 | 100.0 | 0.0 | 0.8 | $0.00 |
| Defended | custom-loop | InjecAgent | Aggressive | 100.0 | 100.0 | 100.0 | 0.0 | 0.7 | $0.00 |
| Defended | custom-loop | InjecAgent | Conservative | 0.0 | 100.0 | 100.0 | 0.0 | 0.9 | $0.00 |
| Defended | custom-loop | InjecAgent | Standard | 0.0 | 100.0 | 100.0 | 0.0 | 0.9 | $0.00 |
| Defended | langgraph | IgnorePrev. | Aggressive | 100.0 | 100.0 | 100.0 | 0.0 | 2.5 | $0.00 |
| Defended | langgraph | IgnorePrev. | Conservative | 0.0 | 100.0 | 100.0 | 0.0 | 2.7 | $0.00 |
| Defended | langgraph | IgnorePrev. | Standard | 0.0 | 100.0 | 100.0 | 0.0 | 2.7 | $0.00 |
| Defended | langgraph | InjecAgent | Aggressive | 100.0 | 100.0 | 100.0 | 0.0 | 2.4 | $0.00 |
| Defended | langgraph | InjecAgent | Conservative | 0.0 | 100.0 | 100.0 | 0.0 | 2.7 | $0.00 |
| Defended | langgraph | InjecAgent | Standard | 0.0 | 100.0 | 100.0 | 0.0 | 3.3 | $0.00 |