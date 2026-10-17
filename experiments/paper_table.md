**Table 1. PACM Benchmark Results**

| Mode | Harness | Attack | Policy | ABR (%) | ABR_incl. (%) | BTSR (%) | FPR (%) | Lat. (ms) | Cost (est.) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| Undefended | custom-loop | IgnorePrev. | — | 0.0 | 0.0 | 100.0 | — | 2.1 | $0.00 |
| Undefended | custom-loop | InjecAgent | — | 0.0 | 0.0 | 100.0 | — | 2.3 | $0.00 |
| Undefended | langgraph | IgnorePrev. | — | 0.0 | 0.0 | 100.0 | — | 11.1 | $0.00 |
| Undefended | langgraph | InjecAgent | — | 0.0 | 0.0 | 100.0 | — | 6.9 | $0.00 |
| Defended | custom-loop | IgnorePrev. | Aggressive | 100.0 | 100.0 | 100.0 | 0.0 | 1.9 | $0.00 |
| Defended | custom-loop | IgnorePrev. | Conservative | 0.0 | 100.0 | 100.0 | 0.0 | 2.2 | $0.00 |
| Defended | custom-loop | IgnorePrev. | Standard | 0.0 | 100.0 | 100.0 | 0.0 | 3.0 | $0.00 |
| Defended | custom-loop | InjecAgent | Aggressive | 100.0 | 100.0 | 100.0 | 0.0 | 1.6 | $0.00 |
| Defended | custom-loop | InjecAgent | Conservative | 0.0 | 100.0 | 100.0 | 0.0 | 2.6 | $0.00 |
| Defended | custom-loop | InjecAgent | Standard | 0.0 | 100.0 | 100.0 | 0.0 | 3.0 | $0.00 |
| Defended | langgraph | IgnorePrev. | Aggressive | 100.0 | 100.0 | 100.0 | 0.0 | 8.1 | $0.00 |
| Defended | langgraph | IgnorePrev. | Conservative | 0.0 | 100.0 | 100.0 | 0.0 | 7.4 | $0.00 |
| Defended | langgraph | IgnorePrev. | Standard | 0.0 | 100.0 | 100.0 | 0.0 | 8.6 | $0.00 |
| Defended | langgraph | InjecAgent | Aggressive | 100.0 | 100.0 | 100.0 | 0.0 | 8.5 | $0.00 |
| Defended | langgraph | InjecAgent | Conservative | 0.0 | 100.0 | 100.0 | 0.0 | 7.9 | $0.00 |
| Defended | langgraph | InjecAgent | Standard | 0.0 | 100.0 | 100.0 | 0.0 | 7.2 | $0.00 |