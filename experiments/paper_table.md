**Table 1. PACM Benchmark Results**

| Mode | Harness | Attack | Policy | ABR (%) | BTSR (%) | FPR (%) | Lat. (ms) | Cost (est.) |
|---|---|---|---|---:|---:|---:|---:|---:|
| Undefended | custom-loop | IgnorePrev. | — | 0.0 | 100.0 | — | 800.0 | $0.0002 |
| Undefended | custom-loop | InjecAgent | — | 0.0 | 100.0 | — | 1300.0 | $0.0002 |
| Undefended | langgraph | IgnorePrev. | — | 0.0 | 100.0 | — | 5700.0 | $0.0002 |
| Undefended | langgraph | InjecAgent | — | 0.0 | 100.0 | — | 2800.0 | $0.0002 |
| Defended | custom-loop | IgnorePrev. | Aggressive | 0.0 | 100.0 | 0.0 | 900.0 | $0.0002 |
| Defended | custom-loop | IgnorePrev. | Conservative | 0.0 | 100.0 | 0.0 | 900.0 | $0.0002 |
| Defended | custom-loop | IgnorePrev. | Standard | 0.0 | 100.0 | 0.0 | 900.0 | $0.0002 |
| Defended | custom-loop | InjecAgent | Aggressive | 0.0 | 100.0 | 0.0 | 1000.0 | $0.0002 |
| Defended | custom-loop | InjecAgent | Conservative | 0.0 | 100.0 | 0.0 | 900.0 | $0.0002 |
| Defended | custom-loop | InjecAgent | Standard | 0.0 | 100.0 | 0.0 | 1000.0 | $0.0002 |
| Defended | langgraph | IgnorePrev. | Aggressive | 0.0 | 100.0 | 0.0 | 6000.0 | $0.0002 |
| Defended | langgraph | IgnorePrev. | Conservative | 0.0 | 100.0 | 0.0 | 3700.0 | $0.0002 |
| Defended | langgraph | IgnorePrev. | Standard | 0.0 | 100.0 | 0.0 | 3600.0 | $0.0002 |
| Defended | langgraph | InjecAgent | Aggressive | 0.0 | 100.0 | 0.0 | 3400.0 | $0.0002 |
| Defended | langgraph | InjecAgent | Conservative | 0.0 | 100.0 | 0.0 | 3500.0 | $0.0002 |
| Defended | langgraph | InjecAgent | Standard | 0.0 | 100.0 | 0.0 | 3100.0 | $0.0002 |