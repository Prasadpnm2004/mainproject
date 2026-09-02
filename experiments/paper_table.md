**Table 1. PACM Benchmark Results**

| Mode | Harness | Attack | Policy | ABR (%) | BTSR (%) | FPR (%) | Lat. (ms) | Cost (est.) |
|---|---|---|---|---:|---:|---:|---:|---:|
| Undefended | custom-loop | IgnorePrev. | — | 0.0 | 100.0 | — | 2200.0 | $0.0002 |
| Undefended | custom-loop | InjecAgent | — | 0.0 | 100.0 | — | 1800.0 | $0.0002 |
| Undefended | langgraph | IgnorePrev. | — | 0.0 | 100.0 | — | 9100.0 | $0.0002 |
| Undefended | langgraph | InjecAgent | — | 0.0 | 100.0 | — | 7200.0 | $0.0002 |
| Defended | custom-loop | IgnorePrev. | Aggressive | 100.0 | 100.0 | 0.0 | 2000.0 | $0.0002 |
| Defended | custom-loop | IgnorePrev. | Conservative | 0.0 | 100.0 | 0.0 | 2200.0 | $0.0002 |
| Defended | custom-loop | IgnorePrev. | Standard | 0.0 | 100.0 | 0.0 | 2200.0 | $0.0002 |
| Defended | custom-loop | InjecAgent | Aggressive | 100.0 | 100.0 | 0.0 | 1500.0 | $0.0002 |
| Defended | custom-loop | InjecAgent | Conservative | 0.0 | 100.0 | 0.0 | 2000.0 | $0.0002 |
| Defended | custom-loop | InjecAgent | Standard | 0.0 | 100.0 | 0.0 | 1900.0 | $0.0002 |
| Defended | langgraph | IgnorePrev. | Aggressive | 100.0 | 100.0 | 0.0 | 6800.0 | $0.0002 |
| Defended | langgraph | IgnorePrev. | Conservative | 0.0 | 100.0 | 0.0 | 7400.0 | $0.0002 |
| Defended | langgraph | IgnorePrev. | Standard | 0.0 | 100.0 | 0.0 | 5300.0 | $0.0002 |
| Defended | langgraph | InjecAgent | Aggressive | 100.0 | 100.0 | 0.0 | 7500.0 | $0.0002 |
| Defended | langgraph | InjecAgent | Conservative | 0.0 | 100.0 | 0.0 | 8300.0 | $0.0002 |
| Defended | langgraph | InjecAgent | Standard | 0.0 | 100.0 | 0.0 | 7600.0 | $0.0002 |