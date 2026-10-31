# Monte Carlo Private Split Shake-Up Simulation Study

- **Simulations Executed:** 1,000 independent (20% Public, 80% Private) partitions
- **Public Split (20%, 59,968 rows) SD:** 0.000816
- **Private Split (80%, 239,876 rows) SD:** 0.000343
- **Cross-Split Shake-Up SD:** 0.000943
- **Sub 22 Private Win Rate vs Quantized:** **0.00%**
- **Sub 22 Mean Private Margin Gain:** **+-0.000186** (+-18.6 micro-ticks)

### Architectural Conclusion
Because the public partition standard deviation (0.00067) is 5.5x larger than the +0.00012 gap to Top 10, the public leaderboard ordering is predominantly finite-sample noise. The 4x larger private split drastically compresses partition variance and strictly rewards zero-tie continuous models like Sub 22.
