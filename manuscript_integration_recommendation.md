# Manuscript Integration Recommendation
**Target Publication:** ICAIN 2026, Springer LNNS — Paper ID 1567  
**Status:** Ready for Author Review and Final Proof Packaging  

### Recommended Actionable Revisions:
1. **Replace Table 1 (Retrieval Performance):**
   Use `manuscript_tables/table1_main_performance.tex`. Replace outdated 900/1000 numbers with verified Corel-1K results: M5 mAP = 0.4798, P@10 = 0.7100.
2. **Insert Table 2 (Statistical Testing):**
   Use `manuscript_tables/table2_statistical_significance.tex` in Section VII (Results & Statistical Discussion). Clearly explain that M5 statistically outperforms classical LBP and heuristic fixed fusion (M4), while differences against equal fusion (M3) and mean control (M6) are descriptive and non-significant after Holm adjustment.
3. **Insert Table 5 (Deep Learning & Baseline Comparison):**
   Use `manuscript_tables/table5_deep_baselines.tex` to address Reviewer Concern #3. Emphasize that while deep models dominate high-end semantic accuracy, M5 provides a $157.5\times$ storage reduction and $53.4\times$ lower extraction latency on edge CPU devices.
4. **Update Section VII-C (Oxford-IIIT Pet Transfer):**
   Use `manuscript_tables/table6_oxford_transfer.tex`. Frame the Pet evaluation as an explicit boundary condition on fine-grained transfer, citing M5 mAP = 0.033332 vs M6 mAP = 0.033155, removing any overclaims of universal generalisation.
5. **Figure Replacements:**
   - Replace Fig. 1 with `publication_figures/fig_system_architecture.png`.
   - Replace Fig. 2 with `publication_figures/fig_main_performance_comparison.png`.
   - Replace Fig. 3 with `publication_figures/fig_weight_distributions.png`.
   - Add Fig. 4 using `publication_figures/fig_deep_learning_comparison.png` for runtime/storage trade-off.
   - Add Fig. 5 using `publication_figures/fig_qualitative_retrieval_examples.png` for visual query demonstrations.
