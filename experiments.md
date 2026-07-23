# 数据集

| Name              | # Nodes    | # Edges     | Summary       |
| ----------------- | ---------- | ----------- | ------------- |
| Ego-Facebook (FA) | 4,039      | 88,234      | Social        |
| Email-Enron (EM)  | 36,692     | 183,831     | Email         |
| Amazon(AM)        | 334,863    | 925,872     | Network       |
| DBLP (DB)         | 317,080    | 1,049,866   | Collaboration |
| Youtube (YO)      | 1,134,890  | 2,987,624   | Social        |
| RoadNet-CA (RN)   | 1,965,206  | 2,766,607   | Network       |
| LiveJournal (LJ)  | 3,997,962  | 34,681,189  | Social        |
| Hollywood (HO)    | 1,107,243  | 56,375,711  | Collaboration |
| Orkut             | 3,072,441  | 117,185,083 | Network       |
| UK-02 (U2)        | 18,520,343 | 261,787,258 | Hyperlinks    |
| UK-05 (U5)        | 39,459,923 | 783,027,125 | Hyperlinks    |

# 主实验表格

压缩后表示大小 / 原图边数（越小约好）

| Dataset | SWeG Rel. Size | SWeG Time (s) | SLUGGER Rel. Size | SLUGGER Time (s) | BEAM Rel. Size | BEAM Time (s) |
| ------- | -------------- | ------------- | ----------------- | ---------------- | -------------- | ------------- |
| FA      | 0.657          | 3.27          | 0.442             | 2.31             | 0.484          | 1.87          |
| EM      | 0.891          | 5.03          | 0.763             | 7.63             | 0.711          | 3.99          |
| AM      | 0.879          | 10.55         | 0.802             | 11.11            | 0.619          | 10.35         |
| DB      | 0.698          | 15.34         | 0.690             | 13.82            | 0.571          | 10.11         |
| YO      | 0.815          | 1127.86       | 0.935             | 103.32           | 0.735          | 68.90         |
| RN      | 0.9993         | 87.65         | 0.9997            | 46.89            | 0.716          | 62.36         |
| LJ      | 0.865          | 435.19        | 0.764             | 1061.45          | 0.756          | 299.03        |
| HO      | 0.598          | 987.03        | 0.445             | 3525.73          | 0.549          | 263.78        |
| Orkut   | -              | -             | -                 | -                | 0.871          | 581.63        |
| U2      | 0.173          | 9442.87       | 0.144             | 8071.43          | 0.126          | 2024.26       |
| U5      | 0.130          | 45681.93      | 0.110             | 24170.22         | 0.101          | 4585.81       |

# Overall Comparison with Speedups

| Dataset           | SWeG Rel. Size | SWeG Time (s) | SLUGGER Rel. Size | SLUGGER Time (s) | BEAM Rel. Size | BEAM Time (s) | Speedup vs SWeG | Speedup vs SLUGGER |
| ----------------- | -------------- | ------------- | ----------------- | ---------------- | -------------- | ------------- | --------------- | ------------------ |
| FA                | 0.657          | 3.27          | **0.442**         | 2.31             | 0.497          | **1.87**      | 1.75×           | 1.24×              |
| EM                | 0.891          | 5.03          | 0.763             | 7.63             | **0.711**      | **3.99**      | 1.26×           | 1.91×              |
| AM                | 0.879          | 10.55         | 0.802             | 11.11            | **0.619**      | **10.35**     | 1.02×           | 1.07×              |
| DB                | 0.698          | 15.34         | 0.690             | 13.82            | **0.571**      | **10.11**     | 1.52×           | 1.37×              |
| YO                | 0.815          | 1127.86       | 0.935             | 103.32           | **0.735**      | **68.90**     | 16.37×          | 1.50×              |
| RN                | 0.999          | 87.65         | **1.000**         | **46.89**        | **0.716**      | 62.36         | 1.41×           | 0.75×              |
| LJ                | 0.865          | 435.19        | 0.764             | 1061.45          | **0.756**      | **299.03**    | 1.46×           | 3.55×              |
| HO                | 0.598          | 987.03        | **0.445**         | 3525.73          | 0.549          | **263.78**    | 3.74×           | 13.37×             |
| U2                | 0.173          | 9442.87       | 0.144             | 8071.43          | **0.126**      | **2024.26**   | 4.66×           | 3.99×              |
| U5                | 0.130          | 45681.93      | 0.110             | 24170.22         | **0.101**      | **4585.81**   | **9.96×**       | **5.27×**          |
| Geo. mean speedup | --             | --            | --                | --               | --             | --            | **2.69×**       | **2.29×**          |

# BEAM Runtime Breakdown

| Dataset | Total (s) | Divide (s) | Merge (s) | Encode (s) | Prepare (s) | Gain Scoring (s) | Merge Share | Prep.+Gain Share | Gain Share | Candidates |
| ------- | --------: | ---------: | --------: | ---------: | ----------: | ---------------: | ----------: | ----------------: | ---------: | ---------: |
| FA      |      1.87 |       0.02 |      1.83 |       0.02 |        0.36 |             0.04 |       97.9% |             21.3% |       2.0% |     11,713 |
| EM      |      3.99 |       0.17 |      3.79 |       0.04 |        1.82 |             0.32 |       94.8% |             53.7% |       8.0% |    359,890 |
| AM      |     10.35 |       2.65 |      7.48 |       0.22 |        2.32 |             1.82 |       72.3% |             40.1% |      17.6% |    299,875 |
| DB      |     10.11 |       2.18 |      7.68 |       0.24 |        2.47 |             1.92 |       76.0% |             43.5% |      19.0% |    380,762 |
| YO      |     68.90 |      11.33 |     56.84 |       0.71 |       38.33 |            10.56 |       82.5% |             70.9% |      15.3% | 13,548,785 |
| RN      |     62.36 |      30.24 |     31.20 |       0.88 |        8.55 |            11.53 |       50.0% |             32.2% |      18.5% |    862,286 |
| LJ      |    299.03 |      81.19 |    208.04 |       9.73 |      112.69 |            44.28 |       69.6% |             52.5% |      14.8% |  4,175,972 |
| HO      |    263.78 |      15.69 |    236.21 |      11.87 |      183.70 |            25.65 |       89.5% |             79.4% |       9.7% |  7,659,141 |
| OR      |    581.63 |      82.22 |    467.60 |      31.77 |      273.60 |            69.38 |       80.4% |             59.0% |      11.9% |  1,801,651 |
| U2      |   2024.26 |     303.31 |   1679.29 |      41.29 |     1465.70 |            99.32 |       83.0% |             77.3% |       4.9% |310,296,688 |
| U5      |   4585.81 |     651.09 |   3789.20 |     144.87 |     3280.82 |           231.95 |       82.6% |             76.6% |       5.1% |823,853,836 |

# CUDA Backend Breakdown

| Dataset | Candidates  | Block Size | CUDA Calls | H2D (s) | Kernel (s) | D2H (s) | Other CUDA Overhead (s) | CUDA Total (s) | Total Time (s) | CUDA Share |
| ------- | ----------- | ---------- | ---------- | ------- | ---------- | ------- | ----------------------: | -------------- | -------------- | ---------- |
| FA      | 11,713      | 64         | 168        | 0.004   | 0.020      | 0.002   |                   0.011 | 0.037          | 1.87           | 2.0%       |
| EM      | 359,890     | 64         | 2,774      | 0.022   | 0.188      | 0.027   |                   0.082 | 0.319          | 3.99           | 8.0%       |
| AM      | 299,875     | 64         | 28,148     | 0.162   | 0.670      | 0.267   |                   0.724 | 1.822          | 10.35          | 17.6%      |
| DB      | 380,762     | 64         | 26,025     | 0.157   | 0.842      | 0.248   |                   0.676 | 1.922          | 10.11          | 19.0%      |
| YO      | 13,548,785  | 64         | 115,821    | 0.652   | 5.814      | 1.106   |                   2.978 | 10.550         | 68.90          | 15.3%      |
| RN      | 862,286     | 64         | 229,476    | 1.139   | 2.606      | 2.142   |                   5.625 | 11.512         | 62.36          | 18.5%      |
| LJ      | 4,175,972   | 64         | 336,759    | 3.231   | 27.546     | 3.302   |                  10.166 | 44.245         | 299.03         | 14.8%      |
| HO      | 7,659,141   | 64         | 40,442     | 2.256   | 18.694     | 0.449   |                   4.248 | 25.648         | 263.78         | 9.7%       |
| OR      | 1,801,651   | 64         | 136,931    | 7.302   | 46.504     | 1.658   |                  13.895 | 69.360         | 581.63         | 11.9%      |
| U2      | 310,296,688 | 64         | 922,330    | 8.474   | 52.065     | 9.485   |                  29.206 | 99.230         | 2024.26        | 4.9%       |
| U5      | 823,853,836 | 64         | 1,607,703  | 19.298  | 136.373    | 17.071  |                  59.048 | 231.791        | 4585.81        | 5.1%       |

# Paper Evidence Registry

This registry controls the claims that may be derived from the experimental results in this file. Numerical results reported in the paper must originate from the final tables in `experiments.md`. The evidence identifiers below provide correctness, provenance, scope, and limitation information; they do not override the final experimental numbers.

| ID   | Paper claim                                                  | Support                                     | Primary evidence                                           | Scope and limitation                                         | Intended paper use                                      |
| ---- | ------------------------------------------------------------ | ------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------- |
| C1   | The integer local merge gain equals the independently recomputed change in global MAGS-compatible partition cost. | Direct                                      | `E10_COST_ORACLE_UNIT`                                     | Verified for every legal merge in 500 fixed-seed random small graphs and named graph families. This is empirical exhaustive validation over the tested graphs, not a formal proof for every implementation backend. | Correctness table and Method/Evaluation text            |
| C2   | The exact objective is consistent with the produced lossless correction-set summary. | Direct                                      | `E11_COST_ORACLE_INTEGRATION`, `E20_PERSISTENT_INITIAL`    | Payload cost, `ExactPartitionCost`, and the reported MAGS-compatible cost agree on seven named small graphs under two CPU merge modes. Lossless reconstruction passes. Persistent updates additionally agree with full rebuilds over randomized merge sequences. | Combined with C1 in one compact correctness table       |
| C3   | Persistent quotient state eliminates repeated original-CSR row reconstruction after initialization. | Direct                                      | `E01_STAGE0_PROFILE`, `E20_PERSISTENT_INITIAL`             | Legacy profiling observes 16.461–19.653 full-graph-equivalent prepare scans across seven datasets. Persistent prepare reports zero post-initialization original-edge scans. Final encoding may still access the original graph. | One row in the persistent-state evaluation              |
| C4   | Zero-copy quotient-row views eliminate prepared-row materialization in the persistent CPU path. | Direct                                      | `E21_ZERO_COPY`                                            | Persistent profiling reports zero copied row entries and zero copied row bytes. This does not imply that all CPU or CUDA data movement is eliminated. | Combined with C3; detailed paired results in supplement |
| C5   | Exact validation is necessary before a proxy-retained pair can be accepted as a beneficial merge. | Partial but sufficient for the stated claim | `E01_STAGE0_PROFILE`                                       | In the audited Facebook run, 289,013 retained pairs were exact-scored and 99,331 had positive exact gain. This is a single-dataset retained-set observation, not a complete proxy-ranking audit. | One short paragraph; no standalone large table          |
| C6   | Reducing the number of exact candidate evaluations does not by itself preserve final compression quality. | Direct within the audited ablation          | `E31_QN_ONLINE_FA`, `E33_RESIDUAL_ONLINE_FA`               | On Facebook, experimental bounded candidate paths reduced exact calls but produced worse final MAGS-compatible cost. The result does not establish failure on every dataset or prove that the default proposal path is globally optimal. | One row in the negative-ablation table                  |
| C7   | Better offline local-ranking diagnostics do not necessarily produce a better online final partition. | Direct within the combined audit scope      | `E32_RESIDUAL_CHECKPOINT_AUDIT`, `E34_SEQUENCE_DIVERGENCE` | Offline checkpoint diagnostics and the online Facebook run have different measurement scopes. They jointly demonstrate that improved local diagnostics need not translate into improved final quality, but not that the exact-best candidate was retained at every production epoch. | One row in the negative-ablation table                  |
| C8   | Current-state transactional commit policies do not materially improve final cost under the fixed candidate graph audited on Facebook and Email-Enron. | Provisional / scope-bounded                 | `E50_COMMIT_FA`, `E51_COMMIT_EM`                           | Best observed improvements are 0.0117% on Facebook and 0.0102% on Email-Enron. The conclusion is limited to the tested policies and fixed-candidate setting. Upgrade this row to direct support only when raw logs, command, commit, and machine provenance are recorded in `experiments.md`. | One row in the negative-ablation table or supplement    |
| C9   | CUDA is an optional implementation backend rather than the core BEAM algorithmic contribution. | Scope statement                             | historical CUDA evidence and implementation records        | Existing CPU and CUDA result sets do not use a matched objective and merge trajectory. They cannot support a pure same-output backend speedup claim. | Implementation discussion or supplement only            |
| C10  | The current evaluation does not establish a matched end-to-end Pareto comparison against MAGS-DM. | Boundary statement                          | experiment provenance audit                                | Existing historical MAGS-DM and BEAM measurements differ in environment, code state, or provenance. They may motivate analysis but must not be used to derive controlled speedups. | Evaluation scope and Limitations                        |

## Claim-use rules

1. Final runtime, compression ratio, speedup, and per-dataset values must be taken from the finalized tables in `experiments.md`.
2. Evidence from different machines, commits, objectives, or algorithm paths must not be combined into a new aggregate result.
3. Evidence marked `Partial`, `Provisional`, or `Scope statement` must retain its dataset and configuration qualification in the paper.
4. Experimental CandidateIndex ranking results must not be attributed to the default EA-proxy path.
5. Work-counter elimination must be reported separately from runtime improvement.
6. Historical CUDA measurements must not be presented as a matched CPU-versus-CUDA backend comparison.
7. Full evidence paths, raw logs, commands, and derivation details remain in `docs/paper_evidence_inventory.md` and the artifact.

# MAGS-DM 历史对比（仅供参考）

| Dataset     | MAGS-DM 时间/s | BEAM 时间/s | BEAM 慢多少 | MAGS-DM ratio | BEAM compatible ratio |
| ----------- | -------------- | ----------- | ----------- | ------------- | --------------------- |
| facebook    | 0.158          | 0.658       | 4.16×       | 0.4949        | 0.4837                |
| email_enron | 0.234          | 3.605       | 15.41×      | **0.6678**    | 0.6908                |
| amazon      | 1.162          | 19.617      | 16.88×      | **0.5728**    | 0.5804                |
| dblp        | 1.162          | 17.606      | 15.15×      | **0.4943**    | 0.5154                |
| youtube     | 4.411          | 90.543      | 20.53×      | **0.6918**    | 0.7294                |
| roadnet     | 6.009          | 158.806     | 26.43×      | **0.6849**    | 0.6941                |
| livejournal | 31.501         | 402.487     | 12.78×      | **0.7413**    | 0.7472                |
| hollywood   | 24.001         | 280.534     | 11.69×      | **0.5217**    | 0.5352                |
| orkut       | 80.989         | 674.630     | 8.33×       | **0.8608**    | 0.8699                |
| uk-2002     |                | 2502.728    |             | 0.0985        | 0.1130                |
| uk-2005     |                | 28243.931   |             | 0.0766        | 0.0937                |
