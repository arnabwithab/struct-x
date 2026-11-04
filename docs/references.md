# Struct-X references (canonical — supersedes inline lists)

Project: **Struct-X: Physics-Aware Event Vision for Robust Structural
Vibration Monitoring** — Arnab Mandal, Department of Computer Science and
Engineering, am483@snu.edu.in. Keywords: event cameras, non-contact
vibration measurement, structural health monitoring, physics-informed
neural networks, spatiotemporal Transformers, domain shift, robustness.
Acknowledgment (carries into the final paper): XJTU-DV is open source from
Xi'an Jiaotong University (Li et al., CJME 2026) via Zenodo 17447083 —
rig, LDV ground truth, and example loaders courtesy of the authors.

## Core (design §14, link status verified 2026-10-23 unless noted)

1. Gallego et al., Event-based vision: A survey, TPAMI 44(1), 2022. (Q1)
   https://doi.org/10.1109/TPAMI.2020.3008413 — LIVE.
2. Gehrig & Scaramuzza, Event-based vision meets computer vision, TPAMI 45(10), 2023.
3. Rebecq et al., High speed and HDR video with an event camera, CVPR 2017.
4. Wu et al., Event-based visual deformation measurement, arXiv:2602.14376, 2026.
   https://arxiv.org/abs/2602.14376 — LIVE (all three numbers PDF-verified).
5. Raissi et al., PINNs, J. Comput. Phys. 378, 2019. (Q1)
   https://doi.org/10.1016/j.jcp.2018.10.045 — LIVE.
6. Wang et al., Structural damage identification by physics-guided residual NN,
   Eng. Struct. 318:118703, 2024. (Q1)
   https://doi.org/10.1016/j.engstruct.2024.118703 — LIVE.
7. Wu/Tang et al., Accurate structural displacement reconstruction from
   acceleration + vision via PINNs, MSSP 235:112961, 2025. (Q1)
   https://doi.org/10.1016/j.ymssp.2025.112961 — LIVE.
8. Zhou & Lai, Structural damage classification under varying environments via
   open-set DA, MSSP 218:111561, 2024. (Q1)
   https://doi.org/10.1016/j.ymssp.2024.111561 — LIVE.
9. Bertasius et al., Is space-time attention all you need for video
   understanding? ICML 2021. (A*)
   https://proceedings.mlr.press/v139/bertasius21a.html — LIVE;
   code https://github.com/facebookresearch/TimeSformer — LIVE (archived,
   CC-NC 4.0, research-only; deployment path uses the ResNet control).
10. Li et al., XJTU-DV, Chinese J. Mech. Eng. 39:100163, 2026. (Q1)
    https://doi.org/10.1016/j.cjme.2025.100163 — LIVE.
11. He et al., Deep residual learning, CVPR 2016.
12. Karniadakis et al., Physics-informed ML, Nature Rev. Phys. 3, 2021.
13. iniVation DAVIS346 MONO €4,400. https://shop.inivation.com/products/davis346
    — LIVE (price + 346×260 specs confirmed 2026-10-23).
14. PCB accelerometers (indicative $427–$3,840 US catalogue snapshot).
    https://www.pcb.com/sensors-for-test-measurement/accelerometers — LIVE
    (category page; exact prices unverified, treat as indicative).
15. Polytec OFV-5000 set $12,728. https://www.nagw.com/product/20260609/356607.html
    — **DEAD (404 on 2026-10-23, UNVERIFIED).**
16. Polytec scanning 500k–1M RMB band corroborated by
    https://ospmotiongo.com/archives/4004 — LIVE (OFV-505/5000 class guide).
17. Aaral Tech (Polytec India distributor). https://www.aaraltech.com — LIVE.

## Data & loaders

- Dataset: Zenodo 17447083, XJTU-DV-Beam.zip 1.7 GB, md5
  d3af5b54a227be70fa932d93323d5920, CC-BY-4.0.
  https://zenodo.org/records/17447083 — LIVE.
- Loader mirror: https://gr.xjtu.edu.cn/web/lixiang/xjtu-dv — LIVE.
  Contact lixiang@xjtu.edu.cn.

## Added during the build (F011–F019)

- Kay et al., The Kinetics human action video dataset, arXiv:1705.06950, 2017.
  Pretrained TimeSformer weights via HuggingFace; URL+SHA pinned in
  `src/models/timesformer.py` (bdb264…, live at fetch 2026-10-26).
- Hu, Delbruck et al., v2e: From video frames to realistic DVS events,
  CVPRW 2021 (SensorsINI, MIT). https://github.com/SensorsINI/v2e — LIVE.
  Used for the F018 threshold/noise sweep (augmentation-grade only).
- Event2Audio (Cai et al. 2025, MIT code). https://github.com/dgalor/Event2Audio
  — LIVE (code); data Box link 404s (unshared/dead, 2026-11-04) — F019 blocked.
- Pedregosa et al., Scikit-learn: Machine learning in Python, JMLR 12, 2011.
  RidgeCV baseline (F013).

## Dead at verification (do not cite as live)

- `kaggle.com/docs/gpu`, NAG Polytec URL (#15 above).
