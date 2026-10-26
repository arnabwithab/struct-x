# Security policy

## Credentials

- Kaggle API credentials (`~/.kaggle/kaggle.json`, `KAGGLE_USERNAME` /
  `KAGGLE_KEY`) must never be committed, logged, or pasted into kernel code,
  metadata, or issues. Kernel logs are public — never print secrets there.
- No other secrets exist in this project (no private APIs, no paid services).

## Data and licensing

- The XJTU-DV-Beam dataset is CC-BY-4.0 (Xi'an Jiaotong University, via Zenodo
  17447083). Keep the license declaration on every dataset derived from it.
- The ported TimeSformer is CC-NC 4.0, research-only. Never ship ported
  TimeSformer weights in a deployment path; use the ResNet control instead.
- Do not commit data files, model checkpoints, or kernel outputs to git.

## Reporting a vulnerability

Contact Arnab Mandal (am483@snu.edu.in) with a description and reproduction.
Please do not open a public issue for credential leaks — rotate the credential
first, then report.
