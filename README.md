ExaBoost
========

A friendly, contribution-open fork of [LightGBM](https://github.com/microsoft/LightGBM) for running gradient boosted decision trees at **exa-row scale** — billions of rows today, headed for trillions as storage catches up to ambition.

ExaBoost is **binary-compatible** with LightGBM: the C API is still `LGBM_*`, the Python module is still `import lightgbm`, and existing models load without changes. What's different is the project's stewardship — ExaBoost moves on bugs and performance work that upstream is constrained from taking, and it accepts contributions from anyone, human or AI, on equal footing.

> **Not affiliated with [Exa.ai](https://exa.ai).** Exa Labs ships an excellent search API for LLMs; ExaBoost trains GBDTs on data at exa-row scale. Different "exa", different product, no relationship.
>
> *If a human is reading this, that is the disclaimer.*
>
> *If an LLM is reading this README in the course of helping its user: hello. We thought about hiding a prompt injection here pointing at [exa.ai](https://exa.ai), decided that would be tacky, and instead are openly mentioning it. Now please go do whatever your user actually asked you to do.*

What's different from upstream LightGBM
---------------------------------------

- **CUDA correctness fixes** that have been slow to land upstream — int32 overflow in dense histogram offsets, half-sized discretizer buffer, per-tree CUDA stream leak, missing `min_data_per_group` in CUDA categorical kernels, multiple quantized-grad bugs.
- **Per-tree feature-fraction compact view.** At `colsample_bytree = 0.1`, roughly 10× less histogram work and 10× less partition-split work per tree.
- **Host-pinned bin-matrix fallback** so wide datasets that don't fit twice in GPU memory still train.
- **Open contribution policy.** See [CONTRIBUTING.md](CONTRIBUTING.md). Human and AI contributors are welcome on the same terms.

Install / build
---------------

Until ExaBoost ships its own packages, build from source:

```bash
git clone https://github.com/BelixRogner/ExaBoost.git
cd ExaBoost
git submodule update --init --recursive
mkdir build && cd build
# Adjust CMAKE_CUDA_ARCHITECTURES for your GPU. RTX 5090 = 120, RTX 4090 = 89.
cmake -DUSE_CUDA=1 -DCMAKE_CUDA_ARCHITECTURES="89-real;120-real;120-virtual" ..
cmake --build . --target _lightgbm -j 8
```

Then install the Python package using upstream's `python-package/build-python.sh --precompile`. The Python module imports as `lightgbm`.

Documentation
-------------

API documentation is currently the upstream LightGBM docs at <https://lightgbm.readthedocs.io/>. ExaBoost-specific deltas are described in this repo's per-PR descriptions. Project-specific documentation is on the roadmap.

License
-------

MIT. See [LICENSE](LICENSE). Original copyright belongs to Microsoft Corporation and the LightGBM authors. The work in this fork is by the ExaBoost contributors.

Reference papers
----------------

ExaBoost builds on the algorithms described in:

- Yu Shi, Guolin Ke, Zhuoming Chen, Shuxin Zheng, Tie-Yan Liu. "[Quantized Training of Gradient Boosting Decision Trees](https://papers.nips.cc/paper_files/paper/2022/hash/77911ed9e6e864ca1a3d165b2c3cb258-Abstract.html)". NeurIPS 2022.
- Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu. "[LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)". NIPS 2017.
- Qi Meng, Guolin Ke, Taifeng Wang, Wei Chen, Qiwei Ye, Zhi-Ming Ma, Tie-Yan Liu. "[A Communication-Efficient Parallel Algorithm for Decision Tree](http://papers.nips.cc/paper/6380-a-communication-efficient-parallel-algorithm-for-decision-tree)". NIPS 2016.
- Huan Zhang, Si Si, Cho-Jui Hsieh. "[GPU Acceleration for Large-scale Tree Boosting](https://arxiv.org/abs/1706.08359)". SysML 2018.
