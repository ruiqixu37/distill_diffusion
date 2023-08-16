# distill_diffusion
Personal pytorch implementation of the 2023 CVPR Award Candidate: *On Distillation of Guided Diffusion Models*[paper link](https://arxiv.org/abs/2210.03142).

## Dependencies
To install the required libraries, run:
```bash
pip install -e . 
```

## Training
```
python main.py -c configs/<config-name.yaml>
```

## Notice

This repository is currently a **work in progress**. Many features are still being implemented and some features may be different from the original paper due to my ongoing understanding of its details. The code has not been fully tested yet and cannot match the original due to my hardware limitations. I will continue to improve this repository in the future. If you have any questions, please feel free to open an issue so we can discuss and make this repo better.

## Acknowledgement

This repository is based on the following repositories:
-  Overall structure planning: [AntixK/PyTorch-VAE](https://github.com/AntixK/PyTorch-VAE)
```
@misc{Subramanian2020,
  author = {Subramanian, A.K},
  title = {PyTorch-VAE},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/AntixK/PyTorch-VAE}}
}
```

- w-embedding and noise scheduler: [google-research/vdm](https://github.com/google-research/vdm)
```
@article{Kingma2021VariationalDM,
  title={Variational Diffusion Models},
  author={Diederik P. Kingma and Tim Salimans and Ben Poole and Jonathan Ho},
  journal={ArXiv},
  year={2021}
}
```

- downloading LAION datasets: [rom1504/laion-prepro](https://github.com/rom1504/laion-prepro)

