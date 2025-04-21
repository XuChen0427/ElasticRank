# ElasticRank in SIGIR'25

## Xu Chen, Ph.D. student of Renming University of China, GSAI
Any question, please mail to xc_chen@ruc.edu.cn or chenxu0427ruc@gmail.com


### Paper: Understanding Accuracy-Fairness Trade-offs in Re-ranking through Elasticity in Economics

Running command

```
python main.py --train_config_file=train_Reranking.yaml
```

The running parameters are stored in train_Reranking.yaml, you can changing your own parameters. The testing results are stored in log\

### Run in unified benchmark
Please note that our method can be also run in the unified benchmark [FairDiverse](https://github.com/XuChen0427/FairDiverse/), which contains over __30__ fair-aware and diversity-aware models that can be fairly compared.


### Citation

please cite the following bib
```
@inproceedings{xu2025elasticity,
  author       = {Chen Xu and Jujia Zhao and Wenjie Wang and Liang Pang and Jun Xu and Tat-Seng Chua and Maarten de Rijke},
  title        = {Understanding Accuracy-Fairness Trade-offs in Re-ranking through Elasticity in Economics},
  booktitle    = {Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '25)},
  year         = {2025},
  isbn         = {979-8-4007-1592-1},
  publisher    = {Association for Computing Machinery},
  address      = {Padua, Italy},
  month        = {July},
  pages        = {},
  doi          = {10.1145/3726302.3730106},
  url          = {https://doi.org/10.1145/3726302.3730106},
}
```


