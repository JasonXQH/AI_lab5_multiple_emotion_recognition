# README

## 项目地址

colab: https://colab.research.google.com/drive/1BmWJFEj0IPEnALF63qISiw2tz4Ln8vsd?usp=sharing

github: https://github.com/JasonXQH/AI_lab5_multiple_emotion_recognition

## Setup

可以直接使用如下指令安装实验所需要的各个依赖库

> nltk                          3.7<br>
> numpy                         1.21.6<br>
> wandb                         0.12.21<br>
> pandas                        1.3.5<br>
> torch                         1.11.0+cu113<br>
> pathlib                       1.0.1<br>
> matplotlib                    3.2.2<br>
> torchvision                   0.12.0+cu113<br>
> transformers                  4.20.1<br>
> torchmetrics                  0.9.2<br>
> torchsummary                  1.5.1<br>
> opencv-python                 4.1.2.30<br>
> pytorch-lightning             1.6.4<br>

```python
pip install -r requirements.txt
```

## Repository structure

```
├── README.md
├── lab5_data
│   ├── img_with_tags.csv        # 图片路径和标签文件(有两个版本)
│   ├── img_with_tags2.csv
│   ├── new_img_txt_tags_df.csv  # 图片路径、文本信息和标签文件
│   ├── test_without_label.txt   # 文本信息和标签文件
│   ├── train										# 训练集，里面包含三类不同标签、已经分类的图片文件
│   │   ├── negative
│   │   ├── neutral
│   │   └── positive
│   ├── train.txt
│   ├── train2
│   │   ├── negative
│   │   ├── neutral
│   │   └── positive
│   ├── txt_with_tags.csv
│   ├── val										# 验证集，里面包含三种不同标签、已经分类的图片文件
│   │   ├── negative
│   │   ├── neutral
│   │   └── positive
│   ├── val2
│       ├── negative
│       ├── neutral
│       └── positive	
├── requirements.txt         # 依赖库
└── 代码											# 代码文件
    └── lab5.ipynb
```

## Run pipeline for big-scale datasets



## Reference

### APIs

+ https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
+ https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html

### Blogs

+ https://mccormickml.com/2019/07/22/BERT-fine-tuning/#11-using-colab-gpu-for-training
+ https://drivendata.co/blog/hateful-memes-benchmark/

### Papers

+ http://www2.ift.ulaval.ca/~chaib/publications/Multimodal%20Multitask%20Emotion%20Recognition%20WCMRL19.pdf
+ http://www.jfdc.cnic.cn/article/2022/1674-9480/1674-9480-4-3-131.shtml