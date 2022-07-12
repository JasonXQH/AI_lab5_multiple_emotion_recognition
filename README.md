# README

## 项目地址

colab: https://colab.research.google.com/drive/1BmWJFEj0IPEnALF63qISiw2tz4Ln8vsd?usp=sharing

github: https://github.com/JasonXQH/AI_lab5_multiple_emotion_recognition

Huggingface(已经训练完的模型): https://huggingface.co/JasonXu/multimodel_emotion_recognize_with_bert_and_resnet/tree/main

如果想看完整的Wandb记录的数据图像，请发邮件给我 xuqihang74@gmail.com ，我可以邀请您加入Wandb Team查看

## Setup

可以直接使用如下指令安装实验所需要的各个依赖库

> nltk                          3.7<br>
> numpy                         1.21.6<br>
> wandb                         0.12.21<br>
> pandas                        1.3.5<br>
> torch                         1.11.0+cu113<br>googletrans           3.1.0a0<br>
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
│   ├── txt_with_tags.csv
│   ├── train.txt
│   ├── test_without_label.txt   # 文本信息和标签文件
│   ├── train										 # 训练集(图片1200x1200),包含三类不同标签、已经分类的图片文件
│   │   ├── negative
│   │   ├── neutral
│   │   └── positive
│   ├── val										   # 验证集(图片1200x1200),包含三种不同标签、已经分类的图片文件
│   │   ├── negative
│   │   ├── neutral
│   │   └── positive
│   ├── train2									# 训练集第二个版本(原始图片)
│   │   ├── negative
│   │   ├── neutral
│   │   └── positive
│   ├── val2										# 训练集第二个版本(原始图片)
│       ├── negative
│       ├── neutral
│       └── positive	
├── requirements.txt            # 依赖库
└── 代码											  # 代码文件
    └── lab5.ipynb
```

## Run pipeline for big-scale datasets



## Reference

### Stackoverflows

+ https://stackoverflow.com/questions/61323621/how-to-understand-hidden-states-of-the-returns-in-bertmodelhuggingface-transfo

### APIs

+ https://docs.wandb.ai/

+ https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html

+ https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
+ https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForSequenceClassification

### Blogs

+ https://drivendata.co/blog/hateful-memes-benchmark/

+ https://mccormickml.com/2019/07/22/BERT-fine-tuning/#11-using-colab-gpu-for-training
+ https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently/notebook

### Papers

+ https://aclanthology.org/2020.semeval-1.114.pdf

+ http://www.jfdc.cnic.cn/article/2022/1674-9480/1674-9480-4-3-131.shtml
+ https://scholar.smu.edu/cgi/viewcontent.cgi?article=1165&context=datasciencereview

+ http://www2.ift.ulaval.ca/~chaib/publications/Multimodal%20Multitask%20Emotion%20Recognition%20WCMRL19.pdf