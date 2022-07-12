# README

## 项目地址

colab:

+ 训练用：https://colab.research.google.com/drive/1BmWJFEj0IPEnALF63qISiw2tz4Ln8vsd?usp=sharing
+ 测试用：https://colab.research.google.com/drive/1SK9lRET5PrF4c5TqzE2OWp0nQ5I4_Vqw?usp=sharing

github: 

+ 项目地址: https://github.com/JasonXQH/AI_lab5_multiple_emotion_recognition

Huggingface

+ 已经训练完的checkpoint: https://huggingface.co/JasonXu/multimodel_emotion_recognize_with_bert_and_resnet/tree/main

预测用数据集

+ https://github.com/JasonXQH/AI_lab5_multiple_emotion_recognition/raw/main/content/drive/Mydrive/lab5_data/test_dataset.pt

如果想看完整的Wandb记录的数据图像，请发邮件给我 xuqihang74@gmail.com ，我可以邀请您加入Wandb Team查看

## Setup

可以直接使用如下指令安装实验所需要的各个依赖库

> nltk                          3.7<br>hf_hub_lightning     0.0.2<br>numpy                         1.21.6<br>
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
.
├── README.md
├── content
│   └── drive
│       └── Mydrive
│           └── lab5_data
│               ├── img_with_tags.csv			#	图片和标签 csv(仅供resnet模型使用)
│               ├── img_with_tags2.csv
│               ├── txt_with_tags.csv			# 文本和标签 csv(仅供Bert模型使用)
│               ├── txt_with_tags2.csv
│               ├── new_img_txt_tags_df.csv # 图片和文本的标签csv(供融合模型使用)
│               ├── null_label_testdata.csv	# 测试样例csv(供测试使用)
│               ├── test_dataset.pt					# 总的训练集dataset(4000组)，需后续划分训练集和验证集
│               ├── train_dataset.pt				# 测试集dataset(511组)
│               ├── train.txt								# 训练样例标签
│               ├── results.txt							# 预测结果
│               ├── test_without_label.txt 	# 文本信息和标签文件
│               ├── train
│               │   ├── negative
│               │   ├── neutral
│               │   └── positive
│               ├── train2				#   训练集第二个版本(原始图片)
│               │   ├── negative
│               │   ├── neutral
│               │   └── positive
│               ├── val 				 #  验证集(图片1200x1200),包含三种不同标签、已经分类的图片文件
│               │   ├── negative
│               │   ├── neutral
│               │   └── positive
│               └── val2				
│                   ├── negative
│                   ├── neutral
│                   └── positive
├── lab5_train.ipynb
├── lab5_predict.ipynb
└── requirements.txt 							#   依赖库
```



## Run pipeline for big-scale datasets

直接复制 测试用colab文件：https://colab.research.google.com/drive/1SK9lRET5PrF4c5TqzE2OWp0nQ5I4_Vqw?usp=sharing 。然后直接运行即可

或者直接运行根目录下的 `lab5_predict.ipynb`文件，即可进行预测

如果要进行训练的话，请务必保证数据保存在`/content/drive/MyDrive/lab5_data` 中，否则会找不到文件

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