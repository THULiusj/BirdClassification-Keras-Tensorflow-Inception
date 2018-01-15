## 鸟图片种类识别

### 实验背景
该实验是基于Caltech-UCSD Birds-200-2011 (CUB-200-2011)这一鸟类图片数据库，使用Transfer Learning(迁移学习)和InceptionV3网络，在Azure数据科学虚拟机里利用预装的Keras和Tensorflow框架完成鸟种类的识别。

### 实验环境
- Azure数据科学虚拟机：NC6类型，Ubuntu
- Keras 2.0.9
- Tensorflow 1.4.0
- Anaconda Python 3.5.2 with CUDA 8.0

#### 1. 虚拟机准备
1) 创建Azure 数据科学虚拟机
在全球版Azure的管理门户上搜索Azure数据科学虚拟机(Data Science Virtual Machine)，开始创建。注意选择Ubuntu系统，HDD磁盘类型，NC6型号虚拟机。创建成功后，通过Putty连接虚拟机。

3) 训练环境配置
配置Keras默认后台为Tensorflow。也可以在/home/<username>/.keras里面找到keras.json文件。
如果使用Python环境运行代码，运行source activate py35，启动虚拟环境。

## 数据说明
数据是来自[Caltech-UCSD Birds-200-2011 (CUB-200-2011)](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)，包含200种鸟类的图片，可以用于进行鸟的识别、部位检测和图像分割。[下载](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz)
<img src="image/CUBshot.png" width="600" height="280" />

## 参考来源
-  https://github.com/DeepLearningSandbox/DeepLearningSandbox/blob/master/transfer_learning

## 代码说明
- dataprocess.py: 数据预处理，拆分训练集和测试集。
- train.py: 数据增强，模型训练。
        
   - 选用50%数据训练时，训练集Accuracy约为96%，验证Accuracy约为70%
   - 所有数据集都作为训练集时，Accuracy约为91%
- test.py: 基于训练好的模型进行网络图片测试。运行方式: python test.py "Internet img url"

