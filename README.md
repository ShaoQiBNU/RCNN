R-CNN算法解读
=============

> 参考 https://blog.csdn.net/wopawn/article/details/52133338. 

# 一. 背景

> 在RCNN之前，overfeat已经是用深度学习的方法做目标检测，但RCNN是第一个可以真正可以工业级应用的解决方案。可以说改变了目标检测领域的主要研究思路，紧随其后的系列文章：*Fast-RCNN* ，*Faster-RCNN*都沿袭*R-CNN*的思路。

# 二. 目标检测流程

> RCNN做物体目标检测主要有以下模块构成，具体如下：

## (一) Region proposals
> 采用selective search方法提取大约2000个建议框，selective search算法具体见：https://github.com/ShaoQiBNU/Selective_Search. 

## (二) Feature extraction

> 先在每个建议框周围加上16个像素值为建议框像素平均值的边框，再直接变形为227×227的大小；将所有建议框像素减去该建议框像素的平均值，再依次将每个227×227的建议框输入AlexNet CNN网络获取4096维的特征，2000个建议框的CNN特征组合成2000×4096维矩阵；

## (三) SVM分类

> 采用线性SVM分类器进行分类；将2000×4096维特征与20个SVM组成的权值矩阵4096×20相乘——20种分类，SVM是二分类器，则有20个SVM，获得2000×20维的矩阵，该矩阵表示每个建议框是某个物体类别的得分；

## (四) IoU非极大值抑制

> 分别对上述2000×20维矩阵中每一列即每一类进行非极大值抑制剔除重叠建议框，得到该列即该类中得分最高的一些建议框。

## 1. IoU

> IoU即表示(A∩B)/(AUB) ，如图所示：

![image](https://github.com/ShaoQiBNU/RCNN/blob/master/images/1.png)

> 在(三)之后，获得2000×20维矩阵表示每个建议框是某个物体类别的得分情况，此时会遇到下图所示情况，同一个车辆目标会被多个建议框包围，这时需要非极大值抑制操作去除得分较低的候选框以减少重叠框。

![image](https://github.com/ShaoQiBNU/RCNN/blob/master/images/2.png)

## 2. IoU流程
```
① 对2000×20维矩阵中每列按从大到小进行排序； 
② 从每列最大的得分建议框开始，分别与该列后面的得分建议框进行IoU计算，若IoU>阈值，则剔除得分较小的建议框，否则认为图像中存在多个同一类物体； 
③ 从每列次大的得分建议框开始，重复步骤②； 
④ 重复步骤③直到遍历完该列所有建议框； 
⑤ 遍历完2000×20维矩阵所有列，即所有物体种类都做一遍非极大值抑制； 
⑥ 最后剔除各个类别中剩余建议框得分少于该类别阈值的建议框
```

## (五) Bounding box regression

> 分别用20个回归器对上述20个类别中剩余的建议框进行回归操作，最终得到每个类别的修正后的得分最高的bounding box。

## 1. 简介
> 目标检测不仅是要对目标进行识别，还要完成定位任务，所以最终获得的bounding-box也决定了目标检测的精度。定位精度可以用算法得出的物体检测框与实际标注的物体边界框的IoU值来近似表示。如下图所示，绿色框为实际标准的卡宴车辆框，即Ground Truth；黄色框为selective search算法得出的建议框，即Region Proposal。即使黄色框中物体被分类器识别为卡宴车辆，但是由于绿色框和黄色框IoU值并不大，所以最后的目标检测精度并不高。采用回归器是为了对建议框进行校正，使得校正后的Region Proposal与selective search更接近， 以提高最终的检测精度。论文中采用bounding-box回归使mAP提高了3~4%。 

![image](https://github.com/ShaoQiBNU/RCNN/blob/master/images/3.png)

## 2. 过程 
> Bounding box regression回归过程如下，如图所示：

![image](https://github.com/ShaoQiBNU/RCNN/blob/master/images/4.png)

> 其中黄色框口*P*表示建议框Region Proposal，绿色窗口*G*表示实际框Ground Truth，红色窗口<a href="https://www.codecogs.com/eqnedit.php?latex=\widehat{G}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\widehat{G}" title="\widehat{G}" /></a>
> 表示Region Proposal进行回归后的预测窗口。目标是找到*P*到<a href="https://www.codecogs.com/eqnedit.php?latex=\widehat{G}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\widehat{G}" title="\widehat{G}" /></a>的线性变换（注意：当Region Proposal与Ground Truth的IoU>0.6时可以认为是线性变换），使得<a href="https://www.codecogs.com/eqnedit.php?latex=\widehat{G}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\widehat{G}" title="\widehat{G}" /></a>与*G*越相近，这就相当于一个简单的可以用最小二乘法解决的线性回归问题。 

> *P*窗口的数学表达式：<a href="https://www.codecogs.com/eqnedit.php?latex=P^{i}=(P_{x}^{i},&space;P_{y}^{i},&space;P_{w}^{i},&space;P_{h}^{i})" target="_blank"><img src="https://latex.codecogs.com/svg.latex?P^{i}=(P_{x}^{i},&space;P_{y}^{i},&space;P_{w}^{i},&space;P_{h}^{i})" title="P^{i}=(P_{x}^{i}, P_{y}^{i}, P_{w}^{i}, P_{h}^{i})" /></a>，其中<a href="https://www.codecogs.com/eqnedit.php?latex=(P_{x}^{i},&space;P_{y}^{i})" target="_blank"><img src="https://latex.codecogs.com/svg.latex?(P_{x}^{i},&space;P_{y}^{i})" title="(P_{x}^{i}, P_{y}^{i})" /></a>表示第一个i窗口的中心点坐标，<a href="https://www.codecogs.com/eqnedit.php?latex=P_{w}^{i},&space;P_{h}^{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?P_{w}^{i},&space;P_{h}^{i}" title="P_{w}^{i}, P_{h}^{i}" /></a>分别为第i个窗口的宽和高；G窗口的数学表达式为：<a href="https://www.codecogs.com/eqnedit.php?latex=G^{i}=(G_{x}^{i},&space;G_{y}^{i},&space;G_{w}^{i},&space;G_{h}^{i})" target="_blank"><img src="https://latex.codecogs.com/svg.latex?G^{i}=(G_{x}^{i},&space;G_{y}^{i},&space;G_{w}^{i},&space;G_{h}^{i})" title="G^{i}=(G_{x}^{i}, G_{y}^{i}, G_{w}^{i}, G_{h}^{i})" /></a>，<a href="https://www.codecogs.com/eqnedit.php?latex=\widehat{G}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\widehat{G}" title="\widehat{G}" /></a>窗口的数学表达式为：<a href="https://www.codecogs.com/eqnedit.php?latex=\widehat{G}^{i}=(\widehat{G}_{x}^{i},&space;\widehat{G}_{y}^{i},&space;\widehat{G}_{w}^{i},&space;\widehat{G}_{h}^{i})" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\widehat{G}^{i}=(\widehat{G}_{x}^{i},&space;\widehat{G}_{y}^{i},&space;\widehat{G}_{w}^{i},&space;\widehat{G}_{h}^{i})" title="\widehat{G}^{i}=(\widehat{G}_{x}^{i}, \widehat{G}_{y}^{i}, \widehat{G}_{w}^{i}, \widehat{G}_{h}^{i})" /></a>。以下省去i上标。


> 定义四种变换函数：<a href="https://www.codecogs.com/eqnedit.php?latex=d_{x}(P),&space;d_{y}(P),&space;d_{w}(P),&space;d_{h}(P)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?d_{x}(P),&space;d_{y}(P),&space;d_{w}(P),&space;d_{h}(P)" title="d_{x}(P), d_{y}(P), d_{w}(P), d_{h}(P)" /></a>，<a href="https://www.codecogs.com/eqnedit.php?latex=d_{x}(P),&space;d_{y}(P)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?d_{x}(P),&space;d_{y}(P)" title="d_{x}(P), d_{y}(P)" /></a>通过平移对x和y进行变化，<a href="https://www.codecogs.com/eqnedit.php?latex=d_{w}(P),&space;d_{h}(P)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?d_{w}(P),&space;d_{h}(P)" title="d_{w}(P), d_{h}(P)" /></a>通过缩放对w和h进行变化，即下面四个式子所示：
>
> <a href="https://www.codecogs.com/eqnedit.php?latex=\widehat{G}_{x}=P_{w}d_{x}(P)&plus;P_{x}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\widehat{G}_{x}=P_{w}d_{x}(P)&plus;P_{x}" title="\widehat{G}_{x}=P_{w}d_{x}(P)+P_{x}" /></a>
>
> <a href="https://www.codecogs.com/eqnedit.php?latex=\widehat{G}_{y}=P_{h}d_{y}(P)&plus;P_{y}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\widehat{G}_{y}=P_{h}d_{y}(P)&plus;P_{y}" title="\widehat{G}_{y}=P_{h}d_{y}(P)+P_{y}" /></a>
>
> <a href="https://www.codecogs.com/eqnedit.php?latex=\widehat{G}_{w}=P_{w}exp(d_{w}(P))" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\widehat{G}_{w}=P_{w}exp(d_{w}(P))" title="\widehat{G}_{w}=P_{w}exp(d_{w}(P))" /></a>
>
> <a href="https://www.codecogs.com/eqnedit.php?latex=\widehat{G}_{h}=P_{h}exp(d_{h}(P))" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\widehat{G}_{h}=P_{h}exp(d_{h}(P))" title="\widehat{G}_{h}=P_{h}exp(d_{h}(P))" /></a>



> 每一个<a href="https://www.codecogs.com/eqnedit.php?latex=d_{*}(P)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?d_{*}(P)" title="d_{*}(P)" /></a>（表示x，y，w，h）都是一个AlexNet CNN网络Pool5层特征<a href="https://www.codecogs.com/eqnedit.php?latex=\phi&space;_{5}(P)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\phi&space;_{5}(P)" title="\phi _{5}(P)" /></a>的线性函数，即<a href="https://www.codecogs.com/eqnedit.php?latex=d_{*}(P)=w_{*}^{T}\phi&space;_{5}(P)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?d_{*}(P)=w_{*}^{T}\phi&space;_{5}(P)" title="d_{*}(P)=w_{*}^{T}\phi _{5}(P)" /></a> ，这里<a href="https://www.codecogs.com/eqnedit.php?latex=w_{*}^{T}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?w_{*}^{T}" title="w_{*}^{T}" /></a>就是所需要学习的回归参数。损失函数即为：
>
> <a href="https://www.codecogs.com/eqnedit.php?latex=Loss&space;=&space;argmin&space;\sum_{i=0}^{N}(t_{*}^{i}-w_{*}^{T}\phi&space;_{5}(P^{i}))^{2}&plus;\lambda&space;||w_{*}||^{2}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?Loss&space;=&space;argmin&space;\sum_{i=0}^{N}(t_{*}^{i}-w_{*}^{T}\phi&space;_{5}(P^{i}))^{2}&plus;\lambda&space;||w_{*}||^{2}" title="Loss = argmin \sum_{i=0}^{N}(t_{*}^{i}-w_{*}^{T}\phi _{5}(P^{i}))^{2}+\lambda ||w_{*}||^{2}" /></a>



> 损失函数中加入正则项<a href="https://www.codecogs.com/eqnedit.php?latex=\lambda&space;||w_{*}||^{2}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\lambda&space;||w_{*}||^{2}" title="\lambda ||w_{*}||^{2}" /></a>是为了避免归回参数<a href="https://www.codecogs.com/eqnedit.php?latex=w_{*}^{T}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?w_{*}^{T}" title="w_{*}^{T}" /></a>过大。其中，回归目标<a href="https://www.codecogs.com/eqnedit.php?latex=t_{*}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?t_{*}" title="t_{*}" /></a>由训练输入对(P，G)按下式计算得来：
>
> <a href="https://www.codecogs.com/eqnedit.php?latex=t_{x}&space;=&space;(G_{x}&space;-&space;P_{x})/P_{w}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?t_{x}&space;=&space;(G_{x}&space;-&space;P_{x})/P_{w}" title="t_{x} = (G_{x} - P_{x})/P_{w}" /></a>
>
> <a href="https://www.codecogs.com/eqnedit.php?latex=t_{y}&space;=&space;(G_{y}&space;-&space;P_{y})/P_{h}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?t_{y}&space;=&space;(G_{y}&space;-&space;P_{y})/P_{h}" title="t_{y} = (G_{y} - P_{y})/P_{h}" /></a>
>
> <a href="https://www.codecogs.com/eqnedit.php?latex=t_{w}&space;=&space;log(G_{w}/P_{w})" target="_blank"><img src="https://latex.codecogs.com/svg.latex?t_{w}&space;=&space;log(G_{w}/P_{w})" title="t_{w} = log(G_{w}/P_{w})" /></a>
>
> <a href="https://www.codecogs.com/eqnedit.php?latex=t_{h}&space;=&space;log(G_{h}/P_{h})" target="_blank"><img src="https://latex.codecogs.com/svg.latex?t_{h}&space;=&space;log(G_{h}/P_{h})" title="t_{h} = log(G_{h}/P_{h})" /></a>



> 回归过程如下：
>
> ①构造样本对。为了提高每类样本框回归的有效性，对每类样本都仅仅采集与Ground Truth相交IoU最大的Region Proposal，并且IoU>0.6的Region Proposal作为样本对<a href="https://www.codecogs.com/eqnedit.php?latex=(P^{i},&space;G^{i})" target="_blank"><img src="https://latex.codecogs.com/svg.latex?(P^{i},&space;G^{i})" title="(P^{i}, G^{i})" /></a>，一共产生20对样本对——20个类别； 
> ②每种类型的回归器单独训练，输入该类型样本对N个：<a href="https://www.codecogs.com/eqnedit.php?latex=\left&space;\{&space;(P^{i},&space;G^{i})&space;\right&space;\}&space;i=1,2,...N" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\left&space;\{&space;(P^{i},&space;G^{i})&space;\right&space;\}&space;i=1,2,...N" title="\left \{ (P^{i}, G^{i}) \right \} i=1,2,...N" /></a>以及<a href="https://www.codecogs.com/eqnedit.php?latex=P^{i}\&space;i=1,2,...N" target="_blank"><img src="https://latex.codecogs.com/svg.latex?P^{i}\&space;i=1,2,...N" title="P^{i}\ i=1,2,...N" /></a>所对应的AlexNet CNN网络Pool5层特征<a href="https://www.codecogs.com/eqnedit.php?latex=\phi&space;_{5}(P^{i})\&space;i=1,2,...N" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\phi&space;_{5}(P^{i})\&space;i=1,2,...N" title="\phi _{5}(P^{i})\ i=1,2,...N" /></a>
> ③利用公式和输入样本对<a href="https://www.codecogs.com/eqnedit.php?latex=\left&space;\{&space;(P^{i},G^{i})&space;\right&space;\}\&space;i=1,2,...N" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\left&space;\{&space;(P^{i},G^{i})&space;\right&space;\}\&space;i=1,2,...N" title="\left \{ (P^{i},G^{i}) \right \}\ i=1,2,...N" /></a>计算<a href="https://www.codecogs.com/eqnedit.php?latex=t_{*}^{i}\&space;i=1,2,...N" target="_blank"><img src="https://latex.codecogs.com/svg.latex?t_{*}^{i}\&space;i=1,2,...N" title="t_{*}^{i}\ i=1,2,...N" /></a>
> ④利用<a href="https://www.codecogs.com/eqnedit.php?latex=\phi&space;_{5}(P^{i})&space;\&space;i=1,2,...N" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\phi&space;_{5}(P^{i})&space;\&space;i=1,2,...N" title="\phi _{5}(P^{i}) \ i=1,2,...N" /></a>和<a href="https://www.codecogs.com/eqnedit.php?latex=t_{*}^{i}\&space;i=1,2,...N" target="_blank"><img src="https://latex.codecogs.com/svg.latex?t_{*}^{i}\&space;i=1,2,...N" title="t_{*}^{i}\ i=1,2,...N" /></a>，根据损失函数进行回归，采用梯度下降或最小二乘法得到使损失函数最小的参数<a href="https://www.codecogs.com/eqnedit.php?latex=w_{*}^{T}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?w_{*}^{T}" title="w_{*}^{T}" /></a>

