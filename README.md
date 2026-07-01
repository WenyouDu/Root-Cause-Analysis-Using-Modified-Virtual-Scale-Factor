# Root-Cause-Analysis-Using-Modified-Virtual-Scale-Factor
An incremental analysis method was proposed in this work to improve the interpretability of data-driven fault detection models. Without additional training, the contribution of each feature to the model decision could be quantified by using only the anomaly detection model and the current sample.
## 关于TEP_dATA
1、该数据与蒋浩天所著教材相同
2、以te结尾的为测试集，无te结尾的为训练集
3、其中d01-d021为21个故障数据，d00为正常数据
4、训练数据部分d00.dat含有500个正常采样，d01.dat-d02.data含有480个采样，从第80个采样起引入故障
5、测试数据部分均包含960个采样，对于故障数据，从第160个采样起引入故障
6、特别注意一点d00.dat中行为变量，列为采样，其余所有数据文件均用列表示变量，行表示采样
7、数据的导入可参考文件夹内的READ_TEP.ipynb脚本
