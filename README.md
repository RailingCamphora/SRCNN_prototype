# 简单SRCNN网络超分辨成像

##使用说明

.pth文件是已经训练好的预训练参数，它们和SRCNN_function.py里的不同网络相对应。
验证显示图片分别为picture里的boat.png,cameraman.png,fruits.png。
验证时使用文件SRCNN_TEST_GRAY.py,根据需要验证的图片，修改图片路径；根据使用的预训练参数，修改预训练参数路径和模型的类型。
##其他说明
似乎最好的是srcnn_model2alter_lr_to_hr.pth
srcnn_modelresnet_lr_to_hr.pth是作者费大力用残差神经网络训练的，作者水平不足，效果反而不及简单的神经网络
得到的图片会有伪影，所以最后测试效果时也进行了简单的中值滤波


