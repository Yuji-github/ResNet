# ResNet
Cifar10 with ResNet

Need at least 12 GB RAM <br>
Highly Recommend: Running at Google Colab <br>

This layer has only 18 layers so far.
If you can up to 34 layers, add this code between 2nd layer and fullconnections <br> 

# 34 layers <br>
# 3rd Identity block layer <>br
model = resnet.identityBlock(model, 64)
<br>

*You might handle over fitting when you add more layers
Dropout, Pool can reduce over fitting somehow
