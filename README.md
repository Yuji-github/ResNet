# ResNet
Cifar10 with ResNet

Need at least 12 GB RAM <br>
Highly Recommend: Running at Google Colab <br>

This layer has only 18 layers so far.
<br> 

# 34 layers <br>
You can up to 34 layers, add this code between the 2nd layer and the fullconnections 
3rd Identity block layer <br>
model = resnet.identityBlock(model, 64)
<br>

*You might handle over fitting when you add more layers <br>
Dropout, Pool can reduce over fitting somehow
