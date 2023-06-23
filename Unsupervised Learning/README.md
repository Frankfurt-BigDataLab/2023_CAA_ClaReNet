# Unsupervised Learning
----

As a unsupervised learning algorithm we used the [DeepCluster](https://github.com/facebookresearch/deepcluster) by Caron et al. (2018)

We cloned the original repository and made some changes to make it easier to use for our purposes. One of these is how the output is stored. We save the final output as a csv and at each epoch we save the features generated in a pkl to use for further processing ( i.e. die analysis).

----
The following parameters were used for the results obtained in the publication:

After separating by size: 
  - ARCH="vgg16" 
  - LR=0.05 
  - K=100 
  - EPOCHS=400 
  - BATCH=128 

-----
After manual evaluation of the previous result: 
  - ARCH="vgg16" 
  - LR=0.05 
  - K=25 
  - EPOCHS=300 
  - BATCH=128 

----
Used to detect and separate between dies: 
  - ARCH="vgg16" 
  - LR=0.025 
  - EPOCHS=250 
  - BATCH=128 
  - CHANGE_CLUSTER=10 #change the algorihm on the last x epochs 
  - CHANGE_ALGORITHM="hierarchical_clustering" 
