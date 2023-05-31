# Unsupervised Learning
----

As a unsupervised learning algorithm we used the [DeepCluster](https://github.com/facebookresearch/deepcluster) by Caron et al.

We cloned the original repository and made some changes to make it easier to use for our purposes. One of these is how the output is stored. We save the final output as a csv and at each epoch we save the features generated in a pkl to use for further processing ( i.e. die analysis).
