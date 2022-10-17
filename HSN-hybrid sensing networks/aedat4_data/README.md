# The dataset folder

please put the aedat4_data here.

We capture a turning disk dataset by DAVIS240. The two steams are the same turning disk with different rolling speed.

You can download the toy dataset on the Tsinghua Netdisk link, https://cloud.tsinghua.edu.cn/d/59c05f980929412480e1/.

Please use the "dvSave-2020_07_23_10_28_47.aedat4" as the trainset, but the "dvSave-2020_07_23_10_28_03.aedat4" as the testset.

* Notebly, these two sequences are just a toy dataset, which can't driectly using in the real and complex scenario. They are just the simplest case to help us understand how the code works. The advantage of such a simple data set is that it is intuitive and easy to train, but the disadvantage is that the training set and the test set are captured in the same scene with different configuration, and there is an overfitting problem. To real applications, please retarin the model with NFS http://ci2cv.net/nfs/index.html or Clevrer dataset http://clevrer.csail.mit.edu/. At the same time, due to copyright reasons, please go to their official website to download the dataset.
