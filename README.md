# Hybrid-neural-networks

Code of paper "A Framework for the General Design and Computation of Hybrid Neural Networks".

You can download our paper at https://www.nature.com/articles/s41467-022-30964-7

## HSN: hybrid sensing network now is available

After running the code, you may get the following results. From the original dataset, the background is APS, the red points are DVS outputs. The green box is the ground truth of objects and the blue box is the HSN outputs. 
You may find that traditional frame-based imagers can only track objects with a larger time interval, leading to poor spatial and temporal consistency. 

We capture a turning disk dataset by DAVIS240. The two steams are the same turning disk with different rolling speed.

You can download the toy dataset on the Tsinghua Netdisk link, https://cloud.tsinghua.edu.cn/d/59c05f980929412480e1/.

Please use the "dvSave-2020_07_23_10_28_47.aedat4" as the trainset, but the "dvSave-2020_07_23_10_28_03.aedat4" as the testset.

* Notebly, these two sequences are just a toy dataset, which can't driectly using in the real and complex scenario. They are just the simplest case to help us understand how the code works. The advantage of such a simple data set is that it is intuitive and easy to train, but the disadvantage is that the training set and the test set are captured in the same scene with different configuration, and there is an overfitting problem. To real applications, please retarin the model with NFS http://ci2cv.net/nfs/index.html or Clevrer dataset http://clevrer.csail.mit.edu/. At the same time, due to copyright reasons, please go to their official website to download the dataset.

![demo_video](https://user-images.githubusercontent.com/18552022/193256636-4ca90f78-d832-4bfd-8d44-2980f740ba75.gif)

## HMN: hybrid modulation network now is available

## HRN: hybrid reasoning network now is available
dataset: [CLEVRER](http://clevrer.csail.mit.edu/).

The visual representations are extracted by RCNN, and then transformed by HU. Each ".pkl" file in the folder "./HRN/data/HU_final/" indicates a video.

The reasoning program are interpreated by SEM, stored in "./HRN/data/programs.pkl".

When running "./HRN/run.py", firstly an HU transform the visual representation into network connections and weights, with the help of "./HRN/executor.py" and "./HRN/simulation.py", which are altered codes of part of [NS-VQA](https://github.com/kexinyi/ns-vqa); secondly another HU transform programs into spiking trains to stimulate the network for reasoning process; lastly a simple read-out program detect the final spiking pattern of the network to provide the answer in text form. 

## Citation

If any part of our paper and code is helpful to your work, please generously cite with:

```
@article{zhao2022framework,
  title={A framework for the general design and computation of hybrid neural networks},
  author={Zhao, Rong and Yang, Zheyu and Zheng, Hao and Wu, Yujie and Liu, Faqiang and Wu, Zhenzhi and Li, Lukai and Chen, Feng and Song, Seng and Zhu, Jun and others},
  journal={Nature communications},
  volume={13},
  number={1},
  pages={1--12},
  year={2022},
  publisher={Nature Publishing Group}
}
```
