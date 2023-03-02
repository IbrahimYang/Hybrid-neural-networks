# Hybrid-neural-networks

Code of paper "A Framework for the General Design and Computation of Hybrid Neural Networks".

You can download our paper at https://www.nature.com/articles/s41467-022-30964-7

Now, our lab updates a new trianing framework and code temporal for HNN, making us can easliy build your own HNNs model. https://github.com/openBII/HNN

## HSN: hybrid sensing network now is available

After running the code, you may get a video like the one below, with high frame rate DVS sensor data drawn above the low frame rate APS background. The green boxes in the video are ground truth target locations and blue boxes are the predictions of HSN model. 

This video is a demonstration of HSN model on a toy dataset of a rotating disk captured by ourselves using a DAVIS240 sensor. This toy dataset is free to download from the following link: https://cloud.tsinghua.edu.cn/d/59c05f980929412480e1/. In it you may find two recordings of the same disk rotating at different speeds.
In our demo, "dvSave-2020_07_23_10_28_47.aedat4" was used as the trainset and "dvSave-2020_07_23_10_28_03.aedat4" the testset.

![demo_video](https://user-images.githubusercontent.com/18552022/193256636-4ca90f78-d832-4bfd-8d44-2980f740ba75.gif)

Notebly, these two sequences are just toy datasets for demonstration and quick experiment. For a more realistic application, the HSN model was also tested on NFS http://ci2cv.net/nfs/index.html and Clevrer http://clevrer.csail.mit.edu/ dataset, which can both be downloaded from their official websites. Please see our paper for more details.

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
