# Hybrid-neural-networks

Code of paper "A Framework for the General Design and Computation of Hybrid Neural Networks".

You can download our paper at https://www.nature.com/articles/s41467-022-30964-7

## HSN: hybrid sensing network now is available
This is an initial version, we will update the dataset and pre-trained model soon...

After running the code, you may get the following results. The green box is the ground truth and the blue box is the HSN output. 
You may find that traditional frame-based imagers can only track objects with a larger time interval, leading to poor spatial and temporal consistency. 


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
