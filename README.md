Dataset Description
# FB15k-237: 310116 triplets, 14541 entities, 237 relations.
# entities - 2500/1000/1500 for train/validate/test
# associated triplets - 72065/6246/9867

问题1
1.验证集中和测试集中的entity是否出现在训练集的entity中的triplets中？
答：没有重叠部分。
2.预训练的entity和relation需要在训练的过程中fine-tune吗？
3.使用support set还是query set去学校unseen entity的embeddings?
4.kl散度的值和loss之间差别很大（量级），正常吗？
5.Meta-test阶段的z只加在unseen的entity上吗？如何利用隐变量z是一个问题
6.head和tail节点编码隐变量z能够用一个线性层？


To run the experiments on FB15k-237 dataset
```python
bash run_fb15k.sh
```