Dataset Description
- FB15k-237: 310116 triplets, 14541 entities, 237 relations.
- entities - 2500/1000/1500 for train/validate/test
- associated triplets - 72065/6246/9867

问题1
1.验证集中和测试集中的entity是否出现在训练集的entity中的triplets中？
答：没有重叠部分。
2.预训练的entity和relation需要在训练的过程中fine-tune吗？
3.使用support set还是query set去学校unseen entity的embeddings?
4.kl散度的值和loss之间差别很大（量级），正常吗？
5.Meta-test阶段的z只加在unseen的entity上吗？如何利用隐变量z是一个问题
6.head和tail节点编码隐变量z能够用一个线性层？
7.query set的batch训练留到后面大数据集再添加。

10/28
7.提前filter掉的节点如何决定？
fb13k-237的节点相关三元组分布如下（累加值）
```python
[   0,    0,    0,    0,    1,    3,   15,   44,   95,  177,  264,
    356,  444,  527,  608,  677,  755,  821,  879,  956, 1011, 1081,
    1136, 1186, 1234, 1287, 1342, 1380, 1412, 1465, 1524, 1576, 1638,
    1687, 1733, 1786, 1825, 1859, 1895, 1925, 1948, 1965, 1990, 2019,
    2041, 2068, 2091, 2121, 2145, 2167, 2184, 2199, 2215, 2234, 2253,
    2270, 2282, 2295, 2307, 2318, 2334, 2350, 2357, 2369, 2381, 2391,
    2397, 2405, 2412, 2420, 2425, 2434, 2443, 2445, 2450, 2454, 2457,
    2464, 2470, 2476, 2482, 2483, 2484, 2485, 2489, 2492, 2494, 2494,
    2494, 2494, 2497, 2499, 2499, 2499, 2500]
```


To run the experiments on FB15k-237 dataset
```python
bash run_fb15k.sh
```