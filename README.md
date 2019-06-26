# Bilateral-Multi-Perspective-Matching-for-Natural-Language-Sentences
自然语言句子匹配的学习与总结。

论文地址：https://arxiv.org/pdf/1702.03814.pdf。Bilateral Multi-Perspective Matching for Natural Language Sentences》

此论文的创新点是：进行双向匹配以及多视角匹配。
双向匹配：假设有两个句子P,Q，不止进行p->Q的匹配，还进行Q->P的匹配。
多视角匹配：建设两个d维向量v1,v2需要匹配,定义一个矩阵w, w.shape=(l x d),匹配结果m=f_m(v1,v2,w),m为一个l维的向量，向量中每一个位置k的值为:
m_k = cosine(w_k 。v1，w_k 。v2),w_k为矩阵w中第k行的数据。 
作者也指出了四种从文本中得到v1 v2的方法。具体详见论文。
作者在三个任务上评估的模型的有效性：释义识别，自然语言推理，问答选择。
其中在SNLI数据集上进行的自然语言推理任务中，准确率达到了88.8%，比之前的单项匹配模型太高了2%左右。
在TREC-QA 和 WikiQA 数据集上进行的问答任务中，通过trec eval-8.0 脚本评估性能，分别达到了0.802和0.718。


代码来自于官方作者实现。https://github.com/zhiguowang/BiMPM 。里面用到了chare_embedding和high_way网络。chare_embedding感觉和fasttex中用到的n-Gram挺像，它能帮助捕捉后缀/前缀的含义.
high_way和残差网络差不多，让部分信息直接通过，不经过神经网络，这解决了梯度消失问题，让网络更好的训练。
Google的《attention is all you need》就使用了残差网络。

