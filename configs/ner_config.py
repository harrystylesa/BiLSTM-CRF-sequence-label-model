ner_config = {
    "continue": False,
    "training": True,
    "batch_size": 8,
    "class_num": 27,
    "decay": 0.85,
    "display_batch": 5,  # 每个 epoch 显示结果
    "epoch_num": 20,
    "embedding_size": 300,  # 字向量长度
    "hidden_size": 300,  # 隐含层节点数
    "input_size": 32,
    "keep_prob": 1.0,  # dropout 的概率
    "layer_num": 2,  # bi-lstm 层数
    "learning_rate": 1e-4,
    "max_epoch": 10,
    "max_len": 32,
    "max_grad_norm": 5.0,  # 最大梯度（超过此值的梯度将被裁剪）
    "max_to_keep": 5,
    "shuffle_size": 10000,
    "time_step_size": 32,
    "train_sample_size": 32890,
    "val_sample_size": 1710,
    "vocab_size": 3000,  # 样本中不同字的个数，根据处理数据的时候得到
    "train": "../data/ner/train_ner.txt.tfrecords",
    "val": "../data/ner/val_ner.txt.tfrecords",
    "test": "../data/ner/test_ner1.txt",
    "word2id": "../data/ner/word2id.csv",
    "id2label": "../data/ner/id2label.csv",
    "ckpt_dir": "../ckpt/ner",  # 模型保存目录
    "summary_dir": "../summary",  # 保存目录
    "log_dir": "../log/ner",
    "meta_file": "../ckpt/bi-lstm-cws.ckpt-10.meta",
    "reset_default_graph": False  # 重置计算图
}
