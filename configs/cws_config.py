cws_config = {
    "continue": False,
    "batch_size": 16,
    "class_num": 5,
    "decay": 0.85,
    "display_batch": 5,  # 每个 epoch 显示结果
    "epoch_num": 20,
    "embedding_size": 300,  # 字向量长度
    "hidden_size": 300,  # 隐含层节点数
    "input_size": 32,
    "keep_prob": 1.0,  # dropout 的概率
    "layer_num": 2,  # bi-lstm 层数
    "learning_rate": 1e-4,
    "max_epoch": 5,
    "max_len": 32,
    "max_grad_norm": 5.0,  # 最大梯度（超过此值的梯度将被裁剪）
    "max_to_keep": 5,
    "shuffle_size": 10000,
    "time_step_size": 32,
    "train_sample_size": 32933,
    "val_sample_size": 1716,
    "vocab_size": 3000,  # 样本中不同字的个数，根据处理数据的时候得到
    "train": "../data/cws/train_cws.txt.tfrecords",
    "val": "../data/cws/val_cws.txt.tfrecords",
    "ckpt_dir": "../ckpt/cws",  # 模型保存目录
    "summary_dir": "../summary",  # 保存目录
    "log_dir": "../log",
    "meta_file": "../ckpt/bi-lstm-cws.ckpt-10.meta",
    "reset_default_graph": True  # 重置计算图
}
