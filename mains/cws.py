import tensorflow as tf

from data_loader.data_generator import DataGenerator
from models.bi_lstm_crf import Bi_LSTM_crf
from trainers.cws_trainer import CWSTrainer
from utils.dirs import create_dirs
from configs import cws_config
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        config = cws_config.cws_config
    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config['summary_dir'], config['ckpt_dir']])
    # create tensorflow session
    session_config = tf.ConfigProto(
        log_device_placement=False,
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0,
    )
    # session_config.gpu_options.allow_growth = True
    # session_config.gpu_options.per_process_gpu_memory_fraction = 0.4

    sess = tf.Session(config=session_config)

    # create your data generator
    train_data = DataGenerator(config, isval=False)
    val_data = DataGenerator(config, isval=True)

    # create an instance of the model you want
    model = Bi_LSTM_crf(config)
    # create trainers and pass all the previous components to it
    trainer = CWSTrainer(sess, model, train_data, val_data, config)
    # load model if exists
    model.load(sess)
    # here you train your model
    trainer.train()

    # trainers.test()


if __name__ == '__main__':
    main()
