import tensorflow as tf
import sys
sys.path.append("/mnt/0DC915E60DC915E6/workspace/python/jupyter/BiLSTM-CRF-sequence-label-model/")
from data_loader.data_generator import DataGenerator
from models.bi_lstm_crf_pretrained import Bi_LSTM_crf_with_pretrained_embedding
from trainers.ner_trainer_pretrained import NERTrainer
from utils.dirs import create_dirs
from configs import ner_config_pretrained
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        config = ner_config_pretrained.ner_config
    except:
        print("missing or invalid arguments")
        exit(0)
    training = config['training']
    # create the experiments dirs
    create_dirs([config['summary_dir'], config['ckpt_dir']])
    # create tensorflow session
    session_config = tf.ConfigProto(
        log_device_placement=False,
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0,
    )
    session_config.gpu_options.allow_growth = True
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.8

    sess = tf.Session(config=session_config)

    # create your data generator
    train_data = DataGenerator(config, isval=False)
    val_data = DataGenerator(config, isval=True)

    # create an instance of the model you want
    model = Bi_LSTM_crf_with_pretrained_embedding(config)
    # create trainers and pass all the previous components to it
    trainer = NERTrainer(sess, model, train_data, val_data, config)
    # load model if exists
    model.load(sess)
    # here you train your model
    if training:
        trainer.train()
    else:
        testfile = config['test']
        trainer.test(testfile)
        pass


if __name__ == '__main__':
    main()
