import os.path
from pickle import Unpickler
from random import shuffle
from absl import app, flags, logging
from absl.flags import FLAGS
from torch.utils.tensorboard import SummaryWriter
import json

from hex.matrix_hex_game import MatrixHexGame
from hex.NNet import NNetWrapper as NNet
from utils import config_rec

"""
Pretrains a new network using an examle file from self-play
"""

flags.DEFINE_enum('board_type', 'hex',
    ['hex', 'vortex'],
    'hex: standard hex grid Hex board, '
    'vortex: random grap board played on a Voronoi diagram (not implemented yet!!!)')
flags.DEFINE_string('pretrain_dir', 'temp/pretrain', 'pretrained weights save path')
flags.DEFINE_string('summary_dir', 'temp/pretrain/logs', 'logs for pretraining')
flags.DEFINE_string('checkpoint_file', None, 'pretrained weights save path')
flags.DEFINE_string('nnet', 'base_gat', 'type of neural net to pre-train for p,v estimation')
flags.DEFINE_string('example_file', 'hex6x6.pth.tar.examples', 'path to example file to use in pre-training')
flags.DEFINE_integer('game_board_size', 6, 'overide default size')
flags.DEFINE_float('learning_rate', 1e-4, 'network learning rate')
flags.DEFINE_integer('batch_size', 64, 'network training batch size')
flags.DEFINE_integer('epochs', 10, 'Number of training epochs to run')
flags.DEFINE_boolean('cont', False, 'load checkpoint and continue training')
flags.DEFINE_string('job_id', 'testrun', 'job identifier from the batch system.  Used in tagging the logs')


def main(_argv):
    sdir = os.path.join(FLAGS.summary_dir, FLAGS.nnet, FLAGS.job_id)
    writers = {
        'train': SummaryWriter(log_dir=os.path.join(sdir, 'train')),
        'val': SummaryWriter(log_dir=os.path.join(sdir, 'validation'))
    }
    writers['train'].add_text(tag="Config", text_string=json.dumps(config_rec(), indent=4, sort_keys=True))
    g = MatrixHexGame(FLAGS.game_board_size, FLAGS.game_board_size)
    nnw = NNet(g, net_type=FLAGS.nnet, lr=FLAGS.learning_rate, batch_size=FLAGS.batch_size, epochs=FLAGS.epochs)
    checkpoint_file = (FLAGS.nnet + '.chk') if FLAGS.checkpoint_file is None else FLAGS.checkpoint_file
    if FLAGS.cont:
        logging.info('Continuing from checkpoint file: {}'.format(checkpoint_file))
        nnw.load_checkpoint(folder=FLAGS.pretrain_dir, filename=checkpoint_file)
    example_file = os.path.join(FLAGS.pretrain_dir, FLAGS.example_file)

    if not os.path.isfile(example_file):
        logging.warning(f'File "{example_file}" with trainExamples not found!')
    else:
        logging.info("File with trainExamples found - {}.\n Loading it...".format(example_file))
        with open(example_file, "rb") as f:
            train_examples_hist = Unpickler(f).load()
        logging.info('Loading done!')

        # shuffle examples before training
        train_examples = []
        for e in train_examples_hist:
            train_examples.extend(e)
        shuffle(train_examples)

        logging.info('Training:')
        nnw.train(train_examples, checkpoint_folder=FLAGS.pretrain_dir, summary_writers=writers)
        logging.info('Saving best checkpoint to: {}'.format(checkpoint_file))
        nnw.save_checkpoint(folder=FLAGS.pretrain_dir, filename=checkpoint_file)

    writers['train'].close()
    writers['val'].close()

if __name__ == '__main__':
    app.run(main)
