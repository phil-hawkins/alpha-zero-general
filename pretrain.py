import numpy as np
import os.path
from pickle import Unpickler
from random import shuffle
from absl import app, flags, logging
from absl.flags import FLAGS

from hex.matrix_hex_game import MatrixHexGame
from hex.graph_hex_game import GraphHexGame
from hex.graph_hex_board import GraphHexBoard
from hex.NNet import NNetWrapper as NNet
from utils import dotdict, config_rec

"""
Pretrains a new network using an examle file from self-play
"""

flags.DEFINE_enum('board_type', 'hex',
    ['hex', 'vortex'],
    'hex: standard hex grid Hex board, '
    'vortex: random grap board played on a Voronoi diagram (not implemented yet!!!)')
flags.DEFINE_string('pretrain_dir', 'temp/pretrain', 'pretrained weights save path')
flags.DEFINE_string('checkpoint_file', None, 'pretrained weights save path')
flags.DEFINE_string('nnet', 'base_gat', 'type of neural net to pre-train for p,v estimation')
flags.DEFINE_string('example_file', 'hex6x6.pth.tar.examples', 'path to example file to use in pre-training')
flags.DEFINE_integer('game_board_size', 6, 'overide default size')


def main(_argv):
    g = MatrixHexGame(FLAGS.game_board_size, FLAGS.game_board_size)
    nnw = NNet(g, net_type=FLAGS.nnet)
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
        nnw.train(train_examples, FLAGS.pretrain_dir)
        checkpoint_file = (FLAGS.nnet + '.chk') if FLAGS.checkpoint_file is None else FLAGS.checkpoint_file
        logging.info('Saving best checkpoint to : {}'.format(checkpoint_file))
        nnw.save_checkpoint(folder=FLAGS.pretrain_dir, filename=checkpoint_file)

if __name__ == '__main__':
    app.run(main)
