import logging
from absl import app, flags
from absl.flags import FLAGS
import coloredlogs
from torch import multiprocessing as mp
import pyximport; pyximport.install()
import os

from fasta0.Coach import Coach
from hex.matrix_hex_game import MatrixHexGame
from hex.graph_hex_game import GraphHexGame
from hex.NNet import NNetWrapper as NNet
from utils import dotdict

flags.DEFINE_integer('numIters', 1, 'Number of training iterations to run')
flags.DEFINE_float('learning_rate', 1e-4, 'network learning rate')
flags.DEFINE_integer('train_steps_per_iteration', 1000, 'Number of training epochs to run')
flags.DEFINE_string('job_id', 'testrun', 'job identifier from the batch system.  Used in tagging the logs')
flags.DEFINE_integer('process_batch_size', 2, 'network training batch size')
flags.DEFINE_integer('train_batch_size', 128, 'network training batch size')
flags.DEFINE_boolean('aggrgate_process_batches', False, 'collate process batches to as close as possible to the training batch size')
flags.DEFINE_float('temp', 3.0, 'tempeature for first episode, reduces to 1.0 until tempThreshold')
flags.DEFINE_integer('tempThreshold', 30, 'temp is a function of start_temp and episodeStep if episodeStep < tempThreshold, and thereafter uses temp=0.')
flags.DEFINE_float('updateThreshold', 0.6, 'During arena playoff, new neural net will be accepted if threshold or more of games are won')
flags.DEFINE_integer('maxlenOfQueue', 200000, 'Number of game examples to train the neural networks')
flags.DEFINE_integer('numMCTSSims', 200, 'Number of games moves for MCTS to simulate')
flags.DEFINE_integer('num_fast_MCTS_sims', 100, 'Number of games moves for MCTS to simulate')
flags.DEFINE_integer('arenaCompare', 40, 'Number of games to play during arena play to determine if new net will be accepted')
flags.DEFINE_float('cpuct', 1, 'constant multiplier for predictor + Upper confidence bound for trees (modified from PUCB in http://gauss.ececs.uc.edu/Conferences/isaim2010/papers/rosin.pdf)')
flags.DEFINE_integer('game_board_size', 5, 'overide default size')
flags.DEFINE_string('nnet', 'base_gat', 'neural net for p,v estimation')
flags.DEFINE_integer('numItersForTrainExamplesHistory', 100, 'Number of training iterations to keep examples for')

flags.DEFINE_boolean('load_model', False, 'load model and training examples from checkpoint')
flags.DEFINE_string('load_folder', './temp/checkpoints', 'load model from folder')
flags.DEFINE_string('load_file', 'best.pth.tar', 'load model from file')
flags.DEFINE_string('examples_file', None, 'load examples from file')
flags.DEFINE_integer('start_iteration', 0, 'Iteration to start training at')
flags.DEFINE_boolean('compare_only', False, 'just do a comparison with the past checkpoint, no training')


log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.


def setup_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def main(_argv):
    out_dir = os.path.join('./selfplay', FLAGS.nnet)
    args = dotdict({
        'run_name': os.path.join(FLAGS.nnet, FLAGS.job_id),
        'workers': mp.cpu_count() - 1,
        'startIter': FLAGS.start_iteration,
        'numIters': FLAGS.numIters,
        'process_batch_size': FLAGS.process_batch_size,
        'train_batch_size': FLAGS.train_batch_size,
        'aggrgate_process_batches': FLAGS.aggrgate_process_batches,
        'train_steps_per_iteration': FLAGS.train_steps_per_iteration,
        # should preferably be a multiple of process_batch_size and workers
        'gamesPerIteration': FLAGS.process_batch_size * (mp.cpu_count()-1),
        'numItersForTrainExamplesHistory': FLAGS.numItersForTrainExamplesHistory,
        'symmetricSamples': True,
        'numMCTSSims': FLAGS.numMCTSSims,
        'numFastSims': FLAGS.num_fast_MCTS_sims,
        'probFastSim': 0.75,
        'tempThreshold': FLAGS.tempThreshold,
        'temp': FLAGS.temp,
        'compareWithRandom': False,
        'arenaCompareRandom': 500,
        'arenaCompare': FLAGS.arenaCompare,
        'arenaTemp': 0.1,
        'arenaMCTS': False,
        'randomCompareFreq': 1,
        'compareWithPast': True,
        'pastCompareFreq': 1,
        'expertValueWeight': dotdict({
            'start': 0,
            'end': 0,
            'iterations': 35
        }),
        'cpuct': 3,
        'checkpoint': os.path.join(out_dir, 'checkpoint'),
        'data': os.path.join(out_dir, 'data')
    })
    setup_dir('./selfplay')
    setup_dir(out_dir)
    setup_dir(args.checkpoint)
    setup_dir(args.data)

    log.info('Config initialised')
    log.info('\tWorkers {}'.format(args.workers))
    log.info('Loading %s...', GraphHexGame.__name__)
    g = MatrixHexGame(FLAGS.game_board_size, FLAGS.game_board_size)
    log.info('Loading %s...', NNet.__name__)
    nnet = NNet(game=g, net_type=FLAGS.nnet, lr=FLAGS.learning_rate, epochs=25, batch_size=FLAGS.train_batch_size)

    if FLAGS.load_model:
        log.info('Loading pretrained checkpoint "%s/%s"...', FLAGS.load_folder, FLAGS.load_file)
        nnet.load_checkpoint(FLAGS.load_folder, FLAGS.load_file)
    else:
        log.warning('Not loading a pretrained checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if FLAGS.compare_only:    
        log.info('Comparison only')
        # the coach sets args.startIter to the number of checkpoints so we need to
        # look at startIter-1 to get the last checkpoint
        c.compareToPast(c.args.startIter-1)
    else:
        log.info('Starting the learning process')
        c.learn()


if __name__ == '__main__':
    app.run(main)
