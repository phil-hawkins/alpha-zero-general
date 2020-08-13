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
flags.DEFINE_integer('epochs', 20, 'Number of training epochs to run')
flags.DEFINE_string('job_id', 'testrun', 'job identifier from the batch system.  Used in tagging the logs')
flags.DEFINE_integer('batch_size', 128, 'network training batch size')
flags.DEFINE_integer('numEps', 100, 'Number of complete self-play games to simulate during a new iteration')
flags.DEFINE_float('temp', 3.0, 'tempeature for first episode, reduces to 1.0 until tempThreshold')
flags.DEFINE_integer('tempThreshold', 30, 'temp is a function of start_temp and episodeStep if episodeStep < tempThreshold, and thereafter uses temp=0.')
flags.DEFINE_float('updateThreshold', 0.6, 'During arena playoff, new neural net will be accepted if threshold or more of games are won')
flags.DEFINE_integer('maxlenOfQueue', 200000, 'Number of game examples to train the neural networks')
flags.DEFINE_integer('numMCTSSims', 200, 'Number of games moves for MCTS to simulate')
flags.DEFINE_integer('num_fast_MCTS_sims', 100, 'Number of games moves for MCTS to simulate')
flags.DEFINE_integer('arenaCompare', 40, 'Number of games to play during arena play to determine if new net will be accepted')
flags.DEFINE_integer('cpuct', 1, 'constant multiplier for predictor + Upper confidence bound for trees (modified from PUCB in http://gauss.ececs.uc.edu/Conferences/isaim2010/papers/rosin.pdf)')
flags.DEFINE_integer('game_board_size', 5, 'overide default size')
flags.DEFINE_string('nnet', 'base_gat', 'neural net for p,v estimation')
flags.DEFINE_string('save_prefix', 'base_gat_', 'prefix for best model save file')
flags.DEFINE_integer('numItersForTrainExamplesHistory', 100, 'Number of training iterations to keep examples for')

flags.DEFINE_boolean('load_model', False, 'load model and training examples from checkpoint')
flags.DEFINE_string('load_folder', './temp/checkpoints', 'load model from folder')
flags.DEFINE_string('load_file', 'best.pth.tar', 'load model from file')
flags.DEFINE_string('examples_file', None, 'load examples from file')
flags.DEFINE_integer('start_iteration', 1, 'Iteration to start training at')


log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.



def main(_argv):
    args = dotdict({
        'run_name': os.path.join(FLAGS.nnet, FLAGS.job_id),
        'workers': mp.cpu_count() - 1,
        'startIter': 1,
        'numIters': FLAGS.numIters,
        'process_batch_size': FLAGS.batch_size,
        'train_batch_size': FLAGS.batch_size,
        'train_steps_per_iteration': FLAGS.epochs,
        # should preferably be a multiple of process_batch_size and workers
        'gamesPerIteration': 4*FLAGS.batch_size*(mp.cpu_count()-1),
        'numItersForTrainExamplesHistory': FLAGS.numItersForTrainExamplesHistory,
        'symmetricSamples': False,
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
        'pastCompareFreq': 3,
        'expertValueWeight': dotdict({
            'start': 0,
            'end': 0,
            'iterations': 35
        }),
        'cpuct': 3,
        'checkpoint': 'checkpoint',
        'data': 'data',
    })
    # args = dotdict({
    #     'numIters': FLAGS.numIters,
    #     'numEps': FLAGS.numEps,
    #     'tempThreshold': FLAGS.tempThreshold,
    #     'updateThreshold': FLAGS.updateThreshold,
    #     'maxlenOfQueue': FLAGS.maxlenOfQueue,
    #     'numMCTSSims': FLAGS.numMCTSSims,
    #     'arenaCompare': FLAGS.arenaCompare,
    #     'cpuct': FLAGS.cpuct,

    #     'checkpoint': FLAGS.load_folder,
    #     'save_prefix': FLAGS.save_prefix,
    #     'load_model': FLAGS.load_model,
    #     'load_folder_file': (FLAGS.load_folder, FLAGS.load_file),
    #     'examples_file': FLAGS.examples_file,
    #     'numItersForTrainExamplesHistory': FLAGS.numItersForTrainExamplesHistory,

    #     'start_iteration': FLAGS.start_iteration,

    #     'gamesPerIteration': 4*128*(mp.cpu_count()-1),
    #     'process_batch_size': FLAGS.batch_size,
    #     'workers': mp.cpu_count() - 1,
    #     'run_name': os.path.join(FLAGS.nnet, FLAGS.job_id),
    #     'expertValueWeight': dotdict({
    #         'start': 0,
    #         'end': 0,
    #         'iterations': 35
    #     })
    # })

    log.info('Loading %s...', GraphHexGame.__name__)
    g = MatrixHexGame(FLAGS.game_board_size, FLAGS.game_board_size)
    log.info('Loading %s...', NNet.__name__)
    nnet = NNet(g, net_type=FLAGS.nnet, lr=FLAGS.learning_rate, epochs=FLAGS.epochs, batch_size=FLAGS.batch_size)
    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)
    log.info('Starting the learning process')
    c.learn()


if __name__ == '__main__':
    app.run(main)
