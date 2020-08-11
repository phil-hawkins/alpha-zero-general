import logging
from absl import app, flags
from absl.flags import FLAGS
import coloredlogs

from Coach import Coach
from hex.HexGame import HexGame
from hex.pytorch.NNet import NNetWrapper as nn, args as nn_args
from utils import dotdict

flags.DEFINE_integer('numIters', 1, 'Number of training iterations to run')
flags.DEFINE_float('learning_rate', 1e-4, 'network learning rate')
flags.DEFINE_integer('batch_size', 64, 'network training batch size')
flags.DEFINE_integer('numEps', 100, 'Number of complete self-play games to simulate during a new iteration')
flags.DEFINE_float('start_temp', 3.0, 'tempeature for first episode, reduces to 1.0 until tempThreshold')
flags.DEFINE_integer('tempThreshold', 30, 'temp is a function of start_temp and episodeStep if episodeStep < tempThreshold, and thereafter uses temp=0.')
flags.DEFINE_float('updateThreshold', 0.6, 'During arena playoff, new neural net will be accepted if threshold or more of games are won')
flags.DEFINE_integer('maxlenOfQueue', 200000, 'Number of game examples to train the neural networks')
flags.DEFINE_integer('numMCTSSims', 500, 'Number of games moves for MCTS to simulate')
flags.DEFINE_integer('arenaCompare', 40, 'Number of games to play during arena play to determine if new net will be accepted')
flags.DEFINE_integer('cpuct', 1, 'constant multiplier for predictor + Upper confidence bound for trees (modified from PUCB in http://gauss.ececs.uc.edu/Conferences/isaim2010/papers/rosin.pdf)')
flags.DEFINE_integer('game_board_size', None, 'overide default size')
flags.DEFINE_string('nnet', 'base_gat', 'neural net for p,v estimation')
flags.DEFINE_string('save_prefix', 'base_gat_', 'prefix for best model save file')
flags.DEFINE_integer('numItersForTrainExamplesHistory', 20, 'Number of training iterations to keep examples for')

flags.DEFINE_boolean('load_model', False, 'load model and training examples from checkpoint')
flags.DEFINE_string('load_folder', './temp/', 'load model from folder')
flags.DEFINE_string('load_file', 'best.pth.tar', 'load model from file')
flags.DEFINE_string('examples_file', None, 'load examples from file')
flags.DEFINE_integer('start_iteration', 1, 'Iteration to start training at')


log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.


def main(_argv):
    args = dotdict({
        'numIters': FLAGS.numIters,
        'numEps': FLAGS.numEps,
        'tempThreshold': FLAGS.tempThreshold,
        'updateThreshold': FLAGS.updateThreshold,
        'maxlenOfQueue': FLAGS.maxlenOfQueue,
        'numMCTSSims': FLAGS.numMCTSSims,
        'arenaCompare': FLAGS.arenaCompare,
        'cpuct': FLAGS.cpuct,

        'checkpoint': FLAGS.load_folder,
        'save_prefix': FLAGS.save_prefix,
        'load_model': FLAGS.load_model,
        'load_folder_file': (FLAGS.load_folder, FLAGS.load_file),
        'examples_file': FLAGS.examples_file,
        'numItersForTrainExamplesHistory': FLAGS.numItersForTrainExamplesHistory,

        'start_iteration': FLAGS.start_iteration
    })
    if FLAGS.start_temp is not None:
        args['start_temp'] = FLAGS.start_temp
    nn_args['lr'] = FLAGS.learning_rate
    nn_args['batch_size'] = FLAGS.batch_size

    log.info('Loading %s...', HexGame.__name__)
    g = HexGame(height=FLAGS.game_board_size, width=FLAGS.game_board_size)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g, net_type=FLAGS.nnet)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process')
    c.learn()


if __name__ == '__main__':
    app.run(main)
