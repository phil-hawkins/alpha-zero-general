import logging
from absl import app, flags
from absl.flags import FLAGS
import coloredlogs

from Coach import Coach
from hex.HexGame import HexGame
from hex.pytorch.NNet import NNetWrapper as nn
from utils import dotdict

flags.DEFINE_integer('numIters', 1000, 'Number of training iterations to run')
flags.DEFINE_integer('numEps', 100, 'Number of complete self-play games to simulate during a new iteration')
flags.DEFINE_integer('tempThreshold', 15, '?')
flags.DEFINE_float('updateThreshold', 0.6, 'During arena playoff, new neural net will be accepted if threshold or more of games are won')
flags.DEFINE_integer('maxlenOfQueue', 200000, 'Number of game examples to train the neural networks')
flags.DEFINE_integer('numMCTSSims', 500, 'Number of games moves for MCTS to simulate')
flags.DEFINE_integer('arenaCompare', 40, 'Number of games to play during arena play to determine if new net will be accepted')
flags.DEFINE_integer('cpuct', 1, '')
flags.DEFINE_integer('game_board_height', None, 'overide default height')
flags.DEFINE_integer('game_board_width', None, 'overide default width')

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

        'checkpoint': './temp/',
        'load_model': False,
        'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
        'numItersForTrainExamplesHistory': 20,

    })    

    log.info('Loading %s...', HexGame.__name__)
    g = HexGame(height=FLAGS.game_board_height, width=FLAGS.game_board_width)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file)
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == '__main__':
    app.run(main)
