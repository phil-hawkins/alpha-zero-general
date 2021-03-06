from .MCTS import MCTS
from .SelfPlayAgent import SelfPlayAgent
import torch
from glob import glob
from torch import multiprocessing as mp
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from .Arena import Arena
from GenericPlayers import RandomPlayer, NNPlayer
from utils import AverageMeter
from queue import Empty
from time import time
import numpy as np
import os
from tqdm import tqdm


class Coach:
    def __init__(self, game, nnet, args):
        np.random.seed()
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game, self.nnet.net_type)  # the competitor network
        self.args = args

        networks = sorted(glob(self.args.checkpoint+'/*'))
        # need to ignore the best-so-far checkpoint in the count of checkpoints
        self.args.startIter = len(networks) - 1
        if self.args.startIter == 0:
            self.nnet.save_checkpoint(
                folder=self.args.checkpoint, filename='iteration-0000.pkl')
            self.args.startIter = 1

        self.nnet.load_checkpoint(
            folder=self.args.checkpoint, filename=f'best.pkl')

        self.agents = []
        self.input_tensors = []
        self.policy_tensors = []
        self.value_tensors = []
        self.batch_ready = []
        self.ready_queue = mp.Queue()
        self.file_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.completed = mp.Value('i', 0)
        self.games_played = mp.Value('i', 0)
        if self.args.run_name != '':
            self.writer = SummaryWriter(log_dir='logs/'+self.args.run_name)
        else:
            self.writer = SummaryWriter()
        self.args.expertValueWeight.current = self.args.expertValueWeight.start

    def learn(self):
        print('Because of batching, it can take a long time before any games finish.')
        for i in range(self.args.startIter, self.args.numIters + 1):
            print(f'------ITER {i}------')
            self.generateSelfPlayAgents()
            self.processSelfPlayBatches()
            self.saveIterationSamples(i)
            self.processGameResults(i)
            self.killSelfPlayAgents()
            self.train(i)
            if self.args.compareWithRandom and (i-1) % self.args.randomCompareFreq == 0:
                if i == 1:
                    print(
                        'Note: Comparisons with Random do not use monte carlo tree search.')
                self.compareToRandom(i)
            if self.args.compareWithPast and (i - 1) % self.args.pastCompareFreq == 0:
                #self.compareToPast(i)
                self.compareToBest(i)
            z = self.args.expertValueWeight
            self.args.expertValueWeight.current = min(
                i, z.iterations)/z.iterations * (z.end - z.start) + z.start
            print()
        self.writer.close()

    def generateSelfPlayAgents(self):
        self.ready_queue = mp.Queue()
        boardx, boardy = self.game.getBoardSize()
        for i in range(self.args.workers):
            self.input_tensors.append(torch.zeros(
                [self.args.process_batch_size, boardx, boardy]))
            self.input_tensors[i].pin_memory()
            self.input_tensors[i].share_memory_()

            self.policy_tensors.append(torch.zeros(
                [self.args.process_batch_size, self.game.getActionSize()]))
            self.policy_tensors[i].pin_memory()
            self.policy_tensors[i].share_memory_()

            self.value_tensors.append(torch.zeros(
                [self.args.process_batch_size, 1]))
            self.value_tensors[i].pin_memory()
            self.value_tensors[i].share_memory_()
            self.batch_ready.append(mp.Event())

            self.agents.append(
                SelfPlayAgent(i, self.game, self.ready_queue, self.batch_ready[i],
                              self.input_tensors[i], self.policy_tensors[i], self.value_tensors[i], self.file_queue,
                              self.result_queue, self.completed, self.games_played, self.args))
            self.agents[i].start()

    def agg_batch(self):
        ''' collects process batches to do a larger training batch
        '''
        train_batch_ratio = self.args.train_batch_size // self.args.process_batch_size
        working_workers = self.args.workers - self.completed.value
        workers_to_batch = min(working_workers, train_batch_ratio)
        start_time = time()

        ids = []
        for _ in range(workers_to_batch):
            if time() - start_time > 1.:
                break
            try:
                ids.append(self.ready_queue.get(timeout=1))
            except Empty:
                break

        pos_processed = 0
        if len(ids) > 0:
            input_tensors = []
            for id in ids:
                input_tensors.append(self.input_tensors[id])
            self.policy, self.value = self.nnet.process(torch.cat(input_tensors, dim=0))
            start_batch_ndx = end_batch_ndx = 0
            for i, id in enumerate(ids):
                end_batch_ndx = start_batch_ndx + input_tensors[i].size(0)
                self.policy_tensors[id].copy_(self.policy[start_batch_ndx:end_batch_ndx])
                self.value_tensors[id].copy_(self.value[start_batch_ndx:end_batch_ndx])
                self.batch_ready[id].set()
                start_batch_ndx = end_batch_ndx
            pos_processed = end_batch_ndx + 1

        return pos_processed

    def single_batch(self):
        pos_processed = 0
        try:
            id = self.ready_queue.get(timeout=1)
            self.policy, self.value = self.nnet.process(
                self.input_tensors[id])
            self.policy_tensors[id].copy_(self.policy)
            self.value_tensors[id].copy_(self.value)
            self.batch_ready[id].set()
            pos_processed = self.value.size(0)
        except Empty:
            pass

        return pos_processed

    def processSelfPlayBatches(self):
        sample_time = AverageMeter()
        network_time = AverageMeter()
        bar = tqdm(desc='Generating Samples', total=self.args.gamesPerIteration)
        end = time()

        n = 0
        while self.completed.value != self.args.workers:
            loop_time = time()
            if self.args.aggrgate_process_batches:
                ex = self.agg_batch()
            else:
                ex = self.single_batch()
            if ex > 0:
                network_time.update((time() - loop_time) / ex)

            size = self.games_played.value
            if size > n:
                self.writer.add_scalar("Sample games", size)
                self.writer.flush()
                sample_time.update((time() - end) / (size - n), size - n)
                n = size
                end = time()

            bar.set_postfix(sample_time=sample_time.avg, network_time=network_time.avg)
            bar.update(size - bar.n)

    def killSelfPlayAgents(self):
        for i in range(self.args.workers):
            self.agents[i].join()
            del self.input_tensors[0]
            del self.policy_tensors[0]
            del self.value_tensors[0]
            del self.batch_ready[0]
        self.agents = []
        self.input_tensors = []
        self.policy_tensors = []
        self.value_tensors = []
        self.batch_ready = []
        self.ready_queue = mp.Queue()
        self.completed = mp.Value('i', 0)
        self.games_played = mp.Value('i', 0)

    def saveIterationSamples(self, iteration):
        num_samples = self.file_queue.qsize()
        print(f'Saving {num_samples} samples')
        boardx, boardy = self.game.getBoardSize()
        data_tensor = torch.zeros([num_samples, boardx, boardy])
        policy_tensor = torch.zeros([num_samples, self.game.getActionSize()])
        value_tensor = torch.zeros([num_samples, 1])
        for i in range(num_samples):
            data, policy, value = self.file_queue.get()
            data_tensor[i] = torch.from_numpy(data)
            policy_tensor[i] = torch.tensor(policy)
            value_tensor[i, 0] = value

        os.makedirs(self.args.data, exist_ok=True)

        torch.save(
            data_tensor, f'{self.args.data}/iteration-{iteration:04d}-data.pkl')
        torch.save(policy_tensor,
                   f'{self.args.data}/iteration-{iteration:04d}-policy.pkl')
        torch.save(
            value_tensor, f'{self.args.data}/iteration-{iteration:04d}-value.pkl')
        del data_tensor
        del policy_tensor
        del value_tensor

    def processGameResults(self, iteration):
        num_games = self.result_queue.qsize()
        p1wins = 0
        p2wins = 0
        draws = 0
        for _ in range(num_games):
            winner = self.result_queue.get()
            if winner == 1:
                p1wins += 1
            elif winner == -1:
                p2wins += 1
            else:
                draws += 1
        self.writer.add_scalar('win_rate/p1 vs p2',
                               (p1wins+0.5*draws)/num_games, iteration)
        self.writer.add_scalar('win_rate/draws', draws/num_games, iteration)

    def train(self, iteration):
        datasets = []

        currentHistorySize = min(
            max(4, (iteration + 4)//2),
            self.args.numItersForTrainExamplesHistory)
        for i in range(max(1, iteration - currentHistorySize), iteration + 1):
            data_tensor = torch.load(
                f'{self.args.data}/iteration-{i:04d}-data.pkl')
            policy_tensor = torch.load(
                f'{self.args.data}/iteration-{i:04d}-policy.pkl')
            value_tensor = torch.load(
                f'{self.args.data}/iteration-{i:04d}-value.pkl')
            datasets.append(TensorDataset(
                data_tensor, policy_tensor, value_tensor))

        dataset = ConcatDataset(datasets)
        dataloader = DataLoader(dataset, batch_size=self.args.train_batch_size, shuffle=True,
                                num_workers=self.args.workers, pin_memory=True)

        l_pi, l_v = self.nnet.fast_a0_train(dataloader, self.args.train_steps_per_iteration, self.writer)
        self.writer.add_scalar('loss/policy', l_pi, iteration)
        self.writer.add_scalar('loss/value', l_v, iteration)
        self.writer.add_scalar('loss/total', l_pi + l_v, iteration)
        self.writer.add_scalar("lr", self.nnet.optimizer.param_groups[0]['lr'], global_step=iteration)

        self.nnet.save_checkpoint(
            folder=self.args.checkpoint, filename=f'iteration-{iteration:04d}.pkl')

        del dataloader
        del dataset
        del datasets

    def compareToPast(self, iteration):
        past = max(0, iteration-self.args.pastCompareFreq)
        self.pnet.load_checkpoint(folder=self.args.checkpoint,
                                  filename=f'iteration-{past:04d}.pkl')
        print(f'PITTING AGAINST ITERATION {past}')
        if(self.args.arenaMCTS):
            pplayer = MCTS(self.game, self.pnet, self.args)
            nplayer = MCTS(self.game, self.nnet, self.args)

            def playpplayer(x, turn):
                if turn <= 2:
                    pplayer.reset()
                temp = self.args.temp if turn <= self.args.tempThreshold else self.args.arenaTemp
                policy = pplayer.getActionProb(x, temp=temp)
                return np.random.choice(len(policy), p=policy)

            def playnplayer(x, turn):
                if turn <= 2:
                    nplayer.reset()
                temp = self.args.temp if turn <= self.args.tempThreshold else self.args.arenaTemp
                policy = nplayer.getActionProb(x, temp=temp)
                return np.random.choice(len(policy), p=policy)

            arena = Arena(playnplayer, playpplayer, self.game)
        else:
            pplayer = NNPlayer(self.game, self.pnet, self.args.arenaTemp)
            nplayer = NNPlayer(self.game, self.nnet, self.args.arenaTemp)

            arena = Arena(nplayer.play, pplayer.play, self.game)
        nwins, pwins, draws = arena.playGames(self.args.arenaCompare)

        print(f'NEW/PAST WINS : {nwins} / {pwins} ; DRAWS : {draws}\n')
        self.writer.add_scalar(
            'win_rate/past', float(nwins + 0.5 * draws) / (pwins + nwins + draws), iteration)

    def compareToBest(self, iteration):
        self.pnet.load_checkpoint(folder=self.args.checkpoint,
                                  filename='best.pkl')
        print(f'PITTING AGAINST BEST PREVIOUS ITERATION')
        if(self.args.arenaMCTS):
            pplayer = MCTS(self.game, self.pnet, self.args)
            nplayer = MCTS(self.game, self.nnet, self.args)

            def playpplayer(x, turn):
                if turn <= 2:
                    pplayer.reset()
                temp = self.args.temp if turn <= self.args.tempThreshold else self.args.arenaTemp
                policy = pplayer.getActionProb(x, temp=temp)
                return np.random.choice(len(policy), p=policy)

            def playnplayer(x, turn):
                if turn <= 2:
                    nplayer.reset()
                temp = self.args.temp if turn <= self.args.tempThreshold else self.args.arenaTemp
                policy = nplayer.getActionProb(x, temp=temp)
                return np.random.choice(len(policy), p=policy)

            arena = Arena(playnplayer, playpplayer, self.game)
        else:
            pplayer = NNPlayer(self.game, self.pnet, self.args.arenaTemp)
            nplayer = NNPlayer(self.game, self.nnet, self.args.arenaTemp)

            arena = Arena(nplayer.play, pplayer.play, self.game)
        nwins, pwins, draws = arena.playGames(self.args.arenaCompare)

        print(f'NEW/PAST WINS : {nwins} / {pwins} ; DRAWS : {draws}\n')

        if nwins/(pwins + nwins + draws) > self.args.updateThreshold:
            print('Saving new best checkpoint!')
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pkl')
        self.writer.add_scalar(
            'win_rate/past', float(nwins + 0.5 * draws) / (pwins + nwins + draws), iteration)

    def compareToRandom(self, iteration):
        r = RandomPlayer(self.game)
        nnplayer = NNPlayer(self.game, self.nnet, self.args.arenaTemp)
        print('PITTING AGAINST RANDOM')

        arena = Arena(nnplayer.play, r.play, self.game)
        nwins, pwins, draws = arena.playGames(self.args.arenaCompareRandom)

        print(f'NEW/RANDOM WINS : {nwins} / {pwins} ; DRAWS : {draws}\n')
        self.writer.add_scalar(
            'win_rate/random', float(nwins + 0.5 * draws) / (pwins + nwins + draws), iteration)
