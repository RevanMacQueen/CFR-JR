import numpy as np
from pathlib import Path
from tqdm import tqdm
import itertools

from cfr_jr.cfr import CFRSolver
from cfr_jr.utils import PureStratGenerator, get_terminals, player_terminal_reach_probs


class CFR_JR:
    def __init__(self, game, num_players, blueprint=False, checkpoint_dir='results'):
        self.game = game 
        self.cfr = CFRSolver(game, blueprint)
        self.num_players = num_players
        self.terminals = get_terminals(self.game)
        self.reachable_inf = []
        self.reachable_zero = []
        self.joint_dist = None
        self.init_reach_probs_arrays() 
        self.checkpoint_dir = Path(checkpoint_dir)


    def init_reach_probs_arrays(self):
        """
        Initializes reach probability arrays 
        """
       
        for player in range(self.num_players):
            g = PureStratGenerator(self.game, player)
            num_terminals = len(self.terminals)

            reacheable_err = np.ones((g.num_pure_strats, num_terminals)) * 100
            # if reacheable_err[s_idx, z_idx] = 1 then z is potentially reachable with s
            # otherwise  reacheable_err[s_idx, z_idx] = 100
            reacheable = np.zeros((g.num_pure_strats, num_terminals)) 
            # if reacheable[s_idx, z_idx] = 1 then z is potentially reachable with s
            # otherwise reacheable[s_idx, z_idx] = 0

            print("initializing for player {}".format(player))
            for s_idx in tqdm(range(g.num_pure_strats)):
                pure_s_i = g.get_next_strat() # a pure strategy for player i
                reach_probs_s_i = player_terminal_reach_probs(self.game, pure_s_i, player) # numpy array with reach probabilities
                reacheable[s_idx, :] = reach_probs_s_i
                
                reach_probs_s_i[reach_probs_s_i==0] = 100 # everything that is not reachable gets a high "cost"
                reach_probs_s_i[reach_probs_s_i<100] = 0 # everything reachable gets a lot cost
                reacheable_err[s_idx, :] = reach_probs_s_i
                
            self.reachable_inf.append(reacheable_err)
            self.reachable_zero.append(reacheable)


    def mixed_from_behavior(self, policy, player):
        """
        Takes a behavour strategy and returns an outcome-equivalent
        mixed strategy

        Implements Algorithm 2 from https://arxiv.org/pdf/1910.06228.pdf

        args:
            behav_s : (openspiel.TabularPolicy) a tabular policy / behaviour strategy profile
        """

        s_i = dict() # the pure strategy
        reach_probs_vec = player_terminal_reach_probs(self.game, policy, player)

        while not np.all(np.isclose(reach_probs_vec, 0)):
            min_weights = reach_probs_vec*self.reachable_zero[player] # the reach probabilities
            min_weights += self.reachable_inf[player] # add a large cost to terminal that are not reachable by that strategy

            min_weights = np.min(min_weights, axis=1)
            max_s_idx = np.argmax(min_weights) # index of maximizing strategy

            weight = min_weights[max_s_idx]
            s_i[max_s_idx] = weight
            reach_probs_vec -= self.reachable_zero[player][max_s_idx, :]*weight

        return s_i



    def train(self, iterations):
        """
        Trains CFR and outputs a joint distribution on deterministic policies/pure strategies

        args: 
            iterations : (int) number of iterations to run cfr
        """

        joint_dist = dict()
        keys_across_itr = list()

        for itr in tqdm(range(iterations)):
            self.cfr.evaluate_and_update_policy()
            curr_policy = self.cfr.current_policy()
            mixed_strats =  [self.mixed_from_behavior(curr_policy, p) for p in range(self.num_players)]

            # next, we aggregate these into a joint distribution. 
            support = [list(s.keys()) for s in mixed_strats]
            support_prod = itertools.product(*support)
            itr_support = [] #(a, s(a)) pairs in the support of the mixed strategy this iteration
            for s in support_prod:
                # for each player p, mixed_strats[p][s[p]] is the probability of playing s[p]
                # multiply these together to get the probability of jointly playing s
                prob_s = np.product([mixed_strats[p][s[p]] for p in range(self.num_players)]) 
                itr_support.append((s, prob_s))

                if joint_dist.get(s, None) is None:
                    joint_dist[s] = prob_s
                else:
                    joint_dist[s] += prob_s

            keys_across_itr.append(len(list(joint_dist.keys())))
        return joint_dist, keys_across_itr
        