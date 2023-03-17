import numpy as np
from cfr_jr.policy import TabularPolicy

def unique(list):
    unique_list = []
    for i in list:
        if i not in unique_list:
            unique_list.append(i)

    return unique_list

class PureStratGenerator:
    def __init__(self, game, player):
        self.game = game
        self.player = player
        self.infosets = self.get_player_infosets()
        self.num_infosets = len(self.infosets)
        self.num_pure_strats = 2**self.num_infosets
        self.num_generated = 0
        self.policy = TabularPolicy(self.game, [self.player])

    def get_player_infosets(self):
        infosets = []
    
        def recurse(state):
            nonlocal infosets 
            if state.is_terminal():
                return
            elif state.is_chance_node():
                for action, action_prob in state.chance_outcomes():
                    new_state = state.child(action)
                    recurse(new_state)
                return
            else:
                # non-terminal, non chance stat
                current_player = state.current_player()
                infoset = state.information_state_string(current_player)

                if infoset not in infosets and self.player == current_player:
                    infosets.append(infoset)
            
                # recurse to children
                for action in state.legal_actions():
                    new_state = state.child(action)
                    recurse(new_state)
                return

        recurse(self.game.new_initial_state())
        return infosets

    
    def get_strat_by_idx(self, idx): 
        strat_bit_vec = bin(idx)[2:] # excluse '0b'
        self.num_generated += 1
        padding = self.num_infosets - len(strat_bit_vec)
        strat_bit_vec = ('0' * padding) + strat_bit_vec  # pads with 0s
        
        # set action probabilities
        for i, infoset in enumerate(self.infosets): 
            state_policy = self.policy.policy_for_key(infoset)
            state_policy[int(strat_bit_vec[i])] = 1
            state_policy[1-int(strat_bit_vec[i])] = 0

        return self.policy

    def get_next_strat(self):
        if self.num_generated > self.num_pure_strats:
            assert False

        strat_bit_vec = bin(self.num_generated)[2:] # excluse '0b'
        self.num_generated += 1
        padding = self.num_infosets - len(strat_bit_vec)
        strat_bit_vec = ('0' * padding) + strat_bit_vec  # pads with 0s
        
        # set action probabilities
        for i, infoset in enumerate(self.infosets): 
            state_policy = self.policy.policy_for_key(infoset)
            state_policy[int(strat_bit_vec[i])] = 1
            state_policy[1-int(strat_bit_vec[i])] = 0
       
        return self.policy

def terminal_reach_probs(game, policy):
    """
    Returns a vector containing the reach probabilities of each terminal history 

    args:
        game : openspiel game
        policy : (openspiel.TabularPolicy) a joint policy (behavior strategy) profile for all players

    returns:
        reach_probs : (dict) dictionary that returns the reach probability of a terminal history 
    """

    reach_probs = dict()

    def recurse(state, reach_prob):
        if state.is_terminal():
            nonlocal reach_probs
            reach_probs[str(state)] = reach_prob
            return

        if state.is_chance_node():
            for action, action_prob in state.chance_outcomes():
                new_state = state.child(action)
                new_reach_prob = reach_prob * action_prob
                recurse(new_state, new_reach_prob)
            return

        current_player = state.current_player()
        info_state = state.information_state_string(current_player)
        info_state_policy = policy.policy_for_key(info_state)

        for action in state.legal_actions():
            action_prob = info_state_policy[action]
            new_state = state.child(action)
            new_reach_prob = reach_prob * action_prob
            recurse(new_state, new_reach_prob)

    recurse(game.new_initial_state(), 1)
    return reach_probs

def chance_terminal_reach_probs(game):
    """
    Returns a vector containing the reach probabilities of each terminal history as if chance
    were following policy and other players are playing to that history

    args:
        game : openspiel game

    returns:
        reach_probs : (np.array) reach probability of each terminal history 
    """

    reach_probs = list()

    def recurse(state, reach_prob):
        if state.is_terminal():
            nonlocal reach_probs
            reach_probs.append(reach_prob)
            return

        if state.is_chance_node():
            for action, action_prob in state.chance_outcomes():
                new_state = state.child(action)
                recurse(new_state, reach_prob*action_prob)
            return

        for action in state.legal_actions():
            new_state = state.child(action)
            recurse(new_state, reach_prob)
    
    recurse(game.new_initial_state(), 1)
    return np.array(reach_probs)

def player_terminal_reach_probs(game, policy, player):
    """
    Returns a vector containing the reach probabilities of each terminal history as if player
    were following policy and other players are playing to that history

    args:
        game : openspiel game
        policy : (openspiel.TabularPolicy) a (potentially joiny) policy (behavior strategy) profile that includes at least 
            all of player's infosets
        player : (int) the player in question

    returns:
        reach_probs : (np.array) reach probability of each terminal history 
    """

    reach_probs = list()

    def recurse(state, reach_prob):
        if state.is_terminal():
            nonlocal reach_probs
            reach_probs.append(reach_prob)
            return

        if state.is_chance_node():
            for action, action_prob in state.chance_outcomes():
                new_state = state.child(action)
                recurse(new_state, reach_prob)
            return

        for action in state.legal_actions():
            new_state = state.child(action)
            new_reach_prob = reach_prob 
            
            nonlocal player
            current_player = state.current_player()
            if current_player == player:
                info_state = state.information_state_string(current_player)
                info_state_policy = policy.policy_for_key(info_state)
                action_prob = info_state_policy[action]
                new_reach_prob *= action_prob
               
            recurse(new_state, new_reach_prob)

    recurse(game.new_initial_state(), 1)
    return np.array(reach_probs)

def get_terminals(game):
    """
    Retuns a list of all terminal states of the game

    args:
        game : openspiel game

    returns:
       terminals : (list) terminals of the game. Each represented as a string
    """

    terminals = []

    def recurse(state):
        if state.is_terminal():
            nonlocal terminals
            terminals.append(str(state))
            return

        if state.is_chance_node():
            for action, action_prob in state.chance_outcomes():
                new_state = state.child(action)
                recurse(new_state)
            return

        for action in state.legal_actions():
            new_state = state.child(action)
            recurse(new_state)

    recurse(game.new_initial_state())
    return terminals

def get_terminal_returns(game):
    """
    Retuns an np.array of all the retuns of terminal states

    args:
        game : openspiel game

    returns:
       returns : (np.Array) returns at each terminals of the game
    """

    terminals = []

    def recurse(state):
        if state.is_terminal():
            nonlocal terminals
            terminals.append(state.returns())
            return

        if state.is_chance_node():
            for action, action_prob in state.chance_outcomes():
                new_state = state.child(action)
                recurse(new_state)
            return

        for action in state.legal_actions():
            new_state = state.child(action)
            recurse(new_state)

    recurse(game.new_initial_state())
    return np.array(terminals).T

def get_all_infosets(game):
    infosets = []

    def recurse(state):
        nonlocal infosets 
        if state.is_terminal():
            return
        elif state.is_chance_node():
            for action, action_prob in state.chance_outcomes():
                new_state = state.child(action)
                recurse(new_state)
            return
        else:
            # non-terminal, non chance stat
            current_player = state.current_player()
            infoset = state.information_state_string(current_player)

            if infoset not in infosets:
                infosets.append(infoset)
        
            # recurse to children
            for action in state.legal_actions():
                new_state = state.child(action)
                recurse(new_state)
            return

    recurse(game.new_initial_state())
    return infosets

def get_player_infosets(player, game):
    infosets = []

    def recurse(state):
        nonlocal infosets 
        if state.is_terminal():
            return
        elif state.is_chance_node():
            for action, action_prob in state.chance_outcomes():
                new_state = state.child(action)
                recurse(new_state)
            return
        else:
            # non-terminal, non chance stat
            current_player = state.current_player()
            infoset = state.information_state_string(current_player)

            if infoset not in infosets and player == current_player:
                infosets.append(infoset)
        
            # recurse to children
            for action in state.legal_actions():
                new_state = state.child(action)
                recurse(new_state)
            return

    recurse(game.new_initial_state())
    return infosets

def evaluate(state, policies, num_players):
    '''
    Evaluates utility of a policy.
    '''

    if state.is_terminal():
        return np.asarray(state.returns())

    if state.is_chance_node():
        state_value = 0.0
        for action, action_prob in state.chance_outcomes():
            new_state = state.child(action)
            state_value += action_prob * evaluate(new_state, policies, num_players)
        return state_value
    
    state_value = np.zeros(num_players)

    current_player = state.current_player()
    info_state_policy = policies[current_player](state)
    for action in state.legal_actions():
        action_prob = info_state_policy.get(action, 0.)

        if action_prob != 0:
            new_state = state.child(action)
            child_utility = evaluate(new_state, policies, num_players)
            state_value += action_prob * child_utility

    return state_value


def unique(list):
    unique_list = []
    for i in list:
        if i not in unique_list:
            unique_list.append(i)

    return unique_list