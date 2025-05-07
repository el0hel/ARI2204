"""

this class simulates the blackjack environment.

state representation (returned by 'get_state()' and 'reset()'):
    - player_total (int): sum of the player's hand (with ace possibly counted as 11 or 1)
    - dealer_card (int): value of the dealer's visible card
    - ace (bool): true if the player has an ace that can be counted as 11 without losing

action space (used with `step(action)`):
    - 0: HIT (draw a new card)
    - 1: STAND (end player's turn and dealer plays)

methods:
    - reset() - starts a new game and returns the initial state.
    - step(action) - applies the action (0 or 1), returns next state, reward, and whether the game is over.
    - get_state() - returns current state representation.
    - print_hands(): method to debug by printing current player and dealer hands.

rewards:
    - +1 if player wins
    -  0 if draw or game continues
    - -1 if player loses

usage example can be seen in test_blackjack.py
"""

import random


class BlackjackEnv:
    def __init__(self):
        self.deck = []
        self.player_hand = []
        self.dealer_hand = []
        self.card_values = {
            'A': 1,  # will be treated as 11 later if usable
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
            '7': 7, '8': 8, '9': 9, '10': 10,
            'J': 10, 'Q': 10, 'K': 10
        }
        self.reset()

    def reset(self):
        self.deck = self._create_deck()
        random.shuffle(self.deck)
        self.player_hand = [self._draw_card(), self._draw_card()]
        self.dealer_hand = [self._draw_card()]
        return self.get_state(), False  # (state, game_over)

    def _create_deck(self):
        # creates 52-card deck
        ranks = ['A'] + [str(i) for i in range(2, 11)] + ['J', 'Q', 'K']
        # 4 of each rank
        return ranks * 4

    def _draw_card(self):
        # pop a card from the list
        return self.deck.pop()

    def _hand_values(self, hand):
        # returns value list of a hand
        return [self.card_values[card] for card in hand]

    def _sum_hand(self, hand):
        values = self._hand_values(hand)
        total = sum(values)
        ace = 'A' in hand and total + 10 <= 21
        return (total + 10 if ace else total), ace

    def _is_over_21(self, hand):
        # returns true if hand sums up to over 21, false otherwise
        total, _ = self._sum_hand(hand)
        return total > 21

    def get_state(self):
        # returns the state representation (player_total, dealer_card_value, ace)
        player_total, ace = self._sum_hand(self.player_hand)
        dealer_card_value = self.card_values[self.dealer_hand[0]]
        return player_total, dealer_card_value, ace

    def step(self, action):
        # takes an action, 0 - HIT, 1 - STAND. returns the new state, the reward, and whether round is over
        if action == 0:  # HIT
            self.player_hand.append(self._draw_card())
            if self._is_over_21(self.player_hand):
                return self.get_state(), -1, True  # player loses
            else:
                return self.get_state(), 0, False  # game continues
        elif action == 1:  # STAND
            # dealer plays
            while True:
                dealer_total, _ = self._sum_hand(self.dealer_hand)
                # strict policy, only hit if total is less than 17
                if dealer_total < 17:
                    self.dealer_hand.append(self._draw_card())
                else:
                    break

            player_total, _ = self._sum_hand(self.player_hand)
            dealer_total, _ = self._sum_hand(self.dealer_hand)

            if self._is_over_21(self.dealer_hand):
                return self.get_state(), 1, True  # player wins
            elif player_total > dealer_total:
                return self.get_state(), 1, True  # player wins
            elif player_total < dealer_total:
                return self.get_state(), -1, True  # player loses
            else:
                return self.get_state(), 0, True  # a draw occurs
        else:
            raise ValueError(f'Invalid action: {action}. To hit, press 0. To stand, press 1.')

    def print_hands(self):
        player_total, _ = self._sum_hand(self.player_hand)
        dealer_total, _ = self._sum_hand(self.dealer_hand)
        print(f"Player's hand: {self.player_hand} (sum = {player_total})")
        print(f"Dealer's hand: {self.dealer_hand} (sum = {dealer_total})")

