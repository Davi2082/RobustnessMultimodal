import random

from OpenAttack.attackers import BERTAttacker

random.seed(1)


class FastBERTAttacker(BERTAttacker):
    def __init__(self, *args, **kwargs):
        super(FastBERTAttacker, self).__init__(*args, **kwargs)
    
    def get_important_scores(self, words, tgt_model, orig_prob, orig_label, orig_probs):
        result = [random.random() for i in range(len(words) - 1)]
        return (result)
