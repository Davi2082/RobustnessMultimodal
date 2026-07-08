import OpenAttack
from OpenAttack.tags import Tag

from attack.modifier import Modifier
from attack.rephraser import Rephraser


class TrepatAttacker(OpenAttack.attackers.ClassificationAttacker):
    TAGS = {Tag("english", "lang"), Tag("get_pred", "victim")}
    
    def __init__(self, model, device, splitter, command, weak=False):
        rephraser = Rephraser(model, device, command)
        self.modifier = Modifier(rephraser, splitter, weak)
    
    def attack(self, victim, input_, goal):
        pred_old = victim.get_pred([input_])[0]
        prob_old = victim.get_prob([input_])[0, pred_old]
        self.modifier.init(input_)
        while True:
            x_new = self.modifier.get_next_variant()
            if x_new is None:
                break
            prob_new = victim.get_prob([x_new])[0, pred_old]
            gain = prob_old - prob_new
            self.modifier.get_feedback(gain)
            if prob_new < 0.5:
                y_new = victim.get_pred([x_new])
                assert (goal.check(x_new, y_new))
                return x_new
        return None
