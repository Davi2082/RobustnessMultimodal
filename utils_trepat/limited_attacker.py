import OpenAttack


class TooManyQueries(Exception):
    def __init__(self):
        self.message = "Too many queries asked to the victim."


class CountingVictim(OpenAttack.victim.Victim):
    def __init__(self, max_queries):
        self.queries = None
        self.real_victim = None
        self.max_queries = max_queries
    
    def reset(self, real_victim):
        self.real_victim = real_victim
        self.queries = set()
    
    def get_pred(self, input_):
        if len(input_) > 1 and (len(self.queries) + len(input_) > self.max_queries):
            return [self.get_pred([input__])[0] for input__ in input_]
        elif len(self.queries) < self.max_queries:
            self.queries.update(input_)
            return self.real_victim.get_pred(input_)
        else:
            raise TooManyQueries()
    
    def get_prob(self, input_):
        if len(input_) > 1 and (len(self.queries) + len(input_) > self.max_queries):
            return [self.get_prob([input__])[0] for input__ in input_]
        elif len(self.queries) < self.max_queries:
            self.queries.update(input_)
            return self.real_victim.get_prob(input_)
        else:
            raise TooManyQueries()


class LimitedAttacker(OpenAttack.attackers.ClassificationAttacker):
    def __init__(self, attacker, max_queries):
        self.attacker = attacker
        self.counting_victim = CountingVictim(max_queries)
        self.querynumbers = []
    
    def attack(self, victim, input_, goal):
        self.counting_victim.reset(victim)
        try:
            result = self.attacker.attack(self.counting_victim, input_, goal)
        except TooManyQueries:
            print("Hit the query limit.")
            result = None
        self.querynumbers.append(len(self.counting_victim.queries))
        return result
    
    def avg_queries(self):
        return sum(self.querynumbers) * 1.0 / len(self.querynumbers)
