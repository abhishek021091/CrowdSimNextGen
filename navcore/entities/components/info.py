class Timeout:
    def __str__(self):
        return "Timeout"


class ReachGoal:
    def __str__(self):
        return 'Reaching goal'


class Danger:
    def __init__(self, min_dist):
        self.min_dist = min_dist

    def __str__(self):
        return 'Too close'


class Collision:
    def __str__(self):
        return 'Collision'

class OutRoad:
    def __str__(self):
        return 'Out of road'

class Nothing:
    def __str__(self):
        return ''
