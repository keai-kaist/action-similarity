import numpy as np

class MotionEmbedding:
    torso = None
    left_arm = None
    right_arm = None
    left_leg = None
    right_leg = None

    #def __init__(self, torso, left_arm, right_arm, left_leg, right_leg, random_init=False):
    def init_by_random(self):
        self.torso = np.random.rand(128)
        self.left_arm = np.random.rand(64)
        self.right_arm = np.random.rand(64)
        self.left_leg = np.random.rand(64)
        self.right_leg = np.random.rand(64)