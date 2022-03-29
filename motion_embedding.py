import numpy as np

class MotionEmbedding:
    torso = None
    left_arm = None
    right_arm = None
    left_leg = None
    right_leg = None

    #def __init__(self, torso, left_arm, right_arm, left_leg, right_leg, random_init=False):
    def init_by_random(self, T = 32):
        self.torso = np.random.rand(T, 128)
        self.left_arm = np.random.rand(T, 64)
        self.right_arm = np.random.rand(T, 64)
        self.left_leg = np.random.rand(T, 64)
        self.right_leg = np.random.rand(T, 64)