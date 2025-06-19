import numpy as np


class Gaussian:
    def __init__(self, id, opacity, color, activate=True):
        self.id = id
        # numpy array of shape (3), meaning x, y, bandwidth
        self.opacity = opacity
        # numpy array of shape (3), meaning r, g, b
        self.color = color
        self.activate = activate
        
    def __copy__(self):
        return Gaussian(self.id, np.copy(self.opacity), np.copy(self.color), self.activate)
        
    def freeze(self):
        self.activate = False
    
    def unfreeze(self):
        self.activate = True
        
    def is_frozen(self):
        return not self.activate
    
    def to_json(self):
        return {
            'id': self.id,
            'opacity': self.opacity.tolist(),
            'color': self.color.tolist(),
            'activate': self.activate
        }
        
    def get(self):
        return self.opacity, self.color
    
    def copy(self):
        return Gaussian(self.id, np.copy(self.opacity), np.copy(self.color), self.activate)
    
    @staticmethod
    def setup_gaussians(opacity, color):
        #兼容原本代码
        if len(color[1:-1]) == len(opacity):
            color = color[1:-1]
            
        gmm_num = len(opacity)
        gaussians = []
        for i in range (gmm_num):
            gaussian = Gaussian(i, opacity[i], color[i])
            gaussians.append(gaussian)
        return gaussians