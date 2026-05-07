import numpy as np

try:
    from base_agent import BaseAgent
except ImportError:
    from agents.base_agent import BaseAgent

# from evaluator.base_agent import BaseAgent

class NumpyWindMasterAgent(BaseAgent):
    def __init__(self, weights_path="final_model.npz", crop_size=32):
        super().__init__()
        self.crop_size = crop_size
        self.weights = np.load(weights_path)
        self.prev_angle = None
        self.goal_pos = np.array([64, 127])

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def conv2d_simple(self, x, weight, bias):
        """Version simplifiée de Conv2d (stride 1, padding same)"""
        out_channels, in_channels, kh, kw = weight.shape
        _, _, h, w = x.shape
        # On assume padding=1 ici pour simplifier comme dans SailingNet
        x_padded = np.pad(x, ((0,0), (0,0), (1,1), (1,1)), mode='constant')
        output = np.zeros((1, out_channels, h, w))
        
        for oc in range(out_channels):
            for ic in range(in_channels):
                for i in range(h):
                    for j in range(w):
                        region = x_padded[0, ic, i:i+kh, j:j+kw]
                        output[0, oc, i, j] += np.sum(region * weight[oc, ic])
            output[0, oc] += bias[oc]
        return output

    def max_pool2d(self, x, kernel_size=2):
        b, c, h, w = x.shape
        out_h, out_w = h // kernel_size, w // kernel_size
        output = x[:, :, :out_h*kernel_size, :out_w*kernel_size]
        output = output.reshape(b, c, out_h, kernel_size, out_w, kernel_size)
        return output.max(axis=(3, 5))

    def get_local_crop(self, obs):
        pos = obs[0:2].astype(int)
        wmap = obs[32774:49158].reshape(128, 128)
        wfield = obs[6:32774].reshape(128, 128, 2)
        
        pad = self.crop_size // 2
        wmap_padded = np.pad(wmap, pad, constant_values=1)
        wfield_padded = np.pad(wfield, ((pad,pad),(pad,pad),(0,0)), constant_values=0)
        
        y, x = pos[1] + pad, pos[0] + pad
        crop_wmap = wmap_padded[y-pad:y+pad, x-pad:x+pad]
        crop_wfield = wfield_padded[y-pad:y+pad, x-pad:x+pad]
        
        combined = np.zeros((1, 3, self.crop_size, self.crop_size))
        combined[0, 0] = crop_wmap
        combined[0, 1:] = crop_wfield.transpose(2, 0, 1)
        return combined

    def act(self, observation):
        # 1. Prise d'infos
        local_map = self.get_local_crop(observation)
        curr_wind = observation[4:6]
        curr_angle = np.arctan2(curr_wind[1], curr_wind[0])
        angle_delta = curr_angle - self.prev_angle if self.prev_angle is not None else 0
        self.prev_angle = curr_angle
        
        to_goal = self.goal_pos - observation[0:2]
        scalars = np.array([[observation[2], observation[3], to_goal[0], to_goal[1], angle_delta]])

        # 2. Forward CNN
        x = self.conv2d_simple(local_map, self.weights['cnn.0.weight'], self.weights['cnn.0.bias'])
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2d_simple(x, self.weights['cnn.3.weight'], self.weights['cnn.3.bias'])
        x = self.relu(x)
        x_cnn = x.flatten().reshape(1, -1)

        # 3. Forward MLP
        x_mlp = self.relu(scalars @ self.weights['mlp.0.weight'].T + self.weights['mlp.0.bias'])

        # 4. Fusion & Actor
        combined = np.concatenate([x_cnn, x_mlp], axis=1)
        x = self.relu(combined @ self.weights['actor.0.weight'].T + self.weights['actor.0.bias'])
        probs = self.softmax(x @ self.weights['actor.2.weight'].T + self.weights['actor.2.bias'])

        # 5. Retourne l'action la plus probable
        return int(np.argmax(probs))

    def reset(self):
        self.prev_angle = None