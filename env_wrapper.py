class EnvWrapper:
    def __init__(self, env, key_mapping, default_action=None):
        self.env = env
        self.key_mapping = key_mapping
        self.default_action = default_action

    def get_action_from_keys(self, keys_pressed):
        for key, action_id in self.key_mapping.items():
            if keys_pressed[key]:
                return action_id
        return self.default_action

    def step_from_pygame_keys(self, keys_pressed):
        action = self.get_action_from_keys(keys_pressed)
        obs, reward, cap1, cap2, cap3 = self.env.step(action)
        return obs, reward, cap1, cap2, cap3

    def reset(self):
        return self.env.reset()