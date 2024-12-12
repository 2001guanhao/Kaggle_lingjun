class DataLoader:
    def __init__(self, config):
        self.use_cache = config['data']['use_cache']
        self._setup_cache()
    
    def load_data(self, cache_key=None):
        if self.use_cache and cache_key:
            return self._load_from_cache(cache_key)
        # 加载数据
        return data 