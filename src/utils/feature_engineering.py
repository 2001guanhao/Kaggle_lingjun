from multiprocessing import Pool
import pandas as pd

class FeatureProcessor:
    def __init__(self, config):
        self.n_jobs = config['processing']['n_jobs']
        
    def process_features(self, df):
        # 并行处理特征
        with Pool(self.n_jobs) as p:
            results = p.map(self._process_single, df)
        return pd.concat(results) 