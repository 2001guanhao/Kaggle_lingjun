from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    DATA_DIR = os.getenv('DATA_DIR', 'data')
    MODEL_DIR = os.getenv('MODEL_DIR', 'models')
    API_KEY = os.getenv('API_KEY') 