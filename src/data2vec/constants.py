import os

from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
IGNORE_INDEX = -100

OPTIMIZER_STATE_PATH = "optim_state.msgpack"
MODEL_PATH = "flax_model.msgpack"
TRAINING_STATE_PATH = "training_state.yaml"
CONFIG_PATH = "config.json"
