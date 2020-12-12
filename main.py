import torch
from src.trainer import *
from src.util import *


def run():
    trainer = DeblurTrainer()
    trainer.train()

if __name__ == "__main__":
    run()

