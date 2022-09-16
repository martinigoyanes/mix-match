from experiments import *
import torch
import gc

gc.collect()

torch.cuda.empty_cache()

def main():
    # TODO: Is linear rampup for lambda_u in MixMatchLoss correct?
    # fully_supervised_baseline()
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # fully_supervised_medium()
    experiment_1()

if __name__ == "__main__":
    main()
