from sweep_SegFormer import sweep_train
import wandb

LIGHT = True
WANDB = True

if not LIGHT:
    PATH_JPGS = "RailNet_DT/rs19_val/jpgs/rs19_val"
    PATH_MASKS = "RailNet_DT/rs19_val/uint8/rs19_val"  # /rails
else:
    PATH_JPGS = "RailNet_DT/rs19_val_light/jpgs/rs19_val"
    PATH_MASKS = "RailNet_DT/rs19_val_light/uint8/rs19_val"

PATH_MODELS = "RailNet_DT/models"
PATH_LOGS = "RailNet_DT/logs"

sweep_config = {
    'method': 'random',  # 'bayes', 'grid'
    'metric': {
        'name': 'MIoU',
        'goal': 'maximize'
    },
    'parameters': {
        'epochs': {
            'value': 10
        },
        'learning_rate': {
            'distribution': 'uniform',
            'min': 0.0001,
            'max': 0.01
        },
        'optimizer': {
            'values': ['adam', 'sgd', 'adagrad']  # Different optimizers to sweep over
        },
        'scheduler': {
            'values': ['ReduceLROnPlateau', 'LinearLR']  # Different schedulers to sweep over
        },
        'batch_size': {
            'distribution': 'q_log_uniform_values',
            'q': 8,
            'min': 8,
            'max': 32
        },
        'image_size': {
            'value': 550  # Fixed image size
        },
        'outs': {
            'value': 13  # Fixed number of outputs
        }
    }
}

if __name__ == "__main__":
        wandb.agent('ovalach/RailNet/1bl4fkdx', sweep_train, count=20)
