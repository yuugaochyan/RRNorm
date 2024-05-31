from transformers import TrainerCallback

class SavePretrainedCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        self._trainer.save_predtrained
        return control