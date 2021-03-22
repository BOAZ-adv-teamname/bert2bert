from bert2bert import Bert2Bert
from dataset import load_dataset
import torch
if __name__ == '__main__':
    trainer = Bert2Bert(
        model_save_path="saved/model.pt",
        batch_size=16,
        num_gpus=torch.cuda.device_count() 
    )

    train = load_dataset(batch_size=trainer.batch_size)
    trainer.fit(train)

