import json

from torch.utils.data import DataLoader


def load_dataset(batch_size):
    train_data = open("dataset/final.jsonl", "r").read().splitlines()

    train_set = []
    for data in train_data:
        try:        
            data = json.loads(data)
            article_original = data["original"].replace('·'," ")
            #article_original = [a.replace("\n", " ") for a in article_original]
            #article_original = " ".join(article_original)
            abstractive = data["summary"].replace("\\"," ")
            train_set.append((article_original, abstractive))
        except:
            pass

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16, #original 32
        pin_memory=True,
    )

    return train_loader
