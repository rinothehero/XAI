import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# 1. Custom Dataset from result.json
class JsonDataset(Dataset):
    def __init__(self, json_path: str, tokenizer: BertTokenizer, max_len: int = 128):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        text = entry['text']
        label = entry['true_label']
        enc = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        input_ids = enc['input_ids'].squeeze(0)  # [L]
        return input_ids, torch.tensor(label, dtype=torch.long)

# 2. MLP Model
class MLPWithEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, input_ids):
        emb = self.embedding(input_ids)          # [B, L, E]
        emb_mean = emb.mean(dim=1)               # [B, E]
        return self.encoder(emb_mean)

# 3. Main training routine
def main():
    # Settings
    JSON_PATH = "../outputs/intermediate_data/bert_token_explanations_for_mlp.json"
    MAX_LEN = 128
    BATCH_SIZE = 64
    EPOCHS = 5
    EMBED_DIM = 128
    HIDDEN_DIM = 256
    N_CLASSES = 4
    LR = 1e-3

    # Tokenizer and Dataset
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = JsonDataset(JSON_PATH, tokenizer, max_len=MAX_LEN)

    # Train/val split
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    # Model
    model = MLPWithEmbedding(tokenizer.vocab_size, EMBED_DIM, HIDDEN_DIM, N_CLASSES).cuda()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Training loop with best model save
    best_acc = 0.0
    try:
        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_loss = 0
            for input_ids, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
                input_ids, labels = input_ids.cuda(), labels.cuda()
                optimizer.zero_grad()
                logits = model(input_ids)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Train Loss: {total_loss / len(train_loader):.4f}")

            # Validation
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for input_ids, labels in val_loader:
                    input_ids, labels = input_ids.cuda(), labels.cuda()
                    logits = model(input_ids)
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            acc = correct / total
            print(f"Validation Acc: {acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                torch.save(model.cpu().state_dict(), "best_mlp_resultjson.pth")
                print(f"[Epoch {epoch}] Best model saved with acc {best_acc:.4f}")
                model.cuda()
    finally:
        torch.save(model.cpu().state_dict(), "last_mlp_resultjson.pth")
        print("Final model saved to last_mlp_resultjson.pth")
        model.cuda()

if __name__ == '__main__':
    main()