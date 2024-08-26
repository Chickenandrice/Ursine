import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, DistilBertForSequenceClassification, DataCollatorWithPadding

# GPU / CPU setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {torch.cuda.get_device_name(device)}" if torch.cuda.is_available() else "Using CPU")

# Loading dataset
dataset = pd.read_csv('datasets/ccomb_dataset.csv') 

# Checking distribution of sentiment
label_distribution = dataset['label'].value_counts()
print("Label distribution:\n", label_distribution)

# Loading pretrained BERT model
num_labels = len(label_distribution) 

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

model = model.to(device)

# Freezing layers
for name, param in model.named_parameters():
    if name.startswith('distilbert.transformer.layer'):
        layer_num = name.split('.')[3]
        if layer_num.isdigit() and int(layer_num) < 4:
            param.requires_grad = False

# Custom Dataset
class CustomTextDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_length: int):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = self.dataframe.iloc[idx]['text']
        label = self.dataframe.iloc[idx]['label']
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        # Convert tensors to the correct format
        return {key: torch.squeeze(val) for key, val in encoding.items()}, torch.tensor(label, dtype=torch.long)

# Splitting dataset
train_df, test_df = train_test_split(dataset, test_size=0.2, random_state=1)


# Hyperparameters
learning_rate = 1e-5
batch_size = 64
epochs = 10    

# Data loaders
def collate_fn(batch):

    batch_data, labels = zip(*batch)
    labels = torch.stack(labels)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    batch = data_collator(batch_data)
    batch['labels'] = labels

    return batch


train_dataset = CustomTextDataset(train_df, tokenizer, max_length=128)
test_dataset = CustomTextDataset(test_df, tokenizer, max_length=128)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

def train_loop(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer):
    model.train()
    size = len(dataloader.dataset)
    for batch, batch_data in enumerate(dataloader):
        input_ids = batch_data['input_ids'].to(device)
        attention_mask = batch_data['attention_mask'].to(device)
        labels = batch_data['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            print(f"loss: {loss.item():>7f}  [{batch * batch_size + len(labels):>5d}/{size:>5d}]")

def test_loop(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for batch_data in dataloader:
            input_ids = batch_data['input_ids'].to(device)
            attention_mask = batch_data['attention_mask'].to(device)
            labels = batch_data['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            test_loss += loss.item()
            preds = outputs.logits.argmax(dim=1)
            correct += (preds == labels).sum().item()

    test_loss /= num_batches
    accuracy = 100 * correct / size
    print(f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Training and testing
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


print("Pretrained Model\n-------------------------------")
test_loop(test_loader, model, loss_fn)


for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    train_loop(train_loader, model, loss_fn, optimizer)
    test_loop(test_loader, model, loss_fn)
print("Done!")

model_save_path = 'distilbert_finetune.pt'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

