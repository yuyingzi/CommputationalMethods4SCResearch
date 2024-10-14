import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Assuming CUDA is available, else fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
    """A custom dataset class for your irony detection task."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def load_dataset(file_path, tokenizer, max_length):
    """Function to load and tokenize the dataset."""
    df = pd.read_csv(file_path, delimiter="\t")
    texts = df['text'].tolist()  # Adjust column name based on your dataset
    labels = df['Label'].tolist()  # Adjust column name based on your dataset

    encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=max_length, return_tensors="pt")
    return CustomDataset(encodings, labels)

def compute_metrics(pred):
    """Function to compute metrics of the model's performance."""
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load and tokenize the dataset
file_path = '/Users/yinterested/Downloads/数据库/nlp_data/new_irony_dataset.csv'
max_length = 256
iron_dataset = load_dataset(file_path, tokenizer, max_length)

# Define the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=iron_dataset,
    eval_dataset=iron_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# 假设模型已经训练完毕，我们将其保存到指定路径
model_save_path = './iron_model'

# 保存模型和tokenizer
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

from transformers import pipeline

# 加载训练好的反讽检测模型
model_path = './iron_model'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
irony_detector = pipeline('text-classification', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# 对情感分类数据集的每条文本进行反讽预测
def predict_irony_labels(texts, tokenizer, model, device):
    irony_labels = []
    for text in texts:
        # Encode the text, ensuring it's truncated to the max length the model can handle
        inputs = tokenizer.encode_plus(
            text, 
            return_tensors='pt', 
            max_length=512, 
            truncation=True, 
            padding='max_length'
        )

        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Assuming using a binary classification model where the second token (index 1) represents "irony"
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        irony_label = torch.argmax(predictions, dim=1).cpu().numpy()[0]  # Get the predicted class (0 or 1)
        irony_labels.append(irony_label)

    return irony_labels

# Load data
data = pd.read_csv("IMDB Dataset.csv", error_bad_lines=False)
data['label'] = data['sentiment'].map({'positive': 1, 'negative': 0})
train_texts, test_texts, train_labels, test_labels = train_test_split(data['review'], data['label'], test_size=0.2)

train_texts = train_texts.reset_index(drop=True)
test_texts = test_texts.reset_index(drop=True)
train_labels = train_labels.reset_index(drop=True)
test_labels = test_labels.reset_index(drop=True)

# 假设你的模型和tokenizer已经定义好了
train_irony_labels = predict_irony_labels(train_texts, tokenizer, model, device)
test_irony_labels = predict_irony_labels(test_texts, tokenizer, model, device)

class CustomDataset1(Dataset):
    def __init__(self, texts, labels, irony_labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.irony_labels = irony_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        item = self.tokenizer(self.texts[idx], truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        item['irony_labels'] = torch.tensor(self.irony_labels[idx], dtype=torch.long)  # New
        return {key: value.squeeze(0) for key, value in item.items()}  # Ensure tensors are correctly shaped

    def __len__(self):
        return len(self.texts)

# Assuming the adjusted CustomDataset is used
train_dataset = CustomDataset1(train_texts, train_labels, train_irony_labels, tokenizer, 256)
test_dataset = CustomDataset1(test_texts, test_labels, test_irony_labels, tokenizer, 256)

model1 = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=500,
    load_best_model_at_end=True,
    # Add the following line to report metrics every evaluation step
    report_to="all"
)

# Initialize the Trainer
trainer1 = Trainer(
    model=model1,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer1.train()

trainer1.evaluate(test_dataset)