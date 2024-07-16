import torch
import pandas as pd
import re
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import numpy as np
import time
import datetime
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import os

# Check if GPU is available
device = torch.device("cuda")

# Load the dataset into a pandas dataframe
df = pd.read_csv("sample2.csv")
df['sentiment'] = df['sentiment'].map({1: 0, 2: 1})
print(df.info())

# Preprocess text
def process(x):
    x = re.sub('[,\.!?:()"]', '', x)
    x = re.sub('<.*?>', ' ', x)
    x = re.sub('http\S+', ' ', x)
    x = re.sub('[^a-zA-Z0-9]', ' ', x)
    x = re.sub('\s+', ' ', x)
    return x.lower().strip()

df['review'] = df['review'].apply(process)

# Split the data into train and test sets
train, test = train_test_split(df, test_size=0.2)
train_sentences = train.review.values
train_labels = train.sentiment.values
test_sentences = test.review.values
test_labels = test.sentiment.values

# Load the BERT tokenizer
print('Loading BERT tokenizer...')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)

# Tokenize and encode the data
def generate_data(data, labels):
    input_ids = []
    attention_masks = []

    for sent in data:
        encoded_dict = tokenizer.encode_plus(
            sent,
            add_special_tokens=True,
            max_length=80,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0).to(device)
    attention_masks = torch.cat(attention_masks, dim=0).to(device)  
    labels = torch.tensor(labels).to(device)

    return input_ids, attention_masks, labels

train_input_ids, train_attention_masks, train_labels = generate_data(train_sentences, train_labels)
test_input_ids, test_attention_masks, test_labels = generate_data(test_sentences, test_labels)

train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)

# Hyperparameters
batch_size = 32
learning_rate = 2e-5
epochs = 4
warmup_steps = 0
weight_decay = 0.01

# Create DataLoader for training and testing
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)

# Load BERT model for sequence classification
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False
)
model.to(device)

# Set up optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8, weight_decay=weight_decay)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

# Function to calculate accuracy
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# Function to format time
def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Set seed value
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Trackers for metrics
training_stats = {
    'train_loss': [],
    'val_loss': [],
    'train_acc': [],
    'val_acc': []
}

total_t0 = time.time()

# Training loop
for epoch_i in range(epochs):
    print(f"\n======== Epoch {epoch_i + 1} / {epochs} ========")
    print('Training...')

    t0 = time.time()
    total_train_loss = 0
    total_train_accuracy = 0
    model.train()

    for step, batch in enumerate(train_dataloader):
        if step % 40 == 0 and step != 0:
            elapsed = format_time(time.time() - t0)
            print(f'  Batch {step} of {len(train_dataloader)}.    Elapsed: {elapsed}.')

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()
        result = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels, return_dict=True)
        loss = result.loss
        logits = result.logits

        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        total_train_accuracy += flat_accuracy(logits, label_ids)

    avg_train_loss = total_train_loss / len(train_dataloader)
    avg_train_accuracy = total_train_accuracy / len(train_dataloader)
    training_time = format_time(time.time() - t0)

    print(f"\n  Average training loss: {avg_train_loss:.2f}")
    print(f"  Average training accuracy: {avg_train_accuracy:.2f}")
    print(f"  Training epoch took: {training_time}")

    training_stats['train_loss'].append(avg_train_loss)
    training_stats['train_acc'].append(avg_train_accuracy)

    # Validation
    print("\nRunning Validation...")
    t0 = time.time()
    model.eval()

    total_eval_loss = 0
    total_eval_accuracy = 0

    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            result = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels, return_dict=True)
        
        loss = result.loss
        logits = result.logits

        total_eval_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(logits, label_ids)

    avg_val_loss = total_eval_loss / len(test_dataloader)
    avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
    validation_time = format_time(time.time() - t0)

    print(f"  Validation Loss: {avg_val_loss:.2f}")
    print(f"  Validation Accuracy: {avg_val_accuracy:.2f}")
    print(f"  Validation took: {validation_time}")

    training_stats['val_loss'].append(avg_val_loss)
    training_stats['val_acc'].append(avg_val_accuracy)

print("\nTraining complete!")
print(f"Total training took {format_time(time.time() - total_t0)} (h:mm:ss)")

# Plot training and validation loss
plt.figure(figsize=(12, 8))
plt.plot(training_stats['train_loss'], label='Training Loss')
plt.plot(training_stats['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plot training and validation accuracy
plt.figure(figsize=(12, 8))
plt.plot(training_stats['train_acc'], label='Training Accuracy')
plt.plot(training_stats['val_acc'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Save the model
output_dir = './model_save/'

# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Load a trained model and vocabulary that you have fine-tuned
model = DistilBertForSequenceClassification.from_pretrained(output_dir)
tokenizer = DistilBertTokenizer.from_pretrained(output_dir)

# Copy the model to the GPU.
model.to(device)

# Predict
model.eval()
predictions, true_labels = [], []

for batch in test_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch

    with torch.no_grad():
        result = model(b_input_ids, attention_mask=b_input_mask, return_dict=True)
    logits = result.logits
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    predictions.append(logits)
    true_labels.append(label_ids)

flat_predictions = [item for sublist in predictions for item in sublist]
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
flat_true_labels = [item for sublist in true_labels for item in sublist]

# Compute the confusion matrix
cf_matrix = confusion_matrix(flat_true_labels, flat_predictions)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cf_matrix, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.show()

# Compute and print the F1 score and accuracy
f1 = f1_score(flat_true_labels, flat_predictions, average='macro')
accuracy = accuracy_score(flat_true_labels, flat_predictions)

print(f'F1 Score: {f1}')
print(f'Accuracy: {accuracy}')

# Predict a single review
def predict_review(review_text):
    # Yorumu ön işleme tabi tutun
    processed_review = process(review_text)

    # Yorumu tokenleştirin
    inputs = tokenizer.encode_plus(
        processed_review,
        add_special_tokens=True,
        max_length=80,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Modeli değerlendirme moduna alın
    model.eval()

    # Tahmin yapın
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, return_dict=True)
    
    logits = outputs.logits
    logits = logits.detach().cpu().numpy()

    # En yüksek olasılığa sahip olan sınıfı alın (0 veya 1)
    pred = np.argmax(logits, axis=1).flatten()

    return pred[0]

# Örnek kullanım
review_text = "This movie was fantastic! I really enjoyed it."
prediction = predict_review(review_text)

print(f'The predicted sentiment for the review is: {prediction} (0 for negative, 1 for positive)')
