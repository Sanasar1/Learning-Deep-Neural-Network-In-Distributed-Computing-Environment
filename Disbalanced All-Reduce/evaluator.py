import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate(model, loader, criterion, device, rank):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    # Progress bar
    progress_bar = tqdm(enumerate(loader), total=len(loader))

    with torch.no_grad():
        for i, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            progress_bar.set_description("Testing")
            progress_bar.set_postfix(loss=running_loss/(i+1), accuracy=100.*correct/total)

    loss = running_loss / len(loader)
    accuracy = 100 * correct / total

    # Convert lists to numpy arrays for metric calculation
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print(all_preds)
    print(all_labels)

    # Calculate precision, recall, and F1-score
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    # Calculate precision, recall, and F1-score using 'weighted' average
    precision_weighted = precision_score(all_labels, all_preds, average='weighted')
    recall_weighted = recall_score(all_labels, all_preds, average='weighted')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')

    # Calculate precision, recall, and F1-score using 'micro' average
    precision_micro = precision_score(all_labels, all_preds, average='micro')
    recall_micro = recall_score(all_labels, all_preds, average='micro')
    f1_micro = f1_score(all_labels, all_preds, average='micro')

    print(f'Worker {rank}, Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.2f}%, Weighted Precision: {precision_weighted:.2f}, Weighted Recall: {recall_weighted:.2f}, Weighted F1 Score: {f1_weighted:.2f}')

    print(f'Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}')

    print(f'Micro recision: {precision_micro:.2f}, Micro ecall: {recall_micro:.2f}, Micro F1 Score: {f1_micro:.2f}')

    return loss, accuracy, all_preds, all_labels