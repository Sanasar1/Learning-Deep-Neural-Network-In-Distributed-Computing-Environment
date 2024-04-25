from pathlib import Path
import matplotlib.pyplot as plt
import os

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_loss_distribution_by_worker(loss_data, output_folder='Graphs'):
    ensure_directory_exists(output_folder)

    fig = plt.figure()
    fig.set_size_inches(16,10)

    plt.boxplot(loss_data, labels=[f'Worker {i}' for i in range(len(loss_data))])

    plt.title('Loss Distribution per Worker')
    plt.xlabel('Worker')
    plt.ylabel('Loss')
    plt.xticks(rotation=45)
    plt.grid(True)

    plt.savefig(Path(output_folder) / 'loss_distribution_by_worker.png')
    plt.clf()  


def plot_loss_distribution_per_epoch(loss_data, output_folder='Graphs'):

    ensure_directory_exists(output_folder)

    fig = plt.figure()
    fig.set_size_inches(16,10)
    
    plt.boxplot(loss_data, labels=[f'Epoch {i+1}' for i in range(len(loss_data))])
    plt.title('Loss Distribution Across All Workers Per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(rotation=45)
    plt.grid(True)

    plt.savefig(Path(output_folder) / 'loss_distribution_per_epoch.png')
    plt.clf()

def plot_loss_distribution_per_epoch_global(loss_data, output_folder='Graphs'):

    ensure_directory_exists(output_folder)

    fig = plt.figure()
    fig.set_size_inches(16,10)
    
    plt.boxplot(loss_data, labels=[f'Epoch {i+1}' for i in range(len(loss_data))])
    plt.title('Loss Distribution Across All Workers Per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(rotation=45)
    plt.grid(True)

    plt.savefig(Path(output_folder) / 'loss_distribution_per_epoch_global.png')
    plt.clf()

def plot_accuracy_distribution_per_epoch_global(loss_data, output_folder='Graphs'):

    ensure_directory_exists(output_folder)

    fig = plt.figure()
    fig.set_size_inches(16,10)
    
    plt.boxplot(loss_data, labels=[f'Epoch {i+1}' for i in range(len(loss_data))])
    plt.title('Accuracy Distribution Across All Workers Per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.grid(True)

    plt.savefig(Path(output_folder) / 'accuracy_distribution_per_epoch_global.png')
    plt.clf()

def plot_metrics_global(epochs, train_loss, train_accuracy, val_loss, val_accuracy, output_folder='Graphs'):
    ensure_directory_exists(output_folder)

    epochs = list(range(1, epochs + 1))

    fig = plt.figure()
    fig.set_size_inches(16,10)

    ax_1 = fig.add_subplot(2,1,1)
    plt.plot(epochs, train_loss, 'o-', label='Train Loss') 
    plt.plot(epochs, val_loss, 'o-', label='Validation Loss')
    plt.title('Individual Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    ax_2 = fig.add_subplot(2,1,2)
    plt.plot(epochs, train_accuracy, 'o-', label='Worker Train Accuracy')
    plt.plot(epochs, val_accuracy, 'o-', label='Worker Val Accuracy')
    plt.title('Individual Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    
    plt.savefig(Path(output_folder) / f'training_metrics.png')

    plt.clf()


def plot_metrics_total(epochs, train_loss, train_accuracy, val_loss, val_accuracy, rank, output_folder='Graphs'):
    ensure_directory_exists(output_folder)

    epochs = list(range(1, epochs + 1))

    fig = plt.figure()
    fig.set_size_inches(16,10)

    ax_1 = fig.add_subplot(2,1,1)
    plt.plot(epochs, train_loss, 'o-', label=f'Worker {rank} Train Loss') 
    plt.plot(epochs, val_loss, 'o-', label=f'Worker {rank} Validation Loss')
    plt.title('Individual Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    ax_2 = fig.add_subplot(2,1,2)
    plt.plot(epochs, train_accuracy, 'o-', label=f'Worker {rank} Train Accuracy')
    plt.plot(epochs, val_accuracy, 'o-', label=f'Worker {rank} Val Accuracy')
    plt.title('Individual Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    
    plt.savefig(Path(output_folder) / f'training_metrics_{rank}.png')

    plt.clf()