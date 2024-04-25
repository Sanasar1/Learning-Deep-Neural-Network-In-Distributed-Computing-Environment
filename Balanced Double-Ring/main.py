import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import argparse
from model import EnhancedCNNModel
from dataloader import get_loaders
from trainer import train_global
from vizualizator import plot_loss_distribution_by_worker, plot_loss_distribution_per_epoch, plot_loss_distribution_per_epoch_global, plot_accuracy_distribution_per_epoch_global, plot_metrics_global, plot_metrics_total
from evaluator import evaluate
from mpi4py import MPI

def main():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    # Model
    model = EnhancedCNNModel().to(device)
    
    # Initialise model weights (Xavier-Uniform)
    def init_weights(m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    # Broadcast model weights from rank 0 to all other nodes
    def broadcast_model_weights(model, rank):
        for name, param in model.state_dict().items():
            param_data = param.clone().cpu().numpy()
            comm.Bcast(param_data, root=0)
            if rank != 0: 
                param.copy_(torch.from_numpy(param_data).to(param.device))

    model.apply(init_weights)
    broadcast_model_weights(model, rank)

    # Loaders
    train_loader, val_loader, test_loader, trainset, valset, train_indices, val_indices = get_loaders(args.batch_size, world_size, rank, model, device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=25)

    # Training and validation
    all_workers_losses, all_epochs_losses, global_epoch_losses, global_epoch_accuracies, global_train_losses, global_train_accuracies, global_val_losses, global_val_accuracies, worker_specific_train_losses, worker_specific_train_accuracies, worker_specific_val_losses, worker_specific_val_accuracies = train_global(
        model, train_loader, val_loader, trainset, valset, train_indices, val_indices, criterion, optimizer, scheduler, device, rank, world_size, args.epochs_local, args.epochs_global, args.time_limit, args.batch_size, args.prev_fraction, args.next_fraction, comm, args.local_weight, args.aggregation_type, args.aggregation_by
    )

    if rank==0:
        loss, accuracy, all_preds, all_labels = evaluate(model, test_loader, criterion, device, rank)


    # Vizualization
    if rank == 0:

        plot_metrics_global(args.epochs_global, global_train_losses, global_train_accuracies, global_val_losses, global_val_accuracies)

        plot_metrics_total(args.epochs_global * args.epochs_local, worker_specific_train_losses, worker_specific_train_accuracies, worker_specific_val_losses, worker_specific_val_accuracies, rank)

        plot_loss_distribution_by_worker(all_workers_losses)

        plot_loss_distribution_per_epoch(all_epochs_losses)

        plot_loss_distribution_per_epoch_global(global_epoch_losses)

        plot_accuracy_distribution_per_epoch_global(global_epoch_accuracies)

    MPI.Finalize()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", type=int, dest='local_rank', help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--backend", type=str, default="gloo", choices=['nccl', 'gloo', 'mpi'], help='Specify back end. Default=gloo. For GPU nccl is recommended, for cpu gloo')
    parser.add_argument('--dist-url', default='localhost:23456', type=str, help='url used to set up distributed training')
    parser.add_argument('--epochs_local', type=int, default=5, help='Specify number of epochs. Default=100')
    parser.add_argument('--epochs_global', type=int, default=20, help='Specify number of epochs. Default=100')
    parser.add_argument('--batch_size', type=int, default=64, help='Specify batch size. Default=64')
    parser.add_argument('--lr', type=float, default=1e-3, help='Specify learning rate. Default=0.001')
    parser.add_argument('--gpu_weight', type=int, default=10, help='Specify gpu weight. Default=10')
    parser.add_argument('--time_limit', type=float, default=60, help='Specify how long the first node that finishes wait for the rest of the nodes. Default=60')
    parser.add_argument('--prev_fraction', type=float, default=0.5, help='Fraction of data to be taken from previous subset')
    parser.add_argument('--next_fraction', type=float, default=0.5, help='Fraction of data to be taken from next subset')
    parser.add_argument('--aggregation_type', type=str, default='equal', choices=['equal', 'weighted'], help='Equal or weighted aggregation')
    parser.add_argument('--aggregation_by', type=str, default='gradients', choices=['gradients', 'weights'], help='Aggregate by weights or gradients')
    parser.add_argument('--local_weight', type=float, default=0.5, help='Weight of node own data, when aggregating')
                    
    args = parser.parse_args()

    main()