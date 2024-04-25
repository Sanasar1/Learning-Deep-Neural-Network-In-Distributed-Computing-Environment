import torch
from tqdm import tqdm
import time
from dataloader import get_subset_loaders
from tqdm import tqdm
from mpi4py import MPI
import numpy as np
from validator import validate
from communication import double_ring_all_reduce, double_ring_all_reduce_weighted

def train_global(model, trainloader, val_loader, trainset, valset, indices_train, indices_val, criterion, optimizer, scheduler, device, rank, world_size, num_local_epochs, num_global_epochs, timelimit, batch_size, prev_fraction, next_fraction, comm, local_weight, aggregation_type, aggregation_by):

    # Initialize metrics storage
    global_train_losses = []
    global_val_losses = []
    global_train_accuracies = []
    global_val_accuracies = []
    worker_specific_train_losses = []
    worker_specific_val_losses = []
    worker_specific_train_accuracies = []
    worker_specific_val_accuracies = []
    all_workers_losses = [[] for _ in range(world_size)]
    all_epochs_losses = []
    global_epoch_losses = []
    global_epoch_accuracies = []

    progress_bar = tqdm(range(num_global_epochs), desc='Global Epochs')

    for global_epoch in progress_bar:

        # Initialize accumulators for global epoch metrics
        epoch_train_losses = torch.tensor([0.0], device=device)
        epoch_train_accuracies = torch.tensor([0.0], device=device)
        epoch_val_losses = torch.tensor([0.0], device=device)
        epoch_val_accuracies = torch.tensor([0.0], device=device)
        current_global_epoch_losses = []
        current_global_epoch_accuracies = []

        start_time = time.time()
        
        # Variable to check if the finish signal has been broadcasted
        epoch_finished_signal = torch.tensor([0], dtype=torch.int, device=device)
        
        # Record the start time of the epoch
        epoch_start_time = None

        for local_epoch in range(num_local_epochs):
            local_loss, local_accuracy, epoch_losses = train_local_epoch(model, trainloader, criterion, optimizer, device, world_size, scheduler, local_epoch)

            # Wrap local_accuracy in a tensor and use all_reduce to sum it across all workers
            local_accuracy_tensor = torch.tensor([local_accuracy], device=device)
            local_accuracy_tensor = comm.allreduce(local_accuracy_tensor, op=MPI.SUM)
            # Divide by the number of workers to get the average
            average_accuracy = local_accuracy_tensor.item() / world_size

            # Determine the max length of loss arrays across all workers
            local_loss_len = torch.tensor(len(epoch_losses), device=device)
            max_loss_len = comm.allreduce(local_loss_len, op=MPI.MAX)

            # Pad the local loss array to match the max length
            epoch_losses_tensor = torch.tensor(epoch_losses, dtype=torch.float32, device=device)
            pad_size = max_loss_len.item() - len(epoch_losses)
            if pad_size > 0:
                epoch_losses_tensor = torch.nn.functional.pad(epoch_losses_tensor, (0, pad_size), "constant", -1.0)

            # Assuming epoch_losses_tensor is a PyTorch tensor, convert it to a NumPy array
            epoch_losses_numpy = epoch_losses_tensor.cpu().numpy()

            # Prepare a receive buffer that is large enough to hold data from all processes
            gathered_losses_numpy = np.empty([world_size, len(epoch_losses_numpy)], dtype=np.float32)

            # Use Allgather to collect losses from all processes
            comm.Allgather([epoch_losses_numpy, MPI.FLOAT], [gathered_losses_numpy, MPI.FLOAT])

            # Process gathered, padded loss arrays
            if rank == 0:
                for i, worker_losses in enumerate(gathered_losses_numpy):
                    # Filter out padding values
                    valid_losses = worker_losses[worker_losses != -1.0].tolist()
                    all_workers_losses[i].extend(valid_losses)
                    
            # Determine the max length of loss arrays across all workers
            local_loss_len = torch.tensor(len(epoch_losses), device=device)
            max_loss_len = comm.allreduce(local_loss_len, op=MPI.MAX)

            # Pad the local loss array to match the max length
            epoch_losses_tensor = torch.tensor(epoch_losses, device=device)
            pad_size = max_loss_len.item() - len(epoch_losses)
            if pad_size > 0:
                epoch_losses_tensor = torch.nn.functional.pad(epoch_losses_tensor, (0, pad_size), "constant", -1.0)

            epoch_losses_numpy = epoch_losses_tensor.cpu().numpy()

            # Prepare a NumPy array to gather losses into
            gathered_losses_numpy = np.empty((world_size, len(epoch_losses_numpy)), dtype=epoch_losses_numpy.dtype)

            # Use MPI to gather arrays
            comm.Allgather([epoch_losses_numpy, MPI.FLOAT], [gathered_losses_numpy, MPI.FLOAT])

            # Process gathered, padded loss arrays
            if rank == 0:
                epoch_losses_all_workers = []
                for losses in gathered_losses_numpy:
                    # Filter out padding values
                    valid_losses = losses[losses != -1.0].tolist()
                    epoch_losses_all_workers.extend(valid_losses)
                    current_global_epoch_losses.extend(valid_losses)
                all_epochs_losses.append(epoch_losses_all_workers)
                current_global_epoch_accuracies.append(average_accuracy)

            val_loss, val_accuracy = validate(
            model, val_loader, criterion, device
        )
            
            print(f"Rank {rank}, Global Epoch {global_epoch+1}, Local Epoch {local_epoch+1}, Loss: {local_loss}, Accuracy: {local_accuracy}")
            print(f'Worker {rank}, Global Epoch {global_epoch + 1}, Validation Loss: {val_loss:.4f}, 'f'Validation Accuracy: {val_accuracy:.2f}%')

            # Check for finish signal or time limit
            epoch_finished_signal = torch.tensor([epoch_finished_signal.item()], device=device)
            epoch_finished_signal = comm.allreduce(epoch_finished_signal, op=MPI.MAX)
            if epoch_finished_signal.item() == 1 and epoch_start_time is None:
                # Start the timer when the first node finishes
                epoch_start_time = time.time()
            elif epoch_start_time and (time.time() - epoch_start_time > timelimit):
                # If timer has started and exceeded the time limit, break
                break 
            
            # Update worker-specific metrics
            if rank == 0:
                worker_specific_train_losses.append(local_loss)
                worker_specific_train_accuracies.append(local_accuracy)
                worker_specific_val_losses.append(val_loss)
                worker_specific_val_accuracies.append(val_accuracy)

            # Update local accumulators
            epoch_train_losses += torch.tensor([local_loss], device=device)
            epoch_train_accuracies += torch.tensor([local_accuracy], device=device)
            epoch_val_losses += torch.tensor([val_loss], device=device)
            epoch_val_accuracies += torch.tensor([val_accuracy], device=device)


        if epoch_finished_signal.item() == 0:
        # Broadcast finish signal, when finishing iteration
            epoch_finished_signal.fill_(1)
            epoch_finished_signal = torch.tensor(comm.allreduce(epoch_finished_signal.item(), op=MPI.MAX))
            if epoch_start_time is None:
                epoch_start_time = time.time()
        
        # Averaging gradients after each global epoch
        if aggregation_by == 'gradients':
            if aggregation_type == 'weighted':
                for param in model.parameters():
                    if param.grad is not None:
                        # Weighted gradient aggregation
                        double_ring_all_reduce_weighted(param.grad.data, rank, world_size, local_weight)
            elif aggregation_type == 'equal':
                for param in model.parameters():
                    if param.grad is not None:
                        # Equal gradient aggregation
                        double_ring_all_reduce(param.grad.data, rank, world_size)
        if aggregation_by == 'weights':
            if aggregation_type == 'weighted':
                for param in model.parameters():
                    if param.requires_grad:
                        # Weighted model weights aggregation
                        double_ring_all_reduce_weighted(param.data, rank, world_size, local_weight)
            elif aggregation_type == 'equal':
                for param in model.parameters():
                    if param.requires_grad:
                        # Equal model weights aggregation
                        double_ring_all_reduce(param.data, rank, world_size)
                
        # Aggregate metrics across workers
        epoch_train_losses = comm.allreduce(epoch_train_losses, op=MPI.SUM)
        epoch_train_accuracies = comm.allreduce(epoch_train_accuracies, op=MPI.SUM)
        epoch_val_losses = comm.allreduce(epoch_val_losses, op=MPI.SUM)
        epoch_val_accuracies = comm.allreduce(epoch_val_accuracies, op=MPI.SUM)

        # Average metrics across workers
        avg_train_loss = (epoch_train_losses / world_size).item() / num_local_epochs
        avg_train_accuracy = (epoch_train_accuracies / world_size).item() / num_local_epochs
        avg_val_loss = (epoch_val_losses / world_size).item() / num_local_epochs
        avg_val_accuracy = (epoch_val_accuracies / world_size).item() / num_local_epochs

        # Update global metrics
        global_train_losses.append(avg_train_loss)
        global_train_accuracies.append(avg_train_accuracy)
        global_val_losses.append(avg_val_loss)
        global_val_accuracies.append(avg_val_accuracy)
                
        global_epoch_losses.append(current_global_epoch_losses)
        global_epoch_accuracies.append(current_global_epoch_accuracies)
        
        # Update progress bar
        progress_bar.set_postfix(loss=avg_train_loss, accuracy=100.*avg_train_accuracy)

        # Synchronize all workers before starting the next global epoch
        comm.Barrier()

        end_time = time.time()
        epoch_duration = end_time - start_time
        epoch_duration_tensor = torch.tensor([epoch_duration], device=device)
        epoch_duration_tensor = comm.allreduce(epoch_duration_tensor, op=MPI.SUM)


        #print(f"Total duration for Global Epoch {global_epoch + 1}: {epoch_duration_tensor.item()} seconds")
            
        trainloader, val_loader, indices_train, indices_val = get_subset_loaders(trainset, valset, indices_train, indices_val, batch_size=batch_size, prev_fraction=prev_fraction, next_fraction=next_fraction, epoch_duration=epoch_duration, epoch_duration_tensor=epoch_duration_tensor)

        epoch_duration_tensor.fill_(0)

    return all_workers_losses, all_epochs_losses, global_epoch_losses, global_epoch_accuracies, global_train_losses, global_train_accuracies, global_val_losses, global_val_accuracies, worker_specific_train_losses, worker_specific_train_accuracies, worker_specific_val_losses, worker_specific_val_accuracies

def train_local_epoch(model, trainloader, criterion, optimizer, device, world_size, scheduler, epoch):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0
    epoch_losses = []

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        epoch_losses.append(loss.item())

    scheduler.step()

    train_loss = running_loss / len(trainloader)
    train_accuracy = 100 * correct / total

    return train_loss, train_accuracy, epoch_losses