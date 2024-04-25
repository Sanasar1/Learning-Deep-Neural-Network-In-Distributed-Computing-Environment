import torch
from tqdm import tqdm
import time
import torch.distributed as dist
from dataloader import get_subset_loaders
from tqdm import tqdm
from validator import validate
from communication import average_gradients_weighted, average_gradients_equal, average_weights_weighted, average_weights_equal


def train_global(model, trainloader, val_loader, trainset, valset, indices_train, indices_val, criterion, optimizer, scheduler, device, rank, world_size, num_local_epochs, num_global_epochs, timelimit, batch_size, prev_fraction, next_fraction, local_weight, aggregation_type, aggregation_by):

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
            local_loss, local_accuracy, epoch_losses = train_local_epoch(model, trainloader, criterion, optimizer, device, scheduler)

            # Wrap local_accuracy in a tensor and use all_reduce to sum it across all workers
            local_accuracy_tensor = torch.tensor([local_accuracy], device=device)
            dist.all_reduce(local_accuracy_tensor, op=dist.ReduceOp.SUM)
            # Divide by the number of workers to get the average
            average_accuracy = local_accuracy_tensor.item() / world_size

            # Determine the max length of loss arrays across all workers
            local_loss_len = torch.tensor(len(epoch_losses), device=device)
            max_loss_len = torch.tensor([0], device=device)
            dist.all_reduce(local_loss_len, op=torch.distributed.ReduceOp.MAX)
            max_loss_len = local_loss_len

            # Pad the local loss array to match the max length
            epoch_losses_tensor = torch.tensor(epoch_losses, dtype=torch.float32, device=device)
            pad_size = max_loss_len.item() - len(epoch_losses)
            if pad_size > 0:
                epoch_losses_tensor = torch.nn.functional.pad(epoch_losses_tensor, (0, pad_size), "constant", -1.0)

            # Gather padded loss arrays from all workers
            gathered_losses = [torch.zeros(max_loss_len.item(), device=device) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_losses, epoch_losses_tensor)

            # Process gathered, padded loss arrays
            if rank == 0:
                for i, worker_losses in enumerate(gathered_losses):
                    # Filter out padding values
                    valid_losses = worker_losses[worker_losses != -1.0].tolist()
                    all_workers_losses[i].extend(valid_losses)
                    
            # Determine the max length of loss arrays across all workers
            local_loss_len = torch.tensor(len(epoch_losses), device=device)
            max_loss_len = torch.tensor([0], device=device)
            dist.all_reduce(local_loss_len, op=torch.distributed.ReduceOp.MAX)
            max_loss_len = local_loss_len

            # Pad the local loss array to match the max length
            epoch_losses_tensor = torch.tensor(epoch_losses, device=device)
            pad_size = max_loss_len.item() - len(epoch_losses)
            if pad_size > 0:
                epoch_losses_tensor = torch.nn.functional.pad(epoch_losses_tensor, (0, pad_size), "constant", -1.0)

            # Gather padded loss arrays from all workers
            gathered_losses = [torch.zeros(max_loss_len.item(), device=device) for _ in range(world_size)]
            dist.all_gather(gathered_losses, epoch_losses_tensor)

            # Process gathered, padded loss arrays
            if rank == 0:
                epoch_losses_all_workers = []
                for losses in gathered_losses:
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
            print(f'Worker {rank}, Global Epoch {global_epoch + 1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

            # Check for finish signal or time limit
            torch.distributed.all_reduce(epoch_finished_signal, op=torch.distributed.ReduceOp.MAX)
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
            torch.distributed.all_reduce(epoch_finished_signal, op=torch.distributed.ReduceOp.MAX)
            if epoch_start_time is None:
                epoch_start_time = time.time()
        
        if aggregation_by == 'gradients':
            if aggregation_type =='equal':
                average_gradients_equal(model, world_size)
            elif aggregation_type == 'weighted':
                average_gradients_weighted(model, world_size, local_weight)
        elif aggregation_by == 'weights':
            if aggregation_type == 'equal':
                average_weights_equal(model, world_size)
            elif aggregation_type == 'weighted':
                average_weights_weighted(model, world_size, local_weight)

        # Aggregate metrics across workers
        dist.all_reduce(epoch_train_losses, op=dist.ReduceOp.SUM)
        dist.all_reduce(epoch_train_accuracies, op=dist.ReduceOp.SUM)
        dist.all_reduce(epoch_val_losses, op=dist.ReduceOp.SUM)
        dist.all_reduce(epoch_val_accuracies, op=dist.ReduceOp.SUM)

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
        dist.barrier()

        end_time = time.time()
        epoch_duration = end_time - start_time
        epoch_duration_tensor = torch.tensor([epoch_duration], device=device)

        # Sum durations across all nodes
        dist.all_reduce(epoch_duration_tensor, op=dist.ReduceOp.SUM)

        #print(f"Total duration for Global Epoch {global_epoch + 1}: {epoch_duration_tensor.item()} seconds")

        trainloader, val_loader, indices_train, indices_val = get_subset_loaders(trainset, valset, indices_train, indices_val, batch_size=batch_size, prev_fraction=prev_fraction, next_fraction=next_fraction, epoch_duration=epoch_duration, epoch_duration_tensor=epoch_duration_tensor)

        epoch_duration_tensor.fill_(0)

    return all_workers_losses, all_epochs_losses, global_epoch_losses, global_epoch_accuracies, global_train_losses, global_train_accuracies, global_val_losses, global_val_accuracies, worker_specific_train_losses, worker_specific_train_accuracies, worker_specific_val_losses, worker_specific_val_accuracies

def train_local_epoch(model, trainloader, criterion, optimizer, device, scheduler):
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