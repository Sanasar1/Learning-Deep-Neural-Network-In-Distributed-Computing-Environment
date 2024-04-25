from mpi4py import MPI
import torch

# Equal aggregation
def ring_all_reduce_equal(tensor, rank, world_size):
    comm = MPI.COMM_WORLD
    # Calculate the next and previous rank in the ring topology
    send_to = (rank + 1) % world_size
    receive_from = (rank - 1 + world_size) % world_size
    
    # Initialize a tensor for receiving gradients
    recv_tensor = torch.zeros_like(tensor)
    
    # Convert tensors to numpy arrays for MPI operations
    send_data = tensor.cpu().numpy()
    recv_data = recv_tensor.cpu().numpy()
    
    # Start non-blocking receive from the previous node
    req_recv = comm.Irecv(recv_data, source=receive_from)
    
    # Send tensor to the next node non-blocking
    req_send = comm.Isend(send_data, dest=send_to)
    
    # Wait for both send and receive to complete
    MPI.Request.Waitall([req_send, req_recv])
    
    tensor = torch.from_numpy(send_data).to(tensor.device)
    recv_tensor = torch.from_numpy(recv_data).to(tensor.device)
    
    tensor.add_(recv_tensor).div_(2)

# Weighted aggregation
def ring_all_reduce_weighted(tensor, rank, world_size, local_weight):
    comm = MPI.COMM_WORLD
    # Calculate the next and previous rank in the ring topology
    send_to = (rank + 1) % world_size
    receive_from = (rank - 1 + world_size) % world_size
    
    # Initialize a tensor for receiving gradients
    recv_tensor = torch.zeros_like(tensor)
    
    # Convert tensors to numpy arrays for MPI operations
    send_data = tensor.cpu().numpy()
    recv_data = recv_tensor.cpu().numpy()
    
    # Start non-blocking receive from the previous node
    req_recv = comm.Irecv(recv_data, source=receive_from)
    
    # Send tensor to the next node non-blocking
    req_send = comm.Isend(send_data, dest=send_to)
    
    # Wait for both send and receive to complete
    MPI.Request.Waitall([req_send, req_recv])
    
    # Convert numpy arrays back to tensors
    tensor = torch.from_numpy(send_data).to(tensor.device)
    recv_tensor = torch.from_numpy(recv_data).to(tensor.device)
    
    # Average the received tensor with the local tensor
    received_weight = 1 - local_weight

    tensor.mul_(local_weight).add_(recv_tensor, alpha=received_weight)