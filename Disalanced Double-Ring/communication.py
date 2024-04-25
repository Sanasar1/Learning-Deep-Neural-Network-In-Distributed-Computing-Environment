from mpi4py import MPI
import torch

# Equal aggregation
def double_ring_all_reduce(tensor, rank, world_size):
    comm = MPI.COMM_WORLD
    # Calculate ranks for the two next and two previous nodes in the double ring topology
    send_to_1 = (rank + 1) % world_size
    send_to_2 = (rank + 2) % world_size
    receive_from_1 = (rank - 1 + world_size) % world_size
    receive_from_2 = (rank - 2 + world_size) % world_size
    
    # Initialize tensors for receiving gradients
    recv_tensor_1 = torch.zeros_like(tensor)
    recv_tensor_2 = torch.zeros_like(tensor)
    
    # Convert tensors to numpy arrays for MPI operations
    send_data = tensor.cpu().numpy()
    recv_data_1 = recv_tensor_1.cpu().numpy()
    recv_data_2 = recv_tensor_2.cpu().numpy()
    
    # Start non-blocking receives from the previous two nodes
    req_recv_1 = comm.Irecv(recv_data_1, source=receive_from_1)
    req_recv_2 = comm.Irecv(recv_data_2, source=receive_from_2)
    
    # Start non-blocking sends to the next two nodes
    req_send_1 = comm.Isend(send_data.copy(), dest=send_to_1)
    req_send_2 = comm.Isend(send_data.copy(), dest=send_to_2)
    
    # Wait for all communications to complete
    MPI.Request.Waitall([req_send_1, req_send_2, req_recv_1, req_recv_2])
    
    # Convert numpy arrays back to tensors
    tensor = torch.from_numpy(send_data).to(tensor.device)
    recv_tensor_1 = torch.from_numpy(recv_data_1).to(tensor.device)
    recv_tensor_2 = torch.from_numpy(recv_data_2).to(tensor.device)
    
    # Average the local tensor with the two received tensors
    tensor.add_(recv_tensor_1).add_(recv_tensor_2)
    tensor.div_(3)  # Divide by 3 to average the contributions

# Weighted aggregation
def double_ring_all_reduce_weighted(tensor, rank, world_size, local_weight):
    comm = MPI.COMM_WORLD
    # Calculate ranks for the two subsequent and two preceding nodes in the ring topology
    send_to_1 = (rank + 1) % world_size
    send_to_2 = (rank + 2) % world_size
    receive_from_1 = (rank - 1 + world_size) % world_size
    receive_from_2 = (rank - 2 + world_size) % world_size
    
    # Initialize tensors for receiving gradients
    recv_tensor_1 = torch.zeros_like(tensor)
    recv_tensor_2 = torch.zeros_like(tensor)
    
    # Convert tensors to numpy arrays for MPI operations
    send_data = tensor.cpu().numpy()
    recv_data_1 = recv_tensor_1.cpu().numpy()
    recv_data_2 = recv_tensor_2.cpu().numpy()
    
    # Start non-blocking receives from the previous two nodes
    req_recv_1 = comm.Irecv(recv_data_1, source=receive_from_1)
    req_recv_2 = comm.Irecv(recv_data_2, source=receive_from_2)
    
    # Start non-blocking sends to the next two nodes
    req_send_1 = comm.Isend(send_data.copy(), dest=send_to_1)
    req_send_2 = comm.Isend(send_data.copy(), dest=send_to_2)
    
    # Wait for all communications to complete
    MPI.Request.Waitall([req_send_1, req_send_2, req_recv_1, req_recv_2])
    
    # Update the original tensor directly
    recv_tensor_1 = torch.from_numpy(recv_data_1).to(tensor.device)
    recv_tensor_2 = torch.from_numpy(recv_data_2).to(tensor.device)
    
    # Average the received tensors with the local tensor
    received_weight = (1 - local_weight) / 2
    tensor.mul_(local_weight).add_(recv_tensor_1, alpha=received_weight).add_(recv_tensor_2, alpha=received_weight)