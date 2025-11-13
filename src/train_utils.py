import torch
import torch.nn.functional as F

def train(model, device, train_loader, optimizer, criterion, epoch):
    """
    Performs one epoch of training using the specified model and data loader.

    Args:
        model (nn.Module): The CNN model to be trained.
        device (torch.device): The device (e.g., 'mps' or 'cpu') to use for computation.
        train_loader (DataLoader): DataLoader for the training dataset.
        optimizer (optim.Optimizer): The optimizer (e.g., SGD) used for updating weights.
        criterion (nn.Module): The loss function (e.g., CrossEntropyLoss).
        epoch (int): The current epoch number.
    """
    model.train() # Set the model to training mode
    
    for batch_idx, (data, target) in enumerate(train_loader):
        
        # Move data and targets to the specified device (GPU/MPS)
        data, target = data.to(device), target.to(device) 
        
        # --- Backpropagation Core Steps ---
        
        # 1. Zero Grad: Clear accumulated gradients from the previous step
        optimizer.zero_grad()
        
        # 2. Forward Pass: Compute model output (Logits)
        output = model(data)
        
        # 3. Compute Loss: Measure the error between prediction and true labels
        loss = criterion(output, target) 
        
        # 4. Backward Pass: Calculate gradients for all parameters
        loss.backward()
        
        # 5. Step: Update weights based on the calculated gradients (using SGD)
        optimizer.step()
        
        # Optional: Print training status every 100 batches
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')


def test(model, device, test_loader, criterion): 
    """
    Evaluates the model's performance on the test dataset.

    Args:
        model (nn.Module): The CNN model to be evaluated.
        device (torch.device): The device (e.g., 'mps' or 'cpu') to use for computation.
        test_loader (DataLoader): DataLoader for the test dataset.
        criterion (nn.Module): The loss function.
    """
    model.eval() # Set the model to evaluation mode (e.g., disables Dropout)
    test_loss = 0
    correct = 0

    with torch.no_grad(): # Disable gradient calculation to save memory and time
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Accumulate loss: Calculate item loss and multiply by batch size to get total batch loss
            test_loss += criterion(output, target).item() * len(data) 
            
            # Find the most likely class index (prediction)
            pred = output.argmax(dim=1, keepdim=True) 
            
            # Compare prediction with true labels (correctly summed up)
            correct += pred.eq(target.view_as(pred)).sum().item() 

    # Calculate average loss per sample
    test_loss /= len(test_loader.dataset)

    # Print final evaluation results
    print(f'\nTest Run:\n  Avg. Loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.1f}%)')


def get_device():
    """
    Checks for the best available device for PyTorch computation (MPS/GPU or CPU).
    
    Returns:
        torch.device: The best available device for training.
    """
    # 1. Check for Apple Silicon GPU (MPS Backend)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🎉 Successfully found Apple Silicon GPU (MPS) for acceleration.")
        
    # 2. Check for general CUDA/GPU
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🎉 Found CUDA GPU for acceleration.")
        
    # 3. Fallback to CPU
    else:
        device = torch.device("cpu")
        print("❌ No GPU found. Falling back to CPU.")
        
    return device