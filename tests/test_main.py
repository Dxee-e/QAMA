from QAMA.QAMultiheadAttention import QAMultiheadAttention
import torch
from icecream import ic

def test_backend_sa():
    """
    Test the backend of the QAMultiheadAttention class with simulated annealing.
    """
    # Initialize the model
    args_solver = {
        'user_id': "69878024601862146", 
        'sdk_code': "0i4T6LY1XygfwN3MWa8Fjq27OaT0sq",
    }
    args_model = {
        'num_soft_selection_qubit': 3,
    }
    model = QAMultiheadAttention(num_heads=4, args_model=args_model, solver_backend='kaiwu_sa', args_solver=args_solver)
    
    # Create random input tensors
    Q = torch.randn(7, 4, 3, 32)
    K = torch.randn(7, 4, 3, 32)
    V = torch.randn(7, 4, 3, 32)
    
    # Forward pass
    output = model(Q, K, V)
    
    # Check the output shape
    assert output.shape == (7, 4, 3, 32), "Output shape is incorrect"