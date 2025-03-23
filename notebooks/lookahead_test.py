import torch
import torch.nn as nn
import torch_optimizer as toptim
from torch_optimizer import Lookahead

# Define a simple dummy model.
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)
    def forward(self, x):
        return self.linear(x)

model = DummyModel()

# Create a base optimizer (RAdam in this case)
base_optimizer = toptim.RAdam(model.parameters(), lr=1e-3)

# Wrap the base optimizer with Lookahead.
optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)

# Modified helper function: if optimizer is Lookahead, use its 'optimizer' attribute.
def get_optimizer_state_dict(optimizer):
    if isinstance(optimizer, Lookahead):
        print("Lookahead detected. Returning inner optimizer state dict.")
        return optimizer.optimizer.state_dict()  # Use 'optimizer.optimizer', not 'inner_optimizer'
    return optimizer.state_dict()

# Retrieve and print the state dict.
state_dict = get_optimizer_state_dict(optimizer)
print("State dict keys:", state_dict.keys())

# Dummy training step to verify things work:
dummy_input = torch.randn(3, 10)
dummy_output = model(dummy_input)
loss = dummy_output.sum()
loss.backward()
optimizer.step()
optimizer.zero_grad()

print("Mini sample run completed without error.")
