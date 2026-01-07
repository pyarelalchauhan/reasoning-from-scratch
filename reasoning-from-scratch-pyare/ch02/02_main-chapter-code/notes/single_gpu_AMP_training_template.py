import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Dummy model/dataset for example ---
class MyModel(nn.Module):
    def __init__(self, in_dim=100, out_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        return self.net(x)

model = MyModel().to(device)

x = torch.randn(1024, 100)
y = torch.randint(0, 10, (1024,))
loader = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# --- AMP setup ---
amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))

num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        # Forward in mixed precision
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # Backward with gradient scaling (needed only for fp16)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    print(f"Epoch {epoch+1}: loss = {epoch_loss:.4f}")
