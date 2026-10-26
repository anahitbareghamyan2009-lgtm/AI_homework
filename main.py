import torch
import torch.nn as nn
import torch.optim as optim

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, num_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, x):
        seq_len = x.size(1)
        out = self.embed(x)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        out = self.transformer(out, mask=mask)
        out = out.mean(dim=1)
        return self.fc_out(out)


vocab_size = 1000
d_model = 64
nhead = 4
num_layers = 2
num_classes = 2 

model = SimpleTransformer(vocab_size, d_model, nhead, num_layers, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
inputs = torch.randint(0, vocab_size, (4, 10))  
targets = torch.tensor([0, 1, 1, 0])            


model.train()
epochs = 5
for e in range(epochs):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{e+1}/{epochs}], Loss: {loss.item():.4f}")
