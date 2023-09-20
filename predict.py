import torch
import torch.nn as nn
import random
input_size = 1
hidden_size = 32
output_size = 1


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Take the output of the last time step
        return out

# Load the saved model
loaded_model = SimpleRNN(input_size, hidden_size, output_size)
loaded_model.load_state_dict(torch.load('rnn_model.pth'))
loaded_model.eval()

# Prepare a single data point (similar to training data preprocessing)
sequence_length = 115
input_size = 1
hidden_size = 32
output_size = 1

input_value = 73.25
input_sequence = torch.Tensor([random.uniform(0, 1) for _ in range(sequence_length - 1)] + [input_value]).unsqueeze(dim=0).unsqueeze(dim=2)

# Pass the input sequence through the model to get predictions
with torch.no_grad():
    predictions = loaded_model(input_sequence)

# Print the predicted output value
predicted_output = predictions.item()
print(f"Predicted Output Value for Input {input_value}: {predicted_output:.4f}")

