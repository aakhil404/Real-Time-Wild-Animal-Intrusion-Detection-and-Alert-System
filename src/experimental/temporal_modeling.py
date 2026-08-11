import torch
import torch.nn as nn
from collections import deque
import time

class DetectionTracker:
    def __init__(self, window_size=10, alert_threshold=4):
        """
        Tracks verified detections in a rolling time/frame window to estimate threat levels.
        - None: No animals verified.
        - Passing: Animal detected in few frames (low threat, brief crossing).
        - Persistent Intrusion: Animal lingering in frame across many frames (high threat).
        """
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        # Double-ended queue to store (timestamp, is_verified)
        self.history = deque(maxlen=window_size)
        
    def update(self, is_verified):
        """Adds a detection event to the rolling window and returns current threat level."""
        current_time = time.time()
        self.history.append((current_time, is_verified))
        return self.get_threat_level()
        
    def get_threat_level(self):
        """Calculates current threat level based on rolling history."""
        if not self.history:
            return "NONE"
            
        verified_count = sum(1 for _, verified in self.history if verified)
        
        if verified_count == 0:
            return "NONE"
        elif verified_count >= self.alert_threshold:
            return "PERSISTENT INTRUSION (HIGH THREAT)"
        else:
            return "PASSING ANIMAL (LOW THREAT)"

# PyTorch LSTM Trajectory Classifier Scaffold
class TrajectoryLSTM(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_classes=2, num_layers=1):
        """
        LSTM Sequence Classifier to analyze movement paths of detected animals.
        Input features per frame: [x_center, y_center, width, height, confidence]
        Output classes: [passing, lingering/intruding]
        """
        super(TrajectoryLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Recurrent LSTM layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Classifier fully connected layer
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        """
        x: Tensor of shape (batch, sequence_length, input_dim)
        """
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward pass through LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the output of the last sequence step
        out = out[:, -1, :]
        
        # Classification
        out = self.fc(out)
        return out

if __name__ == "__main__":
    # Test tracking system
    tracker = DetectionTracker(window_size=5, alert_threshold=3)
    events = [False, True, True, False, True, True]
    
    print("Testing Temporal Tracker:")
    for i, ev in enumerate(events):
        threat = tracker.update(ev)
        print(f"  Frame {i+1} (Detected={ev}) -> Threat level: {threat}")
        
    # Test LSTM shape
    seq_inputs = torch.randn(4, 10, 5) # Batch of 4, sequence length of 10 frames, 5 features per frame
    model = TrajectoryLSTM()
    outputs = model(seq_inputs)
    print("\nTrajectoryLSTM test output shape:", outputs.shape)
