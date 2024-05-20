import torch
import torch.nn as nn

# LSTM-self-attention
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, device="cpu"):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=False)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(self.device)
        output, (h,c) = self.lstm(input_seq, (h_0, c_0))
        return output, (h,c)


class AttentionLSTM(nn.Module):
    def __init__(self, input_features_num, output_len, lstm_hidden, lstm_layers, batch_size, device="cpu"):
        super(AttentionLSTM, self).__init__()
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.lstmunit = LSTM(input_features_num, lstm_hidden, lstm_layers, batch_size, device)
        self.attentionunit = nn.MultiheadAttention(lstm_hidden, 1)
        self.linear = nn.Linear(lstm_hidden, output_len)

    def forward(self, input_seq):
        ula, (h_out, c_out) = self.lstmunit(input_seq)
        att_out, att_weight = self.attentionunit(ula, ula, ula)
        att_out+=ula
        out = att_out.contiguous().view(att_out.shape[0] * att_out.shape[1], self.lstm_hidden)
        out = self.linear(out)
        out = out.view(att_out.shape[0], att_out.shape[1], -1)
        out = out[:, -1, :]
        return out

