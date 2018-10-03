# class LogAndSign(object):
#     def __init__(self, device, p = 10.):
#         super(LogAndSign, self).__init__()
#         self._p = torch.tensor(p).to(device)
#         self.device = device
#     def __call__(self, gradients):
#         # eps = np.finfo(gradients.dtype.as_numpy_dtype).eps
#         eps = 1e-7
#         ndims = len(list(gradients.size()))

#         def clamp(inputs, min_value=None, max_value=None):
#             output = inputs
#             if min_value is not None:
#                 output = torch.max(output, min_value)
#             if max_value is not None:
#                 output = torch.min(output, max_value)
#             return output

#         log_value = torch.log(torch.abs(gradients) + eps).to(self.device)
#         log_or_minusone = clamp(log_value / self._p, min_value=torch.tensor(-1.0).to(self.device))
#         sign = clamp(gradients * torch.exp(self._p), min_value=torch.tensor(-1.0).to(self.device), max_value=torch.tensor(1.0).to(self.device))
#         res = torch.cat([log_or_minusone, sign], ndims - 1)
#         return res
            

# class NodeUpdater(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(NodeUpdater, self).__init__()
#         self.hidden_size = hidden_size
#         self.compute_gate_x = nn.Linear(input_size, 4 * hidden_size)
#         self.compute_gate_h = nn.Linear(hidden_size, 4 * hidden_size)

#         init.xavier_normal_(self.compute_gate_h.weight.data)
#         init.constant_(self.compute_gate_x.bias.data, 0)
#         init.xavier_normal_(self.compute_gate_h.weight.data)
#         init.constant_(self.compute_gate_x.bias.data, 0)
    
#     def forward(self, x, old_h, old_c):
#         gates = self.compute_gate_h(old_h) + self.compute_gate_x(x)
#         gates = gates.split(self.hidden_size, 1)

#         new_c_candid = F.tanh(gates[0])
#         gate_u = F.sigmoid(gates[1])
#         gate_f = F.sigmoid(gates[2])
#         gate_o = F.sigmoid(gates[3])
        
#         new_c = gate_u * new_c_candid + gate_f * old_c
#         new_h = gate_o * F.tanh(new_c)
        
#         return new_h, new_c