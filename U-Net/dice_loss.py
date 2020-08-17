import torch
from torch.autograd import Function

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target, target_widen=None):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        tmp = self.inter
        if target_widen is not None:
            tmp = torch.dot(input.view(-1), target_widen.view(-1))
            
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * tmp.float() + eps) / (self.union.float() - self.inter.float() + tmp.float())
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target
        
def get_precision_recal(input, target, target_widen=None):
    eps = 0.0001
    inter = torch.dot(input.view(-1), target.view(-1))
    tmp = inter
    if target_widen is not None:
        tmp = torch.dot(input.view(-1), target_widen.view(-1))
    
    precision = tmp.float() / (torch.sum(input).float() + eps)
    recall = tmp.float() / (torch.sum(target).float() - inter.float() + tmp.float() + eps)
    return precision, recall

def dice_coeff(input, target, target_widen=None, enable_precision_recall=False):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
        s_p = torch.FloatTensor(1).cuda().zero_()
        s_r = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()
        s_p = torch.FloatTensor(1).zero_()
        s_r = torch.FloatTensor(1).zero_()
    
    if enable_precision_recall:
        if target_widen is None:
            for i, c in enumerate(zip(input, target)):
                s = s + DiceCoeff().forward(c[0], c[1])
                r_p, r_r = get_precision_recal(c[0], c[1])
                s_p = s_p + r_p
                s_r = s_r + r_r
        else:
            for i, c in enumerate(zip(input, target, target_widen)):
                s = s + DiceCoeff().forward(c[0], c[1], c[2])
                r_p, r_r = get_precision_recal(c[0], c[1], c[2])
                s_p = s_p + r_p
                s_r = s_r + r_r
        return s / (i + 1), s_p / (i + 1), s_r / (i + 1)
    else:
        if target_widen is None:
            for i, c in enumerate(zip(input, target)):
                s = s + DiceCoeff().forward(c[0], c[1])
        else:
            for i, c in enumerate(zip(input, target, target_widen)):
                s = s + DiceCoeff().forward(c[0], c[1], c[2])

    return s / (i + 1)
