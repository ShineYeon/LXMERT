### Pytorch optimization for BERT model. ###
import math
import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required
import logging

logger = logging.getLogger(__name__)


def warmup_cosine(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 0.5 * (1.0 + torch.cos(math.pi * x))


def warmup_constant(x, warmup=0.002):
    """ `warmup`*`t_total` training step 동안 선형적으로 learning_rate을 높임. learning rate은 1.0 이상 """
    if x < warmup:
        return x / warmup
    return 1.0


def warmup_linear(x, warmup=0.02):
    """ `warmup`*`t_total`번째 training step이 피크인 triangular learning rate을 지정함,
        `t_total`번째 training step 이후 learning rate은 0임 """
    if x < warmup:
        return x / warmup
    return max((x - 1.) / (warmup - 1.), 0)


SCHEDULES = {
    'warmup_cosine': warmup_cosine,
    'warmup_constant': warmup_constant,
    'warmup_linear': warmup_linear,
}


class BertAdam(Optimizer):
    """ weight decay 고정이면서 Adam 알고리즘 사용하는 BERT 구현
    Params:
        lr: learning rate
        warmup: t_total에서 warmup할 비율로 -1입력시 warmup 없음, Default: -1
        t_total: learning rate 스케쥴에서 training step의 total number로 -1 입력시 constant learning rate을 의미, Default: -1
        schedule: warmup 시 사용할 schedule, Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay: Weight decay. Default: 0.01
        max_grad_norm: gradient를 위한 maximum norm (-1은 no clipping 뜻함), Default: 1.0
        """

    def __init__(self, params, lr=required, warmup=-1, t_total=-1, schedule='warmup_linear', b1=0.9, b2=0.999, e=1e-6,
                 weight_decay=0.01, max_grad_norm=1.0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0] or -1".format(warmup))
        if not 0.0 <= b1 < 1.0:
            raise ValueError("Invalid b1 parameter: {} - should be in [0.0, 1.0]".format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError("Invalid b2 parameter: {} - should be in [0.0, 1.0]".format(b2))
        if not e >= 0.0:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        b1=b1, b2=b2, e=e, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        super(BertAdam, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step'] / group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def step(self, closure=None):
        """ 단일 optimization step 실행.

        Arguments:
            closure (callable, optional): 모델을 재평가하고 loss 반환하는 closure
        """
        loss = None
        if closure is not None:
            loss = closure()

        warned_for_t_total = False

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead.")

                state = self.state[p]

                # State 초기화
                if len(state) == 0:
                    state['step'] = 0
                    # gradient 값의 exponential moving average
                    state['next_m'] = torch.zeros_like(p.data)
                    # squared gradient 값의 exponential moving average
                    state['next_v'] = torch.zeros_like(p.data)

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2']

                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = next_m / (next_v.sqrt() + group['e'])

                if group['weight_decay'] > 0.0:
                    update += group['weight_decay'] * p.data

                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    progress = state['step'] / group['t_total']
                    lr_scheduled = group['lr'] * schedule_fct(progress, group['warmup'])

                    if group['schedule'] == "warmup_linear" and progress > 1. and not warned_for_t_total:
                        logger.warning(
                            "training beyond specified 't_total' steps with schedule '{}'. Learning rate set to {}. "
                            "Please set 't_total' of {} correctly.".format(group['schedule'], lr_scheduled,
                                                                           self.__class__.__name__)
                        )
                        warned_for_t_total = True
                else:
                    lr_scheduled = group['lr']

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

                state['step'] += 1

        return loss
