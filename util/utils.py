import torch
import numpy as np
import random


def seed_everything(seed):
    torch.manual_seed(seed)  # torch를 거치는 모든 난수들의 생성순서를 고정한다
    torch.cuda.manual_seed(seed)  # cuda를 사용하는 메소드들의 난수시드는 따로 고정해줘야한다
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True  # 딥러닝에 특화된 CuDNN의 난수시드도 고정
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)  # numpy를 사용할 경우 고정
    random.seed(seed)  # 파이썬 자체 모듈 random 모듈의 시드 고정

