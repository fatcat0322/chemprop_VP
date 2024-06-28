from chemprop.args import TrainArgs
from typing import Callable
import torch

def Tu(A:torch.FloatTensor,B:torch.FloatTensor,C:torch.FloatTensor,
       D:torch.FloatTensor,E:torch.FloatTensor,T:torch.FloatTensor):
    A,B,C,D,E,T=A.squeeze(),B.squeeze(),C.squeeze(),D.squeeze(),E.squeeze(),T.squeeze()
    lnP = A+torch.div(B,T)-C*torch.log(T)-D*T
    return lnP

def Antonine(A:torch.FloatTensor,B:torch.FloatTensor,C:torch.FloatTensor,
             D:torch.FloatTensor,E:torch.FloatTensor,T:torch.FloatTensor):
    A,B,C,D,E,T=A.squeeze(),B.squeeze(),C.squeeze(),D.squeeze(),E.squeeze(),T.squeeze()
    lnP = A-torch.div(B,(C+T))
    return lnP.unsqueeze(-1)

def Riedel(A:torch.FloatTensor,B:torch.FloatTensor,C:torch.FloatTensor,
           D:torch.FloatTensor,E:torch.FloatTensor,T:torch.FloatTensor):
    A,B,C,D,E,T=A.squeeze(),B.squeeze(),C.squeeze(),D.squeeze(),E.squeeze(),T.squeeze()
    lnP = A-torch.div(B,T)+C*torch.log(T)+D*T.pow(6)
    return lnP.unsqueeze(-1)

def Wagner25(A:torch.FloatTensor,B:torch.FloatTensor,C:torch.FloatTensor,
             D:torch.FloatTensor,E:torch.FloatTensor,T:torch.FloatTensor):
    A,B,C,D,E,T=A.squeeze(),B.squeeze(),C.squeeze(),D.squeeze(),E.squeeze(),T.squeeze()
    tau=1-T
    lnP = A+torch.div((B*tau+C*tau.pow(1.5)+D*tau.pow(2.5)+E*tau.pow(5)),T)
    return lnP.unsqueeze(-1)

def get_equation(args:TrainArgs)->Callable[[torch.FloatTensor],torch.FloatTensor]:
    """Get the empirical equation embedded in the model."""
    eqn=args.equation
    if eqn == 'Tu':
        return Tu
    elif eqn =='Antonine':
        return Antonine
    elif eqn == 'Riedel':
        return Riedel
    elif eqn == 'Wagner25':
        return Wagner25
    else:
        raise ValueError(f'Equation "{eqn}" is not supported.')