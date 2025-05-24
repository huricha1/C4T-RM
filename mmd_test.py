import torch.nn as nn
import torch
import math
def distance(X, Y, sqrt):
    nX = X.size(0)
    nY = Y.size(0)
    X = X.view(nX, -1).cuda()
    X2 = (X * X).sum(1).reshape(nX, 1)
    Y = Y.view(nY, -1).cuda()
    Y2 = (Y * Y).sum(1).reshape(nY, 1)

    M = torch.zeros(nX, nY)
    M.copy_(X2.expand(nX, nY) + Y2.expand(nY, nX).transpose(0, 1) - 2 * torch.mm(X, Y.transpose(0, 1)))

    del X, X2, Y, Y2

    if sqrt:
        M = ((M + M.abs()) / 2).sqrt()

    return M

def mmd_test(Mxx, Mxy, Myy, sigma) :
    scale = Mxx.mean()
    Mxx = torch.exp(-Mxx/(scale*2*sigma*sigma))
    Mxy = torch.exp(-Mxy/(scale*2*sigma*sigma))
    Myy = torch.exp(-Myy/(scale*2*sigma*sigma))
    a = Mxx.mean()+Myy.mean()-2*Mxy.mean()
    mmd = math.sqrt(max(a, 0))
    return mmd
#  ground truth
real = raw_data
# imputed data
fake = imputed_data
Mxx = distance(real, real, False)
Mxy = distance(real, fake, False)
Myy = distance(fake, fake, False)
sigma=1
print('Manual MMD: ', mmd_test(Mxx, Mxy, Myy, sigma))