import numpy as np
import torch
import torch.nn as nn
from data_config import *

# def init_a_layer(layer: nn.Module):
#     print(f"{type(layer)}")
#     for name, param in layer.named_parameters():
#         if name.startswith("weight"):
#             nn.init.kaiming_normal_(param)
#         else:
#             nn.init.constant_(param, 0)

class PlanarNormalizingFlow(nn.Module):
    """
    A single layer Planar Normalizing Flow (Danilo 2016) with `tanh` activation
    function, as well as the invertible trick.  The `x` and `y` are assumed to
    be 1-D random variable (i.e., ``value_ndims == 1``)

    .. math::

        \\begin{aligned}
            \\mathbf{y} &= \\mathbf{x} +
                \\mathbf{\\hat{u}} \\tanh(\\mathbf{w}^\\top\\mathbf{x} + b) \\\\
            \\mathbf{\\hat{u}} &= \\mathbf{u} +
                \\left[m(\\mathbf{w}^\\top \\mathbf{u}) -
                       (\\mathbf{w}^\\top \\mathbf{u})\\right]
                \\cdot \\frac{\\mathbf{w}}{\\|\\mathbf{w}\\|_2^2} \\\\
            m(a) &= -1 + \\log(1+\\exp(a))
        \\end{aligned}
    """
    def build(self, shape=None):
        lim_w = np.sqrt(2. / np.prod(shape))
        lim_u = np.sqrt(2)
        
        self.shape = shape
        w = nn.Parameter(torch.empty(shape)[None], requires_grad=True).to(device=global_device)
        nn.init.uniform_(w, -lim_w, lim_w)
        
        u = nn.Parameter(torch.empty(shape)[None], requires_grad=True).to(device=global_device)
        nn.init.uniform_(u, -lim_u, lim_u)

        b = nn.Parameter(torch.zeros(1), requires_grad=True).to(device=global_device)
        
        wu = w.matmul(u.transpose(-1, -2))
        u_hat = u + (-1 + nn.Softplus()(wu) - wu) * w / torch.sum(torch.square(w))  # shape == [1, n_units]

        self._w, self._b, self._u, self._u_hat = w, b, u, u_hat

    def __init__(self, shape=None):
        """
        Construct a new :class:`PlanarNormalizingFlow`.

        Args:
            w_initializer: The initializer for parameter `w`.
            w_regularizer: The regularizer for parameter `w`.
            b_regularizer: The regularizer for parameter `b`.
            b_initializer: The initializer for parameter `b`.
            u_regularizer: The regularizer for parameter `u`.
            u_initializer: The initializer for parameter `u`.
            trainable (bool): Whether or not the parameters are trainable?
                (default :obj:`True`)
        """
        super(PlanarNormalizingFlow, self).__init__()
        self.shape = shape
        self.build(shape=self.shape)
        # self.apply(init_a_layer)
    
    def reset_parameters(self):
        self.build(shape=self.shape)


    def forward(self, input):

        x, log_det_previous = input
        # flatten x for better performance
        # x_flatten, s1, s2 = flatten_to_ndims(x, 2)  # x.shape == [?, n_units]
        wxb = torch.matmul(x, self._w.transpose(-1, -2)) + self._b  # shape == [?, 1]
        tanh_wxb = torch.tanh(wxb)  # shape == [?, 1]

        # compute y = f(x)
        y = x + self._u_hat * tanh_wxb  # shape == [?, n_units]
        # y = unflatten_from_ndims(y, s1, s2)

        # compute log(det|df/dz|)
        grad = 1. - torch.square(tanh_wxb)  # dtanh(x)/dx = 1 - tanh^2(x)
        phi = grad * self._w  # shape == [?, n_units]
        u_phi = torch.matmul(phi, self._u_hat.transpose(-1, -2))  # shape == [?, 1]
        det_jac = 1. + u_phi  # shape == [?, 1]
        log_det = torch.log(torch.abs(det_jac))  # shape == [?, 1]
        # log_det = unflatten_from_ndims(tf.squeeze(log_det, -1), s1, s2)
        # now returns the transformed sample and log-determinant
        return y, log_det

