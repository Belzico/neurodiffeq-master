from neurodiffeq import diff       # the differentiation operation
from neurodiffeq.ode import solve  # the ANN-based solver
from neurodiffeq.conditions import IVP   # the initial condition
import numpy as np
import matplotlib.pyplot as plt
from neurodiffeq.solvers import GenericSolver
import warnings






class BaseCondition:
    r"""Base class for all conditions.

    A condition is a tool to `re-parameterize` the output(s) of a neural network.
    such that the re-parameterized output(s) will automatically satisfy initial conditions (ICs)
    and boundary conditions (BCs) of the PDEs/ODEs that are being solved.

    .. note::
        - The nouns *(re-)parameterization* and *condition* are used interchangeably in the documentation and library.
        - The verbs *(re-)parameterize* and *enforce* are different in that:

          - *(re)parameterize* is said of network outputs;
          - *enforce* is said of networks themselves.
    """

    def __init__(self):
        self.ith_unit = None

    def parameterize(self, output_tensor, *input_tensors):
        r"""Re-parameterizes output(s) of a network.

        :param output_tensor: Output of the neural network.
        :type output_tensor: `torch.Tensor`
        :param input_tensors: Inputs to the neural network; i.e., sampled coordinates; i.e., independent variables.
        :type input_tensors: `torch.Tensor`
        :return: The re-parameterized output of the network.
        :rtype: `torch.Tensor`

        .. note::
            This method is **abstract** for BaseCondition
        """
        raise ValueError(f"Abstract {self.__class__.__name__} cannot be parameterized")  # pragma: no cover

    def enforce(self, net, *coordinates):
        r"""Enforces this condition on a network.

        :param net: The network whose output is to be re-parameterized.
        :type net: `torch.nn.Module`
        :param coordinates: Inputs of the neural network.
        :type coordinates: `torch.Tensor`
        :return: The re-parameterized output, where the condition is automatically satisfied.
        :rtype: `torch.Tensor`
        """
        # concatenate the coordinates and pass to network
        network_output = net(torch.cat(coordinates, dim=1))
        # if `ith_unit` is set, the condition will only be enforced on the i-th output unit
        if self.ith_unit is not None:
            network_output = network_output[:, self.ith_unit].view(-1, 1)
        # parameterize the raw output and return
        return self.parameterize(network_output, *coordinates)

    def set_impose_on(self, ith_unit):
        r"""**[DEPRECATED]** When training several functions with a single, multi-output network, this method is called
        (by a `Solver` class or a `solve` function) to keep track of which output is being parameterized.

        :param ith_unit: The index of network output to be parameterized.
        :type ith_unit: int

        .. note::
            This method is deprecated and retained for backward compatibility only. Users interested in enforcing
            conditions on multi-output networks should consider using a ``neurodiffeq.conditions.EnsembleCondition``.
        """

        warnings.warn(
            f"`{self.__class__.__name__}.set_impose_on` is deprecated and will be removed in the future",
            DeprecationWarning,
        )
        self.ith_unit = ith_unit



class DirichletBVP3D(BaseCondition):
    r"""An Dirichlet boundary condition on the boundary of :math:`[x_0, x_1] \times [y_0, y_1]`, where

    - :math:`u(x_0, y) = f_0(y)`;
    - :math:`u(x_1, y) = f_1(y)`;
    - :math:`u(x, y_0) = g_0(x)`;
    - :math:`u(x, y_1) = g_1(x)`.

    :param x_min: The lower bound of x, the :math:`x_0`.
    :type x_min: float
    :param x_min_val: The boundary value on :math:`x = x_0`, i.e. :math:`f_0(y)`.
    :type x_min_val: callable
    :param x_max: The upper bound of x, the :math:`x_1`.
    :type x_max: float
    :param x_max_val: The boundary value on :math:`x = x_1`, i.e. :math:`f_1(y)`.
    :type x_max_val: callable
    :param y_min: The lower bound of y, the :math:`y_0`.
    :type y_min: float
    :param y_min_val: The boundary value on :math:`y = y_0`, i.e. :math:`g_0(x)`.
    :type y_min_val: callable
    :param y_max: The upper bound of y, the :math:`y_1`.
    :type y_max: float
    :param y_max_val: The boundary value on :math:`y = y_1`, i.e. :math:`g_1(x)`.
    :type y_max_val: callable
    """
 
    def __init__(self, x_min, x_min_val, x_max, x_max_val, y_min, y_min_val, y_max, y_max_val):
        r"""Initializer method
        """
        super().__init__()
        self.x0, self.f0 = x_min, x_min_val
        self.x1, self.f1 = x_max, x_max_val
        self.y0, self.g0 = y_min, y_min_val
        self.y1, self.g1 = y_max, y_max_val

    def parameterize(self, output_tensor, x, y):
        r"""Re-parameterizes outputs such that the Dirichlet condition is satisfied on all four sides of the domain.

        The re-parameterization is
        :math:`\displaystyle u(x,y)=A(x,y)
        +\tilde{x}\big(1-\tilde{x}\big)\tilde{y}\big(1-\tilde{y}\big)\mathrm{ANN}(x,y)`, where

        :math:`\displaystyle \begin{align*}
        A(x,y)=&\big(1-\tilde{x}\big)f_0(y)+\tilde{x}f_1(y) \\
        &+\big(1-\tilde{y}\big)\Big(g_0(x)-\big(1-\tilde{x}\big)g_0(x_0)+\tilde{x}g_0(x_1)\Big) \\
        &+\tilde{y}\Big(g_1(x)-\big(1-\tilde{x}\big)g_1(x_0)+\tilde{x}g_1(x_1)\Big)
        \end{align*}`

        :math:`\displaystyle\tilde{x}=\frac{x-x_0}{x_1-x_0}`,

        :math:`\displaystyle\tilde{y}=\frac{y-y_0}{y_1-y_0}`,

        and :math:`\mathrm{ANN}` is the neural network.

        :param output_tensor: Output of the neural network.
        :type output_tensor: `torch.Tensor`
        :param x: :math:`x`-coordinates of inputs to the neural network; i.e., the sampled :math:`x`-coordinates.
        :type x: `torch.Tensor`
        :param y: :math:`y`-coordinates of inputs to the neural network; i.e., the sampled :math:`y`-coordinates.
        :type y: `torch.Tensor`
        :return: The re-parameterized output of the network.
        :rtype: `torch.Tensor`
        """
        x_tilde = (x - self.x0) / (self.x1 - self.x0)
        y_tilde = (y - self.y0) / (self.y1 - self.y0)
        x0 = torch.ones_like(x_tilde[0, 0]).expand(*x_tilde.shape) * self.x0
        x1 = torch.ones_like(x_tilde[0, 0]).expand(*x_tilde.shape) * self.x1
        Axy = (1 - x_tilde) * self.f0(y) + x_tilde * self.f1(y) \
              + (1 - y_tilde) * (self.g0(x) - ((1 - x_tilde) * self.g0(x0) + x_tilde * self.g0(x1))) \
              + y_tilde * (self.g1(x) - ((1 - x_tilde) * self.g1(x0) + x_tilde * self.g1(x1)))

        return Axy + x_tilde * (1 - x_tilde) * y_tilde * (1 - y_tilde) * output_tensor
