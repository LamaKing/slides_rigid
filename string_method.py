import numpy as np
from scipy.interpolate import interp1d

class PotentialPathAnalyt:

    def __init__(self, path, en_f, en_in):
        """
        Variable path corresponds to a path class object containing the
        coordinates of the path gamma.
        """
        self.x = path.x
        self.y = path.y
        self.pos = path.pos
        self.en_f = en_f
        self.en_in = en_in

    def update(self, path):
        self.x = path.x
        self.y = path.y
        self.pos = path.pos

    def total(self):
        """
        Mathematical form of the total potential function V along the
        path gamma.
        """
        en_path = [self.en_f(self.pos+[x,y], [x,y], *self.en_in)[0] for x,y in zip(self.x, self.y)]
        return np.array(en_path)

    def grad(self):
        """
        Mathematical form of gradient (dV/dx, dV/dy) along the path gamma.
        """

        grad_path = [self.en_f(self.pos+[x,y], [x,y], *self.en_in)[1] for x,y in zip(self.x, self.y)]
        grad_path = np.array(grad_path)
        return grad_path[:,0], grad_path[:,1]

class Path:

    def __init__(self, a, b, pos, N, fix_ends=False):
        """
        As initial guess, connect the points a and b by a line, parameterised
        with parameter t taking N discrete values between 0 and 1.
        """
        # directional vector along the
        s = np.zeros(2)
        s[0], s[1] = b[0] - a[0], b[1] - a[1]

        # parameter
        self.t = np.linspace(0, 1, N)
        self.N = N

        # line connecting a and b (linear interpolation)
        self.x = s[0]*self.t + a[0]
        self.y = s[1]*self.t + a[1]

        # partiles position in cluster
        self.pos = pos
        self.fix_ends = fix_ends

    def reparametrizeArc(self):
        """
        Equal arc parameterisation of the path.
        """
        s = np.zeros(self.N)
        for j in range(1, self.N):
            s[j] = s[j-1] + np.sqrt((self.x[j] - self.x[j-1])**2 +
                                    (self.y[j] - self.y[j-1])**2)

        t_noneq = s / s[-1]  # parameter non-equal arc spacing
        intx = interp1d(t_noneq, self.x, kind='cubic')
        inty = interp1d(t_noneq, self.y, kind='cubic')

        self.x = intx(self.t)
        self.y = inty(self.t)

    def eulerArc(self, potential, dt=1e-7):
        """
        Euler method to integrate dynamics.
        """
        # move path
        gradx, grady = potential.grad()
        # fix string ends
        if self.fix_ends:
            gradx[0], grady[0] = 0,0
            gradx[-1], grady[-1] = 0,0

        self.x += dt * gradx
        self.y += dt * grady

        # reparametrise path using equal arc parameterisation
        self.reparametrizeArc()
