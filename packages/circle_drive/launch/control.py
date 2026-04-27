class MPCController:
    def __init__(self, horizon=15, dt=0.1,
                 max_linear_velocity=0.3,
                 max_angular_velocity=8.0):

        self.N = horizon
        self.dt = dt
        self.max_linear_v = max_linear_velocity
        self.max_angular_v = max_angular_velocity

        self.candidates = np.linspace(-max_angular_velocity,
                                      max_angular_velocity, 31)

    def simulate_cost(self, h0, a0, omega):
        h = h0
        a = a0
        cost = 0.0
    
        for _ in range(self.N):
            a = a - omega * self.dt
            h = h + a * self.dt
    
            cost += (
                self.q_h * h * h +
                self.q_a * a * a +
                self.r * omega * omega
            )
    
        return cost

    def update(self, horizontal_error, angular_error):
        best_cost = 1e9
        best_u = 0.0

        for omega in self.candidates:
            c = self.simulate_cost(horizontal_error,
                                   angular_error,
                                   omega)
            if c < best_cost:
                best_cost = c
                best_u = omega

        return self.max_linear_v, float(best_u)
