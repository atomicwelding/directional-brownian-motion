import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.stats import qmc

""" TODO: 
        * Hashgrids? 
"""

# UTILS
#   constants are prefixed w/ C_

"""Numerical scheme"""
C_TIMESTEP  = 5_000 # number of timesteps
C_N = 30 # number of particles 
C_dt = 0.001

"""Box"""
C_L = 20

"""Lennard-Jones"""
C_EPSILON   = 1
C_SIGMA     = 1

""" Particles """
C_MASS = 10
C_RADIUS = 0.1
C_POISSON_RADIUS = 1/C_N


# forces
def Flj(r, sigma = C_SIGMA, epsilon = C_EPSILON):
    return 4*epsilon * ( (12*sigma**12)/(r**13) - (6*sigma**6)/(r**7) )

def Fwall(d, L = C_L):
    return 1/d**6 + 1/(L-d)**6


# CLASSES 
class Particle:
    def __init__(self,  timestep = C_TIMESTEP,
                        mass = C_MASS, radius = C_RADIUS,
                        x0  = 0,    y0  =  0,
                        vx0 = 0,    vy0 = 0,
                        ax0 = 0,    ay0 = 0 ):

        self.mass = mass
        self.radius =  radius 

        self.x = np.zeros(timestep)
        self.y = np.zeros(timestep)
    
        self.vx = np.zeros(timestep)
        self.vy = np.zeros(timestep)

        self.ax = np.zeros(timestep)
        self.ay = np.zeros(timestep)
           
        #init position
        self.x[0] = x0
        self.y[0] = y0

        #init velocities
        self.vx[0] = vx0
        self.vy[0] = vy0

        #init acceleration 
        self.ax[0] = ax0
        self.ay[0] = ay0


        # to draw
        self.color = "r"

class System:
    def __init__(self, N = C_N):
        self.particles = np.empty(N, dtype=object)
        self.init_particles(N)

    def init_particles(self, N):
        # crade
        engine = qmc.PoissonDisk(d=2, radius=C_POISSON_RADIUS)
        poss = engine.random(n=N)
        for i in range(len(poss)):
            x0,y0 = poss[i]*C_L - C_L/2
            self.particles[i] = Particle(x0=x0, y0=y0)

class Solver:
    def __init__(self,  timestep = C_TIMESTEP, dt = C_dt,
                        system = System(), scheme='verlet'):
        self.dt = dt
        self.timestep = timestep
        self.system = system

        match scheme:
            case 'verlet':
                self.solve_step = self.verlet_step
            case _:
                raise NotImplementedError

    def solve(self):
        times = np.linspace(0, self.dt*self.timestep, num = self.timestep)
        for t in range(self.timestep-1):
            print(f"{t/(self.timestep-1)*100 :.2f} %")
            self.solve_step(t)

    def verlet_step(self, t):
        """ Should implement the velocity Verlet algorithm
            Since our simulation is using a thermostat to apply gradient of temperature (kinetic energy)
            see: https://en.wikipedia.org/wiki/Verlet_integration#Velocity_Verlet
        """ 
        particles = self.system.particles
        N = len(particles)

        #update positions
        for i in range(N):
            particles[i].x[t + 1] = particles[i].x[t] \
                                    + particles[i].vx[t] * self.dt \
                                    + 0.5 * particles[i].ax[t] * self.dt**2 
            
            particles[i].y[t + 1] = particles[i].y[t] \
                                    + particles[i].vy[t] * self.dt \
                                    + 0.5 * particles[i].ay[t] * self.dt**2

        # compute acceleration
        for i in range(N):
            for j in range(N):
                if(i == j):
                    continue

                xdif = particles[i].x[t+1] - particles[j].x[t+1]
                ydif = particles[i].y[t+1] - particles[j].y[t+1]

                r = np.sqrt( xdif**2 + ydif**2 )
                
                # https://stackoverflow.com/questions/26658917/calculate-angle-between-two-vectors-atan2-discontinuity-issue
                a = particles[i].x[t+1] * particles[j].y[t+1] - particles[j].x[t+1] * particles[i].y[t+1]
                b = particles[i].x[t+1]*particles[j].x[t+1] + particles[i].y[t+1] * particles[j].y[t+1]
                theta = np.mod(np.arctan2(a,b), 2*np.pi)


                # theta = np.arctan(ydif/xdif) # trop couteux, discontinuité

                rhs = Flj(r) / particles[i].mass
                particles[i].ax[t+1] += rhs*np.cos(theta)
                particles[i].ay[t+1] += rhs*np.sin(theta)

                # wall
                particles[i].ax[t+1] +=  Fwall(np.abs(C_L - particles[i].x[t+1])) / particles[i].mass
                particles[i].ay[t+1] +=  Fwall(np.abs(C_L - particles[i].y[t+1])) / particles[i].mass

                particles[i].ax[t+1] -=  Fwall(np.abs(-C_L - particles[i].x[t+1])) / particles[i].mass
                particles[i].ay[t+1] -=  Fwall(np.abs(-C_L - particles[i].y[t+1])) / particles[i].mass
        # velocities
        for i in range(N):
            particles[i].vx[t+1] = particles[i].vx[t] \
                                   + 0.5 *  ( particles[i].ax[t] + particles[i].ax[t+1]) * self.dt
            particles[i].vy[t+1] =  particles[i].vy[t] \
                                    + 0.5 *  ( particles[i].ay[t] + particles[i].ay[t+1]) * self.dt
    

if __name__ == "__main__":
    solver = Solver()
    solver.solve()
    
    fig,ax= plt.subplots()
    line, = ax.plot([], [],'o', color="k")
    ax.set_xlim(left=-C_L/2, right=C_L/2)
    ax.set_ylim(bottom=-C_L/2, top=C_L/2)

    def animate(t):
        x_positions = [particle.x[t] for particle in solver.system.particles]
        y_positions = [particle.y[t] for particle in solver.system.particles]

        print(f"{x_positions[-1]},{y_positions[-1]}")

        line.set_data(x_positions, y_positions)
        return line,
    
    ani = animation.FuncAnimation(fig, animate, frames=C_TIMESTEP,
                                interval=1, blit=True, repeat=False)
    plt.show()
    