import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.stats import qmc

""" TODO: 
        * Hashgrids?
        * Virer disque de poisson
"""

# UTILS
#   constants are prefixed w/ C_

"""Numerical scheme"""
C_TIMESTEP  = 2000 # number of timesteps
C_N = 30 # number of light particles
C_N_HEAVY = 1 # number of heavy particles

C_dt = 0.01

"""Box"""
C_L = 10

"""Lennard-Jones"""
C_EPSILON   = 0.4
C_SIGMA     = 0.3
C_SIGMA_HEAVY = 0.6

C_VELOCITIES_SCALING_FACTOR = 1

""" Particles """
C_MASS = 1
C_MASS_HEAVY = 3

C_RADIUS = C_SIGMA
C_RADIUS_HEAVY = 2


C_POISSON_RADIUS = 1/C_N 


""" AFFICHAGE """
C_INTERVAL = 1 # toutes les X millisecondes, la fenêtre est redessinée 


# FORCES
def Flj(r, sigma = C_SIGMA, epsilon = C_EPSILON):
    return 4*epsilon * ( (12*sigma**12)/(r**13) - (6*sigma**6)/(r**7) )

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


class System:
    def __init__(self, N = C_N, N_HEAVY=C_N_HEAVY):
        self.particles = np.empty(N+N_HEAVY, dtype=object)
        self.init_particles(N, N_HEAVY)

    def init_particles(self, N, N_HEAVY):
        engine = qmc.PoissonDisk(d=2, radius=C_POISSON_RADIUS)
        pos = engine.random(n=N+N_HEAVY)
        vel = engine.random(n=N+N_HEAVY)
        for i in range(N):
             x0,y0 = pos[i]*C_L - C_L/2
             vx0,vy0 = vel[i]*C_VELOCITIES_SCALING_FACTOR # un peu crade pour generer les vitesses de depart
             self.particles[i] = Particle(x0=x0, y0=y0, vx0=vx0, vy0=vy0)

        for i in range(N, len(pos)):
            x0,y0 = pos[i]*C_L - C_L / 3
            self.particles[i] = Particle(mass=C_MASS_HEAVY, x0 = x0, y0 = y0)

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
            print(f"\rComputing: {t/(self.timestep-2)*100 :.2f}% ...", end="")
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

        # update accelerations
        for i in range(N):
            for j in range(N):
                if(i == j):
                    continue

                xdif = particles[i].x[t+1] - particles[j].x[t+1]
                ydif = particles[i].y[t+1] - particles[j].y[t+1]

                r = np.sqrt( xdif**2 + ydif**2 )

                sig = C_SIGMA if particles[i].mass == C_MASS else C_SIGMA_HEAVY
                rhs = Flj(r) / particles[i].mass
                particles[i].ax[t+1] += rhs*xdif/r
                particles[i].ay[t+1] += rhs*ydif/r

        # update velocities
        for i in range(N):
            
            particles[i].vx[t+1] = particles[i].vx[t] \
                                   + 0.5 *  ( particles[i].ax[t] + particles[i].ax[t+1]) * self.dt
            particles[i].vy[t+1] =  particles[i].vy[t] \
                                    + 0.5 *  ( particles[i].ay[t] + particles[i].ay[t+1]) * self.dt

            # wall
            if(particles[i].x[t+1] > C_L/2 or particles[i].x[t+1] < -C_L/2):
                particles[i].vx[t+1] = - particles[i].vx[t+1]
            if(particles[i].y[t+1] > C_L/2 or particles[i].y[t+1] < -C_L/2):
                particles[i].vy[t+1] = - particles[i].vy[t+1]
            
                
    

if __name__ == "__main__":
    solver = Solver()
    solver.solve()
    
    fig,ax= plt.subplots()

    line_light, = ax.plot([], [],'o', color="k", markersize = 72/C_L)
    line_heavy, = ax.plot([], [],'o', color="royalblue", markersize = C_RADIUS_HEAVY * 72/C_L)
    
    ax.set_xlim(left=-C_L/2, right=C_L/2)
    ax.set_ylim(bottom=-C_L/2, top=C_L/2)
    plt.grid()
    print("")

    def animate(t):
        print(f"\rFrame: {t+1}/{C_TIMESTEP} ...", end="")

        # couteux
        light_part = list(filter(lambda part : part.mass == C_MASS, solver.system.particles))
        heavy_part = list(filter(lambda part : part.mass == C_MASS_HEAVY, solver.system.particles))
        
        # light
        x_positions = [particle.x[t] for particle in light_part]
        y_positions = [particle.y[t] for particle in light_part]
        line_light.set_data(x_positions, y_positions)

        # heavy
        x_heavy_positions = [particle.x[t] for particle in heavy_part]
        y_heavy_positions = [particle.y[t] for particle in heavy_part]
        line_heavy.set_data(x_heavy_positions, y_heavy_positions)
        
        return line_light,line_heavy
    
    ani = animation.FuncAnimation(fig, animate, frames=C_TIMESTEP,
                                  interval=C_INTERVAL, blit=True, repeat=True)
    plt.show()
    print("\nAll done!")

    
    
