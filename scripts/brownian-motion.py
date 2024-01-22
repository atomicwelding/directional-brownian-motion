# scripting before implementing it in cpp
import numpy as np

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from random import choices



# UTILS
#   constants are prefixed w/ C_

"""Numerical scheme"""
C_TIMESTEP  = 10_000 # number of timesteps
C_N = 200 # number of light particles
C_N_HEAVY = 20 # number of heavy particles

C_dt = 0.001

"""Box"""
C_L = 80


""" THERMOSTAT """
C_T = 0.1 # for thermostat
C_W_THERM = C_L/4
C_H_THERM = C_L
C_TIME_BEFORE_THERM = 1000

"""Lennard-Jones"""
C_EPSILON   = 0.4
C_SIGMA     = 0.3
C_SIGMA_HEAVY = 0.6


""" Particles """
C_MASS = 1
C_MASS_HEAVY = 3

C_RADIUS = C_SIGMA
C_RADIUS_HEAVY = 2

C_VELOCITIES_SCALING_FACTOR = 1


""" DISPLAY  """
C_INTERVAL = 1 # each C_INTERVAL milliseconds, buffer is redrawn 


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

        # kinetic energy
        self.T = lambda dt : 0.5 * self.mass * (self.vx[dt]**2 + self.vy[dt]**2)

class System:
    def __init__(self, N = C_N, N_HEAVY=C_N_HEAVY):
        self.particles = np.empty(N+N_HEAVY, dtype=object)
        self.init_particles(N, N_HEAVY)


    def init_particles(self, N, N_HEAVY):
        npart = N+N_HEAVY
        nbpos = int(np.sqrt(npart)) + 1

        lower = -C_L/2 + C_RADIUS
        higher = C_L/2 - C_RADIUS
        # generate a grid
        xs = np.linspace(lower, higher, nbpos)
        pos = [(x0,y0) for x0 in xs for y0 in xs]

        # normal distribution
        vp = np.random.normal(size=(npart, 2)) * C_VELOCITIES_SCALING_FACTOR

        # ensure at least one big particle that we will track 
        self.particles[0] = Particle(x0=pos[0][0],
                                          y0=pos[0][1],
                                          vx0=vp[0][0],
                                          vy0=vp[0][1],
                                          mass = C_MASS_HEAVY)
        # proportion
        TOTAL_PARTICLE = N+N_HEAVY
        for i in range(1, TOTAL_PARTICLE):
            psize = choices([-1, 1], [N_HEAVY/TOTAL_PARTICLE, N/TOTAL_PARTICLE])

            #big particle
            if(psize == -1):
                self.particles[i] = Particle(x0=pos[i][0],
                                          y0=pos[i][1],
                                          vx0=vp[i][0],
                                          vy0=vp[i][1],
                                          mass = C_MASS_HEAVY)
            #small ones
            else:
                self.particles[i] = Particle(x0=pos[i][0],
                                          y0=pos[i][1],
                                          vx0=vp[i][0],
                                          vy0=vp[i][1])
            
    
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


            #thermostat
            if(t >= C_TIME_BEFORE_THERM and particles[i].x[t+1] < (-C_L/2 + C_W_THERM)):
                if(particles[i].T(t+1) >= 1):
                    particles[i].vx[t+1] *= (1.-C_T)
                    particles[i].vy[t+1] *= (1.-C_T)

                elif(particles[i].T(t+1) <= 0.8):
                     particles[i].vx[t+1] *= (1.+C_T)
                     particles[i].vy[t+1] *= (1.+C_T)

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
    track_one_particle, = ax.plot([], [], '-', color="royalblue")

    # drawing thermostat 
    rect = mpatches.Rectangle((-C_L/2, -C_L/2), C_W_THERM, C_H_THERM, alpha = 0.1, facecolor="red")
    ax.add_patch(rect)

    
    ax.set_xlim(left=-C_L/2, right=C_L/2)
    ax.set_ylim(bottom=-C_L/2, top=C_L/2)
    plt.grid()
    print("")

    def animate(t):
        print(f"\rFrame: {t+1}/{C_TIMESTEP} ...", end="")

        # not ideal
        light_part = list(filter(lambda part : part.mass == C_MASS, solver.system.particles))
        heavy_part = list(filter(lambda part : part.mass == C_MASS_HEAVY, solver.system.particles))

        tr_x = heavy_part[0].x[0:t]
        tr_y = heavy_part[0].y[0:t]
        track_one_particle.set_data(tr_x, tr_y)
        
        # light
        x_positions = [particle.x[t] for particle in light_part]
        y_positions = [particle.y[t] for particle in light_part]
        line_light.set_data(x_positions, y_positions)

        # heavy
        x_heavy_positions = [particle.x[t] for particle in heavy_part]
        y_heavy_positions = [particle.y[t] for particle in heavy_part]
        line_heavy.set_data(x_heavy_positions, y_heavy_positions)
        
        
        
        return line_light,line_heavy,track_one_particle
    
    ani = animation.FuncAnimation(fig, animate, frames=C_TIMESTEP,
                                  interval=C_INTERVAL, blit=True, repeat=True)
    plt.show()
    print("\nAll done!")

    
    
