import numpy 
import time
from numpy import *
from pylab import *   
from scipy.stats import norm
import matplotlib.mlab as mlab 
from math import sqrt

class MolecularDynamics:

	"""Class that describes the molecular dynamics of a gas of atoms in units where m = epsilon = sigma = kb = 1"""
	
	dt = 0.001 # time increment
	sampleInterval = 100
	
	

	def __init__(self, N=4, L=10.0, initialTemperature=0.0, initialAngularMomentum=0.0):
	
		numpy.random.seed(219) # random number generator used for initial velocities (and sometimes positions) 
		
		self.N = N  # number of particles 
		self.L = L	# length of square side 
		self.initialTemperature = initialTemperature
		self.initialAngularMomentum = initialAngularMomentum
		
		self.t = 0.0 # initial time
		self.tArray = array([self.t]) # array of time steps that is added to during integration
		self.steps = 0
		
		self.EnergyArray = array([]) # list of energy, sampled every sampleInterval time steps
		self.sampleTimeArray = array([])
		self.angularMomentumArray = array([])
		
		# accumulate statistics during time evolution
		self.temperatureArray = array([self.initialTemperature])
		self.temperatureAccumulator = 0.0
		self.angularMomentumArray = array([self.initialAngularMomentum])
		self.angularMomentumAccumulator = 0.0
		
		self.squareTemperatureAccumulator = 0.0


		self.virialAccumulator = 0.0

		self.x = zeros(2*N) # NumPy array of N (x, y) positions
		self.v = zeros(2*N) # array of N (vx, vy) velocities

		self.xArray = array([]) # particle positions that is added to during integration
		self.vArray = array([]) # particle velocities
		
		#lzero=abs((self.xArray[0])*(self.vArray[1])-(self.xArray[1])*(self.vArray[0]))

		self.forceType = "lennardJones"


	
	def minimumImage(self, x): # minimum image approximation (Gould Listing 8.2)
	
		L = self.L
		halfL = 0.5 * L
		
		return (x + halfL) % L - halfL

	
	def force(self): 
	
		if (self.forceType == "lennardJones"):
			f, virial = self.lennardJonesForce()
						
		if (self.forceType == "powerLaw"):
			f, virial = self.powerLawForce()
					
		self.virialAccumulator += virial
		
		return f
		
	
	
	def lennardJonesForce(self): # Gould Eq. 8.3 (NumPy vector form which is faster)
	
		N = self.N
		virial = 0.0
		tiny = 1.0e-40 # prevents division by zero in calculation of self-force
		L = self.L
		halfL = 0.5 * L
		
		x = self.x[arange(0, 2*N, 2)]
		y = self.x[arange(1, 2*N, 2)]	
		f = zeros(2*N)
		
		minimumImage = self.minimumImage
		
		for i in range(N):  # The O(N**2) calculation that slows everything down SOS: ACCELERATE THIS LOOP
		
			dx = minimumImage(x[i] - x)
			dy = minimumImage(y[i] - y)
			
			r2inv = 1.0/(dx**2 + dy**2 + tiny)
			c = 48.0 * r2inv**7 - 24.0 * r2inv**4 
			fx = dx * c
			fy = dy * c
			
			fx[i] = 0.0 # no self force
			fy[i] = 0.0
			f[2*i] = fx.sum()
			f[2*i+1] = fy.sum()
			
			virial += dot(fx, dx) + dot(fy, dy)
				
		return f, 0.5*virial
		
	
	def powerLawForce(self): 
	
		N = self.N
		virial = 0.0
		tiny = 1.0e-40 # prevents division by zero in calculation of self-force
		halfL = 0.5 * self.L
		
		x = self.x[arange(0, 2*N, 2)]
		y = self.x[arange(1, 2*N, 2)]	
		f = zeros(2*N)
		minimumImage = self.minimumImage
		for i in range(N):  # The O(N**2) calculation that slows everything down SOS: ACCELERATE THIS LOOP
		
			dx = minimumImage(x[i] - x)
			dy = minimumImage(y[i] - y)
			
			r2 = dx**2 + dy**2 + tiny
			r6inv = pow(r2, -3)
			fx = dx * r6inv
			fy = dy * r6inv
			
			fx[i] = 0.0 # no self force
			fy[i] = 0.0
			f[2*i] = fx.sum()
			f[2*i+1] = fy.sum()
			
			virial += dot(fx, dx) + dot(fy, dy)	
			
		return f, 0.5 * virial 


# TIME EVOLUTION METHODS 

	def verletStep(self): # Gould Eqs. 8.4a and 8.4b
	
		a = self.force()
		self.x += self.v * self.dt + 0.5 * self.dt**2 * a
		self.x = self.x % self.L 	# periodic boundary conditions
		self.v += 0.5 * self.dt * (a + self.force())
				
		
	def evolve(self, time=10.0):
	
		steps = int(abs(time/self.dt))
		for i in range(steps):
		
			self.verletStep()
			self.zeroTotalMomentum()
			
			self.t += self.dt
			self.tArray = append(self.tArray, self.t)
			
			if (i % self.sampleInterval == 0): # only calculate energy every sampleInterval steps to reduce load
				self.EnergyArray = append(self.EnergyArray, self.energy())
				self.sampleTimeArray = append(self.sampleTimeArray, self.t)
				self.xArray = append(self.xArray, self.x)
				self.vArray = append(self.vArray, self.v)
			
			#self.angularMomentumArray = append(self.angularMomentumArray, self.angularMomentum())
			#self.sampleTimeArray = append(self.sampleTimeArray, self.t)
			
			T = self.temperature()
			self.steps += 1
			self.temperatureArray = append(self.temperatureArray, T)
			self.temperatureAccumulator += T
			self.squareTemperatureAccumulator += T*T

			L = self.angularMomentum()
			self.steps += 1
			self.angularMomentumArray = append(self.angularMomentumArray, L)
			self.angularMomentumAccumulator += L
			
			
	def zeroTotalMomentum(self):
	
		vx = self.v[arange(0, 2*self.N, 2)]
		vy = self.v[arange(1, 2*self.N, 2)]

		vx -= vx.mean() # zero mean momentum
		vy -= vy.mean()
		
		self.v[arange(0, 2*self.N, 2)] = vx
		self.v[arange(1, 2*self.N, 2)] = vy

			
	def reverseTime(self):
	
		self.dt = -self.dt
			
			
	def cool(self, time=1.0):
	
		steps = int(time/self.dt)
		for i in range(steps):
			self.verletStep()
			self.v *= (1.0 - self.dt) # friction slows down atoms
			
		self.resetStatistics()

			

# INITIAL CONDITION METHODS		
		
	def randomPositions(self):
	
		self.x = self.L * numpy.random.random(2*self.N)
		
		self.forceType = "powerLaw" 
		self.cool(time=1.0)
		self.forceType = "lennardJones"


	def triangularLatticePositions(self):
	
		#self.rectangularLatticePositions()
		self.randomPositions()
		self.v += numpy.random.random(2*self.N) - 0.5 # jiggle to break symmetry
		
		self.forceType = "powerLaw" 
		self.cool(time=10.0)
		self.forceType = "lennardJones"


	def rectangularLatticePositions(self): # assume that N is a square integer (4, 16, 64, ...)
	
		nx = int(sqrt(self.N))
		ny = nx
		dx = self.L / nx
		dy = self.L / ny
		
		for i in range(nx):
			x = (i + 0.5) * dx
			for j in range(ny):
				y = (j + 0.5) * dy
				self.x[2*(i*ny+j)] = x
				self.x[2*(i*ny+j)+1] = y
				
		
	def randomVelocities(self):
	
		self.v = numpy.random.random(2*self.N) - 0.5
	
		self.zeroTotalMomentum()
		
		T = self.temperature()
		self.v *= sqrt(self.initialTemperature/T)
		
		
		
# MEASUREMENT METHODS

	

	def kineticEnergy(self):
	
		return 0.5 * (self.v * self.v).sum()
		
	
	def potentialEnergy(self):
	
		#return self.weaveLennardJonesPotentialEnergy()
		return self.lennardJonesPotentialEnergy()
		
	
		
	def lennardJonesPotentialEnergy(self): # Gould Eqs. 8.1 and 8.2
	
		tiny = 1.0e-40 # prevents division by zero in calculation of self-force
		halfL = 0.5 * self.L
		N = self.N
		
		x = self.x[arange(0, 2*N, 2)]
		y = self.x[arange(1, 2*N, 2)]	
		U = 0.0
		minimumImage = self.minimumImage
		for i in range(N):  # The O(N**2) calculation that slows everything down SOS: ACCELERATE THIS LOOP
		
			dx = minimumImage(x[i] - x)
			dy = minimumImage(y[i] - y)

			r2inv = 1.0/(dx**2 + dy**2 + tiny)
			dU = r2inv**6 - r2inv**3
			dU[i] = 0.0 # no self-interaction
			U += dU.sum()

		return 2.0 * U
		
		
			
	def energy(self):
	
		return self.potentialEnergy() + self.kineticEnergy()
		
	def temperature(self): # Gould Eq. 8.6
	
		return self.kineticEnergy() / self.N
		
	def angularMomentum(self): # Calculate z component of angular momentum using definition
		
		vx = self.v[arange(0, 2*self.N, 2)]
		vy = self.v[arange(1, 2*self.N, 2)]

		x = self.x[arange(0, 2*self.N, 2)]
		y = self.x[arange(1, 2*self.N, 2)]
		
		return (x*vy-y*vx).sum() 

# STATISTICS METHODS		
		
	def resetStatistics(self):
	
		self.steps = 0
		self.temperatureAccumulator = 0.0
		self.angularMomentumAccumulator = 0.0
		self.squareTemperatureAccumulator = 0.0
		self.squareaangularMomentumAccumulator = 0.0
		self.virialAccumulator = 0.0
		self.xArray = array([])
		self.vArray = array([])

		
	def meanTemperature(self):
	
		return self.temperatureAccumulator / self.steps

	def meanangularMomentum(self):

		return self.angularMomentumAccumulator / self.steps
	
	def meanSquareTemperature(self):
		
		return self.squareTemperatureAccumulator / self.steps

	def meanSquareangularMomentum(self):

		return self.squareaangularMomentumAccumulator / self.steps
		
		
	def meanPressure(self): # Gould Eq. 8.9
	
		meanVirial = 0.5 * self.virialAccumulator / self.steps # divide by 2 because force is calculated twice per step
		return 1.0 + 0.5 * meanVirial / (self.N * self.meanTemperature())
		
		
	def heatCapacity(self): # Gould Eq. 8.12
	
		meanTemperature = self.meanTemperature()
		meanSquareTemperature = self.meanSquareTemperature()
		sigma2 = meanSquareTemperature - meanTemperature**2
		denom = 1.0 - sigma2 * self.N / meanTemperature**2
		return self.N / denom

	def fluctuations(self):

		meanangularMomentum = self.meanangularMomentum()
		meanSquareangularMomentum = self.meanSquareangularMomentum()
		fluct = meanSquareangularMomentum - meanangularMomentum**2

		return fluct
	
	
	def meanEnergy(self):
	
		return self.EnergyArray.mean()

	
	
	def stdEnergy(self):
	
		return self.EnergyArray.std()
		
# PLOTTING METHODS
	
	def plotPositions(self):
	
		figure(1)
		scatter(self.x[arange(0, 2*self.N, 2)], self.x[arange(1, 2*self.N, 2)], s=5.0, marker='o', alpha=1.0)
		xlabel("x")
		ylabel("y")
	
	def plotTrajectories(self, number=1):
		
		figure(2)
		xlabel("x")
		ylabel("y")
		N = self.N
		size = len(self.xArray)/(2*N) # For python3 this needs to be corrected
		r = reshape(self.xArray, [size, 2*N])
		for i in range(number):
			x = r[:, 2*i]
			y = r[:, 2*i+1]
			plot(x, y, ".")

	def plotTemperature(self):
		
		figure(3)
		plot(self.tArray, self.temperatureArray)
		xlabel("time")
		ylabel("temperature")

	def plotAngularMomentum(self):

		figure(6)
		plot(self.tArray, self.angularMomentumArray)
		xlabel("time")
		ylabel("Angular Momentum")

	def plotEnergy(self):
		
		figure(4)
		plot(self.sampleTimeArray, self.EnergyArray)
		xlabel("time")
		ylabel("Energy")	

	def velocityHistogram(self):
		
		figure(5)
		hist(self.vArray, bins=100, normed=1)
		xlabel("velocity in x- or y-directions")
		ylabel("probability")

	def angularMomentumHistogram(self):

		figure(7)
		hist(self.angularMomentumArray, bins=100, normed=1)
		xlabel('angular momentum')
		ylabel('Probability')

		mu = mean(self.angularMomentumArray)
		variance = var(self.angularMomentumArray)
		sigma = sqrt(variance)
		print("Sigma of the Gaussian Distribution is : ",sigma)

		t = linspace(min(self.angularMomentumArray), max(self.angularMomentumArray), 100)
		plot(t, mlab.normpdf(t, mu, sigma))

	def showPlots(self):
		
		show()	
		
	
# RESULTS METHODS

	def results(self):
		print("\n\nRESULTS\n") 
		print("time = ", md.t, " total energy = ", md.energy(), " and temperature = ", md.temperature())
		if (self.steps > 0):
			print("Mean energy = ", md.meanEnergy(), " and standard deviation = ", md.stdEnergy())
			print("Cv = ", md.heatCapacity(), " and PV/NkT = ", md.meanPressure())
			#print(" Sigma of the Gaussian Distribution is : ", md.

		
start = time.time()
md = MolecularDynamics(N=16, L=4, initialTemperature=5.0, initialAngularMomentum=0.0) # instantiate object
#md = MolecularDynamics(N=16, L=4, initialTemperature=5.0) # instantiate object
#md = MolecularDynamics(N=64, L=16, initialTemperature=1.0) # instantiate object
#md = MolecularDynamics(N=256, L=16, initialTemperature=1.0) # instantiate object
# EQUILIBRATION AND STATISTICS
md.triangularLatticePositions()
md.randomVelocities()
md.plotPositions()
md.results()
md.evolve(time=10.0) # initial time evolution
md.resetStatistics() # remove transient behavior
md.evolve(time=20.0) # accumulate statistics 
md.results()
md.angularMomentum()

end = time.time()

print('Time is %.2f'%(end-start))

md.plotEnergy()
md.plotTrajectories(md.N)
md.plotTemperature()
md.plotAngularMomentum()
md.velocityHistogram()
md.angularMomentumHistogram()
md.showPlots()
