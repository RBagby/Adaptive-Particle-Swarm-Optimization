import numpy as np
from random import randint, random




def objective_function(vector):
	s = sum([i**2 for i in vector])

	return abs(s)


def random_vector(minmax):
	rand_vector = [minmax[i][0] + ((minmax[i][1]-minmax[i][0]) * random()) for i in range(len(minmax))]

	return rand_vector



def create_particle(search_space, vel_space):
	particle = {}
	particle['position'] = random_vector(search_space)
	particle['cost'] = objective_function(particle['position'])
	particle['best position'] = particle['position']
	particle['best cost'] = particle['cost']
	particle['velocity'] = random_vector(vel_space)

	return particle


def initialize_global_best(pop):
	gbest = pop[np.argmin([objective_function(p['position']) for p in pop])]

	return gbest



def get_global_best(population, gbest):
	best_index = np.argmin([objective_function(particle['position']) for particle in population])
	best = population[best_index]

	if best['cost'] < gbest['cost']:
		gbest['position'], gbest['cost'] = best['position'], objective_function(best['position'])

	return gbest



def constants_decay_rate(gen, max_gens, c):
        start_c = c
        end_c = 1.75
        const_c_gens = 0.85 * max_gens  # 10% of learning time
        c_decay_gens = 0.9 * max_gens  # 60% of learning time

        if gen < const_c_gens:
            return start_c

        elif gen < c_decay_gens:
            # Linear decay
            return start_c - (gen - const_c_gens) / \
                               (c_decay_gens - const_c_gens) * (start_c - end_c)
        else:
            return end_c



def max_velocity_decay_rate(gen, max_gens, max_vels):
        start_vel = max_vels
        end_vel = 1.0
        const_vel_gens = 0.1 * max_gens  # 10% 
        vel_decay_gens = 0.6 * max_gens  # 60% 

        if gen < const_vel_gens:
            return max_vels

        elif gen < vel_decay_gens:
            # Linear decay
            return start_vel - (gen - const_vel_gens) / \
                               (vel_decay_gens - const_vel_gens) * (start_vel - end_vel)
        else:
            return end_vel



def update_velocity(particle, gbest, max_v, c1, c2):
	for i,v in enumerate(particle['velocity']):
		v1 = c1 * random() * (particle['best position'][i] - particle['position'][i])
		v2 = c2 * random() * (gbest['position'][i] - particle['position'][i])
		particle['velocity'][i] = v + v1 + v2
		particle['velocity'][i] = max_v if particle['velocity'][i] > max_v else particle['velocity'][i]
		particle['velocity'][i] = -max_v if particle['velocity'][i] < -max_v else particle['velocity'][i]


def update_position(part, bounds):
	for i,p in enumerate(part['position']):
		part['position'][i] = p + part['velocity'][i]

		if part['position'][i] > bounds[i][1]:
			part['position'][i] = part['position'][i] - abs(part['position'][i] - bounds[i][1])
			part['velocity'][i] *= -1.0

		elif part['position'][i] < bounds[i][0]:
			part['position'][i] = part['position'][i] + abs(part['position'][i] - bounds[i][0])
			part['velocity'][i] *= -1.0


	part['cost'] = objective_function(part['position'])




def update_best_position(particle):
	if particle['cost'] < particle['best cost']:
		particle['best cost'] = particle['cost']
		particle['best position'] = particle['position']

	return particle



def search(max_gens, search_space, vel_space, pop_size, max_vel, c1, c2, adaptive):
	pop = [create_particle(search_space, vel_space) for p in range(pop_size)]
	gbest = initialize_global_best(pop)
	print('Starting global best has position {} and cost of {} \n'.format(gbest['position'], gbest['cost']))
	current_vel = max_vel

	for g in range(max_gens):
		print('Generation {} of {}'.format(g+1,max_gens))
		for particle in pop:
			print('Particle {} of {} \n'.format(pop.index(particle)+1, len(pop)))

			current_vel = max_velocity_decay_rate(g, max_gens, max_vel) if adaptive else max_vel
			c1 = constants_decay_rate(g, max_gens, c1) if adaptive else c1
			c2 = constants_decay_rate(g, max_gens, c2) if adaptive else c2

			update_velocity(particle, gbest, current_vel, c1, c2)
			update_position(particle, search_space)
			update_best_position(particle)

			print('Particle cost is now {} with position {} \n'.format(round(particle['cost'],5), particle['position']))
			print('=============================================================================== \n')

		gbest = get_global_best(pop, gbest)
		print('=============================================================================== \n')
		print('Global best position is now {} with cost of {} \n'.format(gbest['position'], gbest['cost']))

	return gbest


# problem configuration
problem_size = 2
search_space = [[-100000, 100000] for p in range(problem_size)]

vel_space = [[-1, 1] for p in range(problem_size)]
max_gens = 300
pop_size = 200
max_vel = 1000.0
c1 = c2 = 2.0
adaptive = True

best = search(max_gens, search_space, vel_space, pop_size, max_vel, c1, c2, adaptive)

print('Best solution has position {} and has cost {}'.format(best['position'], best['cost']))
