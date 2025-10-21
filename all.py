import time
import random
import autogen
import numpy as np

# CONFIGURACION GENERAL
GRID_SIZE = (50, 50)
NUM_AIRPORTS = 4
NUM_PLANES = 8
SIMULATION_MINUTES = 500
MAX_RUNWAYS = 4
RUNWAY_INTERVAL = 2
AVG_PLANE_SPEED = 5
AVG_TAKEOFF_TIME = 2
AVG_LANDING_TIME = 3
WAIT_TIME_AT_AIRPORT = 2

random.seed(42)
def make_dispersion_values(n, media, dispersion=0.3):
    vals = []
    sum_so_far = 0
    for _ in range(n - 1):
        low = max(1, int(media * (1 - dispersion)))
        high = int(media * (1 + dispersion))
        val = random.randint(low, high)
        vals.append(val)
        sum_so_far += val
        last = n * media - sum_so_far
    if last < 1:
        diff = 1 - last
        min_index = vals.index(min(vals))
        vals[min_index] = max(1, vals[min_index] - diff)
        last = 1
    vals.append(int(last))
    return vals

class AirportAgent(autogen.Agent):
    def __init__(self, a_id, position, num_runways):
        name = f"Airport ID: {a_id}"
        super().__init__(name=name)
        self.agent_name = name
        self.position = position
        self.num_runways = num_runways
        self.runways = [True] * num_runways
        self.last_op_minute = [0] * num_runways

    def request_runway(self, current_minute):
        for i in range(self.num_runways):
            if self.runways[i] and (current_minute - self.last_op_minute[i]) >= RUNWAY_INTERVAL:
                self.runways[i] = False
                self.last_op_minute[i] = current_minute
                return i
        return None

    def release_runway(self, runway_index):
        if 0 <= runway_index < self.num_runways:
            self.runways[runway_index] = True

class RL_AirportAgent(AirportAgent):
    def __init__(self, a_id, position, num_runways):
        super().__init__(a_id, position, num_runways)
        self.q_table = {}  # Para almacenar los valores Q de cada par (estado,acción)
        self.epsilon = 0.1 # En un valor bajo, ya que no queremos que tienda tanto a la exploración
        self.alpha = 0.2 # El aprendizaje para actualizar Q
        self.gamma = 0.95 # El descuento de cara a futuros Q

    def state_repr(self):
        """
        Crea una representacion del estado actual, devolvuendo una tupla de valores con el Nºde pistas libres. Este
        estado lo utilizamos para consultar y actualizar q_table. Permite que el agente sepa el número de pistas libres
        antes de decidir
        :return:
        """
        libres = sum(self.runways)
        return (libres,)

    def select_action(self, possible_actions):
        """
        Para la selección de acciones que toma el agente utilizando Q-learning. Va a calcular el estasdo inicial en
        el que se encuentra, para posteriormente con una probabilidad marcada por epsilon, elegir una accion aleatoria.
        Si sale no explorar, va a escoger la accion de mayor Q (explotación, la mejor conocida).
        Las acciones posibles van a ser 0 o 1
        :param possible_actions:
        :return:
        """
        state = self.state_repr()
        if np.random.rand() < self.epsilon:
            return np.random.choice(possible_actions)
        q_vals = [self.q_table.get((state, a), 0) for a in possible_actions]
        return possible_actions[np.argmax(q_vals)]

    def update_q(self, action, reward, next_state):
        """
        Va a actualizar el valor Q correspondiente al par de estado-accion tras tomar una decision y obtener un resultado
        Recupera el valor Q actual para el par
        Calcula el mejor Q
        Aplica la formula de Q-learning (Q(s,a)←Q(s,a)+α[r+γa′maxQ(s′,a′)−Q(s,a)] (Fuente: Tema 4 Bloque 3
        Fundamentos de la Inteligencia Artificial, Máster U. en Ing.Software e IA.
        Posteriormente guarda el nuevo valor Q
        :param action:
        :param reward:
        :param next_state:
        :return:
        """
        state = self.state_repr()
        old_q = self.q_table.get((state, action), 0)
        max_next = max([self.q_table.get((next_state, a), 0) for a in [0,1]], default=0)
        new_q = old_q + self.alpha * (reward + self.gamma * max_next - old_q)
        self.q_table[(state, action)] = new_q

    def request_runway(self, current_minute):
        """
        Con las acciones posibles (0 no dar pista, 1 dar la pista) selecciona la accion llamando a la funcion y busca
        las pistas libres en el horario y su disponibilidad)
        Si el agente consigue asignar pista y hay alguna disponible, la tomará y recibirá una recompensa positiva
        mientras que si no asigna o no puede asignar penalizará.
        Actualiza la q_table en función de la acción tomada y el resultado y devuelve el índice de la pista asignada
        (None en caso de que no se haya asignado ninguna)
        :param current_minute:
        :return:
        """
        possible_actions = [0, 1]
        action = self.select_action(possible_actions)
        libres = [i for i, r in enumerate(self.runways) if r and (current_minute - self.last_op_minute[i]) >= RUNWAY_INTERVAL]
        reward = 0
        next_state = self.state_repr()  # Para RL básico

        if action == 1 and len(libres) > 0:
            pista = libres[0]
            self.runways[pista] = False
            self.last_op_minute[pista] = current_minute
            reward = 1
            self.update_q(action, reward, next_state)
            return pista
        else:
            reward = -1
            self.update_q(action, reward, next_state)
            return None

class PlaneAgent(autogen.Agent):
    def __init__(self, p_id, origin, dest, speed, takeoff_time, landing_time):
        name = f"Plane ID: {p_id}"
        super().__init__(name=name)
        self.agent_name = name
        self.origin = origin
        self.dest = dest
        self.position = origin.position[:]
        self.state = "waiting"
        self.wait_timer = WAIT_TIME_AT_AIRPORT
        self.time_to_operation = 0
        self.delays_takeoff = 0
        self.delays_landing = 0
        self.count_takeoffs = 0
        self.count_landings = 0
        self.runway_assigned = None
        self.speed = speed
        self.takeoff_time = takeoff_time
        self.landing_time = landing_time

    def step(self, airports, current_minute):
        if self.origin is None or self.dest is None:
            raise ValueError(f"{self.agent_name}, origin o dest es None durante step.")

        if self.state == "waiting":
            if self.wait_timer > 0:
                self.wait_timer -= 1
                return
            runway_index = airports[self.origin.agent_name].request_runway(current_minute)
            if runway_index is not None:
                self.state = "taking_off"
                self.time_to_operation = self.takeoff_time
                self.runway_assigned = runway_index
            else:
                self.delays_takeoff += 1

        elif self.state == "taking_off":
            self.time_to_operation -= 1
            if self.time_to_operation <= 0:
                airports[self.origin.agent_name].release_runway(self.runway_assigned)
                self.runway_assigned = None
                self.state = "flying"
                self.target = airports[self.dest.agent_name].position[:]
                self.count_takeoffs += 1

        elif self.state == "flying":
            self.move_towards(self.target)
            if self.position == self.target:
                runway_index = airports[self.dest.agent_name].request_runway(current_minute)
                if runway_index is not None:
                    self.state = "landing"
                    self.time_to_operation = self.landing_time
                    self.runway_assigned = runway_index
                else:
                    self.delays_landing += 1

        elif self.state == "landing":
            self.time_to_operation -= 1
            if self.time_to_operation <= 0:
                airports[self.dest.agent_name].release_runway(self.runway_assigned)
                self.runway_assigned = None
                self.state = "wait_after_landing"
                self.wait_timer = WAIT_TIME_AT_AIRPORT
                self.count_landings += 1

        elif self.state == "wait_after_landing":
            if self.wait_timer > 0:
                self.wait_timer -= 1
            else:
                self.origin, self.dest = self.dest, self.origin
                self.state = "waiting"

    def move_towards(self, goal):
        for _ in range(int(round(self.speed))):  # Speed puede ser decimal
            if self.position == goal:
                break
            dx = goal[0] - self.position[0]
            dy = goal[1] - self.position[1]
            if dx != 0:
                self.position[0] += (1 if dx > 0 else -1)
            elif dy != 0:
                self.position[1] += (1 if dy > 0 else -1)


class AirTrafficSimulation:
    def __init__(self):
        self.airports = {}
        positions = set()
        while len(positions) < NUM_AIRPORTS:
            new_pos = [random.randint(0, GRID_SIZE[0] - 1), random.randint(0, GRID_SIZE[1] - 1)]
            positions.add(tuple(new_pos))
        positions = list(positions)
        for i in range(NUM_AIRPORTS):
            num_runways = random.randint(1, MAX_RUNWAYS)
            airport = RL_AirportAgent(i + 1, list(positions[i]), num_runways)
            self.airports[airport.agent_name] = airport
        self.speeds = make_dispersion_values(NUM_PLANES, AVG_PLANE_SPEED)
        self.takeoff_times = make_dispersion_values(NUM_PLANES, AVG_TAKEOFF_TIME)
        self.landing_times = make_dispersion_values(NUM_PLANES, AVG_LANDING_TIME)
        self.planes = []
        airport_names = list(self.airports.keys())
        for i in range(NUM_PLANES):
            a, b = random.sample(airport_names, 2)
            speed = self.speeds[i]
            takeoff_time = self.takeoff_times[i]
            landing_time = self.landing_times[i]
            self.planes.append(
                PlaneAgent(
                    i + 1,
                    self.airports[a],
                    self.airports[b],
                    speed=speed,
                    takeoff_time=takeoff_time,
                    landing_time=landing_time
                )
            )

        self.minute = 0

    def run(self):
        while self.minute < SIMULATION_MINUTES:
            for plane in self.planes:
                plane.step(self.airports, self.minute)
            self.minute += 1
        self.display_summary()

    def display_summary(self):
        takeoffs = [p.count_takeoffs for p in self.planes]
        landings = [p.count_landings for p in self.planes]
        takeoff_delays = [p.delays_takeoff for p in self.planes]
        landing_delays = [p.delays_landing for p in self.planes]
        num_pistas = [airport.num_runways for airport in self.airports.values()]
        speeds = self.speeds
        takeoff_times = self.takeoff_times
        landing_times = self.landing_times

        def stats(lst):
            return max(lst), min(lst), sum(lst) / len(lst) if lst else 0

        def stats(lst):
            return max(lst), min(lst), sum(lst) / len(lst) if lst else 0

        print("----- RESUMEN FINAL -----")
        print(f"Tiempo total en minutos: {self.minute}")
        print(f"Número total de vuelos: {len(self.planes)}")
        print(f"Número de aeropuertos: {len(self.airports)}")
        print(f"Número de aviones: {len(self.planes)}")
        print(f"Dimensiones de la cuadrícula: {GRID_SIZE}")
        max_p, min_p, avg_p = stats(num_pistas)
        print(f"Pistas de aeropuertos -> Máx: {max_p}, Mín: {min_p}, Media: {avg_p:.2f}")
        max_sp, min_sp, avg_sp = stats(speeds)
        print(
            f"Velocidad de aviones -> Máx: {max_sp:.2f}, Mín: {min_sp:.2f}, Media declarada/calculada: {AVG_PLANE_SPEED:.2f}/{avg_sp:.2f}")
        max_tk, min_tk, avg_tk = stats(takeoff_times)
        print(
            f"Tiempo para despegar por avión -> Máx: {max_tk:.2f}, Mín: {min_tk:.2f}, Media declarada/calculada: {AVG_TAKEOFF_TIME:.2f}/{avg_tk:.2f}")
        max_ld, min_ld, avg_ld = stats(landing_times)
        print(
            f"Tiempo de aterrizaje por avión -> Máx: {max_ld:.2f}, Mín: {min_ld:.2f}, Media declarada/calculada: {AVG_LANDING_TIME:.2f}/{avg_ld:.2f}")
        max_to, min_to, avg_to = stats(takeoffs)
        print(f"Despegues por avión -> Máx: {max_to}, Mín: {min_to}, Media: {avg_to:.2f}")
        max_la, min_la, avg_la = stats(landings)
        print(f"Aterrizajes por avión -> Máx: {max_la}, Mín: {min_la}, Media: {avg_la:.2f}")
        max_dt, min_dt, avg_dt = stats(takeoff_delays)
        print(f"Retrasos en despegues -> Máx: {max_dt}, Mín: {min_dt}, Media: {avg_dt:.2f}")
        max_dl, min_dl, avg_dl = stats(landing_delays)
        print(f"Retrasos en aterrizajes -> Máx: {max_dl}, Mín: {min_dl}, Media: {avg_dl:.2f}")
        print("--------------------------")


if __name__ == "__main__":
    sim = AirTrafficSimulation()
    sim.run()
