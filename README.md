# 🛫 Simulación de Tráfico Aéreo con Agentes Inteligentes (Q-Learning)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Completado-green)

Este proyecto implementa una **simulación multiagente del tráfico aéreo**, en la que aeropuertos y aviones cooperan y compiten por recursos limitados (pistas de aterrizaje y despegue).  
Los aeropuertos inteligentes aprenden mediante **Q-Learning** a optimizar la asignación de pistas, buscando reducir retrasos y mejorar la eficiencia global del sistema aéreo.

---

## 🚀 Características Principales

- Simulación bidimensional con múltiples aeropuertos y aviones.
- Aeropuertos inteligentes que aprenden a tomar decisiones mediante **aprendizaje por refuerzo**.
- Ciclo completo de vuelo: **espera → despegue → vuelo → aterrizaje → espera post-vuelo**.
- Métricas finales detalladas sobre rendimiento y eficiencia.
- Arquitectura extensible para incorporar más agentes o visualización.

---

## ⚙️ Configuración del Sistema

Los parámetros globales se definen al inicio del archivo `all.py`.  
A continuación se describen los principales valores configurables:

| Parámetro | Descripción | Valor por defecto |
|------------|--------------|------------------|
| `GRID_SIZE` | Tamaño del entorno simulado (x, y) | `(50, 50)` |
| `NUM_AIRPORTS` | Número total de aeropuertos | `4` |
| `NUM_PLANES` | Número total de aviones | `8` |
| `SIMULATION_MINUTES` | Duración total de la simulación (en minutos virtuales) | `500` |
| `MAX_RUNWAYS` | Máximo número de pistas por aeropuerto | `4` |
| `RUNWAY_INTERVAL` | Intervalo mínimo entre operaciones en la misma pista | `2` |
| `AVG_PLANE_SPEED` | Velocidad media de los aviones | `5` |
| `AVG_TAKEOFF_TIME` | Tiempo medio de despegue | `2` |
| `AVG_LANDING_TIME` | Tiempo medio de aterrizaje | `3` |
| `WAIT_TIME_AT_AIRPORT` | Tiempo de espera tras aterrizar antes del próximo vuelo | `2` |

---

## 🧠 Arquitectura del Sistema

El proyecto está completamente implementado en **Python** y se apoya en el paquete `autogen` para modelar agentes.  
El flujo principal se articula a través de las siguientes clases:

all.py
│
├── make_dispersion_values() # Genera valores aleatorios con una dispersión controlada
│
├── AirportAgent # Aeropuerto básico con gestión de pistas
├── RL_AirportAgent # Aeropuerto inteligente con aprendizaje Q-Learning
├── PlaneAgent # Avión con ciclo de vuelo completo
└── AirTrafficSimulation # Controlador principal que ejecuta la simulación

---

### 🧩 `RL_AirportAgent`: Aprendizaje por Refuerzo

Cada aeropuerto inteligente utiliza **Q-Learning** para decidir si asignar una pista o no, según su disponibilidad y el contexto actual.

- **Estado (`state_repr`)**: número de pistas libres.
- **Acciones**:  
  `0` → no asignar pista  
  `1` → asignar pista
- **Recompensas**:
  - `+1` si asigna correctamente una pista disponible.  
  - `-1` si intenta asignar sin disponibilidad o decide no hacerlo.
- **Ecuación de actualización Q**:

\[
Q(s,a) -> Q(s,a) + `α` [r + `γ` \max_a Q(s', a') - Q(s,a)]
\]

Donde:
- `α` es la tasa de aprendizaje (0.2 por defecto).  
- `γ` es el factor de descuento (0.95 por defecto).  
- `ε` (epsilon = 0.1) controla el equilibrio entre exploración y explotación.

---

## ▶️ Ejecución de la Simulación

### 1️⃣ Clonar el repositorio

```bash
git clone https://github.com/alvaroreyb/SMA_AirportControlAgents.git
cd <SMA_AiportControlAgents>
```
2️⃣ Instalar dependencias
```
pip install numpy autogen
```
Nota: Se recomienda usar un entorno virtual de Python (venv o conda).

3️⃣ Ejecutar la simulación
```
python all.py
```
4️⃣ Ver los resultados
Al finalizar, la simulación imprimirá un resumen con los datos agregados:

----- RESUMEN FINAL -----
Tiempo total en minutos: 500

Número total de vuelos: 8

Número de aeropuertos: 4

Dimensiones de la cuadrícula: (50, 50)

Pistas de aeropuertos -> Máx: 4, Mín: 1, Media: 2.25

Velocidad de aviones -> Máx: 7.00, Mín: 3.00, Media declarada/calculada: 5.00/4.87

Despegues por avión -> Máx: 5, Mín: 3, Media: 4.12

Aterrizajes por avión -> Máx: 5, Mín: 3, Media: 4.12

Retrasos en despegues -> Máx: 2, Mín: 0, Media: 0.62

Retrasos en aterrizajes -> Máx: 3, Mín: 0, Media: 0.75

--------------------------
📊 Estructura de Datos y Dinámica
Aeropuertos → Gestionan pistas (runways) con intervalos de seguridad (RUNWAY_INTERVAL).

Aviones → Ciclan entre dos aeropuertos (origin y dest), actualizando su estado:

waiting

taking_off

flying

landing

wait_after_landing

Simulación → Itera minuto a minuto, actualizando estados y registrando estadísticas.



📦 Dependencias
Librería	Uso principal
numpy	Cálculos numéricos y manejo de arrays
autogen	Base para la definición de agentes autónomos

Instalación:

pip install numpy autogen
🧑‍💻 Autor
Álvaro Rey
Proyecto académico sobre simulación de tráfico aéreo con agentes inteligentes.
📧 Contacto: [alvaroreyb@alvaroreyb.es]
🌍 Universidad de Málaga — Máster en Ingeniería del Software e Inteligencia Artificial

🪪 Licencia
Este proyecto se distribuye bajo licencia MIT.
Puedes usarlo, modificarlo y redistribuirlo libremente, siempre que se mantenga la atribución al autor original.


MIT License

Copyright (c) 2025 Álvaro Rey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
...

⭐ Contribuciones

Si quieres mejorar el proyecto:

Haz un fork del repositorio.

Crea una nueva rama con tu mejora (git checkout -b feature/nueva-mejora).

Haz commit de los cambios.

Abre un Pull Request con una breve descripción.

💬 Cita Recomendada

Si usas este trabajo en un contexto académico o de investigación:

Rey, Á. (2025). Simulación de tráfico aéreo con agentes inteligentes y aprendizaje por refuerzo. Univeridad de Málaga.
