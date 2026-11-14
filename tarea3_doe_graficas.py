"""Tarea 3: DOE factorial 3x3 para recocido simulado TSP con 2-opt.
Basado en AYO_M4.py. Ejecutar: pip install numpy pandas scipy statsmodels matplotlib
python tarea3_doe_graficas.py --matricula 2011968
Sustituir 2026 por la matricula real antes de entregar.
"""
import argparse
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Las primeras cinco ciudades son exactamente las de AYO_M4.py de mi tarea anterior.
CITIES = {
    'A': (0, 0), 'B': (0, 3), 'E': (2, 5), 'D': (4, 3), 'C': (4, 0),
    'F': (7, 1), 'G': (8, 5), 'H': (6, 8), 'I': (2, 9), 'J': (10, 10),
    'K': (12, 3), 'L': (13, 7), 'M': (9, 13), 'N': (4, 13), 'O': (15, 12),
}
NAMES = list(CITIES)


def tour_length(route):
    return sum(math.dist(CITIES[route[i]], CITIES[route[(i + 1) % len(route)]])
               for i in range(len(route)))


def random_successor(route, rng):
    neighbor = route.copy()
    i, j = sorted(rng.sample(range(len(route)), 2))
    neighbor[i:j + 1] = reversed(neighbor[i:j + 1])
    return neighbor


def simulated_annealing(initial_route, initial_temperature, cooling_rate, seed):
    rng = random.Random(seed)
    current = initial_route.copy()
    current_distance = tour_length(current)
    best_distance = current_distance
    temperature = initial_temperature
    while temperature > 0.01:
        candidate = random_successor(current, rng)
        candidate_distance = tour_length(candidate)
        delta = candidate_distance - current_distance
        if delta <= 0 or rng.random() < math.exp(-delta / temperature):
            current, current_distance = candidate, candidate_distance
        if current_distance < best_distance:
            best_distance = current_distance
        temperature *= cooling_rate
    return best_distance


def bootstrap_median(values, seed, samples=5000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True)
    medians = np.median(draws, axis=1)
    return (float(np.median(medians)), float(np.percentile(medians, 2.5)),
            float(np.percentile(medians, 97.5)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matricula', type=int, default=2026, help='CAMBIAR por tu matricula real')
    parser.add_argument('--tmin', type=float, default=1.0)
    parser.add_argument('--tmax', type=float, default=100.0)
    parser.add_argument('--cooling', type=float, default=0.95)
    parser.add_argument('--output', default='resultados_tarea3')
    parser.add_argument('--no-show', action='store_true', help='Guardar graficas sin abrir ventanas')
    args = parser.parse_args()
    if not 0 < args.tmin < args.tmax or not 0 < args.cooling < 1:
        parser.error('Se requiere 0 < tmin < tmax y 0 < cooling < 1')
    folder = Path(args.output); folder.mkdir(parents=True, exist_ok=True)
    r = random.Random(args.matricula).random()
    levels = [(name, frac, args.tmin + frac * (args.tmax - args.tmin)) for name, frac in [
        ('Bajo', .3*r), ('Medio', .3*r+.3), ('Alto', .3*r+.6)]]
    records = []
    for n in (5, 10, 15):
        route = NAMES[:n]
        for li, (level, frac, temperature) in enumerate(levels):
            for replicate in range(10):
                seed = args.matricula * 10000 + n * 100 + li * 10 + replicate
                cost = simulated_annealing(route, temperature, args.cooling, seed)
                records.append(dict(n=n, nivel=level, temperatura=temperature,
                                    replica=replicate+1, semilla=seed, costo=cost))
    df = pd.DataFrame(records)
    df.to_csv(folder / 'corridas_90.csv', index=False)
    model = ols('costo ~ C(nivel) * C(n)', data=df).fit()
    anova = sm.stats.anova_lm(model, typ=2)
    anova.to_csv(folder / 'anova.csv')
    summaries = []
    for ni, n in enumerate((5, 10, 15)):
        for li, (level, _, temp) in enumerate(levels):
            values = df[(df.n == n) & (df.nivel == level)].costo.to_numpy()
            med, low, high = bootstrap_median(values, args.matricula + ni*10+li)
            summaries.append(dict(n=n, nivel=level, temperatura=temp, mediana=med,
                                  ic_inf=low, ic_sup=high, media=values.mean()))
    summaries = pd.DataFrame(summaries)
    summaries.to_csv(folder / 'bootstrap_por_punto.csv', index=False)
    winners = (summaries.sort_values(['n', 'mediana', 'temperatura'])
               .groupby('n', as_index=False).first().sort_values('n'))
    winners.to_csv(folder / 'temperaturas_optimas.csv', index=False)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    bytemp = df.groupby('nivel').costo.mean().reindex([x[0] for x in levels])
    axs[0].plot([x[2] for x in levels], bytemp.to_numpy(), marker='o')
    axs[0].set(xlabel='Temperatura inicial T0', ylabel='Costo medio', title='Efecto principal: temperatura')
    byn = df.groupby('n').costo.mean()
    axs[1].plot(byn.index, byn.values, marker='o')
    axs[1].set(xlabel='Numero de ciudades', ylabel='Costo medio', title='Efecto principal: tamano')
    for ax in axs: ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(folder/'efectos_principales.png', dpi=180)
    graficas = [fig]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for n in (5, 10, 15):
        sub = summaries[summaries.n == n]
        ax.plot(sub.temperatura, sub.media, marker='o', label=f'n = {n}')
    ax.set(xlabel='Temperatura inicial T0', ylabel='Costo medio',
           title='Interaccion: temperatura x tamano')
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(folder/'interaccion.png', dpi=180)
    graficas.append(fig)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    for ax, n in zip(axs, (5, 10, 15)):
        sub = summaries[summaries.n == n].set_index('nivel').loc[['Bajo', 'Medio', 'Alto']]
        x = np.arange(3)
        y = sub.mediana.to_numpy()
        err = np.vstack([y - sub.ic_inf.to_numpy(), sub.ic_sup.to_numpy() - y])
        ax.errorbar(x, y, yerr=err, fmt='o', capsize=5)
        ax.set_xticks(x, ['Bajo', 'Medio', 'Alto'])
        ax.set(title=f'n = {n}', xlabel='Temperatura', ylabel='Mediana bootstrap del costo')
        ax.grid(alpha=.3)
    fig.suptitle('Medianas bootstrap e intervalos de confianza del 95 %')
    fig.tight_layout()
    fig.savefig(folder/'bootstrap_intervalos.png', dpi=180)
    graficas.append(fig)
    print('r:', r, '\nNiveles:', levels, '\nANOVA:\n', anova,
          '\nMejores temperaturas:\n', winners.to_string(index=False))
    print('Graficas guardadas en:', folder.resolve())
    if args.no_show:
        for figura in graficas:
            plt.close(figura)
    else:
        print('Mostrando graficas. Cierra las ventanas para terminar.')
        plt.show()
    if args.matricula == 2026:
        print('AVISO: la matricula 2026 es un valor de ejemplo. Reemplazala y vuelve a ejecutar.')

if __name__ == '__main__':
    main()
