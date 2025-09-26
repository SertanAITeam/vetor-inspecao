# -*- coding: utf-8 -*-
# vetor_inspecao_final.py
# Requisitos: geopandas, shapely, sqlalchemy, psycopg2-binary, numpy, matplotlib, ortools
# pip install geopandas shapely sqlalchemy psycopg2-binary numpy matplotlib ortools

import os
import math
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString
from shapely.affinity import rotate, translate
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# =============================================================================
# PARÂMETROS DE CONTROLE DO USUÁRIO
# =============================================================================
DB_URL = "postgresql://samuel:sertanai@localhost:5432/sertanai"
SQL_QUERY = "SELECT geom FROM formulario.diagnostico WHERE id = 48"
CRS_PLOT = "EPSG:4326"
CRS_METERS = "EPSG:3857"          # string para consistência com GeoPandas
INTERACTIVE_START = True          # True para escolher o ponto de partida no mapa

# =============================================================================
# FUNÇÕES DE LÓGICA E REGRAS
# =============================================================================

def define_parametros_inspecao(area_ha: float):
    """
    Define nº de pontos e espaçamento das linhas-guia conforme o tamanho do talhão.
    (Ajuste livremente estas regras.)
    """
    if area_ha <= 10:         # Pequeno
        n_points = 8
        spacing_m = 30.0
    elif area_ha <= 50:       # Médio
        n_points = int(round(max(15, area_ha / 2.5)))
        spacing_m = 40.0
    elif area_ha <= 200:      # Grande
        n_points = int(round(max(25, area_ha / 3.5)))
        spacing_m = 60.0
    else:                     # Muito Grande
        n_points = int(round(max(40, area_ha / 5.0)))
        spacing_m = 80.0

    print(f"Regras para {area_ha:.1f} ha: {n_points} pontos | Linhas-guia a cada {spacing_m} m.")
    return n_points, spacing_m

def major_axis_angle(poly_m) -> float:
    """Ângulo (graus) do eixo principal do polígono. 0°=Leste, 90°=Norte, faixa [0,180)."""
    mrr = poly_m.minimum_rotated_rectangle
    xs, ys = zip(*list(mrr.exterior.coords))
    edges = [(xs[i], ys[i], xs[i+1], ys[i+1]) for i in range(4)]
    lens = [math.hypot(x2 - x1, y2 - y1) for (x1, y1, x2, y2) in edges]
    (x1, y1, x2, y2) = edges[int(np.argmax(lens))]
    ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return (ang + 180) % 180

def solve_tsp(points_m):
    """TSP: encontra a ordem que minimiza a distância total entre pontos (distância euclidiana)."""
    if len(points_m) <= 2:
        return points_m

    distance_matrix = [[int(p1.distance(p2)) for p2 in points_m] for p1 in points_m]
    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), 1, 0)  # depósito = nó 0
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        a = manager.IndexToNode(from_index)
        b = manager.IndexToNode(to_index)
        return distance_matrix[a][b]

    transit_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(params)
    if not solution:
        print("Nenhuma solução de rota encontrada! Mantendo ordem original.")
        return points_m

    order = []
    idx = routing.Start(0)
    while not routing.IsEnd(idx):
        order.append(manager.IndexToNode(idx))
        idx = solution.Value(routing.NextVar(idx))
    return [points_m[i] for i in order]

def sample_evenly_on_lines(lines, n_points: int):
    """
    Amostra 'n_points' pontos igualmente espaçados ao longo de um conjunto de linhas (lista de LineString).
    Funciona mesmo quando as linhas não são conectadas (sem usar linemerge).
    """
    parts = []
    total = 0.0
    for ln in lines:
        L = ln.length
        if L > 0:
            parts.append((ln, L))
            total += L
    if total == 0 or not parts:
        return []

    targets = np.linspace(0.0, total, n_points, endpoint=True)

    pts = []
    acc = 0.0
    pi = 0
    current_line, current_len = parts[pi]
    for t in targets:
        while acc + current_len < t and pi < len(parts) - 1:
            acc += current_len
            pi += 1
            current_line, current_len = parts[pi]
        d_local = max(0.0, min(t - acc, current_len))
        pts.append(current_line.interpolate(d_local))
    return pts

# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    # --- 1) CARREGAR POLÍGONO E APLICAR REGRAS ---
    engine = create_engine(DB_URL)
    gdf = gpd.read_postgis(SQL_QUERY, engine, geom_col="geom")
    if gdf.empty:
        raise RuntimeError("Consulta ao DB não retornou geometria.")
    if gdf.crs is None:
        gdf.set_crs(CRS_PLOT, inplace=True)

    gdf_m = gdf.to_crs(CRS_METERS)
    talhao_m = gdf_m.geometry.iloc[0]
    area_ha = talhao_m.area / 10_000.0
    n_points, spacing_m = define_parametros_inspecao(area_ha)

    # --- 2) CRIAR LINHAS-GUIA ROTACIONADAS (COBERTURA COMPLETA) ---
    print("Gerando linhas-guia…")
    theta = major_axis_angle(talhao_m)
    cx, cy = talhao_m.centroid.x, talhao_m.centroid.y

    talhao_c = translate(talhao_m, xoff=-cx, yoff=-cy)
    talhao_r = rotate(talhao_c, -theta, origin=(0, 0), use_radians=False)
    rxmin, rymin, rxmax, rymax = talhao_r.bounds
    rheight = rymax - rymin

    n_lines = int(math.floor(rheight / spacing_m)) + 1
    ys = np.linspace(rymin + spacing_m / 2.0, rymax - spacing_m / 2.0, n_lines) if n_lines > 1 else [(rymin + rymax) / 2.0]

    guide_lines = []
    for y in ys:
        # cria linha horizontal no sistema rotacionado e volta ao original
        rline = LineString([(rxmin - 10 * spacing_m, y), (rxmax + 10 * spacing_m, y)])
        line = rotate(rline, theta, origin=(0, 0), use_radians=False)
        line = translate(line, xoff=cx, yoff=cy)

        # recorta no talhão
        cut = line.intersection(talhao_m)
        if cut.is_empty:
            continue
        if isinstance(cut, LineString):
            guide_lines.append(cut)
        elif isinstance(cut, MultiLineString):
            guide_lines.extend([seg for seg in cut.geoms if seg.length > 0])

    if not guide_lines:
        raise ValueError("Não foi possível criar linhas-guia; verifique o polígono/CRS.")

    # --- 3) AMOSTRAR PONTOS AO LONGO DAS LINHAS (ROBUSTO A MULTI) ---
    print("Distribuindo pontos de amostragem no talhão…")
    sample_points_m = sample_evenly_on_lines(guide_lines, n_points)
    if not sample_points_m:
        raise RuntimeError("Falha ao amostrar pontos nas linhas-guia.")

    # --- 4) DEFINIR PONTO DE PARTIDA E OTIMIZAR ROTA (TSP) ---
    if INTERACTIVE_START:
        print("Clique no mapa perto do ponto onde deseja INICIAR o percurso.")
        gdf_ll = gdf_m.to_crs(CRS_PLOT)
        fig, ax = plt.subplots(figsize=(8, 6))
        gdf_ll.boundary.plot(ax=ax, color='gray')
        gpd.GeoDataFrame(geometry=sample_points_m, crs=CRS_METERS).to_crs(CRS_PLOT).plot(ax=ax, markersize=12)
        ax.set_title("Clique perto do ponto de partida desejado e feche a janela")
        plt.tight_layout()
        clicks = plt.ginput(1, timeout=0)
        plt.close(fig)
        if not clicks:
            raise RuntimeError("Nenhum clique de partida foi capturado.")
        start_pt_m = gpd.GeoSeries([Point(clicks[0])], crs=CRS_PLOT).to_crs(CRS_METERS).iloc[0]
        # fixa o 1º nó como o ponto mais próximo ao clique
        start_node = min(sample_points_m, key=lambda p: p.distance(start_pt_m))
        sample_points_m.remove(start_node)
        sample_points_m.insert(0, start_node)

    print("Calculando a rota mais curta (TSP)…")
    ordered_points_m = solve_tsp(sample_points_m)

    # --- 5) EXPORTAR E PLOTAR RESULTADOS (VERSÃO CORRIGIDA) ---
    percurso_m = LineString(ordered_points_m)
    total_distance_km = percurso_m.length / 1000.0
    print(f"Rota otimizada encontrada! Distância total: {total_distance_km:.2f} km")

    os.makedirs("output", exist_ok=True)

    gpath = gpd.GeoDataFrame(geometry=[percurso_m], crs=CRS_METERS).to_crs(CRS_PLOT)
    gpts = gpd.GeoDataFrame(
        {"pt_id": list(range(1, len(ordered_points_m) + 1))},
        geometry=ordered_points_m,
        crs=CRS_METERS
    ).to_crs(CRS_PLOT)

    gpath.to_file("output/percurso_final_otimizado.geojson", driver="GeoJSON")
    gpts.to_file("output/pontos_final_otimizados.geojson", driver="GeoJSON")

    # --- BLOCO DE PLOTAGEM CORRIGIDO ---
    fig, ax = plt.subplots(figsize=(10, 10))

    # Define o aspecto primeiro, pois ele influencia todo o resto
    ax.set_aspect('equal', adjustable='box')

    # Plota os dados
    gdf_m.to_crs(CRS_PLOT).boundary.plot(ax=ax, linewidth=1.5, color='black', zorder=1)
    gpath.plot(ax=ax, linewidth=2, color='darkorange', zorder=2)
    gpts.plot(ax=ax, markersize=40, color='royalblue', edgecolor='white', zorder=3)

    # Plota os textos e a legenda
    for _, row in gpts.iterrows():
        ax.text(row.geometry.x, row.geometry.y, str(row["pt_id"]),
                fontsize=8, ha="center", va="center", color="black", weight='bold')

    start_point_plot = gpts.iloc[0].geometry
    ax.plot(start_point_plot.x, start_point_plot.y, '*', color='lime', markersize=15,
            markeredgecolor='black', zorder=10, label='Início')
    ax.legend()

    # Define os rótulos e o título
    ax.set_title("Percurso de inspeção (Menor Distância entre Pontos)", pad=15) # 'pad' controla o espaço
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Comando final para exibir o gráfico sem cortes
    plt.show()