# -*- coding: utf-8 -*-
# vetor_inspecao_grade_uniforme.py
# Requisitos: geopandas, shapely, sqlalchemy, psycopg2-binary, numpy, matplotlib, ortools
# pip install geopandas shapely sqlalchemy psycopg2-binary numpy matplotlib ortools

import os
import math
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString
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
CRS_METERS = "EPSG:3857"
INTERACTIVE_START = True

# =============================================================================
# FUNÇÕES DE LÓGICA E REGRAS
# =============================================================================

def define_parametros_inspecao(area_ha: float):
    """Define o número alvo de pontos com base no tamanho do talhão."""
    if area_ha <= 10:
        n_points = 8
    elif area_ha <= 50:
        n_points = int(round(max(15, area_ha / 2.5)))
    elif area_ha <= 200:
        n_points = int(round(max(25, area_ha / 3.5)))
    else:
        n_points = int(round(max(40, area_ha / 5.0)))
    print(f"Regras para {area_ha:.1f} ha: Alvo de {n_points} pontos.")
    return n_points

def generate_uniform_grid_points(polygon, target_n_points: int):
    """
    Gera uma grade de pontos uniforme, ajustando o espaçamento para se
    aproximar do número de pontos alvo. Esta é a nova função de distribuição.
    """
    if target_n_points == 0:
        return []

    xmin, ymin, xmax, ymax = polygon.bounds
    area = polygon.area
    
    # Estimativa inicial para o espaçamento da grade
    spacing_estimate = math.sqrt(area / target_n_points)
    
    best_points = []
    # A diferença mínima encontrada entre o número de pontos gerados e o alvo
    min_diff = float('inf')

    # Tenta vários espaçamentos próximos da estimativa para encontrar o melhor ajuste
    for i in range(20):
        # Varia o espaçamento em +/- 25% da estimativa inicial
        spacing = spacing_estimate * (0.75 + (i / 19.0) * 0.5)
        
        x_coords = np.arange(xmin + spacing/2, xmax, spacing)
        y_coords = np.arange(ymin + spacing/2, ymax, spacing)
        
        # Gera os pontos candidatos da grade
        grid_candidates = [Point(x, y) for x in x_coords for y in y_coords]
        
        # Filtra apenas os pontos que estão dentro do polígono
        points_inside = [p for p in grid_candidates if polygon.contains(p)]
        
        if not points_inside:
            continue
            
        current_diff = abs(len(points_inside) - target_n_points)
        
        # Se esta tentativa gerou um número de pontos mais próximo do alvo, guardamos o resultado
        if current_diff < min_diff:
            min_diff = current_diff
            best_points = points_inside
            # Se acertamos o alvo em cheio, podemos parar
            if min_diff == 0:
                break

    print(f"Distribuição otimizada: Gerados {len(best_points)} pontos (alvo era {target_n_points}).")
    return best_points

def solve_tsp(points_m):
    """Encontra a rota mais curta que visita todos os pontos (TSP)."""
    if len(points_m) <= 2: return points_m
    distance_matrix = [[int(p1.distance(p2)) for p2 in points_m] for p1 in points_m]
    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    def distance_callback(from_index, to_index):
        a = manager.IndexToNode(from_index); b = manager.IndexToNode(to_index)
        return distance_matrix[a][b]
    transit_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    solution = routing.SolveWithParameters(params)
    if not solution: return points_m
    order = []; idx = routing.Start(0)
    while not routing.IsEnd(idx):
        order.append(manager.IndexToNode(idx)); idx = solution.Value(routing.NextVar(idx))
    return [points_m[i] for i in order]

# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    # --- 1) CARREGAR POLÍGONO E APLICAR REGRAS ---
    engine = create_engine(DB_URL)
    gdf = gpd.read_postgis(SQL_QUERY, engine, geom_col="geom")
    if gdf.empty: raise RuntimeError("Consulta ao DB não retornou geometria.")
    if gdf.crs is None: gdf.set_crs(CRS_PLOT, inplace=True)

    gdf_m = gdf.to_crs(CRS_METERS)
    talhao_m = gdf_m.geometry.iloc[0].buffer(0) # Corrige geometrias inválidas
    area_ha = talhao_m.area / 10_000.0
    n_points_target = define_parametros_inspecao(area_ha)

    # --- 2) GERAR PONTOS COM DISTRIBUIÇÃO UNIFORME (NOVA LÓGICA) ---
    sample_points_m = generate_uniform_grid_points(talhao_m, n_points_target)
    if not sample_points_m:
        raise RuntimeError("Falha ao gerar pontos de amostragem na grade.")

    # --- 3) DEFINIR PONTO DE PARTIDA E OTIMIZAR ROTA (TSP) ---
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
        if not clicks: raise RuntimeError("Nenhum clique de partida foi capturado.")
        start_pt_m = gpd.GeoSeries([Point(clicks[0])], crs=CRS_PLOT).to_crs(CRS_METERS).iloc[0]
        start_node = min(sample_points_m, key=lambda p: p.distance(start_pt_m))
        sample_points_m.remove(start_node)
        sample_points_m.insert(0, start_node)

    print("Calculando a rota mais curta (TSP)…")
    ordered_points_m = solve_tsp(sample_points_m)

    # --- 4) EXPORTAR E PLOTAR RESULTADOS ---
    percurso_m = LineString(ordered_points_m)
    total_distance_km = percurso_m.length / 1000.0
    print(f"Rota otimizada encontrada! Distância total: {total_distance_km:.2f} km")
    os.makedirs("output", exist_ok=True)
    gpath = gpd.GeoDataFrame(geometry=[percurso_m], crs=CRS_METERS).to_crs(CRS_PLOT)
    gpts = gpd.GeoDataFrame(
        {"pt_id": list(range(1, len(ordered_points_m) + 1))},
        geometry=ordered_points_m, crs=CRS_METERS
    ).to_crs(CRS_PLOT)
    gpath.to_file("output/percurso_grade_otimizado.geojson", driver="GeoJSON")
    gpts.to_file("output/pontos_grade_otimizados.geojson", driver="GeoJSON")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal', adjustable='box')
    gdf_m.to_crs(CRS_PLOT).boundary.plot(ax=ax, linewidth=1.5, color='black', zorder=1)
    gpath.plot(ax=ax, linewidth=2, color='darkorange', zorder=2)
    gpts.plot(ax=ax, markersize=40, color='royalblue', edgecolor='white', zorder=3)
    for _, row in gpts.iterrows():
        ax.text(row.geometry.x, row.geometry.y, str(row["pt_id"]),
                fontsize=8, ha="center", va="center", color="black", weight='bold')
    start_point_plot = gpts.iloc[0].geometry
    ax.plot(start_point_plot.x, start_point_plot.y, '*', color='lime', markersize=15,
            markeredgecolor='black', zorder=10, label='Início')
    ax.legend()
    ax.set_title("Percurso Otimizado (Grade Uniforme)", pad=15)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    plt.show()