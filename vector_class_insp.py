# -*- coding: utf-8 -*-
# inspetor_zonas_criticas_refinado.py

# REQUISITOS:
# pip install customtkinter tkcalendar geopandas sqlalchemy psycopg2-binary numpy matplotlib earthengine-api Pillow ortools shapely

import os
import ee
import json
import math
import requests
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString, base
from sqlalchemy import create_engine, exc
from datetime import date, timedelta, datetime
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import customtkinter as ctk
from tkinter import messagebox
from tkcalendar import Calendar
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# =============================================================================
# PARÂMETROS GLOBAIS
# =============================================================================
ID_PROP = 48
DB_URL = "postgresql://samuel:sertanai@localhost:5432/sertanai"
PROJECT_ID = "ee-samuelsantosambientalcourse"
INTERACTIVE_START = True
N_CLASSES = 3
CLASSE_CRITICA = 0
PROJ_SCALE = 10
CRS_PLOT = "EPSG:4326"
CRS_METERS = "EPSG:3857"
OUTPUT_DIR = rf"I:\Meu Drive\prototipos\vetor_de_inspecao\output\id_{ID_PROP}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUT_ZONAS_CRITICAS = os.path.join(OUTPUT_DIR, "zonas_criticas.geojson")
OUT_MAPA_ZONAS_NDVI = os.path.join(OUTPUT_DIR, "mapa_zonas_ndvi.jpeg")
OUT_PONTOS_INSPECAO = os.path.join(OUTPUT_DIR, "pontos_inspecao_criticos.geojson")
OUT_ROTA_INSPECAO = os.path.join(OUTPUT_DIR, "rota_inspecao_critica.geojson")
OUT_MAPA_INSPECAO = os.path.join(OUTPUT_DIR, "mapa_inspecao_final.jpeg")

PALETTES = {
    3: ["#d73027", "#fee08b", "#1a9850"],
}

# =============================================================================
# JANELA DE SELEÇÃO DE DATAS (CustomTkinter)
# ... (Nenhuma alteração necessária aqui, o código é idêntico e robusto)
# =============================================================================
class DateSelector:
    # ... (código da classe DateSelector aqui, sem alterações)
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Selecionar Período de Análise")
        self.root.geometry("350x220")
        self.start_date = None
        self.end_date = None

        frame = ctk.CTkFrame(self.root)
        frame.pack(pady=20, padx=20, fill="both", expand=True)

        ctk.CTkLabel(frame, text="Data de Início (DD/MM/AAAA):").grid(row=0, column=0, padx=5, pady=(10, 5), sticky="w")
        self.start_entry = ctk.CTkEntry(frame)
        self.start_entry.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        start_button = ctk.CTkButton(frame, text="...", width=30, command=lambda: self.open_calendar(self.start_entry))
        start_button.grid(row=1, column=1, padx=5, pady=5)

        ctk.CTkLabel(frame, text="Data de Fim (DD/MM/AAAA):").grid(row=2, column=0, padx=5, pady=(10, 5), sticky="w")
        self.end_entry = ctk.CTkEntry(frame)
        self.end_entry.grid(row=3, column=0, padx=5, pady=5, sticky="ew")
        end_button = ctk.CTkButton(frame, text="...", width=30, command=lambda: self.open_calendar(self.end_entry))
        end_button.grid(row=3, column=1, padx=5, pady=5)

        confirm_button = ctk.CTkButton(frame, text="Confirmar Período", command=self.on_confirm, height=35)
        confirm_button.grid(row=4, column=0, columnspan=2, padx=5, pady=(20, 10), sticky="ew")
        
        frame.grid_columnconfigure(0, weight=1)

        today = date.today()
        sixty_days_ago = today - timedelta(days=60)
        self.start_entry.insert(0, sixty_days_ago.strftime('%d/%m/%Y'))
        self.end_entry.insert(0, today.strftime('%d/%m/%Y'))

    def open_calendar(self, entry_widget):
        cal_window = ctk.CTkToplevel(self.root)
        cal_window.title("Escolha a data")
        cal_window.transient(self.root); cal_window.grab_set()
        try:
            current_date = datetime.strptime(entry_widget.get(), '%d/%m/%Y')
            cal = Calendar(cal_window, selectmode='day', date_pattern='dd/mm/y',
                           year=current_date.year, month=current_date.month, day=current_date.day,
                           locale='pt_BR')
        except ValueError:
            cal = Calendar(cal_window, selectmode='day', date_pattern='dd/mm/y', locale='pt_BR')
        cal.pack(pady=10)
        def select_date():
            entry_widget.delete(0, 'end')
            entry_widget.insert(0, cal.get_date())
            cal_window.destroy()
        ok_button = ctk.CTkButton(cal_window, text="Selecionar", command=select_date)
        ok_button.pack(pady=10)

    def on_confirm(self):
        try:
            start_dt = datetime.strptime(self.start_entry.get(), '%d/%m/%Y')
            end_dt = datetime.strptime(self.end_entry.get(), '%d/%m/%Y')
            self.start_date = start_dt.strftime('%Y-%m-%d')
            self.end_date = end_dt.strftime('%Y-%m-%d')
            self.root.destroy()
        except ValueError:
            messagebox.showerror("Formato Inválido", "Por favor, insira as datas no formato DD/MM/AAAA.")

    def run(self):
        self.root.mainloop()
        return self.start_date, self.end_date

# =============================================================================
# FUNÇÕES AUXILIARES
# ... (Nenhuma alteração necessária aqui, as funções são as mesmas)
# =============================================================================
def ee_init():
    """Inicializa ou autentica a API do Earth Engine."""
    try:
        ee.Initialize(project=PROJECT_ID)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=PROJECT_ID)

def mask_s2_clouds(image):
    # ... (código da função mask_s2_clouds, sem alterações)
    s2_cloud_col = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
    cloud_prob_image = ee.Image(s2_cloud_col.filter(ee.Filter.eq('system:index', image.get('system:index'))).first())
    is_cloud = cloud_prob_image.select('probability').gt(50)
    return image.updateMask(is_cloud.Not())

def jenks_breaks(data, n_classes):
    # ... (código da função jenks_breaks, sem alterações)
    data = np.array([x for x in data if np.isfinite(x)], dtype=float)
    data.sort()
    if len(data) == 0: return [-1.0] + [1.0] * n_classes
    mat1 = np.zeros((len(data)+1, n_classes+1)); mat2 = np.zeros((len(data)+1, n_classes+1))
    for i in range(1, len(data)+1):
        mat1[i, 1] = np.sum((data[:i] - data[:i].mean())**2); mat2[i, 1] = 1
        for j in range(2, n_classes+1): mat1[i, j] = np.inf
    for l in range(2, len(data)+1):
        s1 = s2 = w = 0.0
        for m in range(l, 0, -1):
            v = data[m-1]; w += 1; s1 += v; s2 += v*v; sse = s2 - (s1*s1)/w
            if m > 1:
                for j in range(2, n_classes+1):
                    if mat1[m-1, j-1] + sse < mat1[l, j]: mat1[l, j] = mat1[m-1, j-1] + sse; mat2[l, j] = m
    k = len(data); brk = [0]*(n_classes+1); brk[-1] = data[-1]
    for j in range(n_classes, 0, -1):
        m = int(mat2[k, j]); brk[j-1] = data[m-2 if m > 1 else 0]; k = m-1
    return brk

def define_parametros_inspecao(area_ha: float):
    # ... (código da função define_parametros_inspecao, sem alterações)
    if area_ha <= 10: n_points = 8
    elif area_ha <= 50: n_points = int(round(max(15, area_ha / 2.5)))
    elif area_ha <= 200: n_points = int(round(max(25, area_ha / 3.5)))
    else: n_points = int(round(max(40, area_ha / 5.0)))
    print(f"Regras para {area_ha:.2f} ha de zonas críticas: Alvo de {n_points} pontos.")
    return n_points

def generate_uniform_grid_points(polygon: base.BaseGeometry, target_n_points: int) -> list[Point]:
    # ... (código da função generate_uniform_grid_points, sem alterações)
    if target_n_points == 0: return []
    xmin, ymin, xmax, ymax = polygon.bounds
    area = polygon.area
    spacing_estimate = math.sqrt(area / target_n_points)
    best_points = []; min_diff = float('inf')
    for i in range(20):
        spacing = spacing_estimate * (0.75 + (i / 19.0) * 0.5)
        x_coords = np.arange(xmin + spacing/2, xmax, spacing)
        y_coords = np.arange(ymin + spacing/2, ymax, spacing)
        grid_candidates = [Point(x, y) for x in x_coords for y in y_coords]
        points_inside = [p for p in grid_candidates if polygon.contains(p)]
        if not points_inside: continue
        current_diff = abs(len(points_inside) - target_n_points)
        if current_diff < min_diff:
            min_diff = current_diff
            best_points = points_inside
            if min_diff == 0: break
    print(f"Distribuição otimizada: Gerados {len(best_points)} pontos (alvo era {target_n_points}).")
    return best_points

def solve_tsp(points_m: list[Point]) -> list[Point]:
    # ... (código da função solve_tsp, sem alterações)
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
# FUNÇÕES DE LÓGICA PRINCIPAL (REATORAÇÃO)
# =============================================================================
def identificar_zonas_criticas(gdf_propriedade, start_date_str, end_date_str):
    """
    Executa a análise no Google Earth Engine para encontrar e vetorizar zonas críticas.

    Args:
        gdf_propriedade (gpd.GeoDataFrame): GeoDataFrame da propriedade.
        start_date_str (str): Data de início no formato 'YYYY-MM-DD'.
        end_date_str (str): Data de fim no formato 'YYYY-MM-DD'.

    Returns:
        tuple: Um GeoDataFrame das zonas críticas e a data da imagem usada, ou (None, None) se falhar.
    """
    print("--- ETAPA 1: IDENTIFICANDO ZONAS CRÍTICAS ---")
    ee_init()
    aoi = ee.Geometry(gdf_propriedade.unary_union.__geo_interface__)

    criteria = ee.Filter.And(ee.Filter.bounds(aoi), ee.Filter.date(ee.Date(start_date_str), ee.Date(end_date_str)))
    s2_sr_col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filter(criteria)

    if s2_sr_col.size().getInfo() == 0:
        messagebox.showerror("Nenhuma Imagem", f"Nenhuma imagem de satélite encontrada para o período selecionado.")
        return None, None

    best_image = ee.Image(s2_sr_col.map(mask_s2_clouds).sort('CLOUDY_PIXEL_PERCENTAGE').first())
    image_date = ee.Date(best_image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
    print(f"Melhor imagem encontrada: {image_date}")

    ndvi = best_image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    samples = ndvi.sample(region=aoi, scale=PROJ_SCALE, numPixels=5000).aggregate_array("NDVI").getInfo()
    breaks = jenks_breaks(samples, N_CLASSES)

    classified_ndvi = ee.Image(N_CLASSES - 1).toInt()
    for i in range(N_CLASSES - 2, -1, -1):
        classified_ndvi = classified_ndvi.where(ndvi.lte(breaks[i + 1]), i)
    classified_ndvi = classified_ndvi.clip(aoi).rename('class')

    print("Vetorizando as zonas de manejo...")
    vetores_zonas = classified_ndvi.reduceToVectors(geometry=aoi, scale=PROJ_SCALE, geometryType='polygon', eightConnected=False, labelProperty='class')
    zonas_criticas_fc = vetores_zonas.filter(ee.Filter.eq('class', CLASSE_CRITICA))

    num_criticas = zonas_criticas_fc.size().getInfo()
    print(f"Encontrados {num_criticas} polígonos de zonas críticas.")
    if num_criticas == 0:
        messagebox.showinfo("Tudo Certo!", "Nenhuma zona crítica foi encontrada.")
        return None, image_date

    # Salva o mapa de zonas NDVI geral
    viz_img = classified_ndvi.visualize(min=0, max=N_CLASSES-1, palette=PALETTES[N_CLASSES])
    url = viz_img.getThumbURL({'region': aoi, 'format': 'png', 'crs': 'EPSG:4326', 'scale': PROJ_SCALE})
    img = Image.open(BytesIO(requests.get(url).content))
    fig, ax = plt.subplots(figsize=(10,10)); ax.imshow(img); ax.set_title(f'Zonas de Manejo NDVI - {image_date}'); ax.axis('off')
    fig.savefig(OUT_MAPA_ZONAS_NDVI, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"✅ Mapa de zonas NDVI salvo em: {OUT_MAPA_ZONAS_NDVI}")
    
    gdf_zonas_criticas = gpd.GeoDataFrame.from_features(zonas_criticas_fc.getInfo()['features'])
    gdf_zonas_criticas.set_crs(CRS_PLOT, inplace=True)
    gdf_zonas_criticas.to_file(OUT_ZONAS_CRITICAS, driver='GeoJSON')
    print(f"✅ Zonas críticas salvas em: {OUT_ZONAS_CRITICAS}")

    return gdf_zonas_criticas, image_date

def gerar_rota_de_inspecao(gdf_propriedade, gdf_zonas_criticas, image_date):
    """
    Gera e plota a rota de inspeção otimizada para as zonas críticas.

    Args:
        gdf_propriedade (gpd.GeoDataFrame): GeoDataFrame da propriedade (para contexto no plot).
        gdf_zonas_criticas (gpd.GeoDataFrame): GeoDataFrame das zonas a serem inspecionadas.
        image_date (str): Data da imagem para usar no título do mapa.
    """
    print("\n--- ETAPA 2: GERANDO ROTA DE INSPEÇÃO PARA ZONAS CRÍTICAS ---")
    gdf_criticas_m = gdf_zonas_criticas.to_crs(CRS_METERS)
    area_ha = gdf_criticas_m.unary_union.area / 10_000.0
    n_points_target = define_parametros_inspecao(area_ha)

    talhao_critico_m = gdf_criticas_m.unary_union.buffer(0)
    sample_points_m = generate_uniform_grid_points(talhao_critico_m, n_points_target)

    if not sample_points_m:
        print("❌ Falha ao gerar pontos. A área crítica pode ser muito pequena/fragmentada.")
        return

    if INTERACTIVE_START:
        print(">> AÇÃO NECESSÁRIA: Clique no mapa para definir o INÍCIO do percurso e feche a janela.")
        fig, ax = plt.subplots(figsize=(8, 6))
        gdf_propriedade.boundary.plot(ax=ax, color='gray', linewidth=2, label='Contorno da Propriedade')
        gdf_zonas_criticas.plot(ax=ax, color='red', alpha=0.5, label='Zonas Críticas')
        gpd.GeoDataFrame(geometry=sample_points_m, crs=CRS_METERS).to_crs(CRS_PLOT).plot(ax=ax, markersize=12, color='blue')
        ax.set_title("Clique perto do ponto de partida desejado")
        ax.legend(); plt.tight_layout()
        clicks = plt.ginput(1, timeout=0)
        plt.close(fig)
        if not clicks: raise RuntimeError("Nenhum clique de partida foi capturado.")
        start_pt_m = gpd.GeoSeries([Point(clicks[0])], crs=CRS_PLOT).to_crs(CRS_METERS).iloc[0]
        start_node = min(sample_points_m, key=lambda p: p.distance(start_pt_m))
        sample_points_m.remove(start_node)
        sample_points_m.insert(0, start_node)

    print("Calculando a rota mais curta (TSP)…")
    ordered_points_m = solve_tsp(sample_points_m)

    percurso_m = LineString(ordered_points_m)
    total_distance_km = percurso_m.length / 1000.0
    print(f"Rota otimizada encontrada! Distância total: {total_distance_km:.2f} km")

    gpath = gpd.GeoDataFrame(geometry=[percurso_m], crs=CRS_METERS).to_crs(CRS_PLOT)
    gpts = gpd.GeoDataFrame(
        {"pt_id": list(range(1, len(ordered_points_m) + 1))},
        geometry=ordered_points_m, crs=CRS_METERS
    ).to_crs(CRS_PLOT)

    gpath.to_file(OUT_ROTA_INSPECAO, driver="GeoJSON")
    print(f"✅ Rota de inspeção salva em: {OUT_ROTA_INSPECAO}")
    gpts.to_file(OUT_PONTOS_INSPECAO, driver="GeoJSON")
    print(f"✅ Pontos de inspeção salvos em: {OUT_PONTOS_INSPECAO}")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal', adjustable='box')
    gdf_propriedade.boundary.plot(ax=ax, linewidth=1.5, color='black', zorder=1, label='Limite da Propriedade')
    gdf_zonas_criticas.plot(ax=ax, color='red', alpha=0.4, edgecolor='red', linewidth=0.5, zorder=2, label='Zonas Críticas')
    gpath.plot(ax=ax, linewidth=2, color='darkorange', zorder=3, label=f'Rota ({total_distance_km:.2f} km)')
    gpts.plot(ax=ax, markersize=40, color='royalblue', edgecolor='white', zorder=4)
    for _, row in gpts.iterrows():
        ax.text(row.geometry.x, row.geometry.y, str(row["pt_id"]),
                fontsize=8, ha="center", va="center", color="white", weight='bold')
    
    start_point_plot = gpts.iloc[0].geometry
    ax.plot(start_point_plot.x, start_point_plot.y, '*', color='lime', markersize=15,
            markeredgecolor='black', zorder=10, label='Início (Ponto 1)')
            
    ax.legend()
    ax.set_title(f"Rota de Inspeção Otimizada para Zonas Críticas\n(Base: Imagem de {image_date})", pad=15)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    plt.tight_layout()
    fig.savefig(OUT_MAPA_INSPECAO, dpi=200, bbox_inches="tight")
    print(f"✅ Mapa de inspeção final salvo em: {OUT_MAPA_INSPECAO}")
    
    plt.show()

# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================
def main():
    """Função principal que orquestra todo o fluxo de trabalho."""
    date_selector = DateSelector()
    start_date, end_date = date_selector.run()
    if not start_date or not end_date:
        print("Nenhuma data selecionada. Encerrando o script.")
        return

    print(f"Período selecionado: de {start_date} a {end_date}\n")

    try:
        engine = create_engine(DB_URL)
        sql = f"SELECT ST_Transform(geom, 4326) AS geom FROM formulario.diagnostico WHERE id = {ID_PROP}"
        gdf_propriedade = gpd.read_postgis(sql, engine, geom_col="geom")
        if gdf_propriedade.empty:
            raise ValueError(f"Nenhuma geometria encontrada para o ID {ID_PROP}.")
        
        # MELHORIA: Corrige geometrias potencialmente inválidas logo no início.
        gdf_propriedade['geometry'] = gdf_propriedade.geometry.buffer(0)

    except exc.OperationalError as e:
        messagebox.showerror("Erro de Conexão", f"Não foi possível conectar ao banco de dados.\nVerifique a URL e se o serviço está no ar.\n\nErro: {e}")
        return
    except Exception as e:
        messagebox.showerror("Erro", f"Ocorreu um erro ao buscar dados: {e}")
        return

    gdf_criticas, image_date = identificar_zonas_criticas(gdf_propriedade, start_date, end_date)

    if gdf_criticas is not None and not gdf_criticas.empty:
        gerar_rota_de_inspecao(gdf_propriedade, gdf_criticas, image_date)

    print("\nProcesso concluído com sucesso! 🚀")

if __name__ == "__main__":
    main()