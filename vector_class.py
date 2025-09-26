# -*- coding: utf-8 -*-
# gerador_zonas_criticas_final.py
# Requisitos: customtkinter, tkcalendar, geopandas, sqlalchemy, numpy, matplotlib, earthengine-api, Pillow
# pip install customtkinter tkcalendar geopandas sqlalchemy numpy matplotlib earthengine-api Pillow

import os
import ee
import geopandas as gpd
from sqlalchemy import create_engine
from datetime import date, timedelta, datetime
import json
import customtkinter as ctk
from tkinter import messagebox
from tkcalendar import Calendar
from io import BytesIO
import requests
from PIL import Image
import matplotlib.pyplot as plt


# =============================================================================
# PARÂMETROS
# =============================================================================
ID_PROP = 48
OUTPUT_DIR = rf"I:\Meu Drive\diagnostico\output\id_{ID_PROP}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUT_ZONAS_CRITICAS = os.path.join(OUTPUT_DIR, "zonas_criticas.geojson")
OUT_MAPA_ZONAS = os.path.join(OUTPUT_DIR, "mapa_de_zonas.jpeg")

PROJECT_ID = "ee-samuelsantosambientalcourse"

# Parâmetros da Análise
N_CLASSES = 3
CLASSE_CRITICA = 0
PROJ_SCALE = 10

PALETTES = {
    3: ["#d73027","#fee08b","#1a9850"], # Vermelho, Amarelo, Verde
}

# =============================================================================
# JANELA DE SELEÇÃO DE DATAS (VERSÃO CORRIGIDA COM BOTÃO DE TEXTO)
# =============================================================================
class DateSelector:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Selecionar Período")
        self.root.geometry("350x220") # Altura ajustada
        self.start_date = None
        self.end_date = None

        # --- Widgets ---
        frame = ctk.CTkFrame(self.root)
        frame.pack(pady=20, padx=20, fill="both", expand=True)

        # Data de Início
        ctk.CTkLabel(frame, text="Data de Início (DD/MM/AAAA):").grid(row=0, column=0, padx=5, pady=(10, 5), sticky="w")
        
        self.start_entry = ctk.CTkEntry(frame)
        self.start_entry.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        
        # Botão de texto "..." para abrir o calendário
        start_button = ctk.CTkButton(frame, text="...", width=30,
                                       command=lambda: self.open_calendar(self.start_entry))
        start_button.grid(row=1, column=1, padx=5, pady=5)

        # Data de Fim
        ctk.CTkLabel(frame, text="Data de Fim (DD/MM/AAAA):").grid(row=2, column=0, padx=5, pady=(10, 5), sticky="w")
        
        self.end_entry = ctk.CTkEntry(frame)
        self.end_entry.grid(row=3, column=0, padx=5, pady=5, sticky="ew")
        
        # Botão de texto "..." para abrir o calendário
        end_button = ctk.CTkButton(frame, text="...", width=30,
                                     command=lambda: self.open_calendar(self.end_entry))
        end_button.grid(row=3, column=1, padx=5, pady=5)

        # Botão de Confirmação
        confirm_button = ctk.CTkButton(frame, text="Confirmar Período", command=self.on_confirm, height=35)
        confirm_button.grid(row=4, column=0, columnspan=2, padx=5, pady=(20, 10), sticky="ew")
        
        frame.grid_columnconfigure(0, weight=1)

        # Preenche com as datas padrão
        today = date.today()
        sixty_days_ago = today - timedelta(days=60)
        self.start_entry.insert(0, sixty_days_ago.strftime('%d/%m/%Y'))
        self.end_entry.insert(0, today.strftime('%d/%m/%Y'))

    def open_calendar(self, entry_widget):
        """Abre uma janela pop-up com o calendário."""
        cal_window = ctk.CTkToplevel(self.root)
        cal_window.title("Escolha a data")
        cal_window.transient(self.root)
        cal_window.grab_set()

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
        start_str_br = self.start_entry.get()
        end_str_br = self.end_entry.get()
        try:
            start_dt = datetime.strptime(start_str_br, '%d/%m/%Y')
            end_dt = datetime.strptime(end_str_br, '%d/%m/%Y')
            self.start_date = start_dt.strftime('%Y-%m-%d')
            self.end_date = end_dt.strftime('%Y-%m-%d')
            self.root.destroy()
        except ValueError:
            messagebox.showerror("Formato Inválido", "Por favor, insira as datas no formato DD/MM/AAAA.")

    def run(self):
        self.root.mainloop()
        return self.start_date, self.end_date

# --- FUNÇÕES EE E JENKS ---
def ee_init():
    try: ee.Initialize(project=PROJECT_ID)
    except Exception: ee.Authenticate(); ee.Initialize(project=PROJECT_ID)

def mask_s2_clouds(image):
    s2_cloud_col = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
    cloud_prob_image = ee.Image(s2_cloud_col.filter(ee.Filter.eq('system:index', image.get('system:index'))).first())
    is_cloud = cloud_prob_image.select('probability').gt(50)
    return image.updateMask(is_cloud.Not())

def jenks_breaks(data, n_classes):
    import numpy as np
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

# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    # --- ETAPA 0: SELECIONAR DATAS NA JANELA ---
    date_selector = DateSelector()
    start_date_str, end_date_str = date_selector.run()

    if not start_date_str or not end_date_str:
        print("Nenhuma data selecionada. Encerrando o script.")
        exit()

    print(f"Período de busca selecionado: de {start_date_str} a {end_date_str}")
    
    ee_init()

    # --- 1) BUSCAR GEOMETRIA E MELHOR IMAGEM NO PERÍODO SELECIONADO ---
    engine = create_engine("postgresql://samuel:sertanai@localhost:5432/sertanai")
    sql = f"SELECT ST_Transform(geom, 4326) AS geom FROM formulario.diagnostico WHERE id = {ID_PROP}"
    gdf = gpd.read_postgis(sql, engine, geom_col="geom")
    if gdf.empty: raise ValueError(f"Nenhuma geometria encontrada para o ID {ID_PROP}.")
    aoi = ee.Geometry(gdf.union_all().__geo_interface__)

    data_fim = ee.Date(end_date_str)
    data_inicio = ee.Date(start_date_str)

    criteria = ee.Filter.And(ee.Filter.bounds(aoi), ee.Filter.date(data_inicio, data_fim))
    s2_sr_col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filter(criteria)
    
    if s2_sr_col.size().getInfo() == 0:
        messagebox.showerror("Nenhuma Imagem", f"Nenhuma imagem de satélite encontrada para o período de {start_date_str} a {end_date_str}.")
        exit()

    best_image = ee.Image(s2_sr_col.map(mask_s2_clouds).sort('CLOUDY_PIXEL_PERCENTAGE').first())
    image_date = ee.Date(best_image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
    print(f"Melhor imagem encontrada: {image_date}")

    # --- 2) CALCULAR NDVI E CLASSIFICAR EM ZONAS ---
    ndvi = best_image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    
    samples = ndvi.sample(region=aoi, scale=PROJ_SCALE, numPixels=5000).aggregate_array("NDVI").getInfo()
    breaks = jenks_breaks(samples, N_CLASSES)

    classified_ndvi = ee.Image(N_CLASSES - 1).toInt()
    for i in range(N_CLASSES - 2, -1, -1):
        classified_ndvi = classified_ndvi.where(ndvi.lte(breaks[i + 1]), i)
    classified_ndvi = classified_ndvi.clip(aoi).rename('class')

    # --- 3) VETORIZAR ZONAS E FILTRAR AS CRÍTICAS ---
    print("Vetorizando as zonas de manejo...")
    vetores_zonas = classified_ndvi.reduceToVectors(
        geometry=aoi, scale=PROJ_SCALE, geometryType='polygon',
        eightConnected=False, labelProperty='class'
    )
    zonas_criticas_fc = vetores_zonas.filter(ee.Filter.eq('class', CLASSE_CRITICA))

    # --- 4) EXPORTAR ARQUIVOS ---
    print(f"Exportando {zonas_criticas_fc.size().getInfo()} polígonos de zonas críticas...")
    with open(OUT_ZONAS_CRITICAS, "w") as f:
        json.dump(zonas_criticas_fc.getInfo(), f)
    print(f"✅ Zonas críticas salvas em: {OUT_ZONAS_CRITICAS}")

    # --- Salva um mapa visual (patch robusto) ---
    palette = PALETTES[N_CLASSES]

    # (1) garanta uma máscara coerente: pixels válidos do NDVI
    valid_mask = ndvi.mask().rename('mask')
    class_vis_base = classified_ndvi.updateMask(valid_mask)

    # (2) materialize em RGB com palette
    viz_img = class_vis_base.visualize(min=0, max=N_CLASSES-1, palette=palette)

    # (3) passe o ee.Geometry diretamente como region; fixe scale/dimensions
    thumb_params = {
        'region': aoi,
        'format': 'png',
        'crs': 'EPSG:4326',
        'scale': PROJ_SCALE,
    }
    url = viz_img.getThumbURL(thumb_params)

    url = viz_img.getThumbURL(thumb_params)

    r = requests.get(url)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content))

    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(img)
    ax.set_title(f'Zonas de Manejo NDVI - {image_date}')
    ax.axis('off')
    fig.savefig(OUT_MAPA_ZONAS, dpi=200, bbox_inches="tight")
    print(f"✅ Mapa de todas as zonas salvo em: {OUT_MAPA_ZONAS}")
