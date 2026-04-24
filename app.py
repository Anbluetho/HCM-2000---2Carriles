import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# ==========================================
# Configuración inicial de la página
# ==========================================
st.set_page_config(
    page_title="HCM 2000 - LOS Carreteras de 2 Carriles",
    page_icon="🛣️",
    layout="wide"
)

# ==========================================
# Funciones de Cálculo (Metodología HCM - Exhibit 20-1)
# ==========================================

def get_f_LS(lane_width, shoulder_width):
    """Ajuste por ancho de carril y espaldón (Exhibit 20-5) usando rangos fijos (escalones)"""
    if lane_width < 3.0:
        row = 0
    elif lane_width < 3.3:
        row = 1
    elif lane_width < 3.6:
        row = 2
    else:
        row = 3
        
    if shoulder_width < 0.6:
        col = 0
    elif shoulder_width < 1.2:
        col = 1
    elif shoulder_width < 1.8:
        col = 2
    else:
        col = 3
        
    tabla_fLS = [
        [10.3, 7.7, 5.6, 3.5], # Carriles angostos (<3.0)
        [ 8.5, 5.9, 3.8, 1.7], # Carriles 3.0 a <3.3
        [ 7.5, 4.9, 2.8, 0.7], # Carriles 3.3 a <3.6
        [ 6.8, 4.2, 2.1, 0.0]  # Carriles ideales (>=3.6)
    ]
    
    return tabla_fLS[row][col]

# ==============================================================================
# CÁLCULO EXACTO DE f_A (Densidad de Accesos - Método Matemático)
# ==============================================================================
def get_f_A(accesos_por_km):
    """Ajuste por densidad de accesos (Exhibit 20-6)"""
    # Asegurarnos de que el valor sea un número (float)
    accesos = float(accesos_por_km)
    
    # El límite máximo de la tabla es 24 accesos (reducción de 16 km/h)
    if accesos >= 24.0:
        return 16.0
    else:
        # Relación lineal exacta del HCM 2000: (4.0 km/h por cada 6 accesos)
        return accesos * (4.0 / 6.0)

# EXHIBITS 20-9 y 20-10: Equivalencia de Camiones (E_T)
tabla_ET_velocidad_ATS = {
    "Plano": [1.7, 1.2, 1.1],
    "Ondulado": [2.5, 1.9, 1.5],
    "Montañoso": [4.5, 2.7, 2.0] # Valores por defecto mantenidos para montañoso
}

tabla_ET_seguimiento_PTSF = {
    "Plano": [1.1, 1.1, 1.0],
    "Ondulado": [1.8, 1.5, 1.0],
    "Montañoso": [3.9, 2.6, 2.0] # Valores por defecto mantenidos para montañoso
}

def get_factors(terrain, flow_rate, is_ptsf=False):
    """Obtener f_G, E_T, y E_R según el tipo de terreno y la medida (ATS/PTSF)"""
    if flow_rate <= 600: cat = 0
    elif flow_rate <= 1200: cat = 1
    else: cat = 2

    if terrain == "Plano":
        fg = 1.0
    elif terrain == "Ondulado":
        if not is_ptsf: fg = [0.71, 0.93, 0.99][cat]
        else: fg = [0.77, 0.94, 1.00][cat]
    elif terrain == "Montañoso":
        if not is_ptsf: fg = [0.57, 0.85, 0.95][cat]
        else: fg = [0.60, 0.89, 1.00][cat]
    else:
        fg = 1.0

    if not is_ptsf:
        et = tabla_ET_velocidad_ATS.get(terrain, tabla_ET_velocidad_ATS["Plano"])[cat]
        er = 1.1 if terrain in ["Ondulado", "Montañoso"] else 1.0
    else:
        et = tabla_ET_seguimiento_PTSF.get(terrain, tabla_ET_seguimiento_PTSF["Plano"])[cat]
        er = 1.0
            
    return fg, et, er

def calculate_f_HV(p_t, E_T):
    """Calcula el factor de ajuste por vehículos pesados (solo camiones/buses)"""
    p_t = p_t / 100.0
    fhv = 1.0 / (1.0 + p_t * (E_T - 1.0))
    return fhv

# ==============================================================================
# EXHIBIT 20-11: f_np (Efecto de Zonas de No Rebase en ATS)
# ==============================================================================
# Columnas: % Zonas de No Rebase
cols_np_20_11 = [0, 20, 40, 60, 80, 100]
# Filas: Flujo de Demanda vp
filas_vp_20_11 = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200]

matriz_f_np = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # vp = 0
    [0.0, 1.0, 2.3, 3.8, 4.2, 5.6],  # vp = 200
    [0.0, 2.7, 4.3, 5.7, 6.3, 7.3],  # vp = 400
    [0.0, 2.5, 3.8, 4.9, 5.5, 6.2],  # vp = 600
    [0.0, 2.2, 3.1, 3.9, 4.3, 4.9],  # vp = 800
    [0.0, 1.8, 2.5, 3.2, 3.6, 4.2],  # vp = 1000
    [0.0, 1.3, 2.0, 2.6, 3.0, 3.4],  # vp = 1200
    [0.0, 0.9, 1.4, 1.9, 2.3, 2.7],  # vp = 1400
    [0.0, 0.9, 1.3, 1.7, 2.1, 2.4],  # vp = 1600
    [0.0, 0.8, 1.1, 1.6, 1.8, 2.1],  # vp = 1800
    [0.0, 0.8, 1.0, 1.4, 1.6, 1.8],  # vp = 2000
    [0.0, 0.8, 1.0, 1.4, 1.5, 1.7],  # vp = 2200
    [0.0, 0.8, 1.0, 1.3, 1.5, 1.7],  # vp = 2400
    [0.0, 0.8, 1.0, 1.3, 1.4, 1.6],  # vp = 2600
    [0.0, 0.8, 1.0, 1.2, 1.3, 1.4],  # vp = 2800
    [0.0, 0.8, 0.9, 1.1, 1.1, 1.3],  # vp = 3000
    [0.0, 0.8, 0.9, 1.0, 1.0, 1.1]   # vp = 3200
])

f_np_interp = RegularGridInterpolator((filas_vp_20_11, cols_np_20_11), matriz_f_np, bounds_error=False, fill_value=None)

def get_f_np(ffs, vp, no_passing_zones):
    """Ajuste f_np (Exhibit 20-11) mediante matriz interpolada 2D scipy"""
    vp_val = max(min(vp, 3200.0), 0.0)
    npz_val = max(min(no_passing_zones, 100.0), 0.0)
    return float(f_np_interp((vp_val, npz_val)))

# ==============================================================================
# EXHIBIT 20-12: f_d/np (Efecto combinado de Distribución Direccional y No Rebase)
# ==============================================================================

# Las columnas siempre representan el % de Zonas de No Rebase
zonas_no_rebase_cols = [0, 20, 40, 60, 80, 100]

# Estructura del diccionario:
# 'Split': {'vp_filas': [valores de flujo], 'matriz': [[fila 1], [fila 2], ...]}
tabla_fdnp = {
    "50/50": {
        "vp_filas": [200, 400, 600, 800, 1400, 2000, 2600, 3200],
        "matriz": [
            [0.0, 10.1, 17.2, 20.2, 21.0, 21.8],  # vp <= 200
            [0.0, 12.4, 19.0, 22.7, 23.8, 24.8],  # vp = 400
            [0.0, 11.2, 16.0, 18.7, 19.7, 20.5],  # vp = 600
            [0.0,  9.0, 12.3, 14.1, 14.5, 15.4],  # vp = 800
            [0.0,  3.6,  5.5,  6.7,  7.3,  7.9],  # vp = 1400
            [0.0,  1.8,  2.9,  3.7,  4.1,  4.4],  # vp = 2000
            [0.0,  1.1,  1.6,  2.0,  2.3,  2.4],  # vp = 2600
            [0.0,  0.7,  0.9,  1.1,  1.2,  1.4]   # vp >= 3200
        ]
    },
    "60/40": {
        "vp_filas": [200, 400, 600, 800, 1400, 2000, 2600],
        "matriz": [
            [1.6, 11.8, 17.2, 22.5, 23.1, 23.7],  # vp <= 200
            [0.5, 11.7, 16.2, 20.7, 21.5, 22.2],  # vp = 400
            [0.0, 11.5, 15.2, 18.9, 19.8, 20.7],  # vp = 600
            [0.0,  7.6, 10.3, 13.0, 13.7, 14.4],  # vp = 800
            [0.0,  3.7,  5.4,  7.1,  7.6,  8.1],  # vp = 1400
            [0.0,  2.3,  3.4,  3.6,  4.0,  4.3],  # vp = 2000
            [0.0,  0.9,  1.4,  1.9,  2.1,  2.2]   # vp >= 2600
        ]
    },
    "70/30": {
        "vp_filas": [200, 400, 600, 800, 1400, 2000],
        "matriz": [
            [2.8, 13.4, 19.1, 24.8, 25.2, 25.5],  # vp <= 200
            [1.1, 12.5, 17.3, 22.0, 22.6, 23.2],  # vp = 400
            [0.0, 11.6, 15.4, 19.1, 20.0, 20.9],  # vp = 600
            [0.0,  7.7, 10.5, 13.3, 14.0, 14.6],  # vp = 800
            [0.0,  3.8,  5.6,  7.4,  7.9,  8.3],  # vp = 1400
            [0.0,  1.4,  4.9,  3.5,  3.9,  4.2]   # vp >= 2000
        ]
    },
    "80/20": {
        "vp_filas": [200, 400, 600, 800, 1400, 2000],
        "matriz": [
            [5.1, 17.5, 24.3, 31.0, 31.3, 31.6],  # vp <= 200
            [2.5, 15.8, 21.5, 27.1, 27.6, 28.0],  # vp = 400
            [0.0, 14.0, 18.6, 23.2, 23.9, 24.5],  # vp = 600
            [0.0,  9.3, 12.7, 16.0, 16.5, 17.0],  # vp = 800
            [0.0,  4.6,  6.7,  8.7,  9.1,  9.5],  # vp = 1400
            [0.0,  2.4,  3.4,  4.5,  4.7,  4.9]   # vp >= 2000
        ]
    },
    "90/10": {
        "vp_filas": [200, 400, 600, 800, 1400],
        "matriz": [
            [5.6, 21.6, 29.4, 37.2, 37.4, 37.6],  # vp <= 200
            [2.4, 19.0, 25.6, 32.2, 32.5, 32.8],  # vp = 400
            [0.0, 16.3, 21.8, 27.2, 27.6, 28.0],  # vp = 600
            [0.0, 10.9, 14.8, 18.6, 19.0, 19.4],  # vp = 800
            [0.0,  5.5,  7.8, 10.0, 10.4, 10.7]   # vp >= 1400
        ]
    }
}

interpoladores_dnp = {}
for split, data in tabla_fdnp.items():
    vp_filas = data["vp_filas"]
    matriz = np.array(data["matriz"])
    interpoladores_dnp[split] = RegularGridInterpolator((vp_filas, zonas_no_rebase_cols), matriz, bounds_error=False, fill_value=None)

def calcular_f_dnp(split, vp, no_passing_percent):
    """Ajuste f_d/np (Exhibit 20-12) seleccionando la matriz exacta por distribución direccional"""
    if split not in interpoladores_dnp:
        split = '50/50'
        
    vp_filas = tabla_fdnp[split]["vp_filas"]
    vp_val = max(min(vp, float(max(vp_filas))), float(min(vp_filas)))
    npz_val = max(min(no_passing_percent, 100.0), 0.0)
    
    return float(interpoladores_dnp[split]((vp_val, npz_val)))

def determine_los(ats, ptsf, pffs, road_class):
    """Asigna el Nivel de Servicio (LOS) basado en ATS, PTSF y Clase"""
    if road_class == "Clase I (Movilidad)":
        if ptsf <= 35 and ats > 90: return "A"
        elif ptsf <= 50 and ats > 80: return "B"
        elif ptsf <= 65 and ats > 70: return "C"
        elif ptsf <= 80 and ats > 60: return "D"
        elif ptsf > 80 or ats <= 60: return "E"
        
    elif road_class == "Clase II (Proximidad/Turismo)":
        if ptsf <= 40: return "A"
        elif ptsf <= 55: return "B"
        elif ptsf <= 70: return "C"
        elif ptsf <= 85: return "D"
        else: return "E"
        
    elif road_class == "Clase III (Zonas desarrolladas)":
        if pffs >= 91.7: return "A"
        elif pffs >= 83.3: return "B"
        elif pffs >= 75.0: return "C"
        elif pffs >= 66.7: return "D"
        else: return "E"
        
    return "F"

# ==========================================
# Interfaz de Usuario (Streamlit)
# ==========================================

def main():
    st.markdown("""
        <style>
        .los-box {
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            margin: 1.5rem 0;
            color: white;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        .los-A { background: linear-gradient(135deg, #10B981, #059669); }
        .los-B { background: linear-gradient(135deg, #3B82F6, #2563EB); }
        .los-C { background: linear-gradient(135deg, #F59E0B, #D97706); }
        .los-D { background: linear-gradient(135deg, #F97316, #EA580C); }
        .los-E { background: linear-gradient(135deg, #EF4444, #DC2626); }
        .los-F { background: linear-gradient(135deg, #8B5CF6, #6D28D9); }
        .title-gradient {
            background: -webkit-linear-gradient(45deg, #3B82F6, #10B981);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5rem;
            font-weight: 800;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="title-gradient">🛣️ HCM 2000 Pro</div>', unsafe_allow_html=True)
    st.markdown("*Análisis avanzado de Capacidad y Nivel de Servicio para Carreteras de 2 Carriles*")
    st.divider()

    with st.sidebar:
        st.header("⚙️ Panel de Configuración")
        
        st.subheader("1. Clasificación")
        road_class = st.selectbox("Clase de Vía", ["Clase I (Movilidad)", "Clase II (Proximidad/Turismo)", "Clase III (Zonas desarrolladas)"])

        st.subheader("2. Geometría")
        bffs = st.number_input("Vel. Libre Base (BFFS) [km/h]", value=90.0)
        lane_width = st.number_input("Ancho Carril [m]", value=3.6)
        shoulder_width = st.number_input("Ancho Berma [m]", value=1.8)
        access_density = st.number_input("Densidad Accesos [ptos/km]", value=0, step=1)
        no_passing_zones = st.slider("Zonas No Rebase [%]", 0, 100, 20)

        st.subheader("3. Tráfico")
        tpda = st.number_input("TPDA [veh/día]", value=6000, step=100)
        k30 = st.number_input("Factor K30 (fracción)", min_value=0.0, max_value=1.0, value=0.10)
        phf = st.number_input("Factor Pico (PHF)", min_value=0.25, max_value=1.00, value=0.88)
        p_trucks = st.number_input("% Pesados", min_value=0.0, max_value=100.0, value=10.0)
        grade = st.selectbox("Terreno", ["Plano", "Ondulado", "Montañoso"])
        directional_dist = st.selectbox("Dist. Direccional", ["50/50", "60/40", "70/30", "80/20", "90/10"])
        
        st.write("") # Espacio en blanco
        btn_calc = st.button("🚀 Ejecutar Simulación", use_container_width=True, type="primary")

    if btn_calc:
        # Validación explícita de entradas físicas
        errores = []
        if bffs <= 0.0 or bffs > 160.0:
            errores.append("La Velocidad Libre Base (BFFS) debe ser mayor a 0 y máximo 160 km/h.")
        if lane_width < 2.0 or lane_width > 5.0:
            errores.append("El ancho de carril debe ser razonable (entre 2.0 y 5.0 m).")
        if shoulder_width < 0.0 or shoulder_width > 4.0:
            errores.append("El ancho de berma no puede ser negativo ni mayor a 4.0 m.")
        if tpda <= 0 or tpda > 100000:
            errores.append("El TPDA debe ser un valor lógico (entre 1 y 100,000 veh/día).")
            
        if errores:
            for error in errores:
                st.error(f"❌ **Entrada incorrecta:** {error}")
            st.warning("⚠️ Por favor, corrige los datos resaltados en el panel lateral e intenta de nuevo.")
        else:
            with st.spinner("Analizando variables con metodología HCM 2000..."):
                import time
                time.sleep(0.4) # Micro-pausa UX
                
                # Cálculos...
                f_LS = get_f_LS(lane_width, shoulder_width)
                f_A = get_f_A(access_density)
                ffs = bffs - f_LS - f_A

                vhd = tpda * k30
                vp_base = vhd / phf if phf > 0 else 0

                f_g_ats, E_T_ats, _ = get_factors(grade, vp_base, is_ptsf=False)
                f_HV_ats = calculate_f_HV(p_trucks, E_T_ats)
                vp_ats = vhd / (phf * f_g_ats * f_HV_ats) if (phf * f_g_ats * f_HV_ats) > 0 else 0
            
                f_np = get_f_np(ffs, vp_ats, no_passing_zones)
                ats = ffs - 0.0125 * vp_ats - f_np

                f_g_ptsf, E_T_ptsf, _ = get_factors(grade, vp_base, is_ptsf=True)
                f_HV_ptsf = calculate_f_HV(p_trucks, E_T_ptsf)
                vp_ptsf = vhd / (phf * f_g_ptsf * f_HV_ptsf) if (phf * f_g_ptsf * f_HV_ptsf) > 0 else 0
            
                bptsf = 100 * (1 - np.exp(-0.000879 * vp_ptsf))
                f_dnp = calcular_f_dnp(directional_dist, vp_ptsf, no_passing_zones)
                ptsf = bptsf + f_dnp

                pffs = (ats / ffs) * 100 if ffs > 0 else 0
                los = determine_los(ats, ptsf, pffs, road_class)

                if vp_ats > 3200 or vp_ptsf > 3200:
                    los = "F"
                    st.error("⚠️ La demanda excede la capacidad geométrica de la vía.")

                # Dashboard Results
                st.success("✅ Simulación completada con éxito")
            
                # Big LOS Display
                los_letter = los[0] # Get just the letter for the box class
                box_class = f"los-{los_letter}" if los_letter in ["A", "B", "C", "D", "E", "F"] else "los-F"
            
                st.markdown(f"""
                    <div class="los-box {box_class}">
                        <h2 style="margin:0; font-size: 1.5rem; color: rgba(255,255,255,0.9);">Nivel de Servicio (LOS)</h2>
                        <h1 style="margin:0; font-size: 5rem; font-weight: 900; letter-spacing: 2px;">{los_letter}</h1>
                    </div>
                """, unsafe_allow_html=True)
            
                st.header("📊 Medidas de Desempeño")
            
                # Ajustado a un layout de 2x2 para darle más espacio a los textos y unidades
                m1, m2 = st.columns(2)
                m1.metric("Velocidad de Flujo Libre (FFS)", f"{ffs:.1f} km/h")
                m2.metric("Velocidad Media de Viaje (ATS)", f"{ats:.1f} km/h")
            
                m3, m4 = st.columns(2)
                m3.metric("Tiempo en Seguimiento (PTSF)", f"{ptsf:.1f} %")
                if road_class == "Clase III (Zonas desarrolladas)":
                    m4.metric("Porcentaje de FFS (PFFS)", f"{pffs:.1f} %")
                else:
                    m4.metric("Flujo de Demanda Equivalente ($v_p$)", f"{vp_ats:.0f} veh/h")
                
                st.divider()

                with st.expander("📄 Ver Reporte Analítico y Factores de Ajuste"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Ajustes de Velocidad Libre (FFS)**")
                        st.write(f"- Factor ancho ($f_{{LS}}$): {f_LS:.2f} km/h")
                        st.write(f"- Factor accesos ($f_A$): {f_A:.2f} km/h")
                        st.markdown("**Parámetros Volumétricos**")
                        st.write(f"- Volumen Horario Diseño (VHD): {vhd:.1f} veh/h")
                    with c2:
                        st.markdown("**Ajustes de Capacidad (ATS / PTSF)**")
                        st.write(f"- Equivalentes de pesados ($E_T$): {E_T_ats:.1f} / {E_T_ptsf:.1f}")
                        st.write(f"- Ajuste por pesados ($f_{{HV}}$): {f_HV_ats:.3f} / {f_HV_ptsf:.3f}")
                        st.write(f"- Factor zonas no rebase ($f_{{np}}$): {f_np:.2f} km/h")
                        st.write(f"- Factor no rebase/dir ($f_{{d/np}}$): {f_dnp:.2f} %")

    else:
        # PANTALLA DE INICIO (Cuando aún no se ha hecho clic en calcular)
        st.info("👈 Configura los parámetros en el panel lateral y haz clic en 'Ejecutar Simulación' para ver los resultados.")
        
        st.markdown("""
        <div style="text-align: center; opacity: 0.5; margin-top: 50px;">
            <h1 style="font-size: 80px; margin-bottom: 0;">🛣️</h1>
            <p style="font-size: 1.2rem; font-weight: 500;">El motor algorítmico está listo.</p>
        </div>
        """, unsafe_allow_html=True)
if __name__ == "__main__":
    main()
