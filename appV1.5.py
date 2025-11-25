import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import time
from scipy.integrate import quad
from scipy import special, signal
from scipy.special import eval_legendre
import pandas as pd
import io
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor

# ==========================================
# 1. 全域頁面設定與初始化
# ==========================================
st.set_page_config(page_title="電磁學生成小教室", layout="wide", page_icon="⚡")

# CSS 美化
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1E88E5; text-align: center; margin-bottom: 1rem;}
    .stSlider {padding-top: 20px;}
    div.stButton > button:first-child {border-radius: 8px;}
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    .main .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# 初始化 Session State
default_states = {
    'fourier_result': None,
    'point_charges': [{'q': 1.0, 'x': -2.0, 'y': 0.0}, {'q': -1.0, 'x': 2.0, 'y': 0.0}],
    'point_charges_3d': [
        {'x': 1.0, 'y': 0.0, 'z': 0.0, 'q': 1.0}, 
        {'x': -1.0, 'y': 0.0, 'z': 0.0, 'q': -1.0}
    ],
    'point_charges_spherical': [
        {'r': 1.5, 'theta': 0.0, 'phi': 0.0, 'q': 1.0},
        {'r': 1.5, 'theta': 180.0, 'phi': 0.0, 'q': -1.0}
    ],
    'legendre_coeffs': None,
    'legendre_func': None,
    # 模擬結果暫存
    'res_2d_point': None,
    'res_2d_cart_num': None,
    'res_2d_cart_ana': None,
    'res_2d_cart_ana_text': None,
    'res_2d_sphere': None,
    'res_3d_cart': None,
    'res_3d_point': None,
    'res_3d_continuous': None
}

for key, val in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ==========================================
# 2. 核心運算函數 (通用與 2D)
# ==========================================

def get_safe_math_scope(x_val=None):
    scope = {
        "np": np, "signal": signal, "special": special,
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "exp": np.exp, "pi": np.pi, "abs": np.abs, 
        "sqrt": np.sqrt, "log": np.log, "sign": np.sign,
        "maximum": np.maximum, "minimum": np.minimum,
        "square": signal.square, "sawtooth": signal.sawtooth,
        "gamma": special.gamma, "sinh": np.sinh, "cosh": np.cosh,
        "where": np.where, "heaviside": np.heaviside,
        "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
        "legendre": eval_legendre
    }
    if x_val is not None: scope["x"] = x_val
    return scope

def eval_func(func_str, x_val):
    scope = get_safe_math_scope(x_val)
    try:
        return eval(func_str, {"__builtins__": None}, scope)
    except Exception:
        if hasattr(x_val, "__len__"):
            return np.array([eval(func_str, {"__builtins__": None}, get_safe_math_scope(xi)) for xi in x_val])
        return np.nan

def smart_parse(input_str):
    if not input_str or input_str.strip() == "0": return None
    transformations = (standard_transformations + (implicit_multiplication_application,) + (convert_xor,))
    try:
        return parse_expr(input_str, transformations=transformations, local_dict={'e': sp.E, 'pi': sp.pi})
    except:
        return None

def spherical_to_cartesian(r, theta_deg, phi_deg):
    """球座標轉直角座標 (輸入為角度)"""
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

# --- 2D 快取運算 ---
@st.cache_data(show_spinner=False)
def calculate_fourier_coefficients(func_str, a, b, max_n):
    L = b - a
    if L <= 0: return None, "區間錯誤：b 必須大於 a"
    omega = 2 * np.pi / L
    A_coeffs, B_coeffs = [], []
    try:
        val_a0, _ = quad(lambda x: eval_func(func_str, x), a, b, limit=200)
        A_coeffs.append((2.0 / L) * val_a0)
        B_coeffs.append(0.0)
        for n in range(1, max_n + 1):
            val_an, _ = quad(lambda x: eval_func(func_str, x) * np.cos(n * omega * x), a, b, limit=100)
            val_bn, _ = quad(lambda x: eval_func(func_str, x) * np.sin(n * omega * x), a, b, limit=100)
            A_coeffs.append((2.0 / L) * val_an)
            B_coeffs.append((2.0 / L) * val_bn)
        x_vals = np.linspace(a, b, 1000)
        y_original = eval_func(func_str, x_vals)
        return {"A": A_coeffs, "B": B_coeffs, "omega": omega, "x_vals": x_vals, "y_original": y_original, "range": (a, b)}, None
    except Exception as e:
        return None, f"運算錯誤: {str(e)}"

@st.cache_data(show_spinner=False)
def calculate_legendre_coefficients(func_expression, max_n):
    try: _ = eval_func(func_expression, 0.5)
    except Exception as e: return None, None, f"語法解析錯誤: {str(e)}"
    coeffs = []
    try:
        for n in range(max_n + 1):
            factor = (2 * n + 1) / 2
            integrand = lambda x: eval_func(func_expression, x) * eval_legendre(n, x)
            val, _ = quad(integrand, -1, 1, limit=100)
            coeffs.append(factor * val)
        return coeffs, None, None
    except Exception as e: return None, None, f"積分錯誤: {str(e)}"

@st.cache_data(show_spinner=False)
def calculate_point_charge_potential(charges_tuple, grid_size=100):
    charges = list(charges_tuple)
    x = np.linspace(-5, 5, grid_size)
    y = np.linspace(-5, 5, grid_size)
    X, Y = np.meshgrid(x, y)
    V_total = np.zeros_like(X)
    if not charges: return X, Y, V_total
    for charge in charges:
        q = charge['q']; x0 = charge['x']; y0 = charge['y']
        r = np.sqrt((X - x0)**2 + (Y - y0)**2)
        V_total += q / (r + 1e-9) 
    return X, Y, V_total

def plot_heatmap(data, title, xlabel="x", ylabel="y"):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data, cmap='jet', origin='lower', extent=[0, 1, 0, 1], aspect='auto', interpolation='bilinear')
    plt.colorbar(im, ax=ax).set_label('Potential (V)')
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig

# ==========================================
# 3. 3D 核心運算函數
# ==========================================

@st.cache_data(show_spinner=False)
def calculate_3d_physics(N, v_top, v_bottom, v_left, v_right, v_front, v_back, max_iter, tolerance):
    """求解 3D Laplace 方程式"""
    V = np.zeros((N, N, N))
    V[:, :, -1] = v_top;    V[:, :, 0]  = v_bottom
    V[:, -1, :] = v_back;   V[:, 0, :]  = v_front
    V[-1, :, :] = v_right;  V[0, :, :]  = v_left

    for i in range(max_iter):
        V_old = V.copy()
        V[1:-1, 1:-1, 1:-1] = (1/6) * (
            V[2:, 1:-1, 1:-1] + V[:-2, 1:-1, 1:-1] + 
            V[1:-1, 2:, 1:-1] + V[1:-1, :-2, 1:-1] + 
            V[1:-1, 1:-1, 2:] + V[1:-1, 1:-1, :-2]
        )
        V[:, :, -1] = v_top; V[:, :, 0] = v_bottom
        V[:, -1, :] = v_back; V[:, 0, :] = v_front
        V[-1, :, :] = v_right; V[0, :, :] = v_left
        if i % 200 == 0:
            if np.max(np.abs(V - V_old)) < tolerance: break
                
    h = 1.0 / (N - 1)
    grads = np.gradient(V, h)
    Ex, Ey, Ez = -grads[0], -grads[1], -grads[2]
    grid_range = np.linspace(0, 1, N)
    X, Y, Z = np.meshgrid(grid_range, grid_range, grid_range, indexing='ij')
    return X, Y, Z, V, Ex, Ey, Ez, i

@st.cache_data(show_spinner=False)
def calculate_point_charge_field_3d(charges_tuple, grid_range, grid_res):
    """3D 點電荷場計算"""
    charges = list(charges_tuple)
    x = np.linspace(-grid_range, grid_range, grid_res)
    y = np.linspace(-grid_range, grid_range, grid_res)
    z = np.linspace(-grid_range, grid_range, grid_res)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    Ex, Ey, Ez = np.zeros_like(X), np.zeros_like(Y), np.zeros_like(Z)
    V = np.zeros_like(X)
    k = 1.0 

    for charge in charges:
        qx, qy, qz, q_val = charge['x'], charge['y'], charge['z'], charge['q']
        rx = X - qx; ry = Y - qy; rz = Z - qz
        r = np.sqrt(rx**2 + ry**2 + rz**2)
        r[r < 0.2] = 0.2 # Singularity handling
        
        V += k * q_val / r
        E_mag = k * q_val / (r**3)
        Ex += E_mag * rx; Ey += E_mag * ry; Ez += E_mag * rz

    return X, Y, Z, V, Ex, Ey, Ez

@st.cache_data(show_spinner=False)
def calculate_continuous_spherical(dist_type, R, grid_range, grid_res):
    """計算連續球體電荷分佈產生的場 (Discretization Method)"""
    x = np.linspace(-grid_range, grid_range, grid_res)
    y = np.linspace(-grid_range, grid_range, grid_res)
    z = np.linspace(-grid_range, grid_range, grid_res)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    V = np.zeros_like(X)
    Ex = np.zeros_like(X)
    Ey = np.zeros_like(X)
    Ez = np.zeros_like(X)
    
    # 固定來源解析度
    num_r, num_theta, num_phi = 8, 12, 12
    dr = R / num_r
    dtheta = np.pi / num_theta
    dphi = 2 * np.pi / num_phi
    
    r_range = np.linspace(dr/2, R - dr/2, num_r)
    theta_range = np.linspace(dtheta/2, np.pi - dtheta/2, num_theta)
    phi_range = np.linspace(dphi/2, 2*np.pi - dphi/2, num_phi)
    
    source_charges = []
    for r_s in r_range:
        for theta_s in theta_range:
            for phi_s in phi_range:
                if dist_type == "Uniform (均勻)": rho = 1.0
                elif dist_type == "Decaying (1/r)": rho = 1.0 / r_s
                elif dist_type == "Orbital (p-like)": rho = np.abs(np.cos(theta_s)) * 2.0
                else: rho = 1.0
                
                dV = (r_s**2) * np.sin(theta_s) * dr * dtheta * dphi
                dq = rho * dV
                
                # 弧度轉直角 (手動計算避免依賴外部函數)
                cx = r_s * np.sin(theta_s) * np.cos(phi_s)
                cy = r_s * np.sin(theta_s) * np.sin(phi_s)
                cz = r_s * np.cos(theta_s)
                source_charges.append((cx, cy, cz, dq))
    
    k_e = 1.0
    for cx, cy, cz, dq in source_charges:
        dx = X - cx; dy = Y - cy; dz = Z - cz
        dist_sq = dx**2 + dy**2 + dz**2
        dist = np.sqrt(dist_sq)
        dist = np.where(dist < 0.15, 0.15, dist)
        
        V += k_e * dq / dist
        E_common = k_e * dq / (dist**3)
        Ex += E_common * dx; Ey += E_common * dy; Ez += E_common * dz
        
    return X, Y, Z, V, Ex, Ey, Ez, len(source_charges)

# --- 3D 視覺化 (改良版：Log-Binning) ---

def create_potential_figure(X, Y, Z, V, opacity, surface_count, show_caps):
    """繪製 3D 電位等位面"""
    fig = go.Figure(data=go.Isosurface(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=V.flatten(),
        isomin=np.min(V), isomax=np.max(V),
        surface_count=surface_count,
        opacity=opacity,
        caps=dict(x_show=show_caps, y_show=show_caps, z_show=show_caps),
        colorscale='Jet',
        colorbar=dict(title='電位 V'),
        hoverinfo='all'
    ))
    fig.update_layout(
        title="3D 電位分佈 (Isosurfaces)",
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='cube'),
        margin=dict(l=0, r=0, b=0, t=40), height=700
    )
    return fig

def create_field_figure_log(X, Y, Z, Ex, Ey, Ez, scale, stride):
    """繪製 3D 電場 (固定大小 + Log-Scale 彩虹顏色)"""
    X_sub = X[::stride, ::stride, ::stride].flatten()
    Y_sub = Y[::stride, ::stride, ::stride].flatten()
    Z_sub = Z[::stride, ::stride, ::stride].flatten()
    Ex_sub = Ex[::stride, ::stride, ::stride].flatten()
    Ey_sub = Ey[::stride, ::stride, ::stride].flatten()
    Ez_sub = Ez[::stride, ::stride, ::stride].flatten()
    
    E_mag = np.sqrt(Ex_sub**2 + Ey_sub**2 + Ez_sub**2)
    E_mag_safe = np.where(E_mag == 0, 1e-9, E_mag)
    
    U_norm = np.nan_to_num(Ex_sub / E_mag_safe)
    V_norm = np.nan_to_num(Ey_sub / E_mag_safe)
    W_norm = np.nan_to_num(Ez_sub / E_mag_safe)

    log_mag = np.log10(E_mag_safe)
    vmin = np.percentile(log_mag, 5)
    vmax = np.percentile(log_mag, 95)
    
    n_bins = 20
    bins = np.linspace(vmin, vmax, n_bins)
    indices = np.digitize(log_mag, bins) - 1
    indices = np.clip(indices, 0, n_bins - 1)
    
    cmap = plt.get_cmap('jet')
    fig = go.Figure()
    
    for i in range(n_bins):
        mask = (indices == i)
        if not np.any(mask): continue
        color_val = i / (n_bins - 1)
        hex_color = mcolors.to_hex(cmap(color_val))
        
        fig.add_trace(go.Cone(
            x=X_sub[mask], y=Y_sub[mask], z=Z_sub[mask],
            u=U_norm[mask], v=V_norm[mask], w=W_norm[mask],
            colorscale=[[0, hex_color], [1, hex_color]],
            showscale=False,
            sizemode="scaled",
            sizeref=scale,
            anchor="tail",
            hoverinfo='text',
            text=f"Log(|E|) ~ {bins[i]:.2f}",
            name=f"Level {i}"
        ))

    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers',
        marker=dict(
            colorscale='Jet',
            cmin=vmin, cmax=vmax,
            showscale=True,
            colorbar=dict(title='Log10(|E|)', x=0.9)
        )
    ))

    fig.update_layout(
        title="3D 電場向量分佈 (Log-Scale Intensity)",
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='cube'),
        margin=dict(l=0, r=0, b=0, t=40), height=700,
        showlegend=False
    )
    return fig

def create_field_figure_fixed(X, Y, Z, Ex, Ey, Ez, scale, stride):
    """繪製 3D 電場 (固定大小 + 彩虹顏色 - Linear)"""
    X_sub = X[::stride, ::stride, ::stride].flatten()
    Y_sub = Y[::stride, ::stride, ::stride].flatten()
    Z_sub = Z[::stride, ::stride, ::stride].flatten()
    Ex_sub = Ex[::stride, ::stride, ::stride].flatten()
    Ey_sub = Ey[::stride, ::stride, ::stride].flatten()
    Ez_sub = Ez[::stride, ::stride, ::stride].flatten()
    
    E_mag = np.sqrt(Ex_sub**2 + Ey_sub**2 + Ez_sub**2)
    E_mag_safe = np.where(E_mag == 0, 1e-9, E_mag)
    U_norm = np.nan_to_num(Ex_sub / E_mag_safe)
    V_norm = np.nan_to_num(Ey_sub / E_mag_safe)
    W_norm = np.nan_to_num(Ez_sub / E_mag_safe)

    fig = go.Figure()
    
    n_bins = 20
    cmap = plt.get_cmap('jet')
    vmin, vmax = np.percentile(E_mag, 2), np.percentile(E_mag, 98)
    bins = np.linspace(vmin, vmax, n_bins)
    indices = np.digitize(E_mag, bins) - 1
    indices = np.clip(indices, 0, n_bins - 1)
    
    for i in range(n_bins):
        mask = (indices == i)
        if not np.any(mask): continue
        color_val = i / (n_bins - 1)
        hex_color = mcolors.to_hex(cmap(color_val))
        
        fig.add_trace(go.Cone(
            x=X_sub[mask], y=Y_sub[mask], z=Z_sub[mask],
            u=U_norm[mask], v=V_norm[mask], w=W_norm[mask],
            colorscale=[[0, hex_color], [1, hex_color]],
            showscale=False,
            sizemode="scaled",
            sizeref=scale,
            anchor="tail",
            hoverinfo='u+v+w+name',
            name=f"E ~ {bins[i]:.2e}"
        ))

    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers',
        marker=dict(
            colorscale='Jet',
            cmin=vmin, cmax=vmax,
            showscale=True,
            colorbar=dict(title='電場強度 |E|', x=0.9)
        )
    ))

    fig.update_layout(
        title="3D 電場向量分佈 (固定大小 + 彩虹強弱)",
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='cube'),
        margin=dict(l=0, r=0, b=0, t=40), height=700,
        showlegend=False
    )
    return fig

# ==========================================
# 4. 頁面渲染邏輯 (Rendering)
# ==========================================

def render_home():
    st.markdown("<h1 class='main-header'>⚡ 電磁學生成小教室 ⚡</h1>", unsafe_allow_html=True)
    st.markdown("""
    ### 歡迎來到互動學習實驗室！
    請從左側選單選擇您想要探索的主題。所有模擬均需點擊 **「🚀 開始模擬」** 按鈕才會執行運算。
    """)

def render_developing(title):
    st.subheader(f"🚧 {title}")
    st.info("此功能目前正在開發中，敬請期待！")

# --- 2D 函數與模擬 ---
def render_fourier_page():
    st.subheader("📈 傅立葉級數近似")
    fourier_examples = {
        "自訂輸入": "", "方波": "square(x)", "多週期方波": "square(3 * x)", "鋸齒波": "sawtooth(x)", 
        "三角波": "sawtooth(x, 0.5)", "全波整流": "abs(sin(x))", "半波整流": "maximum(sin(x), 0)", "脈衝波": "square(x, duty=0.2)"
    }
    def update_fourier():
        if st.session_state.fourier_preset != "自訂輸入":
            st.session_state.fourier_input = fourier_examples[st.session_state.fourier_preset]
            
    st.sidebar.markdown("---")
    st.sidebar.selectbox("選擇預設波形", list(fourier_examples.keys()), key='fourier_preset', on_change=update_fourier)
    c1, c2, c3, c4 = st.columns(4)
    with c1: func_str = st.text_input("函數 f(x)", value="square(x)", key="fourier_input") 
    with c2: a = st.number_input("起點 a", -3.1415)
    with c3: b = st.number_input("終點 b", 3.1415)
    with c4: max_n = st.number_input("最大項數", 50, step=10)

    if st.button("🚀 計算", use_container_width=True):
        with st.spinner("運算中..."):
            result, error = calculate_fourier_coefficients(func_str, a, b, int(max_n))
            if error: st.error(error)
            else: st.session_state['fourier_result'] = result

    if st.session_state['fourier_result']:
        res = st.session_state['fourier_result']
        st.divider()
        current_n = st.slider("調整 N 值 (疊加項數)", 0, len(res["A"])-1, min(10, len(res["A"])-1))
        n_indices = np.arange(1, current_n + 1)
        A_terms = np.array(res["A"][1:current_n+1]).reshape(-1, 1)
        B_terms = np.array(res["B"][1:current_n+1]).reshape(-1, 1)
        k_omega_x = res["omega"] * np.outer(n_indices, res["x_vals"])
        y_approx = res["A"][0]/2 + np.sum(A_terms * np.cos(k_omega_x) + B_terms * np.sin(k_omega_x), axis=0)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        if res["y_original"] is not None: ax.plot(res["x_vals"], res["y_original"], 'k-', alpha=0.3, label='Original')
        ax.plot(res["x_vals"], y_approx, 'b-', linewidth=2, label=f'N={current_n}')
        ax.legend(); ax.grid(True, alpha=0.3)
        st.pyplot(fig); plt.close(fig)
        
        df = pd.DataFrame({"n": range(len(res["A"])), "An": res["A"], "Bn": res["B"]})
        c1, c2 = st.columns(2)
        buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=150); buf.seek(0)
        c1.download_button("📥 下載圖表", buf, "fourier.png", "image/png", use_container_width=True)
        c2.download_button("📥 下載係數", df.to_csv(index=False, sep='\t').encode(), "coeffs.csv", "text/csv", use_container_width=True)
        with st.expander("查看係數表"): st.dataframe(df, use_container_width=True)

def render_legendre_page():
    st.subheader("🌊 勒讓德級數近似")
    legendre_examples = {
        "自訂輸入": "", "方波": "where(x > 0, 1, 0)", "三角波": "where(x > 0, x, 0)", "絕對值": "abs(x)",
        "多週期方波": "sign(sin(4 * pi * x))", "波包": "sin(15 * x) * exp(-5 * x**2)", "全波整流": "abs(sin(3 * pi * x))",
        "AM 調變": "(1 + 0.5 * cos(10 * x)) * cos(50 * x)", "偶極子": "x", "四極子": "3*x**2 - 1"
    }
    def update_legendre():
        if st.session_state.legendre_preset != "自訂輸入":
            st.session_state.legendre_input = legendre_examples[st.session_state.legendre_preset]
    st.sidebar.markdown("---")
    st.sidebar.selectbox("選擇波形", list(legendre_examples.keys()), key='legendre_preset', on_change=update_legendre)
    c1, c2 = st.columns([3, 1])
    with c1: func_str = st.text_input("輸入 f(x)", value="where(x > 0, 1, 0)", key="legendre_input")
    with c2: max_N = st.number_input("最大階數", 20)

    if st.button("🚀 運算", use_container_width=True):
        with st.spinner("積分中..."):
            coeffs, _, error = calculate_legendre_coefficients(func_str, int(max_N))
            if error: st.error(error)
            else: st.session_state['legendre_coeffs'] = coeffs; st.session_state['legendre_func'] = func_str

    if 'legendre_coeffs' in st.session_state and st.session_state['legendre_coeffs']:
        coeffs = st.session_state['legendre_coeffs']
        func_expr = st.session_state.get('legendre_func', func_str)
        st.divider()
        current_n = st.slider("疊加階數", 0, len(coeffs)-1, len(coeffs)-1)
        x = np.linspace(-1, 1, 400)
        y_target = eval_func(func_expr, x)
        y_approx = sum(coeffs[n] * eval_legendre(n, x) for n in range(current_n + 1))
        
        fig = plt.figure(figsize=(12, 5))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(x, y_target, 'k--', alpha=0.5, label="Target"); ax1.plot(x, y_approx, 'r-', linewidth=2, label="Approx")
        ax1.set_title("Cartesian (x vs f(x))"); ax1.set_xlabel("x = cos(theta)"); ax1.legend(); ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(1, 2, 2, projection='polar')
        theta = np.linspace(0, 2*np.pi, 400)
        r_target_polar = eval_func(func_expr, np.cos(theta))
        r_approx = sum(coeffs[n] * eval_legendre(n, np.cos(theta)) for n in range(current_n + 1))
        ax2.plot(theta, np.abs(r_target_polar), 'k--', alpha=0.5, label='Target')
        ax2.plot(theta, np.abs(r_approx), 'r-', linewidth=2, label='Approx')
        ax2.set_title("Polar (Abs magnitude)"); ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        st.pyplot(fig); plt.close(fig)
        
        df = pd.DataFrame({"n": range(len(coeffs)), "cn": coeffs})
        c1, c2 = st.columns(2)
        buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=150); buf.seek(0)
        c1.download_button("📥 下載圖表", buf, "legendre.png", "image/png", use_container_width=True)
        c2.download_button("📥 下載係數", df.to_csv(index=False, sep='\t').encode(), "coeffs.csv", "text/csv", use_container_width=True)
        with st.expander("查看係數表"): st.dataframe(df, use_container_width=True)

def render_potential_point_charge():
    st.subheader("⚡ 點電荷電位與電場模擬 (2D)")
    st.sidebar.markdown("---"); st.sidebar.header("🔋 電荷控制")
    c1, c2 = st.sidebar.columns(2)
    new_q = c1.number_input("電荷量 (q)", value=1.0, step=0.5)
    c3, c4 = st.sidebar.columns(2)
    new_x = c3.number_input("X 座標", value=0.0, step=0.5, min_value=-5.0, max_value=5.0)
    new_y = c4.number_input("Y 座標", value=0.0, step=0.5, min_value=-5.0, max_value=5.0)
    if st.sidebar.button("➕ 加入電荷", use_container_width=True): st.session_state.point_charges.append({'q': new_q, 'x': new_x, 'y': new_y})
    if st.sidebar.button("🗑️ 清除所有", use_container_width=True): st.session_state.point_charges = []
    st.sidebar.divider()
    if st.session_state.point_charges:
        for i, c in enumerate(st.session_state.point_charges): st.sidebar.text(f"{i+1}. q={c['q']}, ({c['x']}, {c['y']})")
    
    # 視覺化參數
    show_streamlines = st.sidebar.checkbox("顯示流線", value=True)
    grid_res = st.sidebar.slider("網格解析度", 50, 200, 100)

    # 計算按鈕
    if st.button("🚀 開始模擬", use_container_width=True, key="btn_2d_point"):
        if st.session_state.point_charges:
            charges_tuple = tuple(st.session_state.point_charges)
            st.session_state['res_2d_point'] = calculate_point_charge_potential(charges_tuple, grid_res)
        else:
            st.warning("請在左側加入電荷")

    # 繪圖 (使用暫存結果)
    if st.session_state['res_2d_point']:
        X, Y, V = st.session_state['res_2d_point']
        fig, ax = plt.subplots(figsize=(10, 8))
        contour = ax.contourf(X, Y, V, levels=50, cmap='RdBu_r', extend='both')
        ax.contour(X, Y, V, levels=50, colors='k', linewidths=0.5, alpha=0.4)
        if show_streamlines:
            Ey, Ex = np.gradient(-V); mag = np.sqrt(Ex**2 + Ey**2); Ex = np.where(mag > 0, Ex, 0); Ey = np.where(mag > 0, Ey, 0)
            ax.streamplot(X, Y, Ex, Ey, color='#444444', density=1.2, linewidth=0.6, arrowsize=1)
        for charge in st.session_state.point_charges:
            col = '#d62728' if charge['q'] > 0 else '#1f77b4'; sign = '+' if charge['q'] > 0 else '-'
            ax.plot(charge['x'], charge['y'], marker='o', color=col, markersize=15, markeredgecolor='black')
            ax.text(charge['x'], charge['y'], sign, color='white', ha='center', va='center', fontweight='bold')
        ax.set_aspect('equal'); ax.set_title("Electric Potential Landscape"); fig.colorbar(contour, ax=ax)
        st.pyplot(fig); plt.close(fig)

def render_laplace_cartesian_2d():
    st.subheader("🔲 電位模擬 - 笛卡爾座標 (2D)")
    mode = st.radio("模式", ["數值解 (FDM)", "解析解"], horizontal=True)
    
    if mode == "數值解 (FDM)":
        c1, c2 = st.columns([1, 3])
        with c1:
            st.markdown("##### 邊界條件")
            def inp(l, d):
                inf = st.checkbox(f"{l} 接地/無窮", key=f"i_{l}")
                return (True, 0.0) if inf else (False, st.number_input(f"{l} V", float(d), key=f"v_{l}"))
            ti, tv = inp("上", 10.0); bi, bv = inp("下", 0.0); li, lv = inp("左", 0.0); ri, rv = inp("右", 0.0)
            iters = st.slider("迭代", 500, 5000, 2000)
        with c2:
            if st.button("🚀 開始模擬", use_container_width=True, key="btn_2d_fdm"):
                sz=40; pad=sz*3; th=(pad if ti else 0)+sz+(pad if bi else 0); tw=(pad if li else 0)+sz+(pad if ri else 0)
                V = np.zeros((th, tw)); rs=pad if bi else 0; re=rs+sz; cs=pad if li else 0; ce=cs+sz
                if not ti: V[re-1, cs:ce]=tv
                if not bi: V[rs, cs:ce]=bv
                if not li: V[rs:re, cs]=lv
                if not ri: V[rs:re, ce-1]=rv
                p = st.progress(0)
                for i in range(iters):
                    V_old=V.copy(); V[1:-1,1:-1]=0.25*(V_old[0:-2,1:-1]+V_old[2:,1:-1]+V_old[1:-1,0:-2]+V_old[1:-1,2:])
                    if not ti: V[re-1, cs:ce]=tv
                    if not bi: V[rs, cs:ce]=bv
                    if not li: V[rs:re, cs]=lv
                    if not ri: V[rs:re, ce-1]=rv
                    if i%(iters//10)==0: p.progress((i+1)/iters)
                p.progress(1.0)
                st.session_state['res_2d_cart_num'] = V[rs:re, cs:ce]
            
            if st.session_state['res_2d_cart_num'] is not None:
                st.pyplot(plot_heatmap(st.session_state['res_2d_cart_num'], "FDM Result"))
    else:
        st.info("輸入支援 Python 語法")
        c1, c2 = st.columns(2)
        ts = c1.text_input("V(x,1)", "10"); bs = c1.text_input("V(x,0)", "0"); ls = c2.text_input("V(0,y)", "0"); rs = c2.text_input("V(1,y)", "0")
        
        if st.button("🚀 開始模擬", use_container_width=True, key="btn_2d_ana"):
            x,y,n=sp.symbols('x y n'); pi=sp.pi; terms=[]
            def calc(s, sd):
                ex=smart_parse(s); 
                if not ex: return None
                integrand=ex.subs(x if sd in ['top','bottom'] else y, x)
                try: An=2*sp.integrate(integrand*sp.sin(n*pi*x),(x,0,1))
                except: return None
                den=sp.sinh(n*pi)
                if sd=='top': return An*sp.sin(n*pi*x)*sp.sinh(n*pi*y)/den
                if sd=='bottom': return An*sp.sin(n*pi*x)*sp.sinh(n*pi*(1-y))/den
                if sd=='left': return An*sp.sin(n*pi*y)*sp.sinh(n*pi*(1-x))/den
                if sd=='right': return An*sp.sin(n*pi*y)*sp.sinh(n*pi*x)/den
            for s,sd in [(ts,'top'),(bs,'bottom'),(ls,'left'),(rs,'right')]:
                r=calc(s,sd); 
                if r: terms.append(r)
            if terms:
                Vt=sum(terms)
                st.session_state['res_2d_cart_ana_text'] = sp.latex(Vt)
                X,Y=np.meshgrid(np.linspace(0,1,50),np.linspace(0,1,50)); Vn=np.zeros_like(X)
                try:
                    fn=sp.lambdify((n,x,y),Vt,'numpy'); p=st.progress(0)
                    for i in range(1,21): Vn+=np.nan_to_num(fn(i,X,Y)); p.progress(i/20)
                    st.session_state['res_2d_cart_ana'] = Vn
                except Exception as e: st.error(e)
        
        if st.session_state['res_2d_cart_ana'] is not None:
            if st.session_state.get('res_2d_cart_ana_text'):
                st.latex(f"V(x,y) = \\sum_{{n=1}}^{{\\infty}} \\left[ {st.session_state['res_2d_cart_ana_text']} \\right]")
            st.pyplot(plot_heatmap(st.session_state['res_2d_cart_ana'], "Analytical Solution"))

def render_potential_spherical_2d():
    st.subheader("🌐 2D 極座標/球座標切面電位分析")
    PRESETS = {
        "點電荷": "k/r", 
        "電偶極": "k*cos(theta)/r^2", 
        "電四極": "k*(3*cos(theta)**2 - 1)/r^3",
        "均勻電場": "-k*r*cos(theta)",
        "複雜組合": "k/r + r*cos(theta)"
    }
    sel = st.sidebar.selectbox("選擇模型", list(PRESETS.keys()), index=1, key="sp2d")
    user_input = st.sidebar.text_input("輸入 V(r, theta)", value=PRESETS[sel])
    rmax = st.sidebar.slider("範圍", 1.0, 10.0, 5.0)
    grid_res = st.sidebar.slider("網格解析度", 50, 200, 100)
    show_lines = st.sidebar.checkbox("顯示流線", True)

    if st.button("🚀 開始模擬", use_container_width=True, key="btn_2d_sph"):
        if user_input:
            try:
                r, theta, k = sp.symbols('r theta k', real=True)
                trans = (standard_transformations + (implicit_multiplication_application,) + (convert_xor,))
                V_expr = parse_expr(user_input, local_dict={'k':k, 'pi':sp.pi, 'r':r, 'theta':theta}, transformations=trans)
                E_r = -sp.diff(V_expr, r); E_theta = -(1/r)*sp.diff(V_expr, theta)
                
                func_V = sp.lambdify((r, theta), V_expr.subs(k, 1), 'numpy')
                func_Er = sp.lambdify((r, theta), E_r.subs(k, 1), 'numpy')
                func_Et = sp.lambdify((r, theta), E_theta.subs(k, 1), 'numpy')
                
                x_vals = np.linspace(-rmax, rmax, grid_res); X, Y = np.meshgrid(x_vals, x_vals)
                R = np.sqrt(X**2 + Y**2); THETA = np.arctan2(Y, X); R[R<1e-3]=1e-3
                Z_V = func_V(R, THETA)
                if np.isscalar(Z_V): Z_V = np.full_like(R, Z_V)
                
                # Store results including functions for streamlines recalculation if needed
                U_Er = func_Er(R, THETA); U_Et = func_Et(R, THETA)
                if np.isscalar(U_Er): U_Er = np.full_like(R, U_Er)
                if np.isscalar(U_Et): U_Et = np.full_like(R, U_Et)
                
                st.session_state['res_2d_sphere'] = {
                    'X': X, 'Y': Y, 'Z_V': Z_V, 'U_Er': U_Er, 'U_Et': U_Et, 'THETA': THETA,
                    'V_latex': sp.latex(V_expr), 'Er_latex': sp.latex(E_r), 'Et_latex': sp.latex(E_theta)
                }
            except Exception as e: st.error(f"Error: {e}")

    if st.session_state['res_2d_sphere']:
        res = st.session_state['res_2d_sphere']
        c1, c2 = st.columns(2)
        c1.latex(f"V={res['V_latex']}")
        c2.latex(f"\\vec{{E}}=({res['Er_latex']})\\hat{{r}} + ({res['Et_latex']})\\hat{{\\theta}}")
        
        fig, ax = plt.subplots(figsize=(8, 7))
        try:
            contour = ax.contourf(res['X'], res['Y'], res['Z_V'], levels=50, cmap='jet'); plt.colorbar(contour, ax=ax, label='V')
        except: st.warning("數值範圍過大，無法繪製")

        if show_lines:
            Ex = res['U_Er']*np.cos(res['THETA']) - res['U_Et']*np.sin(res['THETA'])
            Ey = res['U_Er']*np.sin(res['THETA']) + res['U_Et']*np.cos(res['THETA'])
            ax.streamplot(res['X'], res['Y'], np.nan_to_num(Ex), np.nan_to_num(Ey), color=(1,1,1,0.5), linewidth=0.8)
        
        ax.set_aspect('equal'); st.pyplot(fig); plt.close(fig)

# ==========================================
# 5. 3D 渲染邏輯
# ==========================================

def render_3d_cartesian():
    st.subheader("🧊 3D 靜電場視覺化：笛卡爾座標")
    st.markdown("使用有限差分法求解 $\\nabla^2 V = 0$。")

    # 1. 側邊欄參數
    with st.sidebar:
        st.markdown("---")
        viz_mode = st.radio(
            "選擇視覺化模式", ["電位分佈 (Potential)", "電場向量 (Electric Field)"], index=0
        )
        
        st.divider()
        st.header("⚙️ 3D 模擬參數")
        grid_n = st.slider("網格點數 (N)", 10, 60, 30)
        
        with st.expander("設定邊界電位 (Boundary)", expanded=True):
            c1, c2 = st.columns(2)
            v_top = c1.number_input("頂面 (Z=1)", value=100.0, step=10.0)
            v_bottom = c2.number_input("底面 (Z=0)", value=-100.0, step=10.0)
            v_back = c1.number_input("後面 (Y=1)", value=0.0, step=10.0)
            v_front = c2.number_input("前面 (Y=0)", value=0.0, step=10.0)
            v_right = c1.number_input("右面 (X=1)", value=0.0, step=10.0)
            v_left = c2.number_input("左面 (X=0)", value=0.0, step=10.0)

        max_iter = st.number_input("最大迭代", 3000, step=500)
        tolerance = st.select_slider("精度", options=[1e-2, 1e-3, 1e-4, 1e-5], value=1e-4)
        
        st.divider()
        st.header("🎨 繪圖微調")
        if viz_mode == "電位分佈 (Potential)":
            surface_count = st.slider("等位面層數", 3, 20, 10)
            opacity = st.slider("透明度", 0.1, 1.0, 0.3)
            show_caps = st.checkbox("顯示封蓋 (Caps)", False)
        else:
            st.info("箭頭顏色(Rainbow)代表強度，箭頭長度固定。")
            cone_scale = st.slider("箭頭固定大小", 0.05, 0.2, 0.1)
            stride_val = st.slider("採樣間隔 (Stride)", 1, 5, 2)

    # 2. 計算按鈕
    if st.sidebar.button("🚀 開始模擬", key="btn_3d_cart"):
        with st.spinner(f'3D 物理運算中...'):
            start_time = time.time()
            results = calculate_3d_physics(
                grid_n, v_top, v_bottom, v_left, v_right, v_front, v_back, max_iter, tolerance
            )
            end_time = time.time()
            st.session_state['res_3d_cart'] = (results, end_time - start_time)

    # 3. 繪圖與統計
    if st.session_state['res_3d_cart']:
        (X, Y, Z, V, Ex, Ey, Ez, actual_iter), elapsed_time = st.session_state['res_3d_cart']
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Max V", f"{np.max(V):.1f} V")
        E_mag = np.sqrt(Ex**2 + Ey**2 + Ez**2)
        c2.metric("Max |E|", f"{np.max(E_mag):.1f} V/m")
        c3.metric("Grid Points", f"{grid_n**3:,}")
        c4.metric("Time", f"{elapsed_time:.3f} s", help=f"Iter: {actual_iter}")

        st.divider()
        if viz_mode == "電位分佈 (Potential)":
            fig = create_potential_figure(X, Y, Z, V, opacity, surface_count, show_caps)
            st.plotly_chart(fig, use_container_width=True)
        else:
            # 使用優化後的 Log-Scale 繪圖
            fig = create_field_figure_log(X, Y, Z, Ex, Ey, Ez, cone_scale, stride_val)
            st.plotly_chart(fig, use_container_width=True)

def render_3d_point_charge():
    st.subheader("⚡ 3D 點電荷模擬")
    st.markdown("在 3D 空間中配置點電荷，即時計算電位與電場分佈。")

    with st.sidebar:
        st.markdown("---")
        viz_mode = st.radio(
            "選擇視覺化模式", ["電位分佈 (Potential)", "電場向量 (Electric Field)"], index=0
        )
        
        st.divider()
        st.header("⚙️ 空間設定")
        grid_range = st.slider("空間範圍 (±)", 1.0, 5.0, 2.5, 0.1)
        grid_res = st.slider("網格解析度", 10, 40, 20)

        st.divider()
        st.header("🔋 電荷管理")
        c1, c2, c3, c4 = st.columns([1,1,1,1.2])
        nx = c1.number_input("X", 0.0, step=0.5, key="nx")
        ny = c2.number_input("Y", 0.0, step=0.5, key="ny")
        nz = c3.number_input("Z", 0.0, step=0.5, key="nz")
        nq = c4.number_input("Q", value=1.0, step=1.0, key="nq")
        
        if st.button("➕ 新增電荷", use_container_width=True):
            st.session_state.point_charges_3d.append({'x':nx, 'y':ny, 'z':nz, 'q':nq})
        
        st.write("目前電荷：")
        if st.session_state.point_charges_3d:
            for i, c in enumerate(st.session_state.point_charges_3d):
                col_info, col_del = st.columns([4,1])
                col_info.caption(f"{i+1}. ({c['x']},{c['y']},{c['z']}) Q:{c['q']}")
                if col_del.button("❌", key=f"del3d_{i}"):
                    st.session_state.point_charges_3d.pop(i)
                    st.rerun()
        else:
            st.warning("無電荷")
            
        if st.button("🗑️ 清空", use_container_width=True):
            st.session_state.point_charges_3d = []
            st.rerun()

        st.divider()
        st.header("🎨 繪圖微調")
        if viz_mode == "電位分佈 (Potential)":
            surface_count = st.slider("等位面層數", 3, 20, 10)
            opacity = st.slider("透明度", 0.1, 1.0, 0.3)
            show_caps = st.checkbox("顯示封蓋", False)
        else:
            st.info("箭頭顏色(Rainbow)代表強度，箭頭長度固定。")
            cone_scale = st.slider("箭頭大小", 0.3, 1.0, 0.5)
            stride_val = st.slider("採樣間隔", 1, 3, 1)

    if not st.session_state.point_charges_3d:
        st.info("請先在左側新增電荷")
        return

    # 計算按鈕
    if st.sidebar.button("🚀 開始模擬", key="btn_3d_point"):
        charges_tuple = tuple(st.session_state.point_charges_3d)
        with st.spinner("3D 庫倫運算中..."):
            start_time = time.time()
            results = calculate_point_charge_field_3d(charges_tuple, grid_range, grid_res)
            end_time = time.time()
            st.session_state['res_3d_point'] = (results, end_time - start_time)

    # 繪圖
    if st.session_state['res_3d_point']:
        (X, Y, Z, V, Ex, Ey, Ez), elapsed_time = st.session_state['res_3d_point']
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Max V", f"{np.max(np.abs(V)):.1f} V") 
        E_mag = np.sqrt(Ex**2 + Ey**2 + Ez**2)
        c2.metric("Max |E|", f"{np.max(E_mag):.1f}")
        c3.metric("Time", f"{elapsed_time:.3f} s")

        st.divider()
        if viz_mode == "電位分佈 (Potential)":
            fig = create_potential_figure(X, Y, Z, V, opacity, surface_count, show_caps)
            # 加上電荷點
            for q in st.session_state.point_charges_3d:
                fig.add_trace(go.Scatter3d(
                    x=[q['x']], y=[q['y']], z=[q['z']],
                    mode='markers', marker=dict(size=5, color='white', line=dict(width=2, color='black')),
                    name=f"Q={q['q']}", showlegend=False
                ))
        else:
            # 使用優化後的 Log-Scale 繪圖
            fig = create_field_figure_log(X, Y, Z, Ex, Ey, Ez, cone_scale, stride_val)
            # 加上電荷點
            for q in st.session_state.point_charges_3d:
                color = 'red' if q['q'] > 0 else 'blue'
                fig.add_trace(go.Scatter3d(
                    x=[q['x']], y=[q['y']], z=[q['z']],
                    mode='markers+text', marker=dict(size=8, color=color, line=dict(width=2, color='black')),
                    text=[f"Q={q['q']}"], textposition="top center", name=f"Q={q['q']}"
                ))
                
        st.plotly_chart(fig, use_container_width=True)

def render_3d_spherical():
    st.subheader("⚡ 3D 球座標模擬：連續電荷分佈")
    st.markdown("模擬**連續電荷**在球體內的分布。程式會將球體切割成數千個微分單元，利用疊加原理計算場。")

    with st.sidebar:
        st.markdown("---")
        viz_mode = st.radio(
            "選擇視覺化模式", ["電位分佈 (Potential)", "電場向量 (Electric Field)"], index=0
        )
        
        st.divider()
        st.header("⚙️ 模擬設定")
        
        # 分佈類型選擇
        dist_type = st.selectbox(
            "電荷分佈模型",
            ["Uniform (均勻)", "Decaying (1/r)", "Orbital (p-like)"]
        )
        
        R = st.slider("球體半徑 (R)", 0.5, 2.0, 1.2)
        grid_range = st.slider("空間範圍 (±)", 1.0, 4.0, 2.0)
        grid_res = st.slider("網格解析度", 10, 30, 18)

        st.divider()
        st.header("🎨 繪圖微調")
        if viz_mode == "電位分佈 (Potential)":
            surface_count = st.slider("等位面層數", 3, 20, 10)
            opacity = st.slider("透明度", 0.1, 1.0, 0.3)
            show_caps = st.checkbox("顯示封蓋", False)
        else:
            st.info("箭頭顏色(Rainbow)代表強度，箭頭長度固定。")
            cone_scale = st.slider("箭頭大小", 0.05, 0.5, 0.15)
            stride_val = st.slider("採樣間隔", 1, 3, 1)

    # 計算按鈕
    if st.sidebar.button("🚀 開始模擬", key="btn_3d_continuous"):
        with st.spinner("正在進行微積分疊加運算 (這可能需要幾秒鐘)..."):
            start_time = time.time()
            results = calculate_continuous_spherical(dist_type, R, grid_range, grid_res)
            end_time = time.time()
            st.session_state['res_3d_continuous'] = (results, end_time - start_time)

    # 繪圖
    if st.session_state['res_3d_continuous']:
        (X, Y, Z, V, Ex, Ey, Ez, n_sources), elapsed_time = st.session_state['res_3d_continuous']
        
        c1, c2, c3 = st.columns(3)
        c1.metric("微分單元數", f"{n_sources}")
        E_mag = np.sqrt(Ex**2 + Ey**2 + Ez**2)
        c2.metric("Max |E|", f"{np.max(E_mag):.1f}")
        c3.metric("Time", f"{elapsed_time:.3f} s")

        st.divider()
        
        if viz_mode == "電位分佈 (Potential)":
            fig = create_potential_figure(X, Y, Z, V, opacity, surface_count, show_caps)
        else:
            fig = create_field_figure_log(X, Y, Z, Ex, Ey, Ez, cone_scale, stride_val)
            
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 6. 主導航邏輯
# ==========================================
st.sidebar.title("⚡ 導航選單")
cat = st.sidebar.selectbox("選擇模組", [
    "首頁", 
    "函數近似", 
    "電位+電場模擬 (2D)", 
    "電位+電場模擬 (3D)"
])

if cat == "首頁": 
    render_home()
elif cat == "函數近似":
    sub = st.sidebar.radio("方法", ["傅立葉近似", "勒讓德近似"])
    if sub == "傅立葉近似": render_fourier_page()
    else: render_legendre_page()
elif cat == "電位+電場模擬 (2D)":
    sub = st.sidebar.radio("結構", ["笛卡爾", "球座標", "點電荷"])
    if sub == "笛卡爾": render_laplace_cartesian_2d()
    elif sub == "球座標": render_potential_spherical_2d()
    elif sub == "點電荷": render_potential_point_charge()
elif cat == "電位+電場模擬 (3D)":
    sub = st.sidebar.radio("結構", ["笛卡爾", "球座標", "點電荷"])
    if sub == "笛卡爾": render_3d_cartesian()
    elif sub == "球座標": render_3d_spherical()
    elif sub == "點電荷": render_3d_point_charge()
    sub = st.sidebar.radio("結構", ["笛卡爾", "球座標", "點電荷"])
    if sub == "笛卡爾": render_3d_cartesian()
    elif sub == "球座標": render_3d_spherical()
    elif sub == "點電荷": render_3d_point_charge()
