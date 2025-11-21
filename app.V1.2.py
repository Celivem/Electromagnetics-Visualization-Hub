import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy import special, signal
from scipy.special import eval_legendre
import pandas as pd
import io
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor

# ==========================================
# 1. 全域頁面設定
# ==========================================
st.set_page_config(page_title="電磁學生成小教室", layout="wide", page_icon="⚡")

# 初始化 Session State
if 'fourier_result' not in st.session_state:
    st.session_state['fourier_result'] = None
if 'point_charges' not in st.session_state:
    st.session_state.point_charges = [{'q': 1.0, 'x': -2.0, 'y': 0.0}, {'q': -1.0, 'x': 2.0, 'y': 0.0}]

# CSS 美化
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1E88E5; text-align: center; margin-bottom: 1rem;}
    .stSlider {padding-top: 20px;}
    div.stButton > button:first-child {border-radius: 8px;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心運算函數 (已優化)
# ==========================================

def get_safe_math_scope(x_val=None):
    """統一管理數學函數的安全命名空間"""
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
    if x_val is not None:
        scope["x"] = x_val
    return scope

# --- 傅立葉 (新增 Cache) ---
@st.cache_data(show_spinner=False)
def calculate_fourier_coefficients(func_str, a, b, max_n):
    def f(x_val):
        return eval(func_str, {"__builtins__": None}, get_safe_math_scope(x_val))

    L = b - a
    if L <= 0: return None, "區間錯誤：b 必須大於 a"
    
    omega = 2 * np.pi / L
    A_coeffs, B_coeffs = [], []
    
    try:
        val_a0, _ = quad(lambda x: f(x), a, b, limit=200)
        A0 = (2.0 / L) * val_a0
    except Exception as e: return None, f"積分錯誤 (A0): {str(e)}"
    
    A_coeffs.append(A0); B_coeffs.append(0.0)
    
    # 優化：批量處理或保持循環但增加錯誤捕捉
    for n in range(1, max_n + 1):
        try:
            val_an, _ = quad(lambda x: f(x) * np.cos(n * omega * x), a, b, limit=100)
            an = (2.0 / L) * val_an
            val_bn, _ = quad(lambda x: f(x) * np.sin(n * omega * x), a, b, limit=100)
            bn = (2.0 / L) * val_bn
            A_coeffs.append(an); B_coeffs.append(bn)
        except Exception as e:
            return None, f"積分錯誤 (n={n}): {str(e)}"
    
    x_vals = np.linspace(a, b, 1000)
    try: 
        y_original = f(x_vals) # 嘗試向量化計算
    except: 
        y_original = [f(val) for val in x_vals] # 回退到列表推導
        
    return {"A": A_coeffs, "B": B_coeffs, "omega": omega, "x_vals": x_vals, "y_original": y_original, "range": (a, b)}, None

# --- 勒讓德 (保持 Cache) ---
@st.cache_data(show_spinner=False)
def calculate_legendre_coefficients(func_expression, max_n):
    def f(x_val):
        return eval(func_expression, {"__builtins__": None}, get_safe_math_scope(x_val))
    
    coeffs = []
    try: _ = f(0.5) 
    except Exception as e: return None, None, f"語法解析錯誤: {str(e)}"
    
    try:
        for n in range(max_n + 1):
            factor = (2 * n + 1) / 2
            integrand = lambda x: f(x) * eval_legendre(n, x)
            val, _ = quad(integrand, -1, 1, limit=100)
            coeffs.append(factor * val)
        return coeffs, None, None
    except Exception as e: return None, None, f"積分錯誤: {str(e)}"

# --- 點電荷 (增加 Cache) ---
@st.cache_data(show_spinner=False)
def calculate_point_charge_potential(charges_tuple, grid_size=100):
    # 注意：st.cache 需要不可變參數，所以 charges 需轉為 tuple 或確保其結構
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

# --- 輔助 ---
def plot_heatmap(data, title, xlabel="x", ylabel="y"):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data, cmap='jet', origin='lower', extent=[0, 1, 0, 1], aspect='auto', interpolation='bilinear')
    plt.colorbar(im, ax=ax).set_label('Potential (V)')
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig

def smart_parse(input_str):
    if not input_str or input_str.strip() == "0": return None
    transformations = (standard_transformations + (implicit_multiplication_application,) + (convert_xor,))
    try: return parse_expr(input_str, transformations=transformations, local_dict={'e': sp.E, 'pi': sp.pi})
    except: return None

# ==========================================
# 3. 頁面渲染邏輯
# ==========================================

def render_home():
    st.markdown("<h1 class='main-header'>⚡ 電磁學生成小教室 ⚡</h1>", unsafe_allow_html=True)
    st.markdown("""
    ### 歡迎來到互動學習實驗室！
    請從左側選單選擇您想要探索的主題：
    * **函數近似**：傅立葉級數、勒讓德多項式。
    * **電位模擬**：笛卡爾座標、球座標(極座標切面)、點電荷。
    * **電場模擬**：(開發中)。
    👈 **請點擊左上角箭頭打開側邊欄！**
    """)

def render_developing(title):
    st.subheader(f"🚧 {title}")
    st.info("此功能目前正在開發中，敬請期待！")

# --- 傅立葉 ---
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
            result, error = calculate_fourier_coefficients(func_str, a, b, max_n)
            if error: st.error(error)
            else: st.session_state['fourier_result'] = result; st.rerun()

    if st.session_state['fourier_result']:
        res = st.session_state['fourier_result']
        st.divider()
        current_n = st.slider("調整 N 值 (疊加項數)", 0, len(res["A"])-1, 10)
        
        # 向量化計算疊加，提升滑桿反應速度
        n_indices = np.arange(1, current_n + 1)
        A_terms = np.array(res["A"][1:current_n+1]).reshape(-1, 1)
        B_terms = np.array(res["B"][1:current_n+1]).reshape(-1, 1)
        k_omega_x = res["omega"] * np.outer(n_indices, res["x_vals"])
        
        y_approx = res["A"][0]/2 + np.sum(A_terms * np.cos(k_omega_x) + B_terms * np.sin(k_omega_x), axis=0)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        if res["y_original"] is not None: 
            ax.plot(res["x_vals"], res["y_original"], 'k-', alpha=0.3, label='Original')
        ax.plot(res["x_vals"], y_approx, 'b-', linewidth=2, label=f'N={current_n}')
        ax.legend(); ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

        df = pd.DataFrame({"n": range(len(res["A"])), "An": res["A"], "Bn": res["B"]})
        c1, c2 = st.columns(2)
        buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=150); buf.seek(0)
        c1.download_button("📥 下載圖表", buf, "fourier.png", "image/png", use_container_width=True)
        c2.download_button("📥 下載係數", df.to_csv(index=False, sep='\t').encode(), "coeffs.csv", "text/csv", use_container_width=True)
        with st.expander("查看係數表"): st.dataframe(df, use_container_width=True)

# --- 勒讓德 ---
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
            coeffs, _, error = calculate_legendre_coefficients(func_str, max_N)
            if error: st.error(error)
            else: st.session_state['legendre_coeffs'] = coeffs; st.session_state['legendre_func'] = func_str; st.rerun()

    if 'legendre_coeffs' in st.session_state:
        coeffs = st.session_state['legendre_coeffs']
        func_expr = st.session_state.get('legendre_func', func_str)
        
        st.divider()
        current_n = st.slider("疊加階數", 0, len(coeffs)-1, len(coeffs)-1)
        x = np.linspace(-1, 1, 400)
        
        try:
            y_target = eval(func_expr, {"__builtins__": None}, get_safe_math_scope(x))
        except: y_target = np.zeros_like(x)
        
        # 向量化計算
        y_approx = np.zeros_like(x)
        for n in range(current_n + 1): 
            y_approx += coeffs[n] * eval_legendre(n, x)
        
        fig = plt.figure(figsize=(12, 5))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(x, y_target, 'k--', alpha=0.3, label="Target"); ax1.plot(x, y_approx, 'r-', label="Approx")
        ax1.set_title("Cartesian Projection"); ax1.legend()
        
        ax2 = fig.add_subplot(1, 2, 2, projection='polar')
        theta = np.linspace(0, 2*np.pi, 400)
        r_approx = np.zeros_like(theta)
        for n in range(current_n + 1): 
            r_approx += coeffs[n] * eval_legendre(n, np.cos(theta))
        ax2.plot(theta, np.abs(r_approx), 'b-'); ax2.set_title("Polar Projection (Abs)")
        
        st.pyplot(fig)
        plt.close(fig)
        
        df = pd.DataFrame({"n": range(len(coeffs)), "cn": coeffs})
        c1, c2 = st.columns(2)
        buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=150); buf.seek(0)
        c1.download_button("📥 下載圖表", buf, "legendre.png", "image/png", use_container_width=True)
        c2.download_button("📥 下載係數", df.to_csv(index=False).encode(), "coeffs.csv", "text/csv", use_container_width=True)
        with st.expander("查看係數表"): st.dataframe(df, use_container_width=True)

# --- 電位: 點電荷 ---
def render_potential_point_charge():
    st.subheader("⚡ 點電荷電位與電場模擬")
    st.sidebar.markdown("---")
    st.sidebar.header("🔋 電荷控制")
    
    c1, c2 = st.sidebar.columns(2); new_q = c1.number_input("電荷量 q", 1.0, step=0.5)
    c3, c4 = st.sidebar.columns(2); new_x = c3.number_input("X 座標", 0.0, step=0.5); new_y = c4.number_input("Y 座標", 0.0, step=0.5)
    
    if st.sidebar.button("➕ 加入電荷", use_container_width=True):
        st.session_state.point_charges.append({'q': new_q, 'x': new_x, 'y': new_y})
    if st.sidebar.button("🗑️ 清除所有", use_container_width=True):
        st.session_state.point_charges = []
        
    st.sidebar.divider()
    st.sidebar.subheader(f"目前電荷 ({len(st.session_state.point_charges)})")
    for i, c in enumerate(st.session_state.point_charges): 
        st.sidebar.text(f"{i+1}. q={c['q']}, ({c['x']},{c['y']})")
        
    show_stream = st.sidebar.checkbox("顯示流線", True)
    grid_res = st.sidebar.slider("網格解析度", 50, 300, 100)

    if st.session_state.point_charges:
        # 使用 tuple 傳遞給 cache 函數以確保可雜湊
        X, Y, V = calculate_point_charge_potential(tuple(st.session_state.point_charges), grid_res)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        contour = ax.contourf(X, Y, V, levels=50, cmap='RdBu_r', extend='both')
        ax.contour(X, Y, V, levels=50, colors='k', alpha=0.4, linewidths=0.5)
        
        if show_stream:
            Ey, Ex = np.gradient(-V)
            # 避免除以零或過大數值影響流線
            mag = np.sqrt(Ex**2 + Ey**2)
            Ex = np.where(mag > 0, Ex, 0)
            Ey = np.where(mag > 0, Ey, 0)
            ax.streamplot(X, Y, Ex, Ey, color='#444444', density=1.2, linewidth=0.6, arrowsize=1)
            
        for c in st.session_state.point_charges:
            col = '#d62728' if c['q']>0 else '#1f77b4'
            s = '+' if c['q']>0 else '-'
            ax.plot(c['x'], c['y'], marker='o', color=col, markersize=15, markeredgecolor='k')
            ax.text(c['x'], c['y'], s, color='w', ha='center', va='center', fontweight='bold')
            
        ax.set_aspect('equal'); ax.set_title("Electric Potential & Field"); fig.colorbar(contour, ax=ax)
        st.pyplot(fig)
        plt.close(fig)
    else: st.warning("請由左側欄位加入電荷")

# --- 電位: 笛卡爾 ---
def render_laplace_cartesian():
    st.subheader("🔲 電位模擬 - 笛卡爾座標")
    mode = st.radio("計算模式", ["數值解 (FDM)", "解析解 (Separation of Variables)"], horizontal=True)
    
    if mode == "數值解 (FDM)":
        c1, c2 = st.columns([1, 3])
        with c1:
            st.markdown("##### 邊界條件")
            def inp(l, d):
                inf = st.checkbox(f"{l} 接地/無窮", key=f"i_{l}")
                return (True, 0.0) if inf else (False, st.number_input(f"{l} 電位 (V)", float(d), key=f"v_{l}"))
            ti, tv = inp("上", 10.0); bi, bv = inp("下", 0.0); li, lv = inp("左", 0.0); ri, rv = inp("右", 0.0)
            iters = st.slider("迭代次數", 500, 5000, 2000)
            
        with c2:
            if st.button("開始模擬", use_container_width=True):
                sz = 40 # 核心網格大小
                pad = sz * 3 # 邊界填充
                # 根據是否無窮遠決定網格大小
                th = (pad if ti else 0) + sz + (pad if bi else 0)
                tw = (pad if li else 0) + sz + (pad if ri else 0)
                
                V = np.zeros((th, tw))
                
                # 定義 ROI (Region of Interest)
                rs = pad if bi else 0
                re = rs + sz
                cs = pad if li else 0
                ce = cs + sz
                
                # 初始化邊界值 (優化：在循環外設置固定邊界)
                if not ti: V[re-1, cs:ce] = tv
                if not bi: V[rs, cs:ce] = bv
                if not li: V[rs:re, cs] = lv
                if not ri: V[rs:re, ce-1] = rv

                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 迭代求解 (優化：雖然 Python loop 慢，但對於 50x50 網格尚可接受，這裡保持結構簡單)
                # 若要極致效能，應使用 numba 或 scipy.signal.convolve2d
                for i in range(iters):
                    V_old = V.copy()
                    # Jacobi / Gauss-Seidel Update (Vectorized)
                    V[1:-1, 1:-1] = 0.25 * (V_old[0:-2, 1:-1] + V_old[2:, 1:-1] + V_old[1:-1, 0:-2] + V_old[1:-1, 2:])
                    
                    # 強制邊界條件 (Dirichlet)
                    if not ti: V[re-1, cs:ce] = tv
                    if not bi: V[rs, cs:ce] = bv
                    if not li: V[rs:re, cs] = lv
                    if not ri: V[rs:re, ce-1] = rv
                    
                    # 處理 "無窮遠" (模擬邊界接地 0) - 這裡已經隱含在 V=0 初始化與不更新邊界中
                    if i % (iters//10) == 0:
                        progress_bar.progress((i+1)/iters)
                
                progress_bar.progress(1.0)
                status_text.success("模擬完成！")
                st.pyplot(plot_heatmap(V[rs:re, cs:ce], "FDM Result (Central Region)"))
                
    elif mode == "解析解 (Separation of Variables)":
        st.info("輸入支援 Python 語法，例如 `x`, `sin(pi*x)`")
        c1, c2 = st.columns(2)
        ts = c1.text_input("V(x,1)", "10"); bs = c1.text_input("V(x,0)", "0")
        ls = c2.text_input("V(0,y)", "0"); rs = c2.text_input("V(1,y)", "0")
        
        if st.button("推導與計算", use_container_width=True):
            x, y, n = sp.symbols('x y n'); pi = sp.pi; terms = []
            
            def calculate_boundary_contribution(input_s, side):
                ex = smart_parse(input_s)
                if not ex: return None
                
                # 根據邊界變換積分變數
                var = x if side in ['left', 'right'] else y
                integrand = ex.subs(x if side in ['top', 'bottom'] else y, x)
                
                try:
                    An = 2 * sp.integrate(integrand * sp.sin(n*pi*x), (x, 0, 1))
                except: return None # 積分失敗忽略

                den = sp.sinh(n*pi)
                if side == 'top': return An * sp.sin(n*pi*x) * sp.sinh(n*pi*y) / den
                if side == 'bottom': return An * sp.sin(n*pi*x) * sp.sinh(n*pi*(1-y)) / den
                if side == 'left': return An * sp.sin(n*pi*y) * sp.sinh(n*pi*(1-x)) / den
                if side == 'right': return An * sp.sin(n*pi*y) * sp.sinh(n*pi*x) / den
                return None

            with st.spinner("符號運算推導中..."):
                for s, sd in [(ts,'top'), (bs,'bottom'), (ls,'left'), (rs,'right')]:
                    r = calculate_boundary_contribution(s, sd)
                    if r: terms.append(r)
            
            if terms:
                Vt = sum(terms)
                st.latex(f"V(x,y) = \\sum_{{n=1}}^{{\\infty}} \\left[ {sp.latex(Vt)} \\right]")
                
                X, Y = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
                Vn = np.zeros_like(X)
                
                # 預先編譯函數以加速循環
                try:
                    fn = sp.lambdify((n, x, y), Vt, 'numpy')
                    progress_bar = st.progress(0)
                    for i in range(1, 21): 
                        term_val = fn(i, X, Y)
                        Vn += np.nan_to_num(term_val)
                        progress_bar.progress(i/20)
                    
                    st.pyplot(plot_heatmap(Vn, "Analytical Solution (First 20 terms)"))
                except Exception as e:
                    st.error(f"數值計算錯誤: {e}")
            else:
                st.warning("沒有有效的邊界條件輸入或積分結果為零")

# --- 電位: 球座標 ---
def render_potential_spherical():
    st.subheader("🌐 2D 極座標/球座標切面電位分析")
    st.markdown("輸入電位 $V(r, \\theta)$，程式將計算電場 $\\vec{E} = -\\nabla V$ 並繪圖。")

    presets = {
        "點電荷": "k / r",
        "電偶極": "k * cos(theta) / r^2",
        "電四極": "k * (3*cos(theta)**2 - 1) / r^3",
        "均勻電場": "-k * r * cos(theta)",
        "殼內電位": "r * sin(theta)"
    }

    st.sidebar.markdown("---")
    st.sidebar.markdown("**極座標參數**")
    sel = st.sidebar.selectbox("選擇模型", list(presets.keys()), index=1)
    user_input = st.sidebar.text_input("輸入 V(r, theta)", value=presets[sel])
    
    rmax = st.sidebar.slider("半徑範圍", 1.0, 10.0, 5.0)
    grid_res = st.sidebar.slider("網格解析度", 50, 300, 100)
    show_lines = st.sidebar.checkbox("顯示電場線", True)

    if user_input:
        try:
            # SymPy 解析 (加上快取邏輯會更好，但這裡直接運算也很快)
            r, theta, k = sp.symbols('r theta k', real=True)
            trans = (standard_transformations + (implicit_multiplication_application,) + (convert_xor,))
            local_d = {'k': k, 'pi': sp.pi, 'e': sp.E, 'r': r, 'theta': theta}
            
            try:
                V_expr = parse_expr(user_input, local_dict=local_d, transformations=trans)
            except Exception as e:
                st.error(f"公式解析失敗: {e}"); return

            E_r = -sp.diff(V_expr, r)
            E_theta = -(1/r) * sp.diff(V_expr, theta)

            c1, c2 = st.columns(2)
            with c1: st.markdown("**電位 V**"); st.latex(sp.latex(V_expr))
            with c2: st.markdown("**電場 E**"); st.latex(f"E_r = {sp.latex(E_r)}, \\quad E_\\theta = {sp.latex(E_theta)}")

            # 數值化
            func_V = sp.lambdify((r, theta), V_expr.subs(k, 1), 'numpy')
            func_Er = sp.lambdify((r, theta), E_r.subs(k, 1), 'numpy')
            func_Et = sp.lambdify((r, theta), E_theta.subs(k, 1), 'numpy')

            # 網格生成
            x = np.linspace(-rmax, rmax, grid_res)
            X, Y = np.meshgrid(x, x)
            R = np.sqrt(X**2 + Y**2)
            THETA = np.arctan2(Y, X)
            # 遮罩掉過小的半徑避免奇異點
            mask = R < 0.1
            R = np.maximum(R, 0.1)

            Z_V = func_V(R, THETA)
            if np.isscalar(Z_V): Z_V = np.full_like(R, Z_V)
            Z_V[mask] = np.nan

            fig, ax = plt.subplots(figsize=(8, 7))
            try:
                contour = ax.contourf(X, Y, Z_V, levels=50, cmap='viridis')
                plt.colorbar(contour, ax=ax, label='Potential (V)')
            except: st.warning("數值範圍過大，無法繪製等位面")

            if show_lines:
                U_Er = func_Er(R, THETA)
                U_Et = func_Et(R, THETA)
                if np.isscalar(U_Er): U_Er = np.full_like(R, U_Er)
                if np.isscalar(U_Et): U_Et = np.full_like(R, U_Et)
                
                # 轉回直角座標向量
                Ex = U_Er * np.cos(THETA) - U_Et * np.sin(THETA)
                Ey = U_Er * np.sin(THETA) + U_Et * np.cos(THETA)
                
                ax.streamplot(X, Y, np.nan_to_num(Ex), np.nan_to_num(Ey), color=(1,1,1,0.5), density=1.2, linewidth=0.8)

            ax.set_aspect('equal'); ax.set_title("Potential & Field Lines")
            ax.set_xlim(-rmax, rmax); ax.set_ylim(-rmax, rmax)
            st.pyplot(fig)
            plt.close(fig)

        except Exception as e:
            st.error(f"運算錯誤: {e}")

# ==========================================
# 4. 主導航邏輯
# ==========================================
st.sidebar.title("⚡ 導航選單")
cat = st.sidebar.selectbox("選擇模組", ["首頁", "函數近似", "電位模擬", "電場模擬"])

if cat == "首頁": render_home()
elif cat == "函數近似":
    sub = st.sidebar.radio("方法", ["傅立葉近似", "勒讓德近似"])
    if sub == "傅立葉近似": render_fourier_page()
    else: render_legendre_page()
elif cat == "電位模擬":
    sub = st.sidebar.radio("結構", ["笛卡爾 (Cartesian)", "球座標 (Spherical)", "柱座標", "點電荷"])
    if sub == "笛卡爾 (Cartesian)": render_laplace_cartesian()
    elif sub == "球座標 (Spherical)": render_potential_spherical()
    elif sub == "點電荷": render_potential_point_charge()
    else: render_developing(f"電位模擬 - {sub}")
else:
    sub = st.sidebar.radio("結構", ["笛卡爾", "球座標", "柱座標", "點電荷"])
    render_developing(f"電場模擬 - {sub}")
