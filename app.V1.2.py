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
# 1. 全域頁面設定與初始化
# ==========================================
st.set_page_config(page_title="電磁學生成小教室", layout="wide", page_icon="⚡")

# 初始化 Session State
default_states = {
    'fourier_result': None,
    'point_charges': [{'q': 1.0, 'x': -2.0, 'y': 0.0}, {'q': -1.0, 'x': 2.0, 'y': 0.0}],
    'legendre_coeffs': None,
    'legendre_func': None
}

for key, val in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = val

# CSS 美化
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1E88E5; text-align: center; margin-bottom: 1rem;}
    .stSlider {padding-top: 20px;}
    div.stButton > button:first-child {border-radius: 8px;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心運算函數 (優化版)
# ==========================================

def get_safe_math_scope(x_val=None):
    """建立安全的數學運算命名空間"""
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

def eval_func(func_str, x_val):
    """
    智能評估函數：
    自動處理 scalar (用於 quad 積分) 與 vector (用於繪圖) 的輸入差異
    """
    scope = get_safe_math_scope(x_val)
    try:
        return eval(func_str, {"__builtins__": None}, scope)
    except Exception:
        # 若向量化失敗，回退到列表推導 (List Comprehension)
        if hasattr(x_val, "__len__"):
            return np.array([eval(func_str, {"__builtins__": None}, get_safe_math_scope(xi)) for xi in x_val])
        return np.nan

def smart_parse(input_str):
    """SymPy 智能解析"""
    if not input_str or input_str.strip() == "0": return None
    transformations = (standard_transformations + (implicit_multiplication_application,) + (convert_xor,))
    try:
        return parse_expr(input_str, transformations=transformations, local_dict={'e': sp.E, 'pi': sp.pi})
    except:
        return None

# --- 快取運算核心 ---

@st.cache_data(show_spinner=False)
def calculate_fourier_coefficients(func_str, a, b, max_n):
    L = b - a
    if L <= 0: return None, "區間錯誤：b 必須大於 a"
    
    omega = 2 * np.pi / L
    A_coeffs, B_coeffs = [], []
    
    try:
        # 計算 A0
        val_a0, _ = quad(lambda x: eval_func(func_str, x), a, b, limit=200)
        A_coeffs.append((2.0 / L) * val_a0)
        B_coeffs.append(0.0)
        
        # 計算 An, Bn
        for n in range(1, max_n + 1):
            val_an, _ = quad(lambda x: eval_func(func_str, x) * np.cos(n * omega * x), a, b, limit=100)
            val_bn, _ = quad(lambda x: eval_func(func_str, x) * np.sin(n * omega * x), a, b, limit=100)
            A_coeffs.append((2.0 / L) * val_an)
            B_coeffs.append((2.0 / L) * val_bn)
            
        x_vals = np.linspace(a, b, 1000)
        y_original = eval_func(func_str, x_vals)
        
        return {
            "A": A_coeffs, 
            "B": B_coeffs, 
            "omega": omega, 
            "x_vals": x_vals, 
            "y_original": y_original, 
            "range": (a, b)
        }, None
        
    except Exception as e:
        return None, f"運算錯誤: {str(e)}"

@st.cache_data(show_spinner=False)
def calculate_legendre_coefficients(func_expression, max_n):
    try:
        # 預先檢查語法
        _ = eval_func(func_expression, 0.5)
    except Exception as e:
        return None, None, f"語法解析錯誤: {str(e)}"
    
    coeffs = []
    try:
        for n in range(max_n + 1):
            factor = (2 * n + 1) / 2
            # 定義被積函數
            integrand = lambda x: eval_func(func_expression, x) * eval_legendre(n, x)
            val, _ = quad(integrand, -1, 1, limit=100)
            coeffs.append(factor * val)
        return coeffs, None, None
    except Exception as e:
        return None, None, f"積分錯誤: {str(e)}"

@st.cache_data(show_spinner=False)
def calculate_point_charge_potential(charges_tuple, grid_size=100):
    # 將 tuple 轉回 list 進行處理
    charges = list(charges_tuple)
    x = np.linspace(-5, 5, grid_size)
    y = np.linspace(-5, 5, grid_size)
    X, Y = np.meshgrid(x, y)
    V_total = np.zeros_like(X)
    
    if not charges: return X, Y, V_total
    
    for charge in charges:
        q = charge['q']
        x0 = charge['x']
        y0 = charge['y']
        r = np.sqrt((X - x0)**2 + (Y - y0)**2)
        # 加上微小值避免除以零
        V_total += q / (r + 1e-9) 
    return X, Y, V_total

# --- 輔助繪圖 ---
def plot_heatmap(data, title, xlabel="x", ylabel="y"):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data, cmap='jet', origin='lower', extent=[0, 1, 0, 1], aspect='auto', interpolation='bilinear')
    plt.colorbar(im, ax=ax).set_label('Potential (V)')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig

# ==========================================
# 3. 頁面渲染邏輯
# ==========================================

def render_home():
    st.markdown("<h1 class='main-header'>⚡ 電磁學生成小教室 ⚡</h1>", unsafe_allow_html=True)
    st.markdown("""
    ### 歡迎來到互動學習實驗室！
    請從左側選單選擇您想要探索的主題：
    * **函數近似**：傅立葉級數、勒讓德多項式。
    * **電位+電場模擬(2D)**：笛卡爾座標、球座標、點電荷。
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
            result, error = calculate_fourier_coefficients(func_str, a, b, int(max_n))
            if error: st.error(error)
            else: st.session_state['fourier_result'] = result

    if st.session_state['fourier_result']:
        res = st.session_state['fourier_result']
        st.divider()
        
        # 滑桿互動
        current_n = st.slider("調整 N 值 (疊加項數)", 0, len(res["A"])-1, min(10, len(res["A"])-1))
        
        # 向量化計算 (Optimization: Vectorized Summation)
        # 避免在 Python 迴圈中逐項累加
        n_indices = np.arange(1, current_n + 1)
        A_terms = np.array(res["A"][1:current_n+1]).reshape(-1, 1)
        B_terms = np.array(res["B"][1:current_n+1]).reshape(-1, 1)
        
        # 建立矩陣: [n, x]
        k_omega_x = res["omega"] * np.outer(n_indices, res["x_vals"])
        
        # 矩陣運算求和
        y_approx = res["A"][0]/2 + np.sum(A_terms * np.cos(k_omega_x) + B_terms * np.sin(k_omega_x), axis=0)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        if res["y_original"] is not None:
            ax.plot(res["x_vals"], res["y_original"], 'k-', alpha=0.3, label='Original')
        ax.plot(res["x_vals"], y_approx, 'b-', linewidth=2, label=f'N={current_n}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        plt.close(fig) # 釋放記憶體

        # 資料下載區
        df = pd.DataFrame({"n": range(len(res["A"])), "An": res["A"], "Bn": res["B"]})
        c1, c2 = st.columns(2)
        buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=150); buf.seek(0)
        c1.download_button("📥 下載圖表", buf, "fourier.png", "image/png", use_container_width=True)
        c2.download_button("📥 下載係數", df.to_csv(index=False, sep='\t').encode(), "coeffs.csv", "text/csv", use_container_width=True)
        with st.expander("查看係數表"): st.dataframe(df, use_container_width=True)

# --- 勒讓德  ---
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
            else: 
                st.session_state['legendre_coeffs'] = coeffs
                st.session_state['legendre_func'] = func_str

    if 'legendre_coeffs' in st.session_state and st.session_state['legendre_coeffs']:
        coeffs = st.session_state['legendre_coeffs']
        func_expr = st.session_state.get('legendre_func', func_str)
        
        st.divider()
        current_n = st.slider("疊加階數", 0, len(coeffs)-1, len(coeffs)-1)
        
        # 1. 準備笛卡爾座標數據
        x = np.linspace(-1, 1, 400)
        y_target = eval_func(func_expr, x)
        y_approx = sum(coeffs[n] * eval_legendre(n, x) for n in range(current_n + 1))
        
        fig = plt.figure(figsize=(12, 5))
        
        # --- 子圖 1: 笛卡爾座標 (Cartesian) ---
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(x, y_target, 'k--', alpha=0.5, label="Target")
        ax1.plot(x, y_approx, 'r-', linewidth=2, label="Approx")
        ax1.set_title("Cartesian Projection (x vs f(x))")
        ax1.set_xlabel("x = cos(theta)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # --- 子圖 2: 極座標 (Polar) ---
        # 對應關係：x = cos(theta)
        ax2 = fig.add_subplot(1, 2, 2, projection='polar')
        theta = np.linspace(0, 2*np.pi, 400)
        
        # 計算目標函數在極座標下的值
        # 我們將 x 替換為 cos(theta) 來獲得目標輪廓
        r_target_polar = eval_func(func_expr, np.cos(theta))
        
        # 計算近似值
        r_approx = sum(coeffs[n] * eval_legendre(n, np.cos(theta)) for n in range(current_n + 1))
        
        # 繪圖 (取絕對值 abs 以便在極座標半徑中顯示大小)
        ax2.plot(theta, np.abs(r_target_polar), 'k--', alpha=0.5, label='Target')
        ax2.plot(theta, np.abs(r_approx), 'r-', linewidth=2, label='Approx')
        
        ax2.set_title("Polar Projection (Abs magnitude)")
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1)) # 將圖例移出避免遮擋
        
        st.pyplot(fig)
        plt.close(fig)
        
        df = pd.DataFrame({"n": range(len(coeffs)), "cn": coeffs})
        c1, c2 = st.columns(2)
        buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=150); buf.seek(0)
        c1.download_button("📥 下載圖表", buf, "legendre.png", "image/png", use_container_width=True)
        c2.download_button("📥 下載係數", df.to_csv(index=False).encode(), "coeffs.csv", "text/csv", use_container_width=True)
        with st.expander("查看係數表"): st.dataframe(df, use_container_width=True)

# --- 電位模擬 (點電荷) ---
def render_potential_point_charge():
    st.subheader("⚡ 點電荷電位與電場模擬")
    st.markdown("透過側邊欄新增電荷，即時觀察電位 ($V$) 與電場線 ($E$) 的變化。")

    # 側邊欄控制 (專屬於此頁面)
    st.sidebar.markdown("---")
    st.sidebar.header("🔋 電荷控制")
    
    col1, col2 = st.sidebar.columns(2)
    new_q = col1.number_input("電荷量 (q)", value=1.0, step=0.5)
    
    col3, col4 = st.sidebar.columns(2)
    new_x = col3.number_input("X 座標", value=0.0, step=0.5, min_value=-5.0, max_value=5.0)
    new_y = col4.number_input("Y 座標", value=0.0, step=0.5, min_value=-5.0, max_value=5.0)

    if st.sidebar.button("➕ 加入電荷", use_container_width=True):
        st.session_state.point_charges.append({'q': new_q, 'x': new_x, 'y': new_y})
    
    if st.sidebar.button("🗑️ 清除所有電荷", use_container_width=True):
        st.session_state.point_charges = []
        
    st.sidebar.divider()
    st.sidebar.subheader("目前電荷列表")
    if not st.session_state.point_charges:
        st.sidebar.info("目前沒有電荷")
    else:
        for i, c in enumerate(st.session_state.point_charges):
            st.sidebar.text(f"{i+1}. q={c['q']}, pos=({c['x']}, {c['y']})")
            
    st.sidebar.divider()
    show_streamlines = st.sidebar.checkbox("顯示電場流線 (Streamlines)", value=True)
    grid_res = st.sidebar.slider("網格解析度", 50, 200, 100)

    # 主畫面繪圖
    if st.session_state.point_charges:
        X, Y, V = calculate_point_charge_potential(st.session_state.point_charges, grid_res)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        v_levels = np.linspace(-3, 3, 50)
        contour = ax.contourf(X, Y, V, levels=v_levels, cmap='RdBu_r', extend='both')
        ax.contour(X, Y, V, levels=v_levels, colors='k', linewidths=0.5, alpha=0.4)
        
        if show_streamlines:
            Ey, Ex = np.gradient(-V)
            ax.streamplot(X, Y, Ex, Ey, color='#444444', density=1.2, linewidth=0.6, arrowsize=1)
        
        for charge in st.session_state.point_charges:
            color = '#d62728' if charge['q'] > 0 else '#1f77b4'
            sign = '+' if charge['q'] > 0 else '-'
            ax.plot(charge['x'], charge['y'], marker='o', color=color, markersize=15, markeredgecolor='black')
            ax.text(charge['x'], charge['y'], sign, color='white', ha='center', va='center', fontweight='bold')

        ax.set_aspect('equal')
        ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
        ax.set_title("Electric Potential Landscape")
        fig.colorbar(contour, ax=ax, label='Electric Potential (V)')
        st.pyplot(fig)
    else:
        st.warning("請在左側側邊欄加入至少一個電荷以開始模擬。")

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
            
            ti, tv = inp("上", 10.0)
            bi, bv = inp("下", 0.0)
            li, lv = inp("左", 0.0)
            ri, rv = inp("右", 0.0)
            iters = st.slider("迭代次數", 500, 5000, 2000)
            
        with c2:
            if st.button("開始模擬", use_container_width=True):
                sz = 40 
                pad = sz * 3
                
                # 根據邊界是否為無窮遠動態調整網格
                th = (pad if ti else 0) + sz + (pad if bi else 0)
                tw = (pad if li else 0) + sz + (pad if ri else 0)
                
                V = np.zeros((th, tw))
                
                rs = pad if bi else 0
                re = rs + sz
                cs = pad if li else 0
                ce = cs + sz
                
                # 初始邊界賦值
                if not ti: V[re-1, cs:ce] = tv
                if not bi: V[rs, cs:ce] = bv
                if not li: V[rs:re, cs] = lv
                if not ri: V[rs:re, ce-1] = rv
                
                progress_bar = st.progress(0)
                
                # FDM 迭代
                for i in range(iters):
                    V_old = V.copy()
                    V[1:-1, 1:-1] = 0.25 * (V_old[0:-2, 1:-1] + V_old[2:, 1:-1] + V_old[1:-1, 0:-2] + V_old[1:-1, 2:])
                    
                    # 強制邊界條件
                    if not ti: V[re-1, cs:ce] = tv
                    if not bi: V[rs, cs:ce] = bv
                    if not li: V[rs:re, cs] = lv
                    if not ri: V[rs:re, ce-1] = rv
                    
                    if i % (iters // 10) == 0:
                        progress_bar.progress((i + 1) / iters)
                
                progress_bar.progress(1.0)
                st.pyplot(plot_heatmap(V[rs:re, cs:ce], "FDM Result (Central Region)"))
                
    elif mode == "解析解 (Separation of Variables)":
        st.info("輸入支援 Python 語法，例如 `x`, `sin(pi*x)`")
        c1, c2 = st.columns(2)
        ts = c1.text_input("V(x,1)", "10")
        bs = c1.text_input("V(x,0)", "0")
        ls = c2.text_input("V(0,y)", "0")
        rs = c2.text_input("V(1,y)", "0")
        
        if st.button("推導與計算", use_container_width=True):
            x, y, n = sp.symbols('x y n')
            pi = sp.pi
            terms = []
            
            def calculate_boundary_contribution(input_s, side):
                ex = smart_parse(input_s)
                if not ex: return None
                
                var = x if side in ['left', 'right'] else y
                integrand = ex.subs(x if side in ['top', 'bottom'] else y, x)
                
                try:
                    An = 2 * sp.integrate(integrand * sp.sin(n * pi * x), (x, 0, 1))
                except: return None

                den = sp.sinh(n * pi)
                if side == 'top': return An * sp.sin(n*pi*x) * sp.sinh(n*pi*y) / den
                if side == 'bottom': return An * sp.sin(n*pi*x) * sp.sinh(n*pi*(1-y)) / den
                if side == 'left': return An * sp.sin(n*pi*y) * sp.sinh(n*pi*(1-x)) / den
                if side == 'right': return An * sp.sin(n*pi*y) * sp.sinh(n*pi*x) / den
                return None

            for s, sd in [(ts, 'top'), (bs, 'bottom'), (ls, 'left'), (rs, 'right')]:
                r = calculate_boundary_contribution(s, sd)
                if r: terms.append(r)
            
            if terms:
                Vt = sum(terms)
                st.latex(f"V(x,y) = \\sum_{{n=1}}^{{\\infty}} \\left[ {sp.latex(Vt)} \\right]")
                
                X, Y = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
                Vn = np.zeros_like(X)
                
                try:
                    # 轉為 numpy 函數前先處理符號
                    fn = sp.lambdify((n, x, y), Vt, 'numpy')
                    progress_bar = st.progress(0)
                    
                    for i in range(1, 21): 
                        Vn += np.nan_to_num(fn(i, X, Y))
                        progress_bar.progress(i / 20)
                    
                    st.pyplot(plot_heatmap(Vn, "Analytical Solution (First 20 terms)"))
                except Exception as e:
                    st.error(f"數值計算錯誤: {e}")
            else:
                st.warning("沒有有效的邊界條件輸入或積分結果為零")

# --- 電位: 球座標 (新整合功能) ---
def render_potential_spherical():
    st.subheader("🌐 2D 極座標/球座標切面電位分析")
    st.markdown(r"""
    輸入電位函數 $V(r, \theta)$，程式將自動計算電場 $\vec{E} = -\nabla V$ 並繪製分佈圖。
    """)
    
    # --- 定義預設範例庫 ---
    PRESETS = {
        "自定義 (Custom)": "",
        "點電荷 (Point Charge)": "k / r",
        "電偶極 (Electric Dipole)": "k * cos(theta) / r^2",
        "電四極 (Electric Quadrupole)": "k * (3*cos(theta)**2 - 1) / r^3",
        "均勻電場 (Uniform Field)": "-k * r * cos(theta)",
        "簡單範例 ": "sin(theta)",
        "複雜組合範例": "k/r + r*cos(theta)"
    }

    # --- 側邊欄設定 ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("**極座標參數設定**")
    
    # 使用 key 避免重整時重置
    selected_preset = st.sidebar.selectbox("選擇模型", list(PRESETS.keys()), index=2, key="sp_preset")
    default_value = PRESETS[selected_preset]
    
    user_input = st.sidebar.text_input(
        "輸入 V(r, theta)", 
        value=default_value,
        help="支援變數: r, theta, k。例如: k*cos(theta)/r^2"
    )

    rmax = st.sidebar.slider("最大範圍 (XY軸邊界)", 1.0, 10.0, 5.0)
    grid_res = st.sidebar.slider("網格解析度", 50, 300, 100)
    show_field_lines = st.sidebar.checkbox("顯示電場線 (Streamlines)", value=True)

    if not user_input:
        st.info("👈 請在左側輸入公式或選擇範例以開始。")
        return

    try:
        # --- 1. SymPy 解析 ---
        r, theta, k = sp.symbols('r theta k', real=True)
        
        transformations = (standard_transformations + 
                           (implicit_multiplication_application,) + 
                           (convert_xor,))
        
        local_dict = {'k': k, 'pi': sp.pi, 'e': sp.E, 'r': r, 'theta': theta}
        
        V_expr = parse_expr(user_input, local_dict=local_dict, transformations=transformations)
        V_expr = sp.simplify(V_expr)

        # 計算電場 (Gradient in Polar Coordinates)
        # E = - grad V = - (dV/dr * r_hat + (1/r)*dV/dtheta * theta_hat)
        E_r = -sp.diff(V_expr, r)
        E_theta = -(1/r) * sp.diff(V_expr, theta)

        # --- 2. 顯示數學結果 ---
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**電位 Potential $V(r, \\theta)$:**")
            st.latex(f"V = {sp.latex(V_expr)}")
        with c2:
            st.markdown("**電場 Electric Field $\\vec{E}$:**")
            st.latex(r"E_r = " + sp.latex(sp.simplify(E_r)))
            st.latex(r"E_\theta = " + sp.latex(sp.simplify(E_theta)))

        # --- 3. 數值計算準備 ---
        # 令 k=1 進行數值模擬
        V_num = V_expr.subs(k, 1)
        Er_num = E_r.subs(k, 1)
        Etheta_num = E_theta.subs(k, 1)

        # 轉為 Python 函數 (Numpy backend)
        func_V = sp.lambdify((r, theta), V_num, modules=['numpy'])
        func_Er = sp.lambdify((r, theta), Er_num, modules=['numpy'])
        func_Etheta = sp.lambdify((r, theta), Etheta_num, modules=['numpy'])

        # --- 4. 建立網格與座標轉換 ---
        # 使用 Cartesian Grid 滿足 streamplot 需求
        x_vals = np.linspace(-rmax, rmax, grid_res)
        y_vals = np.linspace(-rmax, rmax, grid_res)
        X, Y = np.meshgrid(x_vals, y_vals)

        # 直角 -> 極座標
        R = np.sqrt(X**2 + Y**2)
        THETA = np.arctan2(Y, X)
        
        # 處理奇異點 (r=0)
        R[R < 1e-3] = 1e-3

        # 計算電位值
        Z_V = func_V(R, THETA)
        if np.isscalar(Z_V): Z_V = np.full_like(R, Z_V)

        # --- 5. 繪圖 ---
        fig, ax = plt.subplots(figsize=(8, 7))
        
        # 繪製電位 (使用 'jet' 呈現彩虹色)
        try:
            # levels=50 讓漸層更平滑
            contour = ax.contourf(X, Y, Z_V, levels=50, cmap='jet') 
            plt.colorbar(contour, ax=ax, label='Potential V (Volts)')
        except ValueError:
            st.warning("數值範圍過大或包含複數，無法繪製電位圖。")

        # 繪製電場線
        if show_field_lines:
            # 計算極座標下的電場分量
            U_Er = func_Er(R, THETA)
            U_Etheta = func_Etheta(R, THETA)
            
            if np.isscalar(U_Er): U_Er = np.full_like(R, U_Er)
            if np.isscalar(U_Etheta): U_Etheta = np.full_like(R, U_Etheta)

            # 向量分解：極座標向量 -> 直角座標向量
            # Ex = Er * cos(theta) - Etheta * sin(theta)
            # Ey = Er * sin(theta) + Etheta * cos(theta)
            Ex = U_Er * np.cos(THETA) - U_Etheta * np.sin(THETA)
            Ey = U_Er * np.sin(THETA) + U_Etheta * np.cos(THETA)

            # 處理 NaN/Inf
            Ex = np.nan_to_num(Ex)
            Ey = np.nan_to_num(Ey)

            # 繪製 Streamplot (白色半透明)
            ax.streamplot(
                X, Y, Ex, Ey, 
                color=(1, 1, 1, 0.5), 
                linewidth=0.8, 
                density=1.2, 
                arrowsize=1.0
            )

        ax.set_aspect('equal')
        ax.set_title(f"Potential Distribution: ${sp.latex(V_expr)}$")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(-rmax, rmax)
        ax.set_ylim(-rmax, rmax)
        
        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        st.error(f"解析或運算錯誤: {e}")
        st.warning("若包含奇異點 (如 1/r 在 r=0)，中心點數值可能趨近無限大。")

# ==========================================
# 4. 主導航邏輯
# ==========================================
st.sidebar.title("⚡ 導航選單")
cat = st.sidebar.selectbox("選擇模組", ["首頁", "函數近似", "電位+電場模擬"])

if cat == "首頁": 
    render_home()
elif cat == "函數近似":
    sub = st.sidebar.radio("方法", ["傅立葉近似", "勒讓德近似"])
    if sub == "傅立葉近似": render_fourier_page()
    else: render_legendre_page()
elif cat == "電位+電場模擬":
    sub = st.sidebar.radio("結構", ["笛卡爾 (Cartesian)", "球座標 (Spherical)", "點電荷"])
    if sub == "笛卡爾 (Cartesian)": render_laplace_cartesian()
    elif sub == "球座標 (Spherical)": render_potential_spherical()
    elif sub == "點電荷": render_potential_point_charge()
