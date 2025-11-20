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

# 初始化 Session State (傅立葉)
if 'fourier_result' not in st.session_state:
    st.session_state['fourier_result'] = None

# 初始化 Session State (點電荷)
if 'point_charges' not in st.session_state:
    # 預設顯示一個電偶極
    st.session_state.point_charges = [
        {'q': 1.0, 'x': -2.0, 'y': 0.0},
        {'q': -1.0, 'x': 2.0, 'y': 0.0}
    ]

# CSS 美化
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1E88E5; text-align: center; margin-bottom: 1rem;}
    .stSlider {padding-top: 20px;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心運算函數
# ==========================================

# --- 傅立葉運算 ---
def calculate_fourier_coefficients(func_str, a, b, max_n):
    def f(x_val):
        allowed_locals = {
            "x": x_val, "np": np, "signal": signal,
            "sin": np.sin, "cos": np.cos, "tan": np.tan,
            "exp": np.exp, "pi": np.pi, "abs": np.abs, 
            "sqrt": np.sqrt, "log": np.log, "sign": np.sign,
            "maximum": np.maximum, "minimum": np.minimum,
            "square": signal.square, "sawtooth": signal.sawtooth,
            "gamma": special.gamma, "sinh": np.sinh, "cosh": np.cosh,
        }
        return eval(func_str, {"__builtins__": None}, allowed_locals)

    L = b - a
    omega = 2 * np.pi / L
    A_coeffs, B_coeffs = [], []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        val_a0, _ = quad(lambda x: f(x), a, b, limit=200)
        A0 = (2.0 / L) * val_a0
    except Exception as e:
        return None, f"積分錯誤: {str(e)}"

    A_coeffs.append(A0); B_coeffs.append(0.0)

    for n in range(1, max_n + 1):
        val_an, _ = quad(lambda x: f(x) * np.cos(n * omega * x), a, b, limit=100)
        an = (2.0 / L) * val_an
        val_bn, _ = quad(lambda x: f(x) * np.sin(n * omega * x), a, b, limit=100)
        bn = (2.0 / L) * val_bn
        A_coeffs.append(an); B_coeffs.append(bn)
        if n % 5 == 0: 
            progress_bar.progress(n / max_n)
            status_text.text(f"計算中: {n}/{max_n}")

    progress_bar.empty(); status_text.empty()
    
    x_vals = np.linspace(a, b, 1000)
    try: y_original = [f(val) for val in x_vals]
    except: y_original = None

    return {
        "A": A_coeffs, "B": B_coeffs, "omega": omega,
        "x_vals": x_vals, "y_original": y_original,
        "range": (a, b)
    }, None

# --- 勒讓德運算 ---
@st.cache_data(show_spinner=False)
def calculate_legendre_coefficients(func_expression, max_n):
    def f(x_val):
        allowed_locals = {
            "x": x_val, "np": np,
            "sin": np.sin, "cos": np.cos, "tan": np.tan,
            "exp": np.exp, "pi": np.pi, "abs": np.abs, 
            "sqrt": np.sqrt, "log": np.log, "sign": np.sign,
            "where": np.where, "heaviside": np.heaviside,
            "maximum": np.maximum, "minimum": np.minimum,
            "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
            "legendre": eval_legendre 
        }
        return eval(func_expression, {"__builtins__": None}, allowed_locals)

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
    except Exception as e:
        return None, None, f"積分過程錯誤: {str(e)}"

# --- 點電荷電位計算 ---
def calculate_point_charge_potential(charges, grid_size=100):
    x = np.linspace(-5, 5, grid_size)
    y = np.linspace(-5, 5, grid_size)
    X, Y = np.meshgrid(x, y)
    V_total = np.zeros_like(X)
    
    if not charges:
        return X, Y, V_total

    for charge in charges:
        q = charge['q']
        x0 = charge['x']
        y0 = charge['y']
        r = np.sqrt((X - x0)**2 + (Y - y0)**2)
        V_total += q / (r + 1e-9) 
        
    return X, Y, V_total

# --- 輔助：電位繪圖 (FDM/Analytic用) ---
def plot_heatmap(data, title, xlabel="x", ylabel="y"):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data, cmap='jet', origin='lower', extent=[0, 1, 0, 1], aspect='auto', interpolation='bilinear')
    plt.colorbar(im, ax=ax).set_label('Potential (V)')
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    return fig

# --- 輔助：智能解析 ---
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
    
    * **函數近似 (Function Approximation)**
        * 利用傅立葉級數與勒讓德多項式來擬合任意波形。
    * **電位模擬 (Potential Simulation)**
        * 求解拉普拉斯方程式，觀察不同邊界條件下的電位分佈。
    * **電場模擬 (Electric Field Simulation)**
        * (開發中) 視覺化電荷與邊界產生的電場向量場。
    
    👈 **請點擊左上角的箭頭打開側邊欄開始！**
    """)

def render_developing(title):
    st.subheader(f"🚧 {title}")
    st.info("此功能目前正在開發中，敬請期待！")

# --- 傅立葉頁面 ---
def render_fourier_page():
    st.subheader("📈 傅立葉級數近似")
    fourier_examples = {
        "自訂輸入": "", "方波 (Square)": "square(x)", "多週期方波": "square(3 * x)",
        "鋸齒波": "sawtooth(x)", "三角波": "sawtooth(x, 0.5)", "全波整流": "abs(sin(x))",
        "半波整流": "maximum(sin(x), 0)", "脈衝波": "square(x, duty=0.2)"
    }

    def update_fourier_input():
        selection = st.session_state.fourier_preset
        if selection != "自訂輸入":
            st.session_state.fourier_input = fourier_examples[selection]

    st.sidebar.markdown("---")
    st.sidebar.selectbox("選擇預設波形", list(fourier_examples.keys()), key='fourier_preset', on_change=update_fourier_input)

    c1, c2, c3, c4 = st.columns(4)
    with c1: func_str = st.text_input("函數 f(x)", value="square(x)", key="fourier_input") 
    with c2: a = st.number_input("起點 a", -3.1415)
    with c3: b = st.number_input("終點 b", 3.1415)
    with c4: max_n = st.number_input("最大項數", 50, step=10)

    if st.button("🚀 計算並繪圖"):
        with st.spinner("運算中..."):
            result, error = calculate_fourier_coefficients(func_str, a, b, max_n)
            if error: st.error(error)
            else: 
                st.session_state['fourier_result'] = result
                st.rerun()

    if st.session_state['fourier_result']:
        res = st.session_state['fourier_result']
        st.divider()
        current_n = st.slider("調整 N 值", 0, len(res["A"])-1, 10)
        
        y_approx = np.full_like(res["x_vals"], res["A"][0]/2)
        for k in range(1, current_n+1):
            y_approx += res["A"][k]*np.cos(k*res["omega"]*res["x_vals"]) + res["B"][k]*np.sin(k*res["omega"]*res["x_vals"])
        
        fig, ax = plt.subplots(figsize=(10, 4))
        if res["y_original"] is not None: ax.plot(res["x_vals"], res["y_original"], 'k-', alpha=0.3, label='Original')
        ax.plot(res["x_vals"], y_approx, 'b-', linewidth=2, label=f'N={current_n}')
        ax.legend(); ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        st.markdown("### 📊 係數表與下載")
        df = pd.DataFrame({"n": range(len(res["A"])), "An": res["A"], "Bn": res["B"]})
        c1, c2 = st.columns(2)
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        c1.download_button("📥 下載圖表", buf, "fourier.png", "image/png")
        c2.download_button("📥 下載係數", df.to_csv(index=False, sep='\t').encode(), "coeffs.csv", "text/csv")
        with st.expander("查看係數表"): st.dataframe(df)

# --- 勒讓德頁面 ---
def render_legendre_page():
    st.subheader("🌊 勒讓德級數近似")
    legendre_examples = {
        "自訂輸入": "", "方波 (Step)": "where(x > 0, 1, 0)", "三角波 (Ramp)": "where(x > 0, x, 0)",
        "絕對值 (V-Shape)": "abs(x)", "多週期方波": "sign(sin(4 * pi * x))",
        "波包 (Wave Packet)": "sin(15 * x) * exp(-5 * x**2)", "全波整流": "abs(sin(3 * pi * x))",
        "AM 調變訊號": "(1 + 0.5 * cos(10 * x)) * cos(50 * x)", "偶極子": "x", "四極子": "3*x**2 - 1"
    }

    def update_legendre_input():
        selection = st.session_state.legendre_preset
        if selection != "自訂輸入":
            st.session_state.legendre_input = legendre_examples[selection]

    st.sidebar.markdown("---")
    st.sidebar.selectbox("選擇波形模版", list(legendre_examples.keys()), key='legendre_preset', on_change=update_legendre_input)

    c1, c2 = st.columns([3, 1])
    with c1: func_str = st.text_input("輸入 f(x)", value="where(x > 0, 1, 0)", key="legendre_input")
    with c2: max_N = st.number_input("最大階數", 20)

    if st.button("🚀 執行運算 (勒讓德)"):
        with st.spinner("積分中..."):
            coeffs, _, error = calculate_legendre_coefficients(func_str, max_N)
            if error: st.error(error)
            else:
                st.session_state['legendre_coeffs'] = coeffs
                st.session_state['legendre_func'] = func_str
                st.rerun()

    if 'legendre_coeffs' in st.session_state:
        coeffs = st.session_state['legendre_coeffs']
        func_expr = st.session_state.get('legendre_func', func_str)
        
        st.divider()
        current_n = st.slider("疊加階數", 0, len(coeffs)-1, len(coeffs)-1)
        
        x = np.linspace(-1, 1, 400)
        try:
            allowed = {"x": x, "np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp, "pi": np.pi, "abs": np.abs, "where": np.where, "sign": np.sign}
            y_target = eval(func_expr, {"__builtins__": None}, allowed)
        except: y_target = np.zeros_like(x)

        y_approx = np.zeros_like(x)
        for n in range(current_n + 1):
            y_approx += coeffs[n] * eval_legendre(n, x)

        fig = plt.figure(figsize=(12, 5))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(x, y_target, 'k--', alpha=0.3, label='Target')
        ax1.plot(x, y_approx, 'r-', label='Approx')
        ax1.set_title("Cartesian View"); ax1.legend()

        ax2 = fig.add_subplot(1, 2, 2, projection='polar')
        theta = np.linspace(0, 2*np.pi, 400)
        r_approx = np.zeros_like(theta)
        for n in range(current_n + 1):
            r_approx += coeffs[n] * eval_legendre(n, np.cos(theta))
        ax2.plot(theta, np.abs(r_approx), 'b-')
        ax2.set_title("Polar View")
        st.pyplot(fig)

        st.markdown("### 📊 係數表與下載")
        df = pd.DataFrame({"n": range(len(coeffs)), "cn": coeffs})
        c1, c2 = st.columns(2)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        c1.download_button("📥 下載圖表", buf, "legendre.png", "image/png")
        c2.download_button("📥 下載係數", df.to_csv(index=False).encode(), "coeffs.csv", "text/csv")
        with st.expander("查看係數表"): st.dataframe(df)

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

# --- 電位模擬 (笛卡爾) ---
def render_laplace_cartesian():
    st.subheader("🔲 電位模擬 - 笛卡爾座標")
    mode = st.radio("模式", ["數值解 (FDM)", "解析解 (SymPy)"], horizontal=True)

    if mode == "數值解 (FDM)":
        c1, c2 = st.columns([1, 3])
        with c1:
            def input_boundary(label, default_val):
                is_inf = st.checkbox(f"{label} 無窮遠", key=f"inf_{label}")
                if not is_inf:
                    val = st.number_input(f"{label} (V)", value=default_val, key=f"v_{label}")
                    return False, val
                return True, 0.0

            top_inf, top_v = input_boundary("上邊界", 10.0)
            bot_inf, bot_v = input_boundary("下邊界", 0.0)
            left_inf, left_v = input_boundary("左邊界", 0.0)
            right_inf, right_v = input_boundary("右邊界", 0.0)
            iters = st.slider("迭代次數", 1000, 5000, 2000)
        with c2:
            if st.button("開始模擬"):
                sz = 40
                pad = sz * 3
                total_h = (pad if top_inf else 0) + sz + (pad if bot_inf else 0)
                total_w = (pad if left_inf else 0) + sz + (pad if right_inf else 0)
                V = np.zeros((total_h, total_w))
                
                r_start = pad if bot_inf else 0
                r_end = r_start + sz
                c_start = pad if left_inf else 0
                c_end = c_start + sz
                
                for _ in range(iters):
                    V_old = V.copy()
                    V[1:-1, 1:-1] = 0.25*(V_old[0:-2, 1:-1] + V_old[2:, 1:-1] + V_old[1:-1, 0:-2] + V_old[1:-1, 2:])
                    if not top_inf: V[r_end-1, c_start:c_end] = top_v
                    else: V[-1, :] = 0
                    if not bot_inf: V[r_start, c_start:c_end] = bot_v
                    else: V[0, :] = 0
                    if not left_inf: V[r_start:r_end, c_start] = left_v
                    else: V[:, 0] = 0
                    if not right_inf: V[r_start:r_end, c_end-1] = right_v
                    else: V[:, -1] = 0
                
                V_view = V[r_start:r_end, c_start:c_end]
                st.pyplot(plot_heatmap(V_view, "FDM Result"))
    
    elif mode == "解析解 (SymPy)":
        st.info("輸入如 `x`, `sin(pi*x)`")
        c1, c2 = st.columns(2)
        top_s = c1.text_input("V(x,1)", "10")
        bot_s = c1.text_input("V(x,0)", "0")
        left_s = c2.text_input("V(0,y)", "0")
        right_s = c2.text_input("V(1,y)", "0")

        if st.button("推導"):
            x, y, n = sp.symbols('x y n'); pi = sp.pi
            terms = []
            def get_term(s, side):
                expr = smart_parse(s)
                if not expr: return None
                An = 2 * sp.integrate(expr.subs(x if side in ['left','right'] else y, x) * sp.sin(n*pi*x), (x,0,1))
                denom = sp.sinh(n*pi)
                if side=='top': return An*sp.sin(n*pi*x)*sp.sinh(n*pi*y)/denom
                if side=='bottom': return An*sp.sin(n*pi*x)*sp.sinh(n*pi*(1-y))/denom
                if side=='left': return An*sp.sin(n*pi*y)*sp.sinh(n*pi*(1-x))/denom
                if side=='right': return An*sp.sin(n*pi*y)*sp.sinh(n*pi*x)/denom
            
            for s, side in [(top_s,'top'), (bot_s,'bottom'), (left_s,'left'), (right_s,'right')]:
                res = get_term(s, side)
                if res: terms.append(res)
            
            if terms:
                V_total = sum(terms)
                st.latex(f"V(x,y) = \\sum_{{n=1}}^{{\\infty}} ({sp.latex(V_total)})")
                
                X, Y = np.meshgrid(np.linspace(0,1,50), np.linspace(0,1,50))
                V_num = np.zeros_like(X)
                f_np = sp.lambdify((n,x,y), V_total, 'numpy')
                prog = st.progress(0)
                for i in range(1, 21):
                    V_num += np.nan_to_num(f_np(i, X, Y))
                    prog.progress(i/20)
                st.pyplot(plot_heatmap(V_num, "Analytical (Top 20)"))

# ==========================================
# 4. 主導航邏輯
# ==========================================
st.sidebar.title("⚡ 導航選單")
category = st.sidebar.selectbox("選擇課程模組", ["首頁", "函數近似", "電位模擬", "電場模擬"])

if category == "首頁":
    render_home()

elif category == "函數近似":
    sub_category = st.sidebar.radio("選擇近似方法", ["傅立葉近似", "勒壤德近似"])
    if sub_category == "傅立葉近似":
        render_fourier_page()
    elif sub_category == "勒壤德近似":
        render_legendre_page()

elif category == "電位模擬":
    sub_category = st.sidebar.radio("選擇座標/結構", ["笛卡爾 (Cartesian)", "球座標", "柱座標", "點電荷"])
    if sub_category == "笛卡爾 (Cartesian)":
        render_laplace_cartesian()
    elif sub_category == "點電荷":
        render_potential_point_charge()
    else:
        render_developing(f"電位模擬 - {sub_category}")

elif category == "電場模擬":
    sub_category = st.sidebar.radio("選擇座標/結構", ["笛卡爾", "球座標", "柱座標", "點電荷"])
    render_developing(f"電場模擬 - {sub_category}")