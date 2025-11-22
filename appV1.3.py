import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

# ==========================================
# 頁面設定
# ==========================================
st.set_page_config(
    page_title="3D 電位分佈模擬器",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 樣式優化
st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 核心物理引擎 (使用快取加速)
# ==========================================
@st.cache_data(show_spinner=False)
def calculate_potential(N, v_top, v_bottom, v_left, v_right, v_front, v_back, max_iter, tolerance):
    """
    使用有限差分法 (Relaxation Method) 求解 3D Laplace 方程式
    
    Args:
        N (int): 網格大小 (N x N x N)
        v_top, v_bottom... (float): 各個面的邊界電位
        max_iter (int): 最大迭代次數
        tolerance (float): 收斂容許誤差
    """
    # 1. 初始化網格 (全零)
    V = np.zeros((N, N, N))
    
    # 2. 設定邊界條件遮罩 (Boundary Mask)
    # 用來確保在迭代過程中，邊界值不會被改變
    mask = np.zeros((N, N, N), dtype=bool)
    
    # 設定各個面的電位與遮罩
    # Z軸 (Top/Bottom)
    V[:, :, -1] = v_top;    mask[:, :, -1] = True
    V[:, :, 0]  = v_bottom; mask[:, :, 0]  = True
    
    # Y軸 (Front/Back)
    V[:, -1, :] = v_back;   mask[:, -1, :] = True
    V[:, 0, :]  = v_front;  mask[:, 0, :]  = True
    
    # X軸 (Right/Left)
    V[-1, :, :] = v_right;  mask[-1, :, :] = True
    V[0, :, :]  = v_left;   mask[0, :, :]  = True

    # 3. 迭代求解 (使用 NumPy 向量化加速)
    # V_new = (V_x+1 + V_x-1 + V_y+1 + V_y-1 + V_z+1 + V_z-1) / 6
    
    for i in range(max_iter):
        V_old = V.copy()
        
        # 核心計算：只更新內部點 (1:-1)
        V[1:-1, 1:-1, 1:-1] = (1/6) * (
            V[2:, 1:-1, 1:-1] + V[:-2, 1:-1, 1:-1] +  # X 方向鄰居
            V[1:-1, 2:, 1:-1] + V[1:-1, :-2, 1:-1] +  # Y 方向鄰居
            V[1:-1, 1:-1, 2:] + V[1:-1, 1:-1, :-2]    # Z 方向鄰居
        )
        
        # 強制重置邊界條件 (雖然上面的切片未觸及邊界，但為求穩健仍加上邏輯或使用mask)
        # 由於上面只更新內部 [1:-1]，邊界其實未被更動，故此處省略顯式重置以節省效能
        
        # 每 200 次檢查一次收斂性 (減少 np.max 的呼叫次數以提升效能)
        if i % 200 == 0:
            diff = np.max(np.abs(V - V_old))
            if diff < tolerance:
                break
    
    # 建立座標網格 (用於 Plotly 繪圖)
    # linspace 產生 0 到 1 之間的座標
    grid_range = np.linspace(0, 1, N)
    X, Y, Z = np.meshgrid(grid_range, grid_range, grid_range, indexing='ij')
    
    return X, Y, Z, V, i  # 回傳座標, 電位矩陣, 實際迭代次數

# ==========================================
# 視覺化邏輯
# ==========================================
def create_3d_figure(X, Y, Z, V, opacity, surface_count, show_caps):
    """建立 Plotly 3D Isosurface 圖表"""
    
    fig = go.Figure(data=go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=V.flatten(),
        isomin=np.min(V),
        isomax=np.max(V),
        surface_count=surface_count, # 等位面層數
        opacity=opacity,             # 透明度
        caps=dict(x_show=show_caps, y_show=show_caps, z_show=show_caps),
        colorscale='RdBu_r',         # 紅藍色階 (紅=高電位)
        colorbar=dict(title='電位 (V)'),
        hoverinfo='all'
    ))

    fig.update_layout(
        title="3D 電位等位面分佈 (Isosurfaces)",
        scene=dict(
            xaxis_title='X 軸',
            yaxis_title='Y 軸',
            zaxis_title='Z 軸',
            aspectmode='cube', # 保持正立方體比例
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5) # 預設視角
            )
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=700,
    )
    return fig

# ==========================================
# 主應用程式介面
# ==========================================
def main():
    # --- 標題區 ---
    st.title("⚡ 3D 靜電場視覺化：笛卡兒座標")
    st.markdown("""
    本應用程式使用 **有限差分法 (Finite Difference Method)** 解算 Laplace 方程式 $\\nabla^2 V = 0$。
    您可以設定立方體六個面的邊界電位，並觀察內部的電位分佈。
    """)

    # --- 側邊欄：參數控制 ---
    with st.sidebar:
        st.header("⚙️ 設定與參數")
        
        st.subheader("1. 網格精細度")
        grid_n = st.slider("網格點數 (N)", 10, 60, 40, help="數值越大越平滑，但計算越慢。建議 30-50。")
        
        st.subheader("2. 邊界電位 (V)")
        with st.expander("設定六面電位", expanded=True):
            col_z = st.columns(2)
            v_top = col_z[0].number_input("頂面 (Z=1)", value=100.0, step=10.0)
            v_bottom = col_z[1].number_input("底面 (Z=0)", value=-100.0, step=10.0)
            
            col_y = st.columns(2)
            v_back = col_y[0].number_input("後面 (Y=1)", value=0.0, step=10.0)
            v_front = col_y[1].number_input("前面 (Y=0)", value=0.0, step=10.0)
            
            col_x = st.columns(2)
            v_right = col_x[0].number_input("右面 (X=1)", value=0.0, step=10.0)
            v_left = col_x[1].number_input("左面 (X=0)", value=0.0, step=10.0)

        st.subheader("3. 求解參數")
        max_iter = st.number_input("最大迭代次數", value=3000, step=500)
        tolerance = st.select_slider("收斂精度", options=[1e-2, 1e-3, 1e-4, 1e-5], value=1e-4)
        
        st.divider()
        st.markdown("### 👁️ 視覺化選項")
        surface_count = st.slider("等位面層數", 3, 20, 10)
        opacity = st.slider("透明度", 0.1, 1.0, 0.3)
        show_caps = st.checkbox("顯示切面封蓋 (Caps)", value=False, help="開啟後等位面會封閉，關閉則像洋蔥圈便於透視")

    # --- 主邏輯執行 ---
    
    # 計算觸發
    with st.spinner(f'正在進行物理運算 (網格: {grid_n}x{grid_n}x{grid_n})...'):
        start_time = time.time()
        X, Y, Z, V, actual_iter = calculate_potential(
            grid_n, v_top, v_bottom, v_left, v_right, v_front, v_back, max_iter, tolerance
        )
        end_time = time.time()

    # --- 結果顯示區 ---
    
    # 1. 統計數據 Metrics
    st.markdown("### 📊 模擬結果統計")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("最高電位", f"{np.max(V):.1f} V")
    col2.metric("最低電位", f"{np.min(V):.1f} V")
    col3.metric("中心點電位", f"{V[grid_n//2, grid_n//2, grid_n//2]:.1f} V")
    col4.metric("計算耗時", f"{end_time - start_time:.3f} s", help=f"實際迭代: {actual_iter} 次")

    # 2. Plotly 3D 圖表
    st.divider()
    fig = create_3d_figure(X, Y, Z, V, opacity, surface_count, show_caps)
    st.plotly_chart(fig, use_container_width=True)
    
    # 3. 物理原理說明
    with st.expander("📚 物理與數學背景"):
        st.markdown(r"""
        #### 1. 拉普拉斯方程式 (Laplace's Equation)
        在無電荷區域 ($\rho=0$)，靜電位 $V$ 滿足：
        $$
        \nabla^2 V = \frac{\partial^2 V}{\partial x^2} + \frac{\partial^2 V}{\partial y^2} + \frac{\partial^2 V}{\partial z^2} = 0
        $$

        #### 2. 數值解法 (Numerical Solution)
        我們將空間離散化為網格點 $(i, j, k)$。根據平均值定理，若網格夠小，任一點的電位約等於其六個相鄰點的平均值：
        $$
        V_{i,j,k} \approx \frac{1}{6} (V_{i+1,j,k} + V_{i-1,j,k} + V_{i,j+1,k} + V_{i,j-1,k} + V_{i,j,k+1} + V_{i,j,k-1})
        $$
        程式透過不斷重複這個平均化過程 (Relaxation)，直到數值不再變動 (收斂)，即可得到最終的電位分佈。
        """)

if __name__ == "__main__":
    main()
