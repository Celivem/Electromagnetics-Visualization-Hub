# 電磁學生成小教室 · Electromagnetics Visualization Hub

用 Streamlit 寫的互動式電磁學教具：輸入函數或邊界條件，即時看到級數近似、電位與電場分佈。對應大學電磁學（Griffiths 第 2–3 章）中級數展開與 Laplace 方程的內容。

An interactive Streamlit app for undergraduate electromagnetics: Fourier / Legendre series approximation, and 2D / 3D electric potential and field simulations (finite differences, analytic separation-of-variables solutions, point charges, spherical charge distributions).

![傅立葉級數近似](docs/fourier.png)

## 功能

| 模組 | 內容 |
|---|---|
| 函數近似 | **傅立葉級數**：任意函數在區間 [a, b] 的展開，可拉動項數 N 觀察收斂與 Gibbs 現象 |
| | **勒讓德級數**：[-1, 1] 上的 Legendre 展開 |
| 電位＋電場（2D） | **笛卡爾**：矩形區域 Laplace 方程，數值解（有限差分）與解析解（分離變數後四邊疊加，SymPy 推導公式）兩種模式 |
| | **球座標**：輸入 V(r, θ)，內建點電荷、電偶極、電四極、均勻電場等多極範例 |
| | **點電荷**：多個點電荷的電位熱圖 |
| 電位＋電場（3D） | **笛卡爾**：立方體六面給定電位的 3D 有限差分 |
| | **球座標**：球內電荷密度（均勻、1/r 衰減、p 軌域形或自訂 ρ）積分出 3D 電位與電場 |
| | **點電荷**：自由放置電荷，畫等位面、電場向量與電力線 |

![3D 點電荷模擬](docs/3d_point_charge.png)

函數輸入支援一般數學寫法，例如 `x^2`、`sin(pi x)`、`e^-x`、`square(x)`。

## 執行

```bash
pip install -r requirements.txt
streamlit run app.py
```

也可以直接在 GitHub Codespaces 開啟（已附 `.devcontainer`），或部署到 [Streamlit Community Cloud](https://streamlit.io/cloud)，主程式選 `app.py`。

## 沿革

本 repo 整併了先前分開的三個小工具（傅立葉、勒讓德、Laplace），它們的功能都已包含在 `app.py` 中：

- `Celivem/useless-toolbox`（傅立葉級數）
- `Celivem/Legendre-Series`
- `Celivem/laplace-solving`

舊版 `app V1.py`、`app.V1.2.py` 保留在 git 歷史中。
