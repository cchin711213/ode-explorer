# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="ODE Explorer", layout="wide")
st.title("ODE Explorer — solve, plot, download")

# Sidebar: system selection and basic solver settings
st.sidebar.header("System & solver")
system = st.sidebar.selectbox("Choose system", [
    "Van der Pol (2D)",
    "Lorenz (3D)",
    "Harmonic oscillator (2D)",
    "Lotka-Volterra (2D)",
    "Custom (expressions)"
])
t0 = st.sidebar.number_input("t0", value=0.0)
t1 = st.sidebar.number_input("t1", value=20.0)
n_points = st.sidebar.slider("time samples", 200, 20000, 2000)
method = st.sidebar.selectbox("solve_ivp method", ["RK45", "RK23", "DOP853", "Radau", "BDF"])
atol = float(st.sidebar.text_input("atol", "1e-8"))
rtol = float(st.sidebar.text_input("rtol", "1e-6"))

#### prebuilt systems
if system == "Van der Pol (2D)":
    mu = st.sidebar.slider("mu", 0.0, 10.0, 1.5)
    def f(t, y): return [y[1], mu*(1 - y[0]**2)*y[1] - y[0]]
    y0_text = st.sidebar.text_input("Initial state (comma-separated)", "2.0, 0.0")

elif system == "Lorenz (3D)":
    sigma = st.sidebar.number_input("sigma", value=10.0)
    rho = st.sidebar.number_input("rho", value=28.0)
    beta = st.sidebar.number_input("beta", value=8/3)
    def f(t, y):
        return [sigma*(y[1]-y[0]),
                y[0]*(rho-y[2]) - y[1],
                y[0]*y[1] - beta*y[2]]
    y0_text = st.sidebar.text_input("Initial state (comma-separated)", "1.0, 1.0, 1.0")

elif system == "Harmonic oscillator (2D)":
    omega = st.sidebar.number_input("omega", value=1.0)
    def f(t, y): return [y[1], -omega**2 * y[0]]
    y0_text = st.sidebar.text_input("Initial state (comma-separated)", "1.0, 0.0")

elif system == "Lotka-Volterra (2D)":
    a = st.sidebar.number_input("prey growth a", value=1.0)
    b = st.sidebar.number_input("predation b", value=0.1)
    c = st.sidebar.number_input("predator death c", value=1.5)
    d = st.sidebar.number_input("predator efficiency d", value=0.075)
    def f(t, y): return [a*y[0] - b*y[0]*y[1], -c*y[1] + d*b*y[0]*y[1]]
    y0_text = st.sidebar.text_input("Initial state (comma-separated)", "10.0, 5.0")

else:  # custom
    st.sidebar.write("Custom system: enter expressions for dy_i/dt using `t`, `y` (list), and `np`.")
    n_dim = st.sidebar.slider("dimension", 1, 6, 2)
    exprs = []
    for i in range(n_dim):
        default = "y[1]" if i == 0 and n_dim >= 2 else " - y[0]" if i == 1 and n_dim >= 2 else "0"
        exprs.append(st.sidebar.text_input(f"dy[{i}]/dt", value=default, key=f"expr{i}"))
    y0_text = st.sidebar.text_input("Initial state (comma-separated)", ",".join(["0.0"]*n_dim))

    # build function from expressions (note: eval of expressions runs in restricted locals; don't expose to untrusted users)
    def f(t, y):
        local = {"t": t, "y": y, "np": np}
        # protect builtins
        return [eval(expr, {"__builtins__": {}}, local) for expr in exprs]

# parse initial state
try:
    y0 = np.array([float(s) for s in y0_text.split(",")])
except Exception as e:
    st.error("Invalid initial state. Use comma-separated numbers.")
    st.stop()

# run solver
if len(y0) == 0:
    st.error("Initial state empty.")
    st.stop()

t_eval = np.linspace(t0, t1, n_points)
with st.spinner("Solving ODE..."):
    try:
        sol = solve_ivp(f, (t0, t1), y0, t_eval=t_eval, method=method, atol=atol, rtol=rtol)
    except Exception as e:
        st.error(f"Solver error: {e}")
        st.stop()

if not sol.success:
    st.error("ODE solver failed: " + str(sol.message))
    st.stop()

# prepare dataframe
data = {"t": sol.t}
for i in range(sol.y.shape[0]):
    data[f"y{i}"] = sol.y[i]
df = pd.DataFrame(data)

# main area: plotting and download
st.subheader("Time series")
fig, ax = plt.subplots(figsize=(8,4))
for i in range(sol.y.shape[0]):
    ax.plot(sol.t, sol.y[i], label=f"y{i}")
ax.set_xlabel("t"); ax.set_ylabel("y")
ax.legend(); ax.grid(True)
st.pyplot(fig)

if sol.y.shape[0] == 2:
    st.subheader("Phase portrait (y0 vs y1)")
    fig2, ax2 = plt.subplots(figsize=(5,5))
    ax2.plot(sol.y[0], sol.y[1])
    ax2.set_xlabel("y0"); ax2.set_ylabel("y1"); ax2.grid(True)
    st.pyplot(fig2)
elif sol.y.shape[0] == 3:
    st.subheader("3D trajectory preview")
    try:
        from mpl_toolkits.mplot3d import Axes3D
        fig3 = plt.figure(figsize=(6,5))
        ax3 = fig3.add_subplot(111, projection='3d')
        ax3.plot(sol.y[0], sol.y[1], sol.y[2])
        ax3.set_xlabel("y0"); ax3.set_ylabel("y1"); ax3.set_zlabel("y2")
        st.pyplot(fig3)
    except Exception:
        st.write("3D plotting not available in this environment.")

# data download
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", csv, file_name="ode_solution.csv", mime="text/csv")

# optional: quick stats
st.subheader("Quick stats")
st.write(df.describe()) 
