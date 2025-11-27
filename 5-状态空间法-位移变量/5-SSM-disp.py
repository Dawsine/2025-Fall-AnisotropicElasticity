# %% [markdown]
# # 作业6：位移法求解正交各向异性直梁的弯曲

# %% [markdown]
# # 基本方程
# 
# 平面应力问题前提条件：
# 
# 对于厚度较小的直梁，外力平行于梁平面并沿厚度不变，即：
# 
# $\sigma_y = \tau_{xy} = \tau_{yz} = 0$
# 
# 于是三维基本方程均简化：
# 
# + 几何方程：
# 
#     $\varepsilon_x = \frac{\partial u}{\partial x}, \quad \varepsilon_z = \frac{\partial w}{\partial z}, \quad \gamma_{zx} = \frac{\partial w}{\partial x} + \frac{\partial u}{\partial z}$
# 
# + 平衡方程（不计体力）：
# 
#     $\frac{\partial \sigma_x}{\partial x} + \frac{\partial \tau_{zx}}{\partial z} = 0, \quad \frac{\partial \tau_{zx}}{\partial x} + \frac{\partial \sigma_z}{\partial z} = 0$
# 
# + 本构关系（正交各向异性材料）：
# 
#     $\begin{aligned}
#     \sigma_x &= c_{11} \varepsilon_x + c_{13} \varepsilon_z \\
#     \sigma_z &= c_{13} \varepsilon_x + c_{33} \varepsilon_z \\
#     \tau_{zx} &= c_{55} \gamma_{zx}
#     \end{aligned}$
# 
# 其中 $c_{ij} = c_{ij}(x, z)$。
# 
# 

# %% [markdown]
# # 控制方程：
# 
# 从基本方程中推导仅有位移变量的控制方程

# %%
import sympy as sp

# 定义符号变量
x, z = sp.symbols('x z')
u, w = sp.Function('u')(x, z), sp.Function('w')(x, z)
c11, c13, c33, c55 = sp.Function('c11')(x, z), sp.Function('c13')(x, z), sp.Function('c33')(x, z), sp.Function('c55')(x, z)

# %%
print("=== 平面应力问题位移法控制方程推导 ===\n")

# %%
print("1. 几何方程：")
epsilon_x = sp.diff(u, x)
epsilon_z = sp.diff(w, z)
gamma_zx = sp.diff(w, x) + sp.diff(u, z)

print(f"ε_x = ∂u/∂x = {epsilon_x}")
print(f"ε_z = ∂w/∂z = {epsilon_z}")
print(f"γ_zx = ∂w/∂x + ∂u/∂z = {gamma_zx}\n")

# %%
print("2. 本构关系（应力用位移表示）：")
sigma_x = c11 * epsilon_x + c13 * epsilon_z
sigma_z = c13 * epsilon_x + c33 * epsilon_z
tau_zx = c55 * gamma_zx

print(f"σ_x = c11·ε_x + c13·ε_z = {sigma_x}")
print(f"σ_z = c13·ε_x + c33·ε_z = {sigma_z}")
print(f"τ_zx = c55·γ_zx = {tau_zx}\n")

# %%
print("3. 平衡方程：")
print("∂σ_x/∂x + ∂τ_zx/∂z = 0")
print("∂τ_zx/∂x + ∂σ_z/∂z = 0\n")

print("4. 将应力代入平衡方程：")
print("第一个平衡方程：")
eq1 = sp.diff(sigma_x, x) + sp.diff(tau_zx, z)
print(f"∂σ_x/∂x + ∂τ_zx/∂z = {eq1}")


print("\n第二个平衡方程：")
eq2 = sp.diff(tau_zx, x) + sp.diff(sigma_z, z)
print(f"∂τ_zx/∂x + ∂σ_z/∂z = {eq2}\n")

# %%
print("5. 展开并整理控制方程：")
print("第一个控制方程（u方向）：")
# 展开导数
eq1_expanded = sp.expand(eq1)
print(f"展开后: {eq1_expanded}")

# 按位移导数分类整理
u_xx_term = c11 * sp.diff(u, x, x)
u_zz_term = c55 * sp.diff(u, z, z)
u_xz_term = (sp.diff(c11, x) + sp.diff(c55, z)) * sp.diff(u, x) + (c13 + c55) * sp.diff(u, x, z)
w_xz_term = (c13 + c55) * sp.diff(w, x, z) + sp.diff(c13, x) * sp.diff(w, z) + sp.diff(c55, z) * sp.diff(w, x)

eq1_final = u_xx_term + u_zz_term + u_xz_term + w_xz_term
print(f"整理后: {eq1_final} = 0\n")

print("第二个控制方程（w方向）：")
eq2_expanded = sp.expand(eq2)
print(f"展开后: {eq2_expanded}")

# 按位移导数分类整理
w_xx_term = c55 * sp.diff(w, x, x)
w_zz_term = c33 * sp.diff(w, z, z)
w_xz_term = (sp.diff(c55, x) + sp.diff(c33, z)) * sp.diff(w, z) + (c13 + c55) * sp.diff(w, x, z)
u_xz_term = (c13 + c55) * sp.diff(u, x, z) + sp.diff(c55, x) * sp.diff(u, z) + sp.diff(c13, z) * sp.diff(u, x)

eq2_final = w_xx_term + w_zz_term + w_xz_term + u_xz_term
print(f"整理后: {eq2_final} = 0\n")

# %%
print("6. 最终位移法控制方程组：")
print("对于正交各向异性材料，平面应力问题的位移法控制方程为：")
print(f"c11·∂²u/∂x² + c55·∂²u/∂z² + (∂c11/∂x + ∂c55/∂z)·∂u/∂x + (c13 + c55)·∂²u/∂x∂z")
print(f" + (c13 + c55)·∂²w/∂x∂z + ∂c13/∂x·∂w/∂z + ∂c55/∂z·∂w/∂x = 0")
print()
print(f"c55·∂²w/∂x² + c33·∂²w/∂z² + (∂c55/∂x + ∂c33/∂z)·∂w/∂z + (c13 + c55)·∂²w/∂x∂z")
print(f" + (c13 + c55)·∂²u/∂x∂z + ∂c55/∂x·∂u/∂z + ∂c13/∂z·∂u/∂x = 0")

print("\n7. 特殊情况：均匀材料（材料常数与位置无关）")
print("当 c11, c13, c33, c55 为常数时，控制方程简化为：")
print("c11·∂²u/∂x² + c55·∂²u/∂z² + (c13 + c55)·∂²w/∂x∂z = 0")
print("c55·∂²w/∂x² + c33·∂²w/∂z² + (c13 + c55)·∂²u/∂x∂z = 0")


# %%
import sympy as sp
from IPython.display import display, Math, Latex
sp.init_printing()

# 定义符号变量
x, z = sp.symbols('x z')
u, w = sp.Function('u')(x, z), sp.Function('w')(x, z)
c11, c13, c33, c55 = sp.Function('c_{11}')(x, z), sp.Function('c_{13}')(x, z), sp.Function('c_{33}')(x, z), sp.Function('c_{55}')(x, z)

print("=== 平面应力问题位移法控制方程推导 ===\n")

display(Math(r"\text{1.几何方程：}"))
epsilon_x = sp.diff(u, x)
epsilon_z = sp.diff(w, z)
gamma_zx = sp.diff(w, x) + sp.diff(u, z)


display(Math(r"\varepsilon_x = \frac{\partial u}{\partial x} = " + sp.latex(epsilon_x)))
display(Math(r"\varepsilon_z = \frac{\partial w}{\partial z} = " + sp.latex(epsilon_z)))
display(Math(r"\gamma_{zx} = \frac{\partial w}{\partial x} + \frac{\partial u}{\partial z} = " + sp.latex(gamma_zx)))

display(Math(r"\text{2. 本构关系：}"))
sigma_x = c11 * epsilon_x + c13 * epsilon_z
sigma_z = c13 * epsilon_x + c33 * epsilon_z
tau_zx = c55 * gamma_zx

display(Math(r"\text{将几何方程带入本构关系，得到位移表示的应力表达式：}"))
display(Math(r"\sigma_x = c_{11}\varepsilon_x + c_{13}\varepsilon_z = " + sp.latex(sigma_x)))
display(Math(r"\sigma_z = c_{13}\varepsilon_x + c_{33}\varepsilon_z = " + sp.latex(sigma_z)))
display(Math(r"\tau_{zx} = c_{55}\gamma_{zx} = " + sp.latex(tau_zx)))


# %%
display(Math(r"\text{3. 平衡方程：}"))
display(Math(r"\frac{\partial \sigma_x}{\partial x} + \frac{\partial \tau_{zx}}{\partial z} = 0"))
display(Math(r"\frac{\partial \tau_{zx}}{\partial x} + \frac{\partial \sigma_z}{\partial z} = 0"))

display(Math(r"\text{将应力的位移表达代入平衡方程：}"))
eq1 = sp.diff(sigma_x, x) + sp.diff(tau_zx, z)
eq2 = sp.diff(tau_zx, x) + sp.diff(sigma_z, z)
display(Math(r"\text{位移法控制方程：}"))
display(Math(r"\text{方程1: }" + sp.latex(sp.Eq(eq1, 0))))
display(Math(r"\text{方程2: }" + sp.latex(sp.Eq(eq2, 0))))

display(Math(r"\text{展开并整理控制方程：}"))
display(Math(r"\text{第一个控制方程（u方向）：}"))
eq1_expanded = sp.expand(eq1)
display(Math(r"\text{展开后: }" + sp.latex(eq1_expanded)))
u_xx_term = c11 * sp.diff(u, x, x)
u_zz_term = c55 * sp.diff(u, z, z)
u_xz_term = (sp.diff(c11, x) + sp.diff(c55, z)) * sp.diff(u, x) + (c13 + c55) * sp.diff(u, x, z)
w_xz_term = (c13 + c55) * sp.diff(w, x, z) + sp.diff(c13, x) * sp.diff(w, z) + sp.diff(c55, z) * sp.diff(w, x)
eq1_final = u_xx_term + u_zz_term + u_xz_term + w_xz_term
display(Math(r"\text{整理后: }" + sp.latex(sp.Eq(eq1_final, 0))))
display(Math(r"\text{第二个控制方程（w方向）：}"))
eq2_expanded = sp.expand(eq2)
display(Math(r"\text{展开后: }" + sp.latex(eq2_expanded)))
w_xx_term = c55 * sp.diff(w, x, x)
w_zz_term = c33 * sp.diff(w, z, z)
w_xz_term = (sp.diff(c55, x) + sp.diff(c33, z)) * sp.diff(w, z) + (c13 + c55) * sp.diff(w, x, z)
u_xz_term = (c13 + c55) * sp.diff(u, x, z) + sp.diff(c55, x) * sp.diff(u, z) + sp.diff(c13, z) * sp.diff(u, x)
eq2_final = w_xx_term + w_zz_term + w_xz_term + u_xz_term
display(Math(r"\text{整理后: }" + sp.latex(sp.Eq(eq2_final, 0))))
display(Math(r"\text{最终位移法控制方程组：}"))
display(Math(r"\text{对于正交各向异性材料，平面应力问题的位移法控制方程为：}"))
display(Math(r"c_{11}\frac{\partial^2 u}{\partial x^2} + c_{55}\frac{\partial^2 u}{\partial z^2} + \left(\frac{\partial c_{11}}{\partial x} + \frac{\partial c_{55}}{\partial z}\right)\frac{\partial u}{\partial x} + (c_{13} + c_{55})\frac{\partial^2 u}{\partial x \partial z}"))
display(Math(r" + (c_{13} + c_{55})\frac{\partial^2 w}{\partial x \partial z} + \frac{\partial c_{13}}{\partial x}\frac{\partial w}{\partial z} + \frac{\partial c_{55}}{\partial z}\frac{\partial w}{\partial x} = 0"))
display(Math(r"c_{55}\frac{\partial^2 w}{\partial x^2} + c_{33}\frac{\partial^2 w}{\partial z^2} + \left(\frac{\partial c_{55}}{\partial x} + \frac{\partial c_{33}}{\partial z}\right)\frac{\partial w}{\partial z} + (c_{13} + c_{55})\frac{\partial^2 w}{\partial x \partial z}"))
display(Math(r" + (c_{13} + c_{55})\frac{\partial^2 u}{\partial x \partial z} + \frac{\partial c_{55}}{\partial x}\frac{\partial u}{\partial z} + \frac{\partial c_{13}}{\partial z}\frac{\partial u}{\partial x} = 0"))


# %%


display(Math(r"\text{位移法控制方程：}"))
eq1 = sp.diff(sigma_x, x) + sp.diff(tau_zx, z)
eq2 = sp.diff(tau_zx, x) + sp.diff(sigma_z, z)


display(Math(r"\text{方程1: }" + sp.latex(sp.Eq(eq1, 0))))
display(Math(r"\text{方程2: }" + sp.latex(sp.Eq(eq2, 0))))


