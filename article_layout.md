# Iterative Ossification Growth in the Secondary Ossification Center Using CutFEM and Reaction-Diffusion Homogenization

**Cristian Rodrigo Bustamante Porras**

Universidad Nacional de Colombia
Facultad de Ingeniería — Departamento de Ingeniería Mecánica y Mecatrónica
Bogotá, Colombia
2026

---

## 1. Introduction

* Bone development in long bones: endochondral ossification
* Secondary ossification center (SOC) formation and its mechanical regulation
* Computational modeling of bone growth: state of the art
* Objective: coupled mechanical–biochemical iterative framework using CutFEM on unfitted meshes

---

## 2. Method

### 2.1 Biological Model

#### 2.1.1 Domain Structure

* Two-material domain: bone $(E = 500 \text{ MPa}, \nu = 0.2)$ and cartilage $(E = 6 \text{ MPa}, \nu = 0.47)$
* SOC geometry defined by a parametric B-spline boundary
* Material interface at $y = y_{\text{int}}$
* Distributed load applied on the articular surface

---

#### 2.1.2 Carter Stress Approach

Miner’s Index:

$$
\mathrm{MI} = \tau_{\mathrm{oct}} + k_{\mathrm{mi}} \min(\sigma_h,0)
$$

Octahedral shear stress:

$$
\tau_{\mathrm{oct}} = \sqrt{\frac{2}{3}} , \sigma_{\mathrm{vm}}
$$

Hydrostatic stress:

$$
\sigma_h = \frac{1}{3}(\sigma_{xx} + \sigma_{yy} + \sigma_{zz})
$$

Inhibition function:

$$
I(t,s) = (1 - 3t^2 + 2t^3)(1 - 3s^2 + 2s^3)
$$

---

#### 2.1.3 Morphogen Reaction–Diffusion Model

Reaction–diffusion equation in cartilage:

$$

* \nabla^2 u + \alpha u = f \quad \text{in } \Omega_c
  $$

Flux boundary condition:

$$
\nabla u \cdot \mathbf{n} = q \quad \text{on } \Gamma_{\mathrm{oss}}
$$

Zero flux elsewhere:

$$
\nabla u \cdot \mathbf{n} = 0 \quad \text{on } \partial \Omega_c \setminus \Gamma_{\mathrm{oss}}
$$

---

### 2.2 Mathematical Model

#### 2.2.1 Elastic Moving Boundary Problem

Weak form:

$$
\int_{\Omega}
\left(
2\mu , \boldsymbol{\varepsilon}(\mathbf{u}) : \boldsymbol{\varepsilon}(\mathbf{v})

* \lambda (\nabla \cdot \mathbf{u})(\nabla \cdot \mathbf{v})
  \right) d\Omega
*

\int_{\Gamma_D}
\frac{\gamma}{h} \mathbf{u}\cdot\mathbf{v} , d\Gamma
====================================================

\int_{\Gamma_N}
\mathbf{t}\cdot\mathbf{v} , d\Gamma
$$

Strain tensor:

$$
\boldsymbol{\varepsilon}(\mathbf{u}) =
\frac{1}{2} \left( \nabla \mathbf{u} + (\nabla \mathbf{u})^T \right)
$$

Level-set definitions:

$$
\phi_{\mathrm{outer}} = \text{signed distance to SOC boundary}
$$

$$
\phi_{\mathrm{interface}} = y - y_{\text{int}}
$$

$$
\phi_{\mathrm{soc}}^{n+1} = \max(\phi_{\mathrm{soc}}^{n}, \phi_{\mathrm{new}})
$$

---

#### 2.2.2 Iterative Growth Strategy

* Step 0: Mechanical seed
* Step 1: MI-based front expansion
* Step 2: Poisson homogenization
* Step 3: Mechanical solve with updated material
* Step 4: Poisson smoothing
* Repeat until convergence

---

### 2.3 Numerical Method

#### 2.3.1 Cut Finite Element Method

Ghost penalty stabilization:

$$
\sum_{F}
\left(
h^{-1} |[\mathbf{u}]|^2
+
h | [\nabla \mathbf{u} \cdot \mathbf{n}] |^2
\right)
$$

Nitsche penalty:

$$
\gamma = 20 (2\mu_{\text{bone}} + \lambda_{\text{bone}}) \cdot 4
$$

---

#### 2.3.2 Poisson Homogenization Domain

Active domain:

$$
\Omega_c = { \max(\phi_{\mathrm{outer}}, y_{\text{int}} - y) < 0 }
$$

---

## 3. Results

* Evolution of ossification front
* Stress redistribution maps
* Morphogen concentration fields
* Geometry sensitivity (concave vs convex SOC)
* Parameter sensitivity

---

## 4. Discussion

* Biological plausibility of MI–Poisson coupling
* Role of diffusion in front regularization
* Advantages of CutFEM for moving material interfaces
* Limitations: plane strain, simplified constitutive model

---

## 5. Conclusion

* Coupled mechanical–biochemical iterative growth model proposed
* Morphogen transport improves front smoothness
* Future work: full 3D SOC growth, anisotropic cartilage, validation with micro-CT
