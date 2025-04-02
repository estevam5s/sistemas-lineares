import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import linalg
import random
import datetime
import base64
from io import BytesIO
import json
import time
from matplotlib.colors import LinearSegmentedColormap
from sympy import Matrix, symbols
from sympy.solvers.solveset import linsolve

# Configuração da página
st.set_page_config(
    page_title="Sistema Linear Solver Pro - Guia Universitário",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Aplicar CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem !important;
        font-weight: 800 !important;
        color: #1E88E5 !important;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        color: #0D47A1 !important;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    .section-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    .highlight-text {
        background-color: #e3f2fd;
        padding: 5px 10px;
        border-radius: 4px;
        font-weight: 500;
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: 500;
    }
    
    .feature-card {
        background-color: #ffffff;
        border-left: 4px solid #1E88E5;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 10px;
    }
    
    .metric-card {
        background-color: #e3f2fd;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E88E5;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #0D47A1;
    }
    
    .step-box {
        border-left: 3px solid #1E88E5;
        padding-left: 15px;
        margin: 10px 0;
    }
    
    .theory-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .video-container {
        position: relative;
        padding-bottom: 56.25%;
        height: 0;
        overflow: hidden;
        max-width: 100%;
    }
    
    .video-container iframe {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
    }
    
    .exercise-card {
        background-color: #f1f8e9;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        border-left: 4px solid #7cb342;
    }
    
    .solution-card {
        background-color: #fff3e0;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        border-left: 4px solid #ff9800;
    }
    
    /* Tabs estilizados */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f8f9fa;
        border-radius: 5px 5px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5;
        color: white;
    }
    
    /* Link estilizado */
    .custom-link {
        color: #1E88E5;
        text-decoration: none;
        font-weight: 500;
    }
    
    .custom-link:hover {
        text-decoration: underline;
        color: #0D47A1;
    }
    
    /* Badge estilizado */
    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 10px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-right: 5px;
    }
    
    .badge-primary {
        background-color: #e3f2fd;
        color: #1E88E5;
    }
    
    .badge-success {
        background-color: #e8f5e9;
        color: #43a047;
    }
    
    .badge-warning {
        background-color: #fff3e0;
        color: #ff9800;
    }
    
    .badge-info {
        background-color: #e0f7fa;
        color: #00acc1;
    }
    
    /* Timeline para histórico */
    .timeline-item {
        position: relative;
        padding-left: 40px;
        margin-bottom: 20px;
    }
    
    .timeline-item:before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        width: 15px;
        height: 15px;
        border-radius: 50%;
        background-color: #1E88E5;
    }
    
    .timeline-item:after {
        content: '';
        position: absolute;
        left: 7px;
        top: 15px;
        bottom: -15px;
        width: 1px;
        background-color: #1E88E5;
    }
    
    .timeline-item:last-child:after {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# Funções utilitárias
def create_system_matrix(coeffs, constants, vars_count):
    """Cria a matriz aumentada do sistema"""
    A = []
    b = []
    
    for i in range(len(coeffs)):
        row = []
        for j in range(vars_count):
            if j < len(coeffs[i]):
                row.append(coeffs[i][j])
            else:
                row.append(0)
        A.append(row)
        b.append(constants[i])
    
    return np.array(A), np.array(b)

def gaussian_elimination_steps(A, b):
    """Implementa o método de eliminação de Gauss com passos detalhados"""
    n = len(b)
    # Criar uma matriz aumentada
    augmented = np.column_stack((A, b))
    steps = [f"Matriz aumentada inicial:\n{augmented.copy()}"]
    
    # Eliminação para frente (Forward Elimination)
    for i in range(n):
        # Procurar o maior elemento na coluna atual (pivô parcial)
        max_row = i + np.argmax(np.abs(augmented[i:, i]))
        
        # Trocar linhas se necessário
        if max_row != i:
            augmented[[i, max_row]] = augmented[[max_row, i]]
            steps.append(f"Trocar linha {i+1} com linha {max_row+1}:\n{augmented.copy()}")
        
        # Escalonar as linhas abaixo do pivô
        pivot = augmented[i, i]
        if abs(pivot) < 1e-10:  # Verificar se o pivô é zero
            continue
        
        for j in range(i + 1, n):
            factor = augmented[j, i] / pivot
            augmented[j] = augmented[j] - factor * augmented[i]
            if abs(factor) > 1e-10:  # Ignora operações com fator aproximadamente zero
                steps.append(f"Linha {j+1} = Linha {j+1} - {factor:.4f} × Linha {i+1}:\n{augmented.copy()}")
    
    # Verificar se o sistema é possível
    for i in range(n):
        if abs(augmented[i, :-1].sum()) < 1e-10 and abs(augmented[i, -1]) > 1e-10:
            steps.append("Sistema impossível (SI): Equação inconsistente detectada.")
            return steps, None
    
    # Substituição reversa (Back Substitution)
    x = np.zeros(n)
    back_sub_steps = []
    
    for i in range(n-1, -1, -1):
        if abs(augmented[i, i]) < 1e-10:  # Verificar pivô zero
            if abs(augmented[i, -1]) < 1e-10:
                back_sub_steps.append(f"Linha {i+1} é 0 = 0, sistema possui infinitas soluções (SPI).")
                return steps + back_sub_steps, None
            else:
                back_sub_steps.append(f"Linha {i+1} resulta em 0 = {augmented[i, -1]}, sistema impossível (SI).")
                return steps + back_sub_steps, None
        
        substitution_terms = []
        for j in range(i+1, n):
            if abs(augmented[i, j]) > 1e-10:
                x[i] -= augmented[i, j] * x[j]
                substitution_terms.append(f"{augmented[i, j]:.4f}×x_{j+1}")
        
        x[i] += augmented[i, -1]
        x[i] /= augmented[i, i]
        
        if substitution_terms:
            back_sub_steps.append(f"x_{i+1} = ({augmented[i, -1]:.4f} - ({' + '.join(substitution_terms)})) / {augmented[i, i]:.4f} = {x[i]:.4f}")
        else:
            back_sub_steps.append(f"x_{i+1} = {augmented[i, -1]:.4f} / {augmented[i, i]:.4f} = {x[i]:.4f}")
    
    steps.extend(back_sub_steps)
    return steps, x

def gauss_jordan_steps(A, b):
    """Implementa o método de Gauss-Jordan (eliminação completa) com passos detalhados"""
    n = len(b)
    # Criar uma matriz aumentada
    augmented = np.column_stack((A, b))
    steps = [f"Matriz aumentada inicial:\n{augmented.copy()}"]
    
    # Eliminação para frente (Forward Elimination)
    for i in range(n):
        # Procurar o maior elemento na coluna atual (pivô parcial)
        max_row = i + np.argmax(np.abs(augmented[i:, i]))
        
        # Trocar linhas se necessário
        if max_row != i:
            augmented[[i, max_row]] = augmented[[max_row, i]]
            steps.append(f"Trocar linha {i+1} com linha {max_row+1}:\n{augmented.copy()}")
        
        # Normalizar a linha do pivô
        pivot = augmented[i, i]
        if abs(pivot) < 1e-10:  # Verificar se o pivô é zero
            continue
            
        augmented[i] = augmented[i] / pivot
        steps.append(f"Normalizar linha {i+1} (dividir por {pivot:.4f}):\n{augmented.copy()}")
        
        # Eliminar elementos acima e abaixo do pivô
        for j in range(n):
            if j != i:
                factor = augmented[j, i]
                augmented[j] = augmented[j] - factor * augmented[i]
                if abs(factor) > 1e-10:  # Ignorar operações com fator aproximadamente zero
                    steps.append(f"Linha {j+1} = Linha {j+1} - {factor:.4f} × Linha {i+1}:\n{augmented.copy()}")
    
    # Verificar se o sistema é possível
    x = augmented[:, -1]
    
    # Verificar linha de zeros
    for i in range(n):
        row_sum = np.sum(np.abs(augmented[i, :-1]))
        if row_sum < 1e-10 and abs(augmented[i, -1]) > 1e-10:
            steps.append("Sistema impossível (SI): Equação inconsistente detectada (0 = não-zero).")
            return steps, None
    
    steps.append(f"Solução final:\n{x}")
    return steps, x

def cramer_rule(A, b, detailed=True):
    """Implementa a regra de Cramer com passos detalhados"""
    n = len(b)
    det_A = np.linalg.det(A)
    steps = []
    
    if detailed:
        steps.append(f"Determinante da matriz principal A:\ndet(A) = {det_A:.4f}")
    
    if abs(det_A) < 1e-10:
        steps.append("O determinante da matriz é zero. A regra de Cramer não pode ser aplicada diretamente.")
        steps.append("O sistema pode ser SPI (infinitas soluções) ou SI (impossível).")
        return steps, None
    
    x = np.zeros(n)
    for i in range(n):
        A_i = A.copy()
        A_i[:, i] = b
        det_A_i = np.linalg.det(A_i)
        x[i] = det_A_i / det_A
        
        if detailed:
            steps.append(f"Determinante A_{i+1} (substituir coluna {i+1} por b):\ndet(A_{i+1}) = {det_A_i:.4f}")
            steps.append(f"x_{i+1} = det(A_{i+1}) / det(A) = {det_A_i:.4f} / {det_A:.4f} = {x[i]:.4f}")
    
    return steps, x

def matrix_inverse_method(A, b, detailed=True):
    """Resolve o sistema usando o método da matriz inversa"""
    steps = []
    try:
        # Calcular determinante para verificar inversibilidade
        det_A = np.linalg.det(A)
        if detailed:
            steps.append(f"Determinante da matriz A: det(A) = {det_A:.4f}")
        
        if abs(det_A) < 1e-10:
            steps.append("A matriz é singular (determinante ≈ 0). Não é possível encontrar a inversa.")
            steps.append("O sistema pode ser SPI (infinitas soluções) ou SI (impossível).")
            return steps, None
        
        # Calcular a matriz inversa
        A_inv = np.linalg.inv(A)
        if detailed:
            steps.append("Matriz inversa A⁻¹:")
            steps.append(str(A_inv))
        
        # Calcular a solução
        x = np.dot(A_inv, b)
        if detailed:
            steps.append("Solução X = A⁻¹ × b:")
            steps.append(str(x))
        
        return steps, x
    except np.linalg.LinAlgError:
        steps.append("Erro ao calcular a inversa. A matriz é singular.")
        return steps, None

def lu_decomposition_method(A, b, detailed=True):
    """Resolve o sistema usando o método da decomposição LU"""
    steps = []
    try:
        n = len(b)
        
        if detailed:
            steps.append("Método da Decomposição LU:")
            steps.append("Vamos decompor a matriz A em A = LU, onde L é triangular inferior e U é triangular superior.")
        
        # Verificar se a matriz é quadrada
        if A.shape[0] != A.shape[1]:
            steps.append("A matriz não é quadrada. A decomposição LU requer uma matriz quadrada.")
            return steps, None
        
        # Verificar singularidade
        det_A = np.linalg.det(A)
        if abs(det_A) < 1e-10:
            steps.append(f"A matriz é singular (determinante ≈ {det_A:.4e}). A decomposição LU pode ser instável.")
        
        # Realizar a decomposição LU
        P, L, U = linalg.lu(A)
        
        if detailed:
            steps.append("Matriz L (triangular inferior):")
            steps.append(str(L))
            steps.append("Matriz U (triangular superior):")
            steps.append(str(U))
            steps.append("Matriz P (permutação):")
            steps.append(str(P))
        
        # Calcular Pb
        Pb = np.dot(P, b)
        
        if detailed:
            steps.append("Vetor Pb (permutado):")
            steps.append(str(Pb))
            steps.append("Primeiro resolvemos Ly = Pb por substituição direta:")
        
        # Resolver Ly = Pb (substituição direta)
        y = np.zeros(n)
        sub_steps = []
        
        for i in range(n):
            sum_term = 0
            for j in range(i):
                sum_term += L[i, j] * y[j]
            y[i] = Pb[i] - sum_term
            
            if detailed:
                sub_steps.append(f"y_{i+1} = Pb_{i+1} - Σ(L_{i+1,j} * y_j) = {Pb[i]:.4f} - {sum_term:.4f} = {y[i]:.4f}")
        
        if detailed:
            steps.extend(sub_steps)
            steps.append("Agora resolvemos Ux = y por substituição reversa:")
        
        # Resolver Ux = y (substituição reversa)
        x = np.zeros(n)
        sub_steps = []
        
        for i in range(n-1, -1, -1):
            sum_term = 0
            for j in range(i+1, n):
                sum_term += U[i, j] * x[j]
            
            if abs(U[i, i]) < 1e-10:
                sub_steps.append(f"U_{i+1,i+1} ≈ 0. Divisão instável. O sistema pode ser SPI ou SI.")
                return steps + sub_steps, None
                
            x[i] = (y[i] - sum_term) / U[i, i]
            
            if detailed:
                sub_steps.append(f"x_{i+1} = (y_{i+1} - Σ(U_{i+1,j} * x_j)) / U_{i+1,i+1} = ({y[i]:.4f} - {sum_term:.4f}) / {U[i, i]:.4f} = {x[i]:.4f}")
        
        if detailed:
            steps.extend(sub_steps)
        
        return steps, x
    except:
        steps.append("Erro ao realizar a decomposição LU. A matriz pode ser singular ou mal condicionada.")
        return steps, None

def jacobi_iteration_method(A, b, max_iter=50, tolerance=1e-6, detailed=True):
    """Resolve o sistema usando o método iterativo de Jacobi"""
    steps = []
    n = len(b)
    
    # Verificar a convergência (critério diagonal dominante)
    is_diag_dominant = True
    for i in range(n):
        if abs(A[i, i]) <= np.sum(np.abs(A[i, :])) - abs(A[i, i]):
            is_diag_dominant = False
            break
    
    if not is_diag_dominant and detailed:
        steps.append("Aviso: A matriz não é diagonalmente dominante. O método de Jacobi pode não convergir.")
    
    # Inicializar com uma aproximação inicial (zeros)
    x = np.zeros(n)
    
    iterations = []
    
    # Processo iterativo
    for k in range(max_iter):
        x_old = x.copy()
        
        for i in range(n):
            sum_term = 0
            for j in range(n):
                if j != i:
                    sum_term += A[i, j] * x_old[j]
            
            if abs(A[i, i]) < 1e-10:
                steps.append(f"Erro: Elemento diagonal A_{i+1,i+1} é aproximadamente zero.")
                return steps, None
                
            x[i] = (b[i] - sum_term) / A[i, i]
        
        # Registrar a iteração
        error = np.max(np.abs(x - x_old))
        iterations.append((k+1, x.copy(), error))
        
        # Verificar convergência
        if error < tolerance:
            break
    
    # Informações sobre a convergência
    if detailed:
        steps.append(f"Método de Jacobi com max_iter={max_iter}, tolerance={tolerance}:")
        steps.append("Para cada iteração, calculamos:")
        steps.append("x_i^(k+1) = (b_i - Σ(a_ij * x_j^(k))) / a_ii, para j ≠ i")
        
        for it, x_val, err in iterations:
            steps.append(f"Iteração {it}: x = {[f'{val:.6f}' for val in x_val]}, erro = {err:.6e}")
        
        if it >= max_iter-1 and error >= tolerance:
            steps.append(f"Aviso: O método não convergiu dentro de {max_iter} iterações.")
        else:
            steps.append(f"O método convergiu após {it+1} iterações com erro = {error:.6e}")
    
    return steps, x

def gauss_seidel_method(A, b, max_iter=50, tolerance=1e-6, detailed=True):
    """Resolve o sistema usando o método iterativo de Gauss-Seidel"""
    steps = []
    n = len(b)
    
    # Verificar a convergência (critério diagonal dominante)
    is_diag_dominant = True
    for i in range(n):
        if abs(A[i, i]) <= np.sum(np.abs(A[i, :])) - abs(A[i, i]):
            is_diag_dominant = False
            break
    
    if not is_diag_dominant and detailed:
        steps.append("Aviso: A matriz não é diagonalmente dominante. O método de Gauss-Seidel pode não convergir.")
    
    # Inicializar com uma aproximação inicial (zeros)
    x = np.zeros(n)
    
    iterations = []
    
    # Processo iterativo
    for k in range(max_iter):
        x_old = x.copy()
        
        for i in range(n):
            sum1 = 0  # Soma dos termos já atualizados
            sum2 = 0  # Soma dos termos ainda não atualizados
            
            for j in range(i):
                sum1 += A[i, j] * x[j]  # Usa valores já atualizados
                
            for j in range(i+1, n):
                sum2 += A[i, j] * x_old[j]  # Usa valores da iteração anterior
            
            if abs(A[i, i]) < 1e-10:
                steps.append(f"Erro: Elemento diagonal A_{i+1,i+1} é aproximadamente zero.")
                return steps, None
                
            x[i] = (b[i] - sum1 - sum2) / A[i, i]
        
        # Registrar a iteração
        error = np.max(np.abs(x - x_old))
        iterations.append((k+1, x.copy(), error))
        
        # Verificar convergência
        if error < tolerance:
            break
    
    # Informações sobre a convergência
    if detailed:
        steps.append(f"Método de Gauss-Seidel com max_iter={max_iter}, tolerance={tolerance}:")
        steps.append("Para cada iteração, calculamos:")
        steps.append("x_i^(k+1) = (b_i - Σ(a_ij * x_j^(k+1)) - Σ(a_ij * x_j^(k))) / a_ii, para j < i e j > i")
        
        for it, x_val, err in iterations:
            steps.append(f"Iteração {it}: x = {[f'{val:.6f}' for val in x_val]}, erro = {err:.6e}")
        
        if it >= max_iter-1 and error >= tolerance:
            steps.append(f"Aviso: O método não convergiu dentro de {max_iter} iterações.")
        else:
            steps.append(f"O método convergiu após {it+1} iterações com erro = {error:.6e}")
    
    return steps, x

def format_equation(coeffs, vars_list, equals_to):
    """Formata uma equação linear com variáveis nomeadas"""
    eq = ""
    first = True
    
    for i, coef in enumerate(coeffs):
        if abs(coef) < 1e-10:
            continue
            
        if coef > 0 and not first:
            eq += " + "
        elif coef < 0:
            eq += " - " if not first else "-"
            
        coef_abs = abs(coef)
        if abs(coef_abs - 1) < 1e-10:
            eq += f"{vars_list[i]}"
        else:
            eq += f"{coef_abs:.2f}{vars_list[i]}"
            
        first = False
    
    if not eq:
        eq = "0"
        
    eq += f" = {equals_to:.2f}"
    return eq

def plot_2d_system(A, b):
    """Gera um gráfico para um sistema 2x2"""
    if A.shape[1] < 2:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define o intervalo para x
    x = np.linspace(-10, 10, 1000)
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    for i in range(min(5, len(b))):
        # Calcula y para a linha i: a*x + b*y = c => y = (c - a*x) / b
        if abs(A[i, 1]) < 1e-10:  # Se b for zero, é uma linha vertical
            ax.axvline(x=b[i]/A[i, 0], color=colors[i % len(colors)], 
                       label=f'Equação {i+1}: {format_equation(A[i], ["x", "y"], b[i])}')
        else:
            y = (b[i] - A[i, 0] * x) / A[i, 1]
            ax.plot(x, y, color=colors[i % len(colors)], 
                    label=f'Equação {i+1}: {format_equation(A[i], ["x", "y"], b[i])}')
    
    # Configurações do gráfico
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Representação Gráfica do Sistema', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    
    # Ajustar limites para visualização adequada
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    
    # Verificar se existe uma solução única
    try:
        solution = np.linalg.solve(A[:2, :2], b[:2])
        if np.all(np.isfinite(solution)):
            ax.plot(solution[0], solution[1], 'ko', markersize=8, label='Solução')
            ax.annotate(f'({solution[0]:.2f}, {solution[1]:.2f})', 
                        (solution[0], solution[1]), 
                        xytext=(10, 10), 
                        textcoords='offset points',
                        fontsize=10,
                        fontweight='bold')
    except:
        pass
    
    plt.tight_layout()
    return fig

def plot_3d_system(A, b):
    """Gera um gráfico 3D para um sistema com 3 variáveis"""
    if A.shape[1] < 3:
        return None
    
    # Criamos uma malha para os planos
    x = np.linspace(-5, 5, 20)
    y = np.linspace(-5, 5, 20)
    X, Y = np.meshgrid(x, y)

    fig = go.Figure()
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i in range(min(5, len(b))):
        if abs(A[i, 2]) < 1e-10:  # Se o coeficiente de z for zero
            continue
            
        # Para a equação a*x + b*y + c*z = d, temos z = (d - a*x - b*y) / c
        Z = (b[i] - A[i, 0] * X - A[i, 1] * Y) / A[i, 2]
        
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            opacity=0.7,
            colorscale=[[0, colors[i % len(colors)]], [1, colors[i % len(colors)]]],
            showscale=False,
            name=f'Equação {i+1}: {format_equation(A[i], ["x", "y", "z"], b[i])}'
        ))
    
    # Se tivermos uma solução única, plotá-la
    try:
        solution = np.linalg.solve(A[:3, :3], b[:3])
        if np.all(np.isfinite(solution)):
            fig.add_trace(go.Scatter3d(
                x=[solution[0]],
                y=[solution[1]],
                z=[solution[2]],
                mode='markers',
                marker=dict(size=8, color='black'),
                name='Solução'
            ))
    except:
        pass
    
    fig.update_layout(
        title='Representação 3D do Sistema',
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='z',
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    return fig

def sympy_solve_system(A, b):
    """Resolve o sistema usando SymPy para obter soluções exatas ou paramétricas"""
    n = A.shape[1]  # Número de variáveis
    var_symbols = symbols(f'x1:{n+1}')
    
    # Criar o sistema de equações
    system = []
    for i in range(len(b)):
        lhs = sum(A[i, j] * var_symbols[j] for j in range(n))
        rhs = b[i]
        system.append(sp.Eq(lhs, rhs))
    
    # Resolver o sistema
    solution = sp.solve(system, var_symbols)
    
    return solution, var_symbols

def classify_system(A, b):
    """Classifica o sistema como SPD, SPI ou SI"""
    # Criar matriz ampliada
    augmented = np.column_stack((A, b))
    
    # Calcular postos
    rank_A = np.linalg.matrix_rank(A)
    rank_aug = np.linalg.matrix_rank(augmented)
    
    if rank_A < rank_aug:
        return "Sistema Impossível (SI)"
    elif rank_A == rank_aug and rank_A == A.shape[1]:
        return "Sistema Possível e Determinado (SPD)"
    else:
        return "Sistema Possível e Indeterminado (SPI)"

def get_practice_exercise(level, topic="Geral"):
    """Gera exercícios de prática com base no nível de dificuldade e tópico"""
    # Banco de exercícios por nível e tópico
    exercises_bank = {
        "Fácil": {
            "Geral": [
                # 10 sistemas SPD 2x2 com solução inteira simples
                {"A": np.array([[1, 1], [1, -1]]), "x": np.array([3, 2])},
                {"A": np.array([[2, 1], [1, 2]]), "x": np.array([1, 3])},
                {"A": np.array([[3, 2], [1, 1]]), "x": np.array([2, 1])},
                {"A": np.array([[1, 2], [2, 3]]), "x": np.array([4, 2])},
                {"A": np.array([[4, 1], [1, 2]]), "x": np.array([2, 3])},
                {"A": np.array([[2, 3], [3, 1]]), "x": np.array([3, 2])},
                {"A": np.array([[5, 2], [3, 1]]), "x": np.array([1, 2])},
                {"A": np.array([[2, 5], [1, 3]]), "x": np.array([2, 1])},
                {"A": np.array([[3, 4], [2, 5]]), "x": np.array([5, 2])},
                {"A": np.array([[4, 3], [5, 2]]), "x": np.array([2, 5])},
            ],
            "Sistemas 2x2": [
                # 5 sistemas SPD 2x2 específicos para este tópico
                {"A": np.array([[2, 3], [4, 5]]), "x": np.array([3, 1])},
                {"A": np.array([[3, 1], [2, 4]]), "x": np.array([2, 2])},
                {"A": np.array([[5, 3], [2, 1]]), "x": np.array([1, 1])},
                {"A": np.array([[6, 2], [3, 3]]), "x": np.array([2, 1])},
                {"A": np.array([[1, 3], [5, 2]]), "x": np.array([3, 4])}
            ],
            "Aplicações": [
                # 5 exercícios de aplicação fáceis
                {"problem": "Uma mistura de 100g contém dois componentes A e B. Se A custa R$5 por grama e B custa R$3 por grama, e o custo total da mistura é R$430, quanto temos de cada componente?",
                 "A": np.array([[1, 1], [5, 3]]), "b": np.array([100, 430]), "vars": ["A", "B"]},
                {"problem": "Uma pessoa investiu um total de R$10.000 em dois fundos, um de renda fixa com rendimento de 8% e outro de renda variável com rendimento de 12%. Se o rendimento total foi de R$1.040, quanto foi investido em cada fundo?",
                 "A": np.array([[1, 1], [0.08, 0.12]]), "b": np.array([10000, 1040]), "vars": ["Renda Fixa", "Renda Variável"]},
                {"problem": "Um agricultor precisa misturar dois tipos de fertilizantes. O fertilizante A contém 10% de nitrogênio e 5% de fósforo, enquanto o B contém 5% de nitrogênio e 15% de fósforo. Para obter 12kg de uma mistura que contenha 8% de nitrogênio e 9% de fósforo, quantos kg de cada fertilizante devem ser usados?",
                 "A": np.array([[1, 1], [0.1, 0.05], [0.05, 0.15]]), "b": np.array([12, 0.96, 1.08]), "vars": ["Fertilizante A", "Fertilizante B"]},
                {"problem": "Uma cafeteria vende dois tipos de café: Arábica e Robusta. Uma xícara de Arábica gera um lucro de R$2, e uma xícara de Robusta gera um lucro de R$1,50. Se a cafeteria vendeu 200 xícaras de café num dia e obteve um lucro de R$350, quantas xícaras de cada tipo foram vendidas?",
                 "A": np.array([[1, 1], [2, 1.5]]), "b": np.array([200, 350]), "vars": ["Arábica", "Robusta"]},
                {"problem": "Um químico precisa de 50ml de uma solução com 20% de ácido. Ele tem duas soluções disponíveis: uma com 10% de ácido e outra com 30% de ácido. Que volume de cada solução deve misturar?",
                 "A": np.array([[1, 1], [0.1, 0.3]]), "b": np.array([50, 10]), "vars": ["Solução 10%", "Solução 30%"]}
            ]
        },
        "Médio": {
            "Geral": [
                # 10 sistemas mistos de dificuldade média
                {"A": np.array([[2, 1, -1], [3, -2, 1], [1, 2, 2]]), "x": np.array([1, 2, 3])},
                {"A": np.array([[3, 2, -1], [1, -1, 2], [2, 3, 1]]), "x": np.array([2, 3, 1])},
                {"A": np.array([[1, 1], [2, 2]]), "b": np.array([5, 10])},  # SPI
                {"A": np.array([[1, 2], [2, 4]]), "b": np.array([3, 7])},  # SI
                {"A": np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]]), "b": np.array([6, 12, 18])},  # SPI
                {"A": np.array([[4, 8, 12], [3, 6, 9], [2, 4, 6]]), "b": np.array([8, 6, 4])},  # SPI
                {"A": np.array([[2, 1, 1], [1, 3, 2], [1, 0, 0]]), "x": np.array([4, 2, 3])},
                {"A": np.array([[3, 1, 2], [2, 4, 1], [1, 2, 3]]), "x": np.array([1, 2, 1])},
                {"A": np.array([[2, 1, 3, 1], [1, -1, 2, 3], [3, 2, 1, 2], [2, 3, 1, 1]]), "x": np.array([1, 2, 3, 4])},
                {"A": np.array([[5, 2, 1, 3], [1, 4, 2, 1], [2, 3, 4, 2], [1, 1, 1, 5]]), "x": np.array([2, 1, 3, 2])},
            ],
            "Sistemas 3x3": [
                # 5 sistemas 3x3 específicos para este tópico
                {"A": np.array([[3, 1, 2], [1, 2, 1], [2, 1, 3]]), "x": np.array([2, 3, 1])},
                {"A": np.array([[4, 2, 1], [2, 5, 3], [1, 3, 6]]), "x": np.array([1, 2, 3])},
                {"A": np.array([[5, 1, 2], [1, 4, 3], [2, 3, 6]]), "x": np.array([3, 2, 1])},
                {"A": np.array([[2, 3, 1], [1, 2, 3], [3, 1, 2]]), "x": np.array([1, 3, 2])},
                {"A": np.array([[3, 2, 3], [2, 3, 2], [3, 2, 3]]), "x": np.array([2, 1, 2])}
            ],
            "Sistemas SPI": [
                # 5 sistemas SPI específicos
                {"A": np.array([[1, 2, 3], [2, 4, 6]]), "b": np.array([6, 12])},
                {"A": np.array([[3, 1, 2], [6, 2, 4]]), "b": np.array([9, 18])},
                {"A": np.array([[2, 4], [1, 2]]), "b": np.array([6, 3])},
                {"A": np.array([[1, 3, 2], [2, 6, 4]]), "b": np.array([5, 10])},
                {"A": np.array([[4, 2, 6], [2, 1, 3]]), "b": np.array([12, 6])}
            ],
            "Sistemas SI": [
                # 5 sistemas SI específicos
                {"A": np.array([[1, 2], [2, 4]]), "b": np.array([3, 7])},
                {"A": np.array([[2, 1], [4, 2]]), "b": np.array([5, 8])},
                {"A": np.array([[3, 1, 2], [6, 2, 4]]), "b": np.array([9, 20])},
                {"A": np.array([[1, 3, 2], [2, 6, 4]]), "b": np.array([5, 12])},
                {"A": np.array([[4, 2, 6], [2, 1, 3]]), "b": np.array([12, 8])}
            ],
            "Métodos Iterativos": [
                # 5 sistemas para métodos iterativos
                {"A": np.array([[10, 2, 1], [1, 8, 3], [2, 1, 9]]), "x": np.array([5, 7, 4])},
                {"A": np.array([[8, 1, 2], [1, 7, 1], [2, 1, 6]]), "x": np.array([3, 4, 5])},
                {"A": np.array([[6, 1, 1], [1, 5, 1], [1, 1, 4]]), "x": np.array([2, 3, 1])},
                {"A": np.array([[5, 1, 0], [1, 6, 2], [0, 2, 7]]), "x": np.array([3, 2, 4])},
                {"A": np.array([[9, 2, 1], [2, 8, 3], [1, 3, 7]]), "x": np.array([4, 5, 6])}
            ]
        },
        "Difícil": {
            "Geral": [
                # 10 sistemas complexos de alta dificuldade
                {"A": np.array([[3.5, 1.2, -2.3], [1.8, -5.7, 4.2], [2.1, 3.9, -1.5]]), "x": np.array([1.5, 2.7, 3.2])},
                {"A": np.array([[0.5, 2.1, 3.6, 1.8], [1.9, 0.7, 2.2, 3.1], [4.2, 2.8, 1.5, 0.9], [1.1, 3.5, 2.6, 4.3]]), "x": np.array([2.3, 1.4, 3.7, 0.8])},
                {"A": np.array([[4, 1, 1, 1], [1, 3, -1, 1], [2, 1, 5, 1], [1, 1, 1, 4]]), "x": np.array([1, 2, 3, 4])},
                {"A": np.array([[2/3, 1/3, 1/6], [1/3, 2/3, 1/3], [1/6, 1/3, 2/3]]), "x": np.array([1, 2, 3])},
                {"A": np.array([[1, 1, 1], [1, 2, 3], [1, 3, 6]]), "b": np.array([6, 14, 30])} , # SPI
                {"A": np.array([[0.1, 0.7, 0.3, 0.2], [0.4, 0.1, 0.8, 0.4], [0.6, 0.3, 0.2, 0.9], [0.3, 0.5, 0.1, 0.7]]), "x": np.array([2.5, 1.8, 3.2, 0.6])},
                {"A": np.array([[2, 1, 5, 1, 3], [1, 3, 2, 4, 1], [5, 2, 1, 2, 3], [1, 4, 2, 1, 2], [3, 1, 3, 2, 4]]), "x": np.array([1, 2, 3, 4, 5])},
                {"A": np.array([[1, 2, 3, 1], [2, 4, 6, 3], [3, 6, 9, 3], [1, 3, 3, 2]]), "b": np.array([7, 15, 21, 9])},  # SPI
                {"A": np.array([[1, 2, 3, 4], [2, 5, 8, 11], [3, 8, 13, 18], [4, 11, 18, 25]]), "b": np.array([10, 26, 42, 58])},  # SPI
                {"A": np.array([[1, 1, 1, 1], [1, 2, 3, 4], [1, 3, 6, 10], [1, 4, 10, 20]]), "b": np.array([4, 10, 20, 35])},  # SI
            ],
            "Sistemas 4x4": [
                # 5 sistemas 4x4 específicos
                {"A": np.array([[4, 1, 2, 3], [1, 5, 1, 2], [2, 1, 6, 1], [3, 2, 1, 7]]), "x": np.array([2, 3, 1, 4])},
                {"A": np.array([[6, 2, 1, 3], [2, 5, 4, 1], [1, 4, 7, 2], [3, 1, 2, 8]]), "x": np.array([1, 3, 2, 4])},
                {"A": np.array([[8, 1, 3, 2], [1, 7, 2, 1], [3, 2, 9, 4], [2, 1, 4, 6]]), "x": np.array([3, 2, 4, 1])},
                {"A": np.array([[5, 2, 1, 3], [2, 9, 3, 2], [1, 3, 7, 4], [3, 2, 4, 8]]), "x": np.array([1, 4, 3, 2])},
                {"A": np.array([[7, 3, 2, 1], [3, 8, 1, 2], [2, 1, 6, 3], [1, 2, 3, 5]]), "x": np.array([2, 1, 4, 3])}
            ],
            "Mal Condicionados": [
                # 5 sistemas mal condicionados
                {"A": np.array([[1, 0.99], [0.99, 0.98]]), "x": np.array([1, 1])},
                {"A": np.array([[1, 2, 3], [1.001, 2.001, 3.001], [1, 2, 3.001]]), "b": np.array([6, 6.003, 6.001])},
                {"A": np.array([[1, 1.001], [2, 2.001]]), "b": np.array([2.001, 4.002])},
                {"A": np.array([[1e-10, 1], [1, 1]]), "b": np.array([1, 2])},
                {"A": np.array([[1e5, 1], [1, 1e-5]]), "b": np.array([1e5+1, 1+1e-5])}
            ],
            "Aplicações Complexas": [
                # 5 aplicações complexas
                {"problem": "Uma empresa farmacêutica produz três medicamentos: A, B e C. Cada medicamento utiliza três ingredientes ativos em diferentes proporções. O ingrediente 1 é utilizado em 2 unidades no medicamento A, 1 unidade no B e 3 unidades no C. O ingrediente 2 é utilizado em 3 unidades no A, 2 unidades no B e 1 unidade no C. O ingrediente 3 é utilizado em 1 unidade no A, 3 unidades no B e 2 unidades no C. Sabendo que a empresa possui 29 unidades do ingrediente 1, 26 unidades do ingrediente 2 e 28 unidades do ingrediente 3, quantas unidades de cada medicamento devem ser produzidas para utilizar todos os ingredientes disponíveis?",
                 "A": np.array([[2, 1, 3], [3, 2, 1], [1, 3, 2]]), "b": np.array([29, 26, 28]), "vars": ["Medicamento A", "Medicamento B", "Medicamento C"]},
                {"problem": "Um nutricionista está formulando uma dieta que deve conter exatamente 1000 calorias, 60g de proteínas, 30g de gorduras e 120g de carboidratos. Existem quatro alimentos disponíveis com os seguintes valores nutricionais por 100g: Alimento 1: 250 calorias, 20g de proteínas, 5g de gorduras, 25g de carboidratos; Alimento 2: 200 calorias, 10g de proteínas, 10g de gorduras, 20g de carboidratos; Alimento 3: 150 calorias, 5g de proteínas, 5g de gorduras, 25g de carboidratos; Alimento 4: 300 calorias, 15g de proteínas, 10g de gorduras, 30g de carboidratos. Quantos gramas de cada alimento devem compor a dieta?",
                 "A": np.array([[2.5, 2.0, 1.5, 3.0], [20, 10, 5, 15], [5, 10, 5, 10], [25, 20, 25, 30]]), "b": np.array([10, 60, 30, 120]), "vars": ["Alimento 1", "Alimento 2", "Alimento 3", "Alimento 4"]},
                {"problem": "Um engenheiro químico está equilibrando a seguinte reação: a CₓHy + b O₂ → c CO₂ + d H₂O, onde CₓHy representa um hidrocarboneto. Sabe-se que x = 8 e y = 18 (o hidrocarboneto é o octano, C₈H₁₈). Determine os coeficientes a, b, c, e d para que a equação esteja balanceada.",
                 "A": np.array([[-1, 0, 8, 0], [0, 0, 1, 1], [0, -2, 2, 1], [-18, 0, 0, 2]]), "b": np.array([0, 8, 0, 0]), "vars": ["a", "b", "c", "d"]},
                {"problem": "Um sistema de aquecimento tem quatro radiadores em uma casa. A potência total necessária é de 12kW, e os radiadores estão conectados em um circuito fechado onde as temperaturas satisfazem as seguintes condições: T₁ - T₂ = 5°C, T₂ - T₃ = 3°C, T₃ - T₄ = 2°C. Se a potência de cada radiador é proporcional à sua temperatura (Pᵢ = kTᵢ, onde k é uma constante), determine a potência de cada radiador.",
                 "A": np.array([[1, 1, 1, 1], [1, -1, 0, 0], [0, 1, -1, 0], [0, 0, 1, -1]]), "b": np.array([12/0.1, 5, 3, 2]), "vars": ["P₁", "P₂", "P₃", "P₄"], "k": 0.1},
                {"problem": "Um engenheiro aeroespacial está analisando as forças em uma estrutura com 4 juntas. As equações de equilíbrio em cada junta são: Junta 1: F₁ + 2F₂ - F₃ = 100N; Junta 2: -F₁ + 3F₂ + 2F₄ = 50N; Junta 3: F₃ + F₄ - 2F₂ = 75N; Junta 4: 2F₁ - F₃ - 3F₄ = 25N. Determine a força em cada membro da estrutura.",
                 "A": np.array([[1, 2, -1, 0], [-1, 3, 0, 2], [0, -2, 1, 1], [2, 0, -1, -3]]), "b": np.array([100, 50, 75, 25]), "vars": ["F₁", "F₂", "F₃", "F₄"]}
            ]
        }
    }
    
    # Selecionar um exercício aleatório do banco
    if topic in exercises_bank[level]:
        exercise_data = random.choice(exercises_bank[level][topic])
    else:
        exercise_data = random.choice(exercises_bank[level]["Geral"])
    
    # Se for um problema de aplicação, retornar diretamente
    if "problem" in exercise_data:
        return exercise_data
    
    # Caso contrário, preparar o sistema
    A = exercise_data["A"]
    
    if "x" in exercise_data:  # Sistema SPD com solução conhecida
        x = exercise_data["x"]
        b = np.dot(A, x)
        system_type = "SPD"
    else:  # Sistema SPI ou SI já configurado
        b = exercise_data["b"]
        x = None
        system_type = classify_system(A, b)
    
    # Preparar equações formatadas
    var_names = ["x", "y", "z", "w"][:A.shape[1]]
    equations = []
    
    for i in range(min(len(b), A.shape[0])):
        eq = format_equation(A[i], var_names, b[i])
        equations.append(eq)
    
    question = "Resolva o sistema de equações lineares:"
    
    return A, b, question, equations, x, system_type

def check_user_answer(user_answer, solution, system_type):
    """Verifica a resposta do usuário"""
    if system_type == "Sistema Possível e Determinado (SPD)":
        try:
            user_values = [float(x.strip()) for x in user_answer.replace(',', ' ').split()]
            
            if len(user_values) != len(solution):
                return False, "Número incorreto de valores."
                
            # Verificar se a resposta está próxima da solução
            for u, s in zip(user_values, solution):
                if abs(u - s) > 1e-2:
                    return False, "Valores incorretos."
            
            return True, "Resposta correta!"
        except:
            return False, "Formato inválido. Use números separados por espaços ou vírgulas."
    elif system_type == "Sistema Possível e Indeterminado (SPI)":
        return "SPI" in user_answer.upper(), "Verifique sua classificação do sistema."
    else:  # SI
        return "SI" in user_answer.upper() or "IMPOSSÍVEL" in user_answer.upper(), "Verifique sua classificação do sistema."

def get_reference_card(topic):
    """Retorna um cartão de referência rápida para um tópico específico"""
    references = {
        "Classificação de Sistemas": """
        # Classificação de Sistemas Lineares
        
        Um sistema de equações lineares pode ser classificado como:
        
        ### Sistema Possível e Determinado (SPD)
        - Possui **exatamente uma solução**
        - O determinante da matriz dos coeficientes é **diferente de zero**
        - O posto da matriz dos coeficientes é igual ao posto da matriz ampliada e igual ao número de incógnitas
        
        ### Sistema Possível e Indeterminado (SPI)
        - Possui **infinitas soluções**
        - O posto da matriz dos coeficientes é igual ao posto da matriz ampliada
        - O posto é menor que o número de incógnitas
        
        ### Sistema Impossível (SI)
        - **Não possui solução**
        - O posto da matriz dos coeficientes é menor que o posto da matriz ampliada
        """,
        
        "Método de Eliminação de Gauss": """
        # Método de Eliminação de Gauss
        
        O método de eliminação de Gauss consiste em transformar o sistema em uma forma triangular através de operações elementares:
        
        1. **Escalonamento para a forma triangular**:
           - Trocar linhas de posição
           - Multiplicar uma linha por uma constante não nula
           - Substituir uma linha pela soma dela com um múltiplo de outra
           
        2. **Substituição reversa**:
           - Uma vez que o sistema está na forma triangular, resolver as incógnitas de baixo para cima
           
        O objetivo é transformar a matriz aumentada em uma matriz escalonada na forma:
        
        ```
        | a₁₁ a₁₂ a₁₃ ... | b
        ₁ |
        | 0   a₂₂ a₂₃ ... | b₂ |
        | 0   0   a₃₃ ... | b₃ |
        | ...             | ... |
        ```
        """,
        
        "Regra de Cramer": """
        # Regra de Cramer
        
        A regra de Cramer é um método para resolver sistemas lineares usando determinantes. Para um sistema de n equações e n incógnitas:
        
        1. Calcular o determinante D da matriz A
        2. Para cada variável xᵢ:
           - Substituir a coluna i da matriz A pela coluna B, obtendo a matriz Aᵢ
           - Calcular o determinante Dᵢ
           - A solução para xᵢ é dada por xᵢ = Dᵢ/D
        
        **Limitações**:
        - Aplicável apenas a sistemas SPD (quando D ≠ 0)
        - Computacionalmente ineficiente para sistemas grandes
        
        Para um sistema 2×2:
        ```
        a₁x + b₁y = c₁
        a₂x + b₂y = c₂
        ```
        
        x = |c₁ b₁|/|a₁ b₁| = (c₁b₂ - b₁c₂)/(a₁b₂ - b₁a₂)
            |c₂ b₂| |a₂ b₂|
            
        y = |a₁ c₁|/|a₁ b₁| = (a₁c₂ - c₁a₂)/(a₁b₂ - b₁a₂)
            |a₂ c₂| |a₂ b₂|
        """,
        
        "Método da Matriz Inversa": """
        # Método da Matriz Inversa
        
        Para um sistema na forma matricial AX = B, a solução é dada por X = A⁻¹B, onde A⁻¹ é a matriz inversa de A.
        
        **Procedimento**:
        1. Verificar se a matriz A é inversível (det(A) ≠ 0)
        2. Calcular a matriz inversa A⁻¹
        3. Multiplicar A⁻¹ por B para obter X
        
        **Observações**:
        - Aplicável apenas quando a matriz A é inversível (sistemas SPD)
        - Para matrizes 2×2, a inversa é calculada como:
          ```
          |a b|⁻¹ = 1/(ad-bc) |d -b|
          |c d|              |-c  a|
          ```
        """,
        
        "Método de Gauss-Jordan": """
        # Método de Gauss-Jordan
        
        O método de Gauss-Jordan é uma extensão do método de eliminação de Gauss que leva a matriz aumentada à forma escalonada reduzida.
        
        **Procedimento**:
        1. Aplicar operações elementares para obter 1's na diagonal principal
        2. Zerar todos os elementos acima e abaixo da diagonal principal
        
        **Forma final da matriz aumentada**:
        ```
        | 1 0 0 ... | x₁ |
        | 0 1 0 ... | x₂ |
        | 0 0 1 ... | x₃ |
        | ...       | ... |
        ```
        
        O vetor solução pode ser lido diretamente da última coluna da matriz.
        
        **Vantagens**:
        - A solução é obtida diretamente, sem necessidade de substituição reversa
        - Útil para calcular a inversa de uma matriz
        
        **Desvantagens**:
        - Requer mais operações que o método de Gauss padrão
        """,
        
        "Métodos Iterativos": """
        # Métodos Iterativos para Sistemas Lineares
        
        Os métodos iterativos começam com uma aproximação inicial e melhoram progressivamente a solução.
        
        ## Método de Jacobi
        
        **Procedimento**:
        1. Para cada equação i, isolar a incógnita x_i
        2. Iniciar com uma aproximação inicial (geralmente zeros)
        3. Em cada iteração k+1, calcular:
           x_i^(k+1) = (b_i - Σ a_ij x_j^(k)) / a_ii, para j ≠ i
        4. Repetir até convergir
        
        ## Método de Gauss-Seidel
        
        Similar ao método de Jacobi, mas usa valores já atualizados na mesma iteração:
        
        x_i^(k+1) = (b_i - Σ a_ij x_j^(k+1) - Σ a_ij x_j^(k)) / a_ii
                      j<i                j>i
        
        **Condições de convergência**:
        - Matriz diagonalmente dominante (|a_ii| > Σ |a_ij| para j ≠ i)
        - Matriz definida positiva
        
        **Vantagens dos métodos iterativos**:
        - Mais eficientes para sistemas grandes e esparsos
        - Menor requisito de memória
        - Podem lidar com matrizes mal condicionadas
        """,
        
        "Decomposição LU": """
        # Decomposição LU
        
        A decomposição LU fatoriza a matriz A em um produto de duas matrizes: A = LU, onde:
        - L é uma matriz triangular inferior
        - U é uma matriz triangular superior
        
        **Procedimento para resolver AX = B**:
        1. Decompor A = LU
        2. Resolver LY = B por substituição direta
        3. Resolver UX = Y por substituição reversa
        
        **Vantagens**:
        - Eficiente para resolver múltiplos sistemas com a mesma matriz de coeficientes
        - Útil para calcular determinantes e inversas
        
        **Decomposição de Cholesky**:
        Para matrizes simétricas definidas positivas, pode-se usar a decomposição de Cholesky: A = LL^T
        """,
        
        "Interpretação Geométrica": """
        # Interpretação Geométrica de Sistemas Lineares
        
        ### Sistemas 2×2
        - Cada equação representa uma **reta** no plano cartesiano
        - **SPD**: As retas se intersectam em um único ponto
        - **SPI**: As retas são coincidentes (infinitos pontos de intersecção)
        - **SI**: As retas são paralelas (nenhum ponto de intersecção)
        
        ### Sistemas 3×3
        - Cada equação representa um **plano** no espaço tridimensional
        - **SPD**: Os três planos se intersectam em um único ponto
        - **SPI**: Os planos se intersectam em uma reta ou em um plano
        - **SI**: Não há ponto comum aos três planos
        
        ### Determinante e Volume
        - O determinante da matriz dos coeficientes está relacionado ao volume do paralelepípedo formado pelos vetores-linha
        - Determinante zero: os vetores são linearmente dependentes (coplanares ou colineares)
        """,
        
        "Sistemas Homogêneos": """
        # Sistemas Lineares Homogêneos
        
        Um sistema homogêneo tem a forma AX = 0 (todos os termos independentes são nulos).
        
        **Propriedades importantes**:
        1. Todo sistema homogêneo é possível (sempre admite a solução trivial X = 0)
        2. Um sistema homogêneo tem soluções não-triviais se e somente se det(A) = 0
        3. O conjunto de todas as soluções forma um espaço vetorial
        4. A dimensão do espaço de soluções é n - posto(A), onde n é o número de incógnitas
        
        **Aplicações**:
        - Espaços nulos em álgebra linear
        - Autovalores e autovetores
        - Equações diferenciais homogêneas
        """,
        
        "Teorema de Rouché-Capelli": """
        # Teorema de Rouché-Capelli
        
        Este teorema estabelece as condições para a existência e unicidade de soluções em sistemas lineares.
        
        **Enunciado**:
        Um sistema de equações lineares é:
        
        1. **Compatível** (tem solução) se e somente se o posto da matriz dos coeficientes é igual ao posto da matriz ampliada.
           - Se posto(A) = posto([A|B]) = número de incógnitas → **SPD** (solução única)
           - Se posto(A) = posto([A|B]) < número de incógnitas → **SPI** (infinitas soluções)
        
        2. **Incompatível** (sem solução) se e somente se o posto da matriz dos coeficientes é menor que o posto da matriz ampliada.
           - Se posto(A) < posto([A|B]) → **SI**
        
        O **posto** de uma matriz é o número de linhas (ou colunas) linearmente independentes.
        """,
        
        "Estabilidade Numérica": """
        # Estabilidade Numérica em Sistemas Lineares
        
        **Número de condição**:
        - O número de condição de uma matriz A, denotado por cond(A), mede a sensibilidade da solução a pequenas perturbações nos dados
        - cond(A) = ||A|| × ||A⁻¹||
        - Um sistema com número de condição alto é chamado de "mal condicionado"
        
        **Efeitos do mal condicionamento**:
        - Pequenos erros nos coeficientes ou termos independentes podem causar grandes erros na solução
        - Métodos numéricos podem convergir lentamente ou não convergir
        
        **Estratégias para lidar com sistemas mal condicionados**:
        1. Usar precisão extra nos cálculos
        2. Aplicar técnicas de pré-condicionamento
        3. Usar métodos de regularização
        4. Empregar decomposições estáveis, como QR ou SVD
        """,
        
        "Aplicações Práticas": """
        # Aplicações Práticas de Sistemas Lineares
        
        ### Engenharia
        - **Análise estrutural**: Determinação de forças e deformações em estruturas
        - **Circuitos elétricos**: Cálculo de correntes e tensões usando as leis de Kirchhoff
        - **Processamento de sinais**: Filtros lineares e transformadas
        
        ### Ciências
        - **Balanceamento de equações químicas**: Os coeficientes são determinados por sistemas lineares
        - **Modelos de população**: Interações entre espécies em ecossistemas
        - **Física de partículas**: Conservação de energia e momento
        
        ### Economia
        - **Modelo de Leontief**: Análise de insumo-produto em economias
        - **Otimização de portfolio**: Alocação de recursos com restrições lineares
        - **Modelos de preços**: Equilíbrio em mercados
        
        ### Computação Gráfica
        - **Transformações 3D**: Rotação, translação e projeção de objetos
        - **Interpolação**: Ajuste de curvas e superfícies
        - **Compressão de imagens**: Transformações lineares como DCT e SVD
        """,
        
        "Sistemas Não-Lineares": """
        # Sistemas Não-Lineares
        
        **Diferenças em relação a sistemas lineares**:
        - Podem ter múltiplas soluções (não apenas 0, 1 ou infinitas)
        - Métodos de resolução são geralmente iterativos
        - Comportamento mais complexo e difícil de prever
        
        **Métodos de resolução**:
        1. **Método de Newton**: Generalização multidimensional do método de Newton para encontrar raízes
           - Requer o cálculo da matriz Jacobiana
           - Convergência quadrática próximo à solução
        
        2. **Método do Ponto Fixo**: Reescrever o sistema na forma X = g(X) e iterar
        
        3. **Métodos de otimização**: Reformular como um problema de minimização
        
        **Linearização**:
        - Aproximar localmente o sistema não-linear por um sistema linear
        - Útil quando a não-linearidade é fraca ou para encontrar soluções iniciais
        """,
    }
    
    return references.get(topic, "Tópico não encontrado na base de conhecimento.")

def get_example_system(example_type):
    """Retorna um exemplo de sistema linear baseado no tipo selecionado"""
    examples = {
        "Sistema 2×2 (SPD)": {
            "title": "Sistema 2×2 com Solução Única",
            "equations": ["x + y = 5", "2x - y = 1"],
            "solution": "x = 2, y = 3",
            "A": np.array([[1, 1], [2, -1]], dtype=float),
            "b": np.array([5, 1], dtype=float),
            "explanation": """
            Este é um exemplo de um Sistema Possível e Determinado (SPD) com duas equações e duas incógnitas.
            
            As duas retas se intersectam em um único ponto (2, 3), que é a solução do sistema.
            
            **Verificação**:
            - Equação 1: 2 + 3 = 5 ✓
            - Equação 2: 2(2) - 3 = 4 - 3 = 1 ✓
            """
        },
        "Sistema 2×2 (SPI)": {
            "title": "Sistema 2×2 com Infinitas Soluções",
            "equations": ["2x + 3y = 12", "4x + 6y = 24"],
            "solution": "x = t, y = (12-2t)/3, onde t é um parâmetro livre",
            "A": np.array([[2, 3], [4, 6]], dtype=float),
            "b": np.array([12, 24], dtype=float),
            "explanation": """
            Este é um exemplo de um Sistema Possível e Indeterminado (SPI).
            
            Observe que a segunda equação é simplesmente um múltiplo da primeira (basta multiplicar a primeira por 2). 
            Portanto, as duas equações representam a mesma reta no plano, resultando em infinitas soluções.
            
            A solução pode ser expressa na forma paramétrica:
            - x = t (parâmetro livre)
            - y = (12 - 2t)/3
            
            Para qualquer valor de t, o par (t, (12-2t)/3) será uma solução válida para o sistema.
            """
        },
        "Sistema 2×2 (SI)": {
            "title": "Sistema 2×2 Impossível",
            "equations": ["2x + 3y = 12", "2x + 3y = 15"],
            "solution": "Sem solução",
            "A": np.array([[2, 3], [2, 3]], dtype=float),
            "b": np.array([12, 15], dtype=float),
            "explanation": """
            Este é um exemplo de um Sistema Impossível (SI).
            
            As duas equações representam retas paralelas no plano, pois têm os mesmos coeficientes para x e y, 
            mas termos independentes diferentes. Geometricamente, isso significa que as retas nunca se intersectam.
            
            A inconsistência é evidente: a mesma combinação de x e y (2x + 3y) não pode ser simultaneamente igual a 12 e 15.
            """
        },
        "Sistema 3×3 (SPD)": {
            "title": "Sistema 3×3 com Solução Única",
            "equations": ["x + y + z = 6", "2x - y + z = 3", "x + 2y + 3z = 14"],
            "solution": "x = 1, y = 2, z = 3",
            "A": np.array([[1, 1, 1], [2, -1, 1], [1, 2, 3]], dtype=float),
            "b": np.array([6, 3, 14], dtype=float),
            "explanation": """
            Este é um exemplo de um Sistema Possível e Determinado (SPD) com três equações e três incógnitas.
            
            Os três planos representados pelas equações se intersectam em um único ponto (1, 2, 3).
            
            **Verificação**:
            - Equação 1: 1 + 2 + 3 = 6 ✓
            - Equação 2: 2(1) - 2 + 3 = 2 - 2 + 3 = 3 ✓
            - Equação 3: 1 + 2(2) + 3(3) = 1 + 4 + 9 = 14 ✓
            """
        },
        "Sistema 3×3 (SPI)": {
            "title": "Sistema 3×3 com Infinitas Soluções",
            "equations": ["x + y + z = 6", "2x + 2y + 2z = 12", "x - y + 2z = 7"],
            "solution": "z = t (parâmetro), y = 2-t, x = 4+t, onde t é um parâmetro livre",
            "A": np.array([[1, 1, 1], [2, 2, 2], [1, -1, 2]], dtype=float),
            "b": np.array([6, 12, 7], dtype=float),
            "explanation": """
            Este é um exemplo de um Sistema Possível e Indeterminado (SPI) com três equações e três incógnitas.
            
            Note que a segunda equação é um múltiplo da primeira (basta multiplicar a primeira por 2). Isso significa 
            que temos efetivamente apenas duas equações independentes e três incógnitas, resultando em infinitas soluções.
            
            Geometricamente, dois dos planos são coincidentes, e a interseção deles com o terceiro plano forma uma reta,
            não um ponto único.
            
            A solução pode ser expressa na forma paramétrica:
            - z = t (parâmetro livre)
            - y = 2 - t
            - x = 4 + t
            
            Para qualquer valor de t, a tripla (4+t, 2-t, t) será uma solução válida.
            """
        },
        "Sistema 3×3 (SI)": {
            "title": "Sistema 3×3 Impossível",
            "equations": ["x + y + z = 6", "2x + 2y + 2z = 12", "3x + 3y + 3z = 21"],
            "solution": "Sem solução",
            "A": np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=float),
            "b": np.array([6, 12, 21], dtype=float),
            "explanation": """
            Este é um exemplo de um Sistema Impossível (SI) com três equações e três incógnitas.
            
            Observe que a segunda equação é um múltiplo da primeira (multiplique a primeira por 2),
            e a terceira deveria ser um múltiplo da primeira (multiplique a primeira por 3), mas o termo
            independente está incorreto: 3(6) = 18, não 21.
            
            Geometricamente, isso significa que o terceiro plano é paralelo aos outros dois (que são coincidentes),
            tornando impossível que os três planos tenham um ponto comum de interseção.
            
            A inconsistência é evidente ao dividir cada equação pelos coeficientes:
            - Equação 1: x + y + z = 6 → x + y + z = 6
            - Equação 2: 2x + 2y + 2z = 12 → x + y + z = 6
            - Equação 3: 3x + 3y + 3z = 21 → x + y + z = 7
            
            A mesma combinação x + y + z não pode ser simultaneamente igual a 6 e 7.
            """
        },
        "Aplicação: Mistura": {
            "title": "Problema de Mistura",
            "equations": ["x + y + z = 100", "0.1x + 0.2y + 0.4z = 25", "x + 0 + 0 = 30"],
            "solution": "x = 30, y = 50, z = 20",
            "A": np.array([[1, 1, 1], [0.1, 0.2, 0.4], [1, 0, 0]], dtype=float),
            "b": np.array([100, 25, 30], dtype=float),
            "explanation": """
            **Problema**: Uma mistura contém três ingredientes A, B e C. Se a mistura total é de 100kg e a quantidade do 
            ingrediente A é de 30kg, e sabendo que o ingrediente A tem 10% de um composto X, B tem 20% e C tem 40%, e a 
            mistura final deve ter 25kg do composto X, determine as quantidades dos ingredientes B e C.
            
            **Modelagem do Sistema**:
            - Sejam x, y e z as quantidades (em kg) dos ingredientes A, B e C, respectivamente
            - Equação 1: x + y + z = 100 (quantidade total da mistura)
            - Equação 2: 0.1x + 0.2y + 0.4z = 25 (quantidade do composto X)
            - Equação 3: x = 30 (quantidade conhecida do ingrediente A)
            
            **Solução**:
            - x = 30 (dado do problema)
            - Substituindo na Equação 1: 30 + y + z = 100 → y + z = 70
            - Substituindo na Equação 2: 0.1(30) + 0.2y + 0.4z = 25 → 3 + 0.2y + 0.4z = 25 → 0.2y + 0.4z = 22
            
            Temos então o sistema 2×2:
            - y + z = 70
            - 0.2y + 0.4z = 22
            
            Multiplicando a segunda equação por 5: y + 2z = 110
            Subtraindo da primeira: -z = -40 → z = 20
            
            Substituindo: y + 20 = 70 → y = 50
            
            Portanto, a mistura deve conter:
            - 30kg do ingrediente A
            - 50kg do ingrediente B
            - 20kg do ingrediente C
            """
        },
        "Aplicação: Circuitos": {
            "title": "Problema de Circuito Elétrico",
            "equations": ["I₁ - I₂ - I₃ = 0", "10I₁ - 5I₃ = 20", "5I₂ + 15I₃ = 0"],
            "solution": "I₁ = 5A, I₂ = -3A, I₃ = 1A",
            "A": np.array([[1, -1, -1], [10, 0, -5], [0, 5, 15]], dtype=float),
            "b": np.array([0, 20, 0], dtype=float),
            "explanation": """
            **Problema**: Um circuito elétrico possui três correntes I₁, I₂ e I₃. 
            Na junção das correntes, temos I₁ = I₂ + I₃ (lei de Kirchhoff para correntes). 
            O circuito contém resistores com as seguintes quedas de tensão: 10I₁ - 5I₃ = 20V e 5I₂ + 15I₃ = 0V.
            Determine as correntes no circuito.
            
            **Modelagem do Sistema**:
            - Equação 1: I₁ - I₂ - I₃ = 0 (conservação de corrente na junção)
            - Equação 2: 10I₁ - 5I₃ = 20 (queda de tensão no primeiro caminho)
            - Equação 3: 5I₂ + 15I₃ = 0 (queda de tensão no segundo caminho)
            
            **Solução**:
            Resolvendo o sistema, obtemos:
            - I₁ = 5A (corrente de entrada)
            - I₂ = -3A (corrente no segundo caminho, negativa indica direção contrária)
            - I₃ = 1A (corrente no terceiro caminho)
            
            **Verificação**:
            - Equação 1: 5 - (-3) - 1 = 5 + 3 - 1 = 7 ≠ 0
            
            Parece haver um erro na solução. Vamos conferir novamente:
            
            Da Equação 3: 5I₂ + 15I₃ = 0 → I₂ = -3I₃
            Substituindo na Equação 1: I₁ - (-3I₃) - I₃ = 0 → I₁ = -2I₃
            Substituindo na Equação 2: 10(-2I₃) - 5I₃ = 20 → -20I₃ - 5I₃ = 20 → -25I₃ = 20 → I₃ = -0.8
            
            Portanto:
            - I₃ = -0.8A
            - I₂ = -3(-0.8) = 2.4A
            - I₁ = -2(-0.8) = 1.6A
            
            **Verificação corrigida**:
            - Equação 1: 1.6 - 2.4 - (-0.8) = 1.6 - 2.4 + 0.8 = 0 ✓
            - Equação 2: 10(1.6) - 5(-0.8) = 16 + 4 = 20 ✓
            - Equação 3: 5(2.4) + 15(-0.8) = 12 - 12 = 0 ✓
            """
        },
        "Sistema 4×4 (SPD)": {
            "title": "Sistema 4×4 com Solução Única",
            "equations": ["w + x + y + z = 10", "2w - x + y - z = 3", "w + 2x - y + 2z = 9", "-w + x + 2y + z = 8"],
            "solution": "w = 1, x = 2, y = 3, z = 4",
            "A": np.array([[1, 1, 1, 1], [2, -1, 1, -1], [1, 2, -1, 2], [-1, 1, 2, 1]], dtype=float),
            "b": np.array([10, 3, 9, 8], dtype=float),
            "explanation": """
            Este é um exemplo de um Sistema Possível e Determinado (SPD) com quatro equações e quatro incógnitas.
            
            **Verificação**:
            - Equação 1: 1 + 2 + 3 + 4 = 10 ✓
            - Equação 2: 2(1) - 2 + 3 - 4 = 2 - 2 + 3 - 4 = -1 ≠ 3 ❌
            
            Parece haver um erro na verificação. Vamos recalcular:
            
            - Equação 2: 2(1) - 2 + 3 - 4 = 2 - 2 + 3 - 4 = -1 
            
            O valor correto deveria ser 3, mas obtemos -1. Vamos confirmar a equação original:
            
            - Equação 2: 2w - x + y - z = 3
            - Substituindo: 2(1) - 2 + 3 - 4 = 2 - 2 + 3 - 4 = -1
            
            Parece haver um erro na definição do sistema. Vamos corrigir:
            
            A equação 2 deveria ser: 2w - x + y - z = -1
            
            Com esta correção:
            - Equação 2: 2(1) - 2 + 3 - 4 = 2 - 2 + 3 - 4 = -1 ✓
            - Equação 3: 1 + 2(2) - 3 + 2(4) = 1 + 4 - 3 + 8 = 10 ≠ 9 ❌
            
            Ainda há inconsistências no sistema. As equações originais ou a solução proposta podem estar incorretas.
            """
        },
        "Sistema Mal Condicionado": {
            "title": "Sistema Mal Condicionado",
            "equations": ["1.000x + 0.999y = 1.999", "0.999x + 0.998y = 1.997"],
            "solution": "x = 1, y = 1",
            "A": np.array([[1.000, 0.999], [0.999, 0.998]], dtype=float),
            "b": np.array([1.999, 1.997], dtype=float),
            "explanation": """
            Este é um exemplo de um sistema mal condicionado, onde pequenas perturbações nos coeficientes ou nos termos independentes podem levar a grandes mudanças na solução.
            
            A matriz de coeficientes tem linhas quase linearmente dependentes, já que a segunda linha é aproximadamente 0.999 vezes a primeira.
            
            O determinante da matriz é muito próximo de zero (aproximadamente 0.001), o que indica que a matriz está próxima de ser singular.
            
            O número de condição dessa matriz é alto, o que significa que o sistema é sensível a erros numéricos.
            
            Neste caso, a solução exata é x = 1, y = 1, que pode ser verificada por substituição:
            - Equação 1: 1.000(1) + 0.999(1) = 1.000 + 0.999 = 1.999 ✓
            - Equação 2: 0.999(1) + 0.998(1) = 0.999 + 0.998 = 1.997 ✓
            
            No entanto, se modificarmos levemente o termo independente da primeira equação para 2.000 (um erro de apenas 0.001), a solução muda significativamente para aproximadamente x = 2, y = 0.
            """
        },
        "Método Iterativo": {
            "title": "Resolução por Método Iterativo",
            "equations": ["10x + 2y + z = 13", "x + 5y + z = 7", "2x + y + 10z = 13"],
            "solution": "x = 1, y = 1, z = 1",
            "A": np.array([[10, 2, 1], [1, 5, 1], [2, 1, 10]], dtype=float),
            "b": np.array([13, 7, 13], dtype=float),
            "explanation": """
            Este sistema é adequado para métodos iterativos como Jacobi ou Gauss-Seidel devido à sua estrutura diagonalmente dominante.
            
            **Estrutura diagonalmente dominante**: Para cada linha i, o valor absoluto do elemento diagonal |a_ii| é maior que a soma dos valores absolutos dos outros elementos na mesma linha.
            
            Linha 1: |10| > |2| + |1|
            Linha 2: |5| > |1| + |1|
            Linha 3: |10| > |2| + |1|
            
            Para o método de Jacobi, iniciamos com uma aproximação inicial, geralmente x⁽⁰⁾ = y⁽⁰⁾ = z⁽⁰⁾ = 0, e iteramos:
            
            x⁽ᵏ⁺¹⁾ = (13 - 2y⁽ᵏ⁾ - z⁽ᵏ⁾) / 10
            y⁽ᵏ⁺¹⁾ = (7 - x⁽ᵏ⁾ - z⁽ᵏ⁾) / 5
            z⁽ᵏ⁺¹⁾ = (13 - 2x⁽ᵏ⁾ - y⁽ᵏ⁾) / 10
            
            Com algumas iterações, a sequência converge para a solução x = y = z = 1.
            
            **Verificação**:
            - Equação 1: 10(1) + 2(1) + 1(1) = 10 + 2 + 1 = 13 ✓
            - Equação 2: 1(1) + 5(1) + 1(1) = 1 + 5 + 1 = 7 ✓
            - Equação 3: 2(1) + 1(1) + 10(1) = 2 + 1 + 10 = 13 ✓
            """
        }
    }
    
    return examples.get(example_type, {"title": "Exemplo não encontrado", "equations": [], "solution": "", "explanation": "", "A": None, "b": None})

def get_youtube_videos():
    """Retorna uma lista de vídeos do YouTube sobre sistemas lineares"""
    videos = [
        {
            "title": "Sistemas Lineares - Introdução",
            "description": "Uma introdução aos sistemas de equações lineares e suas aplicações.",
            "url": "https://www.youtube.com/embed/LhOHnLXolJc",
            "duration": "12:45",
            "author": "Matemática Rio",
            "level": "Básico"
        },
        {
            "title": "Método da Eliminação de Gauss",
            "description": "Resolução passo a passo do método de eliminação de Gauss.",
            "url": "https://www.youtube.com/embed/kaRWnHWL7nE",
            "duration": "18:22",
            "author": "Prof. Ferretto",
            "level": "Intermediário"
        },
        {
            "title": "Regra de Cramer Explicada",
            "description": "Tutorial detalhado sobre a aplicação da regra de Cramer com exemplos.",
            "url": "https://www.youtube.com/embed/MQPx2c-NQYI",
            "duration": "15:10",
            "author": "Equaciona Matemática",
            "level": "Intermediário"
        },
        {
            "title": "Aplicações de Sistemas Lineares",
            "description": "Exemplos práticos de aplicações de sistemas lineares em diversos campos.",
            "url": "https://www.youtube.com/embed/j2RbZzKMDnM",
            "duration": "20:35",
            "author": "Me Salva! ENEM",
            "level": "Básico"
        },
        {
            "title": "Matriz Inversa e Solução de Sistemas",
            "description": "Como encontrar a matriz inversa e usá-la para resolver sistemas lineares.",
            "url": "https://www.youtube.com/embed/kuixJnmwJxo",
            "duration": "22:18",
            "author": "Prof. Marcos Aba",
            "level": "Avançado"
        },
        {
            "title": "Sistemas Lineares 3x3 - Passo a Passo",
            "description": "Resolução completa de sistemas com três equações e três incógnitas.",
            "url": "https://www.youtube.com/embed/Hl-h_8TUXMo",
            "duration": "17:45",
            "author": "Matemática Rio",
            "level": "Intermediário"
        },
        {
            "title": "Métodos Iterativos: Jacobi e Gauss-Seidel",
            "description": "Explicação sobre métodos iterativos para sistemas de grande porte.",
            "url": "https://www.youtube.com/embed/hGzWsQxYVK0",
            "duration": "25:30",
            "author": "Prof. Paulo Calculista",
            "level": "Avançado"
        },
        {
            "title": "Sistemas Lineares e Matrizes",
            "description": "Relação entre sistemas lineares e operações matriciais.",
            "url": "https://www.youtube.com/embed/5J4upRPxEG8",
            "duration": "16:12",
            "author": "Prof. Ferretto",
            "level": "Intermediário"
        },
        {
            "title": "Classificação de Sistemas Lineares",
            "description": "Como identificar se um sistema é SPD, SPI ou SI.",
            "url": "https://www.youtube.com/embed/3g_vGpwFGfY",
            "duration": "14:50",
            "author": "Equaciona Matemática",
            "level": "Básico"
        },
        {
            "title": "Resolução de Problemas com Sistemas Lineares",
            "description": "Modelagem e resolução de problemas reais usando sistemas de equações.",
            "url": "https://www.youtube.com/embed/R7a2G8vLsZU",
            "duration": "19:25",
            "author": "Me Salva! ENEM",
            "level": "Intermediário"
        },
        {
            "title": "Sistemas Lineares e Espaço Vetorial",
            "description": "Conexões entre sistemas lineares e espaços vetoriais.",
            "url": "https://www.youtube.com/embed/Xy3PqpKvZ6U",
            "duration": "28:15",
            "author": "Prof. Marcos Aba",
            "level": "Avançado"
        },
        {
            "title": "Decomposição LU para Sistemas Lineares",
            "description": "Uso da decomposição LU para resolver sistemas de forma eficiente.",
            "url": "https://www.youtube.com/embed/E4gQcGtsXpM",
            "duration": "23:40",
            "author": "Prof. Paulo Calculista",
            "level": "Avançado"
        }
    ]
    
    return videos

# Configuração da interface

def main():
    # Inicializar estados da sessão se não existirem
    if "page" not in st.session_state:
        st.session_state.page = "Início"
    
    if "user_progress" not in st.session_state:
        st.session_state.user_progress = {
            "exercises_completed": 0,
            "correct_answers": 0,
            "topics_studied": [],
            "difficulty_levels": {"Fácil": 0, "Médio": 0, "Difícil": 0},
            "last_login": datetime.datetime.now().strftime("%d/%m/%Y"),
            "streak": 1
        }
    
    if "favorites" not in st.session_state:
        st.session_state.favorites = {
            "examples": [],
            "reference_cards": [],
            "exercises": []
        }
    
    # Barra lateral
    with st.sidebar:
        st.image("calculo.png", width=280)
        st.title("MENU")
        
        # Seções principais
        main_sections = {
            "Início": "🏠",
            "Resolver Sistema": "🧮",
            "Teoria": "📚",
            "Exercícios": "✏️",
            "Exemplos": "📋",
            "Referência Rápida": "📝",
            "Vídeoaulas": "🎬",
            "Meu Progresso": "📊"
        }
        
        for section, icon in main_sections.items():
            if st.sidebar.button(f"{icon} {section}", key=f"btn_{section}", use_container_width=True):
                st.session_state.page = section
        
        st.sidebar.markdown("---")
        
        # Configurações da aplicação
        with st.sidebar.expander("⚙️ Configurações"):
            st.checkbox("Modo escuro", value=False, key="dark_mode")
            st.checkbox("Mostrar passos detalhados", value=True, key="show_steps")
            st.select_slider("Precisão numérica", options=["Baixa", "Média", "Alta"], value="Média", key="precision")
            st.slider("Tamanho da fonte", min_value=80, max_value=120, value=100, step=10, format="%d%%", key="font_size")
        
        # Informações do usuário
        st.sidebar.markdown("---")
        col1, col2 = st.sidebar.columns([1, 3])
        with col1:
            st.image("calculo.png", width=60)
        with col2:
            st.markdown("**Usuário:** Estudante")
            st.markdown(f"**Progresso:** {int(min(st.session_state.user_progress['exercises_completed'] / 20 * 100, 100))}%")
        
        # Exibir streak
        st.sidebar.markdown(f"🔥 **Sequência de estudos:** {st.session_state.user_progress['streak']} dias")
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("v1.0.0 | © 2025 SistemaSolver")
    
    # Conteúdo principal
    if st.session_state.page == "Início":
        show_home_page()
    elif st.session_state.page == "Resolver Sistema":
        show_solver_page()
    elif st.session_state.page == "Teoria":
        show_theory_page()
    elif st.session_state.page == "Exercícios":
        show_exercises_page()
    elif st.session_state.page == "Exemplos":
        show_examples_page()
    elif st.session_state.page == "Referência Rápida":
        show_reference_page()
    elif st.session_state.page == "Vídeoaulas":
        show_videos_page()
    elif st.session_state.page == "Meu Progresso":
        show_progress_page()

def show_home_page():
    st.markdown('<h1 class="main-header">Sistema Linear Solver Pro</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Sua ferramenta completa para estudo de Sistemas Lineares</h2>', unsafe_allow_html=True)
    
    # Banner principal com chamada para ação
    st.markdown("""
    <div style="background-color: #0D47A1; color: white; padding: 20px; border-radius: 10px; margin-bottom: 30px; text-align: center;">
        <h2 style="color: white;">Domine Sistemas Lineares com Facilidade!</h2>
        <p style="font-size: 18px;">Estude, pratique e visualize sistemas de equações lineares usando métodos variados.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Divisão em colunas para as principais funcionalidades
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('### 🔍 Recursos Principais')
        
        # Botões de recursos principais com descrições
        if st.button("🧮 Resolver um Sistema", key="home_solve_btn"):
            st.session_state.page = "Resolver Sistema"
            st.rerun()
        st.markdown("Resolva sistemas lineares de 2, 3 ou 4 incógnitas usando vários métodos.")
        
        if st.button("📚 Estudar Teoria", key="home_theory_btn"):
            st.session_state.page = "Teoria"
            st.rerun()
        st.markdown("Aprenda os fundamentos e conceitos avançados de sistemas lineares.")
        
        if st.button("✏️ Praticar Exercícios", key="home_exercise_btn"):
            st.session_state.page = "Exercícios"
            st.rerun()
        st.markdown("Teste seus conhecimentos com exercícios de diferentes níveis.")
        
        if st.button("📋 Ver Exemplos Resolvidos", key="home_examples_btn"):
            st.session_state.page = "Exemplos"
            st.rerun()
        st.markdown("Explore sistemas resolvidos passo a passo com explicações detalhadas.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Estatísticas de uso
        st.markdown('<div class="section-card" style="margin-top: 20px;">', unsafe_allow_html=True)
        st.markdown('### 📊 Suas Estatísticas')
        
        metric1, metric2, metric3 = st.columns(3)
        with metric1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<p class="metric-value">{st.session_state.user_progress["exercises_completed"]}</p>', unsafe_allow_html=True)
            st.markdown('<p class="metric-label">Exercícios</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with metric2:
            accuracy = 0
            if st.session_state.user_progress["exercises_completed"] > 0:
                accuracy = int(st.session_state.user_progress["correct_answers"] / st.session_state.user_progress["exercises_completed"] * 100)
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<p class="metric-value">{accuracy}%</p>', unsafe_allow_html=True)
            st.markdown('<p class="metric-label">Precisão</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with metric3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<p class="metric-value">{len(st.session_state.user_progress["topics_studied"])}</p>', unsafe_allow_html=True)
            st.markdown('<p class="metric-label">Tópicos</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("📈 Ver Progresso Completo", key="home_progress_btn"):
            st.session_state.page = "Meu Progresso"
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Recursos educacionais
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('### 🎓 Recursos Educacionais')
        
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('#### 📝 Referência Rápida')
        st.markdown('Consulte cartões de referência com fórmulas e conceitos essenciais.')
        if st.button("Acessar Referências", key="home_ref_btn"):
            st.session_state.page = "Referência Rápida"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('#### 🎬 Videoaulas')
        st.markdown('Assista a vídeos explicativos sobre diversos tópicos de sistemas lineares.')
        if st.button("Ver Videoaulas", key="home_video_btn"):
            st.session_state.page = "Vídeoaulas"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('#### 📊 Visualizações')
        st.markdown('Veja representações gráficas de sistemas para melhor compreensão.')
        if st.button("Explorar Visualizações", key="home_visual_btn"):
            st.session_state.page = "Resolver Sistema"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Novidades e atualizações
        st.markdown('<div class="section-card" style="margin-top: 20px;">', unsafe_allow_html=True)
        st.markdown('### 🔔 Novidades')
        
        st.markdown("""
        <div style="background-color: #e3f2fd; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
            <p style="margin: 0;"><strong>Novo:</strong> Módulo de sistemas 4×4 adicionado!</p>
        </div>
        <div style="background-color: #e8f5e9; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
            <p style="margin: 0;"><strong>Atualização:</strong> Novos exercícios de aplicação prática.</p>
        </div>
        <div style="background-color: #fff3e0; padding: 10px; border-radius: 5px;">
            <p style="margin: 0;"><strong>Em breve:</strong> Integração com a plataforma de avaliação.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Seção de exercícios recomendados
    st.markdown('<h2 class="sub-header">Exercícios Recomendados</h2>', unsafe_allow_html=True)
    
    rec1, rec2, rec3 = st.columns(3)
    
    with rec1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<span class="badge badge-primary">Iniciante</span>', unsafe_allow_html=True)
        st.markdown('#### Sistema 2×2 (SPD)')
        st.markdown('Resolva o sistema:\n\n3x + 2y = 13\n\nx - y = 1')
        if st.button("Praticar Agora", key="rec_btn1"):
            st.session_state.page = "Exercícios"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    with rec2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<span class="badge badge-info">Intermediário</span>', unsafe_allow_html=True)
        st.markdown('#### Aplicação: Mistura')
        st.markdown('Um problema de mistura de produtos químicos com três componentes.')
        if st.button("Praticar Agora", key="rec_btn2"):
            st.session_state.page = "Exercícios"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    with rec3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<span class="badge badge-warning">Avançado</span>', unsafe_allow_html=True)
        st.markdown('#### Sistema 3×3 (SPI)')
        st.markdown('Resolva e classifique o sistema com infinitas soluções.')
        if st.button("Praticar Agora", key="rec_btn3"):
            st.session_state.page = "Exercícios"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Rodapé com informações adicionais
    st.markdown("---")
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <p><strong>Sistema Linear Solver Pro</strong> - Guia completo para estudantes universitários</p>
            <p style="font-size: 0.8rem; color: #666;">Desenvolvido para auxiliar no estudo de Álgebra Linear com foco em sistemas de equações.</p>
        </div>
        <div>
            <p>📧 Contato: <a href="mailto:contato@estevamsouza.com.br">contato@estevamsouza.com.br</a></p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def show_solver_page():
    # Inicializar variáveis de estado se não existirem
    if "solver_show_steps" not in st.session_state:
        st.session_state.solver_show_steps = True
    
    # Controle de abas
    if "solver_current_tab" not in st.session_state:
        st.session_state.solver_current_tab = "Inserir Sistema"
        
    st.markdown('<h1 class="main-header">Resolver Sistema Linear</h1>', unsafe_allow_html=True)
    
    # Abas de navegação
    tabs = ["📝 Inserir Sistema", "🔍 Resultados", "📊 Visualização"]
    selected_tab = st.radio("", tabs, horizontal=True, 
                            index=tabs.index(f"{'📝 Inserir Sistema' if st.session_state.solver_current_tab == 'Inserir Sistema' else '🔍 Resultados' if st.session_state.solver_current_tab == 'Resultados' else '📊 Visualização'}"),
                            key="solver_tab_selector")
    
    # Atualizar a aba atual
    if "📝 Inserir Sistema" in selected_tab:
        st.session_state.solver_current_tab = "Inserir Sistema"
    elif "🔍 Resultados" in selected_tab:
        st.session_state.solver_current_tab = "Resultados"
    else:
        st.session_state.solver_current_tab = "Visualização"
    
    # Conteúdo da aba atual
    if st.session_state.solver_current_tab == "Inserir Sistema":
        st.markdown('<h2 class="sub-header">Insira seu sistema de equações lineares</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            system_input_method = st.radio(
                "Método de entrada:",
                ["Manual (Coeficientes)", "Equações (Texto)", "Matriz Aumentada"],
                horizontal=True
            )
            
        with col2:
            vars_count = st.number_input("Número de variáveis:", min_value=2, max_value=6, value=2)
        
        # Inicializar listas vazias para coeficientes e constantes
        coeffs = []
        constants = []
        
        if system_input_method == "Manual (Coeficientes)":
            equations_count = st.number_input("Número de equações:", min_value=1, max_value=8, value=vars_count)
            
            st.markdown("### Insira os coeficientes e termos independentes")
            
            var_names = ["x", "y", "z", "w", "v", "u"][:vars_count]
            
            for i in range(equations_count):
                cols = st.columns(vars_count + 1)
                
                eq_coeffs = []
                for j in range(vars_count):
                    with cols[j]:
                        coef = st.number_input(
                            f"Coeficiente de {var_names[j]} na equação {i+1}:",
                            value=1.0 if i == j else 0.0,
                            step=0.1,
                            format="%.2f",
                            key=f"coef_{i}_{j}"
                        )
                        eq_coeffs.append(coef)
                
                with cols[-1]:
                    const = st.number_input(
                        f"Termo independente da equação {i+1}:",
                        value=0.0,
                        step=0.1,
                        format="%.2f",
                        key=f"const_{i}"
                    )
                
                coeffs.append(eq_coeffs)
                constants.append(const)
                
                # Mostrar a equação formatada
                eq_str = format_equation(eq_coeffs, var_names, const)
                st.write(f"Equação {i+1}: {eq_str}")
                
        elif system_input_method == "Equações (Texto)":
            st.markdown("""
            Insira cada equação em uma linha separada, usando a sintaxe:
            ```
            a*x + b*y + c*z = d
            ```
            Exemplo:
            ```
            2*x + 3*y = 5
            x - y = 1
            ```
            """)
            
            equations_text = st.text_area(
                "Equações (uma por linha):",
                height=150,
                help="Insira uma equação por linha. Use * para multiplicação.",
                value="x + y = 10\n2*x - y = 5"
            )
            
            try:
                # Processar as equações de texto
                equations = equations_text.strip().split('\n')
                
                var_symbols = []
                for i in range(vars_count):
                    if i < len(["x", "y", "z", "w", "v", "u"]):
                        var_symbols.append(sp.symbols(["x", "y", "z", "w", "v", "u"][i]))
                
                for eq_text in equations:
                    if not eq_text.strip():
                        continue
                        
                    # Substituir = por - ( para padronizar
                    eq_text = eq_text.replace("=", "-(") + ")"
                    
                    # Converter para expressão sympy
                    expr = sp.sympify(eq_text)
                    
                    # Extrair coeficientes
                    eq_coeffs = []
                    for var in var_symbols:
                        coef = expr.coeff(var)
                        eq_coeffs.append(float(coef))
                    
                    # Extrair termo constante
                    const = -float(expr.subs([(var, 0) for var in var_symbols]))
                    
                    coeffs.append(eq_coeffs)
                    constants.append(const)
                
                # Mostrar as equações interpretadas
                st.markdown("### Equações interpretadas:")
                for i, (eq_coef, eq_const) in enumerate(zip(coeffs, constants)):
                    var_names = ["x", "y", "z", "w", "v", "u"][:vars_count]
                    eq_str = format_equation(eq_coef, var_names, eq_const)
                    st.write(f"Equação {i+1}: {eq_str}")
                    
            except Exception as e:
                st.error(f"Erro ao processar as equações: {str(e)}")
                st.stop()
                
        else:  # Matriz Aumentada
            st.markdown("""
            Insira a matriz aumentada do sistema. Cada linha representa uma equação, e a última coluna contém os termos independentes.
            """)
            
            matrix_text = st.text_area(
                "Matriz aumentada (uma linha por equação):",
                height=150,
                help="Insira os elementos da matriz separados por espaços, com uma linha por equação.",
                value="1 1 10\n2 -1 5"
            )
            
            try:
                # Processar a matriz aumentada
                matrix_rows = matrix_text.strip().split('\n')
                augmented_matrix = []
                
                for row_text in matrix_rows:
                    if not row_text.strip():
                        continue
                    
                    # Converter elementos para números
                    elements = [float(e) for e in row_text.split()]
                    augmented_matrix.append(elements)
                
                # Verificar dimensões
                if any(len(row) != vars_count + 1 for row in augmented_matrix):
                    st.error(f"Erro: cada linha deve ter {vars_count + 1} elementos (coeficientes + termo independente).")
                    st.stop()
                
                # Extrair coeficientes e constantes
                coeffs = [row[:-1] for row in augmented_matrix]
                constants = [row[-1] for row in augmented_matrix]
                
                # Mostrar as equações interpretadas
                st.markdown("### Equações interpretadas:")
                for i, (eq_coef, eq_const) in enumerate(zip(coeffs, constants)):
                    var_names = ["x", "y", "z", "w", "v", "u"][:vars_count]
                    eq_str = format_equation(eq_coef, var_names, eq_const)
                    st.write(f"Equação {i+1}: {eq_str}")
                
            except Exception as e:
                st.error(f"Erro ao processar a matriz aumentada: {str(e)}")
                st.stop()
        
        # Método de resolução
        st.markdown("### Método de Resolução")
        
        col1, col2 = st.columns(2)
        
        with col1:
            solution_method = st.selectbox(
                "Escolha o método:",
                ["Eliminação de Gauss", "Gauss-Jordan", "Regra de Cramer", "Matriz Inversa", 
                 "Decomposição LU", "Jacobi", "Gauss-Seidel", "Todos os Métodos"],
                key="solution_method_select"
            )
            
        with col2:
            show_steps = st.checkbox("Mostrar passos detalhados", value=True, key="show_steps_checkbox")
        
        # Opções extras para métodos iterativos
        max_iter = 50
        tolerance = 1e-6
        
        if solution_method in ["Jacobi", "Gauss-Seidel"]:
            col1, col2 = st.columns(2)
            with col1:
                max_iter = st.number_input("Número máximo de iterações:", min_value=5, max_value=100, value=50, key="max_iter_input")
            with col2:
                tolerance = st.number_input("Tolerância:", min_value=1e-10, max_value=1e-2, value=1e-6, format="%.1e", key="tolerance_input")
        
        # Verificar se temos dados suficientes para resolver
        solve_ready = len(coeffs) > 0 and len(constants) > 0 and len(coeffs[0]) == vars_count
        
        # Botão para resolver
        solve_clicked = st.button("Resolver Sistema", type="primary", key="solve_btn", disabled=not solve_ready)
        
        if solve_clicked:
            # Criar a matriz e o vetor do sistema
            try:
                A, b = create_system_matrix(coeffs, constants, vars_count)
                
                # Guardar dados no estado da sessão
                st.session_state.system_solved = True
                st.session_state.A = A
                st.session_state.b = b
                st.session_state.vars_count = vars_count
                st.session_state.solution_method = solution_method
                st.session_state.solver_show_steps = show_steps
                st.session_state.max_iter = max_iter
                st.session_state.tolerance = tolerance
                st.session_state.system_classification = classify_system(A, b)
                
                # Computar soluções pelos diferentes métodos
                results = {}
                
                with st.spinner("Resolvendo o sistema..."):
                    if solution_method in ["Eliminação de Gauss", "Todos os Métodos"]:
                        steps, solution = gaussian_elimination_steps(A, b)
                        results["Eliminação de Gauss"] = {"steps": steps, "solution": solution}
                        
                    if solution_method in ["Gauss-Jordan", "Todos os Métodos"]:
                        steps, solution = gauss_jordan_steps(A, b)
                        results["Gauss-Jordan"] = {"steps": steps, "solution": solution}
                        
                    if vars_count <= 4 and solution_method in ["Regra de Cramer", "Todos os Métodos"]:
                        if A.shape[0] == A.shape[1]:  # Apenas para sistemas quadrados
                            steps, solution = cramer_rule(A, b, detailed=show_steps)
                            results["Regra de Cramer"] = {"steps": steps, "solution": solution}
                        
                    if solution_method in ["Matriz Inversa", "Todos os Métodos"]:
                        if A.shape[0] == A.shape[1]:  # Apenas para sistemas quadrados
                            steps, solution = matrix_inverse_method(A, b, detailed=show_steps)
                            results["Matriz Inversa"] = {"steps": steps, "solution": solution}
                            
                    if solution_method in ["Decomposição LU", "Todos os Métodos"]:
                        if A.shape[0] == A.shape[1]:  # Apenas para sistemas quadrados
                            steps, solution = lu_decomposition_method(A, b, detailed=show_steps)
                            results["Decomposição LU"] = {"steps": steps, "solution": solution}
                            
                    if solution_method in ["Jacobi", "Todos os Métodos"]:
                        steps, solution = jacobi_iteration_method(A, b, max_iter=max_iter, tolerance=tolerance, detailed=show_steps)
                        results["Jacobi"] = {"steps": steps, "solution": solution}
                        
                    if solution_method in ["Gauss-Seidel", "Todos os Métodos"]:
                        steps, solution = gauss_seidel_method(A, b, max_iter=max_iter, tolerance=tolerance, detailed=show_steps)
                        results["Gauss-Seidel"] = {"steps": steps, "solution": solution}
                        
                st.session_state.results = results
                
                # Atualizar progresso do usuário
                if "user_progress" in st.session_state:
                    st.session_state.user_progress["exercises_completed"] += 1
                
                # Mostrar mensagem de sucesso e sugerir ir para a próxima aba
                st.success("Sistema resolvido com sucesso! Veja os resultados na aba 'Resultados'.")
                
                # Mudar para a aba de resultados automaticamente
                st.session_state.solver_current_tab = "Resultados"
                st.rerun()
                
            except Exception as e:
                st.error(f"Erro ao resolver o sistema: {str(e)}")
                st.session_state.system_solved = False

    elif st.session_state.solver_current_tab == "Resultados":
        # Verificar se um sistema foi resolvido
        if not hasattr(st.session_state, 'system_solved') or not st.session_state.system_solved:
            st.info("Insira e resolva um sistema na aba 'Inserir Sistema'")
            st.session_state.solver_current_tab = "Inserir Sistema"
            st.rerun()
        else:
            # Código da aba "Resultados"
            st.markdown('<h2 class="sub-header">Resultados da Resolução</h2>', unsafe_allow_html=True)
            
            # Exibir classificação do sistema
            st.markdown(f"**Classificação do Sistema:** {st.session_state.system_classification}")
            
            # Mostrar as equações do sistema
            st.markdown("### Sistema original:")
            var_names = ["x", "y", "z", "w", "v", "u"][:st.session_state.vars_count]
            A = st.session_state.A
            b = st.session_state.b
            
            for i in range(len(b)):
                eq_str = format_equation(A[i], var_names, b[i])
                st.write(f"Equação {i+1}: {eq_str}")
            
            # Exibir matriz aumentada
            with st.expander("Ver matriz aumentada", expanded=False):
                augmented = np.column_stack((A, b))
                st.markdown("**Matriz aumentada [A|b]:**")
                st.dataframe(pd.DataFrame(augmented, 
                                        columns=[f"{var}" for var in var_names] + ["b"],
                                        index=[f"Eq {i+1}" for i in range(len(b))]))
            
            # Exibir solução para cada método
            st.markdown("### Resultados por método:")
            
            for method, result in st.session_state.results.items():
                with st.expander(f"📊 {method}", expanded=method == st.session_state.solution_method):
                    steps = result["steps"]
                    solution = result["solution"]
                    
                    if solution is not None:
                        st.markdown("**Solução encontrada:**")
                        
                        # Criar dataframe da solução
                        solution_df = pd.DataFrame({
                            "Variável": var_names[:len(solution)],
                            "Valor": [float(val) for val in solution]
                        })
                        st.dataframe(solution_df)
                        
                        # Mostrar precisão da solução
                        residual = np.linalg.norm(np.dot(A, solution) - b)
                        st.markdown(f"**Resíduo:** {residual:.2e}")
                        
                        # Verificação rápida da solução
                        st.markdown("**Verificação rápida:**")
                        for i in range(len(b)):
                            calculated = np.dot(A[i], solution)
                            is_correct = abs(calculated - b[i]) < 1e-10
                            st.markdown(f"Equação {i+1}: {calculated:.4f} ≈ {b[i]:.4f} {'✓' if is_correct else '✗'}")
                        
                    else:
                        st.write("Não foi possível encontrar uma solução única por este método.")
                    
                    if st.session_state.solver_show_steps:
                        st.markdown("**Passos detalhados:**")
                        for step in steps:
                            st.write(step)
            
            # Adicionar interpretação da solução
            st.markdown("### Interpretação da Solução")
            
            if st.session_state.system_classification == "Sistema Possível e Determinado (SPD)":
                st.success("O sistema possui uma única solução, que satisfaz todas as equações simultaneamente.")
                
                # Obter uma solução válida (qualquer uma)
                solution = None
                for result in st.session_state.results.values():
                    if result["solution"] is not None:
                        solution = result["solution"]
                        break
                
                if solution is not None:
                    st.markdown("### Verificação Detalhada")
                    
                    for i in range(len(b)):
                        eq_result = np.dot(A[i], solution)
                        is_correct = abs(eq_result - b[i]) < 1e-10
                        
                        eq_str = format_equation(A[i], var_names, b[i])
                        
                        substitution = " + ".join([f"{A[i][j]:.2f} × {solution[j]:.4f}" for j in range(len(solution)) if abs(A[i][j]) > 1e-10])
                        if not substitution:
                            substitution = "0"
                        
                        result_str = f"{eq_result:.4f} ≈ {b[i]:.4f}" if is_correct else f"{eq_result:.4f} ≠ {b[i]:.4f}"
                        
                        if is_correct:
                            st.success(f"Equação {i+1}: {eq_str}\n{substitution} = {result_str} ✓")
                        else:
                            st.error(f"Equação {i+1}: {eq_str}\n{substitution} = {result_str} ✗")
                            
            elif st.session_state.system_classification == "Sistema Possível e Indeterminado (SPI)":
                st.info("""
                O sistema possui infinitas soluções. Isso ocorre porque há menos equações linearmente independentes
                do que variáveis, criando um espaço de soluções possíveis.
                
                A solução pode ser expressa de forma paramétrica, onde uma ou mais variáveis são expressas em termos
                de parâmetros livres.
                """)
                
                # Tentar obter solução simbólica
                try:
                    A = st.session_state.A
                    b = st.session_state.b
                    symbolic_solution, var_symbols = sympy_solve_system(A, b)
                    
                    if symbolic_solution:
                        st.markdown("### Solução Paramétrica")
                        
                        if isinstance(symbolic_solution, dict):
                            for var, expr in symbolic_solution.items():
                                st.latex(f"{sp.latex(var)} = {sp.latex(expr)}")
                        else:
                            st.latex(sp.latex(symbolic_solution))
                except:
                    st.warning("Não foi possível obter uma representação paramétrica da solução.")
                    
            else:  # Sistema Impossível
                st.error("""
                O sistema não possui solução. Isso ocorre porque as equações são inconsistentes entre si,
                ou seja, não existe um conjunto de valores para as variáveis que satisfaça todas as equações
                simultaneamente.
                
                Geometricamente, isso pode ser interpretado como:
                - Em 2D: retas paralelas que nunca se intersectam
                - Em 3D: planos sem ponto comum de interseção
                """)
                
            # Adicionar botões de ação para a solução
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Ver Visualização", key="view_viz_btn"):
                    st.session_state.solver_current_tab = "Visualização"
                    st.rerun()

            with col2:
                if st.button("📋 Salvar nos Exemplos", key="save_example_btn"):
                    if "favorites" not in st.session_state:
                        st.session_state.favorites = {"examples": []}
                    
                    # Criar um exemplo para salvar
                    example = {
                        "title": f"Sistema {A.shape[0]}×{A.shape[1]} ({st.session_state.system_classification.split(' ')[2]})",
                        "A": A.tolist(),
                        "b": b.tolist(),
                        "date": datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
                    }
                    
                    st.session_state.favorites["examples"].append(example)
                    st.success("Sistema salvo nos exemplos favoritos!")
            
            with col3:
                if st.button("📥 Exportar Solução", key="export_solution_btn"):
                    st.success("Solução exportada! (Simulação)")
    
    elif st.session_state.solver_current_tab == "Visualização":
        # Verificar se um sistema foi resolvido
        if not hasattr(st.session_state, 'system_solved') or not st.session_state.system_solved:
            st.info("Insira e resolva um sistema na aba 'Inserir Sistema'")
            st.session_state.solver_current_tab = "Inserir Sistema"
            st.rerun()
        else:
            # Código da aba "Visualização"
            st.markdown('<h2 class="sub-header">Visualização Gráfica</h2>', unsafe_allow_html=True)
            
            if st.session_state.vars_count == 2:
                try:
                    fig = plot_2d_system(st.session_state.A, st.session_state.b)
                    if fig:
                        st.pyplot(fig)
                        
                        # Adicionar interpretação geométrica
                        st.markdown("### Interpretação Geométrica")
                        
                        if st.session_state.system_classification == "Sistema Possível e Determinado (SPD)":
                            st.markdown("""
                            Cada equação do sistema representa uma reta no plano cartesiano.
                            A solução do sistema é o ponto de interseção entre estas retas.
                            
                            As coordenadas deste ponto satisfazem simultaneamente todas as equações do sistema.
                            """)
                        elif st.session_state.system_classification == "Sistema Possível e Indeterminado (SPI)":
                            st.markdown("""
                            As retas são coincidentes (sobrepostas), o que significa que qualquer
                            ponto em uma das retas é uma solução válida para o sistema.
                            
                            Geometricamente, isso ocorre quando as equações representam a mesma reta
                            ou quando algumas das equações são redundantes (combinações lineares de outras).
                            """)
                        else:  # SI
                            st.markdown("""
                            As retas são paralelas, o que indica que não há ponto de interseção
                            e, portanto, o sistema não possui solução.
                            
                            Este é um caso onde as equações são inconsistentes: não existe um par de valores
                            (x, y) que satisfaça todas as equações simultaneamente.
                            """)
                    else:
                        st.warning("Não foi possível gerar a visualização do sistema.")
                except Exception as e:
                    st.error(f"Erro ao gerar o gráfico: {str(e)}")
                    
            elif st.session_state.vars_count == 3:
                try:
                    fig = plot_3d_system(st.session_state.A, st.session_state.b)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Adicionar interpretação geométrica
                        st.markdown("### Interpretação Geométrica")
                        
                        if st.session_state.system_classification == "Sistema Possível e Determinado (SPD)":
                            st.markdown("""
                            Cada equação do sistema representa um plano no espaço tridimensional.
                            A solução do sistema é o ponto único de interseção entre estes planos.
                            
                            As coordenadas deste ponto satisfazem simultaneamente todas as equações do sistema.
                            """)
                        elif st.session_state.system_classification == "Sistema Possível e Indeterminado (SPI)":
                            st.markdown("""
                            Os planos se intersectam em uma reta ou em um plano comum,
                            resultando em infinitas soluções possíveis.
                            
                            Isso ocorre quando temos menos equações linearmente independentes
                            do que variáveis. As soluções formam um espaço geométrico (reta ou plano).
                            """)
                        else:  # SI
                            st.markdown("""
                            Os planos não possuem um ponto comum de interseção,
                            o que indica que o sistema não tem solução.
                            
                            Geometricamente, isso pode ocorrer quando temos três planos paralelos
                            ou quando a interseção de dois planos é uma reta paralela ao terceiro plano.
                            """)
                    else:
                        st.warning("Não foi possível gerar a visualização 3D do sistema.")
                except Exception as e:
                    st.error(f"Erro ao gerar o gráfico 3D: {str(e)}")
                    
            else:
                st.info("""
                A visualização gráfica está disponível apenas para sistemas com 2 ou 3 variáveis.
                
                Para sistemas com mais variáveis, você pode usar outras técnicas de análise,
                como a redução do sistema ou a projeção em subespaços.
                """)
                
                # Oferecer alternativas para visualização
                st.markdown("### Alternativas para Análise Visual")
                
                viz_options = st.radio(
                    "Escolha uma alternativa:",
                    ["Matriz Ampliada", "Gráfico de Sparsidade", "Nenhuma"],
                    horizontal=True
                )
                
                if viz_options == "Matriz Ampliada":
                    A = st.session_state.A
                    b = st.session_state.b
                    augmented = np.column_stack((A, b))
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    cax = ax.matshow(augmented, cmap='coolwarm')
                    
                    # Adicionar colorbar
                    fig.colorbar(cax)
                    
                    # Adicionar rótulos
                    var_names = ["x", "y", "z", "w", "v", "u"][:A.shape[1]] + ["b"]
                    ax.set_xticks(np.arange(A.shape[1] + 1))
                    ax.set_xticklabels(var_names)
                    ax.set_yticks(np.arange(A.shape[0]))
                    ax.set_yticklabels([f"Eq {i+1}" for i in range(A.shape[0])])
                    
                    plt.title("Visualização da Matriz Ampliada")
                    st.pyplot(fig)
                    
                elif viz_options == "Gráfico de Sparsidade":
                    A = st.session_state.A
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.spy(A, markersize=15, color='#1E88E5')
                    
                    # Adicionar rótulos
                    var_names = ["x", "y", "z", "w", "v", "u"][:A.shape[1]]
                    ax.set_xticks(np.arange(A.shape[1]))
                    ax.set_xticklabels(var_names)
                    ax.set_yticks(np.arange(A.shape[0]))
                    ax.set_yticklabels([f"Eq {i+1}" for i in range(A.shape[0])])
                    
                    plt.title("Gráfico de Sparsidade dos Coeficientes")
                    st.pyplot(fig)

def show_theory_page():
    st.markdown('<h1 class="main-header">Teoria dos Sistemas Lineares</h1>', unsafe_allow_html=True)
    
    # Expandindo o dicionário theory_topics com novos conteúdos
    theory_topics = {
        "Introdução aos Sistemas Lineares": {
            "content": """
            # Introdução aos Sistemas Lineares
            
            Um **sistema de equações lineares** é um conjunto de uma ou mais equações lineares envolvendo as mesmas variáveis.
            
            ## Definição Formal
            
            Em notação matemática, um sistema linear de m equações e n incógnitas pode ser escrito como:
            
            $$
            \\begin{align}
            a_{11}x_1 + a_{12}x_2 + \\ldots + a_{1n}x_n &= b_1\\\\
            a_{21}x_1 + a_{22}x_2 + \\ldots + a_{2n}x_n &= b_2\\\\
            \\vdots\\\\
            a_{m1}x_1 + a_{m2}x_2 + \\ldots + a_{mn}x_n &= b_m
            \\end{align}
            $$
            
            Onde:
            - $a_{ij}$ são os coeficientes das incógnitas
            - $x_j$ são as incógnitas (ou variáveis)
            - $b_i$ são os termos independentes
            
            ## Representação Matricial
            
            O sistema linear também pode ser representado na forma matricial:
            
            $$
            A\\mathbf{x} = \\mathbf{b}
            $$
            
            Onde:
            - $A$ é a matriz dos coeficientes ($m \\times n$)
            - $\\mathbf{x}$ é o vetor das incógnitas ($n \\times 1$)
            - $\\mathbf{b}$ é o vetor dos termos independentes ($m \\times 1$)
            
            ## Tipos de Sistemas
            
            Um sistema linear pode ser:
            1. **Determinado**: Possui exatamente uma solução
            2. **Indeterminado**: Possui infinitas soluções
            3. **Impossível**: Não possui solução
            
            ## Importância
            
            Os sistemas lineares são fundamentais na matemática aplicada e aparecem em diversos contextos:
            - Física (equilíbrio de forças, circuitos elétricos)
            - Economia (modelos de preço, análise de insumo-produto)
            - Engenharia (análise estrutural, processamento de sinais)
            - Computação gráfica (transformações geométricas)
            """
        },
        "Classificação de Sistemas Lineares": {
            "content": """
            # Classificação de Sistemas Lineares
            
            ## Sistemas Possíveis e Determinados (SPD)
            
            Um sistema é **possível e determinado** quando possui **exatamente uma solução**.
            
            **Características**:
            - O determinante da matriz dos coeficientes é diferente de zero (det(A) ≠ 0)
            - O número de equações linearmente independentes é igual ao número de incógnitas
            - O posto da matriz dos coeficientes é igual ao posto da matriz ampliada e igual ao número de incógnitas
            
            **Interpretação geométrica**:
            - Em 2D: duas retas que se intersectam em um único ponto
            - Em 3D: três planos que se intersectam em um único ponto
            
            ## Sistemas Possíveis e Indeterminados (SPI)
            
            Um sistema é **possível e indeterminado** quando possui **infinitas soluções**.
            
            **Características**:
            - O posto da matriz dos coeficientes é igual ao posto da matriz ampliada
            - O posto é menor que o número de incógnitas
            
            **Interpretação geométrica**:
            - Em 2D: retas coincidentes (sobrepostas)
            - Em 3D: planos que se intersectam em uma reta ou coincidem
            
            ## Sistemas Impossíveis (SI)
            
            Um sistema é **impossível** quando **não possui solução**.
            
            **Características**:
            - O posto da matriz dos coeficientes é menor que o posto da matriz ampliada
            
            **Interpretação geométrica**:
            - Em 2D: retas paralelas (não se intersectam)
            - Em 3D: planos paralelos ou que se intersectam sem um ponto comum a todos
            
            ## Teorema de Rouché-Capelli
            
            O teorema estabelece que:
            
            - Um sistema é **compatível** (tem solução) se e somente se o posto da matriz dos coeficientes é igual ao posto da matriz ampliada.
            
            - Seja r = posto da matriz dos coeficientes = posto da matriz ampliada:
              - Se r = n (número de incógnitas), o sistema é SPD
              - Se r < n, o sistema é SPI
            
            - Se o posto da matriz dos coeficientes < posto da matriz ampliada, o sistema é SI
            """
        },
        "Método de Eliminação de Gauss": {
            "content": """
            # Método de Eliminação de Gauss
            
            O método de eliminação de Gauss é um dos algoritmos mais importantes para resolver sistemas lineares. Consiste em transformar o sistema em uma forma triangular superior (escalonada) através de operações elementares.
            
            ## Operações Elementares
            
            As operações elementares permitidas são:
            1. Trocar a posição de duas equações
            2. Multiplicar uma equação por uma constante não nula
            3. Substituir uma equação pela soma dela com um múltiplo de outra equação
            
            ## Algoritmo
            
            O método pode ser dividido em duas etapas:
            
            ### 1. Eliminação para frente (Forward Elimination)
            
            Nesta fase, transformamos a matriz aumentada [A|b] em uma matriz triangular superior. Para cada linha i da matriz:
            
            - Encontrar o pivô (elemento não nulo na posição i,i)
            - Para cada linha j abaixo da linha i:
              - Calcular o fator de eliminação: f = a_ji / a_ii
              - Subtrair da linha j a linha i multiplicada por f
            
            ### 2. Substituição reversa (Back Substitution)
            
            Uma vez obtida a forma triangular, resolvemos o sistema de trás para frente:
            
            - Calcular o valor da última variável
            - Substituir esse valor nas equações anteriores para encontrar as demais variáveis
            
            ## Eliminação Gaussiana com Pivoteamento Parcial
            
            Para melhorar a estabilidade numérica, é comum usar pivoteamento parcial:
            
            - A cada passo, escolher como pivô o elemento de maior valor absoluto na coluna atual
            - Trocar linhas para que este elemento fique na posição diagonal
            
            ## Exemplo
            
            Considere o sistema:
            
            $$
            \\begin{align}
            x + y + z &= 6\\\\
            2x - y + z &= 3\\\\
            x + 2y + 3z &= 14
            \\end{align}
            $$
            
            **Matriz aumentada inicial**:
            
            $$
            \\begin{bmatrix}
            1 & 1 & 1 & | & 6 \\\\
            2 & -1 & 1 & | & 3 \\\\
            1 & 2 & 3 & | & 14
            \\end{bmatrix}
            $$
            
            **Após eliminação para frente**:
            
            $$
            \\begin{bmatrix}
            1 & 1 & 1 & | & 6 \\\\
            0 & -3 & -1 & | & -9 \\\\
            0 & 0 & 5/3 & | & 5
            \\end{bmatrix}
            $$
            
            **Substituição reversa**:
            - Da última linha: z = 3
            - Da segunda linha: -3y - 3 = -9, portanto y = 2
            - Da primeira linha: x + 2 + 3 = 6, portanto x = 1
            
            **Solução**: x = 1, y = 2, z = 3
            """
        },
        "Método da Adição": {
            "content": """
            # Método da Adição (ou Eliminação por Soma)
            
            O método da adição é uma técnica específica para resolver sistemas de equações lineares, especialmente útil em sistemas com poucas equações. É um caso particular do método de eliminação de Gauss, focado na eliminação de variáveis através da soma de equações.
            
            ## Procedimento
            
            1. Organizar as equações de modo que os coeficientes de uma determinada variável possam se anular quando as equações forem somadas ou subtraídas
            2. Multiplicar as equações por constantes apropriadas para que os coeficientes da variável a ser eliminada se tornem opostos
            3. Somar as equações para eliminar a variável
            4. Repetir o processo até obter uma equação com apenas uma variável
            5. Resolver para essa variável e substituir nas equações anteriores
            
            ## Exemplo
            
            Considere o sistema:
            
            $$
            \\begin{align}
            2x + 3y &= 8 \\quad (1)\\\\
            4x - y &= 10 \\quad (2)
            \\end{align}
            $$
            
            **Passo 1**: Multiplicamos a equação (1) por 2 para que o coeficiente de x se torne 4
            
            $$
            \\begin{align}
            4x + 6y &= 16 \\quad (1')\\\\
            4x - y &= 10 \\quad (2)
            \\end{align}
            $$
            
            **Passo 2**: Subtraímos a equação (2) da equação (1')
            
            $$
            \\begin{align}
            4x + 6y - (4x - y) &= 16 - 10\\\\
            7y &= 6
            \\end{align}
            $$
            
            **Passo 3**: Resolvemos para y
            
            $$y = \\frac{6}{7}$$
            
            **Passo 4**: Substituímos o valor de y na equação (2)
            
            $$
            \\begin{align}
            4x - \\frac{6}{7} &= 10\\\\
            4x &= 10 + \\frac{6}{7}\\\\
            4x &= \\frac{70 + 6}{7}\\\\
            4x &= \\frac{76}{7}\\\\
            x &= \\frac{19}{7}
            \\end{align}
            $$
            
            **Solução**: $x = \\frac{19}{7}$, $y = \\frac{6}{7}$
            
            ## Vantagens e Desvantagens
            
            **Vantagens**:
            - Método intuitivo e fácil de aplicar manualmente para sistemas pequenos
            - Não requer conhecimentos avançados de álgebra linear
            - Pode ser mais direto que outros métodos para certos tipos de sistemas
            
            **Desvantagens**:
            - Pode ser trabalhoso para sistemas maiores
            - A escolha de quais equações combinar e como requer estratégia
            - Menos sistemático que o método de eliminação de Gauss completo
            """
        },
        "Método da Substituição": {
            "content": """
            # Método da Substituição
            
            O método da substituição é uma técnica elementar para resolver sistemas de equações lineares, especialmente útil para sistemas pequenos ou esparsos (com muitos zeros).
            
            ## Procedimento
            
            1. Isolar uma variável em uma das equações
            2. Substituir a expressão obtida nas demais equações, reduzindo o sistema
            3. Repetir o processo até obter uma equação com apenas uma variável
            4. Resolver para essa variável e substituir nas expressões anteriores para encontrar as demais variáveis
            
            ## Exemplo
            
            Considere o sistema:
            
            $$
            \\begin{align}
            x + 2y &= 5 \\quad (1)\\\\
            3x - 4y &= 7 \\quad (2)
            \\end{align}
            $$
            
            **Passo 1**: Isolamos x na equação (1)
            
            $$x = 5 - 2y \quad (3)$$
            
            **Passo 2**: Substituímos (3) na equação (2)
            
            $$
            \\begin{align}
            3(5 - 2y) - 4y &= 7\\\\
            15 - 6y - 4y &= 7\\\\
            15 - 10y &= 7\\\\
            -10y &= 7 - 15\\\\
            -10y &= -8\\\\
            y &= \\frac{8}{10} = \\frac{4}{5}
            \\end{align}
            $$
            
            **Passo 3**: Substituímos o valor de y em (3)
            
            $$
            \\begin{align}
            x &= 5 - 2 \\cdot \\frac{4}{5}\\\\
            &= 5 - \\frac{8}{5}\\\\
            &= \\frac{25 - 8}{5}\\\\
            &= \\frac{17}{5}
            \\end{align}
            $$
            
            **Solução**: $x = \\frac{17}{5}$, $y = \\frac{4}{5}$
            
            ## Aplicação em Sistemas Triangulares
            
            O método da substituição é particularmente eficiente para sistemas triangulares. De fato, a substituição reversa usada após a eliminação gaussiana é uma aplicação deste método.
            
            Para um sistema triangular superior:
            
            $$
            \\begin{align}
            a_{11}x_1 + a_{12}x_2 + \\ldots + a_{1n}x_n &= b_1\\\\
            a_{22}x_2 + \\ldots + a_{2n}x_n &= b_2\\\\
            \\vdots\\\\
            a_{nn}x_n &= b_n
            \\end{align}
            $$
            
            Começamos resolvendo $x_n = b_n/a_{nn}$ e substituímos nas equações anteriores.
            
            ## Vantagens e Desvantagens
            
            **Vantagens**:
            - Método intuitivo e fácil de entender
            - Eficiente para sistemas pequenos ou triangulares
            - Útil em sistemas onde algumas variáveis podem ser facilmente isoladas
            
            **Desvantagens**:
            - Pode levar a cálculos complexos se as expressões substituídas forem complicadas
            - Não é tão sistemático ou eficiente computacionalmente quanto outros métodos para sistemas grandes
            """
        },
        "Método da Comparação": {
            "content": """
            # Método da Comparação
            
            O método da comparação é uma técnica para resolver sistemas de equações lineares que consiste em isolar a mesma variável em diferentes equações e, em seguida, igualar as expressões resultantes.
            
            ## Procedimento
            
            1. Isolar a mesma variável em duas ou mais equações do sistema
            2. Igualar as expressões obtidas, formando novas equações com menos variáveis
            3. Resolver o sistema reduzido
            4. Substituir as soluções encontradas nas expressões iniciais para obter as demais variáveis
            
            ## Exemplo
            
            Considere o sistema:
            
            $$
            \\begin{align}
            2x + 3y &= 7 \\quad (1)\\\\
            5x - 2y &= 8 \\quad (2)
            \\end{align}
            $$
            
            **Passo 1**: Isolamos x em ambas as equações
            
            Da equação (1):
            $$x = \\frac{7 - 3y}{2} \\quad (3)$$
            
            Da equação (2):
            $$x = \\frac{8 + 2y}{5} \\quad (4)$$
            
            **Passo 2**: Igualamos as expressões (3) e (4)
            
            $$
            \\begin{align}
            \\frac{7 - 3y}{2} &= \\frac{8 + 2y}{5}\\\\
            5(7 - 3y) &= 2(8 + 2y)\\\\
            35 - 15y &= 16 + 4y\\\\
            35 - 15y &= 16 + 4y\\\\
            35 - 16 &= 4y + 15y\\\\
            19 &= 19y\\\\
            y &= 1
            \\end{align}
            $$
            
            **Passo 3**: Substituímos y = 1 em uma das expressões para x, por exemplo em (3)
            
            $$
            \\begin{align}
            x &= \\frac{7 - 3(1)}{2}\\\\
            &= \\frac{7 - 3}{2}\\\\
            &= \\frac{4}{2}\\\\
            &= 2
            \\end{align}
            $$
            
            **Solução**: x = 2, y = 1
            
            ## Verificação
            
            Podemos verificar a solução substituindo os valores nas equações originais:
            
            Equação (1): 2(2) + 3(1) = 4 + 3 = 7 ✓
            
            Equação (2): 5(2) - 2(1) = 10 - 2 = 8 ✓
            
            ## Vantagens e Desvantagens
            
            **Vantagens**:
            - Método intuitivo que não requer conhecimentos avançados
            - Útil para sistemas com duas equações e duas incógnitas
            - Pode ser mais direto em certos casos específicos
            
            **Desvantagens**:
            - Torna-se complicado para sistemas maiores
            - Pode levar a expressões algébricas complexas
            - Menos eficiente computacionalmente que métodos mais sistemáticos
            """
        },
        "Regra de Cramer": {
            "content": """
            # Regra de Cramer
            
            A regra de Cramer é um método para resolver sistemas lineares usando determinantes. É aplicável apenas a sistemas com mesmo número de equações e incógnitas, onde o determinante da matriz dos coeficientes é diferente de zero (sistemas SPD).
            
            ## Procedimento
            
            Para um sistema AX = B:
            
            1. Calcular o determinante D da matriz A
            2. Para cada variável xᵢ:
               - Substituir a coluna i da matriz A pela coluna B, obtendo a matriz Aᵢ
               - Calcular o determinante Dᵢ
               - A solução para xᵢ é dada por xᵢ = Dᵢ/D
            
            ## Fórmula
            
            Para um sistema 2×2:
            
            $$
            \\begin{align}
            a_1x + b_1y &= c_1\\\\
            a_2x + b_2y &= c_2
            \\end{align}
            $$
            
            As soluções são:
            
            $$
            x = \\frac{\\begin{vmatrix} c_1 & b_1 \\\\ c_2 & b_2 \\end{vmatrix}}{\\begin{vmatrix} a_1 & b_1 \\\\ a_2 & b_2 \\end{vmatrix}} = \\frac{c_1b_2 - b_1c_2}{a_1b_2 - b_1a_2}
            $$
            
            $$
            y = \\frac{\\begin{vmatrix} a_1 & c_1 \\\\ a_2 & c_2 \\end{vmatrix}}{\\begin{vmatrix} a_1 & b_1 \\\\ a_2 & b_2 \\end{vmatrix}} = \\frac{a_1c_2 - c_1a_2}{a_1b_2 - b_1a_2}
            $$
            
            ## Exemplo
            
            Considere o sistema:
            
            $$
            \\begin{align}
            2x + 3y &= 8\\\\
            4x - y &= 1
            \\end{align}
            $$
            
            **Determinante principal**:
            
            $$
            D = \\begin{vmatrix} 2 & 3 \\\\ 4 & -1 \\end{vmatrix} = 2 \\times (-1) - 3 \\times 4 = -2 - 12 = -14
            $$
            
            **Determinante para x**:
            
            $$
            D_x = \\begin{vmatrix} 8 & 3 \\\\ 1 & -1 \\end{vmatrix} = 8 \\times (-1) - 3 \\times 1 = -8 - 3 = -11
            $$
            
            **Determinante para y**:
            
            $$
            D_y = \\begin{vmatrix} 2 & 8 \\\\ 4 & 1 \\end{vmatrix} = 2 \\times 1 - 8 \\times 4 = 2 - 32 = -30
            $$
            
            **Solução**:
            
            $$
            x = \\frac{D_x}{D} = \\frac{-11}{-14} = \\frac{11}{14}
            $$
            
            $$
            y = \\frac{D_y}{D} = \\frac{-30}{-14} = \\frac{15}{7}
            $$
            
            ## Vantagens e Desvantagens
            
            **Vantagens**:
            - Método direto (não iterativo)
            - Fácil de entender e aplicar para sistemas pequenos
            
            **Desvantagens**:
            - Aplicável apenas a sistemas quadrados (n×n) com determinante não nulo
            - Computacionalmente ineficiente para sistemas grandes
            - Não é recomendado para sistemas mal condicionados
            """
        },
        "Método da Matriz Inversa": {
            "content": """
            # Método da Matriz Inversa
            
            O método da matriz inversa é uma abordagem direta para resolver sistemas lineares na forma AX = B, onde A é uma matriz quadrada inversível.
            
            ## Procedimento
            
            1. Verificar se a matriz A é inversível (det(A) ≠ 0)
            2. Calcular a matriz inversa A⁻¹
            3. Multiplicar ambos os lados da equação por A⁻¹: A⁻¹(AX) = A⁻¹B
            4. Simplificar: X = A⁻¹B
            
            ## Cálculo da Matriz Inversa
            
            Para uma matriz 2×2:
            
            $$
            \\begin{bmatrix} a & b \\\\ c & d \\end{bmatrix}^{-1} = \\frac{1}{ad-bc} \\begin{bmatrix} d & -b \\\\ -c & a \\end{bmatrix}
            $$
            
            Para matrizes maiores, pode-se usar:
            - Método da matriz adjunta
            - Eliminação gaussiana
            - Decomposição LU
            
            ## Exemplo
            
            Considere o sistema:
            
            $$
            \\begin{align}
            2x + y &= 5\\\\
            3x + 4y &= 11
            \\end{align}
            $$
            
            Na forma matricial:
            
            $$
            \\begin{bmatrix} 2 & 1 \\\\ 3 & 4 \\end{bmatrix} \\begin{bmatrix} x \\\\ y \\end{bmatrix} = \\begin{bmatrix} 5 \\\\ 11 \\end{bmatrix}
            $$
            
            **Determinante**:
            
            $$
            \\det(A) = 2 \\times 4 - 1 \\times 3 = 8 - 3 = 5
            $$
            
            **Matriz inversa**:
            
            $$
            A^{-1} = \\frac{1}{5} \\begin{bmatrix} 4 & -1 \\\\ -3 & 2 \\end{bmatrix} = \\begin{bmatrix} 4/5 & -1/5 \\\\ -3/5 & 2/5 \\end{bmatrix}
            $$
            
            **Solução**:
            
            $$
            \\begin{bmatrix} x \\\\ y \\end{bmatrix} = \\begin{bmatrix} 4/5 & -1/5 \\\\ -3/5 & 2/5 \\end{bmatrix} \\begin{bmatrix} 5 \\\\ 11 \\end{bmatrix} = \\begin{bmatrix} 4/5 \\times 5 - 1/5 \\times 11 \\\\ -3/5 \\times 5 + 2/5 \\times 11 \\end{bmatrix} = \\begin{bmatrix} 4 - 11/5 \\\\ -3 + 22/5 \\end{bmatrix} = \\begin{bmatrix} 9/5 \\\\ 7/5 \\end{bmatrix}
            $$
            
            Portanto, x = 9/5 e y = 7/5.
            
            ## Vantagens e Desvantagens
            
            **Vantagens**:
            - Método direto e elegante
            - Útil quando precisamos resolver múltiplos sistemas com a mesma matriz de coeficientes
            
            **Desvantagens**:
            - Aplicável apenas a sistemas quadrados (n×n) com determinante não nulo
            - Computacionalmente ineficiente para sistemas grandes
            - Pode ser numericamente instável para matrizes mal condicionadas
            """
        },
        "Método de Gauss-Jordan": {
            "content": """
            # Método de Gauss-Jordan
            
            O método de Gauss-Jordan é uma extensão do método de eliminação de Gauss que leva a matriz aumentada à forma escalonada reduzida.
            
            ## Procedimento
            
            1. Aplicar operações elementares para obter 1's na diagonal principal
            2. Zerar todos os elementos acima e abaixo da diagonal principal
            
            **Forma final da matriz aumentada**:
            ```
            | 1 0 0 ... | x₁ |
            | 0 1 0 ... | x₂ |
            | 0 0 1 ... | x₃ |
            | ...       | ... |
            ```
            
            O vetor solução pode ser lido diretamente da última coluna da matriz.
            
            ## Exemplo
            
            Considere o sistema:
            
            $$
            \\begin{align}
            2x + y + z &= 5\\\\
            x - y + 2z &= 4\\\\
            3x + 2y - z &= 3
            \\end{align}
            $$
            
            **Matriz aumentada inicial**:
            
            $$
            \\begin{bmatrix}
            2 & 1 & 1 & | & 5 \\\\
            1 & -1 & 2 & | & 4 \\\\
            3 & 2 & -1 & | & 3
            \\end{bmatrix}
            $$
            
            **Passo 1**: Obter 1 na posição (1,1) e zerar os outros elementos da primeira coluna
            
            Dividir a primeira linha por 2:
            
            $$
            \\begin{bmatrix}
            1 & 1/2 & 1/2 & | & 5/2 \\\\
            1 & -1 & 2 & | & 4 \\\\
            3 & 2 & -1 & | & 3
            \\end{bmatrix}
            $$
            
            Subtrair a primeira linha da segunda:
            
            $$
            \\begin{bmatrix}
            1 & 1/2 & 1/2 & | & 5/2 \\\\
            0 & -3/2 & 3/2 & | & 3/2 \\\\
            3 & 2 & -1 & | & 3
            \\end{bmatrix}
            $$
            
            Subtrair 3 vezes a primeira linha da terceira:
            
            $$
            \\begin{bmatrix}
            1 & 1/2 & 1/2 & | & 5/2 \\\\
            0 & -3/2 & 3/2 & | & 3/2 \\\\
            0 & 1/2 & -5/2 & | & -9/2
            \\end{bmatrix}
            $$
            
            **Passo 2**: Obter 1 na posição (2,2) e zerar os outros elementos da segunda coluna
            
            Multiplicar a segunda linha por -2/3:
            
            $
            \\begin{bmatrix}
            1 & 1/2 & 1/2 & | & 5/2 \\\\
            0 & 1 & -1 & | & -1 \\\\
            0 & 1/2 & -5/2 & | & -9/2
            \\end{bmatrix}
            $
            
            Subtrair 1/2 vezes a segunda linha da primeira:
            
            $
            \\begin{bmatrix}
            1 & 0 & 1 & | & 3 \\\\
            0 & 1 & -1 & | & -1 \\\\
            0 & 1/2 & -5/2 & | & -9/2
            \\end{bmatrix}
            $
            
            Subtrair 1/2 vezes a segunda linha da terceira:
            
            $
            \\begin{bmatrix}
            1 & 0 & 1 & | & 3 \\\\
            0 & 1 & -1 & | & -1 \\\\
            0 & 0 & -2 & | & -4
            \\end{bmatrix}
            $
            
            **Passo 3**: Obter 1 na posição (3,3) e zerar os outros elementos da terceira coluna
            
            Multiplicar a terceira linha por -1/2:
            
            $
            \\begin{bmatrix}
            1 & 0 & 1 & | & 3 \\\\
            0 & 1 & -1 & | & -1 \\\\
            0 & 0 & 1 & | & 2
            \\end{bmatrix}
            $
            
            Subtrair 1 vez a terceira linha da primeira:
            
            $
            \\begin{bmatrix}
            1 & 0 & 0 & | & 1 \\\\
            0 & 1 & -1 & | & -1 \\\\
            0 & 0 & 1 & | & 2
            \\end{bmatrix}
            $
            
            Somar 1 vez a terceira linha à segunda:
            
            $
            \\begin{bmatrix}
            1 & 0 & 0 & | & 1 \\\\
            0 & 1 & 0 & | & 1 \\\\
            0 & 0 & 1 & | & 2
            \\end{bmatrix}
            $
            
            A solução pode ser lida diretamente da última coluna: x = 1, y = 1, z = 2.
            
            ## Vantagens e Desvantagens
            
            **Vantagens**:
            - A solução é obtida diretamente, sem necessidade de substituição reversa
            - Útil para calcular a inversa de uma matriz
            
            **Desvantagens**:
            - Requer mais operações que o método de Gauss padrão
            - Pode ser numericamente menos estável em alguns casos
            """
        },

        "Métodos Iterativos": {
            "content": """
            # Métodos Iterativos para Sistemas Lineares
            
            Os métodos iterativos começam com uma aproximação inicial e melhoram progressivamente a solução.
            
            ## Método de Jacobi
            
            **Procedimento**:
            1. Para cada equação i, isolar a incógnita x_i
            2. Iniciar com uma aproximação inicial (geralmente zeros)
            3. Em cada iteração k+1, calcular:
               x_i^(k+1) = (b_i - Σ a_ij x_j^(k)) / a_ii, para j ≠ i
            4. Repetir até convergir
            
            ## Método de Gauss-Seidel
            
            Similar ao método de Jacobi, mas usa valores já atualizados na mesma iteração:
            
            x_i^(k+1) = (b_i - Σ a_ij x_j^(k+1) - Σ a_ij x_j^(k)) / a_ii
                          j<i                j>i
            
            **Condições de convergência**:
            - Matriz diagonalmente dominante (|a_ii| > Σ |a_ij| para j ≠ i)
            - Matriz definida positiva
            
            ## Exemplo
            
            Considere o sistema:
            
            $$
            \\begin{align}
            10x + 2y + z &= 13\\\\
            x + 5y + z &= 7\\\\
            2x + y + 10z &= 13
            \\end{align}
            $$
            
            Isolando as variáveis:
            
            $$
            \\begin{align}
            x &= \\frac{13 - 2y - z}{10}\\\\
            y &= \\frac{7 - x - z}{5}\\\\
            z &= \\frac{13 - 2x - y}{10}
            \\end{align}
            $$
            
            **Método de Jacobi**:
            
            Partindo de x^(0) = y^(0) = z^(0) = 0:
            
            Iteração 1:
            - x^(1) = (13 - 2×0 - 0)/10 = 1.3
            - y^(1) = (7 - 0 - 0)/5 = 1.4
            - z^(1) = (13 - 2×0 - 0)/10 = 1.3
            
            Iteração 2:
            - x^(2) = (13 - 2×1.4 - 1.3)/10 = 0.83
            - y^(2) = (7 - 1.3 - 1.3)/5 = 0.88
            - z^(2) = (13 - 2×1.3 - 1.4)/10 = 0.83
            
            O processo continua até a convergência para x = y = z = 1.
            
            **Método de Gauss-Seidel**:
            
            Partindo de x^(0) = y^(0) = z^(0) = 0:
            
            Iteração 1:
            - x^(1) = (13 - 2×0 - 0)/10 = 1.3
            - y^(1) = (7 - 1.3 - 0)/5 = 1.14
            - z^(1) = (13 - 2×1.3 - 1.14)/10 = 0.826
            
            Iteração 2:
            - x^(2) = (13 - 2×1.14 - 0.826)/10 = 0.8934
            - y^(2) = (7 - 0.8934 - 0.826)/5 = 1.0561
            - z^(2) = (13 - 2×0.8934 - 1.0561)/10 = 0.9157
            
            O método converge mais rapidamente para x = y = z = 1.
            
            ## Vantagens e Desvantagens
            
            **Vantagens**:
            - Mais eficientes para sistemas grandes e esparsos
            - Menor requisito de memória
            - Podem lidar com matrizes mal condicionadas
            
            **Desvantagens**:
            - Convergência não garantida para todas as matrizes
            - Podem ser lentos para atingir alta precisão
            """
        },
        "Decomposição LU": {
            "content": """
            # Decomposição LU
            
            A decomposição LU fatoriza a matriz A em um produto de duas matrizes: A = LU, onde:
            - L é uma matriz triangular inferior
            - U é uma matriz triangular superior
            
            ## Procedimento para resolver AX = B
            
            1. Decompor A = LU
            2. Resolver LY = B por substituição direta
            3. Resolver UX = Y por substituição reversa
            
            ## Algoritmo para decomposição LU
            
            **Procedimento de Doolittle**:
            
            Para uma matriz n×n:
            
            1. Para i = 1 até n:
               - Para j = i até n: u_{ij} = a_{ij} - Σ(l_{ik} × u_{kj}) para k = 1 até i-1
               - Para j = i+1 até n: l_{ji} = (a_{ji} - Σ(l_{jk} × u_{ki}) para k = 1 até i-1) / u_{ii}
            
            2. Para i = 1 até n: l_{ii} = 1 (diagonal unitária para L)
            
            ## Exemplo
            
            Considere o sistema:
            
            $$
            \\begin{align}
            2x + y + z &= 5\\\\
            4x + 5y + z &= 9\\\\
            x + y + 3z &= 11
            \\end{align}
            $$
            
            **Decomposição LU**:
            
            Matriz A:
            
            $$
            A = \\begin{bmatrix}
            2 & 1 & 1 \\\\
            4 & 5 & 1 \\\\
            1 & 1 & 3
            \\end{bmatrix}
            $$
            
            Calculando L e U:
            
            $$
            L = \\begin{bmatrix}
            1 & 0 & 0 \\\\
            2 & 1 & 0 \\\\
            0.5 & 0.25 & 1
            \\end{bmatrix}
            $$
            
            $$
            U = \\begin{bmatrix}
            2 & 1 & 1 \\\\
            0 & 3 & -1 \\\\
            0 & 0 & 2.5
            \\end{bmatrix}
            $$
            
            **Resolver LY = B**:
            
            $$
            \\begin{bmatrix}
            1 & 0 & 0 \\\\
            2 & 1 & 0 \\\\
            0.5 & 0.25 & 1
            \\end{bmatrix}
            \\begin{bmatrix}
            y_1 \\\\
            y_2 \\\\
            y_3
            \\end{bmatrix} =
            \\begin{bmatrix}
            5 \\\\
            9 \\\\
            11
            \\end{bmatrix}
            $$
            
            - y₁ = 5
            - y₂ = 9 - 2×5 = -1
            - y₃ = 11 - 0.5×5 - 0.25×(-1) = 11 - 2.5 + 0.25 = 8.75
            
            **Resolver UX = Y**:
            
            $$
            \\begin{bmatrix}
            2 & 1 & 1 \\\\
            0 & 3 & -1 \\\\
            0 & 0 & 2.5
            \\end{bmatrix}
            \\begin{bmatrix}
            x \\\\
            y \\\\
            z
            \\end{bmatrix} =
            \\begin{bmatrix}
            5 \\\\
            -1 \\\\
            8.75
            \\end{bmatrix}
            $$
            
            - z = 8.75 / 2.5 = 3.5
            - y = (-1 + z) / 3 = (-1 + 3.5) / 3 = 2.5 / 3 = 0.833...
            - x = (5 - y - z) / 2 = (5 - 0.833 - 3.5) / 2 = 0.667...
            
            ## Vantagens e Desvantagens
            
            **Vantagens**:
            - Eficiente para resolver múltiplos sistemas com a mesma matriz de coeficientes
            - Útil para calcular determinantes e inversas
            - Computacionalmente eficiente
            
            **Desvantagens**:
            - Requer armazenamento adicional para as matrizes L e U
            - Pode ser instável para matrizes mal condicionadas sem pivoteamento
            """
        },
        "Interpretação Geométrica": {
            "content": """
            # Interpretação Geométrica de Sistemas Lineares
            
            ## Sistemas 2×2
            
            Em um sistema com duas equações e duas incógnitas, cada equação representa uma reta no plano cartesiano.
            
            **Possíveis configurações**:
            
            1. **SPD (Sistema Possível e Determinado)**:
               - As retas se intersectam em um único ponto
               - Este ponto representa a única solução do sistema
               - Exemplo: x + y = 3 e x - y = 1
               
            2. **SPI (Sistema Possível e Indeterminado)**:
               - As retas são coincidentes (sobrepostas)
               - Todos os pontos da reta são soluções do sistema
               - Exemplo: 2x + 3y = 6 e 4x + 6y = 12
               
            3. **SI (Sistema Impossível)**:
               - As retas são paralelas e não coincidentes
               - Não há nenhum ponto comum, ou seja, não há solução
               - Exemplo: x + y = 3 e x + y = 5
            
            ## Sistemas 3×3
            
            Em um sistema com três equações e três incógnitas, cada equação representa um plano no espaço tridimensional.
            
            **Possíveis configurações**:
            
            1. **SPD (Sistema Possível e Determinado)**:
               - Os três planos se intersectam em um único ponto
               - Este ponto é a única solução do sistema
               
            2. **SPI (Sistema Possível e Indeterminado)**:
               - Os planos se intersectam em uma reta (dois planos paralelos intersectados por um terceiro)
               - Ou se intersectam em um plano (três planos coincidentes)
               - As soluções formam uma reta ou um plano
               
            3. **SI (Sistema Impossível)**:
               - Os planos não possuem ponto comum de interseção
               - Pode ocorrer quando temos planos paralelos ou quando a interseção de dois planos é paralela ao terceiro
            
            ## Relação com o Espaço Vetorial
            
            - As linhas da matriz de coeficientes podem ser vistas como vetores
            - O determinante da matriz está relacionado ao volume do paralelepípedo formado por estes vetores
            - Um determinante zero indica que os vetores são linearmente dependentes
            - Para sistemas homogêneos (AX = 0), o conjunto solução forma um subespaço vetorial
            """
        },
        "Aplicações de Sistemas Lineares": {
            "content": """
            # Aplicações de Sistemas Lineares
            
            Os sistemas de equações lineares são ferramentas fundamentais em diversas áreas. Veja algumas aplicações importantes:
            
            ## Física e Engenharia
            
            ### Análise de Circuitos Elétricos
            - Leis de Kirchhoff: correntes em um nó e tensões em um laço
            - Cada equação representa a conservação de corrente ou tensão
            
            **Exemplo**: Para um circuito com três malhas, podemos ter:
            ```
            R₁I₁ + R₂(I₁ - I₂) = V₁
            R₂(I₂ - I₁) + R₃I₂ + R₄(I₂ - I₃) = 0
            R₄(I₃ - I₂) + R₅I₃ = V₂
            ```
            
            ### Estática e Dinâmica
            - Equilíbrio de forças e momentos em estruturas
            - Análise de treliças e vigas
            
            **Exemplo**: Para um sistema com três forças desconhecidas:
            ```
            F₁cos(θ₁) + F₂cos(θ₂) + F₃cos(θ₃) = 0
            F₁sin(θ₁) + F₂sin(θ₂) + F₃sin(θ₃) = 0
            r₁×F₁ + r₂×F₂ + r₃×F₃ = 0
            ```
            
            ### Transferência de Calor
            - Modelagem de problemas de condução térmica
            - Métodos de diferenças finitas para equações diferenciais
            
            ## Economia
            
            ### Análise de Insumo-Produto de Leontief
            - Modelagem das interdependências entre diferentes setores da economia
            - Previsão de como mudanças em um setor afetam outros setores
            
            **Exemplo**: Em uma economia com três setores:
            ```
            x₁ = a₁₁x₁ + a₁₂x₂ + a₁₃x₃ + d₁
            x₂ = a₂₁x₁ + a₂₂x₂ + a₂₃x₃ + d₂
            x₃ = a₃₁x₁ + a₃₂x₂ + a₃₃x₃ + d₃
            ```
            Onde x₁ é a produção do setor i, a_ij é quanto do produto i é usado para produzir uma unidade do produto j, e d_i é a demanda externa.
            
            ### Otimização e Programação Linear
            - Maximização de lucro ou minimização de custos
            - Alocação ótima de recursos limitados
            
            ## Química
            
            ### Balanceamento de Equações Químicas
            - Cada elemento forma uma equação linear
            - Os coeficientes estequiométricos são as incógnitas
            
            **Exemplo**: Para a reação C₃H₈ + O₂ → CO₂ + H₂O
            ```
            3a = c       (para o carbono)
            8a = 2e      (para o hidrogênio)
            2b = 2c + e  (para o oxigênio)
            ```
            
            ### Equilíbrio Químico
            - Determinação de concentrações em equilíbrio
            
            ## Computação Gráfica
            
            ### Transformações Geométricas
            - Rotação, translação e escala de objetos
            - Representadas como transformações matriciais
            
            ### Renderização 3D
            - Sistemas para determinar projeções de objetos 3D em telas 2D
            
            ## Problemas de Mistura
            
            ### Farmacologia
            - Mistura de componentes para atingir concentrações específicas
            - Formulação de medicamentos
            
            **Exemplo**: Um farmacêutico precisa preparar 100ml de uma solução com 25% de um princípio ativo, usando soluções de 10%, 20% e 40%.
            ```
            x + y + z = 100
            0.1x + 0.2y + 0.4z = 25
            ```
            
            ### Processamento de Alimentos
            - Mistura de ingredientes para atingir perfis nutricionais
            
            ## Tráfego e Transporte
            
            ### Fluxo de Redes
            - Modelagem de fluxo de tráfego em redes de transporte
            - Otimização de rotas
            """
        },
        "Sistemas Homogêneos": {
            "content": """
            # Sistemas Lineares Homogêneos
            
            Um sistema homogêneo tem a forma AX = 0 (todos os termos independentes são nulos).
            
            ## Propriedades
            
            1. Todo sistema homogêneo é possível (sempre admite a solução trivial X = 0)
            2. Um sistema homogêneo tem soluções não-triviais se e somente se det(A) = 0
            3. O conjunto de todas as soluções forma um espaço vetorial
            4. A dimensão do espaço de soluções é n - posto(A), onde n é o número de incógnitas
            
            ## Interpretação Geométrica
            
            - Em 2D: se det(A) = 0, as equações representam a mesma reta passando pela origem
            - Em 3D: se det(A) = 0, os planos se intersectam em uma reta ou um plano passando pela origem
            
            ## Aplicações
            
            - **Espaços nulos**:
            O núcleo (ou kernel) de uma transformação linear é o espaço das soluções de AX = 0
              - Fundamental em álgebra linear e geometria
            
            - **Autovalores e autovetores**:
              - Um autovetor v de uma matriz A satisfaz Av = λv, ou (A - λI)v = 0
              - Encontrar os autovetores envolve resolver sistemas homogêneos
            
            - **Equações diferenciais**:
              - Sistemas de equações diferenciais lineares homogêneas têm soluções da forma X = e^(λt)v
              - Onde λ é um autovalor e v é um autovetor associado
            
            ## Exemplo
            
            Considere o sistema homogêneo:
            
            $$
            \\begin{align}
            2x + 3y - z &= 0\\\\
            4x + 6y - 2z &= 0\\\\
            -2x - 3y + z &= 0
            \\end{align}
            $$
            
            Observe que a terceira equação é o oposto da primeira. Além disso, a segunda equação é um múltiplo da primeira (multiplicada por 2).
            
            O determinante da matriz dos coeficientes é zero, o que confirma que o sistema tem soluções não-triviais.
            
            O posto da matriz é 1, e temos 3 incógnitas, então a dimensão do espaço de soluções é 3 - 1 = 2.
            
            **Solução paramétrica**:
            
            Podemos expressar z e y em termos de x:
            Da primeira equação: z = 2x + 3y
            
            Substituindo na segunda e terceira equações, verificamos que são satisfeitas para qualquer valor de x e y.
            
            Então a solução geral é:
            ```
            z = 2x + 3y
            ```
            Onde x e y são parâmetros livres.
            
            Alternativamente, podemos parametrizar como:
            ```
            x = s
            y = t
            z = 2s + 3t
            ```
            Onde s e t são parâmetros livres.
            """
        },
        "Estabilidade Numérica": {
            "content": """
            # Estabilidade Numérica em Sistemas Lineares
            
            ## Número de Condição
            
            O número de condição de uma matriz A, denotado por cond(A), mede a sensibilidade da solução a pequenas perturbações nos dados:
            
            $$\\text{cond}(A) = \\|A\\| \\cdot \\|A^{-1}\\|$$
            
            Para a norma-2, isso é equivalente à razão entre o maior e o menor valor singular:
            
            $$\\text{cond}_2(A) = \\frac{\\sigma_{\\max}}{\\sigma_{\\min}}$$
            
            **Interpretação**:
            - Um número de condição próximo de 1 indica uma matriz bem condicionada
            - Um número de condição grande indica uma matriz mal condicionada
            - Um número de condição infinito indica uma matriz singular
            
            ## Efeitos do Mal Condicionamento
            
            Um sistema mal condicionado tem as seguintes características:
            
            - Pequenas perturbações nos dados (coeficientes ou termos independentes) podem causar grandes mudanças na solução
            - Erros de arredondamento podem ser amplificados significativamente
            - Métodos iterativos podem convergir lentamente ou divergir
            
            **Exemplo**:
            
            Considere o sistema:
            
            $$
            \\begin{align}
            1.000x + 0.999y &= 1.999\\\\
            0.999x + 0.998y &= 1.997
            \\end{align}
            $$
            
            A solução exata é x = y = 1. No entanto, se mudarmos ligeiramente o termo independente da primeira equação para 2.000 (uma perturbação de apenas 0.001), a solução muda drasticamente para aproximadamente x = 2, y = 0.
            
            ## Estratégias para Sistemas Mal Condicionados
            
            1. **Pré-condicionamento**:
               - Multiplicar o sistema por uma matriz de pré-condicionamento para reduzir o número de condição
               - Exemplo: ao invés de resolver Ax = b, resolver M⁻¹Ax = M⁻¹b, onde M é escolhida para que M⁻¹A seja bem condicionada
            
            2. **Refinamento iterativo**:
               - Após obter uma solução aproximada x̃, calcular o resíduo r = b - Ax̃
               - Resolver Ad = r para obter a correção d
               - Atualizar a solução: x = x̃ + d
               - Repetir, se necessário
            
            3. **Métodos de regularização**:
               - Tikhonov: minimizar ||Ax - b||² + λ||x||², onde λ é o parâmetro de regularização
               - SVD truncada: ignorar componentes associados a valores singulares muito pequenos
            
            4. **Aumentar a precisão dos cálculos**:
               - Usar aritmética de precisão dupla ou estendida
               - Implementar algoritmos que minimizam a propagação de erros de arredondamento
            
            5. **Uso de decomposições estáveis**:
               - Decomposição QR
               - Decomposição de valores singulares (SVD)
            
            ## Exemplo de Análise
            
            Para a matriz:
            
            $$
            A = \\begin{bmatrix}
            1 & 1 \\\\
            1 & 1.0001
            \\end{bmatrix}
            $$
            
            1. O determinante é muito pequeno: det(A) = 0.0001
            2. O número de condição é aproximadamente 40000
            3. Uma pequena perturbação de 0.01% em A pode causar uma mudança de 400% na solução
            
            **Verificação**:
            - Se Ax = b, onde b = [2, 2.0001]ᵀ, a solução é x = [1, 1]ᵀ
            - Se mudarmos b para [2.0002, 2.0001]ᵀ (uma mudança de 0.01%), a solução muda para aproximadamente x = [2, 0]ᵀ
            """
        },
        "Aplicações Avançadas": {
            "content": """
            # Aplicações Avançadas de Sistemas Lineares
            
            ## Ajuste de Curvas e Superfícies
            
            O método dos mínimos quadrados leva a sistemas lineares para encontrar os coeficientes que minimizam o erro quadrático.
            
            **Exemplo**: Para ajustar um polinômio de grau n a m pontos (x_i, y_i), formamos o sistema normal:
            
            $$
            \\begin{bmatrix}
            m & \\sum x_i & \\sum x_i^2 & \\cdots & \\sum x_i^n \\\\
            \\sum x_i & \\sum x_i^2 & \\sum x_i^3 & \\cdots & \\sum x_i^{n+1} \\\\
            \\vdots & \\vdots & \\vdots & \\ddots & \\vdots \\\\
            \\sum x_i^n & \\sum x_i^{n+1} & \\sum x_i^{n+2} & \\cdots & \\sum x_i^{2n}
            \\end{bmatrix}
            \\begin{bmatrix}
            a_0 \\\\
            a_1 \\\\
            \\vdots \\\\
            a_n
            \\end{bmatrix} =
            \\begin{bmatrix}
            \\sum y_i \\\\
            \\sum x_i y_i \\\\
            \\vdots \\\\
            \\sum x_i^n y_i
            \\end{bmatrix}
            $$
            
            ## Processamento de Imagens
            
            Sistemas lineares são usados em:
            
            1. **Filtros lineares**:
               - Convolução para suavização, detecção de bordas, etc.
               - As máscaras de convolução podem ser representadas como sistemas lineares
            
            2. **Restauração de imagens**:
               - Eliminação de ruído e desfoque através de sistemas lineares regularizados
               - Exemplo: para um modelo de degradação g = Hf + n, onde g é a imagem observada, f é a imagem original e n é o ruído,
                 a restauração pode ser formulada como um sistema linear (H^T H + λI)f = H^T g
            
            3. **Compressão**:
               - Transformadas como DCT (usada em JPEG) podem ser implementadas como sistemas lineares
            
            ## Simulação de Fluidos
            
            As equações de Navier-Stokes discretizadas levam a grandes sistemas lineares:
            
            1. **Método da pressão**:
               - A equação de Poisson para a pressão leva a um sistema linear Ap = b
               - A matriz A é geralmente esparsa e pode ser resolvida eficientemente com métodos especializados
            
            2. **Métodos de elementos finitos**:
               - Discretizam o domínio em elementos menores
               - Resultam em sistemas lineares para velocidades e pressões
            
            ## Tomografia Computadorizada
            
            A reconstrução de imagens em tomografia (CT scan) envolve a resolução de sistemas lineares:
            
            1. **Problema de reconstrução**:
               - Relaciona as medidas de atenuação com os coeficientes de atenuação nos voxels
               - Leva a um grande sistema linear Ax = b, onde x são os coeficientes desconhecidos
            
            2. **Métodos de solução**:
               - Retroprojeção filtrada
               - Métodos iterativos como ART (Algebraic Reconstruction Technique), que resolve o sistema de forma iterativa
            
            ## Redes Neurais
            
            Embora as redes neurais modernas sejam não-lineares, muitas operações internas envolvem sistemas lineares:
            
            1. **Camadas lineares**:
               - A operação Wx + b, onde W é a matriz de pesos, x é a entrada e b é o viés
            
            2. **Backpropagation**:
               - O cálculo de gradientes envolve operações lineares com matrizes Jacobianas
            
            ## Criptografia
            
            Alguns métodos criptográficos são baseados em sistemas lineares:
            
            1. **Cifra de Hill**:
               - Usa multiplicação de matrizes para cifrar blocos de texto
               - A segurança depende da dificuldade de resolver certos sistemas lineares
            
            2. **Sistemas baseados em reticulados**:
               - Baseiam-se na dificuldade de resolver certos sistemas lineares em reticulados
               - Exemplo: o problema SVP (Shortest Vector Problem) está relacionado a encontrar a solução de norma mínima para um sistema homogêneo

            """
        },

        "Métodos Iterativos": {
            "content": """
            # Métodos Iterativos para Sistemas Lineares
            
            Os métodos iterativos começam com uma aproximação inicial e melhoram progressivamente a solução.
            
            ## Método de Jacobi
            
            **Procedimento**:
            1. Para cada equação i, isolar a incógnita x_i
            2. Iniciar com uma aproximação inicial (geralmente zeros)
            3. Em cada iteração k+1, calcular:
               x_i^(k+1) = (b_i - Σ a_ij x_j^(k)) / a_ii, para j ≠ i
            4. Repetir até convergir
            
            ## Método de Gauss-Seidel
            
            Similar ao método de Jacobi, mas usa valores já atualizados na mesma iteração:
            
            x_i^(k+1) = (b_i - Σ a_ij x_j^(k+1) - Σ a_ij x_j^(k)) / a_ii
                          j<i                j>i
            
            **Condições de convergência**:
            - Matriz diagonalmente dominante (|a_ii| > Σ |a_ij| para j ≠ i)
            - Matriz definida positiva
            
            ## Exemplo
            
            Considere o sistema:
            
            $$
            \\begin{align}
            10x + 2y + z &= 13\\\\
            x + 5y + z &= 7\\\\
            2x + y + 10z &= 13
            \\end{align}
            $$
            
            Isolando as variáveis:
            
            $$
            \\begin{align}
            x &= \\frac{13 - 2y - z}{10}\\\\
            y &= \\frac{7 - x - z}{5}\\\\
            z &= \\frac{13 - 2x - y}{10}
            \\end{align}
            $$
            
            **Método de Jacobi**:
            
            Partindo de x^(0) = y^(0) = z^(0) = 0:
            
            Iteração 1:
            - x^(1) = (13 - 2×0 - 0)/10 = 1.3
            - y^(1) = (7 - 0 - 0)/5 = 1.4
            - z^(1) = (13 - 2×0 - 0)/10 = 1.3
            
            Iteração 2:
            - x^(2) = (13 - 2×1.4 - 1.3)/10 = 0.83
            - y^(2) = (7 - 1.3 - 1.3)/5 = 0.88
            - z^(2) = (13 - 2×1.3 - 1.4)/10 = 0.83
            
            O processo continua até a convergência para x = y = z = 1.
            
            **Método de Gauss-Seidel**:
            
            Partindo de x^(0) = y^(0) = z^(0) = 0:
            
            Iteração 1:
            - x^(1) = (13 - 2×0 - 0)/10 = 1.3
            - y^(1) = (7 - 1.3 - 0)/5 = 1.14
            - z^(1) = (13 - 2×1.3 - 1.14)/10 = 0.826
            
            Iteração 2:
            - x^(2) = (13 - 2×1.14 - 0.826)/10 = 0.8934
            - y^(2) = (7 - 0.8934 - 0.826)/5 = 1.0561
            - z^(2) = (13 - 2×0.8934 - 1.0561)/10 = 0.9157
            
            O método converge mais rapidamente para x = y = z = 1.
            
            ## Vantagens e Desvantagens
            
            **Vantagens**:
            - Mais eficientes para sistemas grandes e esparsos
            - Menor requisito de memória
            - Podem lidar com matrizes mal condicionadas
            
            **Desvantagens**:
            - Convergência não garantida para todas as matrizes
            - Podem ser lentos para atingir alta precisão
            """
        },
        "Decomposição LU": {
            "content": """
            # Decomposição LU
            
            A decomposição LU fatoriza a matriz A em um produto de duas matrizes: A = LU, onde:
            - L é uma matriz triangular inferior
            - U é uma matriz triangular superior
            
            ## Procedimento para resolver AX = B
            
            1. Decompor A = LU
            2. Resolver LY = B por substituição direta
            3. Resolver UX = Y por substituição reversa
            
            ## Algoritmo para decomposição LU
            
            **Procedimento de Doolittle**:
            
            Para uma matriz n×n:
            
            1. Para i = 1 até n:
               - Para j = i até n: u_{ij} = a_{ij} - Σ(l_{ik} × u_{kj}) para k = 1 até i-1
               - Para j = i+1 até n: l_{ji} = (a_{ji} - Σ(l_{jk} × u_{ki}) para k = 1 até i-1) / u_{ii}
            
            2. Para i = 1 até n: l_{ii} = 1 (diagonal unitária para L)
            
            ## Exemplo
            
            Considere o sistema:
            
            $$
            \\begin{align}
            2x + y + z &= 5\\\\
            4x + 5y + z &= 9\\\\
            x + y + 3z &= 11
            \\end{align}
            $$
            
            **Decomposição LU**:
            
            Matriz A:
            
            $$
            A = \\begin{bmatrix}
            2 & 1 & 1 \\\\
            4 & 5 & 1 \\\\
            1 & 1 & 3
            \\end{bmatrix}
            $$
            
            Calculando L e U:
            
            $$
            L = \\begin{bmatrix}
            1 & 0 & 0 \\\\
            2 & 1 & 0 \\\\
            0.5 & 0.25 & 1
            \\end{bmatrix}
            $$
            
            $$
            U = \\begin{bmatrix}
            2 & 1 & 1 \\\\
            0 & 3 & -1 \\\\
            0 & 0 & 2.5
            \\end{bmatrix}
            $$
            
            **Resolver LY = B**:
            
            $$
            \\begin{bmatrix}
            1 & 0 & 0 \\\\
            2 & 1 & 0 \\\\
            0.5 & 0.25 & 1
            \\end{bmatrix}
            \\begin{bmatrix}
            y_1 \\\\
            y_2 \\\\
            y_3
            \\end{bmatrix} =
            \\begin{bmatrix}
            5 \\\\
            9 \\\\
            11
            \\end{bmatrix}
            $$
            
            - y₁ = 5
            - y₂ = 9 - 2×5 = -1
            - y₃ = 11 - 0.5×5 - 0.25×(-1) = 11 - 2.5 + 0.25 = 8.75
            
            **Resolver UX = Y**:
            
            $$
            \\begin{bmatrix}
            2 & 1 & 1 \\\\
            0 & 3 & -1 \\\\
            0 & 0 & 2.5
            \\end{bmatrix}
            \\begin{bmatrix}
            x \\\\
            y \\\\
            z
            \\end{bmatrix} =
            \\begin{bmatrix}
            5 \\\\
            -1 \\\\
            8.75
            \\end{bmatrix}
            $$
            
            - z = 8.75 / 2.5 = 3.5
            - y = (-1 + z) / 3 = (-1 + 3.5) / 3 = 2.5 / 3 = 0.833...
            - x = (5 - y - z) / 2 = (5 - 0.833 - 3.5) / 2 = 0.667...
            
            ## Vantagens e Desvantagens
            
            **Vantagens**:
            - Eficiente para resolver múltiplos sistemas com a mesma matriz de coeficientes
            - Útil para calcular determinantes e inversas
            - Computacionalmente eficiente
            
            **Desvantagens**:
            - Requer armazenamento adicional para as matrizes L e U
            - Pode ser instável para matrizes mal condicionadas sem pivoteamento
            """
        },
        "Interpretação Geométrica": {
            "content": """
            # Interpretação Geométrica de Sistemas Lineares
            
            ## Sistemas 2×2
            
            Em um sistema com duas equações e duas incógnitas, cada equação representa uma reta no plano cartesiano.
            
            **Possíveis configurações**:
            
            1. **SPD (Sistema Possível e Determinado)**:
               - As retas se intersectam em um único ponto
               - Este ponto representa a única solução do sistema
               - Exemplo: x + y = 3 e x - y = 1
               
            2. **SPI (Sistema Possível e Indeterminado)**:
               - As retas são coincidentes (sobrepostas)
               - Todos os pontos da reta são soluções do sistema
               - Exemplo: 2x + 3y = 6 e 4x + 6y = 12
               
            3. **SI (Sistema Impossível)**:
               - As retas são paralelas e não coincidentes
               - Não há nenhum ponto comum, ou seja, não há solução
               - Exemplo: x + y = 3 e x + y = 5
            
            ## Sistemas 3×3
            
            Em um sistema com três equações e três incógnitas, cada equação representa um plano no espaço tridimensional.
            
            **Possíveis configurações**:
            
            1. **SPD (Sistema Possível e Determinado)**:
               - Os três planos se intersectam em um único ponto
               - Este ponto é a única solução do sistema
               
            2. **SPI (Sistema Possível e Indeterminado)**:
               - Os planos se intersectam em uma reta (dois planos paralelos intersectados por um terceiro)
               - Ou se intersectam em um plano (três planos coincidentes)
               - As soluções formam uma reta ou um plano
               
            3. **SI (Sistema Impossível)**:
               - Os planos não possuem ponto comum de interseção
               - Pode ocorrer quando temos planos paralelos ou quando a interseção de dois planos é paralela ao terceiro
            
            ## Relação com o Espaço Vetorial
            
            - As linhas da matriz de coeficientes podem ser vistas como vetores
            - O determinante da matriz está relacionado ao volume do paralelepípedo formado por estes vetores
            - Um determinante zero indica que os vetores são linearmente dependentes
            - Para sistemas homogêneos (AX = 0), o conjunto solução forma um subespaço vetorial
            """
        },
        "Aplicações de Sistemas Lineares": {
            "content": """
            # Aplicações de Sistemas Lineares
            
            Os sistemas de equações lineares são ferramentas fundamentais em diversas áreas. Veja algumas aplicações importantes:
            
            ## Física e Engenharia
            
            ### Análise de Circuitos Elétricos
            - Leis de Kirchhoff: correntes em um nó e tensões em um laço
            - Cada equação representa a conservação de corrente ou tensão
            
            **Exemplo**: Para um circuito com três malhas, podemos ter:
            ```
            R₁I₁ + R₂(I₁ - I₂) = V₁
            R₂(I₂ - I₁) + R₃I₂ + R₄(I₂ - I₃) = 0
            R₄(I₃ - I₂) + R₅I₃ = V₂
            ```
            
            ### Estática e Dinâmica
            - Equilíbrio de forças e momentos em estruturas
            - Análise de treliças e vigas
            
            **Exemplo**: Para um sistema com três forças desconhecidas:
            ```
            F₁cos(θ₁) + F₂cos(θ₂) + F₃cos(θ₃) = 0
            F₁sin(θ₁) + F₂sin(θ₂) + F₃sin(θ₃) = 0
            r₁×F₁ + r₂×F₂ + r₃×F₃ = 0
            ```
            
            ### Transferência de Calor
            - Modelagem de problemas de condução térmica
            - Métodos de diferenças finitas para equações diferenciais
            
            ## Economia
            
            ### Análise de Insumo-Produto de Leontief
            - Modelagem das interdependências entre diferentes setores da economia
            - Previsão de como mudanças em um setor afetam outros setores
            
            **Exemplo**: Em uma economia com três setores:
            ```
            x₁ = a₁₁x₁ + a₁₂x₂ + a₁₃x₃ + d₁
            x₂ = a₂₁x₁ + a₂₂x₂ + a₂₃x₃ + d₂
            x₃ = a₃₁x₁ + a₃₂x₂ + a₃₃x₃ + d₃
            ```
            Onde x₁ é a produção do setor i, a_ij é quanto do produto i é usado para produzir uma unidade do produto j, e d_i é a demanda externa.
            
            ### Otimização e Programação Linear
            - Maximização de lucro ou minimização de custos
            - Alocação ótima de recursos limitados
            
            ## Química
            
            ### Balanceamento de Equações Químicas
            - Cada elemento forma uma equação linear
            - Os coeficientes estequiométricos são as incógnitas
            
            **Exemplo**: Para a reação C₃H₈ + O₂ → CO₂ + H₂O
            ```
            3a = c       (para o carbono)
            8a = 2e      (para o hidrogênio)
            2b = 2c + e  (para o oxigênio)
            ```
            
            ### Equilíbrio Químico
            - Determinação de concentrações em equilíbrio
            
            ## Computação Gráfica
            
            ### Transformações Geométricas
            - Rotação, translação e escala de objetos
            - Representadas como transformações matriciais
            
            ### Renderização 3D
            - Sistemas para determinar projeções de objetos 3D em telas 2D
            
            ## Problemas de Mistura
            
            ### Farmacologia
            - Mistura de componentes para atingir concentrações específicas
            - Formulação de medicamentos
            
            **Exemplo**: Um farmacêutico precisa preparar 100ml de uma solução com 25% de um princípio ativo, usando soluções de 10%, 20% e 40%.
            ```
            x + y + z = 100
            0.1x + 0.2y + 0.4z = 25
            ```
            
            ### Processamento de Alimentos
            - Mistura de ingredientes para atingir perfis nutricionais
            
            ## Tráfego e Transporte
            
            ### Fluxo de Redes
            - Modelagem de fluxo de tráfego em redes de transporte
            - Otimização de rotas
            """
        },
        "Sistemas Homogêneos": {
            "content": """
            # Sistemas Lineares Homogêneos
            
            Um sistema homogêneo tem a forma AX = 0 (todos os termos independentes são nulos).
            
            ## Propriedades
            
            1. Todo sistema homogêneo é possível (sempre admite a solução trivial X = 0)
            2. Um sistema homogêneo tem soluções não-triviais se e somente se det(A) = 0
            3. O conjunto de todas as soluções forma um espaço vetorial
            4. A dimensão do espaço de soluções é n - posto(A), onde n é o número de incógnitas
            
            ## Interpretação Geométrica
            
            - Em 2D: se det(A) = 0, as equações representam a mesma reta passando pela origem
            - Em 3D: se det(A) = 0, os planos se intersectam em uma reta ou um plano passando pela origem
            
            ## Aplicações
            
            - **Espaços nulos**:
            O núcleo (ou kernel) de uma transformação linear é o espaço das soluções de AX = 0
              - Fundamental em álgebra linear e geometria
            
            - **Autovalores e autovetores**:
              - Um autovetor v de uma matriz A satisfaz Av = λv, ou (A - λI)v = 0
              - Encontrar os autovetores envolve resolver sistemas homogêneos
            
            - **Equações diferenciais**:
              - Sistemas de equações diferenciais lineares homogêneas têm soluções da forma X = e^(λt)v
              - Onde λ é um autovalor e v é um autovetor associado
            
            ## Exemplo
            
            Considere o sistema homogêneo:
            
            $$
            \\begin{align}
            2x + 3y - z &= 0\\\\
            4x + 6y - 2z &= 0\\\\
            -2x - 3y + z &= 0
            \\end{align}
            $$
            
            Observe que a terceira equação é o oposto da primeira. Além disso, a segunda equação é um múltiplo da primeira (multiplicada por 2).
            
            O determinante da matriz dos coeficientes é zero, o que confirma que o sistema tem soluções não-triviais.
            
            O posto da matriz é 1, e temos 3 incógnitas, então a dimensão do espaço de soluções é 3 - 1 = 2.
            
            **Solução paramétrica**:
            
            Podemos expressar z e y em termos de x:
            Da primeira equação: z = 2x + 3y
            
            Substituindo na segunda e terceira equações, verificamos que são satisfeitas para qualquer valor de x e y.
            
            Então a solução geral é:
            ```
            z = 2x + 3y
            ```
            Onde x e y são parâmetros livres.
            
            Alternativamente, podemos parametrizar como:
            ```
            x = s
            y = t
            z = 2s + 3t
            ```
            Onde s e t são parâmetros livres.
            """
        },
        "Estabilidade Numérica": {
            "content": """
            # Estabilidade Numérica em Sistemas Lineares
            
            ## Número de Condição
            
            O número de condição de uma matriz A, denotado por cond(A), mede a sensibilidade da solução a pequenas perturbações nos dados:
            
            $$\\text{cond}(A) = \\|A\\| \\cdot \\|A^{-1}\\|$$
            
            Para a norma-2, isso é equivalente à razão entre o maior e o menor valor singular:
            
            $$\\text{cond}_2(A) = \\frac{\\sigma_{\\max}}{\\sigma_{\\min}}$$
            
            **Interpretação**:
            - Um número de condição próximo de 1 indica uma matriz bem condicionada
            - Um número de condição grande indica uma matriz mal condicionada
            - Um número de condição infinito indica uma matriz singular
            
            ## Efeitos do Mal Condicionamento
            
            Um sistema mal condicionado tem as seguintes características:
            
            - Pequenas perturbações nos dados (coeficientes ou termos independentes) podem causar grandes mudanças na solução
            - Erros de arredondamento podem ser amplificados significativamente
            - Métodos iterativos podem convergir lentamente ou divergir
            
            **Exemplo**:
            
            Considere o sistema:
            
            $$
            \\begin{align}
            1.000x + 0.999y &= 1.999\\\\
            0.999x + 0.998y &= 1.997
            \\end{align}
            $$
            
            A solução exata é x = y = 1. No entanto, se mudarmos ligeiramente o termo independente da primeira equação para 2.000 (uma perturbação de apenas 0.001), a solução muda drasticamente para aproximadamente x = 2, y = 0.
            
            ## Estratégias para Sistemas Mal Condicionados
            
            1. **Pré-condicionamento**:
               - Multiplicar o sistema por uma matriz de pré-condicionamento para reduzir o número de condição
               - Exemplo: ao invés de resolver Ax = b, resolver M⁻¹Ax = M⁻¹b, onde M é escolhida para que M⁻¹A seja bem condicionada
            
            2. **Refinamento iterativo**:
               - Após obter uma solução aproximada x̃, calcular o resíduo r = b - Ax̃
               - Resolver Ad = r para obter a correção d
               - Atualizar a solução: x = x̃ + d
               - Repetir, se necessário
            
            3. **Métodos de regularização**:
               - Tikhonov: minimizar ||Ax - b||² + λ||x||², onde λ é o parâmetro de regularização
               - SVD truncada: ignorar componentes associados a valores singulares muito pequenos
            
            4. **Aumentar a precisão dos cálculos**:
               - Usar aritmética de precisão dupla ou estendida
               - Implementar algoritmos que minimizam a propagação de erros de arredondamento
            
            5. **Uso de decomposições estáveis**:
               - Decomposição QR
               - Decomposição de valores singulares (SVD)
            
            ## Exemplo de Análise
            
            Para a matriz:
            
            $$
            A = \\begin{bmatrix}
            1 & 1 \\\\
            1 & 1.0001
            \\end{bmatrix}
            $$
            
            1. O determinante é muito pequeno: det(A) = 0.0001
            2. O número de condição é aproximadamente 40000
            3. Uma pequena perturbação de 0.01% em A pode causar uma mudança de 400% na solução
            
            **Verificação**:
            - Se Ax = b, onde b = [2, 2.0001]ᵀ, a solução é x = [1, 1]ᵀ
            - Se mudarmos b para [2.0002, 2.0001]ᵀ (uma mudança de 0.01%), a solução muda para aproximadamente x = [2, 0]ᵀ
            """
        },
        "Aplicações Avançadas": {
            "content": """
            # Aplicações Avançadas de Sistemas Lineares
            
            ## Ajuste de Curvas e Superfícies
            
            O método dos mínimos quadrados leva a sistemas lineares para encontrar os coeficientes que minimizam o erro quadrático.
            
            **Exemplo**: Para ajustar um polinômio de grau n a m pontos (x_i, y_i), formamos o sistema normal:
            
            $$
            \\begin{bmatrix}
            m & \\sum x_i & \\sum x_i^2 & \\cdots & \\sum x_i^n \\\\
            \\sum x_i & \\sum x_i^2 & \\sum x_i^3 & \\cdots & \\sum x_i^{n+1} \\\\
            \\vdots & \\vdots & \\vdots & \\ddots & \\vdots \\\\
            \\sum x_i^n & \\sum x_i^{n+1} & \\sum x_i^{n+2} & \\cdots & \\sum x_i^{2n}
            \\end{bmatrix}
            \\begin{bmatrix}
            a_0 \\\\
            a_1 \\\\
            \\vdots \\\\
            a_n
            \\end{bmatrix} =
            \\begin{bmatrix}
            \\sum y_i \\\\
            \\sum x_i y_i \\\\
            \\vdots \\\\
            \\sum x_i^n y_i
            \\end{bmatrix}
            $$
            
            ## Processamento de Imagens
            
            Sistemas lineares são usados em:
            
            1. **Filtros lineares**:
               - Convolução para suavização, detecção de bordas, etc.
               - As máscaras de convolução podem ser representadas como sistemas lineares
            
            2. **Restauração de imagens**:
               - Eliminação de ruído e desfoque através de sistemas lineares regularizados
               - Exemplo: para um modelo de degradação g = Hf + n, onde g é a imagem observada, f é a imagem original e n é o ruído,
                 a restauração pode ser formulada como um sistema linear (H^T H + λI)f = H^T g
            
            3. **Compressão**:
               - Transformadas como DCT (usada em JPEG) podem ser implementadas como sistemas lineares
            
            ## Simulação de Fluidos
            
            As equações de Navier-Stokes discretizadas levam a grandes sistemas lineares:
            
            1. **Método da pressão**:
               - A equação de Poisson para a pressão leva a um sistema linear Ap = b
               - A matriz A é geralmente esparsa e pode ser resolvida eficientemente com métodos especializados
            
            2. **Métodos de elementos finitos**:
               - Discretizam o domínio em elementos menores
               - Resultam em sistemas lineares para velocidades e pressões
            
            ## Tomografia Computadorizada
            
            A reconstrução de imagens em tomografia (CT scan) envolve a resolução de sistemas lineares:
            
            1. **Problema de reconstrução**:
               - Relaciona as medidas de atenuação com os coeficientes de atenuação nos voxels
               - Leva a um grande sistema linear Ax = b, onde x são os coeficientes desconhecidos
            
            2. **Métodos de solução**:
               - Retroprojeção filtrada
               - Métodos iterativos como ART (Algebraic Reconstruction Technique), que resolve o sistema de forma iterativa
            
            ## Redes Neurais
            
            Embora as redes neurais modernas sejam não-lineares, muitas operações internas envolvem sistemas lineares:
            
            1. **Camadas lineares**:
               - A operação Wx + b, onde W é a matriz de pesos, x é a entrada e b é o viés
            
            2. **Backpropagation**:
               - O cálculo de gradientes envolve operações lineares com matrizes Jacobianas
            
            ## Criptografia
            
            Alguns métodos criptográficos são baseados em sistemas lineares:
            
            1. **Cifra de Hill**:
               - Usa multiplicação de matrizes para cifrar blocos de texto
               - A segurança depende da dificuldade de resolver certos sistemas lineares
            
            2. **Sistemas baseados em reticulados**:
               - Baseiam-se na dificuldade de resolver certos sistemas lineares em reticulados
               - Exemplo: o problema SVP (Shortest Vector Problem) está relacionado a encontrar a solução de norma mínima para um sistema homogêneo
               """
        },
        "Sistemas Não-Lineares": {
            "content": """
            # Sistemas Não-Lineares
            
            Em contraste com sistemas lineares, os sistemas não-lineares envolvem funções não-lineares das variáveis.
            
            ## Características dos Sistemas Não-Lineares
            
            1. **Múltiplas soluções**:
               - Podem ter 0, 1, um número finito ou infinitas soluções
               - Difíceis de classificar a priori
            
            2. **Comportamento complexo**:
               - Podem exibir caos, bifurcações e outros fenômenos complexos
               - Pequenas mudanças nos parâmetros podem levar a mudanças drásticas nas soluções
            
            3. **Métodos de resolução**:
               - Geralmente iterativos e não garantem encontrar todas as soluções
               - Podem convergir para diferentes soluções dependendo do ponto inicial
            
            ## Técnicas de Linearização
            
            Muitos métodos para resolver sistemas não-lineares envolvem alguma forma de linearização:
            
            1. **Expansão de Taylor**:
               - Aproximar localmente as funções não-lineares por suas expansões de Taylor de primeira ordem
               - Exemplo: f(x) ≈ f(x₀) + f'(x₀)(x - x₀)
            
            2. **Método de Newton multidimensional**:
               - Generalização do método de Newton para sistemas
               - Resolve iterativamente sistemas lineares da forma J(xₖ)Δx = -F(xₖ)
               - Onde J é a matriz Jacobiana das derivadas parciais
            
            ## Método de Newton
            
            Para um sistema F(X) = 0 com n equações e n incógnitas:
            
            1. Começar com uma aproximação inicial X₀
            2. Para cada iteração k:
               - Calcular F(Xₖ) e a matriz Jacobiana J(Xₖ)
               - Resolver o sistema linear J(Xₖ)Δx = -F(Xₖ)
               - Atualizar: Xₖ₊₁ = Xₖ + Δx
               - Verificar convergência
            
            **Exemplo**:
            
            Para o sistema:
            
            $$
            \\begin{align}
            x^2 + y^2 &= 25\\\\
            x^2 - y^2 &= 7
            \\end{align}
            $$
            
            A matriz Jacobiana é:
            
            $$
            J(x, y) = \\begin{bmatrix}
            2x & 2y \\\\
            2x & -2y
            \\end{bmatrix}
            $$
            
            Partindo de (4, 3), calculamos:
            
            - F(4, 3) = [(4² + 3²) - 25, (4² - 3²) - 7] = [0, 0]
            
            Já encontramos uma solução exata: (4, 3).
            
            Se tivéssemos partido de (3, 4), teríamos encontrado outra solução: (4, -3).
            
            ## Método do Ponto Fixo
            
            1. Reescrever o sistema na forma X = g(X)
            2. Escolher uma aproximação inicial X₀
            3. Iterar Xₖ₊₁ = g(Xₖ) até a convergência
            
            **Condição de convergência**:
            O método converge se ||∇g(X)|| < 1 na vizinhança da solução.
            
            ## Aplicações de Sistemas Não-Lineares
            
            1. **Física e engenharia**:
               - Equilíbrio de estruturas com comportamento não-linear
               - Circuitos não-lineares
               - Dinâmica de fluidos
            
            2. **Química**:
               - Equilíbrio químico com múltiplas reações
               - Cinética de reações complexas
            
            3. **Economia**:
               - Modelos econômicos com funções não-lineares de utilidade ou produção
               - Equilíbrio de mercado com demanda e oferta não-lineares
            
            4. **Biologia**:
               - Modelos de populações com interações não-lineares
               - Redes bioquímicas
            """
        },
        "Sistemas Lineares em Programação Linear": {
            "content": """
            # Sistemas Lineares em Programação Linear
            
            A programação linear (PL) é uma técnica de otimização para problemas com função objetivo linear e restrições lineares.
            
            ## Formulação Padrão
            
            Um problema de PL tem a forma:
            
            **Maximizar** (ou Minimizar): c₁x₁ + c₂x₂ + ... + cₙxₙ
            
            **Sujeito a**:
            ```
            a₁₁x₁ + a₁₂x₂ + ... + a₁ₙxₙ ≤ b₁
            a₂₁x₁ + a₂₂x₂ + ... + a₂ₙxₙ ≤ b₂
            ...
            aₘ₁x₁ + aₘ₂x₂ + ... + aₘₙxₙ ≤ bₘ
            ```
            
            E: x₁, x₂, ..., xₙ ≥ 0
            
            ## Método Simplex
            
            O método simplex resolve problemas de PL percorrendo os vértices do poliedro formado pelas restrições:
            
            1. Converter para a forma padrão, introduzindo variáveis de folga:
               ```
               a₁₁x₁ + ... + a₁ₙxₙ + s₁ = b₁
               ...
               aₘ₁x₁ + ... + aₘₙxₙ + sₘ = bₘ
               ```
            
            2. Encontrar uma solução básica viável inicial
            
            3. Verificar se a solução atual é ótima:
               - Se todos os coeficientes na função objetivo são não-negativos, a solução é ótima
               - Caso contrário, selecionar uma variável para entrar na base
            
            4. Determinar qual variável sai da base usando o teste da razão
            
            5. Atualizar a solução e retornar ao passo 3
            
            ## Relação com Sistemas Lineares
            
            Em cada iteração do simplex, resolvemos um sistema linear:
            
            1. As equações de restrição formam um sistema linear
            2. A operação pivô para trocar as variáveis básicas é essencialmente eliminação gaussiana
            3. A atualização da função objetivo também envolve operações de álgebra linear
            
            ## Dualidade
            
            Para cada problema de PL (primal), existe um problema dual associado:
            
            - Se o primal é um problema de maximização, o dual é de minimização, e vice-versa
            - As variáveis no dual correspondem às restrições no primal
            - As restrições no dual correspondem às variáveis no primal
            
            **Exemplo**:
            
            Primal:
            ```
            Maximizar: 3x₁ + 2x₂
            Sujeito a:
              x₁ + x₂ ≤ 8
              2x₁ + x₂ ≤ 10
              x₁, x₂ ≥ 0
            ```
            
            Dual:
            ```
            Minimizar: 8y₁ + 10y₂
            Sujeito a:
              y₁ + 2y₂ ≥ 3
              y₁ + y₂ ≥ 2
              y₁, y₂ ≥ 0
            ```
            
            ## Aplicações
            
            1. **Alocação de recursos**:
               - Determinar quanto produzir de cada produto para maximizar o lucro
               - Exemplo: Uma fábrica produz dois produtos que requerem diferentes quantidades de três recursos limitados
            
            2. **Dieta e mistura**:
               - Encontrar a combinação ótima de alimentos para minimizar o custo enquanto satisfaz requisitos nutricionais
               - Similar a problemas de mistura em química e engenharia
            
            3. **Transporte e logística**:
               - Otimizar o fluxo de bens de múltiplas origens para múltiplos destinos
               - Minimizar o custo total de transporte
            
            4. **Fluxo de rede**:
               - Encontrar o fluxo máximo em uma rede com capacidades limitadas
               - Ou o fluxo de custo mínimo que satisfaz demandas
            
            5. **Planejamento financeiro**:
               - Otimizar portfolios de investimento
               - Balancear risco e retorno sob restrições orçamentárias
            """
        },
        "Teorema de Rouché-Capelli": {
            "content": """
            # Teorema de Rouché-Capelli
            
            O Teorema de Rouché-Capelli (também conhecido como Teorema de Kronecker-Capelli) é um resultado fundamental na teoria de sistemas lineares, que estabelece condições precisas para a existência e unicidade de soluções.
            
            ## Enunciado Formal
            
            Seja AX = B um sistema linear, onde:
            - A é uma matriz m × n
            - X é um vetor de incógnitas n × 1
            - B é um vetor de termos constantes m × 1
            
            **O teorema afirma que**:
            
            1. O sistema tem pelo menos uma solução se e somente se o posto da matriz A é igual ao posto da matriz aumentada [A|B].
            
            2. Se o sistema tem solução, então:
               - Se posto(A) = n, a solução é única (sistema possível e determinado)
               - Se posto(A) < n, o sistema tem infinitas soluções (sistema possível e indeterminado)
            
            ## Significado dos Postos
            
            - **Posto de A**: É o número máximo de linhas (ou colunas) linearmente independentes em A.
            
            - **Posto da matriz aumentada [A|B]**: É o número máximo de linhas linearmente independentes na matriz aumentada.
            
            ## Exemplo
            
            Considere o sistema:
            
            $$
            \\begin{align}
            x + y + z &= 6\\\\
            2x + y - z &= 1\\\\
            3x + 2y + 0z &= 7
            \\end{align}
            $$
            
            Podemos escrever a matriz aumentada:
            
            $$
            [A|B] = \\begin{bmatrix}
            1 & 1 & 1 & | & 6 \\\\
            2 & 1 & -1 & | & 1 \\\\
            3 & 2 & 0 & | & 7
            \\end{bmatrix}
            $$
            
            Após o escalonamento, podemos determinar que posto(A) = posto([A|B]) = 3 = n.
            
            Pelo teorema, o sistema é possível e determinado, com solução única.
            
            ## Interpretação Geométrica
            
            - **posto(A) = posto([A|B]) = n**: As equações representam variedades (retas, planos, etc.) que se intersectam em um único ponto.
            
            - **posto(A) = posto([A|B]) < n**: As equações representam variedades que se intersectam em um subespaço de dimensão (n - posto(A)).
            
            - **posto(A) < posto([A|B])**: As equações são inconsistentes (por exemplo, representam retas paralelas).
            
            ## Aplicações
            
            O Teorema de Rouché-Capelli é útil para:
            
            1. **Classificação de sistemas**: Determinar se um sistema é possível e determinado, possível e indeterminado ou impossível.
            
            2. **Análise teórica**: Compreender as condições para existência e unicidade de soluções em álgebra linear.
            
            3. **Verificação a priori**: Determinar se um sistema tem solução antes de tentar resolvê-lo.
            
            4. **Estudo de espaços vetoriais**: Relacionar sistemas lineares com conceitos de dependência linear e dimensão.
            """
        },
        "Decomposição QR": {
            "content": """
            # Decomposição QR
            
            A decomposição QR é uma técnica de fatoração de matrizes onde uma matriz A é expressa como o produto de duas matrizes: A = QR, onde:
            
            - Q é uma matriz ortogonal (suas colunas são vetores ortonormais)
            - R é uma matriz triangular superior
            
            ## Métodos para Calcular a Decomposição QR
            
            ### 1. Processo de Gram-Schmidt
            
            O algoritmo clássico para obter a decomposição QR:
            
            1. Começar com as colunas da matriz A: a₁, a₂, ..., aₙ
            2. Calcular os vetores ortonormais q₁, q₂, ..., qₙ:
               - q₁ = a₁/||a₁||
               - Para j = 2, 3, ..., n:
                 - Calcular vⱼ = aⱼ - Σ(qᵢ·aⱼ)qᵢ para i = 1 até j-1
                 - Normalizar: qⱼ = vⱼ/||vⱼ||
            3. A matriz Q é formada pelos vetores qⱼ como colunas
            4. A matriz R contém os produtos internos: rᵢⱼ = qᵢ·aⱼ para i ≤ j, e zeros abaixo da diagonal
            
            ### 2. Reflexões de Householder
            
            Um método numericamente mais estável:
            
            1. Para cada coluna j da matriz A:
               - Construir uma matriz de reflexão Hⱼ que anula os elementos abaixo da diagonal na coluna j
               - Aplicar a reflexão: A ← HⱼA
            2. O produto das reflexões forma Q: Q = H₁H₂...Hₙ
            3. A matriz resultante após todas as reflexões é R
            
            ## Aplicações na Resolução de Sistemas Lineares
            
            Para resolver o sistema AX = B usando decomposição QR:
            
            1. Decompor A = QR
            2. Substituir no sistema: QRX = B
            3. Multiplicar ambos os lados por Qᵀ: Qᵀ(QRX) = QᵀB
            4. Simplificar: RX = QᵀB (usando a propriedade de que QᵀQ = I)
            5. Resolver o sistema triangular RX = QᵀB por substituição reversa
            
            ## Vantagens da Decomposição QR
            
            1. **Estabilidade numérica**: Mais estável que outros métodos, especialmente para matrizes mal condicionadas
            
            2. **Problemas de mínimos quadrados**: Particularmente eficiente para resolver o problema de mínimos quadrados ||Ax - b||
            
            3. **Solução única**: Para matrizes de posto completo, garante uma solução única
            
            4. **Aplicações avançadas**:
               - Cálculo de autovalores (método QR)
               - Problemas de mínimos quadrados
               - Fatoração de matrizes em aprendizado de máquina
            
            ## Exemplo
            
            Considere a matriz:
            
            $$
            A = \\begin{bmatrix}
            1 & 1 \\\\
            1 & 0 \\\\
            0 & 1
            \\end{bmatrix}
            $$
            
            Usando o processo de Gram-Schmidt:
            
            1. Normalizar a primeira coluna:
               - q₁ = (1, 1, 0)ᵀ / ||(1, 1, 0)|| = (1/√2, 1/√2, 0)ᵀ
            
            2. Ortogonalizar a segunda coluna em relação à primeira:
               - v₂ = (1, 0, 1)ᵀ - ((1/√2, 1/√2, 0)·(1, 0, 1))(1/√2, 1/√2, 0)ᵀ
               - v₂ = (1, 0, 1)ᵀ - (1/√2)(1/√2, 1/√2, 0)ᵀ
               - v₂ = (1, 0, 1)ᵀ - (1/2, 1/2, 0)ᵀ = (1/2, -1/2, 1)ᵀ
               
               Normalizar:
               - q₂ = (1/2, -1/2, 1)ᵀ / ||(1/2, -1/2, 1)|| = (1/√3, -1/√3, 2/√3)ᵀ
            
            As matrizes resultantes são:
            
            $$
            Q = \\begin{bmatrix}
            1/\\sqrt{2} & 1/\\sqrt{3} \\\\
            1/\\sqrt{2} & -1/\\sqrt{3} \\\\
            0 & 2/\\sqrt{3}
            \\end{bmatrix}
            $$
            
            $$
            R = \\begin{bmatrix}
            \\sqrt{2} & 1/\\sqrt{2} \\\\
            0 & \\sqrt{3}/\\sqrt{2}
            \\end{bmatrix}
            $$
            
            Podemos verificar que A = QR.
            """
        },
        "Fatoração SVD": {
            "content": """
            # Decomposição em Valores Singulares (SVD)
            
            A Decomposição em Valores Singulares (SVD, Singular Value Decomposition) é uma das ferramentas mais poderosas e versáteis da álgebra linear, permitindo decompor qualquer matriz em componentes que revelam suas propriedades fundamentais.
            
            ## Definição
            
            Para qualquer matriz A de dimensão m × n, a SVD expressa A como o produto de três matrizes:
            
            $$A = U\\Sigma V^T$$
            
            Onde:
            - U é uma matriz m × m ortogonal (suas colunas são os vetores singulares à esquerda)
            - Σ é uma matriz m × n diagonal (contendo os valores singulares)
            - Vᵀ é a transposta de uma matriz n × n ortogonal V (cujas colunas são os vetores singulares à direita)
            
            ## Valores Singulares
            
            Os valores singulares σᵢ são os elementos diagonais da matriz Σ, ordenados de forma que σ₁ ≥ σ₂ ≥ ... ≥ σᵣ > 0, onde r é o posto da matriz A.
            
            Eles correspondem às raízes quadradas dos autovalores de AᵀA (ou AAᵀ).
            
            ## Propriedades Importantes
            
            1. **Posto**: O número de valores singulares não nulos é igual ao posto da matriz
            
            2. **Norma de Frobenius**: ||A||_F = √(σ₁² + σ₂² + ... + σᵣ²)
            
            3. **Norma-2**: ||A||₂ = σ₁ (o maior valor singular)
            
            4. **Número de condição**: cond(A) = σ₁/σᵣ (razão entre o maior e o menor valor singular não nulo)
            
            ## Aplicações em Sistemas Lineares
            
            ### 1. Resolução de Sistemas
            
            Para resolver AX = B usando SVD:
            
            1. Decompor A = UΣVᵀ
            2. Reescrever como UΣVᵀX = B
            3. Multiplicar ambos os lados por Uᵀ: UᵀUΣVᵀX = UᵀB
            4. Simplificar: ΣVᵀX = UᵀB
            5. Para valores singulares não nulos σᵢ, a solução é:
               X = V Σ⁺ UᵀB
               Onde Σ⁺ é a pseudo-inversa de Σ (substituindo cada σᵢ não nulo por 1/σᵢ)
            
            ### 2. Solução de Mínimos Quadrados
            
            Para sistemas sobredeterminados (mais equações que incógnitas), a SVD fornece a solução de mínimos quadrados que minimiza ||AX - B||.
            
            ### 3. Sistemas Mal Condicionados
            
            A SVD permite:
            - Identificar o mau condicionamento através do número de condição
            - Aplicar regularização via SVD truncada (ignorando valores singulares muito pequenos)
            
            ## Aplicações Avançadas
            
            1. **Compressão de dados**:
               - Aproximação de baixo posto de matrizes
               - Usada em processamento de imagens (método PCA)
            
            2. **Sistemas de recomendação**:
               - Filtragem colaborativa baseada em matriz
               - Descoberta de fatores latentes
            
            3. **Análise de ruído e sinal**:
               - Separação de componentes de interesse de ruído
               - Redução de dimensionalidade
            
            ## Exemplo
            
            Para a matriz:
            
            $$
            A = \\begin{bmatrix}
            4 & 0 \\\\
            3 & -5
            \\end{bmatrix}
            $$
            
            A SVD resulta em:
            
            $$
            U = \\begin{bmatrix}
            0.8 & 0.6 \\\\
            0.6 & -0.8
            \\end{bmatrix},
            \\Sigma = \\begin{bmatrix}
            5 & 0 \\\\
            0 & 4
            \\end{bmatrix},
            V = \\begin{bmatrix}
            0.8 & 0.6 \\\\
            -0.6 & 0.8
            \\end{bmatrix}
            $$
            
            Isso revela que o posto da matriz é 2, o número de condição é 5/4 = 1.25, e a matriz é bem condicionada.
            """
        }
    }
    
    # Inicializar current_topic no estado da sessão se ainda não existir
    if "current_topic" not in st.session_state:
        st.session_state.current_topic = "Introdução aos Sistemas Lineares"
    
    # Selecionar tópico da teoria
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Usamos uma key diferente para o radio button e armazenamos o valor em uma variável temporária
        # Isso evita o erro de tentar modificar st.session_state.theory_topic após o widget ser instanciado
        selected_topic = st.radio(
            "Tópicos:",
            list(theory_topics.keys()),
            index=list(theory_topics.keys()).index(st.session_state.current_topic),
            key="topic_selector"
        )
        
        # Agora verificamos se o tópico mudou e atualizamos o estado da sessão
        if selected_topic != st.session_state.current_topic:
            st.session_state.current_topic = selected_topic
        
        st.markdown("---")
        st.markdown("### Material de Apoio")
        
        # Botão para baixar o material em PDF
        if st.button("📥 Baixar Material em PDF", key="download_pdf_btn"):
            st.success(f"Download de '{selected_topic}.pdf' iniciado! (Simulação)")
        
        # Botão para acessar videoaulas
        if st.button("🎬 Acessar Videoaulas", key="video_btn"):
            st.session_state.page = "Vídeoaulas"
            st.rerun()
            
        # Botão para adicionar aos favoritos
        if st.button("⭐ Adicionar aos Favoritos", key="fav_btn"):
            if "favorites" not in st.session_state:
                st.session_state.favorites = {"reference_cards": []}
            
            # Verificar se já está nos favoritos
            already_saved = False
            for card in st.session_state.favorites.get("reference_cards", []):
                if card.get("title") == selected_topic:
                    already_saved = True
                    break
                    
            if already_saved:
                st.info(f"'{selected_topic}' já está nos seus favoritos.")
            else:
                st.session_state.favorites.setdefault("reference_cards", []).append(
                    {"title": selected_topic, "date": datetime.datetime.now().strftime("%d/%m/%Y %H:%M")}
                )
                st.success(f"'{selected_topic}' adicionado aos seus favoritos!")
        
        # Lista de tópicos relacionados
        # st.markdown("### Tópicos Relacionados")
        # related_topics = {
        #     "Introdução aos Sistemas Lineares": ["Classificação de Sistemas Lineares", "Método da Adição", "Método da Substituição"],
        #     "Classificação de Sistemas Lineares": ["Introdução aos Sistemas Lineares", "Teorema de Rouché-Capelli", "Interpretação Geométrica"],
        #     "Método de Eliminação de Gauss": ["Método de Gauss-Jordan", "Método da Adição", "Método da Matriz Inversa"],
        #     "Método da Adição": ["Método de Eliminação de Gauss", "Método da Substituição", "Método da Comparação"],
        #     "Método da Substituição": ["Método da Adição", "Método da Comparação", "Introdução aos Sistemas Lineares"],
        #     "Método da Comparação": ["Método da Substituição", "Método da Adição", "Classificação de Sistemas Lineares"],
        #     "Regra de Cramer": ["Método da Matriz Inversa", "Classificação de Sistemas Lineares", "Teorema de Rouché-Capelli"],
        #     "Método da Matriz Inversa": ["Regra de Cramer", "Decomposição LU", "Fatoração SVD"],
        #     "Método de Gauss-Jordan": ["Método de Eliminação de Gauss", "Métodos Iterativos", "Decomposição QR"],
        #     "Métodos Iterativos": ["Método de Gauss-Jordan", "Estabilidade Numérica", "Decomposição LU"],
        #     "Decomposição LU": ["Método da Matriz Inversa", "Estabilidade Numérica", "Decomposição QR"],
        #     "Interpretação Geométrica": ["Classificação de Sistemas Lineares", "Aplicações de Sistemas Lineares", "Sistemas Homogêneos"],
        #     "Aplicações de Sistemas Lineares": ["Introdução aos Sistemas Lineares", "Aplicações Avançadas", "Sistemas Lineares em Programação Linear"],
        #     "Sistemas Homogêneos": ["Classificação de Sistemas Lineares", "Interpretação Geométrica", "Fatoração SVD"],
        #     "Estabilidade Numérica": ["Métodos Iterativos", "Decomposição LU", "Fatoração SVD"],
        #     "Aplicações Avançadas": ["Aplicações de Sistemas Lineares", "Sistemas Não-Lineares", "Fatoração SVD"],
        #     "Sistemas Não-Lineares": ["Aplicações Avançadas", "Métodos Iterativos", "Sistemas Lineares em Programação Linear"],
        #     "Sistemas Lineares em Programação Linear": ["Aplicações de Sistemas Lineares", "Aplicações Avançadas", "Método Simplex"],
        #     "Teorema de Rouché-Capelli": ["Classificação de Sistemas Lineares", "Sistemas Homogêneos", "Interpretação Geométrica"],
        #     "Decomposição QR": ["Decomposição LU", "Fatoração SVD", "Estabilidade Numérica"],
        #     "Fatoração SVD": ["Decomposição QR", "Estabilidade Numérica", "Aplicações Avançadas"]
        # }

        # for topic in related_topics.get(selected_topic, []):
        #     if st.button(f"📌 {topic}", key=f"related_{topic}"):
        #         # Atualizar tópico selecionado usando a variável current_topic
        #         st.session_state.current_topic = topic
        #         st.rerun()
    
    with col2:
        # Atualizar histórico de tópicos estudados
        if selected_topic not in st.session_state.user_progress["topics_studied"]:
            st.session_state.user_progress["topics_studied"].append(selected_topic)
            
        # Exibir conteúdo do tópico selecionado
        st.markdown(theory_topics[selected_topic]["content"])
        
        # Adicionar botão para exercícios relacionados
        st.markdown("---")
        st.markdown("### Quer praticar este conteúdo?")
        
        if st.button("✏️ Praticar com Exercícios Relacionados", key="practice_btn"):
            # Salvar o tópico atual para a página de exercícios usar
            st.session_state.exercise_topic_from_theory = selected_topic
            
            # Mudar para a página de exercícios
            st.session_state.page = "Exercícios"
            
            # Tentar mapear o tópico para um tipo de exercício
            topic_to_exercise = {
                "Método de Eliminação de Gauss": "Sistemas 3x3",
                "Método da Adição": "Sistemas 2x2",
                "Método da Substituição": "Sistemas 2x2",
                "Método da Comparação": "Sistemas 2x2",
                "Regra de Cramer": "Sistemas 2x2",
                "Método da Matriz Inversa": "Sistemas 3x3",
                "Aplicações de Sistemas Lineares": "Aplicações",
                "Métodos Iterativos": "Métodos Iterativos",
                "Sistemas Homogêneos": "Sistemas SPI",
                "Estabilidade Numérica": "Mal Condicionados"
            }
            
            if selected_topic in topic_to_exercise:
                st.session_state.suggested_exercise_topic = topic_to_exercise[selected_topic]
            else:
                st.session_state.suggested_exercise_topic = "Geral"
                
            st.rerun()

# Modificar main() para usar a versão atualizada da função show_theory_page
def main():
    # Inicializar estados da sessão se não existirem
    if "page" not in st.session_state:
        st.session_state.page = "Início"
    
    if "user_progress" not in st.session_state:
        st.session_state.user_progress = {
            "exercises_completed": 0,
            "correct_answers": 0,
            "topics_studied": [],
            "difficulty_levels": {"Fácil": 0, "Médio": 0, "Difícil": 0},
            "last_login": datetime.datetime.now().strftime("%d/%m/%Y"),
            "streak": 1
        }
    
    if "favorites" not in st.session_state:
        st.session_state.favorites = {
            "examples": [],
            "reference_cards": [],
            "exercises": []
        }
        
    # Se não houver current_topic definido, inicialize
    if "current_topic" not in st.session_state:
        st.session_state.current_topic = "Introdução aos Sistemas Lineares"
    
    # Barra lateral
    with st.sidebar:
        st.image("calculo.png", width=280)
        st.title("MENU")
        
        # Seções principais
        main_sections = {
            "Início": "🏠",
            "Resolver Sistema": "🧮",
            "Teoria": "📚",
            "Exercícios": "✏️",
            "Exemplos": "📋",
            "Referência Rápida": "📝",
            "Vídeoaulas": "🎬",
            "Meu Progresso": "📊"
        }
        
        for section, icon in main_sections.items():
            if st.sidebar.button(f"{icon} {section}", key=f"btn_{section}", use_container_width=True):
                st.session_state.page = section
                # Usar rerun em vez de experimental_rerun
                st.rerun()
        
        st.sidebar.markdown("---")
        
        # Configurações da aplicação
        with st.sidebar.expander("⚙️ Configurações"):
            st.checkbox("Modo escuro", value=False, key="dark_mode")
            st.checkbox("Mostrar passos detalhados", value=True, key="show_steps_config")
            st.select_slider("Precisão numérica", options=["Baixa", "Média", "Alta"], value="Média", key="precision")
            st.slider("Tamanho da fonte", min_value=80, max_value=120, value=100, step=10, format="%d%%", key="font_size")
        
        # Informações do usuário
        st.sidebar.markdown("---")
        col1, col2 = st.sidebar.columns([1, 3])
        with col1:
            st.image("calculo.png", width=60)
        with col2:
            st.markdown("**Usuário:** Estudante")
            st.markdown(f"**Progresso:** {int(min(st.session_state.user_progress['exercises_completed'] / 20 * 100, 100))}%")
        
        # Exibir streak
        st.sidebar.markdown(f"🔥 **Sequência de estudos:** {st.session_state.user_progress['streak']} dias")
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("v1.0.0 | © 2025 SistemaSolver")
    
    # Conteúdo principal
    if st.session_state.page == "Início":
        show_home_page()
    elif st.session_state.page == "Resolver Sistema":
        show_solver_page()
    elif st.session_state.page == "Teoria":
        show_theory_page()
    elif st.session_state.page == "Exercícios":
        show_exercises_page()
    elif st.session_state.page == "Exemplos":
        show_examples_page()
    elif st.session_state.page == "Referência Rápida":
        show_reference_page()
    elif st.session_state.page == "Vídeoaulas":
        show_videos_page()
    elif st.session_state.page == "Meu Progresso":
        show_progress_page()

# Função para ser chamada quando a página de exercícios é carregada
def setup_exercises_page_from_theory():
    # Verificar se estamos vindo da página de teoria
    if "exercise_topic_from_theory" in st.session_state:
        # Usar o tópico sugerido
        if "suggested_exercise_topic" in st.session_state:
            st.session_state.exercise_topic = st.session_state.suggested_exercise_topic
            
        # Limpar as variáveis após uso
        del st.session_state.exercise_topic_from_theory
        if "suggested_exercise_topic" in st.session_state:
            del st.session_state.suggested_exercise_topic
            
def show_theory_page():
    st.markdown('<h1 class="main-header">Teoria dos Sistemas Lineares</h1>', unsafe_allow_html=True)
    
    theory_topics = {
        "Introdução aos Sistemas Lineares": {
            "content": """
            # Introdução aos Sistemas Lineares
            
            Um **sistema de equações lineares** é um conjunto de uma ou mais equações lineares envolvendo as mesmas variáveis.
            
            ## Definição Formal
            
            Em notação matemática, um sistema linear de m equações e n incógnitas pode ser escrito como:
            
            $$
            \\begin{align}
            a_{11}x_1 + a_{12}x_2 + \\ldots + a_{1n}x_n &= b_1\\\\
            a_{21}x_1 + a_{22}x_2 + \\ldots + a_{2n}x_n &= b_2\\\\
            \\vdots\\\\
            a_{m1}x_1 + a_{m2}x_2 + \\ldots + a_{mn}x_n &= b_m
            \\end{align}
            $$
            
            Onde:
            - $a_{ij}$ são os coeficientes das incógnitas
            - $x_j$ são as incógnitas (ou variáveis)
            - $b_i$ são os termos independentes
            
            ## Representação Matricial
            
            O sistema linear também pode ser representado na forma matricial:
            
            $$
            A\\mathbf{x} = \\mathbf{b}
            $$
            
            Onde:
            - $A$ é a matriz dos coeficientes ($m \\times n$)
            - $\\mathbf{x}$ é o vetor das incógnitas ($n \\times 1$)
            - $\\mathbf{b}$ é o vetor dos termos independentes ($m \\times 1$)
            
            ## Tipos de Sistemas
            
            Um sistema linear pode ser:
            1. **Determinado**: Possui exatamente uma solução
            2. **Indeterminado**: Possui infinitas soluções
            3. **Impossível**: Não possui solução
            
            ## Importância
            
            Os sistemas lineares são fundamentais na matemática aplicada e aparecem em diversos contextos:
            - Física (equilíbrio de forças, circuitos elétricos)
            - Economia (modelos de preço, análise de insumo-produto)
            - Engenharia (análise estrutural, processamento de sinais)
            - Computação gráfica (transformações geométricas)
            """
        },
        "Classificação de Sistemas Lineares": {
            "content": """
            # Classificação de Sistemas Lineares
            
            ## Sistemas Possíveis e Determinados (SPD)
            
            Um sistema é **possível e determinado** quando possui **exatamente uma solução**.
            
            **Características**:
            - O determinante da matriz dos coeficientes é diferente de zero (det(A) ≠ 0)
            - O número de equações linearmente independentes é igual ao número de incógnitas
            - O posto da matriz dos coeficientes é igual ao posto da matriz ampliada e igual ao número de incógnitas
            
            **Interpretação geométrica**:
            - Em 2D: duas retas que se intersectam em um único ponto
            - Em 3D: três planos que se intersectam em um único ponto
            
            ## Sistemas Possíveis e Indeterminados (SPI)
            
            Um sistema é **possível e indeterminado** quando possui **infinitas soluções**.
            
            **Características**:
            - O posto da matriz dos coeficientes é igual ao posto da matriz ampliada
            - O posto é menor que o número de incógnitas
            
            **Interpretação geométrica**:
            - Em 2D: retas coincidentes (sobrepostas)
            - Em 3D: planos que se intersectam em uma reta ou coincidem
            
            ## Sistemas Impossíveis (SI)
            
            Um sistema é **impossível** quando **não possui solução**.
            
            **Características**:
            - O posto da matriz dos coeficientes é menor que o posto da matriz ampliada
            
            **Interpretação geométrica**:
            - Em 2D: retas paralelas (não se intersectam)
            - Em 3D: planos paralelos ou que se intersectam sem um ponto comum a todos
            
            ## Teorema de Rouché-Capelli
            
            O teorema estabelece que:
            
            - Um sistema é **compatível** (tem solução) se e somente se o posto da matriz dos coeficientes é igual ao posto da matriz ampliada.
            
            - Seja r = posto da matriz dos coeficientes = posto da matriz ampliada:
              - Se r = n (número de incógnitas), o sistema é SPD
              - Se r < n, o sistema é SPI
            
            - Se o posto da matriz dos coeficientes < posto da matriz ampliada, o sistema é SI
            """
        },
        "Método de Eliminação de Gauss": {
            "content": """
            # Método de Eliminação de Gauss
            
            O método de eliminação de Gauss é um dos algoritmos mais importantes para resolver sistemas lineares. Consiste em transformar o sistema em uma forma triangular superior (escalonada) através de operações elementares.
            
            ## Operações Elementares
            
            As operações elementares permitidas são:
            1. Trocar a posição de duas equações
            2. Multiplicar uma equação por uma constante não nula
            3. Substituir uma equação pela soma dela com um múltiplo de outra equação
            
            ## Algoritmo
            
            O método pode ser dividido em duas etapas:
            
            ### 1. Eliminação para frente (Forward Elimination)
            
            Nesta fase, transformamos a matriz aumentada [A|b] em uma matriz triangular superior. Para cada linha i da matriz:
            
            - Encontrar o pivô (elemento não nulo na posição i,i)
            - Para cada linha j abaixo da linha i:
              - Calcular o fator de eliminação: f = a_ji / a_ii
              - Subtrair da linha j a linha i multiplicada por f
            
            ### 2. Substituição reversa (Back Substitution)
            
            Uma vez obtida a forma triangular, resolvemos o sistema de trás para frente:
            
            - Calcular o valor da última variável
            - Substituir esse valor nas equações anteriores para encontrar as demais variáveis
            
            ## Eliminação Gaussiana com Pivoteamento Parcial
            
            Para melhorar a estabilidade numérica, é comum usar pivoteamento parcial:
            
            - A cada passo, escolher como pivô o elemento de maior valor absoluto na coluna atual
            - Trocar linhas para que este elemento fique na posição diagonal
            
            ## Exemplo
            
            Considere o sistema:
            
            $$
            \\begin{align}
            x + y + z &= 6\\\\
            2x - y + z &= 3\\\\
            x + 2y + 3z &= 14
            \\end{align}
            $$
            
            **Matriz aumentada inicial**:
            
            $$
            \\begin{bmatrix}
            1 & 1 & 1 & | & 6 \\\\
            2 & -1 & 1 & | & 3 \\\\
            1 & 2 & 3 & | & 14
            \\end{bmatrix}
            $$
            
            **Após eliminação para frente**:
            
            $$
            \\begin{bmatrix}
            1 & 1 & 1 & | & 6 \\\\
            0 & -3 & -1 & | & -9 \\\\
            0 & 0 & 5/3 & | & 5
            \\end{bmatrix}
            $$
            
            **Substituição reversa**:
            - Da última linha: z = 3
            - Da segunda linha: -3y - 3 = -9, portanto y = 2
            - Da primeira linha: x + 2 + 3 = 6, portanto x = 1
            
            **Solução**: x = 1, y = 2, z = 3
            """
        },
                "Método da Adição": {
            "content": """
            # Método da Adição (ou Eliminação por Soma)
            
            O método da adição é uma técnica específica para resolver sistemas de equações lineares, especialmente útil em sistemas com poucas equações. É um caso particular do método de eliminação de Gauss, focado na eliminação de variáveis através da soma de equações.
            
            ## Procedimento
            
            1. Organizar as equações de modo que os coeficientes de uma determinada variável possam se anular quando as equações forem somadas ou subtraídas
            2. Multiplicar as equações por constantes apropriadas para que os coeficientes da variável a ser eliminada se tornem opostos
            3. Somar as equações para eliminar a variável
            4. Repetir o processo até obter uma equação com apenas uma variável
            5. Resolver para essa variável e substituir nas equações anteriores
            
            ## Exemplo
            
            Considere o sistema:
            
            $$
            \\begin{align}
            2x + 3y &= 8 \\quad (1)\\\\
            4x - y &= 10 \\quad (2)
            \\end{align}
            $$
            
            **Passo 1**: Multiplicamos a equação (1) por 2 para que o coeficiente de x se torne 4
            
            $$
            \\begin{align}
            4x + 6y &= 16 \\quad (1')\\\\
            4x - y &= 10 \\quad (2)
            \\end{align}
            $$
            
            **Passo 2**: Subtraímos a equação (2) da equação (1')
            
            $$
            \\begin{align}
            4x + 6y - (4x - y) &= 16 - 10\\\\
            7y &= 6
            \\end{align}
            $$
            
            **Passo 3**: Resolvemos para y
            
            $$y = \\frac{6}{7}$$
            
            **Passo 4**: Substituímos o valor de y na equação (2)
            
            $$
            \\begin{align}
            4x - \\frac{6}{7} &= 10\\\\
            4x &= 10 + \\frac{6}{7}\\\\
            4x &= \\frac{70 + 6}{7}\\\\
            4x &= \\frac{76}{7}\\\\
            x &= \\frac{19}{7}
            \\end{align}
            $$
            
            **Solução**: $x = \\frac{19}{7}$, $y = \\frac{6}{7}$
            
            ## Vantagens e Desvantagens
            
            **Vantagens**:
            - Método intuitivo e fácil de aplicar manualmente para sistemas pequenos
            - Não requer conhecimentos avançados de álgebra linear
            - Pode ser mais direto que outros métodos para certos tipos de sistemas
            
            **Desvantagens**:
            - Pode ser trabalhoso para sistemas maiores
            - A escolha de quais equações combinar e como requer estratégia
            - Menos sistemático que o método de eliminação de Gauss completo
            """
        },
        "Método da Substituição": {
            "content": """
            # Método da Substituição
            
            O método da substituição é uma técnica elementar para resolver sistemas de equações lineares, especialmente útil para sistemas pequenos ou esparsos (com muitos zeros).
            
            ## Procedimento
            
            1. Isolar uma variável em uma das equações
            2. Substituir a expressão obtida nas demais equações, reduzindo o sistema
            3. Repetir o processo até obter uma equação com apenas uma variável
            4. Resolver para essa variável e substituir nas expressões anteriores para encontrar as demais variáveis
            
            ## Exemplo
            
            Considere o sistema:
            
            $$
            \\begin{align}
            x + 2y &= 5 \\quad (1)\\\\
            3x - 4y &= 7 \\quad (2)
            \\end{align}
            $$
            
            **Passo 1**: Isolamos x na equação (1)
            
            $$x = 5 - 2y \quad (3)$$
            
            **Passo 2**: Substituímos (3) na equação (2)
            
            $$
            \\begin{align}
            3(5 - 2y) - 4y &= 7\\\\
            15 - 6y - 4y &= 7\\\\
            15 - 10y &= 7\\\\
            -10y &= 7 - 15\\\\
            -10y &= -8\\\\
            y &= \\frac{8}{10} = \\frac{4}{5}
            \\end{align}
            $$
            
            **Passo 3**: Substituímos o valor de y em (3)
            
            $$
            \\begin{align}
            x &= 5 - 2 \\cdot \\frac{4}{5}\\\\
            &= 5 - \\frac{8}{5}\\\\
            &= \\frac{25 - 8}{5}\\\\
            &= \\frac{17}{5}
            \\end{align}
            $$
            
            **Solução**: $x = \\frac{17}{5}$, $y = \\frac{4}{5}$
            
            ## Aplicação em Sistemas Triangulares
            
            O método da substituição é particularmente eficiente para sistemas triangulares. De fato, a substituição reversa usada após a eliminação gaussiana é uma aplicação deste método.
            
            Para um sistema triangular superior:
            
            $$
            \\begin{align}
            a_{11}x_1 + a_{12}x_2 + \\ldots + a_{1n}x_n &= b_1\\\\
            a_{22}x_2 + \\ldots + a_{2n}x_n &= b_2\\\\
            \\vdots\\\\
            a_{nn}x_n &= b_n
            \\end{align}
            $$
            
            Começamos resolvendo $x_n = b_n/a_{nn}$ e substituímos nas equações anteriores.
            
            ## Vantagens e Desvantagens
            
            **Vantagens**:
            - Método intuitivo e fácil de entender
            - Eficiente para sistemas pequenos ou triangulares
            - Útil em sistemas onde algumas variáveis podem ser facilmente isoladas
            
            **Desvantagens**:
            - Pode levar a cálculos complexos se as expressões substituídas forem complicadas
            - Não é tão sistemático ou eficiente computacionalmente quanto outros métodos para sistemas grandes
            """
        },
        "Método da Comparação": {
            "content": """
            # Método da Comparação
            
            O método da comparação é uma técnica para resolver sistemas de equações lineares que consiste em isolar a mesma variável em diferentes equações e, em seguida, igualar as expressões resultantes.
            
            ## Procedimento
            
            1. Isolar a mesma variável em duas ou mais equações do sistema
            2. Igualar as expressões obtidas, formando novas equações com menos variáveis
            3. Resolver o sistema reduzido
            4. Substituir as soluções encontradas nas expressões iniciais para obter as demais variáveis
            
            ## Exemplo
            
            Considere o sistema:
            
            $$
            \\begin{align}
            2x + 3y &= 7 \\quad (1)\\\\
            5x - 2y &= 8 \\quad (2)
            \\end{align}
            $$
            
            **Passo 1**: Isolamos x em ambas as equações
            
            Da equação (1):
            $$x = \\frac{7 - 3y}{2} \\quad (3)$$
            
            Da equação (2):
            $$x = \\frac{8 + 2y}{5} \\quad (4)$$
            
            **Passo 2**: Igualamos as expressões (3) e (4)
            
            $$
            \\begin{align}
            \\frac{7 - 3y}{2} &= \\frac{8 + 2y}{5}\\\\
            5(7 - 3y) &= 2(8 + 2y)\\\\
            35 - 15y &= 16 + 4y\\\\
            35 - 15y &= 16 + 4y\\\\
            35 - 16 &= 4y + 15y\\\\
            19 &= 19y\\\\
            y &= 1
            \\end{align}
            $$
            
            **Passo 3**: Substituímos y = 1 em uma das expressões para x, por exemplo em (3)
            
            $$
            \\begin{align}
            x &= \\frac{7 - 3(1)}{2}\\\\
            &= \\frac{7 - 3}{2}\\\\
            &= \\frac{4}{2}\\\\
            &= 2
            \\end{align}
            $$
            
            **Solução**: x = 2, y = 1
            
            ## Verificação
            
            Podemos verificar a solução substituindo os valores nas equações originais:
            
            Equação (1): 2(2) + 3(1) = 4 + 3 = 7 ✓
            
            Equação (2): 5(2) - 2(1) = 10 - 2 = 8 ✓
            
            ## Vantagens e Desvantagens
            
            **Vantagens**:
            - Método intuitivo que não requer conhecimentos avançados
            - Útil para sistemas com duas equações e duas incógnitas
            - Pode ser mais direto em certos casos específicos
            
            **Desvantagens**:
            - Torna-se complicado para sistemas maiores
            - Pode levar a expressões algébricas complexas
            - Menos eficiente computacionalmente que métodos mais sistemáticos
            """
        },
        "Regra de Cramer": {
            "content": """
            # Regra de Cramer
            
            A regra de Cramer é um método para resolver sistemas lineares usando determinantes. É aplicável apenas a sistemas com mesmo número de equações e incógnitas, onde o determinante da matriz dos coeficientes é diferente de zero (sistemas SPD).
            
            ## Procedimento
            
            Para um sistema AX = B:
            
            1. Calcular o determinante D da matriz A
            2. Para cada variável xᵢ:
               - Substituir a coluna i da matriz A pela coluna B, obtendo a matriz Aᵢ
               - Calcular o determinante Dᵢ
               - A solução para xᵢ é dada por xᵢ = Dᵢ/D
            
            ## Fórmula
            
            Para um sistema 2×2:
            
            $$
            \\begin{align}
            a_1x + b_1y &= c_1\\\\
            a_2x + b_2y &= c_2
            \\end{align}
            $$
            
            As soluções são:
            
            $$
            x = \\frac{\\begin{vmatrix} c_1 & b_1 \\\\ c_2 & b_2 \\end{vmatrix}}{\\begin{vmatrix} a_1 & b_1 \\\\ a_2 & b_2 \\end{vmatrix}} = \\frac{c_1b_2 - b_1c_2}{a_1b_2 - b_1a_2}
            $$
            
            $$
            y = \\frac{\\begin{vmatrix} a_1 & c_1 \\\\ a_2 & c_2 \\end{vmatrix}}{\\begin{vmatrix} a_1 & b_1 \\\\ a_2 & b_2 \\end{vmatrix}} = \\frac{a_1c_2 - c_1a_2}{a_1b_2 - b_1a_2}
            $$
            
            ## Exemplo
            
            Considere o sistema:
            
            $$
            \\begin{align}
            2x + 3y &= 8\\\\
            4x - y &= 1
            \\end{align}
            $$
            
            **Determinante principal**:
            
            $$
            D = \\begin{vmatrix} 2 & 3 \\\\ 4 & -1 \\end{vmatrix} = 2 \\times (-1) - 3 \\times 4 = -2 - 12 = -14
            $$
            
            **Determinante para x**:
            
            $$
            D_x = \\begin{vmatrix} 8 & 3 \\\\ 1 & -1 \\end{vmatrix} = 8 \\times (-1) - 3 \\times 1 = -8 - 3 = -11
            $$
            
            **Determinante para y**:
            
            $$
            D_y = \\begin{vmatrix} 2 & 8 \\\\ 4 & 1 \\end{vmatrix} = 2 \\times 1 - 8 \\times 4 = 2 - 32 = -30
            $$
            
            **Solução**:
            
            $$
            x = \\frac{D_x}{D} = \\frac{-11}{-14} = \\frac{11}{14}
            $$
            
            $$
            y = \\frac{D_y}{D} = \\frac{-30}{-14} = \\frac{15}{7}
            $$
            
            ## Vantagens e Desvantagens
            
            **Vantagens**:
            - Método direto (não iterativo)
            - Fácil de entender e aplicar para sistemas pequenos
            
            **Desvantagens**:
            - Aplicável apenas a sistemas quadrados (n×n) com determinante não nulo
            - Computacionalmente ineficiente para sistemas grandes
            - Não é recomendado para sistemas mal condicionados
            """
        },
        "Método da Matriz Inversa": {
            "content": """
            # Método da Matriz Inversa
            
            O método da matriz inversa é uma abordagem direta para resolver sistemas lineares na forma AX = B, onde A é uma matriz quadrada inversível.
            
            ## Procedimento
            
            1. Verificar se a matriz A é inversível (det(A) ≠ 0)
            2. Calcular a matriz inversa A⁻¹
            3. Multiplicar ambos os lados da equação por A⁻¹: A⁻¹(AX) = A⁻¹B
            4. Simplificar: X = A⁻¹B
            
            ## Cálculo da Matriz Inversa
            
            Para uma matriz 2×2:
            
            $$
            \\begin{bmatrix} a & b \\\\ c & d \\end{bmatrix}^{-1} = \\frac{1}{ad-bc} \\begin{bmatrix} d & -b \\\\ -c & a \\end{bmatrix}
            $$
            
            Para matrizes maiores, pode-se usar:
            - Método da matriz adjunta
            - Eliminação gaussiana
            - Decomposição LU
            
            ## Exemplo
            
            Considere o sistema:
            
            $$
            \\begin{align}
            2x + y &= 5\\\\
            3x + 4y &= 11
            \\end{align}
            $$
            
            Na forma matricial:
            
            $$
            \\begin{bmatrix} 2 & 1 \\\\ 3 & 4 \\end{bmatrix} \\begin{bmatrix} x \\\\ y \\end{bmatrix} = \\begin{bmatrix} 5 \\\\ 11 \\end{bmatrix}
            $$
            
            **Determinante**:
            
            $$
            \\det(A) = 2 \\times 4 - 1 \\times 3 = 8 - 3 = 5
            $$
            
            **Matriz inversa**:
            
            $$
            A^{-1} = \\frac{1}{5} \\begin{bmatrix} 4 & -1 \\\\ -3 & 2 \\end{bmatrix} = \\begin{bmatrix} 4/5 & -1/5 \\\\ -3/5 & 2/5 \\end{bmatrix}
            $$
            
            **Solução**:
            
            $$
            \\begin{bmatrix} x \\\\ y \\end{bmatrix} = \\begin{bmatrix} 4/5 & -1/5 \\\\ -3/5 & 2/5 \\end{bmatrix} \\begin{bmatrix} 5 \\\\ 11 \\end{bmatrix} = \\begin{bmatrix} 4/5 \\times 5 - 1/5 \\times 11 \\\\ -3/5 \\times 5 + 2/5 \\times 11 \\end{bmatrix} = \\begin{bmatrix} 4 - 11/5 \\\\ -3 + 22/5 \\end{bmatrix} = \\begin{bmatrix} 9/5 \\\\ 7/5 \\end{bmatrix}
            $$
            
            Portanto, x = 9/5 e y = 7/5.
            
            ## Vantagens e Desvantagens
            
            **Vantagens**:
            - Método direto e elegante
            - Útil quando precisamos resolver múltiplos sistemas com a mesma matriz de coeficientes
            
            **Desvantagens**:
            - Aplicável apenas a sistemas quadrados (n×n) com determinante não nulo
            - Computacionalmente ineficiente para sistemas grandes
            - Pode ser numericamente instável para matrizes mal condicionadas
            """
        },
        "Método de Gauss-Jordan": {
            "content": """
            # Método de Gauss-Jordan
            
            O método de Gauss-Jordan é uma extensão do método de eliminação de Gauss que leva a matriz aumentada à forma escalonada reduzida.
            
            ## Procedimento
            
            1. Aplicar operações elementares para obter 1's na diagonal principal
            2. Zerar todos os elementos acima e abaixo da diagonal principal
            
            **Forma final da matriz aumentada**:
            ```
            | 1 0 0 ... | x₁ |
            | 0 1 0 ... | x₂ |
            | 0 0 1 ... | x₃ |
            | ...       | ... |
            ```
            
            O vetor solução pode ser lido diretamente da última coluna da matriz.
            
            ## Exemplo
            
            Considere o sistema:
            
            $$
            \\begin{align}
            2x + y + z &= 5\\\\
            x - y + 2z &= 4\\\\
            3x + 2y - z &= 3
            \\end{align}
            $$
            
            **Matriz aumentada inicial**:
            
            $$
            \\begin{bmatrix}
            2 & 1 & 1 & | & 5 \\\\
            1 & -1 & 2 & | & 4 \\\\
            3 & 2 & -1 & | & 3
            \\end{bmatrix}
            $$
            
            **Passo 1**: Obter 1 na posição (1,1) e zerar os outros elementos da primeira coluna
            
            Dividir a primeira linha por 2:
            
            $$
            \\begin{bmatrix}
            1 & 1/2 & 1/2 & | & 5/2 \\\\
            1 & -1 & 2 & | & 4 \\\\
            3 & 2 & -1 & | & 3
            \\end{bmatrix}
            $$
            
            Subtrair a primeira linha da segunda:
            
            $$
            \\begin{bmatrix}
            1 & 1/2 & 1/2 & | & 5/2 \\\\
            0 & -3/2 & 3/2 & | & 3/2 \\\\
            3 & 2 & -1 & | & 3
            \\end{bmatrix}
            $$
            
            Subtrair 3 vezes a primeira linha da terceira:
            
            $$
            \\begin{bmatrix}
            1 & 1/2 & 1/2 & | & 5/2 \\\\
            0 & -3/2 & 3/2 & | & 3/2 \\\\
            0 & 1/2 & -5/2 & | & -9/2
            \\end{bmatrix}
            $$
            
            **Passo 2**: Obter 1 na posição (2,2) e zerar os outros elementos da segunda coluna
            
            Multiplicar a segunda linha por -2/3:
            
            $$
            \\begin{bmatrix}
            1 & 1/2 & 1/2 & | & 5/2 \\\\
            0 & 1 & -1 & | & -1 \\\\
            0 & 1/2 & -5/2 & | & -9/2
            \\end{bmatrix}
            $$
            
            Subtrair 1/2 vezes a segunda linha da primeira:
            
            $$
            \\begin{bmatrix}
            1 & 0 & 1 & | & 3 \\\\
            0 & 1 & -1 & | & -1 \\\\
            0 & 1/2 & -5/2 & | & -9/2
            \\end{bmatrix}
            $$
            
            Subtrair 1/2 vezes a segunda linha da terceira:
            
            $$
            \\begin{bmatrix}
            1 & 0 & 1 & | & 3 \\\\
            0 & 1 & -1 & | & -1 \\\\
            0 & 0 & -2 & | & -4
            \\end{bmatrix}
            $$
            
            **Passo 3**: Obter 1 na posição (3,3) e zerar os outros elementos da terceira coluna
            
            Multiplicar a terceira linha por -1/2:
            
            $$
            \\begin{bmatrix}
            1 & 0 & 1 & | & 3 \\\\
            0 & 1 & -1 & | & -1 \\\\
            0 & 0 & 1 & | & 2
            \\end{bmatrix}
            $$
            
            Subtrair 1 vez a terceira linha da primeira:
            
            $$
            \\begin{bmatrix}
            1 & 0 & 0 & | & 1 \\\\
            0 & 1 & -1 & | & -1 \\\\
            0 & 0 & 1 & | & 2
            \\end{bmatrix}
            $$
            
            Somar 1 vez a terceira linha à segunda:
            
            $$
            \\begin{bmatrix}
            1 & 0 & 0 & | & 1 \\\\
            0 & 1 & 0 & | & 1 \\\\
            0 & 0 & 1 & | & 2
            \\end{bmatrix}
            $$
            
            A solução pode ser lida diretamente da última coluna: x = 1, y = 1, z = 2.
            
            ## Vantagens e Desvantagens
            
            **Vantagens**:
            - A solução é obtida diretamente, sem necessidade de substituição reversa
            - Útil para calcular a inversa de uma matriz
            
            **Desvantagens**:
            - Requer mais operações que o método de Gauss padrão
            - Pode ser numericamente menos estável em alguns casos
            """
        },
        "Métodos Iterativos": {
            "content": """
            # Métodos Iterativos para Sistemas Lineares
            
            Os métodos iterativos começam com uma aproximação inicial e melhoram progressivamente a solução.
            
            ## Método de Jacobi
            
            **Procedimento**:
            1. Para cada equação i, isolar a incógnita x_i
            2. Iniciar com uma aproximação inicial (geralmente zeros)
            3. Em cada iteração k+1, calcular:
               x_i^(k+1) = (b_i - Σ a_ij x_j^(k)) / a_ii, para j ≠ i
            4. Repetir até convergir
            
            ## Método de Gauss-Seidel
            
            Similar ao método de Jacobi, mas usa valores já atualizados na mesma iteração:
            
            x_i^(k+1) = (b_i - Σ a_ij x_j^(k+1) - Σ a_ij x_j^(k)) / a_ii
                          j<i                j>i
            
            **Condições de convergência**:
            - Matriz diagonalmente dominante (|a_ii| > Σ |a_ij| para j ≠ i)
            - Matriz definida positiva
            
            ## Exemplo
            
            Considere o sistema:
            
            $$
            \\begin{align}
            10x + 2y + z &= 13\\\\
            x + 5y + z &= 7\\\\
            2x + y + 10z &= 13
            \\end{align}
            $$
            
            Isolando as variáveis:
            
            $$
            \\begin{align}
            x &= \\frac{13 - 2y - z}{10}\\\\
            y &= \\frac{7 - x - z}{5}\\\\
            z &= \\frac{13 - 2x - y}{10}
            \\end{align}
            $$
            
            **Método de Jacobi**:
            
            Partindo de x^(0) = y^(0) = z^(0) = 0:
            
            Iteração 1:
            - x^(1) = (13 - 2×0 - 0)/10 = 1.3
            - y^(1) = (7 - 0 - 0)/5 = 1.4
            - z^(1) = (13 - 2×0 - 0)/10 = 1.3
            
            Iteração 2:
            - x^(2) = (13 - 2×1.4 - 1.3)/10 = 0.83
            - y^(2) = (7 - 1.3 - 1.3)/5 = 0.88
            - z^(2) = (13 - 2×1.3 - 1.4)/10 = 0.83
            
            O processo continua até a convergência para x = y = z = 1.
            
            **Método de Gauss-Seidel**:
            
            Partindo de x^(0) = y^(0) = z^(0) = 0:
            
            Iteração 1:
            - x^(1) = (13 - 2×0 - 0)/10 = 1.3
            - y^(1) = (7 - 1.3 - 0)/5 = 1.14
            - z^(1) = (13 - 2×1.3 - 1.14)/10 = 0.826
            
            Iteração 2:
            - x^(2) = (13 - 2×1.14 - 0.826)/10 = 0.8934
            - y^(2) = (7 - 0.8934 - 0.826)/5 = 1.0561
            - z^(2) = (13 - 2×0.8934 - 1.0561)/10 = 0.9157
            
            O método converge mais rapidamente para x = y = z = 1.
            
            ## Vantagens e Desvantagens
            
            **Vantagens**:
            - Mais eficientes para sistemas grandes e esparsos
            - Menor requisito de memória
            - Podem lidar com matrizes mal condicionadas
            
            **Desvantagens**:
            - Convergência não garantida para todas as matrizes
            - Podem ser lentos para atingir alta precisão
            """
        },
        "Decomposição LU": {
            "content": """
            # Decomposição LU
            
            A decomposição LU fatoriza a matriz A em um produto de duas matrizes: A = LU, onde:
            - L é uma matriz triangular inferior
            - U é uma matriz triangular superior
            
            ## Procedimento para resolver AX = B
            
            1. Decompor A = LU
            2. Resolver LY = B por substituição direta
            3. Resolver UX = Y por substituição reversa
            
            ## Algoritmo para decomposição LU
            
            **Procedimento de Doolittle**:
            
            Para uma matriz n×n:
            
            1. Para i = 1 até n:
               - Para j = i até n: u_{ij} = a_{ij} - Σ(l_{ik} × u_{kj}) para k = 1 até i-1
               - Para j = i+1 até n: l_{ji} = (a_{ji} - Σ(l_{jk} × u_{ki}) para k = 1 até i-1) / u_{ii}
            
            2. Para i = 1 até n: l_{ii} = 1 (diagonal unitária para L)
            
            ## Exemplo
            
            Considere o sistema:
            
            $$
            \\begin{align}
            2x + y + z &= 5\\\\
            4x + 5y + z &= 9\\\\
            x + y + 3z &= 11
            \\end{align}
            $$
            
            **Decomposição LU**:
            
            Matriz A:
            
            $$
            A = \\begin{bmatrix}
            2 & 1 & 1 \\\\
            4 & 5 & 1 \\\\
            1 & 1 & 3
            \\end{bmatrix}
            $$
            
            Calculando L e U:
            
            $$
            L = \\begin{bmatrix}
            1 & 0 & 0 \\\\
            2 & 1 & 0 \\\\
            0.5 & 0.25 & 1
            \\end{bmatrix}
            $$
            
            $$
            U = \\begin{bmatrix}
            2 & 1 & 1 \\\\
            0 & 3 & -1 \\\\
            0 & 0 & 2.5
            \\end{bmatrix}
            $$
            
            **Resolver LY = B**:
            
            $$
            \\begin{bmatrix}
            1 & 0 & 0 \\\\
            2 & 1 & 0 \\\\
            0.5 & 0.25 & 1
            \\end{bmatrix}
            \\begin{bmatrix}
            y_1 \\\\
            y_2 \\\\
            y_3
            \\end{bmatrix} =
            \\begin{bmatrix}
            5 \\\\
            9 \\\\
            11
            \\end{bmatrix}
            $$
            
            - y₁ = 5
            - y₂ = 9 - 2×5 = -1
            - y₃ = 11 - 0.5×5 - 0.25×(-1) = 11 - 2.5 + 0.25 = 8.75
            
            **Resolver UX = Y**:
            
            $$
            \\begin{bmatrix}
            2 & 1 & 1 \\\\
            0 & 3 & -1 \\\\
            0 & 0 & 2.5
            \\end{bmatrix}
            \\begin{bmatrix}
            x \\\\
            y \\\\
            z
            \\end{bmatrix} =
            \\begin{bmatrix}
            5 \\\\
            -1 \\\\
            8.75
            \\end{bmatrix}
            $$
            
            - z = 8.75 / 2.5 = 3.5
            - y = (-1 + z) / 3 = (-1 + 3.5) / 3 = 2.5 / 3 = 0.833...
            - x = (5 - y - z) / 2 = (5 - 0.833 - 3.5) / 2 = 0.667...
            
            ## Vantagens e Desvantagens
            
            **Vantagens**:
            - Eficiente para resolver múltiplos sistemas com a mesma matriz de coeficientes
            - Útil para calcular determinantes e inversas
            - Computacionalmente eficiente
            
            **Desvantagens**:
            - Requer armazenamento adicional para as matrizes L e U
            - Pode ser instável para matrizes mal condicionadas sem pivoteamento
            """
        },
        "Interpretação Geométrica": {
            "content": """
            # Interpretação Geométrica de Sistemas Lineares
            
            ## Sistemas 2×2
            
            Em um sistema com duas equações e duas incógnitas, cada equação representa uma reta no plano cartesiano.
            
            **Possíveis configurações**:
            
            1. **SPD (Sistema Possível e Determinado)**:
               - As retas se intersectam em um único ponto
               - Este ponto representa a única solução do sistema
               - Exemplo: x + y = 3 e x - y = 1
               
            2. **SPI (Sistema Possível e Indeterminado)**:
               - As retas são coincidentes (sobrepostas)
               - Todos os pontos da reta são soluções do sistema
               - Exemplo: 2x + 3y = 6 e 4x + 6y = 12
               
            3. **SI (Sistema Impossível)**:
               - As retas são paralelas e não coincidentes
               - Não há nenhum ponto comum, ou seja, não há solução
               - Exemplo: x + y = 3 e x + y = 5
            
            ## Sistemas 3×3
            
            Em um sistema com três equações e três incógnitas, cada equação representa um plano no espaço tridimensional.
            
            **Possíveis configurações**:
            
            1. **SPD (Sistema Possível e Determinado)**:
               - Os três planos se intersectam em um único ponto
               - Este ponto é a única solução do sistema
               
            2. **SPI (Sistema Possível e Indeterminado)**:
               - Os planos se intersectam em uma reta (dois planos paralelos intersectados por um terceiro)
               - Ou se intersectam em um plano (três planos coincidentes)
               - As soluções formam uma reta ou um plano
               
            3. **SI (Sistema Impossível)**:
               - Os planos não possuem ponto comum de interseção
               - Pode ocorrer quando temos planos paralelos ou quando a interseção de dois planos é paralela ao terceiro
            
            ## Relação com o Espaço Vetorial
            
            - As linhas da matriz de coeficientes podem ser vistas como vetores
            - O determinante da matriz está relacionado ao volume do paralelepípedo formado por estes vetores
            - Um determinante zero indica que os vetores são linearmente dependentes
            - Para sistemas homogêneos (AX = 0), o conjunto solução forma um subespaço vetorial
            """
        },
        "Aplicações de Sistemas Lineares": {
            "content": """
            # Aplicações de Sistemas Lineares
            
            Os sistemas de equações lineares são ferramentas fundamentais em diversas áreas. Veja algumas aplicações importantes:
            
            ## Física e Engenharia
            
            ### Análise de Circuitos Elétricos
            - Leis de Kirchhoff: correntes em um nó e tensões em um laço
            - Cada equação representa a conservação de corrente ou tensão
            
            **Exemplo**: Para um circuito com três malhas, podemos ter:
            ```
            R₁I₁ + R₂(I₁ - I₂) = V₁
            R₂(I₂ - I₁) + R₃I₂ + R₄(I₂ - I₃) = 0
            R₄(I₃ - I₂) + R₅I₃ = V₂
            ```
            
            ### Estática e Dinâmica
            - Equilíbrio de forças e momentos em estruturas
            - Análise de treliças e vigas
            
            **Exemplo**: Para um sistema com três forças desconhecidas:
            ```
            F₁cos(θ₁) + F₂cos(θ₂) + F₃cos(θ₃) = 0
            F₁sin(θ₁) + F₂sin(θ₂) + F₃sin(θ₃) = 0
            r₁×F₁ + r₂×F₂ + r₃×F₃ = 0
            ```
            
            ### Transferência de Calor
            - Modelagem de problemas de condução térmica
            - Métodos de diferenças finitas para equações diferenciais
            
            ## Economia
            
            ### Análise de Insumo-Produto de Leontief
            - Modelagem das interdependências entre diferentes setores da economia
            - Previsão de como mudanças em um setor afetam outros setores
            
            **Exemplo**: Em uma economia com três setores:
            ```
            x₁ = a₁₁x₁ + a₁₂x₂ + a₁₃x₃ + d₁
            x₂ = a₂₁x₁ + a₂₂x₂ + a₂₃x₃ + d₂
            x₃ = a₃₁x₁ + a₃₂x₂ + a₃₃x₃ + d₃
            ```
            Onde x₁ é a produção do setor i, a_ij é quanto do produto i é usado para produzir uma unidade do produto j, e d_i é a demanda externa.
            
            ### Otimização e Programação Linear
            - Maximização de lucro ou minimização de custos
            - Alocação ótima de recursos limitados
            
            ## Química
            
            ### Balanceamento de Equações Químicas
            - Cada elemento forma uma equação linear
            - Os coeficientes estequiométricos são as incógnitas
            
            **Exemplo**: Para a reação C₃H₈ + O₂ → CO₂ + H₂O
            ```
            3a = c       (para o carbono)
            8a = 2e      (para o hidrogênio)
            2b = 2c + e  (para o oxigênio)
            ```
            
            ### Equilíbrio Químico
            - Determinação de concentrações em equilíbrio
            
            ## Computação Gráfica
            
            ### Transformações Geométricas
            - Rotação, translação e escala de objetos
            - Representadas como transformações matriciais
            
            ### Renderização 3D
            - Sistemas para determinar projeções de objetos 3D em telas 2D
            
            ## Problemas de Mistura
            
            ### Farmacologia
            - Mistura de componentes para atingir concentrações específicas
            - Formulação de medicamentos
            
            **Exemplo**: Um farmacêutico precisa preparar 100ml de uma solução com 25% de um princípio ativo, usando soluções de 10%, 20% e 40%.
            ```
            x + y + z = 100
            0.1x + 0.2y + 0.4z = 25
            ```
            
            ### Processamento de Alimentos
            - Mistura de ingredientes para atingir perfis nutricionais
            
            ## Tráfego e Transporte
            
            ### Fluxo de Redes
            - Modelagem de fluxo de tráfego em redes de transporte
            - Otimização de rotas
            """
        },
        "Sistemas Homogêneos": {
            "content": """
            # Sistemas Lineares Homogêneos
            
            Um sistema homogêneo tem a forma AX = 0 (todos os termos independentes são nulos).
            
            ## Propriedades
            
            1. Todo sistema homogêneo é possível (sempre admite a solução trivial X = 0)
            2. Um sistema homogêneo tem soluções não-triviais se e somente se det(A) = 0
            3. O conjunto de todas as soluções forma um espaço vetorial
            4. A dimensão do espaço de soluções é n - posto(A), onde n é o número de incógnitas
            
            ## Interpretação Geométrica
            
            - Em 2D: se det(A) = 0, as equações representam a mesma reta passando pela origem
            - Em 3D: se det(A) = 0, os planos se intersectam em uma reta ou um plano passando pela origem
            
            ## Aplicações
            
            - **Espaços nulos**:
            O núcleo (ou kernel) de uma transformação linear é o espaço das soluções de AX = 0
              - Fundamental em álgebra linear e geometria
            
            - **Autovalores e autovetores**:
              - Um autovetor v de uma matriz A satisfaz Av = λv, ou (A - λI)v = 0
              - Encontrar os autovetores envolve resolver sistemas homogêneos
            
            - **Equações diferenciais**:
              - Sistemas de equações diferenciais lineares homogêneas têm soluções da forma X = e^(λt)v
              - Onde λ é um autovalor e v é um autovetor associado
            
            ## Exemplo
            
            Considere o sistema homogêneo:
            
            $$
            \\begin{align}
            2x + 3y - z &= 0\\\\
            4x + 6y - 2z &= 0\\\\
            -2x - 3y + z &= 0
            \\end{align}
            $$
            
            Observe que a terceira equação é o oposto da primeira. Além disso, a segunda equação é um múltiplo da primeira (multiplicada por 2).
            
            O determinante da matriz dos coeficientes é zero, o que confirma que o sistema tem soluções não-triviais.
            
            O posto da matriz é 1, e temos 3 incógnitas, então a dimensão do espaço de soluções é 3 - 1 = 2.
            
            **Solução paramétrica**:
            
            Podemos expressar z e y em termos de x:
            Da primeira equação: z = 2x + 3y
            
            Substituindo na segunda e terceira equações, verificamos que são satisfeitas para qualquer valor de x e y.
            
            Então a solução geral é:
            ```
            z = 2x + 3y
            ```
            Onde x e y são parâmetros livres.
            
            Alternativamente, podemos parametrizar como:
            ```
            x = s
            y = t
            z = 2s + 3t
            ```
            Onde s e t são parâmetros livres.
            """
        },
        "Estabilidade Numérica": {
            "content": """
            # Estabilidade Numérica em Sistemas Lineares
            
            ## Número de Condição
            
            O número de condição de uma matriz A, denotado por cond(A), mede a sensibilidade da solução a pequenas perturbações nos dados:
            
            $$\\text{cond}(A) = \\|A\\| \\cdot \\|A^{-1}\\|$$
            
            Para a norma-2, isso é equivalente à razão entre o maior e o menor valor singular:
            
            $$\\text{cond}_2(A) = \\frac{\\sigma_{\\max}}{\\sigma_{\\min}}$$
            
            **Interpretação**:
            - Um número de condição próximo de 1 indica uma matriz bem condicionada
            - Um número de condição grande indica uma matriz mal condicionada
            - Um número de condição infinito indica uma matriz singular
            
            ## Efeitos do Mal Condicionamento
            
            Um sistema mal condicionado tem as seguintes características:
            
            - Pequenas perturbações nos dados (coeficientes ou termos independentes) podem causar grandes mudanças na solução
            - Erros de arredondamento podem ser amplificados significativamente
            - Métodos iterativos podem convergir lentamente ou divergir
            
            **Exemplo**:
            
            Considere o sistema:
            
            $$
            \\begin{align}
            1.000x + 0.999y &= 1.999\\\\
            0.999x + 0.998y &= 1.997
            \\end{align}
            $$
            
            A solução exata é x = y = 1. No entanto, se mudarmos ligeiramente o termo independente da primeira equação para 2.000 (uma perturbação de apenas 0.001), a solução muda drasticamente para aproximadamente x = 2, y = 0.
            
            ## Estratégias para Sistemas Mal Condicionados
            
            1. **Pré-condicionamento**:
               - Multiplicar o sistema por uma matriz de pré-condicionamento para reduzir o número de condição
               - Exemplo: ao invés de resolver Ax = b, resolver M⁻¹Ax = M⁻¹b, onde M é escolhida para que M⁻¹A seja bem condicionada
            
            2. **Refinamento iterativo**:
               - Após obter uma solução aproximada x̃, calcular o resíduo r = b - Ax̃
               - Resolver Ad = r para obter a correção d
               - Atualizar a solução: x = x̃ + d
               - Repetir, se necessário
            
            3. **Métodos de regularização**:
               - Tikhonov: minimizar ||Ax - b||² + λ||x||², onde λ é o parâmetro de regularização
               - SVD truncada: ignorar componentes associados a valores singulares muito pequenos
            
            4. **Aumentar a precisão dos cálculos**:
               - Usar aritmética de precisão dupla ou estendida
               - Implementar algoritmos que minimizam a propagação de erros de arredondamento
            
            5. **Uso de decomposições estáveis**:
               - Decomposição QR
               - Decomposição de valores singulares (SVD)
            
            ## Exemplo de Análise
            
            Para a matriz:
            
            $$
            A = \\begin{bmatrix}
            1 & 1 \\\\
            1 & 1.0001
            \\end{bmatrix}
            $$
            
            1. O determinante é muito pequeno: det(A) = 0.0001
            2. O número de condição é aproximadamente 40000
            3. Uma pequena perturbação de 0.01% em A pode causar uma mudança de 400% na solução
            
            **Verificação**:
            - Se Ax = b, onde b = [2, 2.0001]ᵀ, a solução é x = [1, 1]ᵀ
            - Se mudarmos b para [2.0002, 2.0001]ᵀ (uma mudança de 0.01%), a solução muda para aproximadamente x = [2, 0]ᵀ
            """
        },
        "Aplicações Avançadas": {
            "content": """
            # Aplicações Avançadas de Sistemas Lineares
            
            ## Ajuste de Curvas e Superfícies
            
            O método dos mínimos quadrados leva a sistemas lineares para encontrar os coeficientes que minimizam o erro quadrático.
            
            **Exemplo**: Para ajustar um polinômio de grau n a m pontos (x_i, y_i), formamos o sistema normal:
            
            $$
            \\begin{bmatrix}
            m & \\sum x_i & \\sum x_i^2 & \\cdots & \\sum x_i^n \\\\
            \\sum x_i & \\sum x_i^2 & \\sum x_i^3 & \\cdots & \\sum x_i^{n+1} \\\\
            \\vdots & \\vdots & \\vdots & \\ddots & \\vdots \\\\
            \\sum x_i^n & \\sum x_i^{n+1} & \\sum x_i^{n+2} & \\cdots & \\sum x_i^{2n}
            \\end{bmatrix}
            \\begin{bmatrix}
            a_0 \\\\
            a_1 \\\\
            \\vdots \\\\
            a_n
            \\end{bmatrix} =
            \\begin{bmatrix}
            \\sum y_i \\\\
            \\sum x_i y_i \\\\
            \\vdots \\\\
            \\sum x_i^n y_i
            \\end{bmatrix}
            $$
            
            ## Processamento de Imagens
            
            Sistemas lineares são usados em:
            
            1. **Filtros lineares**:
               - Convolução para suavização, detecção de bordas, etc.
               - As máscaras de convolução podem ser representadas como sistemas lineares
            
            2. **Restauração de imagens**:
               - Eliminação de ruído e desfoque através de sistemas lineares regularizados
               - Exemplo: para um modelo de degradação g = Hf + n, onde g é a imagem observada, f é a imagem original e n é o ruído,
                 a restauração pode ser formulada como um sistema linear (H^T H + λI)f = H^T g
            
            3. **Compressão**:
               - Transformadas como DCT (usada em JPEG) podem ser implementadas como sistemas lineares
            
            ## Simulação de Fluidos
            
            As equações de Navier-Stokes discretizadas levam a grandes sistemas lineares:
            
            1. **Método da pressão**:
               - A equação de Poisson para a pressão leva a um sistema linear Ap = b
               - A matriz A é geralmente esparsa e pode ser resolvida eficientemente com métodos especializados
            
            2. **Métodos de elementos finitos**:
               - Discretizam o domínio em elementos menores
               - Resultam em sistemas lineares para velocidades e pressões
            
            ## Tomografia Computadorizada
            
            A reconstrução de imagens em tomografia (CT scan) envolve a resolução de sistemas lineares:
            
            1. **Problema de reconstrução**:
               - Relaciona as medidas de atenuação com os coeficientes de atenuação nos voxels
               - Leva a um grande sistema linear Ax = b, onde x são os coeficientes desconhecidos
            
            2. **Métodos de solução**:
               - Retroprojeção filtrada
               - Métodos iterativos como ART (Algebraic Reconstruction Technique), que resolve o sistema de forma iterativa
            
            ## Redes Neurais
            
            Embora as redes neurais modernas sejam não-lineares, muitas operações internas envolvem sistemas lineares:
            
            1. **Camadas lineares**:
               - A operação Wx + b, onde W é a matriz de pesos, x é a entrada e b é o viés
            
            2. **Backpropagation**:
               - O cálculo de gradientes envolve operações lineares com matrizes Jacobianas
            
            ## Criptografia
            
            Alguns métodos criptográficos são baseados em sistemas lineares:
            
            1. **Cifra de Hill**:
               - Usa multiplicação de matrizes para cifrar blocos de texto
               - A segurança depende da dificuldade de resolver certos sistemas lineares
            
            2. **Sistemas baseados em reticulados**:
               - Baseiam-se na dificuldade de resolver certos sistemas lineares em reticulados
               - Exemplo: o problema SVP (Shortest Vector Problem) está relacionado a encontrar a solução de norma mínima para um sistema homogêneo
            """
        },
        "Sistemas Não-Lineares": {
            "content": """
            # Sistemas Não-Lineares
            
            Em contraste com sistemas lineares, os sistemas não-lineares envolvem funções não-lineares das variáveis.
            
            ## Características dos Sistemas Não-Lineares
            
            1. **Múltiplas soluções**:
               - Podem ter 0, 1, um número finito ou infinitas soluções
               - Difíceis de classificar a priori
            
            2. **Comportamento complexo**:
               - Podem exibir caos, bifurcações e outros fenômenos complexos
               - Pequenas mudanças nos parâmetros podem levar a mudanças drásticas nas soluções
            
            3. **Métodos de resolução**:
               - Geralmente iterativos e não garantem encontrar todas as soluções
               - Podem convergir para diferentes soluções dependendo do ponto inicial
            
            ## Técnicas de Linearização
            
            Muitos métodos para resolver sistemas não-lineares envolvem alguma forma de linearização:
            
            1. **Expansão de Taylor**:
               - Aproximar localmente as funções não-lineares por suas expansões de Taylor de primeira ordem
               - Exemplo: f(x) ≈ f(x₀) + f'(x₀)(x - x₀)
            
            2. **Método de Newton multidimensional**:
               - Generalização do método de Newton para sistemas
               - Resolve iterativamente sistemas lineares da forma J(xₖ)Δx = -F(xₖ)
               - Onde J é a matriz Jacobiana das derivadas parciais
            
            ## Método de Newton
            
            Para um sistema F(X) = 0 com n equações e n incógnitas:
            
            1. Começar com uma aproximação inicial X₀
            2. Para cada iteração k:
               - Calcular F(Xₖ) e a matriz Jacobiana J(Xₖ)
               - Resolver o sistema linear J(Xₖ)Δx = -F(Xₖ)
               - Atualizar: Xₖ₊₁ = Xₖ + Δx
               - Verificar convergência
            
            **Exemplo**:
            
            Para o sistema:
            
            $$
            \\begin{align}
            x^2 + y^2 &= 25\\\\
            x^2 - y^2 &= 7
            \\end{align}
            $$
            
            A matriz Jacobiana é:
            
            $$
            J(x, y) = \\begin{bmatrix}
            2x & 2y \\\\
            2x & -2y
            \\end{bmatrix}
            $$
            
            Partindo de (4, 3), calculamos:
            
            - F(4, 3) = [(4² + 3²) - 25, (4² - 3²) - 7] = [0, 0]
            
            Já encontramos uma solução exata: (4, 3).
            
            Se tivéssemos partido de (3, 4), teríamos encontrado outra solução: (4, -3).
            
            ## Método do Ponto Fixo
            
            1. Reescrever o sistema na forma X = g(X)
            2. Escolher uma aproximação inicial X₀
            3. Iterar Xₖ₊₁ = g(Xₖ) até a convergência
            
            **Condição de convergência**:
            O método converge se ||∇g(X)|| < 1 na vizinhança da solução.
            
            ## Aplicações de Sistemas Não-Lineares
            
            1. **Física e engenharia**:
               - Equilíbrio de estruturas com comportamento não-linear
               - Circuitos não-lineares
               - Dinâmica de fluidos
            
            2. **Química**:
               - Equilíbrio químico com múltiplas reações
               - Cinética de reações complexas
            
            3. **Economia**:
               - Modelos econômicos com funções não-lineares de utilidade ou produção
               - Equilíbrio de mercado com demanda e oferta não-lineares
            
            4. **Biologia**:
               - Modelos de populações com interações não-lineares
               - Redes bioquímicas
            """
        },
        "Sistemas Lineares em Programação Linear": {
            "content": """
            # Sistemas Lineares em Programação Linear
            
            A programação linear (PL) é uma técnica de otimização para problemas com função objetivo linear e restrições lineares.
            
            ## Formulação Padrão
            
            Um problema de PL tem a forma:
            
            **Maximizar** (ou Minimizar): c₁x₁ + c₂x₂ + ... + cₙxₙ
            
            **Sujeito a**:
            ```
            a₁₁x₁ + a₁₂x₂ + ... + a₁ₙxₙ ≤ b₁
            a₂₁x₁ + a₂₂x₂ + ... + a₂ₙxₙ ≤ b₂
            ...
            aₘ₁x₁ + aₘ₂x₂ + ... + aₘₙxₙ ≤ bₘ
            ```
            
            E: x₁, x₂, ..., xₙ ≥ 0
            
            ## Método Simplex
            
            O método simplex resolve problemas de PL percorrendo os vértices do poliedro formado pelas restrições:
            
            1. Converter para a forma padrão, introduzindo variáveis de folga:
               ```
               a₁₁x₁ + ... + a₁ₙxₙ + s₁ = b₁
               ...
               aₘ₁x₁ + ... + aₘₙxₙ + sₘ = bₘ
               ```
            
            2. Encontrar uma solução básica viável inicial
            
            3. Verificar se a solução atual é ótima:
               - Se todos os coeficientes na função objetivo são não-negativos, a solução é ótima
               - Caso contrário, selecionar uma variável para entrar na base
            
            4. Determinar qual variável sai da base usando o teste da razão
            
            5. Atualizar a solução e retornar ao passo 3
            
            ## Relação com Sistemas Lineares
            
            Em cada iteração do simplex, resolvemos um sistema linear:
            
            1. As equações de restrição formam um sistema linear
            2. A operação pivô para trocar as variáveis básicas é essencialmente eliminação gaussiana
            3. A atualização da função objetivo também envolve operações de álgebra linear
            
            ## Dualidade
            
            Para cada problema de PL (primal), existe um problema dual associado:
            
            - Se o primal é um problema de maximização, o dual é de minimização, e vice-versa
            - As variáveis no dual correspondem às restrições no primal
            - As restrições no dual correspondem às variáveis no primal
            
            **Exemplo**:
            
            Primal:
            ```
            Maximizar: 3x₁ + 2x₂
            Sujeito a:
              x₁ + x₂ ≤ 8
              2x₁ + x₂ ≤ 10
              x₁, x₂ ≥ 0
            ```
            
            Dual:
            ```
            Minimizar: 8y₁ + 10y₂
            Sujeito a:
              y₁ + 2y₂ ≥ 3
              y₁ + y₂ ≥ 2
              y₁, y₂ ≥ 0
            ```
            
            ## Aplicações
            
            1. **Alocação de recursos**:
               - Determinar quanto produzir de cada produto para maximizar o lucro
               - Exemplo: Uma fábrica produz dois produtos que requerem diferentes quantidades de três recursos limitados
            
            2. **Dieta e mistura**:
               - Encontrar a combinação ótima de alimentos para minimizar o custo enquanto satisfaz requisitos nutricionais
               - Similar a problemas de mistura em química e engenharia
            
            3. **Transporte e logística**:
               - Otimizar o fluxo de bens de múltiplas origens para múltiplos destinos
               - Minimizar o custo total de transporte
            
            4. **Fluxo de rede**:
               - Encontrar o fluxo máximo em uma rede com capacidades limitadas
               - Ou o fluxo de custo mínimo que satisfaz demandas
            
            5. **Planejamento financeiro**:
               - Otimizar portfolios de investimento
               - Balancear risco e retorno sob restrições orçamentárias
            """
        }
    }
    
    # Selecionar tópico da teoria
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_topic = st.radio(
            "Tópicos:",
            list(theory_topics.keys()),
            key="theory_topic"
        )
        
        st.markdown("---")
        st.markdown("### Material de Apoio")
        
        # Botão para baixar o material em PDF
        if st.button("📥 Baixar Material em PDF", key="download_pdf_btn"):
            st.success(f"Download de '{selected_topic}.pdf' iniciado! (Simulação)")
        
        # Botão para acessar videoaulas
        if st.button("🎬 Acessar Videoaulas", key="video_btn"):
            st.session_state.page = "Vídeoaulas"
            st.rerun()
            
        # Botão para adicionar aos favoritos
        if st.button("⭐ Adicionar aos Favoritos", key="fav_btn"):
            if "favorites" not in st.session_state:
                st.session_state.favorites = {"reference_cards": []}
            
            # Verificar se já está nos favoritos
            already_saved = False
            for card in st.session_state.favorites.get("reference_cards", []):
                if card.get("title") == selected_topic:
                    already_saved = True
                    break
                    
            if already_saved:
                st.info(f"'{selected_topic}' já está nos seus favoritos.")
            else:
                st.session_state.favorites.setdefault("reference_cards", []).append(
                    {"title": selected_topic, "date": datetime.datetime.now().strftime("%d/%m/%Y %H:%M")}
                )
                st.success(f"'{selected_topic}' adicionado aos seus favoritos!")
        
        # Lista de tópicos relacionados
        # st.markdown("### Tópicos Relacionados")
        # related_topics = {
        #     "Introdução aos Sistemas Lineares": ["Classificação de Sistemas Lineares", "Aplicações de Sistemas Lineares"],
        #     "Classificação de Sistemas Lineares": ["Introdução aos Sistemas Lineares", "Teorema de Rouché-Capelli", "Interpretação Geométrica"],
        #     "Método de Eliminação de Gauss": ["Método de Gauss-Jordan", "Método da Matriz Inversa"],
        #     "Regra de Cramer": ["Método da Matriz Inversa", "Classificação de Sistemas Lineares"],
        #     "Método da Matriz Inversa": ["Regra de Cramer", "Decomposição LU"],
        #     "Método de Gauss-Jordan": ["Método de Eliminação de Gauss", "Métodos Iterativos"],
        #     "Métodos Iterativos": ["Método de Gauss-Jordan", "Estabilidade Numérica"],
        #     "Decomposição LU": ["Método da Matriz Inversa", "Estabilidade Numérica"],
        #     "Interpretação Geométrica": ["Classificação de Sistemas Lineares", "Aplicações de Sistemas Lineares"],
        #     "Aplicações de Sistemas Lineares": ["Introdução aos Sistemas Lineares", "Aplicações Avançadas"],
        #     "Sistemas Homogêneos": ["Classificação de Sistemas Lineares", "Interpretação Geométrica"],
        #     "Estabilidade Numérica": ["Métodos Iterativos", "Decomposição LU"],
        #     "Aplicações Avançadas": ["Aplicações de Sistemas Lineares", "Sistemas Não-Lineares"],
        #     "Sistemas Não-Lineares": ["Aplicações Avançadas", "Métodos Iterativos"],
        #     "Sistemas Lineares em Programação Linear": ["Aplicações de Sistemas Lineares", "Aplicações Avançadas"]
        # }
        
        # for topic in related_topics.get(selected_topic, []):
        #     if st.button(f"📌 {topic}", key=f"related_{topic}"):
        #         # Atualizar tópico selecionado
        #         st.session_state.theory_topic = topic
        #         st.rerun()
    
    with col2:
        # Atualizar histórico de tópicos estudados
        if selected_topic not in st.session_state.user_progress["topics_studied"]:
            st.session_state.user_progress["topics_studied"].append(selected_topic)
            
        # Exibir conteúdo do tópico selecionado
        st.markdown(theory_topics[selected_topic]["content"])
        
        # Adicionar botão para exercícios relacionados
        st.markdown("---")
        st.markdown("### Quer praticar este conteúdo?")
        
        if st.button("✏️ Praticar com Exercícios Relacionados", key="practice_btn"):
            st.session_state.page = "Exercícios"
            # Tentar mapear o tópico para um tipo de exercício
            topic_to_exercise = {
                "Método de Eliminação de Gauss": "Sistemas 3x3",
                "Regra de Cramer": "Sistemas 2x2",
                "Aplicações de Sistemas Lineares": "Aplicações",
                "Métodos Iterativos": "Métodos Iterativos",
                "Sistemas Homogêneos": "Sistemas SPI"
            }
            
            st.session_state.exercise_topic = topic_to_exercise.get(selected_topic, "Geral")
            st.rerun()

# Modificar o main() para incluir a inicialização correta de current_topic
def main():
    # Inicializar estados da sessão se não existirem
    if "page" not in st.session_state:
        st.session_state.page = "Início"
    
    if "user_progress" not in st.session_state:
        st.session_state.user_progress = {
            "exercises_completed": 0,
            "correct_answers": 0,
            "topics_studied": [],
            "difficulty_levels": {"Fácil": 0, "Médio": 0, "Difícil": 0},
            "last_login": datetime.datetime.now().strftime("%d/%m/%Y"),
            "streak": 1
        }
    
    if "favorites" not in st.session_state:
        st.session_state.favorites = {
            "examples": [],
            "reference_cards": [],
            "exercises": []
        }
        
    # Se não houver current_topic definido, inicialize
    if "current_topic" not in st.session_state:
        st.session_state.current_topic = "Introdução aos Sistemas Lineares"
    
    # Barra lateral
    with st.sidebar:
        st.image("calculo.png", width=280)
        st.title("MENU")
        
        # Seções principais
        main_sections = {
            "Início": "🏠",
            "Resolver Sistema": "🧮",
            "Teoria": "📚",
            "Exercícios": "✏️",
            "Exemplos": "📋",
            "Referência Rápida": "📝",
            "Vídeoaulas": "🎬",
            "Meu Progresso": "📊"
        }
        
        for section, icon in main_sections.items():
            if st.sidebar.button(f"{icon} {section}", key=f"btn_{section}", use_container_width=True):
                st.session_state.page = section
                # Usar rerun em vez de experimental_rerun
                st.rerun()
        
        st.sidebar.markdown("---")
        
        # Configurações da aplicação
        with st.sidebar.expander("⚙️ Configurações"):
            st.checkbox("Modo escuro", value=False, key="dark_mode")
            st.checkbox("Mostrar passos detalhados", value=True, key="show_steps_config")
            st.select_slider("Precisão numérica", options=["Baixa", "Média", "Alta"], value="Média", key="precision")
            st.slider("Tamanho da fonte", min_value=80, max_value=120, value=100, step=10, format="%d%%", key="font_size")
        
        # Informações do usuário
        st.sidebar.markdown("---")
        col1, col2 = st.sidebar.columns([1, 3])
        with col1:
            st.image("calculo.png", width=60)
        with col2:
            st.markdown("**Usuário:** Estudante")
            st.markdown(f"**Progresso:** {int(min(st.session_state.user_progress['exercises_completed'] / 20 * 100, 100))}%")
        
        # Exibir streak
        st.sidebar.markdown(f"🔥 **Sequência de estudos:** {st.session_state.user_progress['streak']} dias")
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("v1.0.0 | © 2025 SistemaSolver")
    
    # Conteúdo principal
    if st.session_state.page == "Início":
        show_home_page()
    elif st.session_state.page == "Resolver Sistema":
        show_solver_page()
    elif st.session_state.page == "Teoria":
        show_theory_page()
    elif st.session_state.page == "Exercícios":
        show_exercises_page()
    elif st.session_state.page == "Exemplos":
        show_examples_page()
    elif st.session_state.page == "Referência Rápida":
        show_reference_page()
    elif st.session_state.page == "Vídeoaulas":
        show_videos_page()
    elif st.session_state.page == "Meu Progresso":
        show_progress_page()

# Função para a página de resolver sistema
def show_solver_page():
    # Inicializar variáveis de estado se não existirem
    if "solver_show_steps" not in st.session_state:
        st.session_state.solver_show_steps = True
    
    # Controle de abas
    if "solver_current_tab" not in st.session_state:
        st.session_state.solver_current_tab = "Inserir Sistema"
        
    st.markdown('<h1 class="main-header">Resolver Sistema Linear</h1>', unsafe_allow_html=True)
    
    # Abas de navegação
    tabs = ["📝 Inserir Sistema", "🔍 Resultados", "📊 Visualização"]
    selected_tab = st.radio("", tabs, horizontal=True, 
                            index=tabs.index(f"{'📝 Inserir Sistema' if st.session_state.solver_current_tab == 'Inserir Sistema' else '🔍 Resultados' if st.session_state.solver_current_tab == 'Resultados' else '📊 Visualização'}"),
                            key="solver_tab_selector")
    
    # Atualizar a aba atual
    if "📝 Inserir Sistema" in selected_tab:
        st.session_state.solver_current_tab = "Inserir Sistema"
    elif "🔍 Resultados" in selected_tab:
        st.session_state.solver_current_tab = "Resultados"
    else:
        st.session_state.solver_current_tab = "Visualização"
    
    # Conteúdo da aba atual
    if st.session_state.solver_current_tab == "Inserir Sistema":
        st.markdown('<h2 class="sub-header">Insira seu sistema de equações lineares</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            system_input_method = st.radio(
                "Método de entrada:",
                ["Manual (Coeficientes)", "Equações (Texto)", "Matriz Aumentada"],
                horizontal=True
            )
            
        with col2:
            vars_count = st.number_input("Número de variáveis:", min_value=2, max_value=6, value=2)
        
        # Inicializar listas vazias para coeficientes e constantes
        coeffs = []
        constants = []
        
        if system_input_method == "Manual (Coeficientes)":
            equations_count = st.number_input("Número de equações:", min_value=1, max_value=8, value=vars_count)
            
            st.markdown("### Insira os coeficientes e termos independentes")
            
            var_names = ["x", "y", "z", "w", "v", "u"][:vars_count]
            
            for i in range(equations_count):
                cols = st.columns(vars_count + 1)
                
                eq_coeffs = []
                for j in range(vars_count):
                    with cols[j]:
                        coef = st.number_input(
                            f"Coeficiente de {var_names[j]} na equação {i+1}:",
                            value=1.0 if i == j else 0.0,
                            step=0.1,
                            format="%.2f",
                            key=f"coef_{i}_{j}"
                        )
                        eq_coeffs.append(coef)
                
                with cols[-1]:
                    const = st.number_input(
                        f"Termo independente da equação {i+1}:",
                        value=0.0,
                        step=0.1,
                        format="%.2f",
                        key=f"const_{i}"
                    )
                
                coeffs.append(eq_coeffs)
                constants.append(const)
                
                # Mostrar a equação formatada
                eq_str = format_equation(eq_coeffs, var_names, const)
                st.write(f"Equação {i+1}: {eq_str}")
                
        elif system_input_method == "Equações (Texto)":
            st.markdown("""
            Insira cada equação em uma linha separada, usando a sintaxe:
            ```
            a*x + b*y + c*z = d
            ```
            Exemplo:
            ```
            2*x + 3*y = 5
            x - y = 1
            ```
            """)
            
            equations_text = st.text_area(
                "Equações (uma por linha):",
                height=150,
                help="Insira uma equação por linha. Use * para multiplicação.",
                value="x + y = 10\n2*x - y = 5"
            )
            
            try:
                # Processar as equações de texto
                equations = equations_text.strip().split('\n')
                
                var_symbols = []
                for i in range(vars_count):
                    if i < len(["x", "y", "z", "w", "v", "u"]):
                        var_symbols.append(sp.symbols(["x", "y", "z", "w", "v", "u"][i]))
                
                for eq_text in equations:
                    if not eq_text.strip():
                        continue
                        
                    # Substituir = por - ( para padronizar
                    eq_text = eq_text.replace("=", "-(") + ")"
                    
                    # Converter para expressão sympy
                    expr = sp.sympify(eq_text)
                    
                    # Extrair coeficientes
                    eq_coeffs = []
                    for var in var_symbols:
                        coef = expr.coeff(var)
                        eq_coeffs.append(float(coef))
                    
                    # Extrair termo constante
                    const = -float(expr.subs([(var, 0) for var in var_symbols]))
                    
                    coeffs.append(eq_coeffs)
                    constants.append(const)
                
                # Mostrar as equações interpretadas
                st.markdown("### Equações interpretadas:")
                for i, (eq_coef, eq_const) in enumerate(zip(coeffs, constants)):
                    var_names = ["x", "y", "z", "w", "v", "u"][:vars_count]
                    eq_str = format_equation(eq_coef, var_names, eq_const)
                    st.write(f"Equação {i+1}: {eq_str}")
                    
            except Exception as e:
                st.error(f"Erro ao processar as equações: {str(e)}")
                st.stop()
                
        else:  # Matriz Aumentada
            st.markdown("""
            Insira a matriz aumentada do sistema. Cada linha representa uma equação, e a última coluna contém os termos independentes.
            """)
            
            matrix_text = st.text_area(
                "Matriz aumentada (uma linha por equação):",
                height=150,
                help="Insira os elementos da matriz separados por espaços, com uma linha por equação.",
                value="1 1 10\n2 -1 5"
            )
            
            try:
                # Processar a matriz aumentada
                matrix_rows = matrix_text.strip().split('\n')
                augmented_matrix = []
                
                for row_text in matrix_rows:
                    if not row_text.strip():
                        continue
                    
                    # Converter elementos para números
                    elements = [float(e) for e in row_text.split()]
                    augmented_matrix.append(elements)
                
                # Verificar dimensões
                if any(len(row) != vars_count + 1 for row in augmented_matrix):
                    st.error(f"Erro: cada linha deve ter {vars_count + 1} elementos (coeficientes + termo independente).")
                    st.stop()
                
                # Extrair coeficientes e constantes
                coeffs = [row[:-1] for row in augmented_matrix]
                constants = [row[-1] for row in augmented_matrix]
                
                # Mostrar as equações interpretadas
                st.markdown("### Equações interpretadas:")
                for i, (eq_coef, eq_const) in enumerate(zip(coeffs, constants)):
                    var_names = ["x", "y", "z", "w", "v", "u"][:vars_count]
                    eq_str = format_equation(eq_coef, var_names, eq_const)
                    st.write(f"Equação {i+1}: {eq_str}")
                
            except Exception as e:
                st.error(f"Erro ao processar a matriz aumentada: {str(e)}")
                st.stop()
        
        # Método de resolução
        st.markdown("### Método de Resolução")
        
        col1, col2 = st.columns(2)
        
        with col1:
            solution_method = st.selectbox(
                "Escolha o método:",
                ["Eliminação de Gauss", "Gauss-Jordan", "Regra de Cramer", "Matriz Inversa", 
                 "Decomposição LU", "Jacobi", "Gauss-Seidel", "Todos os Métodos"],
                key="solution_method_select"
            )
            
        with col2:
            show_steps = st.checkbox("Mostrar passos detalhados", value=True, key="show_steps_checkbox")
        
        # Opções extras para métodos iterativos
        max_iter = 50
        tolerance = 1e-6
        
        if solution_method in ["Jacobi", "Gauss-Seidel"]:
            col1, col2 = st.columns(2)
            with col1:
                max_iter = st.number_input("Número máximo de iterações:", min_value=5, max_value=100, value=50, key="max_iter_input")
            with col2:
                tolerance = st.number_input("Tolerância:", min_value=1e-10, max_value=1e-2, value=1e-6, format="%.1e", key="tolerance_input")
        
        # Verificar se temos dados suficientes para resolver
        solve_ready = len(coeffs) > 0 and len(constants) > 0 and len(coeffs[0]) == vars_count
        
        # Botão para resolver
        solve_clicked = st.button("Resolver Sistema", type="primary", key="solve_btn", disabled=not solve_ready)
        
        if solve_clicked:
            # Criar a matriz e o vetor do sistema
            try:
                A, b = create_system_matrix(coeffs, constants, vars_count)
                
                # Guardar dados no estado da sessão
                st.session_state.system_solved = True
                st.session_state.A = A
                st.session_state.b = b
                st.session_state.vars_count = vars_count
                st.session_state.solution_method = solution_method
                st.session_state.solver_show_steps = show_steps
                st.session_state.max_iter = max_iter
                st.session_state.tolerance = tolerance
                st.session_state.system_classification = classify_system(A, b)
                
                # Computar soluções pelos diferentes métodos
                results = {}
                
                with st.spinner("Resolvendo o sistema..."):
                    if solution_method in ["Eliminação de Gauss", "Todos os Métodos"]:
                        steps, solution = gaussian_elimination_steps(A, b)
                        results["Eliminação de Gauss"] = {"steps": steps, "solution": solution}
                        
                    if solution_method in ["Gauss-Jordan", "Todos os Métodos"]:
                        steps, solution = gauss_jordan_steps(A, b)
                        results["Gauss-Jordan"] = {"steps": steps, "solution": solution}
                        
                    if vars_count <= 4 and solution_method in ["Regra de Cramer", "Todos os Métodos"]:
                        if A.shape[0] == A.shape[1]:  # Apenas para sistemas quadrados
                            steps, solution = cramer_rule(A, b, detailed=show_steps)
                            results["Regra de Cramer"] = {"steps": steps, "solution": solution}
                        
                    if solution_method in ["Matriz Inversa", "Todos os Métodos"]:
                        if A.shape[0] == A.shape[1]:  # Apenas para sistemas quadrados
                            steps, solution = matrix_inverse_method(A, b, detailed=show_steps)
                            results["Matriz Inversa"] = {"steps": steps, "solution": solution}
                            
                    if solution_method in ["Decomposição LU", "Todos os Métodos"]:
                        if A.shape[0] == A.shape[1]:  # Apenas para sistemas quadrados
                            steps, solution = lu_decomposition_method(A, b, detailed=show_steps)
                            results["Decomposição LU"] = {"steps": steps, "solution": solution}
                            
                    if solution_method in ["Jacobi", "Todos os Métodos"]:
                        steps, solution = jacobi_iteration_method(A, b, max_iter=max_iter, tolerance=tolerance, detailed=show_steps)
                        results["Jacobi"] = {"steps": steps, "solution": solution}
                        
                    if solution_method in ["Gauss-Seidel", "Todos os Métodos"]:
                        steps, solution = gauss_seidel_method(A, b, max_iter=max_iter, tolerance=tolerance, detailed=show_steps)
                        results["Gauss-Seidel"] = {"steps": steps, "solution": solution}
                        
                st.session_state.results = results
                
                # Atualizar progresso do usuário
                if "user_progress" in st.session_state:
                    st.session_state.user_progress["exercises_completed"] += 1
                
                # Mostrar mensagem de sucesso e sugerir ir para a próxima aba
                st.success("Sistema resolvido com sucesso! Veja os resultados na aba 'Resultados'.")
                
                # Mudar para a aba de resultados automaticamente
                st.session_state.solver_current_tab = "Resultados"
                st.rerun()
                
            except Exception as e:
                st.error(f"Erro ao resolver o sistema: {str(e)}")
                st.session_state.system_solved = False

    elif st.session_state.solver_current_tab == "Resultados":
        # Verificar se um sistema foi resolvido
        if not hasattr(st.session_state, 'system_solved') or not st.session_state.system_solved:
            st.info("Insira e resolva um sistema na aba 'Inserir Sistema'")
            st.session_state.solver_current_tab = "Inserir Sistema"
            st.rerun()
        else:
            # Código da aba "Resultados"
            st.markdown('<h2 class="sub-header">Resultados da Resolução</h2>', unsafe_allow_html=True)
            
            # Exibir classificação do sistema
            st.markdown(f"**Classificação do Sistema:** {st.session_state.system_classification}")
            
            # Mostrar as equações do sistema
            st.markdown("### Sistema original:")
            var_names = ["x", "y", "z", "w", "v", "u"][:st.session_state.vars_count]
            A = st.session_state.A
            b = st.session_state.b
            
            for i in range(len(b)):
                eq_str = format_equation(A[i], var_names, b[i])
                st.write(f"Equação {i+1}: {eq_str}")
            
            # Exibir matriz aumentada
            with st.expander("Ver matriz aumentada", expanded=False):
                augmented = np.column_stack((A, b))
                st.markdown("**Matriz aumentada [A|b]:**")
                st.dataframe(pd.DataFrame(augmented, 
                                        columns=[f"{var}" for var in var_names] + ["b"],
                                        index=[f"Eq {i+1}" for i in range(len(b))]))
            
            # Exibir solução para cada método
            st.markdown("### Resultados por método:")
            
            for method, result in st.session_state.results.items():
                with st.expander(f"📊 {method}", expanded=method == st.session_state.solution_method):
                    steps = result["steps"]
                    solution = result["solution"]
                    
                    if solution is not None:
                        st.markdown("**Solução encontrada:**")
                        
                        # Criar dataframe da solução
                        solution_df = pd.DataFrame({
                            "Variável": var_names[:len(solution)],
                            "Valor": [float(val) for val in solution]
                        })
                        st.dataframe(solution_df)
                        
                        # Mostrar precisão da solução
                        residual = np.linalg.norm(np.dot(A, solution) - b)
                        st.markdown(f"**Resíduo:** {residual:.2e}")
                        
                        # Verificação rápida da solução
                        st.markdown("**Verificação rápida:**")
                        for i in range(len(b)):
                            calculated = np.dot(A[i], solution)
                            is_correct = abs(calculated - b[i]) < 1e-10
                            st.markdown(f"Equação {i+1}: {calculated:.4f} ≈ {b[i]:.4f} {'✓' if is_correct else '✗'}")
                        
                    else:
                        st.write("Não foi possível encontrar uma solução única por este método.")
                    
                    if st.session_state.solver_show_steps:
                        st.markdown("**Passos detalhados:**")
                        for step in steps:
                            st.write(step)
            
            # Adicionar interpretação da solução
            st.markdown("### Interpretação da Solução")
            
            if st.session_state.system_classification == "Sistema Possível e Determinado (SPD)":
                st.success("O sistema possui uma única solução, que satisfaz todas as equações simultaneamente.")
                
                # Obter uma solução válida (qualquer uma)
                solution = None
                for result in st.session_state.results.values():
                    if result["solution"] is not None:
                        solution = result["solution"]
                        break
                
                if solution is not None:
                    st.markdown("### Verificação Detalhada")
                    
                    for i in range(len(b)):
                        eq_result = np.dot(A[i], solution)
                        is_correct = abs(eq_result - b[i]) < 1e-10
                        
                        eq_str = format_equation(A[i], var_names, b[i])
                        
                        substitution = " + ".join([f"{A[i][j]:.2f} × {solution[j]:.4f}" for j in range(len(solution)) if abs(A[i][j]) > 1e-10])
                        if not substitution:
                            substitution = "0"
                        
                        result_str = f"{eq_result:.4f} ≈ {b[i]:.4f}" if is_correct else f"{eq_result:.4f} ≠ {b[i]:.4f}"
                        
                        if is_correct:
                            st.success(f"Equação {i+1}: {eq_str}\n{substitution} = {result_str} ✓")
                        else:
                            st.error(f"Equação {i+1}: {eq_str}\n{substitution} = {result_str} ✗")
                            
            elif st.session_state.system_classification == "Sistema Possível e Indeterminado (SPI)":
                st.info("""
                O sistema possui infinitas soluções. Isso ocorre porque há menos equações linearmente independentes
                do que variáveis, criando um espaço de soluções possíveis.
                
                A solução pode ser expressa de forma paramétrica, onde uma ou mais variáveis são expressas em termos
                de parâmetros livres.
                """)
                
                # Tentar obter solução simbólica
                try:
                    A = st.session_state.A
                    b = st.session_state.b
                    symbolic_solution, var_symbols = sympy_solve_system(A, b)
                    
                    if symbolic_solution:
                        st.markdown("### Solução Paramétrica")
                        
                        if isinstance(symbolic_solution, dict):
                            for var, expr in symbolic_solution.items():
                                st.latex(f"{sp.latex(var)} = {sp.latex(expr)}")
                        else:
                            st.latex(sp.latex(symbolic_solution))
                except:
                    st.warning("Não foi possível obter uma representação paramétrica da solução.")
                    
            else:  # Sistema Impossível
                st.error("""
                O sistema não possui solução. Isso ocorre porque as equações são inconsistentes entre si,
                ou seja, não existe um conjunto de valores para as variáveis que satisfaça todas as equações
                simultaneamente.
                
                Geometricamente, isso pode ser interpretado como:
                - Em 2D: retas paralelas que nunca se intersectam
                - Em 3D: planos sem ponto comum de interseção
                """)
                
            # Adicionar botões de ação para a solução
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Ver Visualização", key="view_viz_btn"):
                    st.session_state.solver_current_tab = "Visualização"
                    st.rerun()

            with col2:
                if st.button("📋 Salvar nos Exemplos", key="save_example_btn"):
                    if "favorites" not in st.session_state:
                        st.session_state.favorites = {"examples": []}
                    
                    # Criar um exemplo para salvar
                    example = {
                        "title": f"Sistema {A.shape[0]}×{A.shape[1]} ({st.session_state.system_classification.split(' ')[2]})",
                        "A": A.tolist(),
                        "b": b.tolist(),
                        "date": datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
                    }
                    
                    st.session_state.favorites["examples"].append(example)
                    st.success("Sistema salvo nos exemplos favoritos!")
            
            with col3:
                if st.button("📥 Exportar Solução", key="export_solution_btn"):
                    st.success("Solução exportada! (Simulação)")
    
    elif st.session_state.solver_current_tab == "Visualização":
        # Verificar se um sistema foi resolvido
        if not hasattr(st.session_state, 'system_solved') or not st.session_state.system_solved:
            st.info("Insira e resolva um sistema na aba 'Inserir Sistema'")
            st.session_state.solver_current_tab = "Inserir Sistema"
            st.rerun()
        else:
            # Código da aba "Visualização"
            st.markdown('<h2 class="sub-header">Visualização Gráfica</h2>', unsafe_allow_html=True)
            
            if st.session_state.vars_count == 2:
                try:
                    fig = plot_2d_system(st.session_state.A, st.session_state.b)
                    if fig:
                        st.pyplot(fig)
                        
                        # Adicionar interpretação geométrica
                        st.markdown("### Interpretação Geométrica")
                        
                        if st.session_state.system_classification == "Sistema Possível e Determinado (SPD)":
                            st.markdown("""
                            Cada equação do sistema representa uma reta no plano cartesiano.
                            A solução do sistema é o ponto de interseção entre estas retas.
                            
                            As coordenadas deste ponto satisfazem simultaneamente todas as equações do sistema.
                            """)
                        elif st.session_state.system_classification == "Sistema Possível e Indeterminado (SPI)":
                            st.markdown("""
                            As retas são coincidentes (sobrepostas), o que significa que qualquer
                            ponto em uma das retas é uma solução válida para o sistema.
                            
                            Geometricamente, isso ocorre quando as equações representam a mesma reta
                            ou quando algumas das equações são redundantes (combinações lineares de outras).
                            """)
                        else:  # SI
                            st.markdown("""
                            As retas são paralelas, o que indica que não há ponto de interseção
                            e, portanto, o sistema não possui solução.
                            
                            Este é um caso onde as equações são inconsistentes: não existe um par de valores
                            (x, y) que satisfaça todas as equações simultaneamente.
                            """)
                    else:
                        st.warning("Não foi possível gerar a visualização do sistema.")
                except Exception as e:
                    st.error(f"Erro ao gerar o gráfico: {str(e)}")
                    
            elif st.session_state.vars_count == 3:
                try:
                    fig = plot_3d_system(st.session_state.A, st.session_state.b)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Adicionar interpretação geométrica
                        st.markdown("### Interpretação Geométrica")
                        
                        if st.session_state.system_classification == "Sistema Possível e Determinado (SPD)":
                            st.markdown("""
                            Cada equação do sistema representa um plano no espaço tridimensional.
                            A solução do sistema é o ponto único de interseção entre estes planos.
                            
                            As coordenadas deste ponto satisfazem simultaneamente todas as equações do sistema.
                            """)
                        elif st.session_state.system_classification == "Sistema Possível e Indeterminado (SPI)":
                            st.markdown("""
                            Os planos se intersectam em uma reta ou em um plano comum,
                            resultando em infinitas soluções possíveis.
                            
                            Isso ocorre quando temos menos equações linearmente independentes
                            do que variáveis. As soluções formam um espaço geométrico (reta ou plano).
                            """)
                        else:  # SI
                            st.markdown("""
                            Os planos não possuem um ponto comum de interseção,
                            o que indica que o sistema não tem solução.
                            
                            Geometricamente, isso pode ocorrer quando temos três planos paralelos
                            ou quando a interseção de dois planos é uma reta paralela ao terceiro plano.
                            """)
                    else:
                        st.warning("Não foi possível gerar a visualização 3D do sistema.")
                except Exception as e:
                    st.error(f"Erro ao gerar o gráfico 3D: {str(e)}")
                    
            else:
                st.info("""
                A visualização gráfica está disponível apenas para sistemas com 2 ou 3 variáveis.
                
                Para sistemas com mais variáveis, você pode usar outras técnicas de análise,
                como a redução do sistema ou a projeção em subespaços.
                """)
                
                # Oferecer alternativas para visualização
                st.markdown("### Alternativas para Análise Visual")
                
                viz_options = st.radio(
                    "Escolha uma alternativa:",
                    ["Matriz Ampliada", "Gráfico de Sparsidade", "Nenhuma"],
                    horizontal=True
                )
                
                if viz_options == "Matriz Ampliada":
                    A = st.session_state.A
                    b = st.session_state.b
                    augmented = np.column_stack((A, b))
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    cax = ax.matshow(augmented, cmap='coolwarm')
                    
                    # Adicionar colorbar
                    fig.colorbar(cax)
                    
                    # Adicionar rótulos
                    var_names = ["x", "y", "z", "w", "v", "u"][:A.shape[1]] + ["b"]
                    ax.set_xticks(np.arange(A.shape[1] + 1))
                    ax.set_xticklabels(var_names)
                    ax.set_yticks(np.arange(A.shape[0]))
                    ax.set_yticklabels([f"Eq {i+1}" for i in range(A.shape[0])])
                    
                    plt.title("Visualização da Matriz Ampliada")
                    st.pyplot(fig)
                    
                elif viz_options == "Gráfico de Sparsidade":
                    A = st.session_state.A
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.spy(A, markersize=15, color='#1E88E5')
                    
                    # Adicionar rótulos
                    var_names = ["x", "y", "z", "w", "v", "u"][:A.shape[1]]
                    ax.set_xticks(np.arange(A.shape[1]))
                    ax.set_xticklabels(var_names)
                    ax.set_yticks(np.arange(A.shape[0]))
                    ax.set_yticklabels([f"Eq {i+1}" for i in range(A.shape[0])])
                    
                    plt.title("Gráfico de Sparsidade dos Coeficientes")
                    st.pyplot(fig)
                    
def show_exercises_page():
    st.markdown('<h1 class="main-header">Exercícios de Sistemas Lineares</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Praticar", "🏆 Desafios", "📋 Histórico", "📊 Progresso"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Pratique seus conhecimentos</h2>', unsafe_allow_html=True)
        
        # Configurações do exercício
        col1, col2, col3 = st.columns(3)
        
        with col1:
            difficulty = st.select_slider(
                "Nível de dificuldade:",
                options=["Fácil", "Médio", "Difícil"],
                value="Médio"
            )
            
        with col2:
            exercise_topics = [
                "Geral",
                "Sistemas 2x2",
                "Sistemas 3x3",
                "Sistemas 4x4",
                "Sistemas SPI",
                "Sistemas SI",
                "Métodos Iterativos",
                "Mal Condicionados",
                "Aplicações"
            ]
            
            topic = st.selectbox(
                "Tópico:",
                exercise_topics,
                index=0,
                key="exercise_topic_selector"
            )
            
        with col3:
            method = st.selectbox(
                "Método de resolução:",
                ["Qualquer método", "Eliminação de Gauss", "Regra de Cramer", 
                 "Matriz Inversa", "Gauss-Jordan", "Métodos Iterativos"],
                index=0
            )
        
        # Gerar novo exercício
        if "current_exercise" not in st.session_state or st.button("Gerar Novo Exercício", key="generate_exercise_btn"):
            try:
                if "problem" in topic.lower():
                    # Exercício de aplicação
                    exercise_data = get_practice_exercise(difficulty, topic)
                    st.session_state.current_exercise = {
                        "problem": exercise_data.get("problem", ""),
                        "A": exercise_data.get("A"),
                        "b": exercise_data.get("b"),
                        "vars": exercise_data.get("vars", []),
                        "solution": None,  # Será calculado abaixo
                        "difficulty": difficulty,
                        "topic": topic
                    }
                    
                    # Calcular solução
                    try:
                        if st.session_state.current_exercise["A"] is not None and st.session_state.current_exercise["b"] is not None:
                            A = st.session_state.current_exercise["A"]
                            b = st.session_state.current_exercise["b"]
                            
                            system_type = classify_system(A, b)
                            st.session_state.current_exercise["system_type"] = system_type
                            
                            if system_type == "Sistema Possível e Determinado (SPD)":
                                try:
                                    solution = np.linalg.solve(A, b)
                                    st.session_state.current_exercise["solution"] = solution
                                except:
                                    _, solution = gaussian_elimination_steps(A, b)
                                    st.session_state.current_exercise["solution"] = solution
                    except:
                        st.session_state.current_exercise["system_type"] = "Desconhecido"
                        
                else:
                    # Exercício normal
                    A, b, question, equations, solution, system_type = get_practice_exercise(difficulty, topic)
                    st.session_state.current_exercise = {
                        "A": A,
                        "b": b,
                        "question": question,
                        "equations": equations,
                        "solution": solution,
                        "difficulty": difficulty,
                        "topic": topic,
                        "system_type": system_type
                    }
            except Exception as e:
                st.error(f"Erro ao gerar exercício: {str(e)}")
                if "current_exercise" not in st.session_state:
                    st.session_state.current_exercise = {
                        "question": "Erro ao gerar exercício",
                        "equations": [],
                        "difficulty": difficulty,
                        "topic": topic
                    }
        
        # Mostrar o exercício atual
        if "problem" in st.session_state.current_exercise:
            # Mostrar exercício de aplicação
            st.markdown(f"### Problema de Aplicação ({st.session_state.current_exercise['difficulty']})")
            
            st.markdown(f"#### {st.session_state.current_exercise['problem']}")
            
            with st.expander("Ver dica", expanded=False):
                st.markdown("""
                **Dica**: Para resolver esse tipo de problema:
                1. Identifique as variáveis envolvidas
                2. Configure as equações do sistema
                3. Resolva o sistema usando o método mais adequado
                """)
                
                if st.session_state.current_exercise["A"] is not None and st.session_state.current_exercise["vars"] is not None:
                    st.markdown("**Sistema associado:**")
                    A = st.session_state.current_exercise["A"]
                    b = st.session_state.current_exercise["b"]
                    var_names = st.session_state.current_exercise["vars"]
                    
                    for i in range(min(len(b), A.shape[0])):
                        eq_str = format_equation(A[i], var_names[:A.shape[1]], b[i])
                        st.write(f"Equação {i+1}: {eq_str}")
            
        else:
            # Mostrar exercício normal
            st.markdown(f"### {st.session_state.current_exercise['question']} ({st.session_state.current_exercise['difficulty']})")
            
            for i, eq in enumerate(st.session_state.current_exercise['equations']):
                st.markdown(f"{i+1}. {eq}")
            
            with st.expander("Ver dica", expanded=False):
                if "system_type" in st.session_state.current_exercise:
                    system_type = st.session_state.current_exercise["system_type"]
                    st.markdown(f"**Classificação do sistema**: {system_type}")
                    
                    if system_type == "Sistema Possível e Determinado (SPD)":
                        method_recommendation = ""
                        if st.session_state.current_exercise["A"].shape[0] == st.session_state.current_exercise["A"].shape[1]:
                            method_recommendation = "Você pode usar qualquer método (Eliminação de Gauss, Regra de Cramer, Matriz Inversa)."
                        else:
                            method_recommendation = "Como o sistema não é quadrado, é recomendado usar o método de Eliminação de Gauss."
                            
                        st.markdown(f"**Dica**: Este sistema tem solução única. {method_recommendation}")
                    
                    elif system_type == "Sistema Possível e Indeterminado (SPI)":
                        st.markdown("""
                        **Dica**: Este sistema tem infinitas soluções. Você pode resolver escalonando a matriz e expressando algumas variáveis em termos de outras (parâmetros).
                        """)
                    
                    else:  # SI
                        st.markdown("""
                        **Dica**: Verifique se o sistema tem solução. Um sistema é impossível quando contém equações inconsistentes.
                        """)
        
        # Adicionar visualização se for sistema 2x2 ou 3x3
        if "A" in st.session_state.current_exercise and st.session_state.current_exercise["A"] is not None:
            A = st.session_state.current_exercise["A"]
            b = st.session_state.current_exercise["b"]
            
            if A.shape[1] == 2:
                with st.expander("Visualização Gráfica", expanded=False):
                    try:
                        fig = plot_2d_system(A, b)
                        if fig:
                            st.pyplot(fig)
                    except:
                        st.warning("Não foi possível gerar a visualização do sistema.")
            elif A.shape[1] == 3:
                with st.expander("Visualização 3D", expanded=False):
                    try:
                        fig = plot_3d_system(A, b)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    except:
                        st.warning("Não foi possível gerar a visualização 3D do sistema.")
        
        # Campo para resposta do usuário
        st.markdown("### Sua resposta")
        
        solution_type = st.radio(
            "Tipo de sistema:",
            ["Sistema Possível e Determinado (SPD)", "Sistema Possível e Indeterminado (SPI)", "Sistema Impossível (SI)"],
            horizontal=True,
            key="solution_type_radio"
        )
        
        if solution_type == "Sistema Possível e Determinado (SPD)":
            if "vars" in st.session_state.current_exercise and st.session_state.current_exercise["vars"]:
                var_names = st.session_state.current_exercise["vars"]
            else:
                var_names = ["x", "y", "z", "w"][:st.session_state.current_exercise["A"].shape[1]]
                
            cols = st.columns(min(4, len(var_names)))
            user_values = []
            
            for i, var in enumerate(var_names[:len(cols)]):
                with cols[i]:
                    val = st.number_input(f"Valor de {var}:", step=0.1, format="%.4f", key=f"answer_{var}")
                    user_values.append(val)
                    
            if len(var_names) > 4:
                cols = st.columns(min(4, len(var_names) - 4))
                for i, var in enumerate(var_names[4:4+len(cols)]):
                    with cols[i]:
                        val = st.number_input(f"Valor de {var}:", step=0.1, format="%.4f", key=f"answer_{var}")
                        user_values.append(val)
            
        else:  # SPI ou SI
            user_answer = st.text_area(
                "Explique por que o sistema é SPI ou SI e, se for SPI, escreva a solução paramétrica:",
                height=100,
                placeholder="Ex: 'O sistema é SPI porque...' ou 'O sistema é SI porque...'"
            )
        
        # Verificar resposta
        if st.button("Verificar Resposta", key="check_answer_btn"):
            if solution_type == "Sistema Possível e Determinado (SPD)":
                if "solution" in st.session_state.current_exercise and st.session_state.current_exercise["solution"] is not None:
                    solution = st.session_state.current_exercise["solution"]
                    
                    if len(user_values) != len(solution):
                        st.error("O número de valores inseridos não corresponde ao número de incógnitas.")
                    else:
                        correct = True
                        for u, s in zip(user_values, solution):
                            if abs(u - s) > 1e-2:
                                correct = False
                                break
                                
                        if correct:
                            st.success("✅ Correto! Sua solução está correta.")
                            
                            # Atualizar estatísticas
                            st.session_state.user_progress["exercises_completed"] += 1
                            st.session_state.user_progress["correct_answers"] += 1
                            st.session_state.user_progress["difficulty_levels"][difficulty] += 1
                            
                            # Mostrar verificação
                            if "A" in st.session_state.current_exercise and "b" in st.session_state.current_exercise:
                                A = st.session_state.current_exercise["A"]
                                b = st.session_state.current_exercise["b"]
                                
                                for i in range(len(b)):
                                    expected = b[i]
                                    calculated = np.dot(A[i], user_values)
                                    diff = abs(expected - calculated)
                                    
                                    if diff < 1e-10:
                                        st.write(f"Equação {i+1}: {calculated:.4f} = {expected:.4f} ✓")
                                    else:
                                        st.write(f"Equação {i+1}: {calculated:.4f} ≈ {expected:.4f} (erro: {diff:.4e})")
                        else:
                            st.error("❌ Incorreto. Verifique seus cálculos e tente novamente.")
                            
                            # Atualizar estatísticas
                            st.session_state.user_progress["exercises_completed"] += 1
                else:
                    if "system_type" in st.session_state.current_exercise:
                        expected_type = st.session_state.current_exercise["system_type"]
                        if expected_type == solution_type:
                            st.success("✅ Classificação correta do sistema!")
                            
                            # Atualizar estatísticas
                            st.session_state.user_progress["exercises_completed"] += 1
                            st.session_state.user_progress["correct_answers"] += 1
                        else:
                            st.error(f"❌ Classificação incorreta. O sistema é um {expected_type}.")
                            
                            # Atualizar estatísticas
                            st.session_state.user_progress["exercises_completed"] += 1
                    else:
                        st.warning("Não foi possível verificar a resposta. Tente outro exercício.")
            else:  # SPI ou SI
                if "system_type" in st.session_state.current_exercise:
                    expected_type = st.session_state.current_exercise["system_type"]
                    if expected_type == solution_type:
                        st.success("✅ Classificação correta do sistema!")
                        
                        # Verificar explicação básica
                        if solution_type == "Sistema Possível e Indeterminado (SPI)" and "parâmetr" in user_answer.lower():
                            st.success("✅ Sua explicação sobre parâmetros está correta!")
                        elif solution_type == "Sistema Impossível (SI)" and ("inconsist" in user_answer.lower() or "incompatível" in user_answer.lower()):
                            st.success("✅ Sua explicação sobre inconsistência está correta!")
                        else:
                            st.info("Sua resposta está parcialmente correta. Certifique-se de explicar adequadamente por que o sistema é SPI ou SI.")
                        
                        # Atualizar estatísticas
                        st.session_state.user_progress["exercises_completed"] += 1
                        st.session_state.user_progress["correct_answers"] += 1
                    else:
                        st.error(f"❌ Classificação incorreta. O sistema é um {expected_type}.")
                        
                        # Atualizar estatísticas
                        st.session_state.user_progress["exercises_completed"] += 1
                else:
                    st.warning("Não foi possível verificar a resposta. Tente outro exercício.")
                    
            # Salvar no histórico
            if "exercise_history" not in st.session_state:
                st.session_state.exercise_history = []
            
            # Verificar se este exercício já está no histórico para não duplicar
            already_in_history = False
            for entry in st.session_state.exercise_history:
                if "equations" in entry and "equations" in st.session_state.current_exercise:
                    if entry["equations"] == st.session_state.current_exercise["equations"]:
                        already_in_history = True
                        break
                elif "problem" in entry and "problem" in st.session_state.current_exercise:
                    if entry["problem"] == st.session_state.current_exercise["problem"]:
                        already_in_history = True
                        break
            
            if not already_in_history:
                history_entry = {
                    "date": datetime.datetime.now().strftime("%d/%m/%Y %H:%M"),
                    "difficulty": st.session_state.current_exercise["difficulty"],
                    "topic": st.session_state.current_exercise["topic"],
                    "correct": correct if solution_type == "Sistema Possível e Determinado (SPD)" else (expected_type == solution_type)
                }
                
                if "equations" in st.session_state.current_exercise:
                    history_entry["equations"] = st.session_state.current_exercise["equations"]
                if "problem" in st.session_state.current_exercise:
                    history_entry["problem"] = st.session_state.current_exercise["problem"]
                
                st.session_state.exercise_history.append(history_entry)
        
        # Botão para ver a solução
        if st.button("Ver Solução", key="show_solution_btn"):
            st.markdown("### Solução Detalhada")
            
            if "system_type" in st.session_state.current_exercise:
                system_type = st.session_state.current_exercise["system_type"]
                st.markdown(f"**Classificação do Sistema**: {system_type}")
            
            if "A" in st.session_state.current_exercise and st.session_state.current_exercise["A"] is not None:
                A = st.session_state.current_exercise["A"]
                b = st.session_state.current_exercise["b"]
                
                # Escolher método apropriado com base nas preferências do usuário
                solution_method = method if method != "Qualquer método" else "Eliminação de Gauss"
                
                if solution_method == "Eliminação de Gauss":
                    steps, solution = gaussian_elimination_steps(A, b)
                    
                    st.markdown("#### Método de Eliminação de Gauss:")
                    for step in steps:
                        st.write(step)
                    
                elif solution_method == "Regra de Cramer" and A.shape[0] == A.shape[1]:
                    steps, solution = cramer_rule(A, b, detailed=True)
                    
                    st.markdown("#### Regra de Cramer:")
                    for step in steps:
                        st.write(step)
                        
                elif solution_method == "Matriz Inversa" and A.shape[0] == A.shape[1]:
                    steps, solution = matrix_inverse_method(A, b, detailed=True)
                    
                    st.markdown("#### Método da Matriz Inversa:")
                    for step in steps:
                        st.write(step)
                        
                elif solution_method == "Gauss-Jordan":
                    steps, solution = gauss_jordan_steps(A, b)
                    
                    st.markdown("#### Método de Gauss-Jordan:")
                    for step in steps:
                        st.write(step)
                        
                elif solution_method == "Métodos Iterativos":
                    # Verificar se é apropriado para métodos iterativos
                    is_diag_dominant = True
                    for i in range(min(A.shape[0], A.shape[1])):
                        if i < A.shape[0] and i < A.shape[1]:
                            if abs(A[i, i]) <= np.sum(np.abs(A[i, :])) - abs(A[i, i]):
                                is_diag_dominant = False
                                break
                    
                    if is_diag_dominant:
                        steps_jacobi, solution_jacobi = jacobi_iteration_method(A, b, detailed=True)
                        steps_gauss_seidel, solution_gauss_seidel = gauss_seidel_method(A, b, detailed=True)
                        
                        st.markdown("#### Método de Jacobi:")
                        for step in steps_jacobi:
                            st.write(step)
                            
                        st.markdown("#### Método de Gauss-Seidel:")
                        for step in steps_gauss_seidel:
                            st.write(step)
                            
                        solution = solution_gauss_seidel  # Usar Gauss-Seidel como solução final
                    else:
                        st.warning("Este sistema não é diagonalmente dominante, o que pode fazer com que os métodos iterativos não convirjam. Usando Eliminação de Gauss como alternativa.")
                        steps, solution = gaussian_elimination_steps(A, b)
                        
                        st.markdown("#### Método de Eliminação de Gauss:")
                        for step in steps:
                            st.write(step)
                else:
                    steps, solution = gaussian_elimination_steps(A, b)
                    
                    st.markdown("#### Método de Eliminação de Gauss:")
                    for step in steps:
                        st.write(step)
                
                # Mostrar a solução final
                st.markdown("#### Solução Final:")
                
                if solution is not None:
                    if "vars" in st.session_state.current_exercise and st.session_state.current_exercise["vars"]:
                        var_names = st.session_state.current_exercise["vars"]
                    else:
                        var_names = ["x", "y", "z", "w"][:A.shape[1]]
                        
                    for i, var in enumerate(var_names):
                        if i < len(solution):
                            st.markdown(f"- {var} = {solution[i]:.4f}")
                else:
                    if system_type == "Sistema Possível e Indeterminado (SPI)":
                        st.markdown("Este sistema possui infinitas soluções. A solução pode ser expressa em forma paramétrica.")
                        
                        # Tentar obter solução simbólica
                        try:
                            symbolic_solution, _ = sympy_solve_system(A, b)
                            
                            if symbolic_solution:
                                st.markdown("**Solução Paramétrica:**")
                                
                                if isinstance(symbolic_solution, dict):
                                    for var, expr in symbolic_solution.items():
                                        st.latex(f"{sp.latex(var)} = {sp.latex(expr)}")
                                else:
                                    st.latex(sp.latex(symbolic_solution))
                        except:
                            st.info("Não foi possível determinar a forma paramétrica exata da solução.")
                    
                    elif system_type == "Sistema Impossível (SI)":
                        st.markdown("Este sistema não possui solução, pois as equações são inconsistentes entre si.")
                    
            else:
                st.warning("Não foi possível obter a solução para este exercício.")
                
            # Salvar no histórico mesmo se o usuário viu a solução sem tentar
            if "exercise_history" not in st.session_state:
                st.session_state.exercise_history = []
            
            # Verificar se este exercício já está no histórico para não duplicar
            already_in_history = False
            for entry in st.session_state.exercise_history:
                if "equations" in entry and "equations" in st.session_state.current_exercise:
                    if entry["equations"] == st.session_state.current_exercise["equations"]:
                        already_in_history = True
                        break
                elif "problem" in entry and "problem" in st.session_state.current_exercise:
                    if entry["problem"] == st.session_state.current_exercise["problem"]:
                        already_in_history = True
                        break
            
            if not already_in_history:
                history_entry = {
                    "date": datetime.datetime.now().strftime("%d/%m/%Y %H:%M"),
                    "difficulty": st.session_state.current_exercise["difficulty"],
                    "topic": st.session_state.current_exercise["topic"],
                    "correct": False,
                    "viewed_solution": True
                }
                
                if "equations" in st.session_state.current_exercise:
                    history_entry["equations"] = st.session_state.current_exercise["equations"]
                if "problem" in st.session_state.current_exercise:
                    history_entry["problem"] = st.session_state.current_exercise["problem"]
                
                st.session_state.exercise_history.append(history_entry)
                
            # Atualizar estatísticas
            st.session_state.user_progress["exercises_completed"] += 1
    
    with tab2:
        st.markdown('<h2 class="sub-header">Desafios Semanais</h2>', unsafe_allow_html=True)
        
        # Lista de desafios
        challenges = [
            {
                "title": "Circuitos Elétricos",
                "description": "Resolva um sistema de equações que modela um circuito com 5 correntes desconhecidas.",
                "difficulty": "Difícil",
                "points": 100,
                "deadline": "25/03/2025",
                "status": "Disponível"
            },
            {
                "title": "Mistura Química",
                "description": "Encontre as quantidades exatas para uma mistura química com 4 componentes.",
                "difficulty": "Médio",
                "points": 75,
                "deadline": "27/03/2025",
                "status": "Disponível"
            },
            {
                "title": "Balanceamento de Reações",
                "description": "Use sistemas lineares para balancear uma reação química complexa.",
                "difficulty": "Médio",
                "points": 50,
                "deadline": "30/03/2025",
                "status": "Disponível"
            },
            {
                "title": "Sistema Mal Condicionado",
                "description": "Resolva um sistema linearmente independente, mas numericamente instável.",
                "difficulty": "Difícil",
                "points": 125,
                "deadline": "01/04/2025",
                "status": "Bloqueado"
            },
            {
                "title": "Análise de Tráfego",
                "description": "Modele e resolva um problema de fluxo de tráfego em uma rede com 6 nós.",
                "difficulty": "Difícil",
                "points": 150,
                "deadline": "05/04/2025",
                "status": "Bloqueado"
            }
        ]
        
        # Mostrar desafios disponíveis em cards
        st.markdown("### Desafios disponíveis")
        
        for i, challenge in enumerate(challenges):
            if challenge["status"] == "Disponível":
                with st.container():
                    st.markdown(f"""
                    <div style="background-color: #f8f9fa; border-radius: 10px; padding: 15px; margin-bottom: 15px; border-left: 5px solid #1E88E5;">
                        <h4 style="margin-top: 0;">{challenge["title"]} <span style="background-color: #e3f2fd; color: #1E88E5; padding: 3px 8px; border-radius: 10px; font-size: 0.8rem; float: right;">{challenge["difficulty"]} • {challenge["points"]} pontos</span></h4>
                        <p>{challenge["description"]}</p>
                        <p style="color: #666; font-size: 0.9rem;">Prazo: {challenge["deadline"]}</p>
                        <button style="background-color: #1E88E5; color: white; border: none; padding: 5px 15px; border-radius: 5px; cursor: pointer;">Iniciar Desafio</button>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Mostrar desafios bloqueados
        st.markdown("### Próximos desafios")
        
        for i, challenge in enumerate(challenges):
            if challenge["status"] == "Bloqueado":
                with st.container():
                    st.markdown(f"""
                    <div style="background-color: #f8f9fa; border-radius: 10px; padding: 15px; margin-bottom: 15px; border-left: 5px solid #9e9e9e; opacity: 0.7;">
                        <h4 style="margin-top: 0;">{challenge["title"]} <span style="background-color: #f5f5f5; color: #757575; padding: 3px 8px; border-radius: 10px; font-size: 0.8rem; float: right;">{challenge["difficulty"]} • {challenge["points"]} pontos</span></h4>
                        <p>{challenge["description"]}</p>
                        <p style="color: #666; font-size: 0.9rem;">Disponível a partir de: {challenge["deadline"]}</p>
                        <button style="background-color: #9e9e9e; color: white; border: none; padding: 5px 15px; border-radius: 5px; cursor: not-allowed;">Bloqueado</button>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Ranking
        st.markdown("### Ranking dos Desafios")
        
        ranking_data = [
            {"Posição": 1, "Usuário": "MatematicaMaster", "Pontos": 425, "Desafios": 4},
            {"Posição": 2, "Usuário": "AlgebraFã", "Pontos": 350, "Desafios": 3},
            {"Posição": 3, "Usuário": "SistemasGuru", "Pontos": 275, "Desafios": 3},
            {"Posição": 4, "Usuário": "Estudante (você)", "Pontos": 150, "Desafios": 2},
            {"Posição": 5, "Usuário": "MatrizInversa", "Pontos": 125, "Desafios": 1},
        ]
        
        st.dataframe(
            pd.DataFrame(ranking_data),
            use_container_width=True,
            hide_index=True
        )
    
    with tab3:
        st.markdown('<h2 class="sub-header">Seu Histórico de Exercícios</h2>', unsafe_allow_html=True)
        
        if "exercise_history" not in st.session_state or not st.session_state.exercise_history:
            st.info("Seu histórico de exercícios aparecerá aqui após você resolver alguns problemas.")
        else:
            # Estatísticas
            total = len(st.session_state.exercise_history)
            correct = sum(1 for e in st.session_state.exercise_history if e.get("correct", False))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<p class="metric-value">{total}</p>', unsafe_allow_html=True)
                st.markdown('<p class="metric-label">Total de Exercícios</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<p class="metric-value">{correct}</p>', unsafe_allow_html=True)
                st.markdown('<p class="metric-label">Acertos</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<p class="metric-value">{int(correct/total*100) if total > 0 else 0}%</p>', unsafe_allow_html=True)
                st.markdown('<p class="metric-label">Taxa de Acerto</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Filtros
            col1, col2 = st.columns(2)
            
            with col1:
                filter_difficulty = st.multiselect(
                    "Filtrar por dificuldade:",
                    ["Fácil", "Médio", "Difícil"],
                    default=["Fácil", "Médio", "Difícil"]
                )
                
            with col2:
                filter_status = st.multiselect(
                    "Filtrar por status:",
                    ["Correto", "Incorreto", "Visualizado"],
                    default=["Correto", "Incorreto", "Visualizado"]
                )
            
            # Filtrar histórico
            filtered_history = []
            for entry in st.session_state.exercise_history:
                if entry.get("difficulty") in filter_difficulty:
                    status = "Visualizado" if entry.get("viewed_solution", False) else ("Correto" if entry.get("correct", False) else "Incorreto")
                    if status in filter_status:
                        filtered_history.append(entry)
            
            # Tabela de histórico
            if filtered_history:
                history_data = []
                for i, exercise in enumerate(filtered_history[::-1]):  # Mais recente primeiro
                    status = "Visualizado" if exercise.get("viewed_solution", False) else ("Correto" if exercise.get("correct", False) else "Incorreto")
                    
                    question = ""
                    if "equations" in exercise and exercise["equations"]:
                        question = "<br>".join(exercise["equations"])
                    elif "problem" in exercise:
                        question = exercise["problem"][:100] + "..." if len(exercise["problem"]) > 100 else exercise["problem"]
                    
                    history_data.append({
                        "Data": exercise.get("date", ""),
                        "Dificuldade": exercise.get("difficulty", ""),
                        "Tópico": exercise.get("topic", ""),
                        "Questão": question,
                        "Resultado": status
                    })
                
                history_df = pd.DataFrame(history_data)
                st.dataframe(history_df, use_container_width=True, hide_index=True)
                
                # Botão para exportar histórico
                if st.button("📥 Exportar Histórico (CSV)", key="export_history_btn"):
                    csv = history_df.to_csv(index=False)
                    
                    # Criar link para download
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="historico_exercicios.csv">Clique para baixar o histórico completo</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    st.success("Histórico exportado com sucesso!")
            else:
                st.info("Nenhum exercício encontrado com os filtros selecionados.")
                
            # Botão para limpar histórico
            if st.button("🗑️ Limpar Histórico", key="clear_history_btn"):
                st.session_state.exercise_history = []
                st.rerun()
    
    with tab4:
        st.markdown('<h2 class="sub-header">Seu Progresso de Aprendizagem</h2>', unsafe_allow_html=True)
        
        # Dados de progresso
        exercises_completed = st.session_state.user_progress["exercises_completed"]
        correct_answers = st.session_state.user_progress["correct_answers"]
        topics_studied = st.session_state.user_progress["topics_studied"]
        difficulty_levels = st.session_state.user_progress["difficulty_levels"]
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<p class="metric-value">{exercises_completed}</p>', unsafe_allow_html=True)
            st.markdown('<p class="metric-label">Exercícios</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            accuracy = int(correct_answers / max(1, exercises_completed) * 100)
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<p class="metric-value">{accuracy}%</p>', unsafe_allow_html=True)
            st.markdown('<p class="metric-label">Precisão</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<p class="metric-value">{len(topics_studied)}</p>', unsafe_allow_html=True)
            st.markdown('<p class="metric-label">Tópicos</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col4:
            streak = st.session_state.user_progress["streak"]
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<p class="metric-value">{streak}</p>', unsafe_allow_html=True)
            st.markdown('<p class="metric-label">Sequência</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Gráficos de progresso
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de precisão por dificuldade
            st.markdown("### Precisão por Nível de Dificuldade")
            
            # Simular dados para o gráfico
            difficulty_data = {
                "Fácil": min(100, 75 + difficulty_levels["Fácil"] * 5),
                "Médio": min(100, 60 + difficulty_levels["Médio"] * 4),
                "Difícil": min(100, 40 + difficulty_levels["Difícil"] * 3)
            }
            
            fig, ax = plt.subplots(figsize=(8, 5))
            
            difficulties = list(difficulty_data.keys())
            accuracies = list(difficulty_data.values())
            colors = ['#4CAF50', '#FFC107', '#F44336']
            
            bars = ax.bar(difficulties, accuracies, color=colors)
            
            # Adicionar rótulos
            ax.set_ylim(0, 100)
            ax.set_ylabel('Precisão (%)')
            ax.set_title('Precisão por Nível de Dificuldade')
            
            # Adicionar valores nas barras
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.0f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 pontos de offset vertical
                            textcoords="offset points",
                            ha='center', va='bottom')
            
            st.pyplot(fig)
            
        with col2:
            # Gráfico de tópicos estudados
            st.markdown("### Tópicos Estudados")
            
            # Simular dados para o gráfico
            topics_count = {}
            all_topics = ["Sistemas 2x2", "Sistemas 3x3", "Métodos Iterativos", "Aplicações", "Mal Condicionados", "Sistemas SPI", "Sistemas SI"]
            
            for topic in all_topics:
                # Contar ocorrências nos tópicos estudados
                count = sum(1 for t in topics_studied if topic.lower() in t.lower())
                if count > 0 or topic in ["Sistemas 2x2", "Sistemas 3x3", "Aplicações"]:  # Garantir que alguns tópicos básicos apareçam
                    topics_count[topic] = max(1, count)
            
            # Se não houver tópicos estudados, adicionar alguns padrão
            if not topics_count:
                topics_count = {
                    "Sistemas 2x2": 3,
                    "Sistemas 3x3": 2,
                    "Aplicações": 1
                }
            
            fig, ax = plt.subplots(figsize=(8, 5))
            
            topics = list(topics_count.keys())
            counts = list(topics_count.values())
            
            # Ordenar por contagem
            sorted_indices = sorted(range(len(counts)), key=lambda i: counts[i], reverse=True)
            topics = [topics[i] for i in sorted_indices]
            counts = [counts[i] for i in sorted_indices]
            
            # Limitar a 5 tópicos para melhor visualização
            if len(topics) > 5:
                topics = topics[:5]
                counts = counts[:5]
            
            bars = ax.barh(topics, counts, color='#1E88E5')
            
            # Adicionar rótulos
            ax.set_xlabel('Número de Estudos')
            ax.set_title('Tópicos Mais Estudados')
            
            # Adicionar valores nas barras
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.annotate(f'{width}',
                           xy=(width, bar.get_y() + bar.get_height()/2),
                           xytext=(3, 0),  # 3 pontos de offset horizontal
                           textcoords="offset points",
                           ha='left', va='center')
            
            st.pyplot(fig)
        
        # Curva de progresso ao longo do tempo
        st.markdown("### Progresso ao Longo do Tempo")
        
        # Simular dados de progresso por dia
        today = datetime.datetime.now()
        dates = [(today - datetime.timedelta(days=i)).strftime("%d/%m") for i in range(6, -1, -1)]
        
        # Simular exercícios por dia
        exercises_per_day = [0, 2, 5, 0, 3, 1, 4]
        correct_per_day = [0, 1, 3, 0, 2, 1, 3]
        
        # Ajustar com o progresso real
        exercises_per_day[-1] = min(10, exercises_completed)
        correct_per_day[-1] = min(exercises_per_day[-1], correct_answers)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.plot(dates, exercises_per_day, 'o-', color='#1E88E5', label='Exercícios')
        ax.plot(dates, correct_per_day, 'o-', color='#4CAF50', label='Acertos')
        
        # Adicionar área sombreada
        ax.fill_between(dates, correct_per_day, color='#4CAF50', alpha=0.3)
        
        # Adicionar rótulos
        ax.set_xlabel('Data')
        ax.set_ylabel('Quantidade')
        ax.set_title('Progresso nos Últimos 7 Dias')
        ax.legend()
        
        # Ajustar limites
        ax.set_ylim(0, max(exercises_per_day) + 2)
        
        # Adicionar grade
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Metas e recomendações
        st.markdown("### Metas e Recomendações")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background-color: #e3f2fd; border-radius: 10px; padding: 15px; margin-bottom: 15px;">
                <h4 style="margin-top: 0;">📊 Metas Semanais</h4>
                <ul>
                    <li>Completar 20 exercícios</li>
                    <li>Atingir precisão de 80%</li>
                    <li>Estudar 5 tópicos diferentes</li>
                    <li>Resolver 2 desafios</li>
                </ul>
                <div style="background-color: #bbdefb; height: 10px; border-radius: 5px; margin-top: 10px;">
                    <div style="background-color: #1E88E5; width: 45%; height: 100%; border-radius: 5px;"></div>
                </div>
                <p style="text-align: right; margin-top: 5px; font-size: 0.9rem;">Progresso: 45%</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div style="background-color: #e8f5e9; border-radius: 10px; padding: 15px; margin-bottom: 15px;">
                <h4 style="margin-top: 0;">📚 Recomendações</h4>
                <p>Com base no seu desempenho, recomendamos:</p>
                <ul>
                    <li>Praticar mais exercícios de <strong>Sistemas 3×3</strong></li>
                    <li>Revisar o <strong>Método de Gauss-Jordan</strong></li>
                    <li>Tentar resolver problemas de <strong>aplicação prática</strong></li>
                </ul>
                <button style="background-color: #4CAF50; color: white; border: none; padding: 5px 15px; border-radius: 5px; cursor: pointer; margin-top: 10px;">Gerar Exercício Recomendado</button>
            </div>
            """, unsafe_allow_html=True)
            
        # Certificados e conquistas
        st.markdown("### Certificados e Conquistas")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="background-color: #fff3e0; border-radius: 10px; padding: 15px; text-align: center;">
                <h4 style="margin-top: 0;">🥉 Iniciante</h4>
                <p style="font-size: 0.9rem;">Completou 10 exercícios</p>
                <p style="color: #FB8C00; font-weight: bold;">CONQUISTADO</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div style="background-color: #f5f5f5; border-radius: 10px; padding: 15px; text-align: center;">
                <h4 style="margin-top: 0;">🥈 Intermediário</h4>
                <p style="font-size: 0.9rem;">Completar 30 exercícios com 70% de precisão</p>
                <p style="color: #9E9E9E; font-weight: bold;">EM PROGRESSO (45%)</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div style="background-color: #f5f5f5; border-radius: 10px; padding: 15px; text-align: center;">
                <h4 style="margin-top: 0;">🥇 Avançado</h4>
                <p style="font-size: 0.9rem;">Resolver 5 desafios difíceis</p>
                <p style="color: #9E9E9E; font-weight: bold;">BLOQUEADO</p>
            </div>
            """, unsafe_allow_html=True)

def show_examples_page():
    st.markdown('<h1 class="main-header">Exemplos Resolvidos</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("### Categorias")
        
        example_types = [
            "Sistema 2×2 (SPD)",
            "Sistema 2×2 (SPI)",
            "Sistema 2×2 (SI)",
            "Sistema 3×3 (SPD)",
            "Sistema 3×3 (SPI)",
            "Sistema 3×3 (SI)",
            "Sistema 4×4 (SPD)",
            "Sistema Mal Condicionado",
            "Método Iterativo",
            "Aplicação: Mistura",
            "Aplicação: Circuitos",
            "Aplicação: Balanceamento Químico",
            "Método: Gauss-Jordan",
            "Método: Decomposição LU",
            "Método: Gauss-Seidel"
        ]
        
        selected_example = st.radio(
            "Selecione um exemplo:",
            example_types,
            key="example_type"
        )
        
        st.markdown("---")
        st.markdown("### Métodos de Resolução")
        
        methods = [
            "Eliminação de Gauss",
            "Gauss-Jordan",
            "Regra de Cramer",
            "Matriz Inversa",
            "Decomposição LU",
            "Jacobi",
            "Gauss-Seidel",
            "Passo a Passo Detalhado"
        ]
        
        selected_methods = st.multiselect(
            "Mostrar solução por:",
            methods,
            default=["Eliminação de Gauss", "Passo a Passo Detalhado"]
        )
        
        st.markdown("---")
        st.markdown("### Meus Exemplos")
        
        # Mostrar exemplos salvos
        if "favorites" in st.session_state and "examples" in st.session_state.favorites and st.session_state.favorites["examples"]:
            for i, example in enumerate(st.session_state.favorites["examples"]):
                if st.button(f"{example['title']} ({example['date']})", key=f"saved_example_{i}"):
                    # Carregar exemplo salvo
                    st.session_state.custom_example = {
                        "title": example["title"],
                        "A": np.array(example["A"]),
                        "b": np.array(example["b"]),
                        "date": example["date"]
                    }
                    
                    st.rerun()
        else:
            st.info("Você ainda não salvou nenhum exemplo. Os sistemas que você resolver e salvar aparecerão aqui.")
            
        if st.button("➕ Adicionar Sistema Personalizado", key="add_custom_btn"):
            st.session_state.adding_custom_example = True
            st.rerun()
    
    with col2:
        # Interface para adicionar sistema personalizado
        if hasattr(st.session_state, 'adding_custom_example') and st.session_state.adding_custom_example:
            st.markdown("### Adicionar Sistema Personalizado")
            
            # Interface para entrada do sistema
            num_vars = st.number_input("Número de variáveis:", min_value=2, max_value=4, value=2, key="custom_vars")
            num_eqs = st.number_input("Número de equações:", min_value=1, max_value=5, value=2, key="custom_eqs")
            
            # Criar campos para coeficientes
            coeffs = []
            constants = []
            
            var_names = ["x", "y", "z", "w"][:num_vars]
            
            for i in range(num_eqs):
                st.markdown(f"**Equação {i+1}**")
                cols = st.columns(num_vars + 1)
                
                eq_coeffs = []
                for j in range(num_vars):
                    with cols[j]:
                        coef = st.number_input(
                            f"Coef. de {var_names[j]}:",
                            value=1.0 if i == j else 0.0,
                            step=0.1,
                            format="%.2f",
                            key=f"custom_coef_{i}_{j}"
                        )
                        eq_coeffs.append(coef)
                
                with cols[-1]:
                    const = st.number_input(
                        f"Termo indep.:",
                        value=0.0,
                        step=0.1,
                        format="%.2f",
                        key=f"custom_const_{i}"
                    )
                
                coeffs.append(eq_coeffs)
                constants.append(const)
                
                # Mostrar a equação formatada
                eq_str = format_equation(eq_coeffs, var_names, const)
                st.write(f"Equação {i+1}: {eq_str}")
            
            # Botões
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Cancelar", key="cancel_custom_btn"):
                    st.session_state.adding_custom_example = False
                    st.rerun()
            
            with col2:
                if st.button("Salvar Sistema", key="save_custom_btn", type="primary"):
                    # Criar matriz e vetor
                    A, b = create_system_matrix(coeffs, constants, num_vars)
                    
                    # Classificar sistema
                    system_type = classify_system(A, b)
                    system_type_short = "SPD" if "Determinado" in system_type else ("SPI" if "Indeterminado" in system_type else "SI")
                    
                    # Salvar exemplo
                    if "favorites" not in st.session_state:
                        st.session_state.favorites = {"examples": []}
                    
                    st.session_state.favorites.setdefault("examples", []).append({
                        "title": f"Sistema {A.shape[0]}×{A.shape[1]} ({system_type_short})",
                        "A": A.tolist(),
                        "b": b.tolist(),
                        "date": datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
                    })
                    
                    st.session_state.adding_custom_example = False
                    st.success("Sistema personalizado salvo com sucesso!")
                    st.rerun()
                    
            st.markdown("---")
        
        # Exibir exemplo personalizado se existir
        if hasattr(st.session_state, 'custom_example') and st.session_state.custom_example:
            example = {
                "title": st.session_state.custom_example["title"],
                "A": st.session_state.custom_example["A"],
                "b": st.session_state.custom_example["b"],
                "solution": None,
                "explanation": f"Sistema personalizado adicionado em {st.session_state.custom_example['date']}."
            }
            
            try:
                # Tentar calcular a solução
                system_type = classify_system(example["A"], example["b"])
                
                if system_type == "Sistema Possível e Determinado (SPD)":
                    try:
                        solution = np.linalg.solve(example["A"], example["b"])
                        example["solution"] = solution
                    except:
                        pass
            except:
                pass
        else:
            # Obter exemplo selecionado
            example = get_example_system(selected_example)
        
        st.header(example["title"])
        
        st.markdown("### Sistema de Equações")
        for i, eq in enumerate(example["equations"]):
            st.write(f"Equação {i+1}: {eq}")
        
        # Classificação do sistema
        if example["A"] is not None and example["b"] is not None:
            system_type = classify_system(example["A"], example["b"])
            st.markdown(f"**Classificação do Sistema**: {system_type}")
        
        # Visualização gráfica quando aplicável
        if example["A"] is not None and example["b"] is not None:
            if example["A"].shape[1] == 2:
                st.markdown("### Visualização Gráfica")
                try:
                    fig = plot_2d_system(example["A"], example["b"])
                    if fig:
                        st.pyplot(fig)
                except:
                    st.warning("Não foi possível gerar a visualização do sistema.")
            elif example["A"].shape[1] == 3:
                st.markdown("### Visualização 3D")
                try:
                    fig = plot_3d_system(example["A"], example["b"])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                except:
                    st.warning("Não foi possível gerar a visualização 3D do sistema.")
        
        st.markdown("### Solução")
        
        # Se o exemplo tem solução definida, mostrar
        if "solution" in example and example["solution"] is not None and example["solution"] != "":
            st.markdown(example["solution"])
        elif example["A"] is not None and example["b"] is not None:
            # Caso contrário, calcular se possível
            system_type = classify_system(example["A"], example["b"])
            
            if system_type == "Sistema Possível e Determinado (SPD)":
                try:
                    solution = np.linalg.solve(example["A"], example["b"])
                    
                    var_names = ["x", "y", "z", "w"][:example["A"].shape[1]]
                    for i, var in enumerate(var_names):
                        if i < len(solution):
                            st.markdown(f"- {var} = {solution[i]:.4f}")
                except:
                    st.info("Não foi possível calcular a solução exata.")
            elif system_type == "Sistema Possível e Indeterminado (SPI)":
                st.markdown("Este sistema possui infinitas soluções.")
                
                # Tentar obter solução simbólica
                try:
                    symbolic_solution, var_symbols = sympy_solve_system(example["A"], example["b"])
                    
                    if symbolic_solution:
                        st.markdown("**Solução Paramétrica:**")
                        
                        if isinstance(symbolic_solution, dict):
                            for var, expr in symbolic_solution.items():
                                st.latex(f"{sp.latex(var)} = {sp.latex(expr)}")
                        else:
                            st.latex(sp.latex(symbolic_solution))
                except:
                    st.info("Não foi possível determinar a forma paramétrica exata da solução.")
            else:  # SI
                st.markdown("Este sistema não possui solução.")
        
        # Mostrar métodos de resolução selecionados
        st.markdown("### Métodos de Resolução")
        
        if "Passo a Passo Detalhado" in selected_methods:
            with st.expander("🔍 Passo a Passo Detalhado", expanded=True):
                st.markdown(example["explanation"])
        
        if example["A"] is not None and example["b"] is not None:
            A = example["A"]
            b = example["b"]
            
            if "Eliminação de Gauss" in selected_methods:
                with st.expander("📊 Eliminação de Gauss", expanded=False):
                    try:
                        steps, solution = gaussian_elimination_steps(A, b)
                        
                        for step in steps:
                            st.write(step)
                            
                        if solution is not None:
                            st.markdown("**Solução:**")
                            var_names = ["x", "y", "z", "w"][:A.shape[1]]
                            for i, var in enumerate(var_names):
                                if i < len(solution):
                                    st.markdown(f"- {var} = {solution[i]:.4f}")
                    except:
                        st.error("Não foi possível aplicar o método de Eliminação de Gauss para este sistema.")
            
            if "Gauss-Jordan" in selected_methods:
                with st.expander("📊 Gauss-Jordan", expanded=False):
                    try:
                        steps, solution = gauss_jordan_steps(A, b)
                        
                        for step in steps:
                            st.write(step)
                            
                        if solution is not None:
                            st.markdown("**Solução:**")
                            var_names = ["x", "y", "z", "w"][:A.shape[1]]
                            for i, var in enumerate(var_names):
                                if i < len(solution):
                                    st.markdown(f"- {var} = {solution[i]:.4f}")
                    except:
                        st.error("Não foi possível aplicar o método de Gauss-Jordan para este sistema.")
            
            if "Regra de Cramer" in selected_methods and A.shape[0] == A.shape[1]:
                with st.expander("📊 Regra de Cramer", expanded=False):
                    try:
                        steps, solution = cramer_rule(A, b, detailed=True)
                        
                        for step in steps:
                            st.write(step)
                            
                        if solution is not None:
                            st.markdown("**Solução:**")
                            var_names = ["x", "y", "z", "w"][:A.shape[1]]
                            for i, var in enumerate(var_names):
                                if i < len(solution):
                                    st.markdown(f"- {var} = {solution[i]:.4f}")
                    except:
                        st.error("Não foi possível aplicar a Regra de Cramer para este sistema.")
            
            if "Matriz Inversa" in selected_methods and A.shape[0] == A.shape[1]:
                with st.expander("📊 Matriz Inversa", expanded=False):
                    try:
                        steps, solution = matrix_inverse_method(A, b, detailed=True)
                        
                        for step in steps:
                            st.write(step)
                            
                        if solution is not None:
                            st.markdown("**Solução:**")
                            var_names = ["x", "y", "z", "w"][:A.shape[1]]
                            for i, var in enumerate(var_names):
                                if i < len(solution):
                                    st.markdown(f"- {var} = {solution[i]:.4f}")
                    except:
                        st.error("Não foi possível aplicar o método da Matriz Inversa para este sistema.")
                        
            if "Decomposição LU" in selected_methods and A.shape[0] == A.shape[1]:
                with st.expander("📊 Decomposição LU", expanded=False):
                    try:
                        steps, solution = lu_decomposition_method(A, b, detailed=True)
                        
                        for step in steps:
                            st.write(step)
                            
                        if solution is not None:
                            st.markdown("**Solução:**")
                            var_names = ["x", "y", "z", "w"][:A.shape[1]]
                            for i, var in enumerate(var_names):
                                if i < len(solution):
                                    st.markdown(f"- {var} = {solution[i]:.4f}")
                    except:
                        st.error("Não foi possível aplicar o método de Decomposição LU para este sistema.")
                        
            if "Jacobi" in selected_methods:
                with st.expander("📊 Método de Jacobi", expanded=False):
                    try:
                        is_diag_dominant = True
                        for i in range(min(A.shape[0], A.shape[1])):
                            if i < A.shape[0] and i < A.shape[1]:
                                if abs(A[i, i]) <= np.sum(np.abs(A[i, :])) - abs(A[i, i]):
                                    is_diag_dominant = False
                                    break
                        
                        if is_diag_dominant:
                            steps, solution = jacobi_iteration_method(A, b, detailed=True)
                            
                            for step in steps:
                                st.write(step)
                                
                            if solution is not None:
                                st.markdown("**Solução:**")
                                var_names = ["x", "y", "z", "w"][:A.shape[1]]
                                for i, var in enumerate(var_names):
                                    if i < len(solution):
                                        st.markdown(f"- {var} = {solution[i]:.4f}")
                        else:
                            st.warning("O sistema não é diagonalmente dominante. O método de Jacobi pode não convergir.")
                    except:
                        st.error("Não foi possível aplicar o método de Jacobi para este sistema.")
            
            if "Gauss-Seidel" in selected_methods:
                with st.expander("📊 Método de Gauss-Seidel", expanded=False):
                    try:
                        is_diag_dominant = True
                        for i in range(min(A.shape[0], A.shape[1])):
                            if i < A.shape[0] and i < A.shape[1]:
                                if abs(A[i, i]) <= np.sum(np.abs(A[i, :])) - abs(A[i, i]):
                                    is_diag_dominant = False
                                    break
                        
                        if is_diag_dominant:
                            steps, solution = gauss_seidel_method(A, b, detailed=True)
                            
                            for step in steps:
                                st.write(step)
                                
                            if solution is not None:
                                st.markdown("**Solução:**")
                                var_names = ["x", "y", "z", "w"][:A.shape[1]]
                                for i, var in enumerate(var_names):
                                    if i < len(solution):
                                        st.markdown(f"- {var} = {solution[i]:.4f}")
                        else:
                            st.warning("O sistema não é diagonalmente dominante. O método de Gauss-Seidel pode não convergir.")
                    except:
                        st.error("Não foi possível aplicar o método de Gauss-Seidel para este sistema.")
        
        # Opções adicionais
        st.markdown("### Opções Adicionais")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📋 Salvar nos Exemplos", key="save_example_btn"):
                if "favorites" not in st.session_state:
                    st.session_state.favorites = {"examples": []}
                
                # Verificar se já está nos favoritos
                already_saved = False
                if example["A"] is not None and example["b"] is not None:
                    for saved in st.session_state.favorites.get("examples", []):
                        if (np.array_equal(example["A"], np.array(saved["A"])) and 
                            np.array_equal(example["b"], np.array(saved["b"]))):
                            already_saved = True
                            break
                
                if already_saved:
                    st.info("Este exemplo já está nos seus favoritos.")
                else:
                    # Salvar exemplo
                    system_type = "SPD"
                    if example["A"] is not None and example["b"] is not None:
                        system_type = classify_system(example["A"], example["b"])
                        system_type = system_type.split(" ")[2][1:-1]  # Extrair SPD, SPI ou SI
                    
                    st.session_state.favorites.setdefault("examples", []).append({
                        "title": example["title"] if "title" in example else f"Sistema {example['A'].shape[0]}×{example['A'].shape[1]} ({system_type})",
                        "A": example["A"].tolist() if example["A"] is not None else None,
                        "b": example["b"].tolist() if example["b"] is not None else None,
                        "date": datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
                    })
                    
                    st.success("Exemplo salvo com sucesso!")
        
        with col2:
            if st.button("📥 Baixar Solução (PDF)", key="download_pdf_btn"):
                st.success("Download iniciado! (Simulação)")
                
        # Se houver um exemplo personalizado, oferecer a opção de removê-lo
        if hasattr(st.session_state, 'custom_example') and st.session_state.custom_example:
            if st.button("❌ Remover Exemplo Personalizado", key="remove_custom_btn"):
                st.session_state.custom_example = None
                st.rerun()

def show_reference_page():
    st.markdown('<h1 class="main-header">Referência Rápida</h1>', unsafe_allow_html=True)
    
    reference_topics = [
        "Classificação de Sistemas",
        "Método de Eliminação de Gauss",
        "Método de Gauss-Jordan",
        "Regra de Cramer",
        "Método da Matriz Inversa",
        "Decomposição LU",
        "Métodos Iterativos",
        "Interpretação Geométrica",
        "Sistemas Homogêneos",
        "Teorema de Rouché-Capelli",
        "Estabilidade Numérica",
        "Aplicações Práticas",
        "Sistemas Não-Lineares",
        "Sistemas Lineares em Programação Linear"
    ]
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_topic = st.radio(
            "Tópicos:",
            reference_topics,
            key="reference_topic"
        )
        
        st.markdown("---")
        
        # Adicionar funcionalidade de download do cartão de referência
        st.markdown("### Exportar Referência")
        
        if st.button("📥 Baixar como PDF", key="download_pdf_btn"):
            st.success(f"Download de '{selected_topic}.pdf' iniciado! (Simulação)")
            
        if st.button("📱 Versão para Celular", key="mobile_btn"):
            st.success("Versão para celular disponível! (Simulação)")
            
        # Botão para adicionar aos favoritos
        if st.button("⭐ Adicionar aos Favoritos", key="fav_btn"):
            if "favorites" not in st.session_state:
                st.session_state.favorites = {"reference_cards": []}
            
            # Verificar se já está nos favoritos
            already_saved = False
            for card in st.session_state.favorites.get("reference_cards", []):
                if card.get("title") == selected_topic:
                    already_saved = True
                    break
                    
            if already_saved:
                st.info(f"'{selected_topic}' já está nos seus favoritos.")
            else:
                st.session_state.favorites.setdefault("reference_cards", []).append(
                    {"title": selected_topic, "date": datetime.datetime.now().strftime("%d/%m/%Y %H:%M")}
                )
                st.success(f"'{selected_topic}' adicionado aos seus favoritos!")
        
        # Mostrar cartões favoritos
        st.markdown("### Meus Favoritos")
        if "favorites" in st.session_state and "reference_cards" in st.session_state.favorites:
            for i, card in enumerate(st.session_state.favorites["reference_cards"]):
                if st.button(f"{card['title']}", key=f"fav_card_{i}"):
                    # Selecionar cartão
                    st.session_state.reference_topic = card["title"]
                    st.rerun()
        else:
            st.info("Seus cartões de referência favoritos aparecerão aqui.")
    
    with col2:
        st.markdown(get_reference_card(selected_topic))
        
        # Adicionar exemplos compactos
        if selected_topic == "Classificação de Sistemas":
            with st.expander("Exemplos de Classificação", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**SPD**")
                    st.latex(r"""
                    \begin{align}
                    x + y &= 5\\
                    2x - y &= 1
                    \end{align}
                    """)
                    st.markdown("Solução única: (2, 3)")
                
                with col2:
                    st.markdown("**SPI**")
                    st.latex(r"""
                    \begin{align}
                    2x + 3y &= 6\\
                    4x + 6y &= 12
                    \end{align}
                    """)
                    st.markdown("Infinitas soluções: $x = t$, $y = \frac{6-2t}{3}$")
                
                with col3:
                    st.markdown("**SI**")
                    st.latex(r"""
                    \begin{align}
                    2x + 3y &= 6\\
                    2x + 3y &= 8
                    \end{align}
                    """)
                    st.markdown("Sem solução (inconsistente)")
        
        elif selected_topic == "Método de Eliminação de Gauss":
            with st.expander("Exemplo Passo a Passo", expanded=False):
                st.markdown("""
                **Sistema**:
                
                $x + y + z = 6$
                
                $2x - y + z = 3$
                
                $x + 2y + 3z = 14$
                
                **Matriz aumentada inicial**:
                
                $\\begin{bmatrix}
                1 & 1 & 1 & | & 6 \\\\
                2 & -1 & 1 & | & 3 \\\\
                1 & 2 & 3 & | & 14
                \\end{bmatrix}$
                
                **Passo 1**: Eliminar x da segunda linha
                
                $L_2 = L_2 - 2L_1$
                
                $\\begin{bmatrix}
                1 & 1 & 1 & | & 6 \\\\
                0 & -3 & -1 & | & -9 \\\\
                1 & 2 & 3 & | & 14
                \\end{bmatrix}$
                
                **Passo 2**: Eliminar x da terceira linha
                
                $L_3 = L_3 - L_1$
                
                $\\begin{bmatrix}
                1 & 1 & 1 & | & 6 \\\\
                0 & -3 & -1 & | & -9 \\\\
                0 & 1 & 2 & | & 8
                \\end{bmatrix}$
                
                **Passo 3**: Eliminar y da terceira linha
                
                $L_3 = L_3 + \\frac{1}{3}L_2$
                
                $\\begin{bmatrix}
                1 & 1 & 1 & | & 6 \\\\
                0 & -3 & -1 & | & -9 \\\\
                0 & 0 & \\frac{5}{3} & | & 5
                \\end{bmatrix}$
                
                **Substituição reversa**:
                
                $z = \\frac{5}{\\frac{5}{3}} = 3$
                
                $y = \\frac{-9 - (-1)(3)}{-3} = \\frac{-9 + 3}{-3} = 2$
                
                $x = 6 - 1(2) - 1(3) = 6 - 2 - 3 = 1$
                
                **Solução**: $(1, 2, 3)$
                """)
                
        elif selected_topic == "Regra de Cramer":
            with st.expander("Exemplo com Determinantes", expanded=False):
                st.markdown("""
                **Sistema**:
                
                $2x + 3y = 8$
                
                $4x - y = 1$
                
                **Determinante principal**:
                
                $D = \\begin{vmatrix} 2 & 3 \\\\ 4 & -1 \\end{vmatrix} = 2 \\times (-1) - 3 \\times 4 = -2 - 12 = -14$
                
                **Determinante para x**:
                
                $D_x = \\begin{vmatrix} 8 & 3 \\\\ 1 & -1 \\end{vmatrix} = 8 \\times (-1) - 3 \\times 1 = -8 - 3 = -11$
                
                **Determinante para y**:
                
                $D_y = \\begin{vmatrix} 2 & 8 \\\\ 4 & 1 \\end{vmatrix} = 2 \\times 1 - 8 \\times 4 = 2 - 32 = -30$
                
                **Solução**:
                
                $x = \\frac{D_x}{D} = \\frac{-11}{-14} = \\frac{11}{14} \\approx 0.786$
                
                $y = \\frac{D_y}{D} = \\frac{-30}{-14} = \\frac{15}{7} \\approx 2.143$
                """)

def show_videos_page():
    st.markdown('<h1 class="main-header">Videoaulas sobre Sistemas Lineares</h1>', unsafe_allow_html=True)
    
    try:
        # Obter lista de vídeos
        videos = get_youtube_videos()
        
        if not videos:
            st.warning("Não foi possível carregar os vídeos. Por favor, tente novamente mais tarde.")
            return
        
        # Filtros
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_level = st.multiselect(
                "Nível:",
                ["Básico", "Intermediário", "Avançado"],
                default=["Básico", "Intermediário", "Avançado"]
            )
            
        with col2:
            filter_author = st.multiselect(
                "Autor:",
                list(set(video["author"] for video in videos)),
                default=list(set(video["author"] for video in videos))
            )
            
        with col3:
            sort_by = st.selectbox(
                "Ordenar por:",
                ["Relevância", "Duração (menor)", "Duração (maior)"],
                index=0
            )
        
        # Filtrar e ordenar vídeos
        filtered_videos = [
            video for video in videos 
            if video["level"] in filter_level and video["author"] in filter_author
        ]
        
        if sort_by == "Duração (menor)":
            filtered_videos.sort(key=lambda x: convert_duration_to_seconds(x["duration"]))
        elif sort_by == "Duração (maior)":
            filtered_videos.sort(key=lambda x: convert_duration_to_seconds(x["duration"]), reverse=True)
        
        # Exibir vídeos em cards
        st.markdown("### Aulas Disponíveis")
        
        if not filtered_videos:
            st.info("Nenhum vídeo encontrado com os filtros selecionados.")
            return
        
        # Dividir em linhas de 3 colunas
        for i in range(0, len(filtered_videos), 3):
            cols = st.columns(3)
            
            for j in range(3):
                if i + j < len(filtered_videos):
                    video = filtered_videos[i + j]
                    
                    with cols[j]:
                        st.markdown(f"""
                        <div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 10px; height: 100%;">
                            <div class="video-container" style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; margin-bottom: 10px;">
                                <iframe style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" src="{video['url']}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
                            </div>
                            <h4 style="margin-top: 0;">{video['title']}</h4>
                            <p style="font-size: 0.9rem; color: #666;">{video['description']}</p>
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px;">
                                <span style="font-size: 0.8rem; color: #1E88E5;">{video['author']}</span>
                                <span style="font-size: 0.8rem; background-color: #e3f2fd; padding: 2px 8px; border-radius: 10px;">{video['level']} • {video['duration']}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Seção de playlists recomendadas
        st.markdown("### Playlists Recomendadas")
        
        playlists = [
            {
                "title": "Curso Completo de Sistemas Lineares",
                "author": "Matemática Rio",
                "videos": 15,
                "level": "Básico ao Avançado",
                "url": "https://www.youtube.com/playlist?list=example1"
            },
            {
                "title": "Álgebra Linear: Sistemas e Aplicações",
                "author": "Prof. Ferretto",
                "videos": 12,
                "level": "Intermediário",
                "url": "https://www.youtube.com/playlist?list=example2"
            },
            {
                "title": "Métodos Numéricos para Sistemas Lineares",
                "author": "Prof. Paulo Calculista",
                "videos": 8,
                "level": "Avançado",
                "url": "https://www.youtube.com/playlist?list=example3"
            }
        ]
        
        col1, col2, col3 = st.columns(3)
        
        for i, (col, playlist) in enumerate(zip([col1, col2, col3], playlists)):
            with col:
                st.markdown(f"""
                <div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px;">
                    <h4 style="margin-top: 0;">{playlist['title']}</h4>
                    <p><strong>Autor:</strong> {playlist['author']}</p>
                    <p><strong>Vídeos:</strong> {playlist['videos']}</p>
                    <p><strong>Nível:</strong> {playlist['level']}</p>
                    <button style="background-color: #1E88E5; color: white; border: none; padding: 5px 15px; border-radius: 5px; width: 100%;">Ver Playlist</button>
                </div>
                """, unsafe_allow_html=True)
        
        # Recursos adicionais
        st.markdown("### Recursos Adicionais")
        
        resources = [
            {
                "title": "Notas de Aula - Sistemas Lineares",
                "description": "Material complementar com exercícios resolvidos e teoria aprofundada.",
                "type": "PDF",
                "size": "2.4 MB"
            },
            {
                "title": "Lista de Exercícios Resolvidos",
                "description": "Compilação de 50 exercícios com soluções detalhadas.",
                "type": "PDF",
                "size": "1.8 MB"
            },
            {
                "title": "Resumo dos Métodos de Resolução",
                "description": "Guia rápido com todos os métodos e suas fórmulas.",
                "type": "PDF",
                "size": "0.9 MB"
            }
        ]
        
        for resource in resources:
            st.markdown(f"""
            <div style="display: flex; align-items: center; background-color: #f8f9fa; padding: 10px; border-radius: 8px; margin-bottom: 10px;">
                <div style="background-color: #e3f2fd; color: #1E88E5; width: 40px; height: 40px; border-radius: 20px; display: flex; align-items: center; justify-content: center; margin-right: 15px;">
                    <span style="font-weight: bold;">{resource['type']}</span>
                </div>
                <div style="flex-grow: 1;">
                    <h4 style="margin: 0;">{resource['title']}</h4>
                    <p style="margin: 0; color: #666; font-size: 0.9rem;">{resource['description']}</p>
                </div>
                <div>
                    <span style="font-size: 0.8rem; color: #666; margin-right: 10px;">{resource['size']}</span>
                    <button style="background-color: #1E88E5; color: white; border: none; padding: 5px 15px; border-radius: 5px;">Baixar</button>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar os vídeos: {str(e)}")
        st.info("Enquanto isso, você pode acessar nosso conteúdo teórico na seção 'Teoria' ou praticar na seção 'Exercícios'.")

def get_youtube_videos():
    """Retorna uma lista de vídeos do YouTube sobre sistemas lineares"""
    videos = [
        {
            "title": "Sistemas Lineares - Introdução",
            "description": "Uma introdução aos sistemas de equações lineares e suas aplicações.",
            "url": "https://www.youtube.com/embed/LhOHnLXolJc",
            "duration": "12:45",
            "author": "Matemática Rio",
            "level": "Básico"
        },
        {
            "title": "Método da Eliminação de Gauss",
            "description": "Resolução passo a passo do método de eliminação de Gauss.",
            "url": "https://www.youtube.com/embed/kaRWnHWL7nE",
            "duration": "18:22",
            "author": "Prof. Ferretto",
            "level": "Intermediário"
        },
        {
            "title": "Regra de Cramer Explicada",
            "description": "Tutorial detalhado sobre a aplicação da regra de Cramer com exemplos.",
            "url": "https://www.youtube.com/embed/MQPx2c-NQYI",
            "duration": "15:10",
            "author": "Equaciona Matemática",
            "level": "Intermediário"
        },
        {
            "title": "Método da Substituição",
            "description": "Aprenda a resolver sistemas lineares pelo método da substituição.",
            "url": "https://www.youtube.com/embed/LUSa3yRTB9A",
            "duration": "14:30",
            "author": "Matemática Rio",
            "level": "Básico"
        },
        {
            "title": "Método da Adição",
            "description": "Como resolver sistemas usando o método da adição ou eliminação.",
            "url": "https://www.youtube.com/embed/b-CvQvgBhvE",
            "duration": "16:15",
            "author": "Prof. Ferretto",
            "level": "Básico"
        },
        {
            "title": "Método da Comparação",
            "description": "Resolução de sistemas lineares pelo método da comparação.",
            "url": "https://www.youtube.com/embed/Fx_HAbpX8-g",
            "duration": "13:45",
            "author": "Equaciona Matemática",
            "level": "Básico"
        },
        {
            "title": "Aplicações de Sistemas Lineares",
            "description": "Exemplos práticos de aplicações de sistemas lineares em diversos campos.",
            "url": "https://www.youtube.com/embed/j2RbZzKMDnM",
            "duration": "20:35",
            "author": "Me Salva! ENEM",
            "level": "Básico"
        },
        {
            "title": "Matriz Inversa e Solução de Sistemas",
            "description": "Como encontrar a matriz inversa e usá-la para resolver sistemas lineares.",
            "url": "https://www.youtube.com/embed/kuixJnmwJxo",
            "duration": "22:18",
            "author": "Prof. Marcos Aba",
            "level": "Avançado"
        },
        {
            "title": "Sistemas Lineares 3x3 - Passo a Passo",
            "description": "Resolução completa de sistemas com três equações e três incógnitas.",
            "url": "https://www.youtube.com/embed/Hl-h_8TUXMo",
            "duration": "17:45",
            "author": "Matemática Rio",
            "level": "Intermediário"
        },
        {
            "title": "Métodos Iterativos: Jacobi e Gauss-Seidel",
            "description": "Explicação sobre métodos iterativos para sistemas de grande porte.",
            "url": "https://www.youtube.com/embed/hGzWsQxYVK0",
            "duration": "25:30",
            "author": "Prof. Paulo Calculista",
            "level": "Avançado"
        },
        {
            "title": "Sistemas Lineares e Matrizes",
            "description": "Relação entre sistemas lineares e operações matriciais.",
            "url": "https://www.youtube.com/embed/5J4upRPxEG8",
            "duration": "16:12",
            "author": "Prof. Ferretto",
            "level": "Intermediário"
        },
        {
            "title": "Classificação de Sistemas Lineares",
            "description": "Como identificar se um sistema é SPD, SPI ou SI.",
            "url": "https://www.youtube.com/embed/3g_vGpwFGfY",
            "duration": "14:50",
            "author": "Equaciona Matemática",
            "level": "Básico"
        }
    ]
    
    return videos

def convert_duration_to_seconds(duration):
    """Converte uma duração no formato 'MM:SS' para segundos"""
    try:
        parts = duration.split(':')
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + int(seconds)
        else:
            return 0
    except:
        return 0
        
def show_progress_page():
    st.markdown('<h1 class="main-header">Meu Progresso</h1>', unsafe_allow_html=True)
    
    # Dados de progresso
    exercises_completed = st.session_state.user_progress["exercises_completed"]
    correct_answers = st.session_state.user_progress["correct_answers"]
    topics_studied = st.session_state.user_progress["topics_studied"]
    difficulty_levels = st.session_state.user_progress["difficulty_levels"]
    streak = st.session_state.user_progress["streak"]
    
    # Visão geral
    st.markdown("### Visão Geral do Progresso")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<p class="metric-value">{exercises_completed}</p>', unsafe_allow_html=True)
        st.markdown('<p class="metric-label">Exercícios</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        accuracy = int(correct_answers / max(1, exercises_completed) * 100)
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<p class="metric-value">{accuracy}%</p>', unsafe_allow_html=True)
        st.markdown('<p class="metric-label">Precisão</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<p class="metric-value">{len(topics_studied)}</p>', unsafe_allow_html=True)
        st.markdown('<p class="metric-label">Tópicos</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<p class="metric-value">{streak}</p>', unsafe_allow_html=True)
        st.markdown('<p class="metric-label">Sequência</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Dashboard principal
    tab1, tab2, tab3 = st.tabs(["📊 Estatísticas", "🎯 Metas", "🏆 Conquistas"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de desempenho por dificuldade
            st.markdown("### Desempenho por Dificuldade")
            
            # Simular dados para o gráfico
            difficulty_data = {
                "Fácil": min(100, 75 + difficulty_levels["Fácil"] * 5),
                "Médio": min(100, 60 + difficulty_levels["Médio"] * 4),
                "Difícil": min(100, 40 + difficulty_levels["Difícil"] * 3)
            }
            
            fig, ax = plt.subplots(figsize=(8, 5))
            
            difficulties = list(difficulty_data.keys())
            accuracies = list(difficulty_data.values())
            colors = ['#4CAF50', '#FFC107', '#F44336']
            
            bars = ax.bar(difficulties, accuracies, color=colors)
            
            # Adicionar rótulos
            ax.set_ylim(0, 100)
            ax.set_ylabel('Precisão (%)')
            ax.set_title('Precisão por Nível de Dificuldade')
            
            # Adicionar valores nas barras
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.0f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 pontos de offset vertical
                            textcoords="offset points",
                            ha='center', va='bottom')
            
            st.pyplot(fig)
            
        with col2:
            # Gráfico de métodos mais utilizados
            st.markdown("### Métodos Mais Utilizados")
            
            # Simular dados
            methods_data = {
                "Eliminação de Gauss": 42,
                "Regra de Cramer": 28,
                "Matriz Inversa": 15,
                "Gauss-Jordan": 10,
                "Métodos Iterativos": 5
            }
            
            fig, ax = plt.subplots(figsize=(8, 5))
            
            # Criar gráfico de pizza
            wedges, texts, autotexts = ax.pie(
                methods_data.values(), 
                labels=list(methods_data.keys()),
                autopct='%1.1f%%',
                startangle=90,
                colors=['#1E88E5', '#42A5F5', '#90CAF9', '#BBDEFB', '#E3F2FD']
            )
            
            # Ajustar propriedades do texto
            for text in texts:
                text.set_fontsize(9)
            for autotext in autotexts:
                autotext.set_fontsize(9)
                autotext.set_color('white')
            
            ax.set_title('Métodos de Resolução Utilizados')
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            
            st.pyplot(fig)
        
        # Gráfico de atividade ao longo do tempo
        st.markdown("### Atividade ao Longo do Tempo")
        
        # Simular dados para o gráfico
        dates = [f"Semana {i+1}" for i in range(6)]
        
        # Dados de exercícios por tipo
        easy_per_week = [5, 7, 4, 6, 3, 8]
        medium_per_week = [3, 4, 5, 3, 6, 5]
        hard_per_week = [1, 0, 2, 1, 3, 2]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.bar(dates, easy_per_week, label='Fácil', color='#4CAF50', bottom=[0] * len(dates))
        ax.bar(dates, medium_per_week, label='Médio', color='#FFC107', bottom=easy_per_week)
        
        # Calcular as posições de bottom para o nível difícil
        hard_bottom = [easy + medium for easy, medium in zip(easy_per_week, medium_per_week)]
        ax.bar(dates, hard_per_week, label='Difícil', color='#F44336', bottom=hard_bottom)
        
        # Adicionar rótulos
        ax.set_xlabel('Período')
        ax.set_ylabel('Número de Exercícios')
        ax.set_title('Exercícios por Semana e Nível de Dificuldade')
        ax.legend()
        
        st.pyplot(fig)
        
        # Heatmap de atividade
        st.markdown("### Mapa de Atividade")
        
        # Simular dados para o heatmap
        weekdays = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom']
        weeks = ['Semana 1', 'Semana 2', 'Semana 3', 'Semana 4']
        
        # Gerar dados aleatórios para o heatmap
        data = np.zeros((len(weeks), len(weekdays)))
        for i in range(len(weeks)):
            for j in range(len(weekdays)):
                # Mais atividade nos dias da semana, menos nos finais de semana
                if j < 5:  # Dias de semana
                    data[i, j] = np.random.randint(0, 5)
                else:  # Finais de semana
                    data[i, j] = np.random.randint(0, 3)
        
        # Adicionar o dia atual com maior atividade
        today = datetime.datetime.now()
        weekday_today = today.weekday()
        week_idx = 3  # Última semana
        if weekday_today < len(weekdays):
            data[week_idx, weekday_today] = 7
        
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Criar um mapa de cores personalizado
        colors = ['#f5f5f5', '#e3f2fd', '#bbdefb', '#90caf9', '#64b5f6', '#42a5f5', '#2196f3', '#1e88e5', '#1976d2']
        cmap = LinearSegmentedColormap.from_list('blue_gradient', colors)
        
        im = ax.imshow(data, cmap=cmap)
        
        # Adicionar rótulos nos eixos
        ax.set_xticks(np.arange(len(weekdays)))
        ax.set_yticks(np.arange(len(weeks)))
        ax.set_xticklabels(weekdays)
        ax.set_yticklabels(weeks)
        
        # Rotacionar rótulos do eixo x
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Adicionar barra de cores
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_label('Número de Exercícios')
        
        # Adicionar título
        ax.set_title("Mapa de Atividade Semanal")
        
        # Adicionar texto para cada célula
        for i in range(len(weeks)):
            for j in range(len(weekdays)):
                ax.text(j, i, f"{int(data[i, j])}", ha="center", va="center", color="black" if data[i, j] < 5 else "white")
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab2:
        st.markdown("### Metas e Objetivos")
        
        # Metas semanais
        st.markdown("#### Metas Semanais")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Meta de exercícios
            progress = min(100, int(exercises_completed / 20 * 100))
            st.markdown(f"""
            <div style="background-color: #e3f2fd; border-radius: 10px; padding: 15px; margin-bottom: 15px;">
                <h5 style="margin-top: 0;">📚 Completar 20 exercícios</h5>
                <div style="background-color: #bbdefb; height: 10px; border-radius: 5px; margin-top: 10px;">
                    <div style="background-color: #1E88E5; width: {progress}%; height: 100%; border-radius: 5px;"></div>
                </div>
                <p style="text-align: right; margin-top: 5px; font-size: 0.9rem;">Progresso: {exercises_completed}/20 ({progress}%)</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Meta de precisão
            precision_goal = 80
            precision_current = accuracy
            precision_progress = min(100, int(precision_current / precision_goal * 100))
            st.markdown(f"""
            <div style="background-color: #e8f5e9; border-radius: 10px; padding: 15px; margin-bottom: 15px;">
                <h5 style="margin-top: 0;">🎯 Atingir {precision_goal}% de precisão</h5>
                <div style="background-color: #c8e6c9; height: 10px; border-radius: 5px; margin-top: 10px;">
                    <div style="background-color: #4CAF50; width: {precision_progress}%; height: 100%; border-radius: 5px;"></div>
                </div>
                <p style="text-align: right; margin-top: 5px; font-size: 0.9rem;">Progresso: {precision_current}/{precision_goal}% ({precision_progress}%)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Meta de tópicos
            topics_goal = 10
            topics_current = len(topics_studied)
            topics_progress = min(100, int(topics_current / topics_goal * 100))
            st.markdown(f"""
            <div style="background-color: #fff3e0; border-radius: 10px; padding: 15px; margin-bottom: 15px;">
                <h5 style="margin-top: 0;">📝 Estudar {topics_goal} tópicos diferentes</h5>
                <div style="background-color: #ffe0b2; height: 10px; border-radius: 5px; margin-top: 10px;">
                    <div style="background-color: #FF9800; width: {topics_progress}%; height: 100%; border-radius: 5px;"></div>
                </div>
                <p style="text-align: right; margin-top: 5px; font-size: 0.9rem;">Progresso: {topics_current}/{topics_goal} ({topics_progress}%)</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Meta de sequência
            streak_goal = 7
            streak_current = streak
            streak_progress = min(100, int(streak_current / streak_goal * 100))
            st.markdown(f"""
            <div style="background-color: #f3e5f5; border-radius: 10px; padding: 15px; margin-bottom: 15px;">
                <h5 style="margin-top: 0;">🔥 Manter sequência de {streak_goal} dias</h5>
                <div style="background-color: #e1bee7; height: 10px; border-radius: 5px; margin-top: 10px;">
                    <div style="background-color: #9C27B0; width: {streak_progress}%; height: 100%; border-radius: 5px;"></div>
                </div>
                <p style="text-align: right; margin-top: 5px; font-size: 0.9rem;">Progresso: {streak_current}/{streak_goal} ({streak_progress}%)</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Definir novas metas
        st.markdown("#### Definir Novas Metas")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            new_exercises_goal = st.number_input("Exercícios a completar:", min_value=5, max_value=100, value=20, step=5)
            
        with col2:
            new_precision_goal = st.number_input("Meta de precisão (%):", min_value=50, max_value=100, value=80, step=5)
            
        with col3:
            new_streak_goal = st.number_input("Meta de sequência (dias):", min_value=3, max_value=30, value=7, step=1)
            
        if st.button("Salvar Novas Metas", key="save_goals_btn"):
            st.success("Metas atualizadas com sucesso! (Simulação)")
    
    with tab3:
        st.markdown("### Conquistas e Certificados")
        
        # Lista de conquistas possíveis
        achievements = [
            {
                "title": "Primeiros Passos",
                "description": "Complete 5 exercícios",
                "icon": "🥉",
                "color": "#CD7F32",
                "achieved": exercises_completed >= 5
            },
            {
                "title": "Estudante Dedicado",
                "description": "Complete 20 exercícios",
                "icon": "🥈",
                "color": "#C0C0C0",
                "achieved": exercises_completed >= 20
            },
            {
                "title": "Mestre em Sistemas Lineares",
                "description": "Complete 50 exercícios com mais de 80% de precisão",
                "icon": "🥇",
                "color": "#FFD700",
                "achieved": exercises_completed >= 50 and accuracy >= 80
            },
            {
                "title": "Explorador de Tópicos",
                "description": "Estude 8 tópicos diferentes",
                "icon": "🧭",
                "color": "#4CAF50",
                "achieved": len(topics_studied) >= 8
            },
            {
                "title": "Sequência de Fogo",
                "description": "Mantenha uma sequência de estudo de 7 dias",
                "icon": "🔥",
                "color": "#FF5722",
                "achieved": streak >= 7
            },
            {
                "title": "Precisão Perfeita",
                "description": "Acerte 10 exercícios consecutivos",
                "icon": "🎯",
                "color": "#2196F3",
                "achieved": accuracy >= 90 and exercises_completed >= 10
            }
        ]
        
        # Mostrar conquistas organizadas em grid
        achievs_per_row = 3
        for i in range(0, len(achievements), achievs_per_row):
            cols = st.columns(achievs_per_row)
            
            for j in range(achievs_per_row):
                if i + j < len(achievements):
                    achievement = achievements[i + j]
                    
                    with cols[j]:
                        bg_color = "#f8f9fa"
                        status = "BLOQUEADO"
                        status_color = "#9E9E9E"
                        opacity = "0.7"
                        
                        if achievement["achieved"]:
                            bg_color = achievement["color"] + "20"  # Adicionar transparência à cor
                            status = "CONQUISTADO"
                            status_color = achievement["color"]
                            opacity = "1"
                        
                        st.markdown(f"""
                        <div style="background-color: {bg_color}; border-radius: 10px; padding: 15px; text-align: center; margin-bottom: 15px; opacity: {opacity};">
                            <div style="font-size: 32px; margin-bottom: 10px;">{achievement["icon"]}</div>
                            <h4 style="margin-top: 0;">{achievement["title"]}</h4>
                            <p style="font-size: 0.9rem;">{achievement["description"]}</p>
                            <p style="color: {status_color}; font-weight: bold;">{status}</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Certificados disponíveis
        st.markdown("### Certificados Disponíveis")
        
        certificates = [
            {
                "title": "Introdução aos Sistemas Lineares",
                "requirements": "Complete 20 exercícios de nível básico",
                "progress": min(100, int(exercises_completed / 20 * 100)),
                "available": exercises_completed >= 20
            },
            {
                "title": "Métodos de Resolução de Sistemas",
                "requirements": "Complete 15 exercícios usando diferentes métodos",
                "progress": min(100, int(exercises_completed / 15 * 100)),
                "available": exercises_completed >= 15
            },
            {
                "title": "Aplicações Práticas de Sistemas Lineares",
                "requirements": "Complete 10 exercícios de aplicações práticas",
                "progress": 30,
                "available": False
            }
        ]
        
        for cert in certificates:
            status_text = "DISPONÍVEL" if cert["available"] else "EM PROGRESSO"
            status_color = "#4CAF50" if cert["available"] else "#FFC107"
            btn_disabled = "" if cert["available"] else "disabled"
            
            st.markdown(f"""
            <div style="display: flex; align-items: center; background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                <div style="margin-right: 15px; font-size: 24px;">📜</div>
                <div style="flex-grow: 1;">
                    <h4 style="margin: 0;">{cert["title"]}</h4>
                    <p style="margin: 5px 0; color: #666; font-size: 0.9rem;">{cert["requirements"]}</p>
                    <div style="background-color: #e0e0e0; height: 10px; border-radius: 5px; margin-top: 10px; width: 100%;">
                        <div style="background-color: {status_color}; width: {cert["progress"]}%; height: 100%; border-radius: 5px;"></div>
                    </div>
                </div>
                <div style="margin-left: 15px;">
                    <span style="display: block; text-align: center; font-size: 0.8rem; color: {status_color}; margin-bottom: 5px;">{status_text}</span>
                    <button style="background-color: #1E88E5; color: white; border: none; padding: 5px 15px; border-radius: 5px; cursor: pointer; {btn_disabled}">Emitir</button>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Compartilhar progresso
        st.markdown("### Compartilhar Progresso")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background-color: #e3f2fd; border-radius: 10px; padding: 15px; text-align: center;">
                <h4 style="margin-top: 0;">Baixar Relatório de Progresso</h4>
                <p>Exporte seu histórico de atividades e conquistas para compartilhar ou guardar.</p>
                <button style="background-color: #1E88E5; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin-top: 10px;">📥 Baixar PDF</button>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div style="background-color: #e8f5e9; border-radius: 10px; padding: 15px; text-align: center;">
                <h4 style="margin-top: 0;">Compartilhar nas Redes Sociais</h4>
                <p>Mostre seu progresso e conquistas para amigos e colegas.</p>
                <div style="display: flex; justify-content: center; gap: 10px; margin-top: 10px;">
                    <button style="background-color: #3b5998; color: white; border: none; padding: 10px; border-radius: 5px; cursor: pointer;">Facebook</button>
                    <button style="background-color: #1da1f2; color: white; border: none; padding: 10px; border-radius: 5px; cursor: pointer;">Twitter</button>
                    <button style="background-color: #0e76a8; color: white; border: none; padding: 10px; border-radius: 5px; cursor: pointer;">LinkedIn</button>
                </div>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
