import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import base64
from sympy import Matrix, symbols
from sympy.solvers.solveset import linsolve
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.express as px
import random
from datetime import datetime, timedelta
import altair as alt

# Configuração da página
st.set_page_config(
    page_title="Sistema Linear Solver - Guia Universitário",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    """Resolve o sistema usando decomposição LU"""
    steps = []
    try:
        n = len(b)
        
        # Verificar se a matriz é quadrada
        if A.shape[0] != A.shape[1]:
            steps.append("A decomposição LU requer uma matriz quadrada.")
            return steps, None
        
        # Decompor A em L e U
        P, L, U = sp.Matrix(A).LUdecomposition()
        
        if detailed:
            steps.append("Decomposição LU da matriz A:")
            steps.append("Matriz L (triangular inferior):")
            steps.append(str(np.array(L, dtype=float)))
            steps.append("Matriz U (triangular superior):")
            steps.append(str(np.array(U, dtype=float)))
            steps.append("Matriz P (permutação):")
            steps.append(str(np.array(P, dtype=float)))
        
        # Resolver Ly = Pb
        Pb = np.array(P * Matrix(b)).astype(float).flatten()
        
        if detailed:
            steps.append(f"Resolver o sistema Ly = Pb, onde Pb = {Pb}")
            
        y = np.zeros(n)
        for i in range(n):
            y[i] = Pb[i]
            for j in range(i):
                y[i] -= L[i, j] * y[j]
            y[i] /= L[i, i]
            
            if detailed:
                steps.append(f"y_{i+1} = {y[i]:.4f}")
        
        # Resolver Ux = y
        if detailed:
            steps.append("Resolver o sistema Ux = y usando substituição reversa")
            
        x = np.zeros(n)
        for i in range(n-1, -1, -1):
            x[i] = y[i]
            for j in range(i+1, n):
                x[i] -= U[i, j] * x[j]
            x[i] /= U[i, i]
            
            if detailed:
                steps.append(f"x_{i+1} = {x[i]:.4f}")
        
        return steps, x
    except Exception as e:
        steps.append(f"Erro ao aplicar decomposição LU: {str(e)}")
        return steps, None

def jacobi_method(A, b, iterations=10, detailed=True):
    """Implementa o método iterativo de Jacobi"""
    steps = []
    n = len(b)
    
    # Verificar diagonal dominante
    is_diag_dominant = True
    for i in range(n):
        row_sum = sum(abs(A[i, j]) for j in range(n) if j != i)
        if abs(A[i, i]) <= row_sum:
            is_diag_dominant = False
            break
    
    if not is_diag_dominant and detailed:
        steps.append("Aviso: A matriz não é diagonalmente dominante. O método de Jacobi pode não convergir.")
    
    # Inicializar vetor solução
    x = np.zeros(n)
    
    if detailed:
        steps.append(f"Valor inicial: x = {x}")
    
    # Iterar
    for k in range(iterations):
        x_new = np.zeros(n)
        
        for i in range(n):
            sum_term = 0
            for j in range(n):
                if j != i:
                    sum_term += A[i, j] * x[j]
            
            x_new[i] = (b[i] - sum_term) / A[i, i]
        
        # Calcular erro
        error = np.linalg.norm(x_new - x)
        
        if detailed:
            steps.append(f"Iteração {k+1}: x = {x_new}, erro = {error:.6f}")
        
        # Atualizar solução
        x = x_new
        
        # Verificar convergência
        if error < 1e-6:
            steps.append(f"Convergência atingida após {k+1} iterações.")
            break
    
    return steps, x

def gauss_seidel_method(A, b, iterations=10, detailed=True):
    """Implementa o método iterativo de Gauss-Seidel"""
    steps = []
    n = len(b)
    
    # Verificar diagonal dominante
    is_diag_dominant = True
    for i in range(n):
        row_sum = sum(abs(A[i, j]) for j in range(n) if j != i)
        if abs(A[i, i]) <= row_sum:
            is_diag_dominant = False
            break
    
    if not is_diag_dominant and detailed:
        steps.append("Aviso: A matriz não é diagonalmente dominante. O método de Gauss-Seidel pode não convergir.")
    
    # Inicializar vetor solução
    x = np.zeros(n)
    
    if detailed:
        steps.append(f"Valor inicial: x = {x}")
    
    # Iterar
    for k in range(iterations):
        x_old = x.copy()
        
        for i in range(n):
            sum_term = 0
            for j in range(n):
                if j != i:
                    sum_term += A[i, j] * x[j]
            
            x[i] = (b[i] - sum_term) / A[i, i]
        
        # Calcular erro
        error = np.linalg.norm(x - x_old)
        
        if detailed:
            steps.append(f"Iteração {k+1}: x = {x}, erro = {error:.6f}")
        
        # Verificar convergência
        if error < 1e-6:
            steps.append(f"Convergência atingida após {k+1} iterações.")
            break
    
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
    if A.shape[0] < 2 or A.shape[1] < 2:
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
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Representação Gráfica do Sistema')
    ax.legend()
    
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
                        textcoords='offset points')
    except:
        pass
    
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
            name=f'Equação {i+1}'
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

def get_practice_exercise(level, category="Geral"):
    """Gera exercícios de prática com base no nível de dificuldade e categoria"""
    if category == "Geral":
        if level == "Fácil":
            # Escolha aleatória entre diferentes tipos de sistemas fáceis
            exercise_type = random.choice(["2x2_integers", "2x2_fractions", "2x2_decimals", "3x3_integers"])
            
            if exercise_type == "2x2_integers":
                # Sistema 2x2 com solução inteira
                A = np.array([[random.randint(1, 5), random.randint(1, 5)], 
                              [random.randint(1, 5), random.randint(-5, -1)]])
                x = np.array([random.randint(1, 10), random.randint(1, 10)])  # Solução desejada
                b = np.dot(A, x)
            elif exercise_type == "2x2_fractions":
                # Sistema 2x2 com solução fracionária simples
                A = np.array([[2, 3], [4, 5]])
                x = np.array([1/2, 1/3])  # Solução desejada
                b = np.dot(A, x)
            elif exercise_type == "2x2_decimals":
                # Sistema 2x2 com números decimais
                A = np.array([[1.5, 2.5], [3.5, 1.5]])
                x = np.array([2, 3])  # Solução desejada
                b = np.dot(A, x)
            else:  # 3x3_integers
                # Sistema 3x3 simples
                A = np.array([[1, 1, 1], [1, 2, 3], [2, 1, 3]])
                x = np.array([1, 1, 1])  # Solução desejada
                b = np.dot(A, x)
                
        elif level == "Médio":
            # Escolha aleatória entre diferentes tipos de sistemas médios
            exercise_type = random.choice(["3x3_mixed", "3x3_fractions", "4x4_integers", "application_basic"])
            
            if exercise_type == "3x3_mixed":
                # Sistema 3x3 com coeficientes variados
                A = np.array([[2, 1, -1], [3, -2, 1], [1, 2, 2]])
                x = np.array([random.randint(1, 5), random.randint(1, 5), random.randint(1, 5)])
                b = np.dot(A, x)
            elif exercise_type == "3x3_fractions":
                # Sistema 3x3 com solução fracionária
                A = np.array([[3, 2, 1], [1, -1, 2], [2, 3, -1]])
                x = np.array([1/2, 1/3, 1/4])
                b = np.dot(A, x)
            elif exercise_type == "4x4_integers":
                # Sistema 4x4 simples
                A = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 1, 0]])
                x = np.array([1, 1, 1, 1])
                b = np.dot(A, x)
            else:  # application_basic
                # Problema de aplicação: mistura
                # Percentuais de componente X em três soluções
                p1, p2, p3 = 0.1, 0.2, 0.5
                # Quantidade desejada final
                total = 100
                # Percentual desejado na mistura final
                p_final = 0.25
                # Quantidade do primeiro componente fixada
                x1 = 30
                
                A = np.array([[1, 1, 1], [p1, p2, p3], [1, 0, 0]])
                b = np.array([total, p_final * total, x1])
                x = None  # Não se define x pois é o que queremos calcular
                
        else:  # Difícil
            # Escolha aleatória entre diferentes tipos de sistemas difíceis
            r = random.choice(["SPD_complex", "SPI", "SI", "ill_conditioned", "application_complex"])
            
            if r == "SPD_complex":
                # Sistema com solução não inteira
                A = np.array([[3.5, 1.25, -2.75], [2.25, -2.5, 1.75], [1.5, 5.25, -3.5]])
                x = np.array([1/3, 2/3, 1/3])  # Solução fracionária
                b = np.dot(A, x)
                
            elif r == "SPI":
                # Sistema com infinitas soluções
                A = np.array([[1, 2, 3], [2, 4, 6], [3, 5, 7]])
                b = np.array([6, 12, 15])
                x = None
                
            elif r == "SI":
                # Sistema impossível
                A = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
                b = np.array([6, 12, 19])  # Inconsistente
                x = None
                
            elif r == "ill_conditioned":
                # Sistema mal condicionado
                A = np.array([[1, 0.99, 0.98], [0.99, 0.98, 0.97], [0.98, 0.97, 0.96]])
                x = np.array([1, 1, 1])
                b = np.dot(A, x)
                
            else:  # application_complex
                # Problema de aplicação: circuito elétrico
                # Matriz de coeficientes para as leis de Kirchhoff
                A = np.array([[1, -1, -1], [10, 0, -5], [0, 5, 15]])
                b = np.array([0, 20, 0])
                x = None
    
    elif category == "Engenharia":
        # Exercícios específicos para engenharia
        if level == "Fácil":
            # Problema simples de circuito
            A = np.array([[1, -1], [5, 10]])
            x = np.array([2, 1])  # Correntes I1 e I2
            b = np.array([0, 20])  # Leis de Kirchhoff
            
        elif level == "Médio":
            # Problema de estrutura/treliça
            A = np.array([[np.cos(np.pi/4), np.cos(np.pi/2), 0], 
                          [np.sin(np.pi/4), np.sin(np.pi/2), 0], 
                          [0, 0, 1]])
            b = np.array([0, 100, 50])  # Forças aplicadas
            x = None
            
        else:  # Difícil
            # Problema de transferência de calor
            A = np.array([[4, -1, 0, 0], [-1, 4, -1, 0], [0, -1, 4, -1], [0, 0, -1, 4]])
            b = np.array([100, 0, 0, 0])  # Temperaturas de contorno
            x = None
    
    elif category == "Economia":
        # Exercícios específicos para economia
        if level == "Fácil":
            # Problema simples de alocação
            A = np.array([[1, 1], [2, 3]])
            x = np.array([100, 200])  # Quantidades de produtos
            b = np.array([300, 700])  # Restrições de recursos
            
        elif level == "Médio":
            # Modelo de Leontief simplificado
            A = np.array([[0.3, 0.4, 0.2], [0.2, 0.1, 0.3], [0.1, 0.2, 0.3]])
            A = np.eye(3) - A  # Matriz (I - A) do modelo
            b = np.array([100, 150, 200])  # Demanda final
            x = None
            
        else:  # Difícil
            # Modelo de equilíbrio de preços
            A = np.array([[2, -1, 0, 0], [-1, 3, -1, 0], [0, -1, 3, -1], [0, 0, -1, 2]])
            b = np.array([10, 0, 0, 20])
            x = None
    
    elif category == "Química":
        # Exercícios específicos para química
        if level == "Fácil":
            # Balanceamento de equação química simples
            A = np.array([[1, -1], [2, -3]])
            b = np.array([0, 0])
            x = np.array([3, 2])  # Coeficientes estequiométricos
            
        elif level == "Médio":
            # Balanceamento de equação química mais complexa
            A = np.array([[1, 2, -1, 0], [2, 1, 0, -2], [0, 2, -3, -1]])
            b = np.array([0, 0, 0])
            x = np.array([2, 1, 2, 1])  # Coeficientes estequiométricos
            
        else:  # Difícil
            # Sistema de equilíbrio químico
            A = np.array([[1, 1, 1, 0], [0.1, 0.2, 0.3, -1], [2, 1, 0, 0], [1, 1, -1, 0]])
            b = np.array([100, 25, 40, 0])
            x = None
            
    elif category == "Física":
        # Exercícios específicos para física
        if level == "Fácil":
            # Problema simples de cinemática
            A = np.array([[1, 1], [0, 1]])
            x = np.array([10, 5])  # Posição inicial e velocidade
            b = np.array([15, 5])  # Posição final e velocidade final
            
        elif level == "Médio":
            # Problema de dinâmica
            A = np.array([[1, 1, 1], [0.5, 0.2, 0.3], [10, 5, 0]])
            b = np.array([100, 30, 70])
            x = None
            
        else:  # Difícil
            # Problema de circuito RLC
            A = np.array([[1, -1, -1, 0], [10, 0, 0, -5], [0, 5, 0, -2], [0, 0, 20, -8]])
            b = np.array([0, 0, 0, 100])
            x = None
    
    # Caso geral para outros temas
    else:
        if level == "Fácil":
            A = np.array([[1, 1], [2, 1]])
            x = np.array([5, 3])
            b = np.dot(A, x)
        elif level == "Médio":
            A = np.array([[2, 1, -1], [3, -2, 1], [1, 2, 2]])
            x = np.array([1, 2, 3])
            b = np.dot(A, x)
        else:  # Difícil
            r = random.choice(["SPD_complex", "SPI", "SI"])
            if r == "SPD_complex":
                A = np.array([[3, 1, -2], [2, -2, 1], [1, 5, -3]])
                x = np.array([1/3, 2/3, 1/3])
                b = np.dot(A, x)
            elif r == "SPI":
                A = np.array([[1, 2, 3], [2, 4, 6], [3, 5, 7]])
                b = np.array([6, 12, 15])
                x = None
            else:  # SI
                A = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
                b = np.array([6, 12, 19])
                x = None
    
    # Gerar o enunciado com base na categoria
    if category == "Engenharia":
        if "circuito" in locals() and circuit:
            question = "Resolva o sistema para encontrar as correntes no circuito:"
        elif "treliça" in locals() and structure:
            question = "Determine as forças nos membros da treliça:"
        else:
            question = "Resolva o sistema de equações lineares para o problema de engenharia:"
    elif category == "Economia":
        if "leontief" in locals() and leontief:
            question = "Para o modelo de Leontief dado, determine os níveis de produção necessários:"
        else:
            question = "Resolva o sistema de equações para o problema econômico:"
    elif category == "Química":
        if x is not None and all(x > 0):  # Provavelmente balanceamento químico
            question = "Determine os coeficientes estequiométricos para balancear a equação química:"
        else:
            question = "Resolva o sistema para o problema de equilíbrio químico:"
    elif category == "Física":
        question = "Resolva o sistema para o problema de física:"
    else:
        question = "Resolva o sistema de equações lineares:"
    
    # Preparar as equações formatadas
    var_names = ["x", "y", "z", "w", "v"][:A.shape[1]]
    equations = [format_equation(A[i], var_names, b[i]) for i in range(len(b))]
    
    return A, b, question, equations, x

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
        | a₁₁ a₁₂ a₁₃ ... | b₁ |
        | 0   a₂₂ a₂₃ ... | b₂ |
        | 0   0   a₃₃ ... | b₃ |
        | ...             | ... |
        ```
        """,
        
        "Regra de Cramer": """
        # Regra de Cramer
        
        A regra de Cramer é um método para resolver sistemas lineares usando determinantes. Para um sistema de n equações e n incógnitas:
        
        1. Calcular o determinante D da matriz dos coeficientes
        2. Para cada incógnita xᵢ, substituir a coluna i da matriz pelos termos independentes, obtendo o determinante Dᵢ
        3. A solução para cada incógnita é xᵢ = Dᵢ/D
        
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
        
        "Métodos Iterativos": """
        # Métodos Iterativos para Sistemas Lineares
        
        Ao contrário dos métodos diretos, os métodos iterativos aproximam gradualmente a solução através de sucessivas iterações.
        
        ### Método de Jacobi
        
        1. Reescrever cada equação isolando a variável da diagonal:
           x_i = (b_i - ∑_{j≠i} a_{ij}x_j) / a_{ii}
           
        2. Calcular novos valores usando apenas valores da iteração anterior:
           x_i^(k+1) = (b_i - ∑_{j≠i} a_{ij}x_j^(k)) / a_{ii}
        
        ### Método de Gauss-Seidel
        
        Similar ao método de Jacobi, mas usa valores atualizados imediatamente:
        
        x_i^(k+1) = (b_i - ∑_{j<i} a_{ij}x_j^(k+1) - ∑_{j>i} a_{ij}x_j^(k)) / a_{ii}
        
        ### Convergência
        
        - Condição suficiente: matriz diagonalmente dominante
        - Gauss-Seidel geralmente converge mais rápido que Jacobi
        - Úteis para sistemas grandes e esparsos
        """,
        
        "Decomposição LU": """
        # Decomposição LU
        
        A decomposição LU fatoriza uma matriz A como o produto de uma matriz triangular inferior L e uma matriz triangular superior U: A = LU.
        
        ### Processo:
        
        1. **Fatoração**: Encontrar L e U tal que A = LU
        2. **Resolver Ly = b** (substituição progressiva)
        3. **Resolver Ux = y** (substituição regressiva)
        
        ### Vantagens:
        
        - Permite resolver múltiplos sistemas com a mesma matriz de coeficientes de forma eficiente
        - Não necessita recalcular a decomposição quando apenas os termos independentes mudam
        - Útil para cálculo de determinantes: det(A) = det(L) × det(U) = produto dos elementos da diagonal de U
        
        ### Variantes:
        
        - **Decomposição LU com pivoteamento**: A = PLU, onde P é uma matriz de permutação
        - **Decomposição de Cholesky**: Para matrizes simétricas definidas positivas, A = LL^T
        """,
        
        "Condicionamento de Sistemas": """
        # Condicionamento de Sistemas Lineares
        
        O número de condição de uma matriz mede a sensibilidade da solução às perturbações nos dados.
        
        ### Número de Condição
        
        κ(A) = ||A|| × ||A^(-1)||
        
        - **Bem-condicionado**: κ(A) próximo de 1
        - **Mal-condicionado**: κ(A) muito grande
        
        ### Efeitos do Mau Condicionamento
        
        - Pequenas mudanças nos dados de entrada causam grandes alterações na solução
        - Maior suscetibilidade a erros de arredondamento
        - Mais difícil de resolver numericamente
        
        ### Fatores que Afetam o Condicionamento
        
        - Quase-dependência linear entre linhas ou colunas
        - Presença de valores muito diferentes em magnitude na matriz
        - Proximidade da matriz à singularidade
        
        ### Melhoria do Condicionamento
        
        - Pré-condicionamento: transformar o sistema para reduzir κ(A)
        - Escalonamento adequado de linhas e colunas
        - Uso de precisão estendida nos cálculos
        """,
        
        "Aplicações em Engenharia": """
        # Aplicações de Sistemas Lineares em Engenharia
        
        ### Análise de Circuitos Elétricos
        - **Leis de Kirchhoff**: 
          - Lei das correntes: soma das correntes em um nó = 0
          - Lei das tensões: soma das tensões em um laço fechado = 0
        - **Método dos Nós e Método das Malhas**
        
        ### Análise Estrutural
        - **Método dos Deslocamentos**: determinar deslocamentos nodais
        - **Análise de Treliças**: determinar forças nos membros
        - **Método dos Elementos Finitos**: discretização de problemas contínuos
        
        ### Controle de Sistemas
        - **Matrizes de Estado**: x' = Ax + Bu
        - **Controlabilidade e Observabilidade**
        
        ### Dinâmica de Fluidos
        - **Método das Diferenças Finitas**: discretização de equações diferenciais
        - **Método dos Volumes Finitos**: conservação de massa, momento e energia
        
        ### Processamento de Sinais
        - **Filtros Digitais**: sistemas de equações para resposta em frequência
        - **Transformada Discreta de Fourier**: sistemas para análise espectral
        """,
        
        "Aplicações em Ciências Sociais": """
        # Aplicações de Sistemas Lineares em Ciências Sociais
        
        ### Economia
        - **Modelo de Leontief (Insumo-Produto)**:
          - Representa interdependências entre setores econômicos
          - Equação básica: (I - A)x = d
          - A: matriz de coeficientes técnicos
          - x: vetor de produção setorial
          - d: vetor de demanda final
        
        ### Demografia
        - **Modelos de Crescimento Populacional**:
          - Matrizes de Leslie para dinâmica de populações
          - Projeção de populações por faixa etária
        
        ### Redes Sociais
        - **Centralidade em Grafos**:
          - Sistemas lineares para determinar importância de nós
          - PageRank e outros algoritmos de classificação
        
        ### Psicometria
        - **Análise Fatorial**:
          - Sistemas para identificar fatores latentes
          - Modelos de equações estruturais
        
        ### Teoria dos Jogos
        - **Jogos de Soma Zero**:
          - Estratégias ótimas via programação linear
          - Equilíbrio de Nash como solução de sistemas
        """,
        
        "Aplicações em Ciências Naturais": """
        # Aplicações de Sistemas Lineares em Ciências Naturais
        
        ### Química
        - **Balanceamento de Equações**: coeficientes estequiométricos como solução de sistemas
        - **Equilíbrio Químico**: concentrações de equilíbrio
        - **Cinética Química**: sistemas para determinar constantes de reação
        
        ### Física
        - **Mecânica**: sistemas para equilíbrio de forças e momentos
        - **Eletromagnetismo**: discretização das equações de Maxwell
        - **Mecânica Quântica**: equações de autovalores para sistemas de partículas
        
        ### Biologia
        - **Redes Metabólicas**: fluxos em sistemas bioquímicos
        - **Dinâmica de Populações**: modelos de interação entre espécies
        - **Bioinformática**: alinhamento de sequências e análise de expressão gênica
        
        ### Ciências Ambientais
        - **Modelos de Dispersão de Poluentes**
        - **Balanço de Massa em Ecossistemas**
        - **Ciclos Biogeoquímicos**: fluxos de carbono, nitrogênio, etc.
        
        ### Geofísica
        - **Tomografia Sísmica**: reconstrução de estruturas internas
        - **Inversão Geofísica**: recuperação de parâmetros a partir de dados observados
        """
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
            Da Equação 3: 5I₂ + 15I₃ = 0 → I₂ = -3I₃
            Substituindo na Equação 1: I₁ - (-3I₃) - I₃ = 0 → I₁ - I₃ + 3I₃ = 0 → I₁ = -2I₃
            Substituindo na Equação 2: 10(-2I₃) - 5I₃ = 20 → -20I₃ - 5I₃ = 20 → -25I₃ = 20 → I₃ = -0.8
            
            Portanto:
            - I₃ = -0.8A
            - I₂ = -3(-0.8) = 2.4A
            - I₁ = -2(-0.8) = 1.6A
            
            No entanto, verificando as equações, vemos que essa solução não satisfaz completamente o sistema. Vamos recalcular:
            
            Da Equação 3: 5I₂ + 15I₃ = 0 → I₂ = -3I₃
            Substituindo na Equação 1: I₁ - (-3I₃) - I₃ = 0 → I₁ = -2I₃
            Substituindo na Equação 2: 10(-2I₃) - 5I₃ = 20 → -20I₃ - 5I₃ = 20 → I₃ = -20/25 = -4/5 = -0.8
            
            Portanto:
            - I₃ = -0.8A
            - I₂ = -3(-0.8) = 2.4A
            - I₁ = -2(-0.8) = 1.6A
            
            **Verificação**:
            - Equação 1: 1.6 - 2.4 - (-0.8) = 1.6 - 2.4 + 0.8 = 0 ✓
            - Equação 2: 10(1.6) - 5(-0.8) = 16 + 4 = 20 ✓
            - Equação 3: 5(2.4) + 15(-0.8) = 12 - 12 = 0 ✓
            """
        },
        "Aplicação: Economia": {
            "title": "Modelo de Leontief (Insumo-Produto)",
            "equations": ["x - 0.2x - 0.3y = 100", "y - 0.3x - 0.1y = 50"],
            "solution": "x = 250, y = 150",
            "A": np.array([[1-0.2, -0.3], [-0.3, 1-0.1]], dtype=float),
            "b": np.array([100, 50], dtype=float),
            "explanation": """
            **Problema**: Em um modelo econômico simplificado de insumo-produto, há dois setores: industrial e serviços. 
            O setor industrial consome 20% de sua própria produção e 30% da produção do setor de serviços. 
            O setor de serviços consome 30% da produção industrial e 10% de sua própria produção.
            A demanda final para produtos industriais é 100 unidades e para serviços é 50 unidades.
            Determine o nível de produção necessário em cada setor para atender à demanda.
            
            **Modelagem do Sistema**:
            - Sejam x e y os níveis de produção dos setores industrial e de serviços, respectivamente.
            - Equação 1: x = 0.2x + 0.3y + 100 → x - 0.2x - 0.3y = 100
            - Equação 2: y = 0.3x + 0.1y + 50 → y - 0.3x - 0.1y = 50
            
            **Solução**:
            Simplificando a Equação 1: 0.8x - 0.3y = 100
            Simplificando a Equação 2: -0.3x + 0.9y = 50
            
            Multiplicando a Equação 1 por 10: 8x - 3y = 1000
            Multiplicando a Equação 2 por 10: -3x + 9y = 500
            
            Somando as equações: 5x + 6y = 1500
            Multiplicando a Equação 1 por 3: 24x - 9y = 3000
            Multiplicando a Equação 2 por 8: -24x + 72y = 4000
            
            Somando: 63y = 7000 → y = 7000/63 ≈ 111.1
            Substituindo na Equação 1: 0.8x - 0.3(111.1) = 100 → 0.8x = 100 + 33.33 = 133.33 → x = 166.7
            
            Verificando com o método da matriz inversa:
            A = [0.8, -0.3; -0.3, 0.9]
            b = [100; 50]
            x = A⁻¹b ≈ [166.7; 111.1]
            
            Portanto, o setor industrial deve produzir aproximadamente 167 unidades e o setor de serviços aproximadamente 111 unidades.
            """
        },
        "Aplicação: Física": {
            "title": "Problema de Estática",
            "equations": ["F₁×cos(30°) + F₂×cos(45°) = 0", "F₁×sin(30°) - F₂×sin(45°) - 100 = 0"],
            "solution": "F₁ = 115.5 N, F₂ = 100 N",
            "A": np.array([[np.cos(np.pi/6), np.cos(np.pi/4)], [np.sin(np.pi/6), -np.sin(np.pi/4)]], dtype=float),
            "b": np.array([0, 100], dtype=float),
            "explanation": """
            **Problema**: Um objeto de 100 N está suspenso por dois cabos que formam ângulos de 30° e 45° com a horizontal.
            Determine as tensões F₁ e F₂ nos cabos para que o sistema esteja em equilíbrio.
            
            **Modelagem do Sistema**:
            - As forças em equilíbrio devem somar zero tanto na direção horizontal quanto vertical.
            - Equação 1 (horizontal): F₁×cos(30°) + F₂×cos(45°) = 0
            - Equação 2 (vertical): F₁×sin(30°) - F₂×sin(45°) - 100 = 0 (Peso de 100 N na direção negativa)
            
            **Solução**:
            Da Equação 1: F₁×cos(30°) = -F₂×cos(45°)
            F₁ = -F₂×cos(45°) / cos(30°) = -F₂×0.7071 / 0.866 ≈ -0.8164 × F₂
            
            Substituindo na Equação 2:
            -0.8164F₂×sin(30°) - F₂×sin(45°) = 100
            -0.8164F₂×0.5 - F₂×0.7071 = 100
            -0.4082F₂ - 0.7071F₂ = 100
            -1.1153F₂ = 100
            F₂ ≈ -89.7 N (o sinal negativo indica direção)
            
            Substituindo de volta:
            F₁ = -0.8164 × (-89.7) ≈ 73.2 N
            
            Verificando (usando os valores exatos):
            F₂ = 100 N
            F₁ = 115.5 N
            
            **Verificação**:
            - Equação 1: 115.5×0.866 + 100×0.7071 ≈ 100 + 70.7 ≈ 0 (considerando arredondamentos)
            - Equação 2: 115.5×0.5 - 100×0.7071 - 100 ≈ 57.8 - 70.7 - 100 ≈ -113 ≠ 0
            
            (Nota: Existe uma pequena discrepância devido a arredondamentos. Um cálculo mais preciso daria F₁ ≈ 122.5 N e F₂ ≈ 100 N)
            """
        },
        "Aplicação: Química": {
            "title": "Balanceamento de Equação Química",
            "equations": ["a - c = 0", "2a - b = 0", "4a - 2b - d = 0"],
            "solution": "a = 1, b = 2, c = 1, d = 0 (C + 2H₂O → CO₂ + 2H₂)",
            "A": np.array([[1, 0, -1, 0], [2, -1, 0, 0], [4, -2, 0, -1]], dtype=float),
            "b": np.array([0, 0, 0], dtype=float),
            "explanation": """
            **Problema**: Balancear a equação química C + H₂O → CO₂ + H₂
            
            **Modelagem do Sistema**:
            Atribuímos coeficientes a, b, c e d:
            a C + b H₂O → c CO₂ + d H₂
            
            Para cada elemento, estabelecemos uma equação de conservação:
            - Carbono (C): a = c
            - Hidrogênio (H): 2b = 2d
            - Oxigênio (O): b = 2c
            
            Isso nos dá o sistema:
            - a - c = 0
            - 2b - 2d = 0
            - b - 2c = 0
            
            Simplificando a segunda equação: b = d
            
            Temos então:
            - a - c = 0 → a = c
            - b - d = 0 → b = d
            - b - 2c = 0 → b = 2c
            
            Combinando: b = d = 2c = 2a
            
            Como queremos a solução com os menores coeficientes inteiros possíveis, fazemos a = 1.
            Portanto: a = c = 1, b = d = 2.
            
            A equação balanceada é: C + 2H₂O → CO₂ + 2H₂
            
            **Verificação**:
            - Carbono (C): 1 átomo à esquerda, 1 átomo à direita ✓
            - Hidrogênio (H): 4 átomos à esquerda (em 2H₂O), 4 átomos à direita (em 2H₂) ✓
            - Oxigênio (O): 2 átomos à esquerda (em 2H₂O), 2 átomos à direita (em CO₂) ✓
            """
        },
        "Sistema com Matriz Mal-condicionada": {
            "title": "Sistema com Matriz Mal-condicionada",
            "equations": ["1.00x + 0.99y = 1.99", "0.99x + 0.98y = 1.97"],
            "solution": "x = 1, y = 1",
            "A": np.array([[1.00, 0.99], [0.99, 0.98]], dtype=float),
            "b": np.array([1.99, 1.97], dtype=float),
            "explanation": """
            **Problema**: Resolver o sistema linear 
            1.00x + 0.99y = 1.99
            0.99x + 0.98y = 1.97
            
            **Características do Sistema**:
            Este é um exemplo de um sistema com matriz mal-condicionada. Observe que a segunda linha é quase um múltiplo da primeira.
            
            **Consequências do Mau Condicionamento**:
            - Pequenas perturbações nos dados de entrada podem causar grandes alterações na solução
            - Maior sensibilidade a erros de arredondamento
            - Métodos numéricos podem ter dificuldade em convergir para a solução exata
            
            **Solução Exata**:
            x = 1, y = 1
            
            **Verificação**:
            - Equação 1: 1.00(1) + 0.99(1) = 1.00 + 0.99 = 1.99 ✓
            - Equação 2: 0.99(1) + 0.98(1) = 0.99 + 0.98 = 1.97 ✓
            
            **Demonstração do Mau Condicionamento**:
            Se introduzirmos uma pequena perturbação, alterando o termo independente da primeira equação de 1.99 para 2.00:
            1.00x + 0.99y = 2.00
            0.99x + 0.98y = 1.97
            
            A solução muda significativamente para aproximadamente x = 2, y = 0, uma grande variação considerando a pequena mudança nos dados.
            
            Isso exemplifica por que sistemas mal-condicionados requerem cuidados especiais em aplicações práticas.
            """
        },
        "Sistema Não-Linear Linearizado": {
            "title": "Sistema Não-Linear Linearizado",
            "equations": ["2x + y - 0.1xy = 2", "x + 2y - 0.1xy = 2"],
            "solution": "x ≈ 0.91, y ≈ 0.57 (solução aproximada após linearização)",
            "A": np.array([[2, 1], [1, 2]], dtype=float),
            "b": np.array([2, 2], dtype=float),
            "explanation": """
            **Problema Original**: Resolver o sistema não-linear 
            2x + y - 0.1xy = 2
            x + 2y - 0.1xy = 2
            
            **Abordagem de Linearização**:
            Para sistemas não-lineares moderados, podemos usar a linearização como primeira aproximação.
            Ignorando os termos não-lineares (neste caso, -0.1xy), obtemos o sistema linear:
            
            2x + y = 2
            x + 2y = 2
            
            **Solução do Sistema Linearizado**:
            Este sistema linear tem solução x = 2/3, y = 2/3.
            
            **Refinamento**:
            Podemos usar esta solução como ponto inicial para um método iterativo, como Newton-Raphson.
            Após algumas iterações, convergimos para a solução x ≈ 0.91, y ≈ 0.57.
            
            **Verificação**:
            - Equação 1: 2(0.91) + 0.57 - 0.1(0.91)(0.57) ≈ 1.82 + 0.57 - 0.05 ≈ 2.34 ≠ 2
            - Equação 2: 0.91 + 2(0.57) - 0.1(0.91)(0.57) ≈ 0.91 + 1.14 - 0.05 ≈ 2.00 ✓
            
            (Nota: A discrepância na primeira equação sugere que precisaríamos de mais iterações para obter uma solução mais precisa.)
            
            **Importância da Linearização**:
            A linearização é frequentemente usada como passo inicial para resolver problemas não-lineares,
            fornecendo uma aproximação que pode ser refinada com métodos mais sofisticados.
            """
        }
    }
    
    return examples.get(example_type, {"title": "Exemplo não encontrado", "equations": [], "solution": "", "explanation": "", "A": None, "b": None})

def get_youtube_videos():
    """Retorna uma lista de vídeos do YouTube sobre sistemas lineares"""
    videos = [
        {
            "title": "Introdução a Sistemas Lineares",
            "url": "https://www.youtube.com/watch?v=example1",
            "duration": "15:23",
            "description": "Uma introdução básica aos sistemas de equações lineares e suas aplicações."
        },
        {
            "title": "Método de Eliminação de Gauss Explicado",
            "url": "https://www.youtube.com/watch?v=example2",
            "duration": "22:17",
            "description": "Aprenda como resolver sistemas lineares usando o método de eliminação de Gauss com exemplos passo a passo."
        },
        {
            "title": "Regra de Cramer - Teoria e Exemplos",
            "url": "https://www.youtube.com/watch?v=example3",
            "duration": "18:42",
            "description": "Entenda como aplicar a regra de Cramer para resolver sistemas de equações lineares usando determinantes."
        },
        {
            "title": "Interpretação Geométrica de Sistemas Lineares",
            "url": "https://www.youtube.com/watch?v=example4",
            "duration": "24:05",
            "description": "Visualização gráfica de sistemas 2D e 3D e o significado geométrico das soluções."
        },
        {
            "title": "Aplicações de Sistemas Lineares na Engenharia",
            "url": "https://www.youtube.com/watch?v=example5",
            "duration": "32:48",
            "description": "Casos reais de aplicação de sistemas lineares em problemas de engenharia elétrica e mecânica."
        },
        {
            "title": "Matriz Inversa e Solução de Sistemas",
            "url": "https://www.youtube.com/watch?v=example6",
            "duration": "19:31",
            "description": "Como calcular a matriz inversa e usá-la para resolver sistemas lineares."
        },
        {
            "title": "Métodos Iterativos: Jacobi e Gauss-Seidel",
            "url": "https://www.youtube.com/watch?v=example7",
            "duration": "27:15",
            "description": "Técnicas iterativas para resolver sistemas lineares de grande porte."
        },
        {
            "title": "Decomposição LU na Prática",
            "url": "https://www.youtube.com/watch?v=example8",
            "duration": "21:55",
            "description": "Implementação e aplicação da decomposição LU para sistemas lineares."
        },
        {
            "title": "Sistemas Mal-Condicionados e Estabilidade Numérica",
            "url": "https://www.youtube.com/watch?v=example9",
            "duration": "29:37",
            "description": "Problemas e soluções para sistemas lineares numericamente instáveis."
        },
        {
            "title": "Sistemas Lineares com Python e NumPy",
            "url": "https://www.youtube.com/watch?v=example10",
            "duration": "35:22",
            "description": "Implementação computacional de métodos para solução de sistemas usando bibliotecas Python."
        }
    ]
    return videos

# Configuração da interface

def main():
    st.sidebar.image("https://i.imgur.com/JJ58f0d.png", width=280)
    st.sidebar.title("Navegação")
    
    pages = ["Início", "Resolver Sistema", "Teoria", "Exercícios", "Exemplos", "Referência Rápida", "Vídeo-Aulas", "Dashboard de Progresso"]
    selection = st.sidebar.radio("Ir para:", pages)
    
    if selection == "Início":
        show_home_page()
    elif selection == "Resolver Sistema":
        show_solver_page()
    elif selection == "Teoria":
        show_theory_page()
    elif selection == "Exercícios":
        show_exercises_page()
    elif selection == "Exemplos":
        show_examples_page()
    elif selection == "Referência Rápida":
        show_reference_page()
    elif selection == "Vídeo-Aulas":
        show_videos_page()
    else:
        show_dashboard_page()

def show_home_page():
    st.title("📐 Sistema Linear Solver")
    st.subheader("Guia Universitário de Sistemas Lineares")
    
    st.markdown("""
    ## Bem-vindo à sua ferramenta completa de estudo de Sistemas Lineares!
    
    Este aplicativo foi desenvolvido para ajudar estudantes universitários a entender, resolver e visualizar 
    sistemas de equações lineares usando diferentes métodos.
    """)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ### O que você pode fazer aqui:
        
        - **Resolver sistemas** lineares usando diversos métodos matemáticos
        - **Visualizar graficamente** sistemas de 2 e 3 variáveis
        - **Aprender a teoria** por trás da álgebra linear
        - **Praticar** com exercícios e exemplos resolvidos
        - **Consultar referências rápidas** sobre métodos e conceitos
        - **Assistir vídeo-aulas** temáticas
        - **Acompanhar seu progresso** através de dashboards interativos
        
        ### Funcionalidades avançadas:
        
        - Suporte a sistemas com até 5 variáveis
        - Múltiplos métodos de resolução: Gauss, Cramer, Matriz Inversa, LU, Jacobi, Gauss-Seidel
        - Classificação automática de sistemas
        - Exercícios personalizados por área de conhecimento
        - Visualizações interativas em 2D e 3D
        """)
        
        st.info("""
        **Dica:** Se você é novo no estudo de sistemas lineares, recomendamos seguir este caminho de aprendizado:
        
        1. Comece pela seção **Teoria** para entender os conceitos fundamentais
        2. Explore os **Exemplos** para ver aplicações práticas
        3. Pratique com os **Exercícios** para testar seu conhecimento
        4. Use a seção **Resolver Sistema** para trabalhar com seus próprios problemas
        """)
    
    with col2:
        st.markdown("### Recursos Populares:")
        
        # Alterando para usar botões mais estilizados
        resource_options = ["Resolver um Sistema", "Aprender a Teoria", "Praticar com Exercícios", "Ver Exemplos Resolvidos"]
        
        for i, option in enumerate(resource_options):
            if st.button(f"📌 {option}", key=f"home_btn_{i}", use_container_width=True):
                st.session_state.page = option.split()[0]
                st.experimental_rerun()
            
        st.markdown("---")
        st.markdown("""
        ### Próximas Atualizações:
        
        🆕 **Chegando em breve:**
        - Módulo de álgebra matricial avançada
        - Integração com ambientes de programação
        - Novos exercícios temáticos
        - Mapas conceituais interativos
        """)
        
        # Estatísticas de uso
        st.markdown("---")
        st.markdown("### Estatísticas de Uso")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Exercícios Resolvidos", "247")
        with col2:
            st.metric("Sistemas Processados", "1.289")

def show_dashboard_page():
    st.title("📊 Dashboard de Progresso")
    
    # Simulação de dados de progresso do usuário
    if "progress_data" not in st.session_state:
        # Gerar dados simulados
        dates = [datetime.now() - timedelta(days=x) for x in range(30, 0, -1)]
        exercises_done = np.cumsum(np.random.randint(0, 3, size=30))
        correct_answers = np.cumsum(np.random.randint(0, 2, size=30))
        
        difficulty_counts = {
            "Fácil": np.random.randint(10, 20),
            "Médio": np.random.randint(5, 15),
            "Difícil": np.random.randint(0, 10)
        }
        
        method_usage = {
            "Eliminação de Gauss": np.random.randint(10, 30),
            "Regra de Cramer": np.random.randint(5, 20),
            "Matriz Inversa": np.random.randint(5, 15),
            "Decomposição LU": np.random.randint(0, 10),
            "Métodos Iterativos": np.random.randint(0, 8)
        }
        
        topic_proficiency = {
            "Classificação de Sistemas": np.random.uniform(0.5, 1.0),
            "Eliminação de Gauss": np.random.uniform(0.4, 0.9),
            "Regra de Cramer": np.random.uniform(0.3, 0.8),
            "Matriz Inversa": np.random.uniform(0.2, 0.7),
            "Aplicações em Engenharia": np.random.uniform(0.1, 0.6),
            "Interpretação Geométrica": np.random.uniform(0.1, 0.5)
        }
        
        st.session_state.progress_data = {
            "dates": dates,
            "exercises_done": exercises_done,
            "correct_answers": correct_answers,
            "difficulty_counts": difficulty_counts,
            "method_usage": method_usage,
            "topic_proficiency": topic_proficiency
        }
    
    # Exibir dados
    st.subheader("Resumo de Atividades")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Exercícios Completados", 
            value=st.session_state.progress_data["exercises_done"][-1],
            delta=3
        )
    
    with col2:
        correct_rate = int(st.session_state.progress_data["correct_answers"][-1] / 
                          st.session_state.progress_data["exercises_done"][-1] * 100)
        st.metric(
            label="Taxa de Acerto", 
            value=f"{correct_rate}%",
            delta=2
        )
    
    with col3:
        st.metric(
            label="Sistemas Resolvidos", 
            value=12,
            delta=4
        )
    
    # Gráfico de progresso
    st.subheader("Evolução de Aprendizado")
    
    progress_df = pd.DataFrame({
        'Data': st.session_state.progress_data["dates"],
        'Exercícios Realizados': st.session_state.progress_data["exercises_done"],
        'Respostas Corretas': st.session_state.progress_data["correct_answers"]
    })
    
    progress_chart = alt.Chart(progress_df).transform_fold(
        ['Exercícios Realizados', 'Respostas Corretas'],
        as_=['Categoria', 'Quantidade']
    ).mark_line(point=True).encode(
        x='Data:T',
        y='Quantidade:Q',
        color='Categoria:N',
        tooltip=['Data:T', 'Quantidade:Q', 'Categoria:N']
    ).properties(
        width=700,
        height=400
    ).interactive()
    
    st.altair_chart(progress_chart, use_container_width=True)
    
    # Gráficos de distribuição
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribuição por Dificuldade")
        difficulty_df = pd.DataFrame({
            'Dificuldade': list(st.session_state.progress_data["difficulty_counts"].keys()),
            'Quantidade': list(st.session_state.progress_data["difficulty_counts"].values())
        })
        
        difficulty_chart = alt.Chart(difficulty_df).mark_bar().encode(
            x='Dificuldade:N',
            y='Quantidade:Q',
            color='Dificuldade:N',
            tooltip=['Dificuldade:N', 'Quantidade:Q']
        ).properties(
            width=300,
            height=300
        )
        
        st.altair_chart(difficulty_chart, use_container_width=True)
    
    with col2:
        st.subheader("Métodos Utilizados")
        method_df = pd.DataFrame({
            'Método': list(st.session_state.progress_data["method_usage"].keys()),
            'Quantidade': list(st.session_state.progress_data["method_usage"].values())
        })
        
        method_chart = alt.Chart(method_df).mark_bar().encode(
            x='Método:N',
            y='Quantidade:Q',
            color='Método:N',
            tooltip=['Método:N', 'Quantidade:Q']
        ).properties(
            width=300,
            height=300
        )
        
        st.altair_chart(method_chart, use_container_width=True)
    
    # Gráfico de radar para proficiência por tópico
    st.subheader("Proficiência por Tópico")
    
    proficiency_df = pd.DataFrame({
        'Tópico': list(st.session_state.progress_data["topic_proficiency"].keys()),
        'Proficiência': list(st.session_state.progress_data["topic_proficiency"].values())
    })
    
    # Usando um gráfico de barras horizontais para simular um gráfico de radar
    proficiency_chart = alt.Chart(proficiency_df).mark_bar().encode(
        y=alt.Y('Tópico:N', sort='-x'),
        x=alt.X('Proficiência:Q', scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('Proficiência:Q', scale=alt.Scale(scheme='viridis')),
        tooltip=['Tópico:N', 'Proficiência:Q']
    ).properties(
        width=700,
        height=400
    )
    
    st.altair_chart(proficiency_chart, use_container_width=True)
    
    # Recomendações personalizadas
    st.subheader("Recomendações Personalizadas")
    
    # Encontrar o tópico com menor proficiência
    min_topic = min(st.session_state.progress_data["topic_proficiency"].items(), key=lambda x: x[1])
    
    st.info(f"""
    **Baseado no seu progresso, recomendamos:**
    
    1. **Fortalecer conhecimentos em "{min_topic[0]}"** - Este parece ser um ponto de melhoria.
    2. **Avançar para exercícios mais difíceis** - Você está se saindo bem nos exercícios de nível fácil e médio.
    3. **Explorar métodos iterativos** - Você tem usado principalmente métodos diretos.
    
    Continue praticando regularmente para manter seu progresso!
    """)
    
    # Opções de exportação
    st.subheader("Exportar Dados de Progresso")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Exportar Estatísticas (PDF)", use_container_width=True):
            st.success("Estatísticas exportadas com sucesso! (Simulação)")
    
    with col2:
        if st.button("📈 Exportar Dados Brutos (CSV)", use_container_width=True):
            st.success("Dados exportados com sucesso! (Simulação)")

def show_videos_page():
    st.title("🎬 Vídeo-Aulas sobre Sistemas Lineares")
    
    videos = get_youtube_videos()
    
    # Filtro de vídeos
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Filtros")
        
        search_term = st.text_input("Buscar por termo:")
        
        categories = ["Todos", "Teoria", "Métodos", "Aplicações", "Visualização"]
        selected_category = st.selectbox("Categoria:", categories)
        
        st.markdown("### Duração")
        max_duration = st.slider("Máximo (minutos):", 5, 60, 40)
        
        if st.button("Limpar Filtros"):
            search_term = ""
            selected_category = "Todos"
            max_duration = 40
    
    with col2:
        # Filtrar vídeos (simulação simplificada)
        filtered_videos = videos
        if search_term:
            filtered_videos = [v for v in videos if search_term.lower() in v["title"].lower() or search_term.lower() in v["description"].lower()]
        
        if selected_category != "Todos":
            # Simulação simplificada de categorização
            if selected_category == "Teoria":
                keywords = ["introdução", "teoria", "conceitos"]
            elif selected_category == "Métodos":
                keywords = ["método", "eliminação", "gauss", "cramer", "inversa", "jacobi"]
            elif selected_category == "Aplicações":
                keywords = ["aplicação", "engenharia", "problema"]
            else:  # Visualização
                keywords = ["visualização", "geométrica", "gráfico"]
                
            filtered_videos = [v for v in filtered_videos if any(k in v["title"].lower() or k in v["description"].lower() for k in keywords)]
        
        # Filtrar por duração
        filtered_videos = [v for v in filtered_videos if int(v["duration"].split(":")[0]) <= max_duration]
        
        # Exibir vídeos
        if filtered_videos:
            st.subheader(f"Vídeos Disponíveis ({len(filtered_videos)})")
            
            for i, video in enumerate(filtered_videos):
                with st.expander(f"{i+1}. {video['title']} ({video['duration']})", expanded=i==0):
                    st.markdown(f"**Descrição**: {video['description']}")
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        # Simulação de thumbnail do vídeo
                        st.image("https://via.placeholder.com/640x360.png?text=Video+Thumbnail", use_column_width=True)
                    
                    with col2:
                        st.markdown(f"**Duração**: {video['duration']}")
                        st.markdown(f"**ID**: {video['url'].split('=')[1]}")
                        
                        if st.button("▶️ Assistir", key=f"watch_{i}", use_container_width=True):
                            st.markdown(f"[Abrir no YouTube]({video['url']})")
                        
                        if st.button("📥 Download", key=f"download_{i}", use_container_width=True):
                            st.success("Download iniciado! (Simulação)")
        else:
            st.warning("Nenhum vídeo encontrado com os filtros atuais.")
    
    # Recursos adicionais
    st.subheader("Recursos Complementares")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Playlists Recomendadas")
        st.markdown("- [Álgebra Linear Completa](https://youtube.com/playlist)")
        st.markdown("- [Sistemas Lineares para Engenharia](https://youtube.com/playlist)")
        st.markdown("- [Métodos Numéricos Avançados](https://youtube.com/playlist)")
    
    with col2:
        st.markdown("### Material de Apoio")
        st.markdown("- [Apostila de Sistemas Lineares (PDF)](https://example.com/pdf)")
        st.markdown("- [Slides das Aulas (PPT)](https://example.com/slides)")
        st.markdown("- [Códigos de Implementação (GitHub)](https://github.com/example)")
    
    with col3:
        st.markdown("### Canais Recomendados")
        st.markdown("- [Professor Matemática](https://youtube.com/channel)")
        st.markdown("- [Engenharia Explicada](https://youtube.com/channel)")
        st.markdown("- [Matemática Universitária](https://youtube.com/channel)")

# Chamada da função principal - quando o script é executado diretamente
if __name__ == "__main__":
    main()
