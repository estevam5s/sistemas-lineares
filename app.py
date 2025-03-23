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

def get_practice_exercise(level):
    """Gera exercícios de prática com base no nível de dificuldade"""
    if level == "Fácil":
        # Sistema 2x2 com solução inteira
        A = np.array([[1, 1], [2, 1]])
        x = np.array([5, 3])  # Solução desejada
        b = np.dot(A, x)
        
        question = "Resolva o sistema de equações lineares:"
        equations = [
            f"{format_equation(A[0], ['x', 'y'], b[0])}",
            f"{format_equation(A[1], ['x', 'y'], b[1])}"
        ]
        
        return A, b, question, equations, x
        
    elif level == "Médio":
        # Sistema 3x3 com solução inteira
        A = np.array([[2, 1, -1], [3, -2, 1], [1, 2, 2]])
        x = np.array([1, 2, 3])  # Solução desejada
        b = np.dot(A, x)
        
        question = "Resolva o sistema de equações lineares:"
        equations = [
            f"{format_equation(A[0], ['x', 'y', 'z'], b[0])}",
            f"{format_equation(A[1], ['x', 'y', 'z'], b[1])}",
            f"{format_equation(A[2], ['x', 'y', 'z'], b[2])}"
        ]
        
        return A, b, question, equations, x
        
    else:  # Difícil
        # Sistema com solução não inteira ou classificação especial
        r = np.random.choice(["SPD_complex", "SPI", "SI"])
        
        if r == "SPD_complex":
            A = np.array([[3, 1, -2], [2, -2, 1], [1, 5, -3]])
            x = np.array([1/3, 2/3, 1/3])  # Solução fracionária
            b = np.dot(A, x)
            
        elif r == "SPI":
            # Sistema com infinitas soluções
            A = np.array([[1, 2, 3], [2, 4, 6], [3, 5, 7]])
            b = np.array([6, 12, 15])
            x = None
            
        else:  # SI
            # Sistema impossível
            A = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
            b = np.array([6, 12, 19])  # Inconsistente
            x = None
        
        question = "Resolva e classifique o sistema de equações lineares:"
        equations = [
            f"{format_equation(A[0], ['x', 'y', 'z'], b[0])}",
            f"{format_equation(A[1], ['x', 'y', 'z'], b[1])}",
            f"{format_equation(A[2], ['x', 'y', 'z'], b[2])}"
        ]
        
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
        }
    }
    
    return examples.get(example_type, {"title": "Exemplo não encontrado", "equations": [], "solution": "", "explanation": "", "A": None, "b": None})

# Configuração da interface

def main():
    st.sidebar.image("https://i.imgur.com/JJ58f0d.png", width=280)
    st.sidebar.title("Navegação")
    
    pages = ["Início", "Resolver Sistema", "Teoria", "Exercícios", "Exemplos", "Referência Rápida"]
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
    else:
        show_reference_page()

def show_home_page():
    st.title("📐 Sistema Linear Solver")
    st.subheader("Guia Universitário de Sistemas Lineares")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Bem-vindo à sua ferramenta de estudo de Sistemas Lineares!
        
        Este aplicativo foi desenvolvido para ajudar estudantes universitários a entender, resolver e visualizar 
        sistemas de equações lineares usando diferentes métodos.
        
        ### O que você pode fazer aqui:
        
        - **Resolver sistemas** lineares usando diversos métodos
        - **Visualizar graficamente** sistemas de 2 e 3 variáveis
        - **Aprender a teoria** por trás da álgebra linear
        - **Praticar** com exercícios e exemplos resolvidos
        - **Consultar referências rápidas** sobre métodos e conceitos
        
        ### Começar a usar:
        
        Use o menu de navegação à esquerda para explorar as diferentes funcionalidades do aplicativo.
        """)
        
        st.info("""
        **Dica:** Se você é novo no estudo de sistemas lineares, recomendamos começar pela seção "Teoria" 
        para entender os conceitos fundamentais, e depois explorar os "Exemplos" para ver aplicações práticas.
        """)
        
    with col2:
        st.markdown("""
        ### Recursos Populares:
        """)
        
        if st.button("🔍 Resolver um Sistema", key="home_btn1"):
            st.session_state.page = "Resolver Sistema"
            st.experimental_rerun()
            
        if st.button("📚 Aprender a Teoria", key="home_btn2"):
            st.session_state.page = "Teoria"
            st.experimental_rerun()
            
        if st.button("✏️ Praticar com Exercícios", key="home_btn3"):
            st.session_state.page = "Exercícios"
            st.experimental_rerun()
            
        if st.button("📋 Ver Exemplos Resolvidos", key="home_btn4"):
            st.session_state.page = "Exemplos"
            st.experimental_rerun()
        
        st.markdown("---")
        st.markdown("""
        ### Próxima Aula:
        📅 **Determinantes e Matrizes Inversas**
        🕒 Amanhã, 14h00
        """)

def show_solver_page():
    st.title("🧮 Resolver Sistema Linear")
    
    tab1, tab2, tab3 = st.tabs(["📝 Inserir Sistema", "🔍 Resultados", "📊 Visualização"])
    
    with tab1:
        st.subheader("Insira seu sistema de equações lineares")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            system_input_method = st.radio(
                "Método de entrada:",
                ["Manual (Coeficientes)", "Equações (Texto)"],
                horizontal=True
            )
            
        with col2:
            vars_count = st.number_input("Número de variáveis:", min_value=2, max_value=4, value=2)
            
        if system_input_method == "Manual (Coeficientes)":
            equations_count = st.number_input("Número de equações:", min_value=1, max_value=5, value=vars_count)
            
            coeffs = []
            constants = []
            
            st.markdown("### Insira os coeficientes e termos independentes")
            
            var_names = ["x", "y", "z", "w"][:vars_count]
            
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
                
        else:  # Entrada por texto
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
                coeffs = []
                constants = []
                
                var_symbols = []
                for i in range(vars_count):
                    if i < len(["x", "y", "z", "w"]):
                        var_symbols.append(sp.symbols(["x", "y", "z", "w"][i]))
                
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
                    var_names = ["x", "y", "z", "w"][:vars_count]
                    eq_str = format_equation(eq_coef, var_names, eq_const)
                    st.write(f"Equação {i+1}: {eq_str}")
                    
            except Exception as e:
                st.error(f"Erro ao processar as equações: {str(e)}")
                st.stop()
        
        # Método de resolução
        st.markdown("### Método de Resolução")
        
        col1, col2 = st.columns(2)
        
        with col1:
            solution_method = st.selectbox(
                "Escolha o método:",
                ["Eliminação de Gauss", "Regra de Cramer", "Matriz Inversa", "Todos os Métodos"]
            )
            
        with col2:
            show_steps = st.checkbox("Mostrar passos detalhados", value=True)
        
        A, b = create_system_matrix(coeffs, constants, vars_count)
        
        # Botão para resolver
        if st.button("Resolver Sistema", type="primary"):
            st.session_state.system_solved = True
            st.session_state.A = A
            st.session_state.b = b
            st.session_state.vars_count = vars_count
            st.session_state.solution_method = solution_method
            st.session_state.show_steps = show_steps
            st.session_state.system_classification = classify_system(A, b)
            
            # Computar soluções pelos diferentes métodos
            results = {}
            
            if solution_method in ["Eliminação de Gauss", "Todos os Métodos"]:
                steps, solution = gaussian_elimination_steps(A, b)
                results["Eliminação de Gauss"] = {"steps": steps, "solution": solution}
                
            if vars_count <= 4 and solution_method in ["Regra de Cramer", "Todos os Métodos"]:
                steps, solution = cramer_rule(A, b, detailed=show_steps)
                results["Regra de Cramer"] = {"steps": steps, "solution": solution}
                
            if solution_method in ["Matriz Inversa", "Todos os Métodos"]:
                steps, solution = matrix_inverse_method(A, b, detailed=show_steps)
                results["Matriz Inversa"] = {"steps": steps, "solution": solution}
                
            st.session_state.results = results
            
            # Mudar para a aba de resultados
            st.experimental_rerun()
            
    with tab2:
        if not hasattr(st.session_state, 'system_solved') or not st.session_state.system_solved:
            st.info("Insira e resolva um sistema na aba 'Inserir Sistema'")
            st.stop()
        
        st.subheader("Resultados da Resolução")
        
        # Exibir classificação do sistema
        st.markdown(f"**Classificação do Sistema:** {st.session_state.system_classification}")
        
        # Exibir solução para cada método
        for method, result in st.session_state.results.items():
            st.markdown(f"### {method}")
            
            steps = result["steps"]
            solution = result["solution"]
            
            if solution is not None:
                st.markdown("**Solução encontrada:**")
                for i, val in enumerate(solution):
                    var_name = ["x", "y", "z", "w"][i] if i < 4 else f"x_{i+1}"
                    st.markdown(f"- {var_name} = {val:.4f}")
            else:
                st.write("Não foi possível encontrar uma solução única por este método.")
            
            if st.session_state.show_steps:
                with st.expander("Ver passos detalhados", expanded=False):
                    for step in steps:
                        st.write(step)
        
        # Adicionar interpretação da solução
        if st.session_state.system_classification == "Sistema Possível e Determinado (SPD)":
            st.markdown("### Interpretação da Solução")
            st.write("O sistema possui uma única solução, que satisfaz todas as equações simultaneamente.")
            
            # Verificação da solução
            st.markdown("### Verificação")
            
            # Obter uma solução válida (qualquer uma)
            solution = None
            for result in st.session_state.results.values():
                if result["solution"] is not None:
                    solution = result["solution"]
                    break
            
            if solution is not None:
                A = st.session_state.A
                b = st.session_state.b
                
                for i in range(len(b)):
                    eq_result = np.dot(A[i], solution)
                    is_correct = abs(eq_result - b[i]) < 1e-10
                    
                    var_names = ["x", "y", "z", "w"][:st.session_state.vars_count]
                    eq_str = format_equation(A[i], var_names, b[i])
                    
                    substitution = " + ".join([f"{A[i][j]:.2f} × {solution[j]:.4f}" for j in range(len(solution)) if abs(A[i][j]) > 1e-10])
                    
                    result_str = f"{eq_result:.4f} ≈ {b[i]:.4f}" if is_correct else f"{eq_result:.4f} ≠ {b[i]:.4f}"
                    
                    if is_correct:
                        st.success(f"Equação {i+1}: {eq_str}\n{substitution} = {result_str} ✓")
                    else:
                        st.error(f"Equação {i+1}: {eq_str}\n{substitution} = {result_str} ✗")
                        
        elif st.session_state.system_classification == "Sistema Possível e Indeterminado (SPI)":
            st.markdown("### Interpretação da Solução")
            st.write("""
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
            st.markdown("### Interpretação da Solução")
            st.write("""
            O sistema não possui solução. Isso ocorre porque as equações são inconsistentes entre si,
            ou seja, não existe um conjunto de valores para as variáveis que satisfaça todas as equações
            simultaneamente.
            
            Geometricamente, isso pode ser interpretado como:
            - Em 2D: retas paralelas que nunca se intersectam
            - Em 3D: planos sem ponto comum de interseção
            """)
    
    with tab3:
        if not hasattr(st.session_state, 'system_solved') or not st.session_state.system_solved:
            st.info("Insira e resolva um sistema na aba 'Inserir Sistema'")
            st.stop()
        
        st.subheader("Visualização Gráfica")
        
        if st.session_state.vars_count == 2:
            try:
                fig = plot_2d_system(st.session_state.A, st.session_state.b)
                if fig:
                    st.pyplot(fig)
                    
                    # Adicionar interpretação geométrica
                    st.markdown("### Interpretação Geométrica")
                    
                    if st.session_state.system_classification == "Sistema Possível e Determinado (SPD)":
                        st.write("""
                        Cada equação do sistema representa uma reta no plano cartesiano.
                        A solução do sistema é o ponto de interseção entre estas retas.
                        """)
                    elif st.session_state.system_classification == "Sistema Possível e Indeterminado (SPI)":
                        st.write("""
                        As retas são coincidentes (sobrepostas), o que significa que qualquer
                        ponto em uma das retas é uma solução válida para o sistema.
                        """)
                    else:  # SI
                        st.write("""
                        As retas são paralelas, o que indica que não há ponto de interseção
                        e, portanto, o sistema não possui solução.
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
                        st.write("""
                        Cada equação do sistema representa um plano no espaço tridimensional.
                        A solução do sistema é o ponto único de interseção entre estes planos.
                        """)
                    elif st.session_state.system_classification == "Sistema Possível e Indeterminado (SPI)":
                        st.write("""
                        Os planos se intersectam em uma reta ou em um plano comum,
                        resultando em infinitas soluções possíveis.
                        """)
                    else:  # SI
                        st.write("""
                        Os planos não possuem um ponto comum de interseção,
                        o que indica que o sistema não tem solução.
                        """)
                else:
                    st.warning("Não foi possível gerar a visualização 3D do sistema.")
            except Exception as e:
                st.error(f"Erro ao gerar o gráfico 3D: {str(e)}")
                
        else:
            st.info("A visualização gráfica está disponível apenas para sistemas com 2 ou 3 variáveis.")

def show_theory_page():
    st.title("📚 Teoria dos Sistemas Lineares")
    
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
            
            O método de Eliminação de Gauss é um dos algoritmos mais importantes para resolver sistemas lineares. Consiste em transformar o sistema em uma forma triangular superior (escalonada) através de operações elementares.
            
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
        "Regra de Cramer": {
            "content": """
            # Regra de Cramer
            
            A Regra de Cramer é um método para resolver sistemas lineares usando determinantes. É aplicável apenas a sistemas com mesmo número de equações e incógnitas, onde o determinante da matriz dos coeficientes é diferente de zero (sistemas SPD).
            
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
        "Aplicações de Sistemas Lineares": {
            "content": """
            # Aplicações de Sistemas Lineares
            
            Os sistemas de equações lineares são ferramentas fundamentais em diversas áreas. Veja algumas aplicações importantes:
            
            ## Física e Engenharia
            
            ### Análise de Circuitos Elétricos
            - Leis de Kirchhoff: correntes em um nó e tensões em um laço
            - Cada equação representa a conservação de corrente ou tensão
            
            ### Estática e Dinâmica
            - Equilíbrio de forças e momentos em estruturas
            - Análise de treliças e vigas
            
            ### Transferência de Calor
            - Modelagem de problemas de condução térmica
            - Métodos de diferenças finitas para equações diferenciais
            
            ## Economia
            
            ### Análise de Insumo-Produto de Leontief
            - Modelagem das interdependências entre diferentes setores da economia
            - Previsão de como mudanças em um setor afetam outros setores
            
            ### Otimização e Programação Linear
            - Maximização de lucro ou minimização de custos
            - Alocação ótima de recursos limitados
            
            ## Química
            
            ### Balanceamento de Equações Químicas
            - Cada elemento forma uma equação linear
            - Os coeficientes estequiométricos são as incógnitas
            
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
            
            ### Processamento de Alimentos
            - Mistura de ingredientes para atingir perfis nutricionais
            
            ## Tráfego e Transporte
            
            ### Fluxo de Redes
            - Modelagem de fluxo de tráfego em redes de transporte
            - Otimização de rotas
            
            ## Exemplo Prático: Problema de Mistura
            
            **Problema**: Uma farmácia precisa preparar 100 ml de uma solução com concentração de 25% de um princípio ativo. Dispõe-se de soluções com concentrações de 10%, 20% e 50%. Quanto de cada solução deve ser usado?
            
            **Modelagem**:
            - Sejam x, y e z as quantidades (em ml) das soluções com 10%, 20% e 50%
            - Equação 1: x + y + z = 100 (volume total)
            - Equação 2: 0.10x + 0.20y + 0.50z = 0.25 × 100 (quantidade do princípio ativo)
            
            **Sistema**:
            
            $$
            \\begin{align}
            x + y + z &= 100\\\\
            0.1x + 0.2y + 0.5z &= 25
            \\end{align}
            $$
            
            Este sistema tem infinitas soluções (SPI). Uma possível estratégia é fixar um valor para uma das variáveis e resolver para as outras duas.
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
        
        # Botão para baixar o material em PDF (simulado)
        if st.button("📥 Baixar Material em PDF"):
            st.success("Download iniciado! (Simulação)")
        
        # Botão para acessar videoaulas (simulado)
        if st.button("🎬 Acessar Videoaulas"):
            st.info("Redirecionando para as videoaulas... (Simulação)")
    
    with col2:
        st.markdown(theory_topics[selected_topic]["content"])

def show_exercises_page():
    st.title("✏️ Exercícios de Sistemas Lineares")
    
    tab1, tab2 = st.tabs(["📝 Praticar", "📋 Histórico"])
    
    with tab1:
        st.subheader("Pratique seus conhecimentos")
        
        difficulty = st.select_slider(
            "Nível de dificuldade:",
            options=["Fácil", "Médio", "Difícil"],
            value="Médio"
        )
        
        if "current_exercise" not in st.session_state or st.button("Gerar Novo Exercício"):
            A, b, question, equations, solution = get_practice_exercise(difficulty)
            st.session_state.current_exercise = {
                "A": A,
                "b": b,
                "question": question,
                "equations": equations,
                "solution": solution,
                "difficulty": difficulty,
                "system_type": classify_system(A, b)
            }
        
        # Mostrar o exercício atual
        st.markdown(f"### {st.session_state.current_exercise['question']}")
        
        for i, eq in enumerate(st.session_state.current_exercise["equations"]):
            st.markdown(f"{i+1}. {eq}")
        
        # Adicionar visualização se for sistema 2x2 ou 3x3
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
        user_answer = st.text_input(
            "Sua resposta:",
            help="Para SPD, insira os valores das variáveis separados por espaço ou vírgula. Para SPI ou SI, classifique o sistema."
        )
        
        # Verificar resposta
        if st.button("Verificar Resposta"):
            if user_answer:
                is_correct, message = check_user_answer(
                    user_answer, 
                    st.session_state.current_exercise["solution"], 
                    st.session_state.current_exercise["system_type"]
                )
                
                if is_correct:
                    st.success(f"✅ Correto! {message}")
                    
                    # Salvar no histórico
                    if "exercise_history" not in st.session_state:
                        st.session_state.exercise_history = []
                    
                    st.session_state.exercise_history.append({
                        "question": st.session_state.current_exercise["question"],
                        "equations": st.session_state.current_exercise["equations"],
                        "answer": user_answer,
                        "is_correct": True,
                        "difficulty": st.session_state.current_exercise["difficulty"]
                    })
                else:
                    st.error(f"❌ Incorreto. {message}")
            else:
                st.warning("Por favor, insira uma resposta.")
        
        # Botão para ver a solução
        if st.button("Ver Solução"):
            st.markdown("### Solução Detalhada")
            
            system_type = st.session_state.current_exercise["system_type"]
            st.markdown(f"**Classificação do Sistema**: {system_type}")
            
            if system_type == "Sistema Possível e Determinado (SPD)":
                solution = st.session_state.current_exercise["solution"]
                
                # Mostrar solução usando eliminação de Gauss
                steps, _ = gaussian_elimination_steps(A, b)
                
                st.markdown("**Método de Eliminação de Gauss:**")
                for step in steps:
                    st.write(step)
                
                st.markdown("**Resposta final:**")
                var_names = ["x", "y", "z", "w"][:A.shape[1]]
                for i, val in enumerate(solution):
                    st.markdown(f"- {var_names[i]} = {val}")
                
            elif system_type == "Sistema Possível e Indeterminado (SPI)":
                st.write("""
                Este sistema possui infinitas soluções. 
                
                A matriz dos coeficientes tem posto menor que o número de variáveis, o que significa 
                que algumas variáveis podem ser expressas em termos de outras (parâmetros livres).
                """)
                
                # Tentar obter solução simbólica
                try:
                    symbolic_solution, var_symbols = sympy_solve_system(A, b)
                    
                    if symbolic_solution:
                        st.markdown("**Solução Paramétrica:**")
                        
                        if isinstance(symbolic_solution, dict):
                            for var, expr in symbolic_solution.items():
                                st.latex(f"{sp.latex(var)} = {sp.latex(expr)}")
                        else:
                            st.latex(sp.latex(symbolic_solution))
                except:
                    st.warning("Não foi possível obter uma representação paramétrica da solução.")
                
            else:  # SI
                st.write("""
                Este sistema não possui solução. 
                
                O posto da matriz de coeficientes é menor que o posto da matriz ampliada, o que significa 
                que o sistema contém equações inconsistentes entre si.
                """)
            
            # Salvar no histórico mesmo se o usuário viu a solução sem tentar
            if "exercise_history" not in st.session_state:
                st.session_state.exercise_history = []
            
            # Verificar se este exercício já está no histórico para não duplicar
            already_in_history = False
            for entry in st.session_state.exercise_history:
                if entry["equations"] == st.session_state.current_exercise["equations"]:
                    already_in_history = True
                    break
            
            if not already_in_history:
                st.session_state.exercise_history.append({
                    "question": st.session_state.current_exercise["question"],
                    "equations": st.session_state.current_exercise["equations"],
                    "answer": "Solução vista",
                    "is_correct": False,
                    "difficulty": st.session_state.current_exercise["difficulty"]
                })
    
    with tab2:
        if "exercise_history" not in st.session_state or not st.session_state.exercise_history:
            st.info("Seu histórico de exercícios aparecerá aqui após você resolver alguns problemas.")
        else:
            st.subheader("Seu histórico de exercícios")
            
            # Estatísticas
            total = len(st.session_state.exercise_history)
            correct = sum(1 for e in st.session_state.exercise_history if e["is_correct"])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total de Exercícios", total)
            with col2:
                st.metric("Acertos", correct)
            with col3:
                st.metric("Taxa de Acerto", f"{int(correct/total*100)}%" if total > 0 else "0%")
            
            # Tabela de histórico
            history_data = []
            for i, exercise in enumerate(st.session_state.exercise_history[::-1]):  # Mais recente primeiro
                history_data.append({
                    "ID": i+1,
                    "Nível": exercise["difficulty"],
                    "Questão": "\n".join(exercise["equations"]),
                    "Resposta": exercise["answer"],
                    "Resultado": "✅ Correto" if exercise["is_correct"] else "❌ Incorreto"
                })
            
            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df, use_container_width=True)
            
            # Botão para limpar histórico
            if st.button("Limpar Histórico"):
                st.session_state.exercise_history = []
                st.experimental_rerun()

def show_examples_page():
    st.title("📋 Exemplos Resolvidos")
    
    example_types = [
        "Sistema 2×2 (SPD)",
        "Sistema 2×2 (SPI)",
        "Sistema 2×2 (SI)",
        "Sistema 3×3 (SPD)",
        "Sistema 3×3 (SPI)",
        "Sistema 3×3 (SI)",
        "Aplicação: Mistura",
        "Aplicação: Circuitos"
    ]
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_example = st.radio(
            "Selecione um exemplo:",
            example_types,
            key="example_type"
        )
        
        st.markdown("---")
        st.markdown("### Métodos de Resolução")
        
        methods = [
            "Eliminação de Gauss",
            "Regra de Cramer",
            "Matriz Inversa"
        ]
        
        selected_methods = st.multiselect(
            "Mostrar solução por:",
            methods,
            default=["Eliminação de Gauss"]
        )
    
    with col2:
        example = get_example_system(selected_example)
        
        st.header(example["title"])
        
        st.markdown("### Sistema de Equações")
        for i, eq in enumerate(example["equations"]):
            st.write(f"Equação {i+1}: {eq}")
        
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
        st.markdown(example["solution"])
        
        # Mostrar métodos de resolução selecionados
        if "Eliminação de Gauss" in selected_methods and example["A"] is not None and example["b"] is not None:
            st.markdown("### Método de Eliminação de Gauss")
            
            try:
                steps, solution = gaussian_elimination_steps(example["A"], example["b"])
                
                with st.expander("Ver passos detalhados", expanded=False):
                    for step in steps:
                        st.write(step)
                
                if solution is not None:
                    st.markdown("**Resultado:**")
                    var_names = ["x", "y", "z", "w"][:example["A"].shape[1]]
                    for i, val in enumerate(solution):
                        st.markdown(f"- {var_names[i]} = {val:.4f}")
                else:
                    st.write("Este método não pode determinar uma solução única.")
            except:
                st.error("Não foi possível aplicar o método de Eliminação de Gauss para este sistema.")
        
        if "Regra de Cramer" in selected_methods and example["A"] is not None and example["b"] is not None:
            st.markdown("### Regra de Cramer")
            
            # Verificar se a matriz é quadrada
            if example["A"].shape[0] == example["A"].shape[1]:
                try:
                    steps, solution = cramer_rule(example["A"], example["b"], detailed=True)
                    
                    with st.expander("Ver passos detalhados", expanded=False):
                        for step in steps:
                            st.write(step)
                    
                    if solution is not None:
                        st.markdown("**Resultado:**")
                        var_names = ["x", "y", "z", "w"][:example["A"].shape[1]]
                        for i, val in enumerate(solution):
                            st.markdown(f"- {var_names[i]} = {val:.4f}")
                    else:
                        st.write("A Regra de Cramer não pode ser aplicada a este sistema (determinante zero).")
                except:
                    st.error("Não foi possível aplicar a Regra de Cramer para este sistema.")
            else:
                st.info("A Regra de Cramer só pode ser aplicada a sistemas com o mesmo número de equações e incógnitas.")
        
        if "Matriz Inversa" in selected_methods and example["A"] is not None and example["b"] is not None:
            st.markdown("### Método da Matriz Inversa")
            
            # Verificar se a matriz é quadrada
            if example["A"].shape[0] == example["A"].shape[1]:
                try:
                    steps, solution = matrix_inverse_method(example["A"], example["b"], detailed=True)
                    
                    with st.expander("Ver passos detalhados", expanded=False):
                        for step in steps:
                            st.write(step)
                    
                    if solution is not None:
                        st.markdown("**Resultado:**")
                        var_names = ["x", "y", "z", "w"][:example["A"].shape[1]]
                        for i, val in enumerate(solution):
                            st.markdown(f"- {var_names[i]} = {val:.4f}")
                    else:
                        st.write("O método da Matriz Inversa não pode ser aplicado a este sistema (matriz singular).")
                except:
                    st.error("Não foi possível aplicar o método da Matriz Inversa para este sistema.")
            else:
                st.info("O método da Matriz Inversa só pode ser aplicado a sistemas com o mesmo número de equações e incógnitas.")
        
        st.markdown("### Explicação")
        st.markdown(example["explanation"])

def show_reference_page():
    st.title("📚 Referência Rápida")
    
    reference_topics = [
        "Classificação de Sistemas",
        "Método de Eliminação de Gauss",
        "Regra de Cramer",
        "Método da Matriz Inversa",
        "Interpretação Geométrica",
        "Teorema de Rouché-Capelli"
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
        
        if st.button("📥 Baixar como PDF"):
            st.success("Download iniciado! (Simulação)")
            
        if st.button("📱 Versão para Celular"):
            st.success("Versão para celular disponível! (Simulação)")
    
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

# Executar a aplicação
if __name__ == "__main__":
    main()
