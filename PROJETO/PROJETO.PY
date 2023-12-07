import numpy as np

def entrada_float(mensagem):
    return float(input(mensagem))

def simplex_interativo():
    # Entrada do número de variáveis e restrições
    num_vars = int(input("Digite o número de variáveis: "))
    num_constraints = int(input("Digite o número de restrições: "))

    # Entrada dos coeficientes da função objetivo
    c = np.array([entrada_float(f"Digite o coeficiente da variável x{i+1} na função objetivo: ") for i in range(num_vars)])

    # Entrada dos coeficientes das restrições
    A = np.array([[entrada_float(f"Digite o coeficiente da variável x{j+1} na restrição {i+1}: ") for j in range(num_vars)] for i in range(num_constraints)])

    # Entrada do lado direito das inequações
    b = np.array([entrada_float(f"Digite o valor do lado direito da restrição {i+1}: ") for i in range(num_constraints)])

    # Adicionando variáveis de folga
    A = np.hstack((A, np.eye(num_constraints)))
    c = np.concatenate((c, np.zeros(num_constraints)))

    # Construindo o tableau inicial
    tableau = np.vstack((np.column_stack((A, b)), np.concatenate((-c, [0]))))

    while True:
        # Verificando se a solução é ótima
        if np.all(tableau[-1, :-num_constraints] >= 0):
            break

        # Selecionando a coluna pivô
        pivot_col = np.argmin(tableau[-1, :-num_constraints])

        # Calculando as razões para determinar a linha pivô
        ratios = tableau[:-1, -1] / tableau[:-1, pivot_col]
        valid_ratios = ratios[ratios > 0]
        if len(valid_ratios) == 0:
            return None  # Sem solução
        pivot_row = np.where(ratios == np.min(valid_ratios))[0][0]

        # Atualizando o tableau
        tableau[pivot_row] /= tableau[pivot_row, pivot_col]
        for i in range(len(tableau)):
            if i != pivot_row:
                tableau[i] -= tableau[i, pivot_col] * tableau[pivot_row]

    # Extraindo a solução
    solution = np.zeros(num_vars)
    for i in range(num_vars):
        col = tableau[:, i]
        if np.sum(col == 1) == 1 and np.sum(col == 0) == len(col) - 1:
            solution[i] = tableau[np.where(col == 1)[0][0], -1]

    # Calculando o valor ótimo
    valor_otimo = tableau[-1, -1]

    # Preços sombra
    preco_sombra = tableau[-1, num_vars:num_vars + num_constraints]

    return solution, valor_otimo, preco_sombra

# Executando a função com entradas interativas
solucao, valor_otimo, preco_sombra = simplex_interativo()
print("Solução Ótima:", solucao)
print("Valor Ótimo:", valor_otimo)
print("Preços Sombra:", preco_sombra)