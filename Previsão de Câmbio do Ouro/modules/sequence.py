import numpy as np

def create_sequences(data, T):
    X = []
    y = []

    # Loop para criar sequências a partir de 'data' com janela de tamanho T
    # O loop vai até 'len(data) - T' para evitar acessar índices fora do limite
    for t in range(len(data) - T):
        # Adiciona a sequência de entrada (T valores consecutivos de data)
        X.append(data[t:t + T])
        # Adiciona o valor seguinte à sequência como a saída (Y)
        y.append(data[t + T])
    
    # Retorna X como um array NumPy de forma adequada, Y como array, e N como o número de sequências
    X = np.array(X).reshape(-1, T)  # Formato [amostras, timesteps, features]
    y = np.array(y).reshape(-1, 1)     # Formato [amostras, 1] para saída
    
    return X, y



    

