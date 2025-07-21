# perceptron

O *Perceptron* é uma rede neural artificial simplificada, proposta por Frank Rosenblatt, composta por apenas um neurônio, com apenas uma saída, que de acordo com o tipo de problema abordado pode possuir n entradas.

![](perceptron.png)

Representação matemática do modelo
O *Perceptron* implementa o modelo *y* = *g(w·x + b)*, onde:

*w* são os pesos sinápticos;
*x* são as entradas;
*b* é o limiar de ativação (bias);
*g* é a função de ativação.

Função de Ativação
Implementamos a função *degrau unitário*, caso particular da *sigmoide* quando β tende ao infinito, que retorna 1 se a entrada for ≥ 0 e -1, por outro lado.

Treinamento
O algoritmo segue a regra de atualização: *W_atual* = *W_anterior + η(d - y)X*. O critério de parada é quando não há mais erros de classificação ou quando atinge o número máximo de épocas.

Operação
Aplicamos ao modelo um problema de classificação com classes linearmente separáveis. Após o treinamento, o *Perceptron* pode ser usado para fazer predições em novos dados.

Saída Esperada
Após a execução do código, obtemos os pesos finais do Perceptron após o treinamento, as predições para cada entrada do conjunto de treinamento, um gráfico mostrando a fronteira de decisão que separa as classes e um gráfico mostrando a diminuição dos erros ao longo das épocas de treinamento.
