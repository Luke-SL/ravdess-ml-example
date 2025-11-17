# 🎭 Classificador de Emoções de Áudio - Material para Apresentação

## 🚀 Como Usar

### Opção 1: Notebook Jupyter (Recomendado para Apresentação)

O notebook é ideal para apresentações interativas:

```bash
# 1. Instalar Jupyter
pip install jupyter

# 2. Abrir o notebook
jupyter notebook emotion_classifier_notebook.ipynb
```

O notebook contém:
- ✅ Explicações didáticas
- ✅ Código comentado
- ✅ Visualizações
- ✅ Exemplo completo passo a passo

### Opção 2: Script Python

Para execução direta:

```bash
# 1. Instalar dependências
pip install librosa scikit-learn matplotlib seaborn numpy

# 2. Executar
python emotion_classifier.py
```

---

## 📊 O Que o Algoritmo Faz?

### Pipeline de Classificação:

```
ÁUDIO (.wav) 
    ↓
PRÉ-PROCESSAMENTO
    ↓
EXTRAÇÃO DE FEATURES
  • MFCCs (textura do som)
  • Pitch (tom da voz)
  • Energia RMS (intensidade)
  • Zero Crossing Rate
  • Spectral Contrast
  • Chroma (informação tonal)
    ↓
NORMALIZAÇÃO
    ↓
MODELO ML (SVM/Random Forest)
    ↓
PREDIÇÃO: feliz/triste/raiva/neutro
```

---

## 🎯 Features Extraídas

O algoritmo extrai **62 features** de cada áudio:

| Feature | Quantidade | O que representa |
|---------|-----------|------------------|
| MFCCs | 26 | Envelope espectral (textura) |
| Chroma | 24 | Informação tonal |
| Spectral Contrast | 7 | Diferença picos/vales |
| Zero Crossing Rate | 2 | Taxa de mudança de sinal |
| RMS Energy | 2 | Intensidade do som |
| Pitch | 1 | Frequência fundamental |

---

## 🤖 Modelos Implementados

### 1. SVM (Support Vector Machine)
- ✅ Melhor para datasets pequenos
- ✅ Boa generalização
- ⚙️ Kernel RBF

### 2. Random Forest
- ✅ Mais interpretável
- ✅ Fornece importância de features
- ⚙️ 100 árvores

---

## 📈 Resultados Esperados

Com dados **reais** e bem balanceados:
- 🎯 **Acurácia típica:** 60-80%
- 📊 **Melhores emoções:** raiva e feliz (mais distintas)
- 🔄 **Confusão comum:** neutro vs. triste

Com os **dados sintéticos** da demo:
- 🎯 **Acurácia:** ~80% (SVM)

---

## 🗂️ Como Usar com Seus Dados

### Estrutura de Diretórios:

```
seu_projeto/
├── data/
│   ├── feliz/
│   │   ├── audio1.wav
│   │   ├── audio2.wav
│   │   └── ...
│   ├── triste/
│   │   ├── audio1.wav
│   │   └── ...
│   ├── raiva/
│   │   └── ...
│   └── neutro/
│       └── ...
├── emotion_classifier.py
└── emotion_classifier_notebook.ipynb
```

### Código para Carregar Dados:

```python
import os

# Carregar arquivos
audio_files = []
labels = []

for emotion in ['feliz', 'triste', 'raiva', 'neutro']:
    emotion_dir = f'data/{emotion}'
    for filename in os.listdir(emotion_dir):
        if filename.endswith('.wav'):
            audio_files.append(os.path.join(emotion_dir, filename))
            labels.append(emotion)

# Extrair features
X, y = prepare_dataset(audio_files, labels)

# Treinar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
classifier = EmotionClassifier(model_type='svm')
classifier.train(X_train, y_train)
```

---

## 📚 Datasets Recomendados

Para treinar com dados reais, use estes datasets públicos:

1. **RAVDESS** (Ryerson Audio-Visual Database)
   - 7.356 arquivos
   - 24 atores
   - 8 emoções
   - Download: https://zenodo.org/record/1188976

2. **TESS** (Toronto Emotional Speech Set)
   - 2.800 arquivos
   - 2 atrizes
   - 7 emoções
   - Download: https://tspace.library.utoronto.ca/handle/1807/24487

3. **CREMA-D**
   - 7.442 arquivos
   - 91 atores
   - 6 emoções
   - Download: https://github.com/CheyneyComputerScience/CREMA-D

---

## 🎨 Visualizações Incluídas

### 1. Matriz de Confusão
Mostra onde o modelo acerta e erra:
- Diagonal principal = acertos
- Fora da diagonal = confusões

### 2. Probabilidades de Emoção
Para cada predição, mostra:
- Confiança em cada classe
- Emoção vencedora destacada

### 3. Pipeline Diagram
Visualização completa do fluxo de processamento

---

## 🔧 Dependências

```
librosa>=0.10.0      # Processamento de áudio
scikit-learn>=1.3.0  # Machine learning
matplotlib>=3.7.0    # Visualização
seaborn>=0.12.0      # Gráficos estatísticos
numpy>=1.24.0        # Operações numéricas
```

Instalar todas:
```bash
pip install librosa scikit-learn matplotlib seaborn numpy
```

---

## 💡 Dicas para Apresentação

### Pontos Principais:

1. **Problema:** Como computadores podem identificar emoções humanas?

2. **Solução:** Machine learning + análise acústica

3. **Como funciona:**
   - Extraímos características sonoras (MFCCs, pitch, energia...)
   - Treinamos um modelo para reconhecer padrões
   - Modelo aprende que raiva = pitch alto + energia alta, etc.

4. **Aplicações:**
   - Call centers (detectar clientes insatisfeitos)
   - Saúde mental (monitorar estado emocional)
   - Games e entretenimento
   - Assistentes virtuais mais empáticos

### Estrutura Sugerida:

1. Introdução (2 min)
2. Como emoções afetam a voz (3 min)
3. Features extraídas (5 min)
4. Pipeline e modelo (5 min)
5. Resultados e visualizações (3 min)
6. Demo ao vivo (2 min) ← Use o notebook!
7. Conclusão e perguntas (5 min)

---

## 🚀 Melhorias Possíveis

Para expandir o projeto:

### Nível Intermediário:
- ✅ Adicionar mais emoções (surpresa, medo, nojo)
- ✅ Data augmentation (pitch shift, time stretch)
- ✅ Grid search para otimizar hiperparâmetros

### Nível Avançado:
- 🧠 Deep Learning com CNNs (processar espectrogramas)
- 🔄 RNNs/LSTMs (capturar dependências temporais)
- 🎯 Transfer learning (usar modelos pré-treinados)
- 📱 Deploy como API REST ou app mobile

---

## 🐛 Troubleshooting

### Erro: "No module named 'librosa'"
```bash
pip install librosa
```

### Erro ao carregar áudio
- Certifique-se que o arquivo é .wav ou .mp3
- Taxa de amostragem recomendada: 16kHz ou 22kHz
- Áudio mono (1 canal)

### Acurácia muito baixa
- Verifique se os dados estão balanceados
- Tente aumentar o número de amostras
- Experimente outros modelos (RF vs SVM)
- Normalize os dados

---

## 📞 Suporte

Para dúvidas sobre o código ou implementação:
- Consulte o notebook interativo
- Leia os comentários no código
- Experimente com os exemplos fornecidos

---

## ✅ Checklist para Apresentação

- [ ] Testei o código localmente
- [ ] Entendo o que cada feature representa
- [ ] Posso explicar o pipeline completo
- [ ] Preparei exemplos de áudio para demo
- [ ] Revisei os resultados e métricas
- [ ] Preparei respostas para perguntas comuns

---

## 🎓 Conceitos-Chave para Explicar

### MFCCs
"Coeficientes que capturam a forma do envelope espectral da voz, similar a como nosso ouvido processa som"

### SVM
"Algoritmo que encontra o melhor hiperplano para separar as classes no espaço de features"

### Spectral Contrast
"Mede a diferença entre picos e vales no espectro de frequências"

### Zero Crossing Rate
"Quantas vezes o sinal de áudio cruza o eixo zero - alto em sons sibilantes"

---

## 📄 Licença

Este código é fornecido como material educacional.
Sinta-se livre para usar, modificar e compartilhar.

---

**Boa sorte na sua apresentação! 🎉**
"# ravdess-ml-example" 
