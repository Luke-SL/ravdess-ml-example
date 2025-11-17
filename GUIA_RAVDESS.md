# 🚀 Guia Rápido - Treinar com RAVDESS

## 📋 Checklist Antes de Começar

- [✅] Baixou o arquivo `Audio_Speech_Actors_01-24.zip` do Zenodo
- [ ] Extraiu o arquivo ZIP em uma pasta (ex: `RAVDESS/`)
- [ ] Tem Python instalado (3.7+)
- [ ] Tem as bibliotecas instaladas

---

## 🔧 Passo 1: Instalar Dependências

```bash
pip install librosa scikit-learn matplotlib seaborn numpy
```

---

## 📂 Passo 2: Extrair o Dataset

Depois que o download terminar:

1. **Descompacte o arquivo** `Audio_Speech_Actors_01-24.zip`
2. Você terá uma pasta com subpastas: `Actor_01`, `Actor_02`, ..., `Actor_24`

**Estrutura esperada:**
```
RAVDESS/
├── Actor_01/
│   ├── 03-01-01-01-01-01-01.wav
│   ├── 03-01-01-01-01-02-01.wav
│   └── ...
├── Actor_02/
│   └── ...
└── Actor_24/
    └── ...
```

---

## ⚙️ Passo 3: Configurar o Script

Abra o arquivo `train_ravdess.py` e ajuste:

```python
# Linha 22: Ajuste o caminho da sua pasta RAVDESS
RAVDESS_PATH = 'RAVDESS/'  # ← Coloque o caminho correto aqui!

# Linha 25: Escolha as emoções (comece com 4)
EMOTIONS_TO_USE = ['feliz', 'triste', 'raiva', 'neutro']

# Linha 28: Escolha o modelo
MODEL_TYPE = 'svm'  # 'svm' ou 'rf'
```

---

## 🏃 Passo 4: Executar o Treinamento

```bash
python train_ravdess.py
```

**O que vai acontecer:**
1. ✅ Carrega os arquivos do RAVDESS (segundos)
2. ⏳ Extrai features de todos os áudios (~10-15 minutos)
3. 💾 Salva as features em `ravdess_features.pkl`
4. 🤖 Treina o modelo (segundos)
5. 📊 Avalia e mostra os resultados
6. 💾 Salva o modelo treinado
7. 📈 Gera visualizações

**Tempo total:** ~10-15 minutos na primeira vez

**Próximas vezes:** ~1 minuto (usa features salvas!)

---

## 📊 Passo 5: Analisar os Resultados

O script irá mostrar:

```
🎯 ACURÁCIA GERAL: 78.5%

              precision    recall  f1-score   support

       feliz       0.82      0.85      0.83        20
      neutro       0.71      0.68      0.69        19
       raiva       0.84      0.79      0.81        19
      triste       0.77      0.79      0.78        19

    accuracy                           0.78        77
```

**Arquivos gerados:**
- ✅ `ravdess_features.pkl` - Features extraídas (não precisa reprocessar)
- ✅ `emotion_model_svm_ravdess.pkl` - Modelo treinado
- ✅ `ravdess_confusion_matrix.png` - Matriz de confusão
- ✅ `ravdess_accuracy_per_emotion.png` - Acurácia por emoção

---

## 🎯 Passo 6: Usar o Modelo em Novos Áudios

```bash
python predict_with_model.py
```

Ou use no seu próprio código:

```python
import pickle
from emotion_classifier import extract_features

# Carregar modelo
with open('emotion_model_svm_ravdess.pkl', 'rb') as f:
    classifier = pickle.load(f)

# Analisar novo áudio
features = extract_features('meu_audio.wav').reshape(1, -1)
emotion = classifier.predict(features)[0]
probabilities = classifier.predict_proba(features)[0]

print(f"Emoção: {emotion}")
print(f"Confiança: {probabilities.max():.1%}")
```

---

## 🎨 Experimentar Diferentes Configurações

### Treinar com Todas as 8 Emoções

```python
EMOTIONS_TO_USE = ['neutro', 'calmo', 'feliz', 'triste', 
                   'raiva', 'medo', 'nojo', 'surpresa']
```

### Comparar SVM vs Random Forest

```python
# Teste 1
MODEL_TYPE = 'svm'
# Execute: python train_ravdess.py

# Teste 2
MODEL_TYPE = 'rf'
# Execute: python train_ravdess.py
```

### Reprocessar Tudo (Forçar Nova Extração)

1. Delete o arquivo `ravdess_features.pkl`
2. Execute `python train_ravdess.py`

---

## 📈 Resultados Esperados

### Com 4 Emoções (feliz, triste, raiva, neutro):
- **Acurácia:** 70-85%
- **Melhores:** raiva (mais distinta)
- **Mais difíceis:** neutro vs calmo

### Com 8 Emoções:
- **Acurácia:** 55-70%
- **Mais desafiador:** mais classes para distinguir

---

## ❓ Troubleshooting

### Erro: "Pasta não encontrada"
```
❌ ERRO: Pasta não encontrada: RAVDESS/
```
**Solução:** Ajuste `RAVDESS_PATH` no `train_ravdess.py`

### Erro: "Nenhuma pasta de ator encontrada"
```
❌ ERRO: Nenhuma pasta de ator encontrada!
```
**Solução:** Verifique se extraiu o ZIP corretamente. Deve ter pastas `Actor_01`, `Actor_02`, etc.

### Erro: "No module named 'librosa'"
```
ModuleNotFoundError: No module named 'librosa'
```
**Solução:** 
```bash
pip install librosa
```

### Processo muito lento
- ⏳ Primeira execução é lenta (extrai features)
- ✅ Próximas execuções são rápidas (usa features salvas)
- 💡 Reduza o número de emoções para testar mais rápido

---

## 🎓 Para sua Apresentação

### Métricas Importantes:

1. **Acurácia Geral:** Porcentagem de acertos
2. **Precision:** Quando prediz X, quantas vezes está certo?
3. **Recall:** De todos os X verdadeiros, quantos foram detectados?
4. **F1-Score:** Média harmônica de precision e recall

### Pontos para Destacar:

✅ Dataset profissional (RAVDESS)
✅ 24 atores diferentes
✅ Áudios controlados em laboratório
✅ Múltiplas emoções
✅ Resultados comparáveis a trabalhos acadêmicos

---

## 📚 Referência do RAVDESS

**Citação:**
```
Livingstone SR, Russo FA (2018) 
The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): 
A dynamic, multimodal set of facial and vocal expressions in North American English. 
PLoS ONE 13(5): e0196391.
```

---

## ✅ Resumo dos Comandos

```bash
# 1. Instalar
pip install librosa scikit-learn matplotlib seaborn numpy

# 2. Treinar
python train_ravdess.py

# 3. Testar em novos áudios
python predict_with_model.py
```

---

## 🎉 Próximos Passos

Depois do treinamento bem-sucedido:

1. ✅ Analise a matriz de confusão
2. ✅ Teste com áudios próprios
3. ✅ Compare SVM vs Random Forest
4. ✅ Experimente com mais/menos emoções
5. ✅ Use nas visualizações da sua apresentação!

---

**Boa sorte! 🚀**
