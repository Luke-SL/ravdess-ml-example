"""
Script para Treinar Classificador de Emoções com RAVDESS
Dataset: Audio_Speech_Actors_01-24.zip

Instruções:
1. Extraia o arquivo ZIP em uma pasta (ex: RAVDESS/)
2. Ajuste o caminho RAVDESS_PATH abaixo
3. Execute este script
"""

import os
import numpy as np
import pickle
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Importar funções do seu classificador
from emotion_classifier import extract_features, EmotionClassifier

# ============================================
# CONFIGURAÇÕES
# ============================================

# 🔧 AJUSTE ESTE CAMINHO PARA SUA PASTA DO RAVDESS
RAVDESS_PATH = 'RAVDESS/'  # Pasta onde você extraiu o ZIP

# Emoções para treinar (comece com 4, depois experimente com todas)
EMOTIONS_TO_USE = ['feliz', 'triste', 'raiva', 'neutro']
# Para usar todas: ['neutro', 'calmo', 'feliz', 'triste', 'raiva', 'medo', 'nojo', 'surpresa']

# Modelo a usar
MODEL_TYPE = 'svm'  # 'svm' ou 'rf'

# ============================================
# FUNÇÕES PARA PROCESSAR RAVDESS
# ============================================

def parse_ravdess_filename(filename):
    """
    Decodifica o nome do arquivo RAVDESS
    Exemplo: 03-01-06-01-02-01-12.wav
    
    Formato: Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor
    """
    parts = filename.replace('.wav', '').split('-')
    
    emotion_map = {
        '01': 'neutro',
        '02': 'calmo', 
        '03': 'feliz',
        '04': 'triste',
        '05': 'raiva',
        '06': 'medo',
        '07': 'nojo',
        '08': 'surpresa'
    }
    
    intensity_map = {
        '01': 'normal',
        '02': 'forte'
    }
    
    statement_map = {
        '01': 'Kids are talking by the door',
        '02': 'Dogs are sitting by the door'
    }
    
    try:
        emotion_code = parts[2]
        intensity_code = parts[3]
        statement_code = parts[4]
        repetition = parts[5]
        actor_id = int(parts[6])
        
        return {
            'emotion': emotion_map.get(emotion_code, 'desconhecido'),
            'intensity': intensity_map.get(intensity_code, 'normal'),
            'statement': statement_map.get(statement_code, ''),
            'repetition': int(repetition),
            'actor': actor_id,
            'gender': 'masculino' if actor_id % 2 == 1 else 'feminino'
        }
    except:
        return None


def load_ravdess_dataset(ravdess_path, emotions_to_use):
    """
    Carrega arquivos do RAVDESS Speech dataset
    
    Retorna:
    - audio_files: lista com caminhos completos dos arquivos
    - labels: lista com as emoções
    - metadata: lista com informações extras (gênero, ator, etc)
    """
    audio_files = []
    labels = []
    metadata = []
    
    print("="*60)
    print("📂 CARREGANDO DATASET RAVDESS")
    print("="*60)
    print(f"Pasta: {ravdess_path}")
    print(f"Emoções selecionadas: {emotions_to_use}")
    print()
    
    # Verificar se a pasta existe
    if not os.path.exists(ravdess_path):
        print(f"❌ ERRO: Pasta não encontrada: {ravdess_path}")
        print("   Ajuste o caminho RAVDESS_PATH no início do script!")
        return [], [], []
    
    # Percorrer todas as subpastas (Actor_01, Actor_02, etc)
    actor_folders = sorted([d for d in os.listdir(ravdess_path) 
                           if os.path.isdir(os.path.join(ravdess_path, d)) 
                           and d.startswith('Actor_')])
    
    if not actor_folders:
        print(f"❌ ERRO: Nenhuma pasta de ator encontrada!")
        print("   Estrutura esperada: RAVDESS/Actor_01/, Actor_02/, etc.")
        return [], [], []
    
    print(f"✅ Encontradas {len(actor_folders)} pastas de atores")
    print()
    
    # Processar cada ator
    for actor_folder in actor_folders:
        actor_path = os.path.join(ravdess_path, actor_folder)
        
        for filename in os.listdir(actor_path):
            # Filtrar apenas arquivos de Speech (começam com 03-01)
            if filename.endswith('.wav') and filename.startswith('03-01'):
                filepath = os.path.join(actor_path, filename)
                info = parse_ravdess_filename(filename)
                
                if info and info['emotion'] in emotions_to_use:
                    audio_files.append(filepath)
                    labels.append(info['emotion'])
                    metadata.append(info)
    
    print(f"✅ Total de arquivos carregados: {len(audio_files)}")
    print()
    
    # Mostrar distribuição
    distribution = Counter(labels)
    print("📊 DISTRIBUIÇÃO DAS EMOÇÕES:")
    print("-" * 40)
    for emotion in sorted(emotions_to_use):
        count = distribution.get(emotion, 0)
        bar = '█' * (count // 5)
        print(f"   {emotion:10s}: {count:3d} arquivos {bar}")
    print()
    
    # Mostrar distribuição por gênero
    gender_dist = Counter([m['gender'] for m in metadata])
    print("👥 DISTRIBUIÇÃO POR GÊNERO:")
    print("-" * 40)
    for gender, count in gender_dist.items():
        print(f"   {gender}: {count} arquivos")
    print()
    
    return audio_files, labels, metadata


def extract_and_save_features(audio_files, labels, save_path='ravdess_features.pkl'):
    """
    Extrai features de todos os áudios e salva em arquivo
    (Para não precisar reprocessar toda vez)
    """
    print("="*60)
    print("🎵 EXTRAINDO FEATURES DOS ÁUDIOS")
    print("="*60)
    print("⏳ Isso pode demorar alguns minutos...")
    print(f"   Total de arquivos: {len(audio_files)}")
    print()
    
    X = []
    y = []
    failed = []
    
    for i, (audio_file, label) in enumerate(zip(audio_files, labels), 1):
        try:
            # Extrair features
            features = extract_features(audio_file)
            X.append(features)
            y.append(label)
            
            # Mostrar progresso a cada 50 arquivos
            if i % 50 == 0:
                print(f"   Processados: {i}/{len(audio_files)} ({i/len(audio_files)*100:.1f}%)")
        
        except Exception as e:
            print(f"   ⚠️  Erro no arquivo {os.path.basename(audio_file)}: {str(e)[:50]}")
            failed.append(audio_file)
    
    X = np.array(X)
    y = np.array(y)
    
    print()
    print(f"✅ Extração concluída!")
    print(f"   Sucessos: {len(X)}/{len(audio_files)}")
    if failed:
        print(f"   Falhas: {len(failed)}")
    print(f"   Shape das features: {X.shape}")
    print()
    
    # Salvar features em arquivo
    print(f"💾 Salvando features em: {save_path}")
    with open(save_path, 'wb') as f:
        pickle.dump({
            'X': X,
            'y': y,
            'feature_shape': X.shape
        }, f)
    print(f"✅ Features salvas!")
    print()
    
    return X, y


def load_saved_features(save_path='ravdess_features.pkl'):
    """
    Carrega features já extraídas anteriormente
    """
    print(f"📂 Carregando features salvas de: {save_path}")
    with open(save_path, 'rb') as f:
        data = pickle.load(f)
    print(f"✅ Features carregadas! Shape: {data['X'].shape}")
    print()
    return data['X'], data['y']


def train_and_evaluate(X, y, model_type='svm'):
    """
    Treina e avalia o modelo
    """
    print("="*60)
    print("🤖 TREINAMENTO DO MODELO")
    print("="*60)
    
    # Dividir em treino e teste (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y  # Manter proporção de classes
    )
    
    print(f"Dados de treino: {len(X_train)} amostras")
    print(f"Dados de teste: {len(X_test)} amostras")
    print()
    
    # Treinar
    print(f"Treinando modelo {model_type.upper()}...")
    classifier = EmotionClassifier(model_type=model_type)
    classifier.train(X_train, y_train)
    print()
    
    # Avaliar
    print("="*60)
    print("📊 AVALIAÇÃO NO CONJUNTO DE TESTE")
    print("="*60)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n🎯 ACURÁCIA GERAL: {accuracy:.2%}")
    print()
    print(classification_report(y_test, y_pred))
    
    return classifier, X_test, y_test, y_pred


def plot_results(y_test, y_pred, emotions):
    """
    Cria visualizações dos resultados
    """
    print("="*60)
    print("📈 GERANDO VISUALIZAÇÕES")
    print("="*60)
    
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred, labels=emotions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=emotions, yticklabels=emotions,
                cbar_kws={'label': 'Contagem'})
    plt.title('Matriz de Confusão - RAVDESS', fontsize=16, fontweight='bold')
    plt.ylabel('Emoção Real', fontsize=12)
    plt.xlabel('Emoção Predita', fontsize=12)
    plt.tight_layout()
    plt.savefig('ravdess_confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("✅ Salva: ravdess_confusion_matrix.png")
    
    # Acurácia por emoção
    accuracies = []
    for emotion in emotions:
        mask = y_test == emotion
        if mask.sum() > 0:
            acc = accuracy_score(y_test[mask], y_pred[mask])
            accuracies.append(acc)
        else:
            accuracies.append(0)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(emotions, accuracies, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Acurácia por Emoção - RAVDESS', fontsize=14, fontweight='bold')
    plt.ylabel('Acurácia', fontsize=12)
    plt.ylim([0, 1])
    plt.grid(axis='y', alpha=0.3)
    
    # Adicionar valores nas barras
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('ravdess_accuracy_per_emotion.png', dpi=150, bbox_inches='tight')
    print("✅ Salva: ravdess_accuracy_per_emotion.png")
    
    plt.show()
    print()


# ============================================
# SCRIPT PRINCIPAL
# ============================================

def main():
    """
    Fluxo principal de execução
    """
    print("\n" + "="*60)
    print("🎭 CLASSIFICADOR DE EMOÇÕES - RAVDESS DATASET")
    print("="*60)
    print()
    
    # Verificar se já existem features salvas
    features_file = 'ravdess_features.pkl'
    
    if os.path.exists(features_file):
        print("📂 Features já extraídas anteriormente encontradas!")
        print()
        resposta = input("Deseja usar as features salvas? (s/n): ").lower()
        
        if resposta == 's':
            X, y = load_saved_features(features_file)
        else:
            print("\n🔄 Reprocessando arquivos...")
            audio_files, labels, metadata = load_ravdess_dataset(RAVDESS_PATH, EMOTIONS_TO_USE)
            
            if not audio_files:
                print("❌ Nenhum arquivo encontrado. Verifique o caminho!")
                return
            
            X, y = extract_and_save_features(audio_files, labels, features_file)
    else:
        # Primeira vez - processar tudo
        audio_files, labels, metadata = load_ravdess_dataset(RAVDESS_PATH, EMOTIONS_TO_USE)
        
        if not audio_files:
            print("❌ Nenhum arquivo encontrado. Verifique o caminho!")
            return
        
        X, y = extract_and_save_features(audio_files, labels, features_file)
    
    # Treinar e avaliar
    classifier, X_test, y_test, y_pred = train_and_evaluate(X, y, MODEL_TYPE)
    
    # Visualizar resultados
    emotions = sorted(EMOTIONS_TO_USE)
    plot_results(y_test, y_pred, emotions)
    
    # Salvar modelo treinado
    model_file = f'emotion_model_{MODEL_TYPE}_ravdess.pkl'
    print(f"💾 Salvando modelo treinado em: {model_file}")
    with open(model_file, 'wb') as f:
        pickle.dump(classifier, f)
    print(f"✅ Modelo salvo!")
    print()
    
    print("\n" + "="*60)
    print("✅ PROCESSO COMPLETO!")
    print("="*60)
    print(f"""
Arquivos gerados:
  - {features_file} (features extraídas)
  - {model_file} (modelo treinado)
  - ravdess_confusion_matrix.png
  - ravdess_accuracy_per_emotion.png

💡 Para visualizar os gráficos:
  - Abra os arquivos .png com qualquer visualizador de imagens
  - Ou use: plt.show() no seu código

Próximos passos:
  1. Analise as visualizações (abra os arquivos .png)
  2. Teste com diferentes emoções (edite EMOTIONS_TO_USE)
  3. Compare SVM vs Random Forest (edite MODEL_TYPE)
  4. Use o modelo para classificar novos áudios!
     Execute: python predict_with_model.py
    """)
    
    print("="*60)
    print("🎉 Treinamento finalizado! O terminal está livre.")
    print("="*60)


if __name__ == "__main__":
    main()
