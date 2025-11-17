"""
Script para Usar o Modelo Treinado em Novos Áudios

Use este script DEPOIS de treinar com train_ravdess.py
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from emotion_classifier import extract_features

def load_trained_model(model_path='emotion_model_svm_ravdess.pkl'):
    """
    Carrega o modelo treinado
    """
    print(f"📂 Carregando modelo de: {model_path}")
    with open(model_path, 'rb') as f:
        classifier = pickle.load(f)
    print("✅ Modelo carregado!")
    return classifier


def predict_emotion(classifier, audio_path, show_plot=True):
    """
    Prediz a emoção de um áudio
    """
    print(f"\n🎵 Analisando: {audio_path}")
    
    # Extrair features
    print("   Extraindo features...")
    features = extract_features(audio_path).reshape(1, -1)
    
    # Predição
    emotion = classifier.predict(features)[0]
    probabilities = classifier.predict_proba(features)[0]
    
    # Obter todas as emoções
    emotions = classifier.model.classes_
    
    print(f"\n🎯 EMOÇÃO DETECTADA: {emotion.upper()}")
    print(f"   Confiança: {probabilities.max():.1%}")
    print()
    print("📊 Probabilidades de todas as emoções:")
    for emo, prob in sorted(zip(emotions, probabilities), key=lambda x: x[1], reverse=True):
        bar = '█' * int(prob * 50)
        print(f"   {str(emo):10s}: {prob:6.1%} {bar}")
    
    # Visualizar
    if show_plot:
        plt.figure(figsize=(10, 6))
        colors = ['#FF6B6B' if e == emotion else '#4ECDC4' for e in emotions]
        bars = plt.bar(emotions, probabilities, color=colors, alpha=0.8, edgecolor='black')
        
        plt.title(f'Predição de Emoção\nArquivo: {audio_path}\nResultado: {emotion.upper()}', 
                  fontsize=14, fontweight='bold')
        plt.ylabel('Probabilidade', fontsize=12)
        plt.ylim([0, 1])
        plt.grid(axis='y', alpha=0.3)
        
        # Adicionar valores
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if show_plot:
            # Salvar ao invés de mostrar (para não travar o terminal)
            plot_filename = f'prediction_{emotion}.png'
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            print(f"\n💾 Gráfico salvo em: {plot_filename}")
            print("   (Abra o arquivo .png para visualizar)")
            plt.close()
        else:
            plt.close()
    
    return emotion, probabilities


def batch_predict(classifier, audio_files):
    """
    Prediz emoções para múltiplos áudios
    """
    print(f"\n📊 Analisando {len(audio_files)} arquivos...")
    print()
    
    results = []
    for i, audio_path in enumerate(audio_files, 1):
        try:
            features = extract_features(audio_path).reshape(1, -1)
            emotion = classifier.predict(features)[0]
            confidence = classifier.predict_proba(features)[0].max()
            
            results.append({
                'file': audio_path,
                'emotion': emotion,
                'confidence': confidence
            })
            
            print(f"{i}. {audio_path}")
            print(f"   → {emotion} ({confidence:.1%})")
            print()
        
        except Exception as e:
            print(f"{i}. {audio_path}")
            print(f"   ❌ Erro: {str(e)[:50]}")
            print()
    
    return results


# ============================================
# EXEMPLOS DE USO
# ============================================

if __name__ == "__main__":
    print("="*60)
    print("🎭 CLASSIFICADOR DE EMOÇÕES - PREDIÇÃO")
    print("="*60)
    print()
    
    # Carregar modelo
    classifier = load_trained_model('emotion_model_svm_ravdess.pkl')
    print()
    
    print("="*60)
    print("EXEMPLOS DE USO:")
    print("="*60)
    
    # EXEMPLO 1: Predizer um único áudio
    print("\n1️⃣ EXEMPLO: Predição em arquivo único")
    print("-" * 60)
    print("""
# Substitua pelo caminho do seu áudio
audio_file = 'meu_audio.wav'
emotion, probs = predict_emotion(classifier, audio_file)
    """)

    audio_file = 'file_path.wav'  # Substitua pelo caminho do seu áudio
    emotion, probs = predict_emotion(classifier, audio_file)
    
    # EXEMPLO 2: Predição em lote
    print("\n2️⃣ EXEMPLO: Predição em vários arquivos")
    print("-" * 60)
    print("""
# Lista de arquivos para analisar
audio_files = [
    'audio1.wav',
    'audio2.wav',
    'audio3.wav'
]
results = batch_predict(classifier, audio_files)
    """)
    
    # EXEMPLO 3: Testar com arquivos do RAVDESS
    print("\n3️⃣ EXEMPLO: Testar com arquivo do RAVDESS")
    print("-" * 60)
    print("""
# Arquivo do RAVDESS para teste
# Exemplo: 03-01-05-02-01-01-01.wav (raiva, intensidade forte)
ravdess_file = 'RAVDESS/Actor_01/03-01-05-02-01-01-01.wav'
emotion, probs = predict_emotion(classifier, ravdess_file)
    """)
    
    print("\n" + "="*60)
    print("💡 DICA: Descomente um dos exemplos acima e execute!")
    print("="*60)
    print()
    
    # Você pode descomentar as linhas abaixo para testar:
    
    # TESTE COM ARQUIVO DO RAVDESS (ajuste o caminho)
    # test_file = 'RAVDESS/Actor_01/03-01-05-02-01-01-01.wav'
    # if os.path.exists(test_file):
    #     emotion, probs = predict_emotion(classifier, test_file)
