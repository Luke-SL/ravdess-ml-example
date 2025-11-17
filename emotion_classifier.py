"""
Classificador de Emoções a partir de Áudio
Exemplo didático para apresentação

Emoções suportadas: feliz, triste, raiva, neutro
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. EXTRAÇÃO DE FEATURES
# ============================================

def extract_features(audio_path, duration=3):
    """
    Extrai features acústicas de um arquivo de áudio
    
    Parameters:
    - audio_path: caminho do arquivo de áudio
    - duration: duração em segundos para análise
    
    Returns:
    - array com features extraídas
    """
    # Carregar áudio
    y, sr = librosa.load(audio_path, duration=duration, sr=22050)
    
    # Feature 1: MFCCs (Mel-Frequency Cepstral Coefficients)
    # Capturam a forma do envelope espectral
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    
    # Feature 2: Chroma (informação tonal)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)
    
    # Feature 3: Spectral Contrast (diferença entre picos e vales)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)
    
    # Feature 4: Zero Crossing Rate (taxa de mudança de sinal)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)
    
    # Feature 5: RMS Energy (energia do sinal)
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)
    
    # Feature 6: Pitch (frequência fundamental)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_mean = np.mean(pitches[pitches > 0]) if len(pitches[pitches > 0]) > 0 else 0
    
    # Concatenar todas as features
    features = np.concatenate([
        mfccs_mean, mfccs_std,
        chroma_mean, chroma_std,
        contrast_mean,
        [zcr_mean, zcr_std, rms_mean, rms_std, pitch_mean]
    ])
    
    return features


def visualize_audio_features(audio_path, emotion):
    """
    Visualiza características do áudio para apresentação
    """
    y, sr = librosa.load(audio_path, duration=3)
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f'Análise de Áudio - Emoção: {emotion}', fontsize=16, fontweight='bold')
    
    # 1. Forma de onda
    axes[0, 0].plot(np.linspace(0, len(y)/sr, len(y)), y, color='blue', alpha=0.7)
    axes[0, 0].set_title('Forma de Onda')
    axes[0, 0].set_xlabel('Tempo (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Espectrograma
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=axes[0, 1], cmap='viridis')
    axes[0, 1].set_title('Espectrograma')
    fig.colorbar(img, ax=axes[0, 1], format='%+2.0f dB')
    
    # 3. MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    img2 = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=axes[1, 0], cmap='coolwarm')
    axes[1, 0].set_title('MFCCs (Coeficientes Mel)')
    axes[1, 0].set_ylabel('Coeficiente')
    fig.colorbar(img2, ax=axes[1, 0])
    
    # 4. Spectral Centroid (brilho do som)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    frames = range(len(spectral_centroids))
    t = librosa.frames_to_time(frames, sr=sr)
    axes[1, 1].plot(t, spectral_centroids, color='red', linewidth=2)
    axes[1, 1].set_title('Centroide Espectral (Brilho)')
    axes[1, 1].set_xlabel('Tempo (s)')
    axes[1, 1].set_ylabel('Hz')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 5. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    t = librosa.frames_to_time(range(len(zcr)), sr=sr)
    axes[2, 0].plot(t, zcr, color='green', linewidth=2)
    axes[2, 0].set_title('Taxa de Cruzamento por Zero')
    axes[2, 0].set_xlabel('Tempo (s)')
    axes[2, 0].set_ylabel('Taxa')
    axes[2, 0].grid(True, alpha=0.3)
    
    # 6. RMS Energy
    rms = librosa.feature.rms(y=y)[0]
    t = librosa.frames_to_time(range(len(rms)), sr=sr)
    axes[2, 1].plot(t, rms, color='purple', linewidth=2)
    axes[2, 1].set_title('Energia RMS')
    axes[2, 1].set_xlabel('Tempo (s)')
    axes[2, 1].set_ylabel('Amplitude')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ============================================
# 2. PREPARAÇÃO DOS DADOS
# ============================================

def prepare_dataset(audio_files, labels):
    """
    Prepara o dataset extraindo features de múltiplos áudios
    
    Parameters:
    - audio_files: lista de caminhos para arquivos de áudio
    - labels: lista de rótulos (emoções) correspondentes
    
    Returns:
    - X: matriz de features
    - y: vetor de labels
    """
    X = []
    y = []
    
    print("Extraindo features dos áudios...")
    for i, (audio_file, label) in enumerate(zip(audio_files, labels)):
        try:
            features = extract_features(audio_file)
            X.append(features)
            y.append(label)
            print(f"Processado {i+1}/{len(audio_files)}: {label}")
        except Exception as e:
            print(f"Erro ao processar {audio_file}: {e}")
    
    return np.array(X), np.array(y)


# ============================================
# 3. TREINAMENTO DO MODELO
# ============================================

class EmotionClassifier:
    """
    Classificador de Emoções
    """
    def __init__(self, model_type='svm'):
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Escolher modelo
        if model_type == 'svm':
            self.model = SVC(kernel='rbf', probability=True, random_state=42)
        elif model_type == 'rf':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError("Tipo de modelo não suportado")
    
    def train(self, X_train, y_train):
        """Treina o modelo"""
        # Normalizar features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Codificar labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        # Treinar
        print(f"\nTreinando modelo {self.model_type.upper()}...")
        self.model.fit(X_train_scaled, y_train_encoded)
        print("Treinamento concluído!")
    
    def predict(self, X_test):
        """Faz predições"""
        X_test_scaled = self.scaler.transform(X_test)
        y_pred_encoded = self.model.predict(X_test_scaled)
        return self.label_encoder.inverse_transform(y_pred_encoded)
    
    def predict_proba(self, X_test):
        """Retorna probabilidades de cada classe"""
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict_proba(X_test_scaled)
    
    def evaluate(self, X_test, y_test):
        """Avalia o modelo"""
        y_pred = self.predict(X_test)
        
        print("\n" + "="*50)
        print("RELATÓRIO DE CLASSIFICAÇÃO")
        print("="*50)
        print(classification_report(y_test, y_pred))
        
        return y_pred


# ============================================
# 4. VISUALIZAÇÕES
# ============================================

def plot_confusion_matrix(y_true, y_pred, labels):
    """Plota matriz de confusão"""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Contagem'})
    plt.title('Matriz de Confusão', fontsize=16, fontweight='bold')
    plt.ylabel('Emoção Real', fontsize=12)
    plt.xlabel('Emoção Predita', fontsize=12)
    plt.tight_layout()
    return plt.gcf()


def plot_emotion_probabilities(probabilities, emotions, predicted_emotion):
    """Plota probabilidades de cada emoção"""
    plt.figure(figsize=(10, 6))
    colors = ['#FF6B6B' if emotion == predicted_emotion else '#4ECDC4' 
              for emotion in emotions]
    
    bars = plt.bar(emotions, probabilities, color=colors, alpha=0.7, edgecolor='black')
    plt.title(f'Probabilidades de Emoção\nPredição: {predicted_emotion}', 
              fontsize=14, fontweight='bold')
    plt.ylabel('Probabilidade', fontsize=12)
    plt.xlabel('Emoção', fontsize=12)
    plt.ylim([0, 1])
    plt.grid(axis='y', alpha=0.3)
    
    # Adicionar valores nas barras
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob:.2%}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return plt.gcf()


def plot_feature_importance(classifier, feature_names, top_n=20):
    """Plota importância das features (para Random Forest)"""
    if classifier.model_type != 'rf':
        print("Importância de features disponível apenas para Random Forest")
        return
    
    importances = classifier.model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(12, 8))
    plt.title(f'Top {top_n} Features Mais Importantes', fontsize=14, fontweight='bold')
    plt.barh(range(top_n), importances[indices], color='teal', alpha=0.7)
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Importância', fontsize=12)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    return plt.gcf()


# ============================================
# 5. EXEMPLO DE USO
# ============================================

def demo_example():
    """
    Demonstração com dados sintéticos para apresentação
    """
    print("="*60)
    print("DEMONSTRAÇÃO: Classificador de Emoções de Áudio")
    print("="*60)
    
    # Simular dataset (normalmente você teria arquivos de áudio reais)
    print("\n[INFO] Em uma aplicação real, você teria:")
    print("- Dataset com arquivos .wav/.mp3 de diferentes emoções")
    print("- Exemplos: RAVDESS, TESS, CREMA-D, etc.")
    print("\n[INFO] Estrutura típica:")
    print("  data/")
    print("    ├── feliz/")
    print("    │   ├── audio1.wav")
    print("    │   └── audio2.wav")
    print("    ├── triste/")
    print("    ├── raiva/")
    print("    └── neutro/")
    
    # Exemplo de como seria o código real:
    print("\n" + "="*60)
    print("CÓDIGO DE EXEMPLO PARA USO REAL:")
    print("="*60)
    
    example_code = """
# Carregar seus dados
audio_files = [
    'data/feliz/audio1.wav',
    'data/feliz/audio2.wav',
    'data/triste/audio1.wav',
    'data/triste/audio2.wav',
    # ... mais arquivos
]

labels = ['feliz', 'feliz', 'triste', 'triste', ...]

# Extrair features
X, y = prepare_dataset(audio_files, labels)

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Treinar modelo
classifier = EmotionClassifier(model_type='svm')  # ou 'rf'
classifier.train(X_train, y_train)

# Avaliar
y_pred = classifier.evaluate(X_test, y_test)

# Visualizar resultados
plot_confusion_matrix(y_test, y_pred, classifier.label_encoder.classes_)
plt.show()

# Fazer predição em novo áudio
new_audio = 'novo_audio.wav'
features = extract_features(new_audio).reshape(1, -1)
emotion = classifier.predict(features)[0]
probabilities = classifier.predict_proba(features)[0]

print(f"Emoção detectada: {emotion}")
plot_emotion_probabilities(probabilities, 
                           classifier.label_encoder.classes_, 
                           emotion)
plt.show()
"""
    print(example_code)
    
    # Criar visualização de exemplo
    print("\n[INFO] Gerando visualização de exemplo do pipeline...")
    create_pipeline_diagram()


def create_pipeline_diagram():
    """Cria diagrama do pipeline de classificação"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Definir etapas
    steps = [
        "1. ÁUDIO\nArquivo .wav",
        "2. PRÉ-PROCESSAMENTO\n- Normalização\n- Remoção de ruído",
        "3. EXTRAÇÃO DE FEATURES\n- MFCCs\n- Pitch\n- Energia\n- ZCR\netc.",
        "4. NORMALIZAÇÃO\nStandardScaler",
        "5. MODELO ML\nSVM / Random Forest",
        "6. PREDIÇÃO\nEmoção + Probabilidades"
    ]
    
    colors = ['#FFE5E5', '#E5F3FF', '#E5FFE5', '#FFF5E5', '#F0E5FF', '#FFE5F5']
    
    # Desenhar caixas
    y_pos = 0.8
    for i, (step, color) in enumerate(zip(steps, colors)):
        x_pos = 0.1 + (i % 3) * 0.3
        if i == 3:
            y_pos = 0.3
        
        # Caixa
        rect = plt.Rectangle((x_pos, y_pos), 0.25, 0.15, 
                            facecolor=color, edgecolor='black', 
                            linewidth=2, alpha=0.8)
        ax.add_patch(rect)
        
        # Texto
        ax.text(x_pos + 0.125, y_pos + 0.075, step, 
               ha='center', va='center', fontsize=10, 
               fontweight='bold', wrap=True)
        
        # Setas
        if i < len(steps) - 1:
            if i == 2:  # Seta para baixo
                ax.arrow(x_pos + 0.125, y_pos, 0, -0.2, 
                        head_width=0.03, head_length=0.02, 
                        fc='black', ec='black', linewidth=2)
            else:  # Seta para direita
                ax.arrow(x_pos + 0.25, y_pos + 0.075, 0.04, 0, 
                        head_width=0.02, head_length=0.02, 
                        fc='black', ec='black', linewidth=2)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.title('Pipeline de Classificação de Emoções', 
             fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    return fig


# ============================================
# EXEMPLO COM DADOS SINTÉTICOS
# ============================================

def generate_synthetic_example():
    """
    Gera exemplo sintético para demonstração
    """
    print("\n[DEMO] Gerando exemplo com dados sintéticos...")
    
    # Simular features extraídas
    np.random.seed(42)
    n_samples = 100
    n_features = 50
    
    # Criar dados sintéticos que simulam padrões reais
    X_feliz = np.random.randn(n_samples//4, n_features) + 0.5
    X_triste = np.random.randn(n_samples//4, n_features) - 0.5
    X_raiva = np.random.randn(n_samples//4, n_features) + 1.0
    X_neutro = np.random.randn(n_samples//4, n_features)
    
    X = np.vstack([X_feliz, X_triste, X_raiva, X_neutro])
    y = ['feliz'] * (n_samples//4) + ['triste'] * (n_samples//4) + \
        ['raiva'] * (n_samples//4) + ['neutro'] * (n_samples//4)
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Treinar modelos
    print("\n[DEMO] Treinando SVM...")
    clf_svm = EmotionClassifier(model_type='svm')
    clf_svm.train(X_train, y_train)
    
    print("\n[DEMO] Treinando Random Forest...")
    clf_rf = EmotionClassifier(model_type='rf')
    clf_rf.train(X_train, y_train)
    
    # Avaliar SVM
    print("\n" + "="*60)
    print("RESULTADOS - SVM")
    print("="*60)
    y_pred_svm = clf_svm.evaluate(X_test, y_test)
    
    # Avaliar Random Forest
    print("\n" + "="*60)
    print("RESULTADOS - RANDOM FOREST")
    print("="*60)
    y_pred_rf = clf_rf.evaluate(X_test, y_test)
    
    # Visualizações
    print("\n[DEMO] Gerando visualizações...")
    
    # Matriz de confusão
    plot_confusion_matrix(y_test, y_pred_rf, clf_rf.label_encoder.classes_)
    plt.savefig('/mnt/user-data/outputs/confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("✓ Matriz de confusão salva")
    
    # Probabilidades de exemplo
    sample_idx = 0
    sample_probs = clf_rf.predict_proba(X_test[sample_idx:sample_idx+1])[0]
    sample_emotion = y_pred_rf[sample_idx]
    plot_emotion_probabilities(sample_probs, clf_rf.label_encoder.classes_, sample_emotion)
    plt.savefig('/mnt/user-data/outputs/emotion_probabilities.png', dpi=150, bbox_inches='tight')
    print("✓ Gráfico de probabilidades salvo")
    
    # Pipeline
    create_pipeline_diagram()
    plt.savefig('/mnt/user-data/outputs/pipeline_diagram.png', dpi=150, bbox_inches='tight')
    print("✓ Diagrama do pipeline salvo")
    
    print("\n[CONCLUÍDO] Arquivos gerados em /mnt/user-data/outputs/")


if __name__ == "__main__":
    # Executar demonstração
    demo_example()
    
    print("\n" + "="*60)
    print("Gerando exemplo com dados sintéticos...")
    print("="*60)
    generate_synthetic_example()
    
    print("\n" + "="*60)
    print("✓ DEMONSTRAÇÃO COMPLETA!")
    print("="*60)
    print("\nArquivos gerados:")
    print("  - confusion_matrix.png")
    print("  - emotion_probabilities.png")
    print("  - pipeline_diagram.png")
    print("\nCódigo pronto para apresentação!")
