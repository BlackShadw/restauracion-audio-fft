# Restauración de Audio mediante Filtrado FFT

> **Proyecto DSP** - Implementación de filtrado espectral (Notch Filter) para eliminar interferencias en señales de audio usando la Transformada Rápida de Fourier (FFT).

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![DSP](https://img.shields.io/badge/DSP-Audio%20Processing-orange)](https://github.com/topics/dsp)

---

## 📋 Descripción

Este proyecto implementa la restauración de grabaciones de audio contaminadas mediante **filtrado en el dominio de la frecuencia**. Utiliza la **FFT (Fast Fourier Transform)** para transformar la señal temporal al dominio espectral, aplica **filtros muesca (notch filters)** para anular selectivamente frecuencias de interferencia conocidas (hum de red 50/60 Hz, tonos armónicos, acoplamientos de alta frecuencia), y reconstruye la señal limpia mediante **IFFT**.

### Resultados del Reporte Técnico
| Métrica | Entrada (Contaminada) | Salida (Filtrada) | Mejora |
|---------|----------------------|-------------------|--------|
| **SNR** | -14.42 dB | +19.82 dB | **+18.40 dB** |
| **Hum 60 Hz** | -6.02 dBFS | -68.40 dBFS | -62.38 dB |
| **Tono 2.5 kHz** | -7.95 dBFS | -71.12 dBFS | -63.17 dB |
| **Tiempo proceso** | 18.42 ms (DFT) | 0.0028 ms (FFT) | **6,500×** |

---

## 🚀 Instalación Rápida

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/restauracion-audio-fft.git
cd restauracion-audio-fft

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt
```

---

## 🎯 Uso

### 1. Demostración Sintética (Reporte Técnico)
Ejecuta el experimento completo del reporte con señal generada:

```bash
python fft_filter.py --demo
```

**Salida esperada:**
```
======================================================================
DEMOSTRACIÓN: Restauración de Audio con FFT - Señal Sintética
======================================================================

1. Generando señal base (voz sintética: 440 Hz + 880 Hz)...
2. Inyectando interferencias controladas...
   - Hum de red: 60 Hz (amplitud 0.5)
   - Tono agudo: 2500 Hz (amplitud 0.4)
   - Ruido blanco gaussiano: σ=0.05
3. Calculando FFT...
4. Aplicando filtro notch espectral (60 Hz y 2500 Hz, ancho 12 Hz)...
5. Evaluando rendimiento (SNR)...

======================================================================
RESULTADOS
======================================================================
SNR Inicial:    -14.42 dB
SNR Final:      +19.82 dB
Ganancia SNR:   +18.40 dB
======================================================================

Atenuación de componentes espurias:
  60 Hz: -62.38 dB
  2500 Hz: -63.17 dB
```

### 2. Procesar Archivo WAV Real
```bash
# Sintaxis básica
python fft_filter.py -i entrada.wav -o salida.wav -f 60,2500

# Con ancho de muesca personalizado
python fft_filter.py -i audio_ruidoso.wav -o audio_limpio.wav -f 50,100,150 -w 10
```

**Parámetros:**
| Flag | Descripción | Ejemplo |
|------|-------------|---------|
| `-i`, `--input` | Archivo WAV de entrada | `grabacion.wav` |
| `-o`, `--output` | Archivo WAV de salida | `grabacion_limpia.wav` |
| `-f`, `--frecuencias` | Frecuencias a eliminar (Hz, separadas por comas) | `60,2500` |
| `-w`, `--ancho` | Ancho de muesca en Hz (default: 12) | `10` |

---

## 🔬 Fundamento Teórico

### Transformada Rápida de Fourier (FFT)
El algoritmo **Cooley-Tukey** reduce la complejidad de la DFT de $O(N^2)$ a $O(N \log_2 N)$, permitiendo procesamiento en tiempo real.

Para audio a 44.1 kHz (10 seg = 441k muestras):
- **DFT directa**: ~1.94×10¹¹ operaciones
- **FFT**: ~8.2×10⁶ operaciones
- **Aceleración**: **23,600×**

### Filtrado en Dominio de Frecuencia
Aplicando el **teorema de la convolución**:
```
y[n] = x[n] ⊛ h[n]    ⇔    Y[k] = X[k] · H[k]
```

El **filtro notch** anula selectivamente:
$$H[k] = \begin{cases} 0 & |f_k - f_{notch}| \leq \Delta f/2 \\ 1 & \text{en otro caso} \end{cases}$$

**Ventajas:**
- ✅ Fase lineal perfecta (simetría hermitiana $H[k] = H^*[N-k]$)
- ✅ Atenuación extrema en frecuencias objetivo (>60 dB)
- ✅ Preservación de transitorios y armónicos de interés
- ✅ Computacionalmente eficiente (una sola FFT + IFFT)

---

## 📁 Estructura del Proyecto

```
restauracion-audio-fft/
├── fft_filter.py          # Script principal (CLI + funciones core)
├── generate_test_audio.py # Generador de archivos WAV de prueba
├── requirements.txt       # Dependencias Python
├── README.md             # Este archivo
├── LICENSE               # Licencia MIT
├── docs/
│   └── reporte_tecnico.md # Versión Markdown del reporte técnico
├── data/
│   ├── original.wav      # Señal contaminada (generada)
│   └── filtrado.wav      # Señal restaurada (resultado)
└── plots/                # Espectrogramas y gráficas (opcional)
```

---

## 🎵 Generar Archivos de Prueba

```bash
# Genera original.wav (contaminado) y filtrado.wav (procesado)
python generate_test_audio.py
```

Esto crea en `data/`:
- `original.wav` - Señal sintética con hum 60Hz + tono 2.5kHz + ruido
- `filtrado.wav` - Resultado después de filtrado notch

---

## 📊 Visualización (Opcional)

Para generar gráficas del espectro antes/después:

```python
# En Python interactivo o script propio
from fft_filter import *
import matplotlib.pyplot as plt

t, s = generar_senal_base()
x = contaminar_senal(s, t)
y, Y, X, H = filtrar_espectral(x, 44100, [60, 2500])

# Graficar espectros
freqs = fftfreq(len(x), 1/44100)
plt.figure(figsize=(12, 6))
plt.semilogx(freqs[:len(freqs)//2], np.abs(X[:len(X)//2]), label='Contaminada')
plt.semilogx(freqs[:len(freqs)//2], np.abs(Y[:len(Y)//2]), label='Filtrada')
plt.axvline(60, color='r', linestyle='--', alpha=0.5)
plt.axvline(2500, color='r', linestyle='--', alpha=0.5)
plt.xlabel('Frecuencia (Hz)'); plt.ylabel('Magnitud'); plt.legend()
plt.show()
```

---

## 🧪 Validación y Tests

```bash
# Ejecutar demo (valida SNR ~ +18.4 dB)
python fft_filter.py --demo

# Test con archivo generado
python generate_test_audio.py
python fft_filter.py -i data/original.wav -o data/test_output.wav -f 60,2500
```

---

## 📚 Referencias

1. **Cooley & Tukey (1965)** - *An algorithm for the machine calculation of complex Fourier series*
2. **Oppenheim & Schafer (1989)** - *Discrete-Time Signal Processing*
3. **Lyons (2010)** - *Understanding Digital Signal Processing*
4. **Proakis & Manolakis (2007)** - *Digital Signal Processing: Principles, Algorithms, and Applications*

---

## 🤝 Contribuir

1. Fork el proyecto
2. Crea tu rama: `git checkout -b feature/nueva-funcionalidad`
3. Commit: `git commit -m 'Añade nueva funcionalidad'`
4. Push: `git push origin feature/nueva-funcionalidad`
5. Abre un Pull Request

---

## 📄 Licencia

MIT License - Ver [LICENSE](LICENSE) para detalles.

---

## 👨‍💻 Autor

**Proyecto Fourier DSP** - Noviembre 2026  
*Disciplina: Procesamiento Digital de Señales (DSP)*  
*Conceptos: Series de Fourier, DFT, FFT Cooley-Tukey, Filtros Notch*"# restauracion-audio-fft" 
