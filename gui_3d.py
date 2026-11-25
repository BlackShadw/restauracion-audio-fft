#!/usr/bin/env python3
"""
Interfaz Gráfica 3D para Restauración de Audio con FFT
=======================================================
Visualizaciones 3D interactivas: espectrograma, espectros, waterfall plots.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import sys
import numpy as np

# Matplotlib para gráficas 3D
import matplotlib
matplotlib.use('TkAgg')  # Backend para Tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fft_filter import (
    demo_sintetica,
    procesar_archivo_wav,
    generar_senal_base,
    contaminar_senal,
    filtrar_espectral,
    calcular_snr
)
from scipy.fft import fft, fftfreq, fftshift
from scipy.io import wavfile
from scipy.signal import spectrogram


class AudioRestorationGUI3D:
    def __init__(self, root):
        self.root = root
        self.root.title("🎵 Restauración de Audio 3D - Filtro Notch FFT")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        
        # Variables
        self.input_file = tk.StringVar()
        self.output_file = tk.StringVar()
        self.frequencies = tk.StringVar(value="60, 2500")
        self.notch_width = tk.DoubleVar(value=12.0)
        self.is_processing = False
        
        # Datos de audio para gráficas
        self.audio_data = None
        self.fs = None
        self.filtered_data = None
        self.freqs_notch = [60, 2500]
        
        self.setup_ui()
        
    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Layout principal: panel izquierdo (controles) + panel derecho (gráficas 3D)
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # === PANEL IZQUIERDO: CONTROLES ===
        left_frame = ttk.Frame(main_paned, width=350)
        main_paned.add(left_frame, weight=1)
        
        # Notebook para controles
        control_notebook = ttk.Notebook(left_frame)
        control_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Pestaña: Archivo
        tab_file = ttk.Frame(control_notebook, padding=10)
        control_notebook.add(tab_file, text="📁 Archivo")
        self.setup_file_controls(tab_file)
        
        # Pestaña: Parámetros
        tab_params = ttk.Frame(control_notebook, padding=10)
        control_notebook.add(tab_params, text="⚙️ Parámetros")
        self.setup_param_controls(tab_params)
        
        # Pestaña: Visualización 3D
        tab_viz = ttk.Frame(control_notebook, padding=10)
        control_notebook.add(tab_viz, text="📊 Gráficas 3D")
        self.setup_viz_controls(tab_viz)
        
        # Pestaña: Demo/Info
        tab_demo = ttk.Frame(control_notebook, padding=10)
        control_notebook.add(tab_demo, text="🧪 Demo & Info")
        self.setup_demo_controls(tab_demo)
        
        # === PANEL DERECHO: GRÁFICAS 3D ===
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=3)
        
        self.setup_3d_plots(right_frame)
        
        # Barra de estado
        self.status_var = tk.StringVar(value="Listo - Cargue un archivo o ejecute la demo")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=(0, 5))
        
    def setup_file_controls(self, parent):
        ttk.Label(parent, text="Procesar Archivo WAV", font=('Segoe UI', 12, 'bold')).pack(pady=(0, 15), anchor=tk.W)
        
        # Entrada
        frame_in = ttk.LabelFrame(parent, text="Entrada (contaminado)", padding=8)
        frame_in.pack(fill=tk.X, pady=5)
        ttk.Entry(frame_in, textvariable=self.input_file).pack(fill=tk.X, pady=2)
        ttk.Button(frame_in, text="📂 Examinar...", command=self.browse_input).pack(fill=tk.X, pady=2)
        
        # Salida
        frame_out = ttk.LabelFrame(parent, text="Salida (restaurado)", padding=8)
        frame_out.pack(fill=tk.X, pady=5)
        ttk.Entry(frame_out, textvariable=self.output_file).pack(fill=tk.X, pady=2)
        ttk.Button(frame_out, text="💾 Guardar como...", command=self.browse_output).pack(fill=tk.X, pady=2)
        
        # Botones acción
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=15)
        
        self.process_btn = ttk.Button(btn_frame, text="🚀 PROCESAR Y VISUALIZAR 3D", 
                                       command=self.process_and_plot, style='Accent.TButton')
        self.process_btn.pack(fill=tk.X, pady=3, ipady=8)
        
        ttk.Button(btn_frame, text="📂 Cargar y Ver (sin procesar)", command=self.load_and_plot).pack(fill=tk.X, pady=3, ipady=5)
        ttk.Button(btn_frame, text="🧹 Limpiar", command=self.clear_all).pack(fill=tk.X, pady=3, ipady=5)
        
        # Log
        log_frame = ttk.LabelFrame(parent, text="Log", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, font=('Consolas', 8))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.config(state=tk.DISABLED)
        
    def setup_param_controls(self, parent):
        ttk.Label(parent, text="Parámetros del Filtro Notch", font=('Segoe UI', 12, 'bold')).pack(pady=(0, 15), anchor=tk.W)
        
        # Frecuencias
        frame_freq = ttk.LabelFrame(parent, text="Frecuencias a Eliminar (Hz)", padding=8)
        frame_freq.pack(fill=tk.X, pady=5)
        
        ttk.Entry(frame_freq, textvariable=self.frequencies).pack(fill=tk.X, pady=2)
        
        # Presets
        preset_frame = ttk.Frame(frame_freq)
        preset_frame.pack(fill=tk.X, pady=5)
        presets = [
            ("🔌 50Hz EU", "50, 100, 150, 200"),
            ("🔌 60Hz US", "60, 120, 180, 240"),
            ("🎵 Demo", "60, 2500"),
            ("🎹 Piano A4", "440, 880"),
        ]
        for label, val in presets:
            ttk.Button(preset_frame, text=label, width=12,
                       command=lambda v=val: self.frequencies.set(v)).pack(side=tk.LEFT, padx=2, pady=2)
        
        # Ancho muesca
        frame_width = ttk.LabelFrame(parent, text="Ancho de Muesca (Hz)", padding=8)
        frame_width.pack(fill=tk.X, pady=5)
        
        self.width_label_var = tk.StringVar(value="12 Hz")
        scale = ttk.Scale(frame_width, from_=1, to=100, variable=self.notch_width, orient=tk.HORIZONTAL)
        scale.pack(fill=tk.X, pady=2)
        ttk.Label(frame_width, textvariable=self.width_label_var).pack()
        self.notch_width.trace_add('write', lambda *a: self.width_label_var.set(f"{self.notch_width.get():.0f} Hz"))
        
        # Opciones avanzadas
        frame_adv = ttk.LabelFrame(parent, text="Opciones de Visualización", padding=8)
        frame_adv.pack(fill=tk.X, pady=5)
        
        self.show_spectrogram = tk.BooleanVar(value=True)
        self.show_spectrum3d = tk.BooleanVar(value=True)
        self.show_waterfall = tk.BooleanVar(value=True)
        self.show_comparison = tk.BooleanVar(value=True)
        self.log_scale = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(frame_adv, text="Espectrograma 3D (Tiempo-Frecuencia)", variable=self.show_spectrogram).pack(anchor=tk.W)
        ttk.Checkbutton(frame_adv, text="Espectro 3D Antes/Después", variable=self.show_spectrum3d).pack(anchor=tk.W)
        ttk.Checkbutton(frame_adv, text="Waterfall Plot", variable=self.show_waterfall).pack(anchor=tk.W)
        ttk.Checkbutton(frame_adv, text="Comparación Lado a Lado", variable=self.show_comparison).pack(anchor=tk.W)
        ttk.Checkbutton(frame_adv, text="Escala Logarítmica (dB)", variable=self.log_scale).pack(anchor=tk.W)
        
    def setup_viz_controls(self, parent):
        ttk.Label(parent, text="Controles de Visualización 3D", font=('Segoe UI', 12, 'bold')).pack(pady=(0, 15), anchor=tk.W)
        
        # Vista actual
        frame_view = ttk.LabelFrame(parent, text="Vista 3D Activa", padding=8)
        frame_view.pack(fill=tk.X, pady=5)
        
        self.current_view = tk.StringVar(value="Espectrograma 3D")
        views = ["Espectrograma 3D", "Espectro 3D Comparativo", "Waterfall", "Superficie Espectral", "Fase 3D"]
        view_combo = ttk.Combobox(frame_view, textvariable=self.current_view, values=views, state="readonly")
        view_combo.pack(fill=tk.X, pady=2)
        view_combo.bind('<<ComboboxSelected>>', lambda e: self.update_3d_view())
        
        # Controles cámara
        frame_cam = ttk.LabelFrame(parent, text="Control de Cámara", padding=8)
        frame_cam.pack(fill=tk.X, pady=5)
        
        ttk.Label(frame_cam, text="Elevación:").pack(anchor=tk.W)
        self.elev_var = tk.DoubleVar(value=30)
        ttk.Scale(frame_cam, from_=-90, to=90, variable=self.elev_var, orient=tk.HORIZONTAL, 
                  command=lambda v: self.update_camera()).pack(fill=tk.X)
        
        ttk.Label(frame_cam, text="Azimut:").pack(anchor=tk.W, pady=(5,0))
        self.azim_var = tk.DoubleVar(value=45)
        ttk.Scale(frame_cam, from_=-180, to=180, variable=self.azim_var, orient=tk.HORIZONTAL,
                  command=lambda v: self.update_camera()).pack(fill=tk.X)
        
        ttk.Label(frame_cam, text="Distancia:").pack(anchor=tk.W, pady=(5,0))
        self.dist_var = tk.DoubleVar(value=10)
        ttk.Scale(frame_cam, from_=3, to=20, variable=self.dist_var, orient=tk.HORIZONTAL,
                  command=lambda v: self.update_camera()).pack(fill=tk.X)
        
        ttk.Button(frame_cam, text="🔄 Reset Vista", command=self.reset_camera).pack(fill=tk.X, pady=5)
        
        # Animación
        frame_anim = ttk.LabelFrame(parent, text="Animación", padding=8)
        frame_anim.pack(fill=tk.X, pady=5)
        
        self.anim_running = False
        self.anim_btn = ttk.Button(frame_anim, text="▶ Rotar Automático", command=self.toggle_animation)
        self.anim_btn.pack(fill=tk.X, pady=2)
        
        ttk.Label(frame_anim, text="Velocidad:").pack(anchor=tk.W, pady=(5,0))
        self.anim_speed = tk.DoubleVar(value=1.0)
        ttk.Scale(frame_anim, from_=0.1, to=5.0, variable=self.anim_speed, orient=tk.HORIZONTAL).pack(fill=tk.X)
        
        # Exportar
        frame_export = ttk.LabelFrame(parent, text="Exportar", padding=8)
        frame_export.pack(fill=tk.X, pady=5)
        
        ttk.Button(frame_export, text="💾 Guardar Imagen 3D", command=self.save_3d_image).pack(fill=tk.X, pady=2)
        ttk.Button(frame_export, text="📹 Grabar Rotación (GIF)", command=self.record_rotation).pack(fill=tk.X, pady=2)
        
    def setup_demo_controls(self, parent):
        ttk.Label(parent, text="Demostración y Información", font=('Segoe UI', 12, 'bold')).pack(pady=(0, 15), anchor=tk.W)
        
        ttk.Button(parent, text="▶ EJECUTAR DEMO SINTÉTICA 3D", command=self.run_demo_3d, 
                   style='Accent.TButton').pack(fill=tk.X, pady=5, ipady=10)
        
        ttk.Button(parent, text="🎼 GENERAR ARCHIVOS DE PRUEBA", command=self.generate_test_files).pack(fill=tk.X, pady=5, ipady=8)
        
        # Info
        info_frame = ttk.LabelFrame(parent, text="Resultados Esperados (Demo)", padding=8)
        info_frame.pack(fill=tk.X, pady=10)
        
        info_text = """SNR Inicial:    ~ -14 dB
SNR Final:      ~ +20 dB
Mejora SNR:     ~ +34 dB

Atenuación 60Hz:    > 60 dB
Atenuación 2.5kHz:  > 60 dB

Tiempo FFT:     < 3 ms (132k muestras)"""
        ttk.Label(info_frame, text=info_text, font=('Consolas', 9), justify=tk.LEFT).pack(anchor=tk.W)
        
        # Ayuda rápida
        help_frame = ttk.LabelFrame(parent, text="Atajos de Teclado (en gráfica 3D)", padding=8)
        help_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        help_text = """Mouse: Rotar vista 3D
Scroll: Zoom
Shift+Mouse: Pan
R: Reset vista
G: Toggle grid
L: Toggle log scale
S: Guardar imagen
Espacio: Play/Pause animación"""
        ttk.Label(help_frame, text=help_text, font=('Consolas', 8), justify=tk.LEFT).pack(anchor=tk.W)
        
    def setup_3d_plots(self, parent):
        # Notebook para múltiples vistas 3D
        self.plot_notebook = ttk.Notebook(parent)
        self.plot_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Crear figuras 3D para cada pestaña
        self.figures = {}
        self.canvases = {}
        self.axes_3d = {}
        
        plot_tabs = [
            ("📈 Espectrograma 3D", "spectrogram"),
            ("📊 Espectro 3D", "spectrum3d"),
            ("🌊 Waterfall", "waterfall"),
            ("🔄 Comparación", "comparison"),
        ]
        
        for title, key in plot_tabs:
            frame = ttk.Frame(self.plot_notebook)
            self.plot_notebook.add(frame, text=title)
            
            fig = Figure(figsize=(8, 6), dpi=100)
            ax = fig.add_subplot(111, projection='3d')
            
            canvas = FigureCanvasTkAgg(fig, frame)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Toolbar
            toolbar = NavigationToolbar2Tk(canvas, frame)
            toolbar.update()
            
            self.figures[key] = fig
            self.canvases[key] = canvas
            self.axes_3d[key] = ax
            
            # Inicializar vacío
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'{title} - Cargue audio para ver')
            
        # Bind eventos de teclado en canvas
        for canvas in self.canvases.values():
            canvas.mpl_connect('key_press_event', self.on_key_press)
            
    def on_key_press(self, event):
        """Manejar atajos de teclado en gráficas 3D"""
        key = event.key
        if key == 'r':
            self.reset_camera()
        elif key == 'g':
            for ax in self.axes_3d.values():
                ax.grid(not ax.zaxis._gridOn)
            self.redraw_all()
        elif key == 'l':
            self.log_scale.set(not self.log_scale.get())
            self.update_3d_view()
        elif key == ' ':
            self.toggle_animation()
        elif key == 's':
            self.save_3d_image()
            
    def browse_input(self):
        filename = filedialog.askopenfilename(
            title="Seleccionar archivo de audio",
            filetypes=[("WAV", "*.wav"), ("Todos", "*.*")]
        )
        if filename:
            self.input_file.set(filename)
            base, ext = os.path.splitext(filename)
            self.output_file.set(f"{base}_restaurado{ext}")
            self.log(f"📂 Cargado: {os.path.basename(filename)}")
            
    def browse_output(self):
        filename = filedialog.asksaveasfilename(
            title="Guardar como",
            defaultextension=".wav",
            filetypes=[("WAV", "*.wav")]
        )
        if filename:
            self.output_file.set(filename)
            
    def log(self, msg):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.root.update_idletasks()
        
    def clear_all(self):
        self.input_file.set("")
        self.output_file.set("")
        self.frequencies.set("60, 2500")
        self.notch_width.set(12.0)
        self.audio_data = None
        self.filtered_data = None
        self.fs = None
        self.clear_plots()
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.status_var.set("Listo")
        
    def clear_plots(self):
        for ax in self.axes_3d.values():
            ax.clear()
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        for canvas in self.canvases.values():
            canvas.draw()
            
    def load_and_plot(self):
        """Cargar archivo y mostrar gráficas 3D sin procesar"""
        if not self.input_file.get() or not os.path.exists(self.input_file.get()):
            messagebox.showerror("Error", "Seleccione un archivo válido")
            return
            
        self.status_var.set("Cargando y generando gráficas 3D...")
        thread = threading.Thread(target=self._load_and_plot_thread)
        thread.daemon = True
        thread.start()
        
    def _load_and_plot_thread(self):
        try:
            fs, data = wavfile.read(self.input_file.get())
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
                
            self.audio_data = data
            self.fs = fs
            self.filtered_data = None
            
            self.root.after(0, lambda: self.log(f"✅ Cargado: {len(data)/fs:.1f}s @ {fs}Hz"))
            self.root.after(0, self.generate_all_3d_plots)
            self.root.after(0, lambda: self.status_var.set("Gráficas 3D generadas"))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error cargando: {e}"))
            self.root.after(0, lambda: self.status_var.set("Error"))
            
    def process_and_plot(self):
        """Procesar archivo con filtro notch y mostrar gráficas 3D comparativas"""
        if not self.input_file.get() or not os.path.exists(self.input_file.get()):
            messagebox.showerror("Error", "Seleccione archivo de entrada")
            return
        if not self.output_file.get():
            messagebox.showerror("Error", "Especifique archivo de salida")
            return
            
        try:
            freqs = [float(f.strip()) for f in self.frequencies.get().split(',') if f.strip()]
            self.freqs_notch = freqs
        except:
            messagebox.showerror("Error", "Frecuencias inválidas")
            return
            
        self.is_processing = True
        self.process_btn.config(state=tk.DISABLED, text="⏳ Procesando...")
        self.status_var.set("Procesando con FFT y generando 3D...")
        
        thread = threading.Thread(target=self._process_and_plot_thread, args=(freqs,))
        thread.daemon = True
        thread.start()
        
    def _process_and_plot_thread(self, freqs):
        try:
            # Cargar
            fs, data = wavfile.read(self.input_file.get())
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
                
            self.audio_data = data
            self.fs = fs
            
            self.root.after(0, lambda: self.log(f"📂 Cargado: {len(data)/fs:.1f}s @ {fs}Hz"))
            self.root.after(0, lambda: self.log(f"🎯 Filtrando: {freqs} Hz (ancho {self.notch_width.get()}Hz)"))
            
            # Filtrar
            from fft_filter import filtrar_espectral
            y, Y, X, H = filtrar_espectral(data, fs, freqs, self.notch_width.get())
            self.filtered_data = y
            
            # Guardar
            y_clip = np.clip(y, -1.0, 1.0)
            y_int16 = (y_clip * 32767).astype(np.int16)
            wavfile.write(self.output_file.get(), fs, y_int16)
            
            # SNR
            snr_in = calcular_snr(data, data)  # placeholder
            snr_out = calcular_snr(data, y)
            
            self.root.after(0, lambda: self.log(f"💾 Guardado: {os.path.basename(self.output_file.get())}"))
            self.root.after(0, lambda: self.log(f"📊 SNR salida: {snr_out:.1f} dB"))
            
            # Generar gráficas 3D
            self.root.after(0, self.generate_all_3d_plots)
            self.root.after(0, lambda: self.status_var.set("✅ Procesado y visualizado en 3D"))
            self.root.after(0, lambda: messagebox.showinfo("Éxito", "Audio procesado y gráficas 3D generadas"))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error: {e}"))
            self.root.after(0, lambda: self.status_var.set("Error"))
        finally:
            self.is_processing = False
            self.root.after(0, lambda: self.process_btn.config(state=tk.NORMAL, text="🚀 PROCESAR Y VISUALIZAR 3D"))
            
    def generate_all_3d_plots(self):
        """Generar todas las visualizaciones 3D"""
        if self.audio_data is None:
            return
            
        self.log("🎨 Generando visualizaciones 3D...")
        
        if self.show_spectrogram.get():
            self.plot_spectrogram_3d()
        if self.show_spectrum3d.get():
            self.plot_spectrum_3d()
        if self.show_waterfall.get():
            self.plot_waterfall_3d()
        if self.show_comparison.get() and self.filtered_data is not None:
            self.plot_comparison_3d()
            
        self.log("✅ Gráficas 3D actualizadas")
        
    def plot_spectrogram_3d(self):
        """Espectrograma 3D: Tiempo x Frecuencia x Magnitud (superficie)"""
        ax = self.axes_3d['spectrogram']
        ax.clear()
        
        # Calcular espectrograma
        nperseg = min(1024, len(self.audio_data) // 8)
        f, t, Sxx = spectrogram(self.audio_data, self.fs, nperseg=nperseg, noverlap=nperseg//2)
        
        # Limitar frecuencia para mejor visualización
        max_freq_idx = np.where(f <= 5000)[0]
        if len(max_freq_idx) > 0:
            f = f[max_freq_idx]
            Sxx = Sxx[max_freq_idx, :]
            
        # Convertir a dB
        if self.log_scale.get():
            Sxx_db = 10 * np.log10(Sxx + 1e-10)
            z_label = 'Magnitud (dB)'
        else:
            Sxx_db = Sxx
            z_label = 'Magnitud'
            
        # Crear malla 3D
        T, F = np.meshgrid(t, f)
        
        # Submuestrear para rendimiento
        stride = max(1, len(t) // 100)
        T_sub = T[::5, ::stride]
        F_sub = F[::5, ::stride]
        S_sub = Sxx_db[::5, ::stride]
        
        # Plot superficie
        surf = ax.plot_surface(T_sub, F_sub, S_sub, cmap=cm.viridis, 
                                linewidth=0, antialiased=True, alpha=0.9,
                                rstride=1, cstride=1)
        
        ax.set_xlabel('Tiempo (s)', labelpad=10)
        ax.set_ylabel('Frecuencia (Hz)', labelpad=10)
        ax.set_zlabel(z_label, labelpad=10)
        ax.set_title('Espectrograma 3D - Señal Original', fontsize=11, pad=10)
        
        # Marcar frecuencias notch
        for freq in self.freqs_notch:
            if freq <= f[-1]:
                ax.axhline(y=freq, color='red', linestyle='--', alpha=0.5, linewidth=1)
                
        ax.view_init(elev=self.elev_var.get(), azim=self.azim_var.get())
        ax.dist = self.dist_var.get()
        self.canvases['spectrogram'].draw()
        
    def plot_spectrum_3d(self):
        """Espectro de frecuencia 3D: barras 3D antes/después"""
        ax = self.axes_3d['spectrum3d']
        ax.clear()
        
        N = len(self.audio_data)
        X = fft(self.audio_data)
        freqs = fftshift(fftfreq(N, 1/self.fs))
        mag = fftshift(np.abs(X))
        
        # Solo frecuencias positivas hasta 5kHz
        pos_mask = (freqs >= 0) & (freqs <= 5000)
        freqs_pos = freqs[pos_mask]
        mag_pos = mag[pos_mask]
        
        if self.log_scale.get():
            mag_pos = 20 * np.log10(mag_pos + 1e-10)
            z_label = 'Magnitud (dB)'
        else:
            z_label = 'Magnitud'
            
        # Crear barras 3D
        x_pos = freqs_pos
        y_pos = np.zeros_like(x_pos)
        z_pos = np.zeros_like(x_pos)
        dx = np.diff(freqs_pos).mean() * 0.8 if len(freqs_pos) > 1 else 10
        dy = 0.5
        dz = mag_pos
        
        # Colorear según si está en notch
        colors = []
        for f in freqs_pos:
            in_notch = any(abs(f - fn) < self.notch_width.get()/2 for fn in self.freqs_notch)
            colors.append('red' if in_notch else 'blue')
            
        ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=colors, alpha=0.7, shade=True)
        
        # Si hay filtrada, superponer
        if self.filtered_data is not None:
            Y = fft(self.filtered_data)
            mag_y = fftshift(np.abs(Y))[pos_mask]
            if self.log_scale.get():
                mag_y = 20 * np.log10(mag_y + 1e-10)
            # Barras transparentes para filtrada
            ax.bar3d(x_pos, y_pos + 0.6, z_pos, dx, dy, mag_y, 
                     color='green', alpha=0.4, shade=True)
            
        ax.set_xlabel('Frecuencia (Hz)', labelpad=10)
        ax.set_ylabel('Canal', labelpad=10)
        ax.set_zlabel(z_label, labelpad=10)
        ax.set_title('Espectro 3D: Original (azul/rojo) vs Filtrado (verde)', fontsize=11, pad=10)
        ax.set_yticks([0.25, 0.85])
        ax.set_yticklabels(['Original', 'Filtrado'])
        
        ax.view_init(elev=self.elev_var.get(), azim=self.azim_var.get())
        ax.dist = self.dist_var.get()
        self.canvases['spectrum3d'].draw()
        
    def plot_waterfall_3d(self):
        """Waterfall plot 3D: múltiples espectros en el tiempo"""
        ax = self.axes_3d['waterfall']
        ax.clear()
        
        # Dividir en ventanas temporales
        window_size = self.fs // 10  # 100ms windows
        hop_size = window_size // 2
        n_windows = min(50, (len(self.audio_data) - window_size) // hop_size)
        
        if n_windows < 2:
            ax.text(0.5, 0.5, 0.5, 'Audio muy corto\npara waterfall', 
                    transform=ax.transAxes, ha='center', va='center')
            self.canvases['waterfall'].draw()
            return
            
        spectra = []
        times = []
        
        for i in range(n_windows):
            start = i * hop_size
            end = start + window_size
            window = self.audio_data[start:end] * np.hanning(window_size)
            spec = np.abs(fft(window))[:window_size//2]
            freqs = fftfreq(window_size, 1/self.fs)[:window_size//2]
            
            # Solo hasta 5kHz
            mask = freqs <= 5000
            spectra.append(spec[mask])
            times.append(start / self.fs)
            
        spectra = np.array(spectra).T  # freq x time
        freqs_plot = freqs[mask]
        
        if self.log_scale.get():
            spectra = 20 * np.log10(spectra + 1e-10)
            z_label = 'Magnitud (dB)'
        else:
            z_label = 'Magnitud'
            
        # Malla para waterfall
        T, F = np.meshgrid(times, freqs_plot)
        
        # Plot líneas 3D (waterfall)
        for i in range(0, len(times), max(1, len(times)//30)):
            z = spectra[:, i]
            ax.plot(times[i] * np.ones_like(freqs_plot), freqs_plot, z, 
                    'b-', alpha=0.6, linewidth=0.8)
                    
        # También superficie semitransparente
        stride_t = max(1, len(times) // 30)
        stride_f = max(1, len(freqs_plot) // 50)
        surf = ax.plot_surface(T[::stride_f, ::stride_t], F[::stride_f, ::stride_t], 
                                spectra[::stride_f, ::stride_t],
                                cmap=cm.plasma, alpha=0.6, linewidth=0, antialiased=True)
        
        ax.set_xlabel('Tiempo (s)', labelpad=10)
        ax.set_ylabel('Frecuencia (Hz)', labelpad=10)
        ax.set_zlabel(z_label, labelpad=10)
        ax.set_title('Waterfall 3D - Evolución Espectral', fontsize=11, pad=10)
        
        ax.view_init(elev=self.elev_var.get(), azim=self.azim_var.get())
        ax.dist = self.dist_var.get()
        self.canvases['waterfall'].draw()
        
    def plot_comparison_3d(self):
        """Comparación 3D lado a lado: Original vs Filtrado"""
        ax = self.axes_3d['comparison']
        ax.clear()
        
        # Espectros de ambos
        N = len(self.audio_data)
        X = fftshift(fft(self.audio_data))
        Y = fftshift(fft(self.filtered_data))
        freqs = fftshift(fftfreq(N, 1/self.fs))
        
        pos_mask = (freqs >= 0) & (freqs <= 5000)
        f = freqs[pos_mask]
        mag_x = np.abs(X[pos_mask])
        mag_y = np.abs(Y[pos_mask])
        
        if self.log_scale.get():
            mag_x = 20 * np.log10(mag_x + 1e-10)
            mag_y = 20 * np.log10(mag_y + 1e-10)
            z_label = 'Magnitud (dB)'
        else:
            z_label = 'Magnitud'
            
        # Crear superficie de diferencia
        diff = mag_x - mag_y
        
        # Plot 3D: x=freq, y=canal (0=orig, 1=filt), z=magnitud
        # Original
        x_orig = f
        y_orig = np.zeros_like(f)
        z_orig = mag_x
        
        # Filtrado
        x_filt = f
        y_filt = np.ones_like(f)
        z_filt = mag_y
        
        # Barras para original
        dx = (f[1] - f[0]) * 0.8 if len(f) > 1 else 10
        dy = 0.4
        ax.bar3d(x_orig, y_orig, np.zeros_like(z_orig), dx, dy, z_orig, 
                 color='blue', alpha=0.6, shade=True, label='Original')
        
        # Barras para filtrado
        ax.bar3d(x_filt, y_filt, np.zeros_like(z_filt), dx, dy, z_filt,
                 color='green', alpha=0.6, shade=True, label='Filtrado')
        
        # Línea de diferencia (atenuación)
        ax.plot(f, 0.5 * np.ones_like(f), diff, 'r-', linewidth=2, label='Atenuación', alpha=0.8)
        
        # Marcar notches
        for freq in self.freqs_notch:
            if freq <= f[-1]:
                ax.axvline(x=freq, color='red', linestyle=':', alpha=0.4)
                ax.text(freq, 0.5, max(mag_x.max(), mag_y.max()) * 0.9, 
                        f'{freq}Hz', rotation=90, fontsize=7, color='red')
                
        ax.set_xlabel('Frecuencia (Hz)', labelpad=10)
        ax.set_ylabel('Señal', labelpad=10)
        ax.set_zlabel(z_label, labelpad=10)
        ax.set_yticks([0.2, 0.8])
        ax.set_yticklabels(['Original', 'Filtrado'])
        ax.set_title('Comparación 3D: Original vs Filtrado + Atenuación (rojo)', fontsize=11, pad=10)
        ax.legend()
        
        ax.view_init(elev=self.elev_var.get(), azim=self.azim_var.get())
        ax.dist = self.dist_var.get()
        self.canvases['comparison'].draw()
        
    def update_3d_view(self):
        """Actualizar vista 3D según pestaña activa"""
        current_tab = self.current_view.get()
        tab_map = {
            "Espectrograma 3D": "spectrogram",
            "Espectro 3D Comparativo": "spectrum3d",
            "Waterfall": "waterfall",
            "Superficie Espectral": "spectrogram",
            "Fase 3D": "comparison",
        }
        key = tab_map.get(current_tab, "spectrogram")
        
        # Regenerar la gráfica activa
        if key == "spectrogram":
            self.plot_spectrogram_3d()
        elif key == "spectrum3d":
            self.plot_spectrum_3d()
        elif key == "waterfall":
            self.plot_waterfall_3d()
        elif key == "comparison":
            self.plot_comparison_3d()
            
    def update_camera(self, *args):
        """Actualizar cámara de todas las gráficas 3D"""
        elev = self.elev_var.get()
        azim = self.azim_var.get()
        dist = self.dist_var.get()
        
        for ax in self.axes_3d.values():
            ax.view_init(elev=elev, azim=azim)
            ax.dist = dist
            
        self.redraw_all()
        
    def reset_camera(self):
        self.elev_var.set(30)
        self.azim_var.set(45)
        self.dist_var.set(10)
        self.update_camera()
        
    def redraw_all(self):
        for canvas in self.canvases.values():
            canvas.draw_idle()
            
    def toggle_animation(self):
        self.anim_running = not self.anim_running
        if self.anim_running:
            self.anim_btn.config(text="⏸ Pausar Rotación")
            self.animate_rotation()
        else:
            self.anim_btn.config(text="▶ Rotar Automático")
            
    def animate_rotation(self):
        if not self.anim_running:
            return
            
        current_azim = self.azim_var.get()
        speed = self.anim_speed.get()
        new_azim = (current_azim + speed) % 360
        if new_azim > 180:
            new_azim -= 360
        self.azim_var.set(new_azim)
        self.update_camera()
        
        self.root.after(50, self.animate_rotation)
        
    def save_3d_image(self):
        """Guardar imagen de la gráfica 3D actual"""
        current_tab = self.current_view.get()
        tab_map = {
            "Espectrograma 3D": "spectrogram",
            "Espectro 3D Comparativo": "spectrum3d",
            "Waterfall": "waterfall",
            "Superficie Espectral": "spectrogram",
            "Fase 3D": "comparison",
        }
        key = tab_map.get(current_tab, "spectrogram")
        
        filename = filedialog.asksaveasfilename(
            title="Guardar imagen 3D",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")]
        )
        if filename:
            self.figures[key].savefig(filename, dpi=300, bbox_inches='tight')
            self.log(f"💾 Imagen guardada: {filename}")
            messagebox.showinfo("Guardado", f"Imagen 3D guardada en:\n{filename}")
            
    def record_rotation(self):
        """Grabar rotación 360° como GIF"""
        messagebox.showinfo("GIF", "Función de grabación GIF requiere imageio.\nInstale: pip install imageio\n\nPor ahora use 'Guardar Imagen 3D' para capturas estáticas.")
        
    def run_demo_3d(self):
        """Ejecutar demo sintética y mostrar en 3D"""
        self.is_processing = True
        self.status_var.set("Ejecutando demo sintética 3D...")
        
        thread = threading.Thread(target=self._run_demo_3d_thread)
        thread.daemon = True
        thread.start()
        
    def _run_demo_3d_thread(self):
        try:
            # Generar señal sintética
            fs = 44100
            duracion = 3.0
            t, s = generar_senal_base(fs, duracion)
            x = contaminar_senal(s, t, fs)
            y, Y, X, H = filtrar_espectral(x, fs, [60, 2500], 12)
            
            snr_ini = calcular_snr(s, x)
            snr_fin = calcular_snr(s, y)
            
            self.audio_data = x
            self.filtered_data = y
            self.fs = fs
            self.freqs_notch = [60, 2500]
            
            self.root.after(0, lambda: self.log(f"🧪 Demo: SNR {snr_ini:.1f} → {snr_fin:.1f} dB (ganancia: {snr_fin-snr_ini:.1f} dB)"))
            self.root.after(0, self.generate_all_3d_plots)
            self.root.after(0, lambda: self.status_var.set("✅ Demo 3D completada"))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Demo falló: {e}"))
        finally:
            self.is_processing = False
            
    def generate_test_files(self):
        """Generar archivos de prueba"""
        thread = threading.Thread(target=self._generate_test_files_thread)
        thread.daemon = True
        thread.start()
        
    def _generate_test_files_thread(self):
        try:
            from generate_test_audio import main as gen_main
            import io, sys
            from contextlib import redirect_stdout
            
            buf = io.StringIO()
            with redirect_stdout(buf):
                gen_main()
                
            output = buf.getvalue()
            for line in output.strip().split('\n'):
                self.root.after(0, lambda l=line: self.log(l))
                
            self.root.after(0, lambda: messagebox.showinfo("Listo", "Archivos generados en data/"))
            self.root.after(0, lambda: self.status_var.set("Archivos de prueba generados"))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))


def main():
    root = tk.Tk()
    
    style = ttk.Style()
    style.configure('Accent.TButton', font=('Segoe UI', 10, 'bold'))
    
    # Configurar matplotlib para mejor rendimiento
    matplotlib.rcParams['animation.html'] = 'none'
    
    app = AudioRestorationGUI3D(root)
    root.mainloop()


if __name__ == "__main__":
    main()