#!/usr/bin/env python3
"""
Simple Ultimate Otoya GUI - Basit ve hızlı arayüz
"""

import sys
import os
sys.path.append('.')

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import time
import numpy as np
from datetime import datetime

# Ultimate Otoya modüllerini import et
try:
    from ultimate_otoya import UltimateConfig, UltimateOtoyaModel
    from quantum_tensor import QuantumTensor
    from train_ultimate import UltimateTrainer
    HAS_ULTIMATE = True
except ImportError as e:
    HAS_ULTIMATE = False
    print(f"Import error: {e}")

class SimpleUltimateGUI:
    """Basit Ultimate Otoya GUI Sınıfı"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🚀 Ultimate Otoya - Basit Arayüz")
        self.root.geometry("900x700")
        
        # Model ve trainer referansları
        self.model = None
        self.trainer = None
        self.training_thread = None
        self.is_training = False
        
        # Setup GUI
        self.setup_gui()
        
        # Status variables
        self.training_status = "Hazır"
        self.current_step = 0
        self.total_steps = 100
        self.training_start_time = 0
        
    def setup_gui(self):
        """Basit GUI bileşenlerini kur"""
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="🚀 ULTIMATE OTOYA - BASİT ARAYÜZ",
            font=("Helvetica", 16, "bold")
        )
        title_label.grid(row=0, column=0, pady=(0, 20))
        
        # Configuration frame
        config_frame = ttk.LabelFrame(main_frame, text="⚙️ Model Ayarları", padding="10")
        config_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Simple configuration parameters
        config_params = [
            ("vocab_size", "Vocab Size:", 50000, 1000, 1000000),
            ("d_model", "Model Boyutu:", 512, 128, 4096),
            ("num_layers", "Katman Sayısı:", 12, 1, 100),
            ("context_length", "Context Uzunluğu:", 1024, 64, 8192),
        ]
        
        self.config_vars = {}
        for i, (key, label, default, min_val, max_val) in enumerate(config_params):
            # Label
            ttk.Label(config_frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=5)
            
            # Entry
            var = tk.DoubleVar(value=default)
            entry = ttk.Entry(config_frame, textvariable=var, width=15)
            entry.grid(row=i, column=1, sticky=tk.W, pady=5, padx=(10, 0))
            
            self.config_vars[key] = var
        
        # Learning rate
        ttk.Label(config_frame, text="Learning Rate:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.lr_var = tk.DoubleVar(value=0.0003)
        lr_entry = ttk.Entry(config_frame, textvariable=self.lr_var, width=15)
        lr_entry.grid(row=4, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        
        # Training parameters frame
        train_frame = ttk.LabelFrame(main_frame, text="🎮 Eğitim Ayarları", padding="10")
        train_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Batch size
        ttk.Label(train_frame, text="Batch Size:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.batch_size_var = tk.IntVar(value=2)  # Küçük başlangıç
        batch_spin = ttk.Spinbox(train_frame, from_=1, to=8, textvariable=self.batch_size_var, width=10)
        batch_spin.grid(row=0, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        
        # Sequence length
        ttk.Label(train_frame, text="Sequence Uzunluğu:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.seq_len_var = tk.IntVar(value=128)  # Küçük başlangıç
        seq_spin = ttk.Spinbox(train_frame, from_=32, to=512, textvariable=self.seq_len_var, width=10)
        seq_spin.grid(row=1, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        
        # Total steps
        ttk.Label(train_frame, text="Toplam Adım:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.total_steps_var = tk.IntVar(value=50)  # Küçük başlangıç
        steps_spin = ttk.Spinbox(train_frame, from_=10, to=500, textvariable=self.total_steps_var, width=10)
        steps_spin.grid(row=2, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        
        # Control buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, pady=(10, 20))
        
        # Create model button
        self.create_model_btn = ttk.Button(
            button_frame, 
            text="🧠 Model Oluştur", 
            command=self.create_model_simple,
            width=15
        )
        self.create_model_btn.grid(row=0, column=0, padx=5)
        
        # Start training button
        self.start_train_btn = ttk.Button(
            button_frame, 
            text="🚀 Eğitimi Başlat", 
            command=self.start_training_simple,
            width=15,
            state=tk.DISABLED
        )
        self.start_train_btn.grid(row=0, column=1, padx=5)
        
        # Stop training button
        self.stop_train_btn = ttk.Button(
            button_frame, 
            text="⏹️ Eğitimi Durdur", 
            command=self.stop_training_simple,
            width=15,
            state=tk.DISABLED
        )
        self.stop_train_btn.grid(row=0, column=2, padx=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(
            main_frame, 
            variable=self.progress_var,
            maximum=100.0,
            length=400
        )
        self.progress_bar.grid(row=4, column=0, pady=(0, 10))
        
        # Status label
        self.status_label = ttk.Label(
            main_frame,
            text="Hazır",
            font=("Helvetica", 10, "bold")
        )
        self.status_label.grid(row=5, column=0, pady=(0, 10))
        
        # Output text area
        output_frame = ttk.LabelFrame(main_frame, text="📊 Çıktı", padding="10")
        output_frame.grid(row=6, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        self.output_text = scrolledtext.ScrolledText(
            output_frame,
            width=80,
            height=15,
            wrap=tk.WORD,
            font=("Courier", 9)
        )
        self.output_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure output frame grid
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
        
        # Memory info frame
        mem_frame = ttk.LabelFrame(main_frame, text="💾 Bellek Bilgisi", padding="10")
        mem_frame.grid(row=7, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.mem_label = ttk.Label(
            mem_frame,
            text="RAM: N/A | VRAM: N/A",
            font=("Courier", 9)
        )
        self.mem_label.grid(row=0, column=0)
        
        # Update memory info periodically
        self.update_memory_info()
    
    def log_message(self, message):
        """Mesajı log'a ekle"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        self.output_text.insert(tk.END, formatted_message)
        self.output_text.see(tk.END)
        
        # Update status label
        short_msg = message[:60] + "..." if len(message) > 60 else message
        self.status_label.config(text=short_msg)
    
    def update_memory_info(self):
        """Bellek bilgisini güncelle"""
        try:
            if HAS_ULTIMATE:
                # Get memory stats
                mem_stats = QuantumTensor.get_memory_stats()
                
                # Update label
                ram_gb = mem_stats.get('allocated_ram_gb', 0)
                vram_gb = mem_stats.get('allocated_vram_gb', 0)
                ram_percent = mem_stats.get('ram_usage_percent', 0)
                status = mem_stats.get('memory_status', 'UNKNOWN')
                
                self.mem_label.config(
                    text=f"RAM: {ram_gb:.2f}GB ({ram_percent:.1f}%) | Status: {status}"
                )
                
                # Warning if memory usage is high
                if ram_percent > 70:
                    self.mem_label.config(foreground='orange')
                elif ram_percent > 85:
                    self.mem_label.config(foreground='red')
                else:
                    self.mem_label.config(foreground='black')
        except Exception as e:
            self.mem_label.config(text=f"RAM: N/A | Error: {str(e)[:30]}")
        
        # Schedule next update
        self.root.after(2000, self.update_memory_info)
    
    def create_model_simple(self):
        """Basit model oluştur - Lazy initialization ile donma sorununu çöz"""
        try:
            self.log_message("Model oluşturuluyor (lazy initialization)...")
            
            # Kullanıcının 16GB VRAM/RAM için 14.5GB limit istedi
            # Windows 1GB kullanıyor, biz 14.5GB limit koyalım
            max_memory_gb = 14.5
            
            # Küçük model için ayarlar - donma sorununu önlemek için çok küçük başlangıç
            vocab_size = min(int(self.config_vars['vocab_size'].get()), 5000)  # 10k'dan 5k'ya düşür
            d_model = min(int(self.config_vars['d_model'].get()), 128)        # 256'dan 128'e düşür
            num_layers = min(int(self.config_vars['num_layers'].get()), 4)    # 6'dan 4'e düşür
            context_length = min(int(self.config_vars['context_length'].get()), 128)  # 256'dan 128'e düşür
            learning_rate = min(self.lr_var.get(), 0.0005)                    # 0.001'den 0.0005'e düşür
            
            # Memory limit bilgisi
            self.log_message(f"💾 Memory limit: {max_memory_gb}GB (16GB sistem için)")
            
            # Get configuration values with memory limits
            config = UltimateConfig(
                vocab_size=vocab_size,
                d_model=d_model,
                num_layers=num_layers,
                num_heads=2,  # 4'ten 2'ye düşür
                context_length=context_length,
                learning_rate=learning_rate,
                batch_size=1,  # Küçük batch size
            )
            
            self.log_message(f"Model oluşturuluyor: {config}")
            
            # Create model with lazy initialization
            self.model = UltimateOtoyaModel(config)
            
            # Create trainer with small batch size
            self.trainer = UltimateTrainer(config)
            
            # Parameter count (lazy olarak hesapla)
            param_count = 0
            try:
                param_count = sum(p.data.size for p in self.model.parameters())
            except:
                # Lazy initialization durumunda tahmini hesapla
                param_count = config.vocab_size * config.d_model + \
                             config.num_layers * (4 * config.d_model * config.d_model)
            
            # Memory estimation
            mem_estimate_mb = param_count * 4 / (1024 * 1024)  # float32 için
            mem_estimate_gb = mem_estimate_mb / 1024
            
            self.log_message(f"✅ Model başarıyla oluşturuldu!")
            self.log_message(f"   • Parametreler: {param_count:,}")
            self.log_message(f"   • Tahmini Bellek: {mem_estimate_gb:.2f}GB")
            self.log_message(f"   • d_model: {d_model}")
            self.log_message(f"   • Layers: {num_layers}")
            self.log_message(f"   • Context: {context_length}")
            self.log_message(f"   • Memory Limit: {max_memory_gb}GB (kullanılan: {mem_estimate_gb:.2f}GB)")
            
            # Enable training button
            self.start_train_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            self.log_message(f"❌ Model oluşturma hatası: {e}")
            import traceback
            self.log_message(traceback.format_exc())
    
    def start_training_simple(self):
        """Basit eğitim başlat"""
        if not self.model or not self.trainer:
            self.log_message("❌ Önce model oluşturmalısınız!")
            return
        
        self.is_training = True
        self.training_start_time = time.time()
        self.current_step = 0
        
        # Update UI
        self.start_train_btn.config(state=tk.DISABLED)
        self.stop_train_btn.config(state=tk.NORMAL)
        self.create_model_btn.config(state=tk.DISABLED)
        
        self.log_message("🚀 Eğitim başlatılıyor...")
        
        # Start training in a separate thread
        self.training_thread = threading.Thread(target=self.run_training_simple)
        self.training_thread.daemon = True
        self.training_thread.start()
    
    def run_training_simple(self):
        """Basit eğitim çalıştır - PC donma sorununu çözmek için"""
        try:
            total_steps = min(self.total_steps_var.get(), 50)   # Maksimum 50 adım (daha küçük)
            batch_size = min(self.batch_size_var.get(), 2)      # Maksimum batch size 2 (daha küçük)
            seq_len = min(self.seq_len_var.get(), 64)           # Maksimum sequence length 64 (daha küçük)
            
            # Memory safety check
            self.log_message(f"💾 Eğitim başlatılıyor:")
            self.log_message(f"   • Batch size: {batch_size}")
            self.log_message(f"   • Sequence length: {seq_len}")
            self.log_message(f"   • Total steps: {total_steps}")
            self.log_message(f"   • Memory limit: 14.5GB")
            
            for step in range(total_steps):
                if not self.is_training:
                    break
                
                # Update progress
                self.current_step = step + 1
                progress = (step + 1) / total_steps * 100
                
                # Update progress bar (must be done in main thread)
                self.root.after(0, lambda p=progress: self.progress_var.set(p))
                
                # Training step
                result = self.trainer.training_step(batch_size, seq_len)
                
                # Log progress every 3 steps (daha sık)
                if (step + 1) % 3 == 0 or step == 0:
                    message = (f"Step {step+1:3d}/{total_steps}: "
                             f"Loss: {result['loss']:.4f} | "
                             f"Acc: {result['accuracy']:.3f} | "
                             f"Time: {result['step_time']:.3f}s | "
                             f"Batch: {result['batch_size']}x{result['seq_len']}")
                    
                    # Update UI in main thread
                    self.root.after(0, lambda m=message: self.log_message(m))
                
                # Small delay to prevent UI freezing
                time.sleep(0.1)  # 100ms delay (daha uzun)
                
                # Force garbage collection every 10 steps
                if (step + 1) % 10 == 0:
                    import gc
                    gc.collect()
            
            # Training completed
            self.root.after(0, self.training_completed_simple)
            
        except MemoryError as e:
            error_msg = f"❌ MEMORY ERROR: PC donma riski! Bellek limiti aşıldı."
            self.root.after(0, lambda m=error_msg: self.log_message(m))
            self.root.after(0, self.stop_training_simple)
            
        except Exception as e:
            error_msg = f"❌ Eğitim hatası: {e}"
            self.root.after(0, lambda m=error_msg: self.log_message(m))
            import traceback
            self.root.after(0, lambda: self.log_message(traceback.format_exc()))
            self.root.after(0, self.stop_training_simple)
    
    def stop_training_simple(self):
        """Eğitimi durdur"""
        self.is_training = False
        
        # Update UI
        self.stop_train_btn.config(state=tk.DISABLED)
        self.start_train_btn.config(state=tk.NORMAL)
        self.create_model_btn.config(state=tk.NORMAL)
        
        self.log_message("⏹️ Eğitim durduruldu")
    
    def training_completed_simple(self):
        """Eğitim tamamlandı"""
        self.is_training = False
        
        # Update UI
        self.stop_train_btn.config(state=tk.DISABLED)
        self.start_train_btn.config(state=tk.NORMAL)
        self.create_model_btn.config(state=tk.NORMAL)
        
        # Get final stats
        if self.trainer and self.trainer.stats['losses']:
            avg_loss = np.mean(self.trainer.stats['losses'])
            avg_acc = np.mean(self.trainer.stats['accuracies'])
            total_time = time.time() - self.training_start_time
            
            self.log_message("=" * 60)
            self.log_message(f"🎉 Eğitim tamamlandı!")
            self.log_message(f"   • Ortalama Loss: {avg_loss:.4f}")
            self.log_message(f"   • Ortalama Accuracy: {avg_acc:.3f}")
            self.log_message(f"   • Toplam Süre: {total_time:.1f}s")
            self.log_message(f"   • Adım/Saniye: {self.current_step/total_time:.1f}")
            self.log_message("=" * 60)

def main():
    """Ana fonksiyon"""
    root = tk.Tk()
    gui = SimpleUltimateGUI(root)
    
    # Center window
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    root.mainloop()

if __name__ == "__main__":
    main()