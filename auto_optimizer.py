#!/usr/bin/env python3
"""
Ultimate Otoya Auto Optimizer - Hedef parametreye göre otomatik optimizasyon
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class OptimizationResult:
    """Optimizasyon sonucu"""
    config: Dict
    estimated_params: int
    estimated_memory_gb: float
    max_batch_size: int
    recommended_seq_len: int
    optimization_score: float
    warnings: list
    recommendations: list

class UltimateAutoOptimizer:
    """Ultimate Otoya Otomatik Optimizasyon Sınıfı"""
    
    def __init__(self, max_vram_gb: float = 15.5, max_ram_gb: float = 15.5):
        """
        Args:
            max_vram_gb: Maksimum VRAM (GB)
            max_ram_gb: Maksimum RAM (GB)
        """
        self.max_vram_gb = max_vram_gb
        self.max_ram_gb = max_ram_gb
        
        # Bellek katsayıları (her parametre tipi için)
        self.memory_coefficients = {
            'embedding': 4.0,  # bytes per parameter (float32)
            'attention': 4.0,
            'ffn': 4.0,
            'layer_norm': 4.0,
            'output': 4.0,
            'activations': 8.0,  # forward + backward
            'optimizer': 8.0,    # momentum + variance
        }
        
        # Performans profilleri
        self.performance_profiles = {
            'speed': {
                'd_model': 512,
                'num_heads': 8,
                'ffn_multiplier': 2,
                'batch_size': 8,
                'seq_len': 512,
                'priority': 'inference_speed'
            },
            'balanced': {
                'd_model': 768,
                'num_heads': 12,
                'ffn_multiplier': 4,
                'batch_size': 4,
                'seq_len': 1024,
                'priority': 'training_stability'
            },
            'quality': {
                'd_model': 1024,
                'num_heads': 16,
                'ffn_multiplier': 4,
                'batch_size': 2,
                'seq_len': 2048,
                'priority': 'model_capacity'
            }
        }
    
    def estimate_model_parameters(self, config: Dict) -> int:
        """Model parametrelerini tahmin et"""
        vocab_size = config.get('vocab_size', 50000)
        d_model = config.get('d_model', 512)
        num_layers = config.get('num_layers', 12)
        num_heads = config.get('num_heads', 8)
        
        # Embedding parametreleri
        embedding_params = vocab_size * d_model
        
        # Attention parametreleri (her layer için)
        # Q, K, V, output projections
        head_dim = d_model // num_heads
        attention_params_per_layer = 4 * d_model * d_model  # Q, K, V, O
        
        # FFN parametreleri
        ffn_multiplier = config.get('ffn_multiplier', 4)
        ffn_hidden = d_model * ffn_multiplier
        ffn_params_per_layer = 2 * d_model * ffn_hidden  # up + down
        
        # Layer norm parametreleri
        layer_norm_params_per_layer = 2 * d_model  # gamma + beta
        
        # Output layer parametreleri
        output_params = d_model * vocab_size
        
        # Toplam parametreler
        total_params = embedding_params
        total_params += num_layers * (attention_params_per_layer + ffn_params_per_layer + layer_norm_params_per_layer)
        total_params += output_params
        
        return int(total_params)
    
    def estimate_training_memory(self, config: Dict, batch_size: int, seq_len: int) -> float:
        """Eğitim için gerekli belleği tahmin et (GB)"""
        d_model = config.get('d_model', 512)
        num_layers = config.get('num_layers', 12)
        vocab_size = config.get('vocab_size', 50000)
        
        # Parametre belleği
        total_params = self.estimate_model_parameters(config)
        param_memory_gb = (total_params * self.memory_coefficients['embedding']) / (1024**3)
        
        # Aktivasyon belleği (forward + backward)
        # Her layer için: attention activations + FFN activations
        activations_per_token = d_model * 10  # yaklaşık değer
        total_activations = batch_size * seq_len * activations_per_token * num_layers
        activation_memory_gb = (total_activations * self.memory_coefficients['activations']) / (1024**3)
        
        # Optimizer state belleği (AdamW için)
        optimizer_memory_gb = (total_params * self.memory_coefficients['optimizer']) / (1024**3)
        
        # Gradyan belleği
        gradient_memory_gb = (total_params * 4) / (1024**3)  # float32 gradyanlar
        
        # Toplam bellek
        total_memory_gb = param_memory_gb + activation_memory_gb + optimizer_memory_gb + gradient_memory_gb
        
        # Güvenlik payı (%10)
        total_memory_gb *= 1.1
        
        return total_memory_gb
    
    def optimize_for_target_params(self, target_params: int, profile: str = 'balanced') -> OptimizationResult:
        """
        Hedef parametre sayısına göre otomatik optimizasyon
        
        Args:
            target_params: Hedef parametre sayısı (örn: 1_000_000_000)
            profile: Optimizasyon profili ('speed', 'balanced', 'quality')
        
        Returns:
            OptimizationResult: Optimizasyon sonucu
        """
        if profile not in self.performance_profiles:
            profile = 'balanced'
        
        base_profile = self.performance_profiles[profile]
        
        # Başlangıç konfigürasyonu
        config = {
            'vocab_size': 50000,
            'd_model': base_profile['d_model'],
            'num_layers': 12,
            'num_heads': base_profile['num_heads'],
            'context_length': base_profile['seq_len'],
            'learning_rate': 0.0003,
            'ffn_multiplier': base_profile['ffn_multiplier'],
        }
        
        # Parametre sayısını hedefe yaklaştır
        current_params = self.estimate_model_parameters(config)
        
        # Optimizasyon stratejisi
        if target_params > current_params:
            # d_model'i artır (en etkili)
            scale_factor = (target_params / current_params) ** 0.5
            config['d_model'] = int(config['d_model'] * scale_factor)
            
            # num_heads'i güncelle (d_model/64 oranında)
            config['num_heads'] = max(8, config['d_model'] // 64)
            
            # num_layers'i ayarla
            if profile == 'quality':
                config['num_layers'] = min(48, int(12 * scale_factor))
            elif profile == 'balanced':
                config['num_layers'] = min(24, int(12 * scale_factor))
            else:  # speed
                config['num_layers'] = min(16, int(12 * scale_factor))
        
        # Batch size ve sequence length optimizasyonu
        estimated_memory = self.estimate_training_memory(config, 1, config['context_length'])
        
        # Maksimum batch size hesapla
        max_batch_size = 1
        while True:
            test_memory = self.estimate_training_memory(config, max_batch_size + 1, config['context_length'])
            if test_memory > self.max_vram_gb * 0.9:  # %90 VRAM kullanımı
                break
            max_batch_size += 1
        
        # Sequence length optimizasyonu
        recommended_seq_len = config['context_length']
        if profile == 'speed':
            recommended_seq_len = min(512, config['context_length'])
        elif profile == 'balanced':
            recommended_seq_len = min(1024, config['context_length'])
        
        # Optimizasyon skoru
        final_params = self.estimate_model_parameters(config)
        param_ratio = min(1.0, final_params / target_params) if target_params > 0 else 1.0
        
        memory_efficiency = 1.0 - (estimated_memory / self.max_vram_gb)
        memory_efficiency = max(0.0, min(1.0, memory_efficiency))
        
        optimization_score = (param_ratio * 0.6 + memory_efficiency * 0.4) * 100
        
        # Uyarılar ve öneriler
        warnings = []
        recommendations = []
        
        if estimated_memory > self.max_vram_gb * 0.8:
            warnings.append(f"Yüksek VRAM kullanımı: {estimated_memory:.1f}GB")
            recommendations.append("Batch size veya sequence length'i azaltın")
        
        if final_params < target_params * 0.8:
            warnings.append(f"Hedef parametreye ulaşılamadı: {final_params:,} < {target_params:,}")
            recommendations.append("d_model veya num_layers'ı artırın")
        
        if max_batch_size < 2:
            warnings.append("Çok küçük batch size")
            recommendations.append("Model boyutunu azaltın veya sequence length'i kısaltın")
        
        # Öneriler
        if profile == 'speed' and config['d_model'] > 1024:
            recommendations.append("Speed profili için d_model <= 1024 önerilir")
        
        if profile == 'quality' and config['num_layers'] < 24:
            recommendations.append("Quality profili için daha fazla layer ekleyin")
        
        return OptimizationResult(
            config=config,
            estimated_params=final_params,
            estimated_memory_gb=estimated_memory,
            max_batch_size=max_batch_size,
            recommended_seq_len=recommended_seq_len,
            optimization_score=optimization_score,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def optimize_for_memory(self, target_memory_gb: float, profile: str = 'balanced') -> OptimizationResult:
        """
        Hedef belleğe göre otomatik optimizasyon
        
        Args:
            target_memory_gb: Hedef bellek kullanımı (GB)
            profile: Optimizasyon profili
        
        Returns:
            OptimizationResult: Optimizasyon sonucu
        """
        if target_memory_gb > self.max_vram_gb:
            target_memory_gb = self.max_vram_gb * 0.8
        
        # Binary search ile optimal konfigürasyon bul
        low_params = 1_000_000
        high_params = 10_000_000_000
        
        best_result = None
        
        for _ in range(20):  # 20 iterasyon
            mid_params = (low_params + high_params) // 2
            result = self.optimize_for_target_params(mid_params, profile)
            
            if result.estimated_memory_gb <= target_memory_gb:
                best_result = result
                low_params = mid_params
            else:
                high_params = mid_params
        
        return best_result if best_result else self.optimize_for_target_params(1_000_000_000, profile)
    
    def get_optimization_presets(self) -> Dict[str, Dict]:
        """Optimizasyon preset'lerini getir"""
        presets = {
            'tiny': {
                'target_params': 100_000_000,
                'profile': 'speed',
                'description': 'Hızlı testler için'
            },
            'small': {
                'target_params': 500_000_000,
                'profile': 'balanced',
                'description': 'Hızlı eğitim için'
            },
            'medium': {
                'target_params': 1_500_000_000,
                'profile': 'balanced',
                'description': 'Genel kullanım için'
            },
            'large': {
                'target_params': 3_000_000_000,
                'profile': 'quality',
                'description': 'Yüksek kalite için'
            },
            'xlarge': {
                'target_params': 7_000_000_000,
                'profile': 'quality',
                'description': 'Maksimum performans için'
            }
        }
        
        return presets
    
    def generate_config_summary(self, result: OptimizationResult) -> str:
        """Konfigürasyon özeti oluştur"""
        summary = []
        summary.append("=" * 60)
        summary.append("🚀 ULTIMATE OTOYA - OTOMATİK OPTİMİZASYON")
        summary.append("=" * 60)
        summary.append("")
        summary.append("📊 OPTİMİZASYON SONUÇLARI:")
        summary.append(f"  • Optimizasyon Skoru: {result.optimization_score:.1f}/100")
        summary.append(f"  • Tahmini Parametreler: {result.estimated_params:,}")
        summary.append(f"  • Tahmini Bellek: {result.estimated_memory_gb:.1f}GB")
        summary.append(f"  • Maksimum Batch Size: {result.max_batch_size}")
        summary.append(f"  • Önerilen Seq Length: {result.recommended_seq_len}")
        summary.append("")
        summary.append("⚙️ ÖNERİLEN KONFİGÜRASYON:")
        for key, value in result.config.items():
            summary.append(f"  • {key}: {value}")
        summary.append("")
        
        if result.warnings:
            summary.append("⚠️ UYARILAR:")
            for warning in result.warnings:
                summary.append(f"  • {warning}")
            summary.append("")
        
        if result.recommendations:
            summary.append("💡 ÖNERİLER:")
            for rec in result.recommendations:
                summary.append(f"  • {rec}")
            summary.append("")
        
        summary.append("=" * 60)
        
        return "\n".join(summary)

# Test fonksiyonu
def test_auto_optimizer():
    """Auto optimizer test"""
    optimizer = UltimateAutoOptimizer(max_vram_gb=15.5, max_ram_gb=15.5)
    
    print("🧪 Auto Optimizer Testi")
    print("-" * 40)
    
    # Hedef parametre testi
    print("\n1. Hedef Parametre: 1 Milyar")
    result = optimizer.optimize_for_target_params(1_000_000_000, 'balanced')
    print(optimizer.generate_config_summary(result))
    
    print("\n2. Hedef Parametre: 3 Milyar")
    result = optimizer.optimize_for_target_params(3_000_000_000, 'quality')
    print(optimizer.generate_config_summary(result))
    
    print("\n3. Preset'ler:")
    presets = optimizer.get_optimization_presets()
    for name, preset in presets.items():
        print(f"  • {name}: {preset['description']} ({preset['target_params']:,} params)")

if __name__ == "__main__":
    test_auto_optimizer()