"""
scalable_jit.py
================

Eita / Otoya için blok‑bazlı, geriye dönük uyumlu, hata‑toleranslı
bir JIT sarmalayıcısı.

Amaç
----
* TinyJit'in 96M+ parametreli modellerde yaşadığı **dev graph patlamasını**
  hafifletmek.
* TinyJit'in *kullanıcı tarafı* interface'ini bozmadan, onun önüne bir
  katman koymak.
* Herhangi bir derleme/çalıştırma hatasında otomatik olarak emniyetli
  yola (eager TinyJit veya saf eager) düşmek.

Çalışma şekli
-------------
1.  Modeldeki blok listesini (`model.layers` veya `model.blocks`)
    otomatik keşfeder. Bulamazsa tüm modeli tek parça olarak bırakır.
2.  Her bloğa ince bir sargı (`_RealizeWrappedBlock`) takar. Sargı
    *her N blokta bir* ara tensörü realize eder → derleyicinin
    gördüğü graph küçülür.
3.  Kullanıcı tarafı `__call__(...)` aynen TinyJit gibi çalışır:
        scaler = ScalableJIT(_forward_backward, model=model)
        loss, lm_loss = scaler(x, y, strength)
4.  Derleme hatası olursa: bloğun sargısı sessizce devre dışı kalır,
    model olduğu gibi çalışmaya devam eder.

Dikkat
------
Bu sınıf Tinygrad iç API'sine (LazyOp, run_graph, vs.) bağlanmaz.
Sadece public Tensor / TinyJit yüzeyini kullanır; bu yüzden
Tinygrad sürüm değişikliklerine karşı dayanıklıdır.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, List, Optional, Sequence

from tinygrad import Tensor  # tip kontrolü + isinstance için

logger = logging.getLogger("eita.scalable_jit")


# ---------------------------------------------------------------------------
# Konfigürasyon
# ---------------------------------------------------------------------------

def _default_params_m(model) -> float:
    """Modelin parametre sayısını (yaklaşık) megabyte cinsinden döndür."""
    try:
        from tinygrad.nn.state import get_parameters
        n = 0
        for p in get_parameters(model):
            try:
                n += int(p.numel())
            except Exception:
                pass
        return n / 1e6
    except Exception:
        return 0.0


def discover_blocks(model) -> Optional[List[Any]]:
    """Modelin blok listesini bul. Bulunamazsa None.

    ÖNEMLİ: Orijinal container referansını döndürür (kopya değil).
    ScalableJIT referans eşitliğine dayanır.
    """
    for attr in ("layers", "blocks", "h"):
        blocks = getattr(model, attr, None)
        if isinstance(blocks, (list, tuple)) and len(blocks) > 1:
            return blocks
    return None


def unwrap_model_blocks(model) -> bool:
    """Modelde daha once takilmis `_RealizeWrappedBlock` sargilarini sok.

    Faz gecislerinde yeni bir ScalableJIT olusturulurken eski sargilar model
    uzerinde kalirsa wrapper'lar ust uste binebilir. Bu da her fazda ek call
    overhead ve yanlis realize sinirlari uretir.
    """
    changed = False
    for attr in ("layers", "blocks", "h"):
        blocks = getattr(model, attr, None)
        if not isinstance(blocks, (list, tuple)) or not blocks:
            continue
        if not all(isinstance(blk, _RealizeWrappedBlock) for blk in blocks):
            continue
        raw_blocks = [blk.block for blk in blocks]
        setattr(model, attr, tuple(raw_blocks) if isinstance(blocks, tuple) else list(raw_blocks))
        changed = True
    return changed


# ---------------------------------------------------------------------------
# Ara realizasyon sargısı
# ---------------------------------------------------------------------------

class _RealizeWrappedBlock:
    """
    Bir nn modülünü (örn. OtoyaBlock) sarar; her K çağrıda bir,
    forward çıktısını realize eder. Realize, derleyicinin gördüğü
    computation graph'ını fiziksel olarak kırar; böylece TinyJit
    çok büyük tek bir kernel yerine birkaç küçük kernel üretir.
    """
    __slots__ = ("block", "period", "block_index", "errors", "enabled")

    def __init__(self, block: Any, period: int, block_index: int):
        self.block = block
        self.period = max(1, int(period))
        self.block_index = int(block_index)
        self.errors = 0
        self.enabled = True

    def disable(self) -> None:
        """Bir hatadan sonra sessizce devre dışı bırak."""
        self.enabled = False

    def __getattr__(self, name: str):
        """Şeffaf proxy: sarmalanmış bloğun tüm attribute'larına erişim.

        `wrapped.phase` → `self.block.phase`,
        `wrapped.weight` → `self.block.weight` vb.

        Böylece `set_phase_trainable(l, ...)` gibi kodlar sargının
        varlığından haberdar olmak zorunda kalmaz.
        """
        # `__getattr__` yalnızca normal lookup başarısız olunca çağrılır.
        # Dahili nitelikler ve `block` için sonsuz döngüyü kır.
        if name in ("block", "enabled", "block_index", "errors", "period",
                    "__dict__", "__class__", "__slots__", "__getattr__"):
            raise AttributeError(name)
        # `__slots__` kullandığımız için `self.__dict__` mevcut değil;
        # `__getattr__` içinde `self.__dict__` aramak sonsuz döngü yaratır.
        # Bu yüzden `object.__getattribute__` ile bypass ediyoruz.
        try:
            block = object.__getattribute__(self, "block")
        except AttributeError:
            raise AttributeError(name)
        return getattr(block, name)

    def __repr__(self) -> str:
        try:
            block = object.__getattribute__(self, "block")
        except AttributeError:
            return "_RealizeWrappedBlock(<uninit>)"
        return f"_RealizeWrappedBlock({type(block).__name__})"

    def __call__(self, *args, **kwargs):
        out = self.block(*args, **kwargs)
        if not self.enabled:
            return out
        # realize_period blok indeksine gore uygulanir:
        # period=2 ise 2., 4., 6. bloklar her forward'da realize edilir.
        # Step sayacina bagli alternating graph olusmasin.
        if (self.block_index + 1) % self.period == 0:
            try:
                # Çıktı bir tuple ise (örn. OtoyaBlock: (x, stats)) ilk
                # elemanı realize et — asıl aktivasyon tensörü orada.
                if isinstance(out, tuple):
                    target = out[0]
                else:
                    target = out
                if isinstance(target, Tensor):
                    target.realize()
            except Exception as exc:  # noqa: BLE001
                self.errors += 1
                if self.errors >= 3:
                    logger.warning(
                        "ScalableJIT: realize başarısız (%s), sargı devre dışı bırakıldı.",
                        exc,
                    )
                    self.enabled = False
        return out


# ---------------------------------------------------------------------------
# Ana sınıf
# ---------------------------------------------------------------------------

class ScalableJIT:
    """
    TinyJit'e benzeyen ama:
      * Modeldeki blokları otomatik keşfeder.
      * Her N blokta bir ara `realize()` koyar (graph kırıcı).
      * Bir hata olursa emniyetli yola (TinyJit) düşer.
      * Hata olmazsa bile büyük modellerde kernel patlamasını önler.

    Kullanım (eita.py içinde):

        from scalable_jit import ScalableJIT
        jit_fb = ScalableJIT(
            _forward_backward, model=model,
            policy="auto",        # "auto" | "block" | "off"
            realize_period=2,     # her 2 blokta bir realize
        )
    """

    # policy -> (realize_period veya None, açıklama)
    _POLICY_TABLE = {
        "auto":  None,  # model büyüklüğüne göre otomatik karar
        "off":   0,     # sarmalama yok; saf TinyJit
        "block": 1,     # her bloktan sonra realize
        "layer": 2,     # her 2 blokta bir
        "wide":  3,     # her 3 blokta bir (daha az kırma, daha çok fusion)
    }

    def __init__(
        self,
        fn: Callable,
        model: Any = None,
        policy: str = "auto",
        realize_period: Optional[int] = None,
        use_tiny_jit: bool = True,
        token_load_hint: Optional[int] = None,
        extra_block_attrs: Sequence[str] = (),
    ):
        self._fn = fn
        self._model = model
        self._policy = policy
        self._explicit_period = realize_period
        self._use_tiny_jit = use_tiny_jit
        self._wrapped_blocks: List[_RealizeWrappedBlock] = []
        self._tiny_jit = None
        self._fallback_reason: Optional[str] = None
        self._compile_time_s: float = 0.0
        self._call_count: int = 0
        self._token_load_hint: Optional[int] = int(token_load_hint) if token_load_hint is not None else None
        self._container_attr: Optional[str] = None
        self._original_blocks: Optional[Sequence[Any]] = None

        # Karar ver ve uygula
        self._setup()

    # ----- kurulum ---------------------------------------------------------

    def _resolve_period(self) -> int:
        """Hangi 'realize_period' kullanılacak?"""
        if self._explicit_period is not None:
            return max(0, int(self._explicit_period))
        if self._policy == "off":
            return 0
        period = self._POLICY_TABLE.get(self._policy)
        if isinstance(period, int):
            return period
        # "auto" — model büyüklüğüne göre
        params_m = _default_params_m(self._model) if self._model is not None else 0.0
        token_load = self._token_load_hint
        if token_load is not None:
            if params_m < 25.0 and token_load >= 4096:
                return 0
            if params_m < 96.0 and token_load >= 16384:
                return 0
        if params_m <= 0:
            return 2
        if params_m < 25.0:
            return 3      # küçük model: az kır, çok fusion
        if params_m < 96.0:
            return 2      # orta: 2 blokta bir
        if params_m < 250.0:
            return 1      # büyük: her blok
        return 1          # çok büyük: her blok

    def _setup(self) -> None:
        if self._model is not None:
            unwrap_model_blocks(self._model)

        period = self._resolve_period()
        blocks = discover_blocks(self._model) if self._model is not None else None

        if blocks is None or period <= 0:
            # Sarmalama yapma; sadece TinyJit veya eager kullan
            if self._use_tiny_jit:
                self._wrap_tiny_jit()
            else:
                self._fallback_reason = "no blocks / period=0 / no tinyjit"
            return

        # Her bloğu sarmala
        original_blocks = list(blocks)
        wrapped: List[_RealizeWrappedBlock] = []
        try:
            # Bloğun tutulduğu attribute ismini bul (model.layers / .blocks / .h)
            container_attr: Optional[str] = None
            for attr in ("layers", "blocks", "h"):
                cont = getattr(self._model, attr, None)
                if cont is blocks:
                    container_attr = attr
                    break
            if container_attr is None:
                # Listeyi kapsayıcı bulamadık — sarmalama yapma
                self._fallback_reason = "blok konteyner attribute'ü bulunamadı"
                logger.warning(
                    "ScalableJIT: model.layers/.blocks/.h eşleşmedi, fallback."
                )
                return

            wrapped = [_RealizeWrappedBlock(blk, period, idx) for idx, blk in enumerate(original_blocks)]
            new_container = type(blocks)(wrapped) if not isinstance(blocks, tuple) else tuple(wrapped)
            setattr(self._model, container_attr, new_container)
            self._wrapped_blocks = wrapped
            self._container_attr = container_attr
            self._original_blocks = original_blocks
        except Exception as exc:  # noqa: BLE001
            # Model attribute'u set edilemedi — geri al
            self._fallback_reason = f"wrap failed: {exc}"
            logger.warning("ScalableJIT: blok sarma başarısız (%s), fallback.", exc)
            return

        # TinyJit'i de sarmala
        if self._use_tiny_jit:
            self._wrap_tiny_jit()
        else:
            self._fallback_reason = "use_tiny_jit=False"

        logger.info(
            "ScalableJIT hazır: %d blok sarmalandı, realize_period=%d, tinyjit=%s",
            len(self._wrapped_blocks), period,
            "aktif" if self._tiny_jit is not None else "pasif",
        )

    def _wrap_tiny_jit(self) -> None:
        try:
            from tinygrad import Tensor  # noqa: F401  (sürüm uyumu)
        except Exception:
            pass
        TinyJit = None
        for path in ("tinygrad.engine.jit", "tinygrad.jit", "tinygrad"):
            try:
                mod = __import__(path, fromlist=["TinyJit"])
                TinyJit = getattr(mod, "TinyJit", None)
                if TinyJit is not None:
                    break
            except Exception:
                continue
        if TinyJit is None:
            self._fallback_reason = "TinyJit import edilemedi"
            return
        t0 = time.time()
        try:
            self._tiny_jit = TinyJit(self._fn)
        except Exception as exc:  # noqa: BLE001
            self._fallback_reason = f"TinyJit derleme hatası: {exc}"
            logger.warning("ScalableJIT: TinyJit derleme hatası (%s), eager moda düşüldü.", exc)
            self._tiny_jit = None
            return
        self._compile_time_s = time.time() - t0

    # ----- public API ------------------------------------------------------

    def __call__(self, *args, **kwargs):
        """Eğitim adımı — TinyJit ile aynı imza."""
        self._call_count += 1
        if self._tiny_jit is not None:
            return self._tiny_jit(*args, **kwargs)
        # Emniyetli yol: eager
        return self._fn(*args, **kwargs)

    def close(self) -> None:
        """Takilan block sargilarini modelden geri sok."""
        if self._model is None:
            return
        if self._container_attr is None or self._original_blocks is None:
            unwrap_model_blocks(self._model)
            return
        try:
            restored = tuple(self._original_blocks) if isinstance(getattr(self._model, self._container_attr), tuple) else list(self._original_blocks)
            setattr(self._model, self._container_attr, restored)
        except Exception:
            unwrap_model_blocks(self._model)

    @staticmethod
    def boundary() -> None:
        """
        İleride auto-partition için no-op işaretleyici.
        Şimdilik hiçbir şey yapmaz; var olmasının tek sebebi
        model tarafında güvenle çağrılabilmesidir.
        """
        return None

    # ----- introspection ---------------------------------------------------

    @property
    def info(self) -> dict:
        return {
            "policy": self._policy,
            "realize_period": self._resolve_period() if self._wrapped_blocks else 0,
            "wrapped_blocks": len(self._wrapped_blocks),
            "tinyjit_active": self._tiny_jit is not None,
            "fallback_reason": self._fallback_reason,
            "compile_time_s": round(self._compile_time_s, 4),
            "call_count": self._call_count,
            "token_load_hint": self._token_load_hint,
        }

    def __repr__(self) -> str:
        return f"ScalableJIT({self.info})"


# ---------------------------------------------------------------------------
# Yardımcı: modelin blok listesini toplam fonksiyon olarak sarma (ileride)
# ---------------------------------------------------------------------------

def chain_forward_blocks(model, x, *extra_args, **kwargs):
    """
    EitaModel'in blok zincirini elle çalıştırır (ileride partition='block'
    modunda kullanılmak üzere yer tutucu). Şu an sadece model.layers
    varsa blokları sırayla çağırır; yoksa olduğu gibi x döner.
    """
    blocks = discover_blocks(model)
    if blocks is None:
        return x
    out = x
    for blk in blocks:
        result = blk(out, *extra_args, **kwargs)
        if isinstance(result, tuple):
            out = result[0]
        else:
            out = result
    return out
