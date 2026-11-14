"""Validated architecture configuration; checkpoints must store the full configuration."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 32000
    context_length: int = 512
    d_model: int = 512
    num_layers: int = 12
    num_heads: int = 8
    num_kv_heads: int = 2
    d_ff: int = 1408
    rope_theta: float = 10000.0
    attention: str = 'standard'
    hopfield_alpha: float = 0.5
    hopfield_alpha_prime: float = 0.5

    def validate(self):
        bounds = {'vocab_size': (512, 65535), 'context_length': (32, 2048),
                  'd_model': (64, 1024), 'num_layers': (1, 24), 'num_heads': (1, 32),
                  'num_kv_heads': (1, 32), 'd_ff': (64, 4096)}
        for key, (lo, hi) in bounds.items():
            value = getattr(self, key)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError(f'{key} must be an integer in [{lo}, {hi}]')
        if self.d_model % self.num_heads:
            raise ValueError('d_model must be divisible by query heads')
        if self.num_heads % self.num_kv_heads:
            raise ValueError('query heads must be divisible by KV heads')
        if (self.d_model // self.num_heads) % 2:
            raise ValueError('RoPE head dimension must be even')
        if self.d_ff < self.d_model or self.d_ff % 32:
            raise ValueError('d_ff must be >= d_model and a multiple of 32')
        if self.attention not in ['standard', 'hopfield', 'hopfield_official_v8']:
            raise ValueError('Unsupported attention')
        if not 0 <= self.hopfield_alpha < 1 or not 0 <= self.hopfield_alpha_prime < 1:
            raise ValueError('Hopfield coefficients must be in [0, 1)')
        if not 100 <= self.rope_theta <= 1000000:
            raise ValueError('Invalid RoPE theta')
        return self

    def dictionary(self):
        self.validate()
        return asdict(self)

    def parameter_estimate(self):
        self.validate()
        d, kv = self.d_model, self.num_kv_heads * (self.d_model // self.num_heads)
        per_block = 2 * d * d + 2 * d * kv + 3 * d * self.d_ff + 2 * d
        return 2 * self.vocab_size * d + self.num_layers * per_block + d
