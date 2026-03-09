from typing import Union, List, Optional
from textwrap import dedent

from omegaconf import DictConfig
import torch

class RandomizerBase:
    def __init__(
        self,
        shape: Union[int, List[int], torch.Size],
        default_value: Union[float, bool],
        device: torch.device,
        enabled: bool = True,
        dtype: torch.dtype = torch.float,
    ):
        self.value = torch.zeros(shape, device=device, dtype=dtype)
        self.default_value = default_value
        self.enabled = enabled
        self.excluded_attributes = [
            "excluded_attributes",
            "value",
            "default_value",
            "randomize",
            "default",
            "enabled",
        ]
        self.default()
    
    def __getattr__(self, name: str):
        if name not in self.excluded_attributes and hasattr(self.value, name):
            return getattr(self.value, name)
        else:
            return getattr(self, name)

    def __str__(self) -> str:
        return str(self.value)
    
    def randomize(self, idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError
    
    def default(self) -> torch.Tensor:
        self.value = torch.full_like(self.value, self.default_value)
        return self.value
    
    def __add__(self, other):
        return self.value + other
    def __rsub__(self, other):
        return other - self.value
    def __sub__(self, other):
        return self.value - other
    def __rmul__(self, other):
        return other * self.value
    def __mul__(self, other):
        return self.value * other
    def __div__(self, other):
        return self.value / other
    def __neg__(self):
        return -self.value
    def reshape(self, shape: Union[int, List[int], torch.Size]):
        return self.value.reshape(shape)
    def squeeze(self, dim: int = -1):
        return self.value.squeeze(dim)
    def unsqueeze(self, dim: int = -1):
        return self.value.unsqueeze(dim)

class UniformRandomizer(RandomizerBase):
    def __init__(
        self,
        shape: Union[int, List[int], torch.Size],
        default_value: Union[float, bool],
        device: torch.device,
        enabled: bool = True,
        low: float = 0.0,
        high: float = 1.0,
        dtype: torch.dtype = torch.float,
    ):
        self.low = low
        self.high = high
        super().__init__(shape, default_value, device, enabled, dtype)
        self.excluded_attributes.extend(["low", "high"])
    
    def randomize(self, idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        if idx is not None:
            mask = torch.zeros_like(self.value, dtype=torch.bool)
            mask[idx] = True
            new = torch.rand_like(self.value) * (self.high - self.low) + self.low
            self.value = torch.where(mask, new, self.value)
        else:
            self.value.uniform_(self.low, self.high)
        return self.value
    
    def __repr__(self) -> str:
        return dedent(f"""
            UniformRandomizer(
                enabled={self.enabled},
                low={self.low},
                high={self.high},
                default={self.default_value},
                shape={self.value.shape},
                device={self.value.device},
                dtype={self.value.dtype}
            )""")
    
    @staticmethod
    def build(
        cfg: DictConfig,
        shape: Union[int, List[int], torch.Size],
        device: torch.device,
        dtype: torch.dtype = torch.float
    ) -> 'UniformRandomizer':
        return UniformRandomizer(
            shape=shape,
            default_value=cfg.default,
            device=device,
            enabled=cfg.enabled,
            low=cfg.min,
            high=cfg.max,
            dtype=dtype,
        )

class ChoiceRandomizer(RandomizerBase):
    def __init__(
        self,
        shape: Union[int, List[int], torch.Size],
        default_value: Union[float, bool],
        device: torch.device,
        choices: Union[List[int], List[float], List[bool]],
        enabled: bool = True,
        dtype: torch.dtype = torch.float,
    ):
        self.choices = torch.tensor(choices, device=device, dtype=dtype)
        self.n_choices = len(choices)
        super().__init__(shape, default_value, device, enabled, dtype)
        self.excluded_attributes.extend(["choices"])
    
    def randomize(self, idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        if idx is not None:
            mask = torch.zeros_like(self.value, dtype=torch.bool)
            mask[idx] = True
            new = self.choices[torch.randint(0, self.n_choices, self.value.shape, device=self.value.device)]
            self.value = torch.where(mask, new, self.value)
        else:
            self.value = self.choices[torch.randint(0, self.n_choices, self.value.shape, device=self.value.device)]
        return self.value
    
    def __repr__(self) -> str:
        return dedent(f"""
            ChoiceRandomizer(enabled={self.enabled},
                choices={self.choices},
                default={self.default_value},
                shape={self.value.shape},
                device={self.value.device},
                dtype={self.value.dtype}
            )""")
    
    @staticmethod
    def build(
        cfg: DictConfig,
        shape: Union[int, List[int], torch.Size],
        device: torch.device,
        dtype: torch.dtype = torch.float
    ) -> 'ChoiceRandomizer':
        return ChoiceRandomizer(
            shape=shape,
            default_value=cfg.default,
            device=device,
            enabled=cfg.enabled,
            choices=cfg.choices,
            dtype=dtype,
        )

class NormalRandomizer(RandomizerBase):
    def __init__(
        self,
        shape: Union[int, List[int], torch.Size],
        default_value: Union[float, bool],
        device: torch.device,
        enabled: bool = True,
        mean: float = 0.0,
        std: float = 1.0,
        dtype: torch.dtype = torch.float,
    ):
        self.mean = mean
        self.std = std
        super().__init__(shape, default_value, device, enabled, dtype)
        self.excluded_attributes.extend(["mean", "std"])
    
    def randomize(self, idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        if idx is not None:
            mask = torch.zeros_like(self.value, dtype=torch.bool)
            mask[idx] = True
            new = torch.randn_like(self.value) * self.std + self.mean
            self.value = torch.where(mask, new, self.value)
        else:
            self.value.normal_(self.mean, self.std)
        return self.value
    
    def __repr__(self) -> str:
        return dedent(f"""
            NormalRandomizer(enabled={self.enabled},
                mean={self.mean},
                std={self.std},
                default={self.default_value},
                shape={self.value.shape},
                device={self.value.device},
                dtype={self.value.dtype}
            )""")
    
    @staticmethod
    def build(
        cfg: DictConfig,
        shape: Union[int, List[int], torch.Size],
        device: torch.device,
        dtype: torch.dtype = torch.float
    ) -> 'NormalRandomizer':
        return NormalRandomizer(
            shape=shape,
            default_value=cfg.default,
            device=device,
            enabled=cfg.enabled,
            mean=cfg.mean,
            std=cfg.std,
            dtype=dtype,
        )

class RandomizerManager:
    randomizers: List[Union[UniformRandomizer, NormalRandomizer, ChoiceRandomizer]] = []
    def __init__(
        self, 
        cfg: DictConfig,
    ):
        self.enabled: bool = cfg.enabled
        
    def refresh(self, idx: Optional[torch.Tensor] = None):
        for randomizer in self.randomizers:
            if self.enabled and randomizer.enabled:
                randomizer.randomize(idx)
            else:
                randomizer.default()
    
    def __str__(self) -> str:
        return (
            "RandomizeManager(\n\t" + 
            f"Enabled: {self.enabled},\n\t" +
            ",\n        ".join([randomizer.__repr__() for randomizer in self.randomizers]) + 
            "\n)"
        )

def build_randomizer(
    cfg: DictConfig,
    shape: Union[int, List[int], torch.Size],
    device: torch.device,
    dtype: torch.dtype = torch.float,
) -> Union[UniformRandomizer, NormalRandomizer, ChoiceRandomizer]:
    if hasattr(cfg, "min") and hasattr(cfg, "max"):
        randomizer = UniformRandomizer.build(cfg, shape, device, dtype)
    elif hasattr(cfg, "mean") and hasattr(cfg, "std"):
        randomizer = NormalRandomizer.build(cfg, shape, device, dtype)
    elif hasattr(cfg, "choices"):
        randomizer = ChoiceRandomizer.build(cfg, shape, device, dtype)
    else:
        raise ValueError("Invalid randomizer configuration. Must contain 'min' and 'max' for UniformRandomizer or 'mean' and 'std' for NormalRandomizer or 'choices' for ChoiceRandomizer.")
    RandomizerManager.randomizers.append(randomizer)
    return randomizer

if __name__ == "__main__":
    # Example usage
    print(build_randomizer(DictConfig({"default": 0.5, "enabled": True, "min": 0, "max": 1}), [2, 3], torch.device("cpu")).randomize())
    print(build_randomizer(DictConfig({"default": 0.5, "enabled": True, "mean": 0, "std": 1}), [2, 3], torch.device("cpu")).randomize())
    print(build_randomizer(DictConfig({"default": 0, "enabled": True, "choices": [0, 1, 2, 3]}), [4, 5], torch.device("cpu")).randomize())