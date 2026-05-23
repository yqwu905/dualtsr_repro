import os
from pathlib import Path
from typing import Optional

class DumpConfig:
    _instance: Optional['DumpConfig'] = None
    
    def __init__(self):
        self._enable = os.getenv("DUMP_ENABLE", "false").lower() in ("true", "1", "yes")
        self._dump_input = os.getenv("DUMP_INPUT", "false").lower() in ("true", "1", "yes")
        self._dump_init = os.getenv("DUMP_INIT", "false").lower() in ("true", "1", "yes")
        self._dump_blocks = os.getenv("DUMP_BLOCKS", "false").lower() in ("true", "1", "yes")
        self._dump_attn = os.getenv("DUMP_ATTN", "false").lower() in ("true", "1", "yes")
        self._dump_linear = os.getenv("DUMP_LINEAR", "false").lower() in ("true", "1", "yes")
        
        self._base_path = os.getenv("DUMP_BASE_PATH", "dump")
        
        self._offline_input_path = os.path.join(self._base_path, "offline_input")
        self._init_data_path = os.path.join(self._base_path, "init_data")
        self._per_block_result_path = os.path.join(self._base_path, "pre_block_result")
        self._attn_path = os.path.join(self._base_path, "sparse_attn")
        self._linear_path = os.path.join(self._base_path, "linear")
        
        self._input_shape = os.getenv("INPUT_SHAPE", None)
        
    @classmethod
    def get_instance(cls) -> 'DumpConfig':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset(cls):
        cls._instance = None
    
    @property
    def enable(self) -> bool:
        return self._enable and any([
            self._dump_input, self._dump_init, self._dump_blocks, self._dump_attn, self._dump_linear
        ])
    
    @property
    def dump_input(self) -> bool:
        return self.enable and self._dump_input
    
    @property
    def dump_init(self) -> bool:
        return self.enable and self._dump_init
    
    @property
    def dump_blocks(self) -> bool:
        return self.enable and self._dump_blocks
    
    @property
    def dump_attn(self) -> bool:
        return self.enable and self._dump_attn

    @property
    def dump_linear(self) -> bool:
        return self.enable and self._dump_linear
    
    @property
    def offline_input_path(self) -> Optional[str]:
        return self._offline_input_path if self.dump_input else None
    
    @property
    def init_data_path(self) -> Optional[str]:
        return self._init_data_path if self.dump_init else None
    
    @property
    def per_block_result_path(self) -> Optional[str]:
        return self._per_block_result_path if self.dump_blocks else None
    
    @property
    def attn_path(self) -> Optional[str]:
        return self._attn_path if self.dump_attn else None
    
    @property
    def linear_path(self) -> Optional[str]:
        return self._linear_path if self.dump_linear else None
    
    @property
    def input_shape(self) -> Optional[str]:
        return self._input_shape
    
    def ensure_dirs(self):
        if self.dump_input:
            os.makedirs(self._offline_input_path, exist_ok=True)
        if self.dump_init:
            os.makedirs(self._init_data_path, exist_ok=True)
        if self.dump_blocks:
            os.makedirs(self._per_block_result_path, exist_ok=True)
        if self.dump_attn:
            os.makedirs(self._attn_path, exist_ok=True)
        if self.dump_linear:
            os.makedirs(self._linear_path, exist_ok=True)


DUMP_CFG = DumpConfig.get_instance()
DUMP_CFG.ensure_dirs()

DUMP_OFFLINE_INPUT_PATH = DUMP_CFG.offline_input_path
DUMP_INIT_DATA_PATH = DUMP_CFG.init_data_path
DUMP_PER_BLOCK_RESULT_PATH = DUMP_CFG.per_block_result_path
DUMP_PATH = DUMP_CFG.attn_path
DUMP_LINEAR_PATH = DUMP_CFG.linear_path
INPUT_SHAPE = DUMP_CFG.input_shape