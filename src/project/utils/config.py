from copy import deepcopy
from typing import Union
_ITEM_TYPES = Union[float,int,str]
class Config(dict):
    """
    A wrapper for a dictionary which acts as the global config for any experiment
    Unlike a dict, a Config should be **one time writeable**, meaning you can set an element but never change one
    This gauruntees that Config fields are always immutable
    """

    def __init__(self, base_config={}):
        # Prevents references to the base_config from affecting us
        c = deepcopy(base_config)
        for k, v in c.items():
            self[k] = v
    
    def __setitem__(self, __k: str, v: _ITEM_TYPES) -> None:
        if __k in self:
            raise KeyError(f"Config field {__k} is already set")
        return super().__setitem__(__k, v)
    


