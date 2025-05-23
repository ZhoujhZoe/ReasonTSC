from functools import partial
import numpy as np
from dataclasses import dataclass

@dataclass
class SerializerSettings:
    """Configuration for number serialization."""
    base: int = 10
    prec: int = 3
    signed: bool = True
    fixed_length: bool = False
    max_val: float = 1e7
    time_sep: str = ', '
    bit_sep: str = ''
    plus_sign: str = ''
    minus_sign: str = ' -'
    half_bin_correction: bool = True
    decimal_point: str = ''
    missing_str: str = ' Nan'

def vec_num2repr(val, base, prec, max_val):
    """Convert numbers to base representation with precision."""
    base = float(base)
    sign = 1 * (val >= 0) - 1 * (val < 0)
    val = np.abs(val)
    max_bit_pos = int(np.ceil(np.log(max_val) / np.log(base)).item())

    before_decimals = []
    for i in range(max_bit_pos):
        digit = (val / base**(max_bit_pos - i - 1)).astype(int)
        before_decimals.append(digit)
        val -= digit * base**(max_bit_pos - i - 1)
    
    before_decimals = np.stack(before_decimals, axis=-1)

    if prec > 0:
        after_decimals = []
        for i in range(prec):
            digit = (val / base**(-i - 1)).astype(int)
            after_decimals.append(digit)
            val -= digit * base**(-i - 1)
        digits = np.concatenate([before_decimals, after_decimals], axis=-1)
    else:
        digits = before_decimals
    return sign, digits

def serialize_arr(arr, settings):
    """Serialize array of numbers to string representation."""
    assert np.all(np.abs(arr[~np.isnan(arr)])) <= settings.max_val
    
    vnum2repr = partial(vec_num2repr, 
                       base=settings.base,
                       prec=settings.prec,
                       max_val=settings.max_val)
    
    sign_arr, digits_arr = vnum2repr(np.where(np.isnan(arr), 0, arr))
    is_missing = np.isnan(arr)
    
    bit_strs = []
    for sign, digits, missing in zip(sign_arr, digits_arr, is_missing):
        if missing:
            bit_strs.append(settings.missing_str)
            continue
            
        if not settings.fixed_length:
            nonzero = np.where(digits != 0)[0]
            digits = digits[nonzero[0]:] if len(nonzero) > 0 else np.array([0])
            
        digits_str = settings.bit_sep.join(map(str, digits))
        sign_str = settings.plus_sign if sign == 1 else settings.minus_sign
        bit_strs.append(sign_str + digits_str)
    
    return settings.time_sep.join(bit_strs) + settings.time_sep