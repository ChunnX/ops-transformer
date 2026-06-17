#!/usr/bin/env python3
import struct
import math


def float32_to_hex(f):
    return f"0x{struct.unpack('>I', struct.pack('>f', f))[0]:08X}"


def hex_to_float32(h):
    h = h.strip()
    if h.startswith(('0x', '0X')):
        h = h[2:]
    return struct.unpack('>f', struct.pack('>I', int(h, 16)))[0]


def float16_to_hex(f):
    return f"0x{struct.unpack('>H', struct.pack('>e', f))[0]:04X}"


def hex_to_float16(h):
    h = h.strip()
    if h.startswith(('0x', '0X')):
        h = h[2:]
    return struct.unpack('>e', struct.pack('>H', int(h, 16)))[0]


def bfloat16_to_hex(f):
    f32 = struct.unpack('>I', struct.pack('>f', f))[0]
    frac = f32 & 0xFFFF
    guard = (frac >> 15) & 1
    rest = frac & 0x7FFF
    bf16 = f32 >> 16
    if guard and (rest > 0 or (bf16 & 1)):
        bf16 += 1
    bf16 &= 0xFFFF
    return f"0x{bf16:04X}"


def hex_to_bfloat16(h):
    h = h.strip()
    if h.startswith(('0x', '0X')):
        h = h[2:]
    bf16 = int(h, 16)
    f32 = bf16 << 16
    return struct.unpack('>f', struct.pack('>I', f32))[0]


def float_to_e4m3(value):
    if math.isnan(value):
        return 0x7F

    f32 = struct.unpack('>I', struct.pack('>f', value))[0]
    sign = (f32 >> 31) & 1
    exp32 = (f32 >> 23) & 0xFF
    mant32 = f32 & 0x7FFFFF

    if exp32 == 0:
        return sign << 7

    e_unbiased = exp32 - 127
    e4 = e_unbiased + 7

    if e4 >= 15:
        if e4 == 15 and mant32 == 0:
            return (sign << 7) | 0x7E
        if e4 == 15:
            frac = mant32 / (1 << 23)
            scaled = 1.0 + frac
            if scaled < 1.75:
                m3 = round((scaled - 1.0) * 8)
                if m3 < 8:
                    return (sign << 7) | (0xF << 3) | m3
            return (sign << 7) | 0x7E
        return (sign << 7) | 0x7F

    if e4 >= 1:
        frac = mant32 / (1 << 23)
        scaled = frac * 8.0
        m3 = int(scaled)
        rem = scaled - m3
        if rem > 0.5:
            m3 += 1
        elif rem == 0.5:
            if m3 & 1:
                m3 += 1
        if m3 >= 8:
            e4 += 1
            m3 = 0
            if e4 >= 15:
                if e4 == 15 and m3 == 0:
                    return (sign << 7) | (0xE << 3)
                return (sign << 7) | 0x7E
        return (sign << 7) | (e4 << 3) | m3

    if e4 >= -6:
        full_mant = (1 << 23) | mant32
        shift = 1 - e4
        ideal = full_mant / (1 << (23 + shift - 3))
        m3 = int(ideal)
        rem = ideal - m3
        if rem > 0.5:
            m3 += 1
        elif rem == 0.5:
            if m3 & 1:
                m3 += 1
        if m3 >= 8:
            return (sign << 7) | (1 << 3)
        if m3 == 0:
            return sign << 7
        return (sign << 7) | m3

    if e4 == -7:
        full_mant = (1 << 23) | mant32
        ideal = full_mant / (1 << 27)
        if ideal > 0.5:
            return (sign << 7) | 1
        if ideal == 0.5:
            return sign << 7
        return sign << 7

    return sign << 7


def e4m3_to_float(bits):
    bits &= 0xFF
    sign = (bits >> 7) & 1
    exp4 = (bits >> 3) & 0xF
    mant3 = bits & 0x7

    if exp4 == 0xF and mant3 == 0x7:
        return float('nan')

    if exp4 == 0:
        val = (mant3 / 8.0) * (2 ** -6)
    else:
        val = (1.0 + mant3 / 8.0) * (2 ** (exp4 - 7))

    return -val if sign else val


def float_to_e8m0(value):
    if math.isnan(value):
        return 0xFF
    if value == 0.0:
        return 0x00
    if value < 0 or math.isinf(value):
        return 0xFF

    exp_floor = int(math.floor(math.log2(value)))
    lower = 2.0 ** exp_floor
    upper = 2.0 ** (exp_floor + 1)
    threshold = lower * math.sqrt(2.0)

    if value > threshold:
        exp_nearest = exp_floor + 1
    elif value < threshold:
        exp_nearest = exp_floor
    else:
        exp_nearest = exp_floor if (exp_floor % 2 == 0) else exp_floor + 1

    biased = exp_nearest + 127
    if biased <= 0:
        return 0x00
    if biased >= 255:
        return 0xFF
    return biased


def e8m0_to_float(bits):
    bits &= 0xFF
    if bits == 0x00:
        return 0.0
    if bits == 0xFF:
        return float('nan')
    return 2.0 ** (bits - 127)


def parse_byte(s):
    s = s.strip()
    if s.startswith(('0x', '0X')):
        return int(s, 16)
    if s.startswith(('0b', '0B')):
        return int(s, 2)
    return int(s)


class FPConverterApp:
    def __init__(self, root, tk, ttk):
        self.root = root
        self.tk = tk
        self.ttk = ttk
        self.root.title("Floating Point Format Converter")
        self.root.resizable(True, False)

        style = ttk.Style()
        style.configure('Title.TLabel', font=('Helvetica', 14, 'bold'))

        ttk.Label(root, text="Float Format Converter", style='Title.TLabel').grid(
            row=0, column=0, columnspan=3, padx=10, pady=(10, 5))

        self.entries = {}
        self.vars = {}
        self._updating = False

        fields = [
            ("float", "Decimal Float"),
            ("float32_hex", "Float32 Hex"),
            ("e4m3", "Float8 E4M3"),
            ("e8m0", "Float8 E8M0"),
            ("float16", "Float16 Hex"),
            ("bfloat16", "BFloat16 Hex"),
        ]

        for i, (key, label) in enumerate(fields):
            ttk.Label(root, text=label + ":").grid(
                row=i + 1, column=0, sticky='e', padx=(10, 5), pady=3)
            var = tk.StringVar()
            entry = ttk.Entry(root, textvariable=var, width=28, font=('Courier', 11))
            entry.grid(row=i + 1, column=1, padx=(0, 5), pady=3)
            entry.bind('<Return>', lambda e, k=key: self.on_enter(k))
            btn = ttk.Button(root, text="Convert", width=8,
                             command=lambda k=key: self.on_enter(k))
            btn.grid(row=i + 1, column=2, padx=(0, 10), pady=3)
            self.entries[key] = entry
            self.vars[key] = var

        btn_frame = ttk.Frame(root)
        btn_frame.grid(row=len(fields) + 1, column=0, columnspan=3, pady=10)
        ttk.Button(btn_frame, text="Clear All", command=self.clear_all).pack(side='left', padx=5)

    def on_enter(self, source):
        if self._updating:
            return
        self._updating = True
        try:
            raw = self.vars[source].get().strip()
            if not raw:
                return

            if source == 'float':
                value = float(raw)
            elif source == 'float32_hex':
                value = hex_to_float32(raw)
            elif source == 'e4m3':
                value = e4m3_to_float(parse_byte(raw))
            elif source == 'e8m0':
                value = e8m0_to_float(parse_byte(raw))
            elif source == 'float16':
                value = hex_to_float16(raw)
            elif source == 'bfloat16':
                value = hex_to_bfloat16(raw)
            else:
                return

            for key in self.vars:
                if key == source:
                    continue
                if key == 'float':
                    self.vars[key].set(repr(value))
                elif key == 'float32_hex':
                    self.vars[key].set(float32_to_hex(value))
                elif key == 'e4m3':
                    bits = float_to_e4m3(value)
                    self.vars[key].set(f"0x{bits:02X}  ({e4m3_to_float(bits)})")
                elif key == 'e8m0':
                    bits = float_to_e8m0(value)
                    self.vars[key].set(f"0x{bits:02X}  ({e8m0_to_float(bits)})")
                elif key == 'float16':
                    self.vars[key].set(float16_to_hex(value))
                elif key == 'bfloat16':
                    self.vars[key].set(bfloat16_to_hex(value))
        except Exception as e:
            self.root.title(f"Error: {e}")
        finally:
            self._updating = False

    def clear_all(self):
        for var in self.vars.values():
            var.set('')
        self.root.title("Floating Point Format Converter")


if __name__ == '__main__':
    import tkinter as tk
    from tkinter import ttk
    root = tk.Tk()
    app = FPConverterApp(root, tk, ttk)
    root.mainloop()
