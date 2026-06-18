#!/usr/bin/env python3
import struct
import math
import curses


def float32_to_hex(f):
    return "0x{:08X}".format(struct.unpack('>I', struct.pack('>f', f))[0])


def hex_to_float32(h):
    h = h.strip()
    if h.startswith(('0x', '0X')):
        h = h[2:]
    return struct.unpack('>f', struct.pack('>I', int(h, 16)))[0]


def float16_to_hex(f):
    return "0x{:04X}".format(struct.unpack('>H', struct.pack('>e', f))[0])


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
    return "0x{:04X}".format(bf16)


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
    return int(s, 16)


FIELDS = [
    ("float",       "Decimal Float  "),
    ("float32_hex", "Float32 Hex    "),
    ("e4m3",        "Float8 E4M3    "),
    ("e8m0",        "Float8 E8M0    "),
    ("float16",     "Float16 Hex    "),
    ("bfloat16",    "BFloat16 Hex   "),
]


def convert_from(source, raw):
    if source == 'float':
        return float(raw)
    elif source == 'float32_hex':
        return hex_to_float32(raw)
    elif source == 'e4m3':
        return e4m3_to_float(parse_byte(raw))
    elif source == 'e8m0':
        return e8m0_to_float(parse_byte(raw))
    elif source == 'float16':
        return hex_to_float16(raw)
    elif source == 'bfloat16':
        return hex_to_bfloat16(raw)


def format_field(key, value):
    if key == 'float':
        return repr(value)
    elif key == 'float32_hex':
        return float32_to_hex(value)
    elif key == 'e4m3':
        bits = float_to_e4m3(value)
        return "0x{:02X}  ({})".format(bits, e4m3_to_float(bits))
    elif key == 'e8m0':
        bits = float_to_e8m0(value)
        return "0x{:02X}  ({})".format(bits, e8m0_to_float(bits))
    elif key == 'float16':
        return float16_to_hex(value)
    elif key == 'bfloat16':
        return bfloat16_to_hex(value)


def main(stdscr):
    curses.curs_set(1)
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_CYAN, -1)
    curses.init_pair(2, curses.COLOR_YELLOW, -1)
    curses.init_pair(3, curses.COLOR_GREEN, -1)
    curses.init_pair(4, curses.COLOR_RED, -1)

    cur = 0
    values = {k: "" for k, _ in FIELDS}
    input_buf = ""
    error_msg = ""
    FIELD_WIDTH = 40

    while True:
        stdscr.erase()
        h, w = stdscr.getmaxyx()

        title = "  Float Format Converter"
        stdscr.addstr(0, 2, title, curses.A_BOLD | curses.color_pair(1))

        hint = "[Up/Down] navigate  [Enter] convert  [Esc/Ctrl-C] quit  [l] clear"
        if w > len(hint) + 4:
            stdscr.addstr(1, 2, hint, curses.color_pair(2))

        for i, (key, label) in enumerate(FIELDS):
            row = i + 3
            is_active = (i == cur)

            if is_active:
                attr_label = curses.A_BOLD | curses.color_pair(1)
                attr_val = curses.A_BOLD
            else:
                attr_label = curses.A_NORMAL
                attr_val = curses.A_NORMAL

            marker = "> " if is_active else "  "
            stdscr.addstr(row, 2, marker, attr_label)
            stdscr.addstr(row, 4, label, attr_label)
            stdscr.addstr(row, 4 + len(label), ": ", attr_label)

            col = 4 + len(label) + 2
            if is_active:
                stdscr.addstr(row, col, "[", curses.color_pair(2))
                stdscr.addstr(row, col + 1, input_buf.ljust(FIELD_WIDTH), curses.A_BOLD)
                stdscr.addstr(row, col + 1 + FIELD_WIDTH, "]", curses.color_pair(2))
                prev = values[key]
                if prev:
                    stdscr.addstr(row, col + FIELD_WIDTH + 4, "= " + prev, curses.color_pair(3))
                stdscr.move(row, col + 1 + len(input_buf))
            else:
                val = values[key]
                if val:
                    stdscr.addstr(row, col, val, curses.color_pair(3))
                else:
                    stdscr.addstr(row, col, "-" * FIELD_WIDTH, curses.A_DIM)

        if error_msg:
            err_row = len(FIELDS) + 4
            if err_row < h:
                stdscr.addstr(err_row, 2, "Error: " + error_msg, curses.color_pair(4))

        stdscr.refresh()

        try:
            ch = stdscr.getch()
        except KeyboardInterrupt:
            break

        if ch == 27 or ch == ord('q'):
            break
        elif ch == curses.KEY_UP:
            cur = (cur - 1) % len(FIELDS)
            input_buf = ""
            error_msg = ""
        elif ch == curses.KEY_DOWN:
            cur = (cur + 1) % len(FIELDS)
            input_buf = ""
            error_msg = ""
        elif ch == ord('l'):
            values = {k: "" for k, _ in FIELDS}
            input_buf = ""
            error_msg = ""
        elif ch == curses.KEY_BACKSPACE or ch == 127 or ch == 8:
            input_buf = input_buf[:-1]
            error_msg = ""
        elif ch == 10 or ch == 13:
            raw = input_buf.strip()
            if raw:
                source_key = FIELDS[cur][0]
                try:
                    value = convert_from(source_key, raw)
                    values[source_key] = raw
                    for k, _ in FIELDS:
                        if k != source_key:
                            values[k] = format_field(k, value)
                    error_msg = ""
                except Exception as e:
                    error_msg = str(e)
            input_buf = ""
        elif 32 <= ch <= 126:
            input_buf += chr(ch)
            error_msg = ""


if __name__ == '__main__':
    curses.wrapper(main)
