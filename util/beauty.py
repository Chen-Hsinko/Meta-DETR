import functools
from typing import Dict


class colors:
    '''Colors class:reset all colors with colors.reset; two
    sub classes fg for foreground
    and bg for background; use as colors.subclass.colorname.
    i.e. colors.fg.red or colors.bg.greenalso, the generic bold, disable,
    underline, reverse, strike through,
    and invisible work with the main class i.e. colors.bold'''
    reset = '\033[0m'
    bold = '\033[01m'
    disable = '\033[02m'
    underline = '\033[04m'
    reverse = '\033[07m'
    strikethrough = '\033[09m'
    invisible = '\033[08m'

    class fg:
        black = '\033[30m'
        red = '\033[31m'
        green = '\033[32m'
        orange = '\033[33m'
        blue = '\033[34m'
        purple = '\033[35m'
        cyan = '\033[36m'
        lightgrey = '\033[37m'
        darkgrey = '\033[90m'
        lightred = '\033[91m'
        lightgreen = '\033[92m'
        yellow = '\033[93m'
        lightblue = '\033[94m'
        pink = '\033[95m'
        lightcyan = '\033[96m'

    class bg:
        black = '\033[40m'
        red = '\033[41m'
        green = '\033[42m'
        orange = '\033[43m'
        blue = '\033[44m'
        purple = '\033[45m'
        cyan = '\033[46m'
        lightgrey = '\033[47m'


color_map = {
    "detr": [colors.fg.green, colors.bg.lightgrey],
    "backbone": [colors.fg.blue, colors.bg.lightgrey],
    "transformer": [colors.fg.cyan, colors.bg.lightgrey],
    "joiner": [colors.fg.orange, colors.bg.lightgrey]
    # 'default': [colors.fg.purple, colors.bg.lightgrey]
}


def colored_str(s, color_map: Dict=color_map, fg=True, bg=False):
    c = s.lower()
    reset_color = '\033[0m'
    if c in color_map:
        fg_color = color_map[c][0]
        bg_color = color_map[c][1]
    else:   # default color setting
        fg_color = colors.fg.purple
        bg_color = colors.bg.lightgrey
    str_colored = s + reset_color
    if fg: str_colored = fg_color + str_colored
    if bg: str_colored = bg_color + str_colored
    return str_colored


if __name__ == "__main__":
    print(colors.bg.green, "SKk", colors.fg.red, "Amartya", colors.reset)
    print(colors.bg.lightgrey, "SKk", colors.fg.red, "Amartya", colors.reset)
    print(colored_str('build', color_map=color_map, bg=False))

