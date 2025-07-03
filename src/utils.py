"""
utils.py
--------------------------------------------
Pequeña utilidad de logging coloreado basada
en colorama para mensajes uniformes.
--------------------------------------------
"""

from colorama import Fore, Style, init as _init_colorama

DEBUG_MODE = False

def set_debug(enable: bool) -> None:
    """Enable or disable debug output."""
    global DEBUG_MODE
    DEBUG_MODE = enable

_init_colorama(autoreset=True)

class Logger:
    """Very small wrapper for colored terminal output."""
    @staticmethod
    def info(msg: str) -> None:
        print(Fore.CYAN + "[INFO] " + Style.RESET_ALL + msg)

    @staticmethod
    def success(msg: str) -> None:
        print(Fore.GREEN + "[OK]   " + Style.RESET_ALL + msg)

    @staticmethod
    def warning(msg: str) -> None:
        print(Fore.YELLOW + "[WARN] " + Style.RESET_ALL + msg)

    @staticmethod
    def error(msg: str) -> None:
        print(Fore.RED + "[ERR]  " + Style.RESET_ALL + msg)

    @staticmethod
    def debug(msg: str) -> None:
        if DEBUG_MODE:
            print(Fore.MAGENTA + "[DBG]  " + Style.RESET_ALL + msg)
