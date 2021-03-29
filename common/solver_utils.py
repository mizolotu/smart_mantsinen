import winreg

from subprocess import Popen, DEVNULL

def get_solver_path(hkey=r'Software\WOW6432Node\Mevea\Mevea Simulation Software'):
    regkey = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, hkey)
    (solver_path, _) = winreg.QueryValueEx(regkey, 'InstallPath')
    solver_path += r'\Bin\MeveaSolver.exe'
    winreg.CloseKey(regkey)
    return solver_path

def start_solver(solver_path, model_path, headless=True):
    solver_args = [solver_path, r'/mvs', model_path]
    if headless:
        solver_args.append(r'/headless')
    solver_proc = Popen(solver_args, stderr=DEVNULL, stdout=DEVNULL)
    return solver_proc

def stop_solver(solver_proc):
    solver_proc.terminate()