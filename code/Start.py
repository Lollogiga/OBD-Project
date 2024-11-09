# Scarichiamo tutti i pacchetti necessari:
import subprocess
import sys


def verify_pip():
    try:
        import pip
        print("Pip already installed")
    except ImportError:
        print("Pip not installed. Downloading...")
        subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
        print("pip installato correttamente.")


def install_packages(package_list):
    """
    Installa una lista di pacchetti tramite pip.

    Parameters:
        package_list (list): Lista di nomi dei pacchetti da installare.
    """
    for package_name in package_list:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"Package '{package_name}' installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f" Error installing package '{package_name}': {e}")

verify_pip()
packages = ["numpy", "matplotlib", "pandas", "scikit-learn"]
install_packages(packages)

print("All necessary packages have been installed. Ready to use")
