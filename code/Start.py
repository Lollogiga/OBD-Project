import subprocess
import sys

#Start function for Windows users:
def verify_pip():
    try:
        import pip
        print("Pip already installed")
    except ImportError:
        print("Pip not installed. Downloading...")
        subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
        print("pip correctly installed.")

def install_packages(package_list):
    """
    Install package through pip

    Parameters:
        package_list (list): List of package to install
    """
    for package_name in package_list:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"Package '{package_name}' installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f" Error installing package '{package_name}': {e}")

# Verify if pip is installed
verify_pip()


packages = ["numpy", "matplotlib", "pandas", "scikit-learn", "imbalanced-learn"]
install_packages(packages)

print("All necessary packages have been installed. Ready to use")
