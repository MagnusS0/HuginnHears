import streamlit

import streamlit.web.cli as stcli
import os, sys

#Set link to CUDA
os.enviro(LD_LIBRARY_PATH="/usr/local/cuda/lib64") 

def resolve_path(path):
    resolved_path = os.path.abspath(os.path.join(os.getcwd(), path))
    return resolved_path

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "huginn_hears")))
    sys.argv = [
        "streamlit",
        "run",
        resolve_path("streamlit_app/app.py"),
        "--global.developmentMode=false",
    ]
    sys.exit(stcli.main())