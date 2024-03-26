from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need
# fine tuning.
build_options = {'packages': ['streamlit', 'torch', 'llmlingua','transformers',
                              'llama_cpp','faster_whisper','langchain_openai',
                              'langchain', 'av'], 
                 'excludes': ['ipykernel', 'ipython', 'jupyter_client', 'jupyter_core',],
                 "include_files": [
                     ("streamlit_app/", "streamlit_app/"),
                     ("huginn_hears/", "streamlit_app/huginn_hears/"),
                     ],
                 "replace_paths": [("*", "")], # Makes sure all paths are relative so that the app can be run from any directory
                     }

base = None

executables = [
    Executable('streamlit_app/run.py', base=base, target_name = 'huginn-hears')
]

setup(name='HuginnHears',
      version = '0.1.0',
      description = 'Transcribe speech and summerize in Norwegian',
      options = {'build_exe': build_options},
      executables = executables)
