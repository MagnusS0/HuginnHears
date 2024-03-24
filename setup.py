from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need
# fine tuning.
build_options = {'packages': ['streamlit', 'torch', 'llmlingua','transformers',
                              'llama_cpp','faster_whisper','langchain_openai',
                              'langchain', 'av'], 
                 'excludes': [],
                 "include_files": [
                     ("streamlit_app/", "streamlit_app/"),
                     ("huginn_hears/", "streamlit_app/huginn_hears/"),
                     ]
                     }

base = 'console'

executables = [
    Executable('streamlit_app/run.py', base=base, target_name = 'huginn-hears')
]

setup(name='Huginn-hears',
      version = '1',
      description = 'Transcribe speech and summerize in Norwegian',
      options = {'build_exe': build_options},
      executables = executables)
