import importlib
import os

# automatically import all python files in the directory
for _file in os.listdir(os.path.dirname(__file__)):
    if _file.endswith('.py') and not _file.startswith('_'):
        module = _file[:_file.find('.py')]
        importlib.import_module(__name__ + '.' + module)
